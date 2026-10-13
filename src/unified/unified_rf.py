"""The single Random Forest extractor, trained on the same dataset as the CNN.

Replaces the two web-set extractors (url_only, full) and their routing. One forest reads the
link's 14 text features and the 17 page/domain features; where the page was not reached the
17 columns are NaN and the forest reads what is left (scikit-learn forests accept NaN).
Output per link: P, u and the 31 per-feature contributions (rf_feature_extractor.py).

Leak guard: the head must never see P/u/c_ from a forest that was fit on the same row.
`oof_outputs` gives every training row its P/u/c_ from a forest fit on the other folds, grouped
by registrable domain / template so one domain is never on both sides. Rows outside the
training set (rf_fit, test, live links) are scored by the final forest, `fit`, fit on all
training link rows.

Only link_status == "link" rows reach the forest. Rows without a link, and rows whose link
was redacted to the <URL> placeholder, get P = u = -1 and all c_ = 0.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from src.data.split_dataset import split_groups
from src.features.model.rf_extractor import PARAM_GRID, URL_TEXT_FEATURES, VARIANTS, fit_rf
from src.features.model.rf_feature_extractor import FULL_FEATURES, OUTPUT_COLUMNS, extract_vector

INPUTS = {"full": VARIANTS["full"], "url_only": URL_TEXT_FEATURES}
DEFAULT_PARAMS = {"n_estimators": 200, "min_samples_leaf": 2, "max_features": "sqrt"}


def link_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["link_status"] == "link").to_numpy()


def labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] == "smishing").to_numpy().astype(int)


def tune(df: pd.DataFrame, inputs: list[str], seed: int, n_splits: int = 3) -> tuple[dict, pd.DataFrame]:
    """Pick forest settings on the training link rows only, by grouped CV. Highest mean F1, ties
    broken by lower log loss (P is used as a probability downstream)."""
    rows = df[link_mask(df)]
    y, groups = labels(rows), split_groups(rows).to_numpy()
    folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(rows, y, groups))
    candidates = list(ParameterGrid(PARAM_GRID))
    scores = []
    for params in candidates:
        f1s, losses = [], []
        for train_idx, val_idx in folds:
            rf = fit_rf(rows.iloc[train_idx][inputs], y[train_idx], seed, **params)
            p = rf.predict_proba(rows.iloc[val_idx][inputs])[:, 1]
            f1s.append(f1_score(y[val_idx], p >= 0.5))
            losses.append(log_loss(y[val_idx], p, labels=[0, 1]))
        scores.append({"f1": float(np.mean(f1s)), "log_loss": float(np.mean(losses))})
    best = min(range(len(candidates)), key=lambda i: (-scores[i]["f1"], scores[i]["log_loss"]))
    grid = pd.DataFrame([{**str_params(p), **s} for p, s in zip(candidates, scores)])
    return candidates[best], grid


def str_params(params: dict) -> dict:
    return {k: str(v) for k, v in params.items()}


def fit(df: pd.DataFrame, inputs: list[str], seed: int, params: dict) -> dict:
    """Final forest on every link row of `df`. Returns the bundle extract_vector reads."""
    rows = df[link_mask(df)]
    return {"model": fit_rf(rows[inputs], labels(rows), seed, **params), "features": list(inputs)}


def _blank(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=df.index, columns=OUTPUT_COLUMNS)
    out[["P", "u"]] = -1.0
    return out


def outputs(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """P, u and the 31 contributions for each row of `df` (same index), from the final forest."""
    out = _blank(df)
    mask = link_mask(df)
    if mask.any():
        rows = df.loc[mask, FULL_FEATURES].copy()
        rows["has_url"] = 1
        out.loc[mask] = extract_vector(bundle, rows).to_numpy()
    return out


def oof_outputs(df: pd.DataFrame, inputs: list[str], seed: int, params: dict, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold P, u, c_ for every row of `df`: a row is scored by a forest that never saw its group."""
    out = _blank(df)
    mask = link_mask(df)
    rows = df[mask]
    y, groups = labels(rows), split_groups(rows).to_numpy()
    folds = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in folds.split(rows, y, groups):
        bundle = fit(rows.iloc[train_idx], inputs, seed, params)
        out.loc[rows.index[val_idx]] = outputs(bundle, rows.iloc[val_idx]).to_numpy()
    return out


def gate_importance(bundle: dict, df: pd.DataFrame, seed: int, n_repeats: int = 30) -> pd.Series:
    """Permutation importance (F1) of each input on held-out link rows, indexed by feature name.
    Features the forest does not read get 0. Feeds the RF-output gates."""
    rows = df[link_mask(df)]
    result = permutation_importance(
        bundle["model"], rows[bundle["features"]], labels(rows), scoring="f1",
        n_repeats=n_repeats, random_state=seed, n_jobs=-1,
    )
    importance = pd.Series(result.importances_mean, index=bundle["features"])
    return importance.reindex(FULL_FEATURES).fillna(0.0)
