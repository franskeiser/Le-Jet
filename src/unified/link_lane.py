"""The C-prune link lane: one Random Forest on the 37 link features (docs/c_prune_architecture.md Phase 3).

Reuses the forest code of the current model (rf_extractor.fit_rf / PARAM_GRID); only the inputs change:
the 20 URL + 17 page/domain features of the crowd dataset (prepare_web_37.FEATURES_37). Missing values
stay -1, as in the dataset (section 4.3).

Step 1, tune: the forest settings per seed and fold, on the fold's TRAIN part only, by grouped
cross-validation. The groups follow the rule the splits were made with (split_report.json), so one
domain is never on both sides of a tuning fold either. Val and test are never read here.
Choice: highest mean F1, ties broken by lower log loss (P is used as a probability later), the same
rule as src/unified/unified_rf.tune.

Step 2, oof: out-of-fold outputs for every TRAIN row. The train part is cut into 5 grouped folds; each
row is scored by a forest (step 1 settings) fit on the other 4, so no row is scored by a forest that saw
it or its group. Per row:
    P     mean smishing probability over the trees
    u     spread of the trees' probabilities
    c_<feature>   the 37 Saabas contributions (rf_feature_extractor), with P = bias + sum of c_

Step 3, final: one forest (step 1 settings) fit on ALL train rows scores the val and test rows, which it
never saw. The forest is saved for the later phases and the live app. Only val is summarised: the test
scores are written but not evaluated here (Phase 7 reads the test once, after everything is fixed).

Writes, per seed and fold, to experiments/c_prune/<dataset>/link_lane/:
    params_seed<S>_fold<F>.json   the chosen settings                     (tune)
    grid_seed<S>_fold<F>.csv      every candidate's mean F1 and log loss  (tune)
    oof_seed<S>_fold<F>.csv       message_id, P, u, 37 c_                 (oof)
    final_seed<S>_fold<F>.csv     message_id, part (val/test), P, u, 37 c_  (final)
and models/c_prune/<dataset>/link_rf_seed<S>_fold<F>.joblib               (final)

Usage:
    python -m src.unified.link_lane --pilot tune
    python -m src.unified.link_lane --pilot oof
    python -m src.unified.link_lane --pilot final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from src.config import ARTIFACTS_ROOT, CROWD_DATASET_PATH, CROWD_N_FOLDS, CROWD_PILOT_PATH, CROWD_SPLITS_DIR, SEEDS
from src.data.split_crowd import load_fold, make_groups
from src.features.model.rf_extractor import PARAM_GRID, fit_rf, p_and_u
from src.features.model.rf_feature_extractor import path_contributions
from src.unified.prepare_web_37 import FEATURES_37

ROOT = Path(__file__).resolve().parents[2]
TUNE_SPLITS = 3
OOF_SPLITS = 5
CONTRIBUTIONS = [f"c_{f}" for f in FEATURES_37]
OUTPUTS = ["P", "u"] + CONTRIBUTIONS


def labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] == "smishing").to_numpy().astype(int)


def group_rule(dataset: Path) -> str:
    report = ROOT / CROWD_SPLITS_DIR / dataset.stem / "split_report.json"
    return json.loads(report.read_text(encoding="utf-8"))["group_by"]


def tune(train: pd.DataFrame, rule: str, seed: int, n_splits: int = TUNE_SPLITS) -> Tuple[dict, pd.DataFrame]:
    """Settings for the link forest, chosen by grouped CV on `train` only."""
    X, y = train[FEATURES_37], labels(train)
    groups = make_groups(train, rule, n_splits).to_numpy()
    folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups))
    candidates = list(ParameterGrid(PARAM_GRID))
    scores = []
    for params in candidates:
        f1s, losses = [], []
        for fit_idx, held_idx in folds:
            rf = fit_rf(X.iloc[fit_idx], y[fit_idx], seed, **params)
            p = rf.predict_proba(X.iloc[held_idx])[:, 1]
            f1s.append(f1_score(y[held_idx], p >= 0.5))
            losses.append(log_loss(y[held_idx], p, labels=[0, 1]))
        scores.append({"f1": float(np.mean(f1s)), "log_loss": float(np.mean(losses))})
    best = min(range(len(candidates)), key=lambda i: (-scores[i]["f1"], scores[i]["log_loss"]))
    grid = pd.DataFrame([{**{k: str(v) for k, v in p.items()}, **s} for p, s in zip(candidates, scores)])
    return candidates[best], grid


def score(rf, rows: pd.DataFrame) -> pd.DataFrame:
    """P, u and the 37 contributions of `rows` (same index) from one fitted forest."""
    X = rows[FEATURES_37]
    p, u = p_and_u(rf, X)
    contribution, _ = path_contributions(rf, X.to_numpy(dtype=np.float32))
    out = pd.DataFrame(contribution, index=rows.index, columns=CONTRIBUTIONS)
    out.insert(0, "u", u)
    out.insert(0, "P", p)
    return out


def oof_outputs(train: pd.DataFrame, rule: str, seed: int, params: dict,
                n_splits: int = OOF_SPLITS) -> pd.DataFrame:
    """Out-of-fold P, u, c_ for every row of `train`: each row is scored by a forest that never saw its group."""
    y = labels(train)
    groups = make_groups(train, rule, n_splits).to_numpy()
    parts = []
    for fit_idx, held_idx in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(train, y, groups):
        rf = fit_rf(train.iloc[fit_idx][FEATURES_37], y[fit_idx], seed, **params)
        parts.append(score(rf, train.iloc[held_idx]))
    out = pd.concat(parts).loc[train.index]          # back in row order; fails if a row was missed
    assert out.index.is_unique and len(out) == len(train), "a row was scored twice or not at all"
    return out


def final_outputs(parts: dict, seed: int, params: dict) -> tuple:
    """(forest fit on all of parts["train"], its P, u, c_ for the val and test rows with a `part` column)."""
    train = parts["train"]
    rf = fit_rf(train[FEATURES_37], labels(train), seed, **params)
    scored = []
    for name in ("val", "test"):
        out = score(rf, parts[name])
        out.insert(0, "part", name)
        out.insert(0, "message_id", parts[name]["message_id"])
        scored.append(out)
    return rf, pd.concat(scored, ignore_index=True)


def load_params(dataset: Path, seed: int, fold: int) -> dict:
    return json.loads((out_dir(dataset) / f"params_seed{seed}_fold{fold}.json").read_text(encoding="utf-8"))


def out_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "experiments" / "c_prune" / dataset.stem / "link_lane"


def run_tune(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = ROOT / (CROWD_PILOT_PATH if pilot else CROWD_DATASET_PATH)
    rule = group_rule(dataset)
    out = out_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            train = load_fold(seed, fold, pilot=pilot)["train"]
            params, grid = tune(train, rule, seed)
            (out / f"params_seed{seed}_fold{fold}.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
            grid.to_csv(out / f"grid_seed{seed}_fold{fold}.csv", index=False)
            best = grid.sort_values(["f1", "log_loss"], ascending=[False, True]).iloc[0]
            rows.append({"seed": seed, "fold": fold, "train_rows": len(train), **params,
                         "cv_f1": round(best["f1"], 4), "cv_log_loss": round(best["log_loss"], 4)})
            print(f"seed {seed} fold {fold}: {params} f1={best['f1']:.4f}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "tune_summary.csv", index=False)
    return summary


def run_oof(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = ROOT / (CROWD_PILOT_PATH if pilot else CROWD_DATASET_PATH)
    rule = group_rule(dataset)
    out = out_dir(dataset)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            train = load_fold(seed, fold, pilot=pilot)["train"]
            oof = oof_outputs(train, rule, seed, load_params(dataset, seed, fold))
            oof.insert(0, "message_id", train["message_id"])
            oof.to_csv(out / f"oof_seed{seed}_fold{fold}.csv", index=False)
            y = labels(train)
            rows.append({"seed": seed, "fold": fold, "train_rows": len(train),
                         "oof_f1": round(f1_score(y, oof["P"] >= 0.5), 4),
                         "oof_log_loss": round(log_loss(y, oof["P"].clip(1e-6, 1 - 1e-6), labels=[0, 1]), 4),
                         "mean_u": round(float(oof["u"].mean()), 4)})
            print(f"seed {seed} fold {fold}: oof F1 {rows[-1]['oof_f1']}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "oof_summary.csv", index=False)
    return summary


def model_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "models" / "c_prune" / dataset.stem


def run_final(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = ROOT / (CROWD_PILOT_PATH if pilot else CROWD_DATASET_PATH)
    out, models = out_dir(dataset), model_dir(dataset)
    models.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            parts = load_fold(seed, fold, pilot=pilot)
            params = load_params(dataset, seed, fold)
            rf, scored = final_outputs(parts, seed, params)
            joblib.dump({"model": rf, "features": list(FEATURES_37), "params": params},
                        models / f"link_rf_seed{seed}_fold{fold}.joblib")
            scored.to_csv(out / f"final_seed{seed}_fold{fold}.csv", index=False)
            val = scored[scored["part"] == "val"]
            y = labels(parts["val"])
            rows.append({"seed": seed, "fold": fold, "val_rows": len(val), "test_rows_scored": int((scored["part"] == "test").sum()),
                         "val_f1": round(f1_score(y, val["P"] >= 0.5), 4),
                         "val_log_loss": round(log_loss(y, val["P"].clip(1e-6, 1 - 1e-6), labels=[0, 1]), 4)})
            print(f"seed {seed} fold {fold}: val F1 {rows[-1]['val_f1']}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "final_summary.csv", index=False)
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("step", choices=["tune", "oof", "final"])
    a = ap.parse_args(argv)
    summary = {"tune": run_tune, "oof": run_oof, "final": run_final}[a.step](a.pilot)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
