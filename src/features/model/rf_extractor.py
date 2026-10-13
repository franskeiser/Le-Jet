"""Phase 2: train the Random Forest feature extractor on the public web phishing set.

Plan: data/dataset_and_training_plan.md section 5. Design decision: docs/decision_web_inspection.md.

The extractor turns one link's features into two numbers that are concatenated with the
CNN and URL features before gating:
    P -- mean of the trees' phishing probabilities
    u -- standard deviation of the trees' probabilities (tree disagreement)
Messages with no link get P = u = -1. P, u and has_url get a fixed gate of 1.0.

Two extractors are trained, matching experiment rows 6 and 7 of the plan. Each one only
ever sees the kind of data it was trained on, so no masking is needed:
    url_only -- the 14 features computed from the link text alone. Used for every old SMS
                link, and for fresh links whose page could not be fetched. No page_rank: it
                is a lookup made today, and for old SMS links today's answer mostly says
                "the scam domain has died", not "this is a scam" (Phase 0: 82% of smishing
                domains no longer resolve).
    full     -- link text + page_rank + webpage + external features, trained on complete
                pages only. Used only for fresh links whose page was fetched within hours.
route_p_u() picks between them per link.

Usage (from the project root):
    python -m src.features.model.rf_extractor
    python -m src.features.model.rf_extractor --top-k 25
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid

from src.config import ARTIFACTS_ROOT, SEEDS

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data" / "processed" / "web_phishing_clean.csv"
DEFAULT_MODEL_DIR = ARTIFACTS_ROOT / "models" / "rf_extractor"
DEFAULT_REPORT_DIR = ARTIFACTS_ROOT / "experiments" / "rf_extractor"

LABEL_TO_IDX = {"legitimate": 0, "phishing": 1}

# has_url is left out of every RF input: it is always 1 in the web set (zero information),
# and rows without a link never reach the forest -- they get P = u = -1 directly.
# Computed from the link string alone, so they mean the same today as when the SMS was sent.
URL_TEXT_FEATURES = [
    "url_length", "https_presence", "number_of_hyphens", "number_of_slashes",
    "subdomain_count", "special_char_count", "suspicious_url_token_count",
    "shortened_url_indicator", "has_ip_address", "suspicious_keyword_count",
    "path_length", "random_domain", "length_hostname", "tld_in_path",
]
# Open PageRank lookup -- in the URL group of the feature list, but it describes the domain
# at lookup time, like the external features.
PAGE_RANK = "page_rank"
WEBPAGE_FEATURES = [
    "nb_hyperlinks", "ratio_intHyperlinks", "nb_extCSS", "login_form", "external_favicon",
    "links_in_tags", "ratio_intMedia", "ratio_extMedia", "iframe", "empty_title",
    "domain_in_title", "domain_with_copyright",
]
EXTERNAL_FEATURES = [
    "whois_registered_domain", "domain_registration_length", "domain_age", "dns_record",
]
# Routing column for route_p_u(): 1 when the inspector fetched the page, else 0. It is
# never an input to either forest.
PAGE_FETCHED = "page_fetched"

VARIANTS = {
    "url_only": URL_TEXT_FEATURES,
    "full": URL_TEXT_FEATURES + [PAGE_RANK] + WEBPAGE_FEATURES + EXTERNAL_FEATURES,
}

# Shortcut ablations (plan section 5, step 5), run on the full variant.
ABLATIONS = {
    "all_features": [],
    "no_length_path": ["url_length", "path_length", "number_of_slashes", "length_hostname"],
    "no_domain_age": ["domain_age"],
    "no_page_rank": [PAGE_RANK],
}

# Tuned on the validation part only. Selection: highest validation F1, ties broken by
# lower log loss (P is fed downstream as a probability, so calibration matters too).
PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 20],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", 0.5],
}

# Offline public suffix list (bundled snapshot), private suffixes on, so x.blogspot.com is
# its own domain -- the same rule as lejet.urlnorm and the Phase 0 script.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)


def registrable_domain(url: str) -> str:
    host = (urlsplit(url if "://" in url else "http://" + url).hostname or "").rstrip(".")
    return _EXTRACT(host).top_domain_under_public_suffix or host


def load_dataset(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["status"].map(LABEL_TO_IDX)
    df["domain"] = df["url"].map(registrable_domain)
    return df


def grouped_split(df: pd.DataFrame, seed: int, val_size=0.15, test_size=0.15):
    # 70/15/15 by domain count (GroupShuffleSplit sizes by groups, so row counts vary a
    # little per seed), grouped by registrable domain so no domain is in two parts.
    groups = df["domain"].to_numpy()
    rest_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        .split(df, groups=groups)
    )
    rest_groups = groups[rest_idx]
    train_rel, val_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
        .split(rest_idx, groups=rest_groups)
    )
    return rest_idx[train_rel], rest_idx[val_rel], test_idx


def fit_rf(X: pd.DataFrame, y: np.ndarray, seed: int, **params) -> RandomForestClassifier:
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    rf.fit(X, y)
    return rf


def tree_probabilities(rf: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    # (n_trees, n_rows) phishing probability from each tree. The trees were fit on a plain
    # array inside the forest, so they get one too.
    values = X.to_numpy(dtype=np.float32)
    return np.stack([tree.predict_proba(values)[:, 1] for tree in rf.estimators_])


def p_and_u(rf: RandomForestClassifier, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    per_tree = tree_probabilities(rf, X)
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(ece)


def evaluate(rf: RandomForestClassifier, X: pd.DataFrame, y: np.ndarray) -> dict:
    p, u = p_and_u(rf, X)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "mean_u": float(u.mean()),
    }


def tune(X_train, y_train, X_val, y_val, seed: int) -> tuple[dict, pd.DataFrame]:
    rows = []
    for params in ParameterGrid(PARAM_GRID):
        rf = fit_rf(X_train, y_train, seed, **params)
        rows.append({**params, **evaluate(rf, X_val, y_val)})
    results = pd.DataFrame(rows)
    best = results.sort_values(["f1", "log_loss"], ascending=[False, True]).iloc[0]
    best_params = {k: best[k] for k in PARAM_GRID}
    # pandas turns None into NaN and ints into floats; restore the sklearn types.
    best_params["max_depth"] = None if pd.isna(best_params["max_depth"]) else int(best_params["max_depth"])
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    if best_params["max_features"] != "sqrt":
        best_params["max_features"] = float(best_params["max_features"])
    return best_params, results


def run_seed(df: pd.DataFrame, seed: int, top_k: int | None, model_dir: Path) -> dict:
    train_idx, val_idx, test_idx = grouped_split(df, seed)
    train, val, test = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    y_train, y_val, y_test = (part["label"].to_numpy() for part in (train, val, test))

    metrics, ablations, importances, tuning = [], [], [], []
    split_info = {
        "seed": seed, "train_n": len(train), "val_n": len(val), "test_n": len(test),
        "train_domains": train["domain"].nunique(),
        "test_phishing_share": float(y_test.mean()),
    }

    for variant, features in VARIANTS.items():
        params, grid = tune(train[features], y_train, val[features], y_val, seed)
        tuning.append(grid.assign(seed=seed, variant=variant))

        # Permutation importance on the validation part (plan section 5, step 7).
        rf = fit_rf(train[features], y_train, seed, **params)
        perm = permutation_importance(
            rf, val[features], y_val, scoring="f1", n_repeats=30, random_state=seed, n_jobs=-1,
        )
        importances.append(pd.DataFrame({
            "seed": seed, "variant": variant, "feature": features,
            "perm_mean": perm.importances_mean, "perm_std": perm.importances_std,
            "gini": rf.feature_importances_,
        }))

        if top_k is not None and top_k < len(features):
            order = np.argsort(perm.importances_mean)[::-1][:top_k]
            features = [features[i] for i in sorted(order)]
            rf = fit_rf(train[features], y_train, seed, **params)

        for split_name, part in (("val", val), ("test", test)):
            metrics.append({
                "seed": seed, "variant": variant, "split": split_name,
                "n_features": len(features), **params,
                **evaluate(rf, part[features], part["label"].to_numpy()),
            })

        # Freeze (plan section 5, step 8). Never retrained on SMS data.
        bundle = {
            "model": rf, "features": features, "variant": variant, "seed": seed,
            "params": params, "label_to_idx": LABEL_TO_IDX,
        }
        joblib.dump(bundle, model_dir / f"{variant}_seed{seed}.joblib")

    # Shortcut ablations on the full variant with its tuned parameters.
    full_params = next(m for m in metrics if m["variant"] == "full")
    full_params = {k: full_params[k] for k in PARAM_GRID}
    for name, dropped in ABLATIONS.items():
        features = [f for f in VARIANTS["full"] if f not in dropped]
        rf = fit_rf(train[features], y_train, seed, **full_params)
        ablations.append({
            "seed": seed, "ablation": name, "dropped": ",".join(dropped),
            **evaluate(rf, test[features], y_test),
        })

    return {
        "split": split_info, "metrics": metrics, "ablations": ablations,
        "importances": pd.concat(importances), "tuning": pd.concat(tuning),
    }


def extract_p_u(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    """P and u from one extractor, ready to concatenate before gating.

    `features` holds the extractor's input columns plus has_url. Rows with has_url == 0
    get P = u = -1, and so do rows whose features are ALL -1 (a link existed but nothing
    is known about it, e.g. the <URL> placeholder) -- the forest is never asked to score
    a row of pure sentinels. For the old SMS data, call this with the url_only extractor.
    """
    out = pd.DataFrame({"P": -1.0, "u": -1.0}, index=features.index)
    unknown = (features[bundle["features"]] == -1).all(axis=1)
    has_link = (features["has_url"] == 1) & ~unknown
    if has_link.any():
        p, u = p_and_u(bundle["model"], features.loc[has_link, bundle["features"]])
        out.loc[has_link, "P"] = p
        out.loc[has_link, "u"] = u
    return out


def route_p_u(url_only: dict, full: dict, features: pd.DataFrame) -> pd.DataFrame:
    """P and u for freshly inspected links: full extractor where the page was fetched,
    url_only everywhere else. `features` needs has_url and page_fetched (0/1); the page and
    external columns are only read for rows with page_fetched == 1."""
    fetched = (features["has_url"] == 1) & (features[PAGE_FETCHED] == 1)
    out = extract_p_u(url_only, features.loc[~fetched])
    if fetched.any():
        out = pd.concat([out, extract_p_u(full, features.loc[fetched])])
    out["extractor"] = np.where(fetched.reindex(out.index), "full", "url_only")
    return out.loc[features.index]


def load_extractor(variant: str, seed: int, model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    return joblib.load(model_dir / f"{variant}_seed{seed}.joblib")


def summarize(frame: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    agg = frame.groupby(by)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--top-k", type=int, default=None,
                        help="retrain on the top-k features by validation permutation importance")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    constant = [f for f in VARIANTS["full"] if df[f].nunique() == 1]
    if constant:
        print(f"Constant in the whole dataset (no signal for the forest): {constant}")

    results = []
    for seed in args.seeds:
        print(f"Seed {seed} ...")
        results.append(run_seed(df, seed, args.top_k, args.model_dir))

    metrics = pd.DataFrame([m for r in results for m in r["metrics"]])
    ablations = pd.DataFrame([a for r in results for a in r["ablations"]])
    importances = pd.concat([r["importances"] for r in results])
    tuning = pd.concat([r["tuning"] for r in results])
    splits = pd.DataFrame([r["split"] for r in results])

    score_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier", "ece", "mean_u"]
    metrics.to_csv(args.report_dir / "metrics_per_seed.csv", index=False)
    ablations.to_csv(args.report_dir / "ablations_per_seed.csv", index=False)
    importances.to_csv(args.report_dir / "importance_per_seed.csv", index=False)
    tuning.to_csv(args.report_dir / "tuning_grid.csv", index=False)
    splits.to_csv(args.report_dir / "splits.csv", index=False)

    metrics_summary = summarize(metrics, ["variant", "split"], score_cols)
    ablation_summary = summarize(ablations, ["ablation"], ["accuracy", "f1", "roc_auc"])
    importance_summary = (
        summarize(importances, ["variant", "feature"], ["perm_mean", "gini"])
        .sort_values(["variant", "perm_mean_mean"], ascending=[True, False])
    )
    metrics_summary.to_csv(args.report_dir / "metrics_summary.csv", index=False)
    ablation_summary.to_csv(args.report_dir / "ablations_summary.csv", index=False)
    importance_summary.to_csv(args.report_dir / "importance_summary.csv", index=False)
    (args.report_dir / "run_config.json").write_text(json.dumps({
        "data": str(args.data), "seeds": args.seeds, "top_k": args.top_k,
        "param_grid": PARAM_GRID, "variants": VARIANTS, "ablations": ABLATIONS,
        "constant_features": constant,
    }, indent=2))

    pd.set_option("display.width", 200)
    print("\nSplits:\n", splits.to_string(index=False))
    print("\nMetrics (mean/std over seeds):\n", metrics_summary.round(4).to_string(index=False))
    print("\nAblations on test (full variant):\n", ablation_summary.round(4).to_string(index=False))
    print("\nTop features (full):\n",
          importance_summary[importance_summary.variant == "full"].head(15).round(4).to_string(index=False))
    print("\nTop features (url_only):\n",
          importance_summary[importance_summary.variant == "url_only"].round(4).to_string(index=False))
    print(f"\nModels: {args.model_dir}\nReports: {args.report_dir}")


if __name__ == "__main__":
    main()
