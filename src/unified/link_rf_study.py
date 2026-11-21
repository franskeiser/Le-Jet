"""Two questions about the Link RF, answered on the TRAINING and VALIDATION parts only (the test folds are never read).

trees   How many trees? For each of the 15 training parts (3 seeds x 5 folds), grouped 3-fold CV on the messages that have a
        link (the same CV as link_lane.tune). For each of the 8 other settings (link_lane.PARAM_GRID without n_estimators) one
        1,000-tree forest is fit per inner fold and its first k trees are scored, k in TREE_COUNTS. With a fixed random_state,
        sklearn builds tree i the same way whatever n_estimators is, so the first k trees ARE the k-tree forest (asserted once).
        Per training part and k, the setting link_lane.tune would pick (best F1, then log loss) is kept.
        Writes trees_grid.csv (every part x setting x k) and trees_summary.csv (per k: mean and SD over the 15 parts).
        Decision rule (fixed before reading the results): the k with the highest mean inner-CV F1; if a smaller k is within
        0.001 F1 of it, the smallest such k (more trees cost time per message for no measurable gain).

rows    Which messages should the forest learn from? (A) every training message with a link (the code today) or (B) only those
        whose page loaded ("trained on complete pages", the diagram). Both are fit with the same settings (--trees, the fold's
        tuned setting at that count) and score the VALIDATION messages that have a link, as inspected. Reported by page condition
        (page loaded / page not loaded). Writes rows_by_fold.csv and rows_summary.csv.

Usage:
    python -m src.unified.link_rf_study trees
    python -m src.unified.link_rf_study rows --trees 300
Output: experiments/c_prune/<dataset>/link_rf_study/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from src.config import CROWD_DATASET_PATH, CROWD_N_FOLDS, SEEDS
from src.data.split_crowd import load_fold, make_groups
from src.features.model.rf_extractor import PARAM_GRID, fit_rf, tree_probabilities
from src.unified import link_lane as ll
from src.unified.prepare_web_37 import FEATURES_37

ROOT = Path(__file__).resolve().parents[2]
TREE_COUNTS = [25, 50, 100, 200, 300, 500, 750, 1000]
OTHER_GRID = {k: v for k, v in PARAM_GRID.items() if k != "n_estimators"}
TIE = 0.001


def out_dir() -> Path:
    d = ll.out_dir(ROOT / CROWD_DATASET_PATH).parent / "link_rf_study"
    d.mkdir(parents=True, exist_ok=True)
    return d


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"f1": f1_score(y, p >= 0.5, zero_division=0), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "bal_acc": balanced_accuracy_score(y, p >= 0.5)}


def prefix_check(X, y, seed, params) -> None:
    big = fit_rf(X, y, seed, n_estimators=60, **params)
    small = fit_rf(X, y, seed, n_estimators=20, **params)
    assert np.allclose(tree_probabilities(big, X.iloc[:200])[:20], tree_probabilities(small, X.iloc[:200])), \
        "the first k trees of a bigger forest are not the k-tree forest: the prefix shortcut is invalid"


def run_trees() -> pd.DataFrame:
    dataset = ROOT / CROWD_DATASET_PATH
    rule = ll.group_rule(dataset)
    rows, checked, t0 = [], False, time.time()
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            train = load_fold(seed, fold)["train"]
            train = train[ll.has_link(train)]
            X, y = train[FEATURES_37], ll.labels(train)
            groups = make_groups(train, rule, ll.TUNE_SPLITS).to_numpy()
            inner = list(StratifiedGroupKFold(ll.TUNE_SPLITS, shuffle=True, random_state=seed).split(X, y, groups))
            for params in ParameterGrid(OTHER_GRID):
                if not checked:
                    prefix_check(X, y, seed, params)
                    checked = True
                per_k = {k: [] for k in TREE_COUNTS}
                for fit_idx, held_idx in inner:
                    rf = fit_rf(X.iloc[fit_idx], y[fit_idx], seed, n_estimators=max(TREE_COUNTS), **params)
                    cum = np.cumsum(tree_probabilities(rf, X.iloc[held_idx]), axis=0)
                    for k in TREE_COUNTS:
                        per_k[k].append(metrics(y[held_idx], cum[k - 1] / k))
                for k in TREE_COUNTS:
                    rows.append({"seed": seed, "fold": fold, **{p: str(v) for p, v in params.items()}, "n_trees": k,
                                 **{m: float(np.mean([r[m] for r in per_k[k]])) for m in ("f1", "log_loss", "bal_acc")}})
            print(f"seed {seed} fold {fold} done ({time.time() - t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir() / "trees_grid.csv", index=False)
    # per part and k: the setting tune would pick
    best = (grid.sort_values(["f1", "log_loss"], ascending=[False, True])
                .groupby(["seed", "fold", "n_trees"], as_index=False).first())
    summary = best.groupby("n_trees").agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"), log_loss_mean=("log_loss", "mean"),
                                          bal_acc_mean=("bal_acc", "mean")).reset_index()
    top = summary.f1_mean.max()
    summary["within_tie_of_best"] = summary.f1_mean >= top - TIE
    chosen = int(summary.loc[summary.within_tie_of_best, "n_trees"].min())
    summary["chosen"] = summary.n_trees == chosen
    summary.to_csv(out_dir() / "trees_summary.csv", index=False)
    best.to_csv(out_dir() / "trees_best_setting_per_part.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"CHOSEN n_trees = {chosen} (best mean F1 {top:.4f}; rule: smallest count within {TIE} of the best)")
    return summary


def run_rows(n_trees: int) -> pd.DataFrame:
    best = pd.read_csv(out_dir() / "trees_best_setting_per_part.csv")
    rows = []
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            r = best[(best.seed == seed) & (best.fold == fold) & (best.n_trees == n_trees)].iloc[0]
            params = {"max_depth": None if str(r.max_depth) in ("None", "nan") else int(float(r.max_depth)),
                      "min_samples_leaf": int(r.min_samples_leaf),
                      "max_features": r.max_features if r.max_features == "sqrt" else float(r.max_features)}
            parts = load_fold(seed, fold)
            train, val = parts["train"], parts["val"]
            train, val = train[ll.has_link(train)], val[ll.has_link(val)]
            yv = ll.labels(val)
            loaded = val.page_fetched.astype(int).eq(1).to_numpy()
            fits = {"A_all_linked": train, "B_complete_pages_only": train[train.page_fetched.astype(int) == 1]}
            for name, rows_fit in fits.items():
                rf = fit_rf(rows_fit[FEATURES_37], ll.labels(rows_fit), seed, n_estimators=n_trees, **params)
                p = rf.predict_proba(val[FEATURES_37])[:, 1]
                for cond, mask in (("all_linked", np.ones_like(loaded)), ("page_loaded", loaded), ("page_not_loaded", ~loaded)):
                    rows.append({"seed": seed, "fold": fold, "variant": name, "train_rows": len(rows_fit), "val_condition": cond,
                                 "n_val": int(mask.sum()), **metrics(yv[mask], p[mask])})
            print(f"seed {seed} fold {fold} done", flush=True)
    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(out_dir() / "rows_by_fold.csv", index=False)
    summary = by_fold.groupby(["val_condition", "variant"]).agg(
        n_val_mean=("n_val", "mean"), f1_mean=("f1", "mean"), f1_sd=("f1", "std"), bal_acc_mean=("bal_acc", "mean"),
        log_loss_mean=("log_loss", "mean")).reset_index()
    summary.to_csv(out_dir() / "rows_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["trees", "rows"])
    ap.add_argument("--trees", type=int, help="rows: the tree count chosen by the trees step")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    run_trees() if a.step == "trees" else run_rows(a.trees)


if __name__ == "__main__":
    main()
