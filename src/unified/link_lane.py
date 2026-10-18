"""The C-prune link lane: one Random Forest on the 37 link features (docs/c_prune_architecture.md Phase 3).

Reuses the forest code of the current model (rf_extractor.fit_rf / PARAM_GRID); only the inputs change:
the 20 URL + 17 page/domain features of the crowd dataset (prepare_web_37.FEATURES_37). Missing values
stay -1, as in the dataset (section 4.3).

Step 1 (this file so far): tune the forest per seed and fold, on the fold's TRAIN part only, by grouped
cross-validation. The groups follow the rule the splits were made with (split_report.json), so one
domain is never on both sides of a tuning fold either. Val and test are never read here.
Choice: highest mean F1, ties broken by lower log loss (P is used as a probability later), the same
rule as src/unified/unified_rf.tune.

Writes, per seed and fold, to experiments/c_prune/<dataset>/link_lane/:
    params_seed<S>_fold<F>.json   the chosen settings
    grid_seed<S>_fold<F>.csv      every candidate's mean F1 and log loss

Usage:
    python -m src.unified.link_lane --pilot tune
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from src.config import ARTIFACTS_ROOT, CROWD_DATASET_PATH, CROWD_N_FOLDS, CROWD_PILOT_PATH, CROWD_SPLITS_DIR, SEEDS
from src.data.split_crowd import load_fold, make_groups
from src.features.model.rf_extractor import PARAM_GRID, fit_rf
from src.unified.prepare_web_37 import FEATURES_37

ROOT = Path(__file__).resolve().parents[2]
TUNE_SPLITS = 3


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


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("step", choices=["tune"])
    a = ap.parse_args(argv)
    summary = run_tune(a.pilot)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
