"""Make the forest's static gates more stable, and measure it (docs/progress.md, gate improvements).

The first fit (train_static_gating.py, stage rfgates) turned a single permutation importance on one small gate slice
into gates with a min-max rescale. Three weaknesses showed: the importances barely agree across seeds (rank
correlation 0.35-0.56), 13-22 of 37 are negative noise, and one feature (page_rank) is 3x the next, so min-max squeezes
every other feature into a narrow band.

This script builds the importances four ways and the gates three ways, and compares them across the three seeds:

  importance sets
    slice_f1       forest permutation importance (F1) on the web gate slice (the first fit; read from its csv)
    slice_logloss  same slice, scored by log-loss (smooth, no threshold)
    oof_f1         5 grouped folds over web_train + gate slice; a forest trained without the fold scores it; averaged
    oof_logloss    same, scored by log-loss
  transforms
    minmax         0.1 + (importance - min) / (max - min) * 2.9      (the first fit)
    rank           0.1 + percentile rank * 2.9                        (one feature no longer sets the scale)
    rankfloor      features not clearly above zero (mean - 2 * spread <= 0) get 0.1; the rest get 0.5-3.0 by rank

Stability = agreement of the gate vectors across seeds (Spearman and Pearson, mean absolute gate difference, overlap of
the top 5 features). The test split is never read. This shows whether the gates are more stable, not whether they help.

Writes experiments/static_gating/gate_variants_f37.csv, gate_stability_f37.csv, and with --export the chosen gates.

Usage:
    python -m src.unified.improve_gates
    python -m src.unified.improve_gates --export oof_logloss rankfloor
"""
import argparse
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SEEDS
from src.data.split_dataset import split_groups
from src.features.model.rf_extractor import fit_rf
from src.features.model.rf_gating_module import gate_transform
from src.unified import unified_rf as urf
from src.unified.train_static_gating import FEATURE_SETS, MODEL_DIR, OUT_DIR, WEB_SPLITS, blank, load_web, part

FS = FEATURE_SETS["37"]
SCORING = {"f1": "f1", "logloss": "neg_log_loss"}
K_FOLDS = 5
TUNED = ("n_estimators", "max_depth", "max_features", "min_samples_leaf")
SETS = ["slice_f1", "slice_logloss", "oof_f1", "oof_logloss"]
TRANSFORMS = ["minmax", "rank", "rankfloor"]


def permute(model, X: pd.DataFrame, y: np.ndarray, seed: int, n_repeats: int) -> dict:
    r = permutation_importance(model, X, y, scoring=SCORING, n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return {k: (r[k].importances_mean, r[k].importances_std) for k in SCORING}


def slice_importance(seed: int, repeats: int) -> dict:
    bundle = joblib.load(MODEL_DIR / f"forest_seed{seed}{FS.suffix}.joblib")
    gate_rows = part(load_web(FS), WEB_SPLITS, seed, "rf_fit")
    gate_rows = gate_rows[urf.link_mask(gate_rows)]
    return permute(bundle["model"], gate_rows[FS.inputs], urf.labels(gate_rows), seed, repeats)


def oof_importance(seed: int, repeats: int) -> dict:
    bundle = joblib.load(MODEL_DIR / f"forest_seed{seed}{FS.suffix}.joblib")
    params = {k: v for k, v in bundle["model"].get_params().items() if k in TUNED}
    web = load_web(FS)
    pool = pd.concat([part(web, WEB_SPLITS, seed, "train"), part(web, WEB_SPLITS, seed, "rf_fit")], ignore_index=True)
    pool = pool[urf.link_mask(pool)].reset_index(drop=True)
    y, groups = urf.labels(pool), split_groups(pool).to_numpy()
    folds = []
    for i, (fit_idx, val_idx) in enumerate(StratifiedGroupKFold(K_FOLDS, shuffle=True, random_state=seed).split(pool, y, groups)):
        fit = pool.iloc[fit_idx]
        aug = pd.concat([fit, blank(fit, FS.page)], ignore_index=True)          # same recipe as the main forest
        rf = fit_rf(aug[FS.inputs], urf.labels(aug), seed, **params)
        val = pool.iloc[val_idx]
        folds.append(permute(rf, val[FS.inputs], urf.labels(val), seed + i, repeats))
        print(f"[seed {seed}] fold {i + 1}/{K_FOLDS} done", flush=True)
    out = {}
    for k in SCORING:
        per_fold = np.stack([f[k][0] for f in folds])                          # (folds, features)
        out[k] = (per_fold.mean(axis=0), per_fold.std(axis=0, ddof=1) / np.sqrt(K_FOLDS))
    return out


def transform(name: str, importance: np.ndarray, spread: np.ndarray) -> np.ndarray:
    n = len(importance)
    if name == "minmax":
        return gate_transform(importance)
    pct = (rankdata(importance) - 1) / (n - 1)
    if name == "rank":
        return 0.1 + pct * 2.9
    clear = (importance - 2 * spread) > 0                                      # clearly above zero
    gates = np.full(n, 0.1)
    if clear.any():
        r = rankdata(importance[clear]) - 1
        gates[clear] = 0.5 + 2.5 * (r / max(len(r) - 1, 1))
    return gates


def stability(gates_by_seed: list[np.ndarray], importance_by_seed: list[np.ndarray]) -> dict:
    sp, pe, diff, top = [], [], [], []
    for a, b in itertools.combinations(range(len(gates_by_seed)), 2):
        ga, gb = pd.Series(gates_by_seed[a]), pd.Series(gates_by_seed[b])
        sp.append(ga.corr(gb, method="spearman")); pe.append(ga.corr(gb)); diff.append(float(np.abs(ga - gb).mean()))
        top.append(len(set(np.argsort(-importance_by_seed[a])[:5]) & set(np.argsort(-importance_by_seed[b])[:5])))
    return {"spearman": np.nanmean(sp), "pearson": np.nanmean(pe), "mean_abs_gate_diff": np.mean(diff), "top5_overlap": np.mean(top)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--slice-repeats", type=int, default=30)
    ap.add_argument("--oof-repeats", type=int, default=10)
    ap.add_argument("--export", nargs=2, metavar=("SET", "TRANSFORM"), help="save the chosen gates as rf_gates_seed*_f37_fix.npy")
    args = ap.parse_args()
    variants = OUT_DIR / "gate_variants_f37.csv"

    if args.export:
        table = pd.read_csv(variants)
        chosen = table[(table["set"] == args.export[0]) & (table["transform"] == args.export[1])]
        assert len(chosen), f"no rows for {args.export}"
        for seed, rows in chosen.groupby("seed"):
            gates = rows.set_index("feature").loc[FS.inputs, "gate"].to_numpy(dtype=np.float32)
            np.save(MODEL_DIR / f"rf_gates_seed{seed}{FS.suffix}_fix.npy", gates)
            print(f"[seed {seed}] saved {len(gates)} gates ({args.export[0]} + {args.export[1]})")
        return

    importances = {}                                                           # (set, seed) -> (mean, spread)
    for seed in args.seeds:
        s = slice_importance(seed, args.slice_repeats)
        importances[("slice_f1", seed)], importances[("slice_logloss", seed)] = s["f1"], s["logloss"]
        o = oof_importance(seed, args.oof_repeats)
        importances[("oof_f1", seed)], importances[("oof_logloss", seed)] = o["f1"], o["logloss"]

    rows, stab = [], []
    for name in SETS:
        for tname in TRANSFORMS:
            gates, imps = [], []
            for seed in args.seeds:
                imp, spread = importances[(name, seed)]
                g = transform(tname, imp, spread)
                gates.append(g); imps.append(imp)
                rows += [{"seed": seed, "set": name, "transform": tname, "feature": f, "importance": float(i), "spread": float(sp), "gate": float(x)}
                         for f, i, sp, x in zip(FS.inputs, imp, spread, g)]
            noise = np.mean([(importances[(name, s)][0] - 2 * importances[(name, s)][1] <= 0).sum() for s in args.seeds])
            stab.append({"set": name, "transform": tname, **stability(gates, imps), "not_clearly_positive": float(noise),
                         "top_gate_share_above_1": float(np.mean([(g > 1.0).mean() for g in gates]))})
    pd.DataFrame(rows).to_csv(variants, index=False)
    report = pd.DataFrame(stab)
    report.to_csv(OUT_DIR / "gate_stability_f37.csv", index=False)
    pd.set_option("display.width", 220)
    print(report.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
