"""The link lane: one Random Forest on the 37 link features.

Reuses the forest code of the current model (rf_extractor.fit_rf / PARAM_GRID); only the inputs change:
the 20 URL + 17 page/domain features of the crowd dataset (prepare_web_37.FEATURES_37). Missing values
stay -1, as in the dataset.

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
never saw. The forest is saved for the later steps and the live app. Only val is summarised: the test
scores are written but not evaluated here (the final comparison reads the test once, after everything is fixed).

The 40 link numbers a model can read (step 4): P, u, the 37 c_ and page_missing = 1 - page_fetched.

Messages with no link (has_url = 0, when the dataset has that column): the forest is tuned, fitted and scored on the messages that HAVE a
link only. A no-link message gets P = 0, u = 0, every c_ = 0 and page_missing = 1 (no page), which the models never read as evidence:
E2 skips its generator for it (has_url = 0) and the concatenation control reads has_url next to them. Summaries are over link messages.
has_url is gone: every message has a link. Steps 2 and 3 write every row twice, scored by the SAME forest:
    full    the row as inspected
    blank   the page hidden: the 13 page columns set to -1, page_missing = 1. The 4 outside columns
            (WHOIS, domain age, page rank) stay, as for a real dead page.
Modality dropout (while training a fused model) swaps a training row's full numbers for its blank ones, so a hidden page
never carries the full page's P and contributions.

Step 5, check: P = bias + sum of the 37 c_ on the written files (the unit test checks it on toy data).
    final  P - sum(c_) equals the saved forest's bias on every val and test row, both versions;
    oof    P - sum(c_) takes at most 5 values (one bias per inner forest, which is not saved).

Writes, per seed and fold, to experiments/c_prune/<dataset>/link_lane/:
    params_seed<S>_fold<F>.json   the chosen settings                     (tune)
    grid_seed<S>_fold<F>.csv      every candidate's mean F1 and log loss  (tune)
    oof_seed<S>_fold<F>.csv       message_id, the 40 numbers              (oof)
    oof_blank_seed<S>_fold<F>.csv   the same, page hidden                 (oof)
    final_seed<S>_fold<F>.csv     message_id, part (val/test), the 40 numbers  (final)
    final_blank_seed<S>_fold<F>.csv the same, page hidden                 (final)
and models/c_prune/<dataset>/link_rf_seed<S>_fold<F>.joblib               (final)

Usage:
    python -m src.unified.link_lane --pilot tune
    python -m src.unified.link_lane --pilot oof
    python -m src.unified.link_lane --pilot final      (run oof and final again after a change to step 4)
    python -m src.unified.link_lane --pilot check
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
from src.unified.prepare_web_37 import FEATURES_37, PAGE

ROOT = Path(__file__).resolve().parents[2]
TUNE_SPLITS = 3
OOF_SPLITS = 5
CONTRIBUTIONS = [f"c_{f}" for f in FEATURES_37]
OUTPUTS = ["P", "u"] + CONTRIBUTIONS
LANE = OUTPUTS + ["page_missing"]   # the 40 link numbers
MISSING = -1


def labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] == "smishing").to_numpy().astype(int)


def has_link(df: pd.DataFrame) -> pd.Series:
    """True for the messages that carry a link. A dataset without a has_url column (the first pilot) has a link in every message."""
    return df["has_url"].astype(int).eq(1) if "has_url" in df.columns else pd.Series(True, index=df.index)


def group_rule(dataset: Path) -> str:
    report = ROOT / CROWD_SPLITS_DIR / dataset.stem / "split_report.json"
    return json.loads(report.read_text(encoding="utf-8"))["group_by"]


def tune(train: pd.DataFrame, rule: str, seed: int, n_splits: int = TUNE_SPLITS) -> Tuple[dict, pd.DataFrame]:
    """Settings for the link forest, chosen by grouped CV on `train` only (the messages that have a link)."""
    train = train[has_link(train)]
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


def blank_page(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows as if the page had not been reached: 13 page columns -1, the 4 outside columns kept."""
    out = rows.copy()
    out[PAGE] = MISSING
    out["page_fetched"] = 0
    return out


def lane_numbers(rf, rows: pd.DataFrame) -> pd.DataFrame:
    """The 40 link numbers of `rows` (same index). A message with no link gets zeros and page_missing = 1; the forest never sees it."""
    linked = has_link(rows).to_numpy()
    out = pd.DataFrame(0.0, index=rows.index, columns=OUTPUTS)
    if linked.any():
        out.loc[linked, OUTPUTS] = score(rf, rows[linked])[OUTPUTS].to_numpy()
    out["page_missing"] = 1 - rows["page_fetched"].astype(int)
    return out


def oof_outputs(train: pd.DataFrame, rule: str, seed: int, params: dict,
                n_splits: int = OOF_SPLITS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold 40 numbers for every row of `train`, (full, blank): each row is scored by a forest that
    never saw its group, and its blank version by the same forest."""
    y = labels(train)
    groups = make_groups(train, rule, n_splits).to_numpy()
    full, blank = [], []
    for fit_idx, held_idx in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(train, y, groups):
        fit_rows = train.iloc[fit_idx]
        fit_linked = has_link(fit_rows).to_numpy()
        rf = fit_rf(fit_rows[fit_linked][FEATURES_37], y[fit_idx][fit_linked], seed, **params)
        held = train.iloc[held_idx]
        full.append(lane_numbers(rf, held))
        blank.append(lane_numbers(rf, blank_page(held)))
    result = []
    for parts in (full, blank):
        out = pd.concat(parts).loc[train.index]      # back in row order; fails if a row was missed
        assert out.index.is_unique and len(out) == len(train), "a row was scored twice or not at all"
        result.append(out)
    return result[0], result[1]


def final_outputs(parts: dict, seed: int, params: dict) -> tuple:
    """(forest fit on all of parts["train"], the 40 numbers of the val and test rows with message_id and part,
    the same with the page hidden)."""
    train = parts["train"]
    train = train[has_link(train)]
    rf = fit_rf(train[FEATURES_37], labels(train), seed, **params)
    full, blank = [], []
    for name in ("val", "test"):
        for result, rows in ((full, parts[name]), (blank, blank_page(parts[name]))):
            out = lane_numbers(rf, rows)
            out.insert(0, "part", name)
            out.insert(0, "message_id", parts[name]["message_id"])
            result.append(out)
    return rf, pd.concat(full, ignore_index=True), pd.concat(blank, ignore_index=True)


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
            oof, oof_blank = oof_outputs(train, rule, seed, load_params(dataset, seed, fold))
            for frame, name in ((oof, "oof"), (oof_blank, "oof_blank")):
                frame.insert(0, "message_id", train["message_id"])
                frame.to_csv(out / f"{name}_seed{seed}_fold{fold}.csv", index=False)
            linked = has_link(train).to_numpy()                       # the forest speaks only for the messages that have a link
            y = labels(train)[linked]
            rows.append({"seed": seed, "fold": fold, "train_rows": len(train), "train_rows_with_link": int(linked.sum()),
                         "oof_f1": round(f1_score(y, oof["P"][linked] >= 0.5), 4),
                         "oof_log_loss": round(log_loss(y, oof["P"][linked].clip(1e-6, 1 - 1e-6), labels=[0, 1]), 4),
                         "mean_u": round(float(oof["u"][linked].mean()), 4),
                         "oof_f1_page_hidden": round(f1_score(y, oof_blank["P"][linked] >= 0.5), 4)})
            print(f"seed {seed} fold {fold}: oof F1 {rows[-1]['oof_f1']}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "oof_summary.csv", index=False)
    return summary


def load_lane(seed: int, fold: int, pilot: bool = False) -> dict:
    """{train, val, test} -> (full, blank) frames of message_id + the 40 numbers, for Phases 6-7."""
    out = out_dir(ROOT / (CROWD_PILOT_PATH if pilot else CROWD_DATASET_PATH))
    lane = {"train": tuple(pd.read_csv(out / f"{n}_seed{seed}_fold{fold}.csv") for n in ("oof", "oof_blank"))}
    final = tuple(pd.read_csv(out / f"{n}_seed{seed}_fold{fold}.csv") for n in ("final", "final_blank"))
    for part in ("val", "test"):
        lane[part] = tuple(f[f["part"] == part].drop(columns="part").reset_index(drop=True) for f in final)
    for full, blank in lane.values():
        assert list(full.columns) == ["message_id"] + LANE and "has_url" not in full.columns
        assert full["message_id"].equals(blank["message_id"]) and (blank["page_missing"] == 1).all()
    return lane


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
            rf, scored, scored_blank = final_outputs(parts, seed, params)
            joblib.dump({"model": rf, "features": list(FEATURES_37), "params": params},
                        models / f"link_rf_seed{seed}_fold{fold}.joblib")
            scored.to_csv(out / f"final_seed{seed}_fold{fold}.csv", index=False)
            scored_blank.to_csv(out / f"final_blank_seed{seed}_fold{fold}.csv", index=False)
            val = scored[scored["part"] == "val"]
            linked = has_link(parts["val"]).to_numpy()
            y = labels(parts["val"])[linked]
            rows.append({"seed": seed, "fold": fold, "val_rows": len(val), "val_rows_with_link": int(linked.sum()),
                         "test_rows_scored": int((scored["part"] == "test").sum()),
                         "val_f1": round(f1_score(y, val["P"][linked] >= 0.5), 4),
                         "val_log_loss": round(log_loss(y, val["P"][linked].clip(1e-6, 1 - 1e-6), labels=[0, 1]), 4),
                         "val_f1_page_hidden": round(f1_score(
                             y, scored_blank.loc[scored_blank["part"] == "val", "P"][linked] >= 0.5), 4)})
            print(f"seed {seed} fold {fold}: val F1 {rows[-1]['val_f1']}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "final_summary.csv", index=False)
    return summary


def forest_bias(rf) -> float:
    """The phishing fraction at the roots, averaged over the trees (the `bias` of P = bias + sum c_)."""
    roots = [t.tree_.value[0, 0, :] for t in rf.estimators_]
    return float(np.mean([r[1] / r.sum() for r in roots]))


def run_check(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS, atol: float = 1e-5) -> pd.DataFrame:
    dataset = ROOT / (CROWD_PILOT_PATH if pilot else CROWD_DATASET_PATH)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            lane = load_lane(seed, fold, pilot=pilot)
            parts = load_fold(seed, fold, pilot=pilot)
            bias = forest_bias(joblib.load(model_dir(dataset) / f"link_rf_seed{seed}_fold{fold}.joblib")["model"])

            def linked(frame, part):          # P = bias + sum c_ holds for the messages the forest scored: those with a link
                ids = set(parts[part].loc[has_link(parts[part]), "message_id"])
                return frame[frame["message_id"].isin(ids)]
            final_gap = max(float((linked(f, part)["P"] - linked(f, part)[CONTRIBUTIONS].sum(axis=1) - bias).abs().max())
                            for part in ("val", "test") for f in lane[part])
            oof_biases = max(int((linked(f, "train")["P"] - linked(f, "train")[CONTRIBUTIONS].sum(axis=1)).round(5).nunique())
                             for f in lane["train"])
            assert final_gap < atol, f"seed {seed} fold {fold}: P != bias + sum c_ (gap {final_gap})"
            assert oof_biases <= OOF_SPLITS, f"seed {seed} fold {fold}: {oof_biases} different oof biases"
            rows.append({"seed": seed, "fold": fold, "bias": round(bias, 4),
                         "final_max_gap": final_gap, "oof_distinct_biases": oof_biases})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir(dataset) / "check_summary.csv", index=False)
    print(f"P = bias + sum c_ holds on all {len(rows)} seed-folds (max gap {summary['final_max_gap'].max():.1e})")
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("step", choices=["tune", "oof", "final", "check"])
    a = ap.parse_args(argv)
    summary = {"tune": run_tune, "oof": run_oof, "final": run_final, "check": run_check}[a.step](a.pilot)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
