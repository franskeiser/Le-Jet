"""The C-prune Gate RF: ranks the 64 filters of the Phase A CNN (docs/c_prune_architecture.md Phase 5).

Step 1, scores: the 64 pooled filter scores of every TRAIN message, per seed and fold. Each fold's Phase A
CNN is rebuilt from its sidecar (text_cnn.load_phase_a, which refuses a changed file) and run in eval mode,
so dropout is off and the scores are the ones the Conv layer really sends on: ReLU, then the global max-pool
(TextEncoder.encode). The text is encoded with the fold's own vocabulary and SEQ_LEN_L, as in Phase A.
Only the train part is scored: the Gate RF learns and is judged on it alone (its own grouped folds, step 2),
so val and test stay unseen until fine-tuning and Phase 7.
Column filter_<i> is the CNN's filter i: the original ids, which the ranking, keep_idx and the
explanations (Phase 8) all refer to.

Step 2, importance: how much each filter helps the Gate RF, per seed and fold (the "oof_logloss" setting of
src/unified/improve_gates.py). The step 1 table is cut into 5 folds grouped by link_domain, so no domain is on
both sides. For each fold, a Gate RF (200 trees, otherwise sklearn defaults) is fit on the other 4 and scores
the held-out fold; then each filter's column is shuffled, 30 times, and the rise in log-loss is its importance
on that fold. Log-loss, not F1: it is smooth, so a filter that only makes the forest less sure still counts.
    importance   mean over the 5 folds (higher = the forest leans on it more; about 0 or below = no help)
    spread       standard error over the 5 folds (std / sqrt(5)), for "clearly above zero" (importance -
                 2 * spread > 0, the improve_gates rule)
The held-out predictions also show whether the Gate RF learns anything at all: its out-of-fold log-loss is
compared with the prior (always predicting the smishing share). A Gate RF no better than the prior gives a
ranking that is noise (section 7, "RF ranking ~ random").
The fitted forests are not saved: only the ranking is used, and the Gate RF is discarded (section 2.2).

Writes, per seed and fold, to experiments/c_prune/<dataset>/gate_rf/:
    filter_scores_seed<S>_fold<F>.csv   message_id, label (1 = smishing), filter_00 .. filter_63   (scores)
    importance_seed<S>_fold<F>.csv      filter, importance, spread, importance_fold0 .. 4          (importance)
and scores_summary.csv: rows, smishing share, and the dead filters (score 0 on every train message: they
can only be removed, whatever the ranking says); importance_summary.csv: the Gate RF's out-of-fold F1 and
log-loss against the prior, and how many filters are clearly above zero.

Usage:
    python -m src.unified.gate_rf --pilot scores
    python -m src.unified.gate_rf --pilot importance
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ARTIFACTS_ROOT, CROWD_N_FOLDS, SEEDS
from src.data.split_crowd import load_fold, make_groups
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.rf_extractor import fit_rf
from src.unified.text_cnn import dataset_path, load_phase_a, model_dir, tensors

BATCH_SIZE = 512
GATE_SPLITS = 5
GATE_REPEATS = 30
GATE_TREES = 200


def out_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "experiments" / "c_prune" / dataset.stem / "gate_rf"


def filter_columns(num_filters: int) -> List[str]:
    return [f"filter_{i:02d}" for i in range(num_filters)]


def filter_scores(cnn: TextEncoder, ids: torch.Tensor) -> np.ndarray:
    """(messages, filters) pooled scores, eval mode; the CNN's own mode is restored afterwards."""
    was_training = cnn.training
    cnn.eval()
    with torch.no_grad():
        scores = torch.cat([cnn.encode(batch) for batch in ids.split(BATCH_SIZE)])
    cnn.train(was_training)
    return scores.numpy()


def score_table(cnn: TextEncoder, part: pd.DataFrame, word_to_idx: dict, seq_len: int) -> pd.DataFrame:
    """message_id, label and one column per filter for the rows of `part` (same order)."""
    ids, labels = tensors(part, word_to_idx, seq_len)
    table = pd.DataFrame(filter_scores(cnn, ids), columns=filter_columns(cnn.conv.out_channels))
    table.insert(0, "label", labels.numpy())
    table.insert(0, "message_id", part["message_id"].to_numpy())
    return table


def run_scores(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    out = out_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            cnn, word_to_idx, sidecar = load_phase_a(model_dir(dataset) / f"phase_a_seed{seed}_fold{fold}.pt")
            train = load_fold(seed, fold, pilot=pilot)["train"]
            table = score_table(cnn, train, word_to_idx, sidecar["seq_len_l"])
            table.to_csv(out / f"filter_scores_seed{seed}_fold{fold}.csv", index=False)
            scores = table[filter_columns(sidecar["num_filters"])]
            assert len(table) == len(train) and (scores >= 0).all().all(), "pooled ReLU scores are never negative"
            dead = [int(c.split("_")[1]) for c in scores.columns if (scores[c] == 0).all()]
            rows.append({"seed": seed, "fold": fold, "train_rows": len(table),
                         "smishing_share": round(float(table["label"].mean()), 4),
                         "mean_score": round(float(scores.to_numpy().mean()), 4),
                         "dead_filters": len(dead), "dead_ids": " ".join(map(str, dead))})
            print(f"seed {seed} fold {fold}: {len(table)} train messages, {len(dead)} dead filters")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "scores_summary.csv", index=False)
    return summary


def gate_folds(y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = GATE_SPLITS) -> list:
    """(fit, held-out) index pairs; one group never sits on both sides."""
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(y, y, groups))


def gate_importance(table: pd.DataFrame, groups: np.ndarray, seed: int, n_splits: int = GATE_SPLITS,
                    n_repeats: int = GATE_REPEATS) -> tuple:
    """(importance frame, out-of-fold P(smishing)) of the filter columns of a step 1 table."""
    columns = [c for c in table.columns if c.startswith("filter_")]
    X, y = table[columns], table["label"].to_numpy()
    per_fold, oof = [], np.full(len(y), np.nan)
    for i, (fit_idx, held_idx) in enumerate(gate_folds(y, groups, seed, n_splits)):
        rf = fit_rf(X.iloc[fit_idx], y[fit_idx], seed, n_estimators=GATE_TREES)
        # One core per forest call and the columns in parallel instead: the result does not depend on the
        # worker count (random_state fixes every shuffle), and single-core sums are bit-for-bit repeatable.
        rf.set_params(n_jobs=1)
        oof[held_idx] = rf.predict_proba(X.iloc[held_idx])[:, 1]
        r = permutation_importance(rf, X.iloc[held_idx], y[held_idx], scoring="neg_log_loss",
                                   n_repeats=n_repeats, random_state=seed + i, n_jobs=-1)
        per_fold.append(r.importances_mean)   # neg_log_loss: the drop in score = the rise in log-loss
    per_fold = np.stack(per_fold)             # (folds, filters)
    importance = pd.DataFrame({"filter": [int(c.split("_")[1]) for c in columns],
                               "importance": per_fold.mean(axis=0),
                               "spread": per_fold.std(axis=0, ddof=1) / np.sqrt(n_splits)})
    for i, fold_values in enumerate(per_fold):
        importance[f"importance_fold{i}"] = fold_values
    return importance, oof


def run_importance(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    out = out_dir(dataset)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            table = pd.read_csv(out / f"filter_scores_seed{seed}_fold{fold}.csv")
            train = load_fold(seed, fold, pilot=pilot)["train"]
            domains = table[["message_id"]].merge(train[["message_id", "link_domain"]], on="message_id",
                                                  how="left", validate="one_to_one")
            assert len(domains) == len(train) and domains["link_domain"].notna().all(), "scores and train differ"
            groups = make_groups(domains, "domain").to_numpy()
            importance, oof = gate_importance(table, groups, seed)
            importance.to_csv(out / f"importance_seed{seed}_fold{fold}.csv", index=False)
            y = table["label"].to_numpy()
            clear = importance["importance"] - 2 * importance["spread"] > 0
            rows.append({"seed": seed, "fold": fold, "train_rows": len(table),
                         "oof_f1": round(f1_score(y, oof >= 0.5), 4),
                         "oof_log_loss": round(log_loss(y, oof, labels=[0, 1]), 4),
                         "prior_log_loss": round(log_loss(y, np.full(len(y), y.mean()), labels=[0, 1]), 4),
                         "clearly_above_zero": int(clear.sum()),
                         "at_or_below_zero": int((importance["importance"] <= 0).sum()),
                         "top5": " ".join(map(str, importance.nlargest(5, "importance")["filter"]))})
            print(f"seed {seed} fold {fold}: Gate RF oof log-loss {rows[-1]['oof_log_loss']} "
                  f"(prior {rows[-1]['prior_log_loss']}), {rows[-1]['clearly_above_zero']} filters clearly above 0")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "importance_summary.csv", index=False)
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("step", choices=["scores", "importance"])
    a = ap.parse_args(argv)
    summary = run_scores(a.pilot) if a.step == "scores" else run_importance(a.pilot)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
