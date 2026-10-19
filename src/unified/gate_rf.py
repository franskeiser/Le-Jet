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

Step 3, groups: the same importance, with similar filters shuffled TOGETHER. Filters that fire on the same
messages cover for each other: shuffle one and the forest reads its twin, so both look useless (the pilot: in
2 of 15 seed-folds no single filter was clearly above zero). Here:
  * filters are grouped by how alike their scores are: Spearman correlation over the train messages, then
    average-linkage clustering of 1 - |rho|, cut at |rho| = 0.8 (two groups join while their filters
    correlate at 0.8 or more on average). Labels are not used, so this reads nothing the forest is judged on.
    A dead filter correlates with nothing and stays alone;
  * per fold, the SAME forests as step 2 (same folds, same seed, refit) score the held-out fold with one
    group's columns shuffled by one shared row order, 30 times: the group keeps its own pattern, but loses
    its link to the label. The rise in log-loss is the group's importance; averaged over the 5 folds, with the
    same spread as step 2.

Step 4, rank: one order of the 64 filters, best first, from steps 2 and 3 (decided 2026-10-19). One filter per
group in turns, like picking teams:
  * groups are taken from the most to the least important (step 3; ties: lower group number first);
  * round 1 takes each group's best filter (highest step 2 importance; ties: lower filter id), round 2 each
    group's second best, and so on, until every filter is placed;
  * dead filters (score 0 on every train message) go last, whatever their group.
So a kept set of k holds one filter of each of the top k groups before any group gets a second: copies of one
pattern are removed first, and the RF decides which patterns stay. The top k of this order is keep_idx.

Step 4 check, compare: is this order better than the others? Measured on each fold's VAL messages, which no
ranking read, with the Phase A CNN and its temporary head, before any fine-tuning. Removing a filter is done
by setting its pooled score to 0: exactly what the deleted filter would send (section 2.3). For k = 48, 32, 16
it scores, by val log-loss (and F1):
    group_turns    this rule (C)
    single         the step 2 order alone (A)
    whole_groups   the step 3 groups in order, every member of a group before the next group (B)
    reverse        group_turns backwards: keep the least important, should do worst
    random         20 random kept sets (seeded), averaged: the order must beat this [S11]
Fine-tuning (Phase 6) can repair some of the damage, so this compares the orders, not the final models;
the Phase 7 controls stay the real test.

Step 5, keep: keep_idx for k = 48, 32, 16 (config.PRUNE_KEEP), by rf_gating_module.select_top_k: the top k of
the step 4 ranking, in ascending id order (kept position i of the pruned CNN is original filter keep_idx[i]).
Checked: each kept set lies inside the larger ones (the same ranking cut shorter). load_keep_idx gives one
set to the later phases.

Step 9, mask: the gate mask sidecar per seed and fold, next to its Phase A checkpoint, for k = 48, 32, 16.
It holds keep_idx and the gate mask (64 numbers, 1 kept and 0 removed) per k, the full ranking, the ranking
rule (RANKING_RULE), the Gate RF settings, and the sha256 of the Phase A checkpoint and of the ranking files.
load_gate_mask refuses it when any of those files changed (a filter id only means something for the CNN the
mask was made from), and checks it agrees with itself: the mask's 1s are keep_idx, keep_idx is the top k of the
ranking. Every later step reads keep_idx from here (pruned_phase_a included). Run it before prune.

Phase 6, lock: the ranking rule, decided once on the real data's val (docs/c_prune_architecture.md Phase 6). From
the compare step: keep group_turns unless single has the lower mean val_log_loss_refit over every seed, fold
and k. Writes experiments/c_prune/<dataset>/gate_rf/ranking_rule_decision.json (date, rule, the numbers) and
refuses to run again once it exists. The mask step then refuses a RANKING_RULE other than the locked one. On
the pilot it is a dry run only: it prints what it would decide and locks nothing.

Step 6, prune: cnn_encoder.prune_encoder builds the k-filter CNN from the Phase A CNN and keep_idx: the kept
filters' Conv rows and biases, and the matching columns of the layer reading them (prune_input_columns; here
the temporary head, in Phase 6 any Dense that reads [filters | 40 link numbers]). pruned_phase_a gives one to
Phase 6. Checked for every seed, fold and k: the shapes, the embedding still frozen, and the trainable
parameters equal section 5.3 (Conv 300 x 3 x k + k, temporary head 2k + 2). Nothing is saved here: the pruned
CNN is saved after fine-tuning (Phase 6).
Step 7 check, in the same run: on each fold's val messages, the pruned CNN against the full CNN with its gate
mask applied (gated_copy: W <- g * W, b <- g * b, g = 0 for removed filters). The removed filters must send
exactly 0, the kept scores must be equal bit for bit, and the output equal up to float rounding (< 1e-5: the
head sums k terms instead of 64 with zeros). The unit tests in tests/test_prune.py check the same on toy models.
Step 8 check, in the same run: pruned with all 64 filters kept, every Phase A CNN gives exactly the same scores
and output on its val messages, bit for bit (same sizes, same weights, same order: nothing to round).

Writes, per seed and fold, to experiments/c_prune/<dataset>/gate_rf/:
    filter_scores_seed<S>_fold<F>.csv     message_id, label (1 = smishing), filter_00 .. filter_63    (scores)
    importance_seed<S>_fold<F>.csv        filter, importance, spread, importance_fold0 .. 4           (importance)
    group_importance_seed<S>_fold<F>.csv  group, size, filters, importance, spread, importance_fold0 .. 4 (groups)
    filter_importance_seed<S>_fold<F>.csv rank (0 = best), filter, group, pick (round in its group), dead,
                                          group_importance, importance, spread                         (rank)
and scores_summary.csv: rows, smishing share, and the dead filters (score 0 on every train message: they
can only be removed, whatever the ranking says); importance_summary.csv: the Gate RF's out-of-fold F1 and
log-loss against the prior, and how many filters are clearly above zero; group_summary.csv: the number of
groups, the largest, and how many groups (and the filters in them) are clearly above zero;
ranking_compare.csv (every seed, fold, k and order) and ranking_compare_summary.csv (per k and order: mean
val log-loss and F1, and in how many seed-folds group_turns has the lower log-loss); keep_summary.csv (per
seed, fold and k: keep_idx, and how many of the groups and dead filters it holds); prune_summary.csv (per
seed, fold and k: the pruned CNN's trainable parameters against the full one's, and the step 7 gaps);
mask_summary.csv. And to models/c_prune/<dataset>/:
    gate_mask_seed<S>_fold<F>.json        the step 9 sidecar                                          (mask)

Usage:
    python -m src.unified.gate_rf --pilot scores
    python -m src.unified.gate_rf --pilot importance
    python -m src.unified.gate_rf --pilot groups
    python -m src.unified.gate_rf --pilot rank
    python -m src.unified.gate_rf --pilot compare
    python -m src.unified.gate_rf --pilot keep
    python -m src.unified.gate_rf --pilot mask
    python -m src.unified.gate_rf --pilot prune      (reads the masks)
    python -m src.unified.gate_rf --pilot lock       (a dry run on the pilot; on the real data it locks, once)
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ARTIFACTS_ROOT, CROWD_N_FOLDS, PRUNE_KEEP, SEEDS
from src.data.split_crowd import load_fold, make_groups
from src.features.model.cnn_encoder import TextEncoder, prune_encoder
from src.features.model.rf_extractor import fit_rf
from src.features.model.rf_gating_module import select_top_k
from src.unified.text_cnn import dataset_path, load_phase_a, model_dir, sha256, tensors, trainable_parameters

BATCH_SIZE = 512
GATE_SPLITS = 5
GATE_REPEATS = 30
GATE_TREES = 200
CORR_THRESHOLD = 0.8
RANDOM_DRAWS = 20
ORDERS = ["group_turns", "single", "whole_groups", "reverse", "random"]
# The ranking rule the gate masks are made with (step 4; locked on the real data's val, Phase 6): group_turns
# (one filter per group in turns) or single (step 2 alone). After a change, run the mask step again.
RANKING_RULE = "group_turns"


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


def score_columns(table: pd.DataFrame) -> List[str]:
    return [c for c in table.columns if c.startswith("filter_")]


def gate_forests(table: pd.DataFrame, groups: np.ndarray, seed: int, n_splits: int = GATE_SPLITS):
    """Per fold: (fold, Gate RF fit on the other folds, held-out X, held-out y, held-out row indices)."""
    X, y = table[score_columns(table)], table["label"].to_numpy()
    for i, (fit_idx, held_idx) in enumerate(gate_folds(y, groups, seed, n_splits)):
        rf = fit_rf(X.iloc[fit_idx], y[fit_idx], seed, n_estimators=GATE_TREES)
        # One core per forest call and the columns in parallel instead: the result does not depend on the
        # worker count (random_state fixes every shuffle), and single-core sums are bit-for-bit repeatable.
        rf.set_params(n_jobs=1)
        yield i, rf, X.iloc[held_idx], y[held_idx], held_idx


def importance_frame(per_fold: List[np.ndarray], first: dict, n_splits: int) -> pd.DataFrame:
    """`first` columns, then the mean over the folds, its standard error and one column per fold."""
    per_fold = np.stack(per_fold)             # (folds, filters or groups)
    frame = pd.DataFrame({**first, "importance": per_fold.mean(axis=0),
                          "spread": per_fold.std(axis=0, ddof=1) / np.sqrt(n_splits)})
    for i, fold_values in enumerate(per_fold):
        frame[f"importance_fold{i}"] = fold_values
    return frame


def gate_importance(table: pd.DataFrame, groups: np.ndarray, seed: int, n_splits: int = GATE_SPLITS,
                    n_repeats: int = GATE_REPEATS) -> tuple:
    """(importance frame, out-of-fold P(smishing)) of the filter columns of a step 1 table."""
    per_fold, oof = [], np.full(len(table), np.nan)
    for i, rf, X_held, y_held, held_idx in gate_forests(table, groups, seed, n_splits):
        oof[held_idx] = rf.predict_proba(X_held)[:, 1]
        r = permutation_importance(rf, X_held, y_held, scoring="neg_log_loss",
                                   n_repeats=n_repeats, random_state=seed + i, n_jobs=-1)
        per_fold.append(r.importances_mean)   # neg_log_loss: the drop in score = the rise in log-loss
    filters = [int(c.split("_")[1]) for c in score_columns(table)]
    return importance_frame(per_fold, {"filter": filters}, n_splits), oof


def correlated_groups(table: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> np.ndarray:
    """A group number per filter column (0, 1, ... in order of each group's first filter)."""
    rho = table[score_columns(table)].corr("spearman").fillna(0.0).abs().to_numpy().copy()
    np.fill_diagonal(rho, 1.0)               # a dead (constant) filter: NaN -> 0, alike with nothing but itself
    labels = fcluster(linkage(squareform(1 - rho, checks=False), "average"), 1 - threshold, "distance")
    order = {label: n for n, label in enumerate(dict.fromkeys(labels))}
    return np.array([order[label] for label in labels])


def shuffled_log_loss(rf, X_held: pd.DataFrame, y_held: np.ndarray, members: np.ndarray, rng_seed: list,
                      n_repeats: int) -> float:
    """Mean log-loss over `n_repeats` shuffles of the `members` columns, all by one row order per shuffle."""
    rng = np.random.default_rng(rng_seed)
    losses = []
    for _ in range(n_repeats):
        shuffled = X_held.copy()
        shuffled.iloc[:, members] = X_held.iloc[rng.permutation(len(X_held)), members].to_numpy()
        losses.append(log_loss(y_held, rf.predict_proba(shuffled)[:, 1], labels=[0, 1]))
    return float(np.mean(losses))


def group_importance(table: pd.DataFrame, groups: np.ndarray, filter_groups: np.ndarray, seed: int,
                     n_splits: int = GATE_SPLITS, n_repeats: int = GATE_REPEATS) -> pd.DataFrame:
    """Importance of each group of filter columns, shuffled together; step 2's folds and forests."""
    members = [np.flatnonzero(filter_groups == g) for g in range(filter_groups.max() + 1)]
    per_fold = []
    for i, rf, X_held, y_held, _ in gate_forests(table, groups, seed, n_splits):
        base = log_loss(y_held, rf.predict_proba(X_held)[:, 1], labels=[0, 1])
        shuffled = Parallel(n_jobs=-1)(delayed(shuffled_log_loss)(rf, X_held, y_held, m, [seed, i, g], n_repeats)
                                       for g, m in enumerate(members))
        per_fold.append(np.array(shuffled) - base)
    filters = [int(c.split("_")[1]) for c in score_columns(table)]
    first = {"group": list(range(len(members))), "size": [len(m) for m in members],
             "filters": [" ".join(str(filters[j]) for j in m) for m in members]}
    return importance_frame(per_fold, first, n_splits)


def domain_groups(table: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """The link_domain group of every row of a step 1 table, from the fold's train part."""
    domains = table[["message_id"]].merge(train[["message_id", "link_domain"]], on="message_id",
                                          how="left", validate="one_to_one")
    assert len(domains) == len(train) and domains["link_domain"].notna().all(), "scores and train differ"
    return make_groups(domains, "domain").to_numpy()


def run_importance(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    out = out_dir(dataset)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            table = pd.read_csv(out / f"filter_scores_seed{seed}_fold{fold}.csv")
            groups = domain_groups(table, load_fold(seed, fold, pilot=pilot)["train"])
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


def run_groups(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    out = out_dir(dataset)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            table = pd.read_csv(out / f"filter_scores_seed{seed}_fold{fold}.csv")
            groups = domain_groups(table, load_fold(seed, fold, pilot=pilot)["train"])
            frame = group_importance(table, groups, correlated_groups(table), seed)
            frame.to_csv(out / f"group_importance_seed{seed}_fold{fold}.csv", index=False)
            clear = frame["importance"] - 2 * frame["spread"] > 0
            rows.append({"seed": seed, "fold": fold, "groups": len(frame), "largest": int(frame["size"].max()),
                         "singletons": int((frame["size"] == 1).sum()),
                         "groups_clearly_above_zero": int(clear.sum()),
                         "filters_in_those_groups": int(frame.loc[clear, "size"].sum()),
                         "top3": " | ".join(frame.nlargest(3, "importance")["filters"])})
            print(f"seed {seed} fold {fold}: {len(frame)} groups, {rows[-1]['groups_clearly_above_zero']} clearly "
                  f"above 0 ({rows[-1]['filters_in_those_groups']} filters)")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "group_summary.csv", index=False)
    return summary


def group_members(single: pd.DataFrame, grouped: pd.DataFrame, dead: set) -> List[List[int]]:
    """The live filters of each group, best first; the groups from most to least important."""
    by_filter = single.set_index("filter")["importance"]
    members = []
    for _, group in grouped.sort_values(["importance", "group"], ascending=[False, True]).iterrows():
        live = [f for f in map(int, group["filters"].split()) if f not in dead]
        members.append(sorted(live, key=lambda f: (-by_filter[f], f)))
    return members


def rank_filters(single: pd.DataFrame, grouped: pd.DataFrame, dead: set) -> pd.DataFrame:
    """Step 4: one filter per group in turns, dead filters last. One row per filter, best first."""
    members = group_members(single, grouped, dead)
    group_of = {f: int(g) for g, filters in zip(grouped["group"], grouped["filters"]) for f in map(int, filters.split())}
    order, picks = [], []
    for pick in range(max(map(len, members), default=0)):
        for live in members:
            if pick < len(live):
                order.append(live[pick])
                picks.append(pick)
    for f in sorted(dead):
        order.append(f)
        picks.append(-1)
    assert sorted(order) == sorted(single["filter"]), "every filter is ranked exactly once"
    by_filter = single.set_index("filter")
    group_importance = grouped.set_index("group")["importance"]
    return pd.DataFrame({"rank": range(len(order)), "filter": order, "group": [group_of[f] for f in order],
                         "pick": picks, "dead": [f in dead for f in order],
                         "group_importance": [group_importance[group_of[f]] for f in order],
                         "importance": by_filter.loc[order, "importance"].to_numpy(),
                         "spread": by_filter.loc[order, "spread"].to_numpy()})


def dead_filters(table: pd.DataFrame) -> set:
    return {int(c.split("_")[1]) for c in score_columns(table) if (table[c] == 0).all()}


def other_orders(single: pd.DataFrame, grouped: pd.DataFrame, dead: set, ranking: pd.DataFrame) -> dict:
    """The orders group_turns is compared with (the random sets are drawn separately)."""
    live_single = single[~single["filter"].isin(dead)].sort_values(["importance", "filter"], ascending=[False, True])
    return {"group_turns": ranking["filter"].tolist(),
            "single": live_single["filter"].tolist() + sorted(dead),
            "whole_groups": [f for live in group_members(single, grouped, dead) for f in live] + sorted(dead),
            "reverse": ranking["filter"].tolist()[::-1]}


def run_rank(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    out = out_dir(dataset_path(pilot))
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            name = f"seed{seed}_fold{fold}.csv"
            dead = dead_filters(pd.read_csv(out / f"filter_scores_{name}"))
            ranking = rank_filters(pd.read_csv(out / f"importance_{name}"),
                                   pd.read_csv(out / f"group_importance_{name}"), dead)
            ranking.to_csv(out / f"filter_importance_{name}", index=False)
            rows.append({"seed": seed, "fold": fold, "dead": len(dead),
                         **{f"groups_in_top{k}": ranking["group"].head(k).nunique() for k in PRUNE_KEEP},
                         "top5": " ".join(map(str, ranking["filter"].head(5)))})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "rank_summary.csv", index=False)
    return summary


def kept_scores(pooled: torch.Tensor, keep: List[int]) -> torch.Tensor:
    """Pooled scores with every filter outside `keep` set to 0, as a deleted filter would send."""
    mask = torch.zeros(pooled.shape[1])
    mask[keep] = 1.0
    return pooled * mask


def val_scores(cnn: TextEncoder, pooled: torch.Tensor, y: np.ndarray, train: tuple, keep: List[int]) -> dict:
    """Val log-loss and F1 with only the `keep` filters, two ways:
    as is      the Phase A head unchanged (it still expects the removed filters' votes)
    refit      a new head (logistic regression) fit on the TRAIN messages' kept scores: a quick stand-in for
               the fine-tuning of Phase 6, which lets the kept filters take over the removed ones' weight
    """
    with torch.no_grad():
        p = torch.softmax(cnn.temp_head(kept_scores(pooled, keep)), dim=1)[:, 1].numpy()
    train_X, train_y = train
    head = LogisticRegression(max_iter=2000).fit(train_X[:, keep], train_y)
    q = head.predict_proba(pooled.numpy()[:, keep])[:, 1]
    return {"val_log_loss": log_loss(y, p, labels=[0, 1]), "val_f1": f1_score(y, p >= 0.5, zero_division=0),
            "val_log_loss_refit": log_loss(y, q, labels=[0, 1]), "val_f1_refit": f1_score(y, q >= 0.5, zero_division=0)}


def run_compare(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    out = out_dir(dataset)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            name = f"seed{seed}_fold{fold}.csv"
            cnn, word_to_idx, sidecar = load_phase_a(model_dir(dataset) / f"phase_a_seed{seed}_fold{fold}.pt")
            ids, labels = tensors(load_fold(seed, fold, pilot=pilot)["val"], word_to_idx, sidecar["seq_len_l"])
            with torch.no_grad():
                pooled = cnn.encode(ids)          # eval mode (load_phase_a): dropout off
            y = labels.numpy()
            table = pd.read_csv(out / f"filter_scores_{name}")
            train = (table[score_columns(table)].to_numpy(), table["label"].to_numpy())
            dead = dead_filters(table)
            single, grouped = pd.read_csv(out / f"importance_{name}"), pd.read_csv(out / f"group_importance_{name}")
            orders = other_orders(single, grouped, dead, pd.read_csv(out / f"filter_importance_{name}"))
            base = {"seed": seed, "fold": fold}
            rows.append({**base, "k": pooled.shape[1], "order": "all",
                         **val_scores(cnn, pooled, y, train, list(range(pooled.shape[1])))})
            rng = np.random.default_rng([seed, fold])
            for k in PRUNE_KEEP:
                for order, filters in orders.items():
                    rows.append({**base, "k": k, "order": order, **val_scores(cnn, pooled, y, train, filters[:k])})
                draws = pd.DataFrame([val_scores(cnn, pooled, y, train,
                                                 rng.choice(pooled.shape[1], k, replace=False).tolist())
                                      for _ in range(RANDOM_DRAWS)])
                rows.append({**base, "k": k, "order": "random", **draws.mean().to_dict()})
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "ranking_compare.csv", index=False)
    summary = compare_summary(detail)
    summary.to_csv(out / "ranking_compare_summary.csv", index=False)
    return summary


def compare_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per k and order: the mean metrics, and in how many seed-folds group_turns has the lower log-loss."""
    metrics = ["val_log_loss", "val_f1", "val_log_loss_refit", "val_f1_refit"]
    summary = detail.groupby(["k", "order"], sort=False)[metrics].mean().round(4).reset_index()
    pruned = detail[detail["order"] != "all"]
    for metric, column in (("val_log_loss", "turns_lower_as_is"), ("val_log_loss_refit", "turns_lower_refit")):
        wide = pruned.pivot_table(index=["seed", "fold", "k"], columns="order", values=metric)
        wins = []
        for k, order in zip(summary["k"], summary["order"]):
            if order in ("all", "group_turns"):
                wins.append("")
                continue
            part = wide.xs(k, level="k")
            wins.append(f"{int((part['group_turns'] < part[order]).sum())}/{len(part)}")
        summary[column] = wins
    return summary


def ranking_inputs(seed: int, fold: int, rule: str, pilot: bool = False) -> List[Path]:
    """The files one rule's order is made from."""
    out, name = out_dir(dataset_path(pilot)), f"seed{seed}_fold{fold}.csv"
    if rule == "group_turns":
        return [out / f"filter_importance_{name}"]
    if rule == "single":
        return [out / f"importance_{name}", out / f"filter_scores_{name}"]
    raise ValueError(f"unknown ranking rule {rule!r}: use group_turns or single")


def ranking_order(seed: int, fold: int, rule: str = None, pilot: bool = False) -> List[int]:
    """Every filter id once, best first, by `rule` (default RANKING_RULE)."""
    rule = rule or RANKING_RULE
    files = ranking_inputs(seed, fold, rule, pilot)
    if rule == "group_turns":
        return pd.read_csv(files[0]).sort_values("rank")["filter"].astype(int).tolist()
    single, dead = pd.read_csv(files[0]), dead_filters(pd.read_csv(files[1]))
    live = single[~single["filter"].isin(dead)].sort_values(["importance", "filter"], ascending=[False, True])
    return live["filter"].astype(int).tolist() + sorted(dead)


def load_keep_idx(seed: int, fold: int, k: int, pilot: bool = False) -> np.ndarray:
    """keep_idx of one seed and fold: the top k of its ranking (RANKING_RULE), ascending ids."""
    return select_top_k(ranking_order(seed, fold, pilot=pilot), k)


def gate_mask_path(seed: int, fold: int, pilot: bool = False) -> Path:
    return model_dir(dataset_path(pilot)) / f"gate_mask_seed{seed}_fold{fold}.json"


def write_gate_mask(seed: int, fold: int, pilot: bool = False) -> dict:
    """Step 9: the gate mask sidecar of one seed and fold, next to its Phase A checkpoint."""
    locked = locked_rule(pilot)
    if locked is not None and locked != RANKING_RULE:
        raise ValueError(f'the ranking rule is locked to {locked}: set RANKING_RULE = "{locked}" in gate_rf.py')
    checkpoint = model_dir(dataset_path(pilot)) / f"phase_a_seed{seed}_fold{fold}.pt"
    phase_a = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    ranking = ranking_order(seed, fold, pilot=pilot)
    assert len(ranking) == phase_a["num_filters"], "the ranking and the Phase A CNN differ in filters"
    keep = {}
    for k in PRUNE_KEEP:
        keep_idx = select_top_k(ranking, k)
        mask = np.zeros(len(ranking), dtype=int)
        mask[keep_idx] = 1
        keep[str(k)] = {"keep_idx": keep_idx.tolist(), "gate_mask": mask.tolist()}
    sidecar = {
        "seed": seed, "fold": fold, "num_filters": len(ranking), "ranking_rule": RANKING_RULE,
        "phase_a_checkpoint": checkpoint.name,
        "gate_rf": {"splits": GATE_SPLITS, "repeats": GATE_REPEATS, "trees": GATE_TREES,
                    "corr_threshold": CORR_THRESHOLD, "scoring": "log_loss", "grouped_by": "link_domain"},
        "sha256": {checkpoint.name: sha256(checkpoint),
                   **{f.relative_to(ARTIFACTS_ROOT).as_posix(): sha256(f)
                      for f in ranking_inputs(seed, fold, RANKING_RULE, pilot)}},
        "ranking": ranking,
        "keep": keep,
    }
    gate_mask_path(seed, fold, pilot).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def load_gate_mask(seed: int, fold: int, k: int, pilot: bool = False) -> np.ndarray:
    """keep_idx from the gate mask sidecar; refuses it if a file it was made from changed or it disagrees inside."""
    path = gate_mask_path(seed, fold, pilot)
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in sidecar["sha256"].items():
        source = path.parent / name if name == sidecar["phase_a_checkpoint"] else ARTIFACTS_ROOT / name
        if sha256(source) != digest:
            raise ValueError(f"{source} changed since the gate mask was made: run the mask step again")
    if str(k) not in sidecar["keep"]:
        raise ValueError(f"no gate mask for k = {k} in {path.name} (it has {', '.join(sidecar['keep'])})")
    keep_idx = np.array(sidecar["keep"][str(k)]["keep_idx"])
    mask = np.array(sidecar["keep"][str(k)]["gate_mask"])
    assert np.array_equal(np.flatnonzero(mask), keep_idx), f"{path.name}: gate mask and keep_idx differ"
    assert np.array_equal(select_top_k(sidecar["ranking"], k), keep_idx), f"{path.name}: keep_idx is not the top {k}"
    return keep_idx


def run_mask(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            write_gate_mask(seed, fold, pilot)
            for k in PRUNE_KEEP:
                keep = load_gate_mask(seed, fold, k, pilot)   # read back through every check
                assert np.array_equal(keep, load_keep_idx(seed, fold, k, pilot))
                rows.append({"seed": seed, "fold": fold, "k": k, "rule": RANKING_RULE, "kept": len(keep),
                             "file": gate_mask_path(seed, fold, pilot).name})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir(dataset_path(pilot)) / "mask_summary.csv", index=False)
    return summary


def run_keep(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    out = out_dir(dataset_path(pilot))
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            ranking = pd.read_csv(out / f"filter_importance_seed{seed}_fold{fold}.csv").set_index("filter")
            kept = {k: load_keep_idx(seed, fold, k, pilot) for k in sorted(PRUNE_KEEP, reverse=True)}
            sizes = sorted(kept)
            # The same ranking cut shorter: a smaller kept set lies inside every larger one.
            assert all(set(kept[a]) <= set(kept[b]) for a, b in zip(sizes, sizes[1:])), "kept sets not nested"
            for k, keep in kept.items():
                rows.append({"seed": seed, "fold": fold, "k": k, "kept": len(keep),
                             "groups_kept": ranking.loc[keep, "group"].nunique(),
                             "groups_total": ranking["group"].nunique(),
                             "dead_kept": int(ranking.loc[keep, "dead"].sum()),
                             "keep_idx": " ".join(map(str, keep))})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "keep_summary.csv", index=False)
    return summary


def pruned_phase_a(seed: int, fold: int, k: int, pilot: bool = False) -> tuple:
    """(k-filter CNN in eval mode, keep_idx, word_to_idx, Phase A sidecar) of one seed and fold. For Phase 6."""
    dataset = dataset_path(pilot)
    cnn, word_to_idx, sidecar = load_phase_a(model_dir(dataset) / f"phase_a_seed{seed}_fold{fold}.pt")
    keep = load_gate_mask(seed, fold, k, pilot)
    return prune_encoder(cnn, keep), keep, word_to_idx, sidecar


def gated_copy(cnn: TextEncoder, keep) -> TextEncoder:
    """The full CNN with its gate mask applied to the Conv weights: W <- g * W, b <- g * b (g = 1 kept, 0 removed)."""
    gated = copy.deepcopy(cnn)
    g = torch.zeros(cnn.conv.out_channels)
    g[torch.as_tensor(keep)] = 1.0
    with torch.no_grad():
        gated.conv.weight.mul_(g[:, None, None])
        gated.conv.bias.mul_(g)
    return gated


def pruning_gaps(cnn: TextEncoder, pruned: TextEncoder, keep, ids: torch.Tensor) -> tuple:
    """(largest score gap, largest output gap) between the pruned CNN and the gated full CNN, eval mode."""
    gated = gated_copy(cnn, keep)
    with torch.no_grad():
        scores = gated.encode(ids)
        score_gap = (pruned.encode(ids) - scores[:, torch.as_tensor(keep)]).abs().max().item()
        removed = [i for i in range(cnn.conv.out_channels) if i not in set(np.asarray(keep).tolist())]
        assert not removed or scores[:, removed].abs().max().item() == 0.0, "a gated filter must send exactly 0"
        output_gap = (pruned(ids) - gated(ids)).abs().max().item()
    return score_gap, output_gap


def keep_all_unchanged(cnn: TextEncoder, ids: torch.Tensor) -> bool:
    """Step 8: pruned with every filter kept, the CNN gives exactly the same output, bit for bit."""
    same = prune_encoder(cnn, list(range(cnn.conv.out_channels)))
    with torch.no_grad():
        return torch.equal(same.encode(ids), cnn.encode(ids)) and torch.equal(same(ids), cnn(ids))


def run_prune(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            cnn, word_to_idx, sidecar = load_phase_a(model_dir(dataset_path(pilot)) / f"phase_a_seed{seed}_fold{fold}.pt")
            ids, _ = tensors(load_fold(seed, fold, pilot=pilot)["val"], word_to_idx, sidecar["seq_len_l"])
            assert keep_all_unchanged(cnn, ids), f"seed {seed} fold {fold}: keeping all 64 filters changed the output"
            for k in PRUNE_KEEP:
                pruned, keep, _, _ = pruned_phase_a(seed, fold, k, pilot)
                score_gap, output_gap = pruning_gaps(cnn, pruned, keep, ids)
                assert score_gap == 0.0 and output_gap < 1e-5, f"seed {seed} fold {fold} k {k}: pruning not exact"
                dim = pruned.embedding.embedding_dim
                conv, head = dim * sidecar["kernel_size"] * k + k, k * 2 + 2   # section 5.3
                assert pruned.conv.weight.shape == (k, dim, sidecar["kernel_size"]) and pruned.temp_head.in_features == k
                assert not pruned.embedding.weight.requires_grad and not pruned.training
                assert trainable_parameters(pruned) == conv + head, "trainable parameters differ from section 5.3"
                rows.append({"seed": seed, "fold": fold, "k": k, "conv_parameters": conv, "head_parameters": head,
                             "trainable_parameters": trainable_parameters(pruned),
                             "full_trainable_parameters": sidecar["trainable_parameters"],
                             "val_score_gap": score_gap, "val_output_gap": output_gap})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir(dataset_path(pilot)) / "prune_summary.csv", index=False)
    return summary


def ranking_decision(detail: pd.DataFrame) -> dict:
    """The plan's rule (Phase 6): keep group_turns unless single has the lower mean val_log_loss_refit, over every
    seed, fold and k of the compare step."""
    pair = detail[detail["order"].isin(["group_turns", "single"])]
    means = pair.groupby("order")["val_log_loss_refit"].mean()
    wide = pair.pivot_table(index=["seed", "fold", "k"], columns="order", values="val_log_loss_refit")
    per_k = pair.pivot_table(index="k", columns="order", values="val_log_loss_refit")
    return {"rule": "single" if means["single"] < means["group_turns"] else "group_turns",
            "mean_val_log_loss_refit": {o: round(float(v), 6) for o, v in means.items()},
            "per_k": {str(k): {o: round(float(v), 6) for o, v in row.items()} for k, row in per_k.iterrows()},
            "single_lower_in": f"{int((wide['single'] < wide['group_turns']).sum())}/{len(wide)}"}


def decision_path(pilot: bool = False) -> Path:
    return out_dir(dataset_path(pilot)) / "ranking_rule_decision.json"


def locked_rule(pilot: bool = False) -> str | None:
    """The locked ranking rule of the real data, or None before the lock (and always on the pilot)."""
    path = decision_path(pilot)
    return None if pilot or not path.exists() else json.loads(path.read_text(encoding="utf-8"))["rule"]


def run_lock(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    """Phase 6, lock the ranking rule: once, on the real data's val. The pilot only gets a dry run."""
    detail = pd.read_csv(out_dir(dataset_path(pilot)) / "ranking_compare.csv")
    expected = {(s, f) for s in seeds for f in range(n_folds)}
    assert set(zip(detail["seed"], detail["fold"])) == expected, "the compare step did not cover every seed and fold"
    decision = ranking_decision(detail)
    if pilot:
        print(f"DRY RUN on the synthetic pilot, nothing locked: the rule would be {decision['rule']}")
    else:
        path = decision_path(pilot)
        if path.exists():
            raise SystemExit(f"the ranking rule is already locked ({path}); it is decided once and never changed")
        path.write_text(json.dumps({"locked_on": date.today().isoformat(), "dataset": dataset_path(pilot).name,
                                    "metric": "mean val_log_loss_refit over every seed, fold and k (lower wins)",
                                    "rule_before": RANKING_RULE, **decision}, indent=2), encoding="utf-8")
        print(f"LOCKED: the ranking rule is {decision['rule']} ({path})")
    if decision["rule"] != RANKING_RULE:
        print(f'next: set RANKING_RULE = "{decision["rule"]}" in src/unified/gate_rf.py, then run the mask and '
              "prune steps and fusion train again")
    rows = [{"order": o, "mean_val_log_loss_refit": v} for o, v in decision["mean_val_log_loss_refit"].items()]
    return pd.DataFrame(rows + [{"order": f"rule: {decision['rule']}",
                                 "mean_val_log_loss_refit": f"single lower in {decision['single_lower_in']}"}])


STEPS = {"scores": run_scores, "importance": run_importance, "groups": run_groups, "rank": run_rank,
         "compare": run_compare, "keep": run_keep, "mask": run_mask, "prune": run_prune, "lock": run_lock}


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("step", choices=list(STEPS))
    a = ap.parse_args(argv)
    summary = STEPS[a.step](a.pilot)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
