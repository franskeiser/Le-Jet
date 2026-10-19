"""
Tests for src/unified/gate_rf.py (docs/c_prune_architecture.md Phase 5).

Run with:  python -m pytest tests/test_gate_rf.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.features.model.cnn_encoder import TextEncoder
from src.features.model.embedding import build_torch_embedding_layer
from src.unified import gate_rf as gr
from src.unified import text_cnn as tc

VOCAB = ["<PAD>", "<URL>", "verify", "now", "account", "locked", "your", "statement", "is", "ready"]


def _cnn():
    word_to_idx = {w: i for i, w in enumerate(VOCAB)}
    matrix = np.random.default_rng(0).normal(size=(len(VOCAB), 8)).astype(np.float32)
    matrix[0] = 0.0
    torch.manual_seed(0)
    return TextEncoder(build_torch_embedding_layer(matrix, word_to_idx), 64), word_to_idx


def _part(n=3):
    return pd.DataFrame({"message_id": [f"m{i}" for i in range(2 * n)],
                         "sms_text": ["verify now your account is locked https://a.xyz",
                                      "your statement is ready https://b.com"] * n,
                         "label": ["smishing", "legitimate"] * n})


def test_scores_are_eval_mode_pooled_conv_outputs():
    cnn, word_to_idx = _cnn()
    cnn.train()   # dropout on: the step must switch it off, then put the mode back
    part = _part()
    table = gr.score_table(cnn, part, word_to_idx, seq_len=8)
    assert cnn.training
    assert list(table.columns) == ["message_id", "label"] + [f"filter_{i:02d}" for i in range(64)]
    assert table["message_id"].tolist() == part["message_id"].tolist()
    assert table["label"].tolist() == [1, 0] * 3
    ids, _ = tc.tensors(part, word_to_idx, 8)
    cnn.eval()
    with torch.no_grad():
        expected = torch.relu(cnn.conv(cnn.embedding(ids).transpose(1, 2))).max(dim=2).values.numpy()
    assert np.array_equal(table.iloc[:, 2:].to_numpy(), expected)


def _score_table(n=200):
    """filter_00 follows the label, filter_01 is weaker, filter_02 / 03 are noise; 40 domains of 5 rows."""
    rng = np.random.default_rng(0)
    y = np.tile([1, 0], n // 2)
    table = pd.DataFrame({"message_id": [f"m{i}" for i in range(n)], "label": y,
                          "filter_00": y + rng.normal(0, 0.3, n), "filter_01": y + rng.normal(0, 1.5, n),
                          "filter_02": rng.normal(size=n), "filter_03": rng.normal(size=n)})
    return table, np.array([f"d:{i // 5}" for i in range(n)])


def test_gate_folds_keep_each_domain_on_one_side():
    table, groups = _score_table()
    folds = gr.gate_folds(table["label"].to_numpy(), groups, seed=42)
    assert len(folds) == gr.GATE_SPLITS
    for fit_idx, held_idx in folds:
        assert not set(groups[fit_idx]) & set(groups[held_idx])
    assert sorted(np.concatenate([held for _, held in folds]).tolist()) == list(range(len(table)))


def test_importance_ranks_the_informative_filter_first(monkeypatch):
    monkeypatch.setattr(gr, "GATE_TREES", 50)
    table, groups = _score_table()
    importance, oof = gr.gate_importance(table, groups, seed=42, n_repeats=5)
    assert importance["filter"].tolist() == [0, 1, 2, 3]
    assert list(importance.columns) == ["filter", "importance", "spread"] + [f"importance_fold{i}" for i in range(5)]
    assert importance["importance"].idxmax() == 0
    assert importance.loc[0, "importance"] - 2 * importance.loc[0, "spread"] > 0
    assert importance.loc[[2, 3], "importance"].abs().max() < importance.loc[0, "importance"] / 5
    # Every row is scored once, by a forest that did not see it.
    assert not np.isnan(oof).any() and ((oof >= 0.5) == table["label"]).mean() > 0.9


def test_importance_is_repeatable(monkeypatch):
    monkeypatch.setattr(gr, "GATE_TREES", 20)
    table, groups = _score_table()
    first = gr.gate_importance(table, groups, seed=7, n_repeats=3)
    second = gr.gate_importance(table, groups, seed=7, n_repeats=3)
    assert first[0].equals(second[0]) and np.array_equal(first[1], second[1])


def _twin_table(n=200):
    """filter_00 and filter_01 are the same useful signal (twins); 02 / 03 are noise; 04 is dead."""
    table, groups = _score_table(n)
    table["filter_01"] = table["filter_00"] * 2 + 0.01 * np.random.default_rng(1).normal(size=n)
    table["filter_04"] = 0.0
    return table, groups


def test_correlated_groups_join_the_twins_only():
    table, _ = _twin_table()
    assert gr.correlated_groups(table).tolist() == [0, 0, 1, 2, 3]


def test_twins_look_useless_alone_but_useful_together(monkeypatch):
    monkeypatch.setattr(gr, "GATE_TREES", 50)
    table, groups = _twin_table()
    single, _ = gr.gate_importance(table, groups, seed=42, n_repeats=5)
    grouped = gr.group_importance(table, groups, gr.correlated_groups(table), seed=42, n_repeats=5)
    assert grouped["filters"].tolist() == ["0 1", "2", "3", "4"] and grouped["size"].tolist() == [2, 1, 1, 1]
    twins = grouped.loc[0, "importance"]
    assert twins - 2 * grouped.loc[0, "spread"] > 0
    assert twins > 3 * single.loc[[0, 1], "importance"].max()
    assert grouped.loc[3, "importance"] == 0.0   # shuffling a dead filter changes nothing


def test_group_importance_is_repeatable(monkeypatch):
    monkeypatch.setattr(gr, "GATE_TREES", 20)
    table, groups = _twin_table()
    filter_groups = gr.correlated_groups(table)
    first = gr.group_importance(table, groups, filter_groups, seed=7, n_repeats=3)
    assert first.equals(gr.group_importance(table, groups, filter_groups, seed=7, n_repeats=3))


def _ranking_inputs():
    """Group 0 = filters 0, 1, 2 (most important); group 1 = 3, 4; group 2 = 5 (dead)."""
    single = pd.DataFrame({"filter": range(6), "importance": [0.01, 0.03, 0.02, 0.0, 0.05, 0.0],
                           "spread": [0.001] * 6})
    grouped = pd.DataFrame({"group": [0, 1, 2], "size": [3, 2, 1], "filters": ["0 1 2", "3 4", "5"],
                            "importance": [0.2, 0.1, 0.0], "spread": [0.01] * 3})
    return single, grouped, {5}


def test_rank_takes_one_filter_per_group_in_turns_and_dead_last():
    ranking = gr.rank_filters(*_ranking_inputs())
    # Round 1: group 0's best (1), group 1's best (4); round 2: 2, 3; round 3: 0; then the dead filter.
    assert ranking["filter"].tolist() == [1, 4, 2, 3, 0, 5]
    assert ranking["pick"].tolist() == [0, 0, 1, 1, 2, -1] and ranking["rank"].tolist() == list(range(6))
    assert ranking["group"].tolist() == [0, 1, 0, 1, 0, 2] and ranking["dead"].tolist() == [False] * 5 + [True]


def test_other_orders_are_permutations_of_every_filter():
    single, grouped, dead = _ranking_inputs()
    orders = gr.other_orders(single, grouped, dead, gr.rank_filters(single, grouped, dead))
    assert orders["single"] == [4, 1, 2, 0, 3, 5]
    assert orders["whole_groups"] == [1, 2, 0, 4, 3, 5]
    assert orders["reverse"] == [5, 0, 3, 2, 4, 1]
    assert all(sorted(o) == list(range(6)) for o in orders.values())


def test_removed_filters_send_exactly_zero_like_a_deleted_filter():
    cnn, word_to_idx = _cnn()
    cnn.eval()
    ids, _ = tc.tensors(_part(), word_to_idx, 8)
    keep = [0, 5, 9]
    with torch.no_grad():
        pooled = gr.kept_scores(cnn.encode(ids), keep)
        cnn.conv.weight[[i for i in range(64) if i not in keep]] = 0.0   # the removed filters, really zeroed
        cnn.conv.bias[[i for i in range(64) if i not in keep]] = 0.0
        assert torch.equal(pooled, cnn.encode(ids))


def test_scores_are_repeatable_and_batching_does_not_change_them(monkeypatch):
    cnn, word_to_idx = _cnn()
    part = _part(10)
    whole = gr.score_table(cnn, part, word_to_idx, seq_len=8)
    monkeypatch.setattr(gr, "BATCH_SIZE", 3)
    assert gr.score_table(cnn, part, word_to_idx, seq_len=8).equals(whole)
