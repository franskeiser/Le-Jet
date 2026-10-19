"""
Tests for the C-prune pruning steps (docs/c_prune_architecture.md Phase 5, steps 5-9).

Run with:  python -m pytest tests/test_prune.py
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.features.model.cnn_encoder import TextEncoder, prune_encoder, prune_input_columns
from src.features.model.embedding import build_torch_embedding_layer
from src.features.model.rf_gating_module import select_top_k
from src.unified import gate_rf as gr

KEEP = [0, 5, 9, 33, 63]


def _encoder(num_filters=64):
    vocab = ["<PAD>", "<URL>", "verify", "now", "account", "locked"]
    matrix = np.random.default_rng(0).normal(size=(len(vocab), 8)).astype(np.float32)
    matrix[0] = 0.0
    embedding = build_torch_embedding_layer(matrix, {w: i for i, w in enumerate(vocab)})
    embedding.weight.requires_grad = False
    torch.manual_seed(0)
    return TextEncoder(embedding, num_filters)


def test_select_top_k_takes_the_first_k_in_ascending_id_order():
    ranking = [7, 2, 9, 0, 5, 1, 3, 4, 6, 8]
    assert select_top_k(ranking, 3).tolist() == [2, 7, 9]
    assert select_top_k(ranking, 10).tolist() == list(range(10))   # k = all: every filter, unchanged order
    assert select_top_k(np.array(ranking), 1).tolist() == [7]


@pytest.mark.parametrize("ranking", [[0, 1, 1, 3], [0, 1, 2, 5], [1, 2, 3, 4]])
def test_select_top_k_refuses_a_ranking_that_is_not_every_filter_once(ranking):
    with pytest.raises(ValueError, match="exactly once"):
        select_top_k(ranking, 2)


@pytest.mark.parametrize("k", [0, -1, 5])
def test_select_top_k_refuses_an_impossible_k(k):
    with pytest.raises(ValueError, match="k must be between 1 and 4"):
        select_top_k([3, 1, 0, 2], k)


def test_load_keep_idx_reads_the_saved_ranking(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "out_dir", lambda dataset: tmp_path)
    # Saved shuffled on purpose: the order comes from the rank column, not the row order.
    pd.DataFrame({"rank": [2, 0, 3, 1], "filter": [0, 3, 1, 2]}).to_csv(
        tmp_path / "filter_importance_seed42_fold0.csv", index=False)
    assert gr.load_keep_idx(42, 0, 2, pilot=True).tolist() == [2, 3]


def test_prune_encoder_copies_only_the_kept_filters():
    full = _encoder()
    pruned = prune_encoder(full, KEEP)
    assert pruned.conv.weight.shape == (5, 8, 3) and pruned.conv.bias.shape == (5,)
    assert torch.equal(pruned.conv.weight, full.conv.weight[KEEP])
    assert torch.equal(pruned.conv.bias, full.conv.bias[KEEP])
    # Kept position i is original filter KEEP[i], in the temporary head too.
    assert pruned.temp_head.weight.shape == (2, 5)
    assert torch.equal(pruned.temp_head.weight, full.temp_head.weight[:, KEEP])
    assert torch.equal(pruned.temp_head.bias, full.temp_head.bias)
    assert torch.equal(pruned.embedding.weight, full.embedding.weight)


def test_prune_encoder_keeps_the_embedding_frozen_and_the_mode():
    full = _encoder()
    full.eval()
    pruned = prune_encoder(full, KEEP)
    assert not pruned.embedding.weight.requires_grad and not pruned.training
    # Trainable: Conv 8 x 3 x k + k and the head 2k + 2 (section 5.3 with an 8-d toy embedding).
    assert sum(p.numel() for p in pruned.parameters() if p.requires_grad) == 8 * 3 * 5 + 5 + 2 * 5 + 2


def test_prune_encoder_leaves_the_full_model_untouched():
    full = _encoder()
    before = {k: v.clone() for k, v in full.state_dict().items()}
    pruned = prune_encoder(full, KEEP)
    with torch.no_grad():
        for p in pruned.parameters():
            p.add_(1.0)   # changing the pruned model must not reach the full one: nothing is shared
    assert all(torch.equal(before[k], v) for k, v in full.state_dict().items())


def test_prune_input_columns_drops_filter_columns_and_keeps_the_rest():
    dense = nn.Linear(64 + 40, 64)   # Dense 64 reading [64 filters | 40 link numbers]
    pruned = prune_input_columns(dense, KEEP, cnn_dim=64)
    assert pruned.in_features == 5 + 40 and pruned.out_features == 64
    assert torch.equal(pruned.weight[:, :5], dense.weight[:, KEEP])
    assert torch.equal(pruned.weight[:, 5:], dense.weight[:, 64:])
    assert torch.equal(pruned.bias, dense.bias)


@pytest.mark.parametrize("keep", [[], [5, 0], [0, 0, 1], [-1, 3], [3, 64]])
def test_prune_refuses_a_bad_keep_idx(keep):
    with pytest.raises(ValueError, match="ascending, unique ids in 0..63"):
        prune_encoder(_encoder(), keep)


def _gated(full: TextEncoder, keep) -> TextEncoder:
    """The full model with its gate mask applied: W <- g * W, b <- g * b, g = 1 kept, 0 removed (section 2.4)."""
    gated = copy.deepcopy(full)
    g = torch.zeros(full.conv.out_channels)
    g[keep] = 1.0
    with torch.no_grad():
        gated.conv.weight.mul_(g[:, None, None])
        gated.conv.bias.mul_(g)
    return gated


def _ids(n=12, seq_len=10):
    torch.manual_seed(1)
    return torch.randint(0, 6, (n, seq_len))


@pytest.mark.parametrize("keep", [KEEP, list(range(0, 64, 2)), [7]])
def test_pruned_output_equals_full_output_with_removed_filters_gated_to_zero(keep):
    full = _encoder()
    full.eval()
    pruned, gated, ids = prune_encoder(full, keep), _gated(full, keep), _ids()
    with torch.no_grad():
        scores = gated.encode(ids)
        removed = [i for i in range(64) if i not in keep]
        assert torch.equal(scores[:, removed], torch.zeros(len(ids), len(removed)))   # a gated filter sends 0
        assert torch.equal(pruned.encode(ids), scores[:, keep])                       # the kept scores: bit for bit
        # The output: the same up to float rounding (the head sums k terms instead of 64 with zeros).
        assert torch.allclose(pruned(ids), gated(ids), atol=1e-6, rtol=0)


def test_pruned_fusion_output_equals_full_with_removed_filters_gated_to_zero():
    # Phase 6's head: Dense 64 reads [filter scores | 40 link numbers]; the pruned one drops the removed columns.
    full = _encoder()
    full.eval()
    gated, pruned = _gated(full, KEEP), prune_encoder(full, KEEP)
    torch.manual_seed(2)
    dense, link = nn.Linear(64 + 40, 64), torch.randn(12, 40)
    pruned_dense = prune_input_columns(dense, KEEP, cnn_dim=64)
    ids = _ids()
    with torch.no_grad():
        expected = dense(torch.cat([gated.encode(ids), link], dim=1))
        assert torch.allclose(pruned_dense(torch.cat([pruned.encode(ids), link], dim=1)), expected, atol=1e-6, rtol=0)
