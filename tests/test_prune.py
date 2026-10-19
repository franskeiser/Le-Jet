"""
Tests for the C-prune pruning steps (docs/c_prune_architecture.md Phase 5, steps 5-9).

Run with:  python -m pytest tests/test_prune.py
"""

from __future__ import annotations

import copy
import json

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


def test_keeping_all_64_filters_changes_nothing():
    full = _encoder()
    full.eval()
    # The whole chain: a shuffled ranking -> select_top_k(64) -> prune_encoder.
    ranking = np.random.default_rng(3).permutation(64)
    pruned = prune_encoder(full, select_top_k(ranking, 64))
    assert pruned.state_dict().keys() == full.state_dict().keys()
    assert all(torch.equal(full.state_dict()[k], v) for k, v in pruned.state_dict().items())
    ids = _ids()
    with torch.no_grad():
        assert torch.equal(pruned.encode(ids), full.encode(ids))
        assert torch.equal(pruned(ids), full(ids))   # bit for bit: same sizes, same weights, same order


def test_keeping_all_columns_leaves_a_fusion_dense_unchanged():
    dense = nn.Linear(64 + 40, 64)
    same = prune_input_columns(dense, list(range(64)), cnn_dim=64)
    assert torch.equal(same.weight, dense.weight) and torch.equal(same.bias, dense.bias)
    x = torch.randn(5, 104)
    with torch.no_grad():
        assert torch.equal(same(x), dense(x))


def _mask_setup(tmp_path, monkeypatch):
    """A 6-filter 'Phase A checkpoint' + sidecar and the ranking files, under a temporary root."""
    models, out = tmp_path / "models", tmp_path / "experiments"
    models.mkdir()
    out.mkdir()
    for name, value in (("ARTIFACTS_ROOT", tmp_path), ("PRUNE_KEEP", [4, 2]), ("RANKING_RULE", "group_turns")):
        monkeypatch.setattr(gr, name, value)
    monkeypatch.setattr(gr, "model_dir", lambda dataset: models)
    monkeypatch.setattr(gr, "out_dir", lambda dataset: out)
    (models / "phase_a_seed42_fold0.pt").write_bytes(b"weights")
    (models / "phase_a_seed42_fold0.json").write_text(json.dumps({"num_filters": 6}), encoding="utf-8")
    pd.DataFrame({"rank": range(6), "filter": [4, 1, 5, 0, 2, 3]}).to_csv(
        out / "filter_importance_seed42_fold0.csv", index=False)
    pd.DataFrame({"filter": range(6), "importance": [0.1, 0.5, 0.0, 0.9, 0.2, 0.3]}).to_csv(
        out / "importance_seed42_fold0.csv", index=False)
    scores = pd.DataFrame({"message_id": ["a", "b"], "label": [1, 0],
                           **{f"filter_{i:02d}": [1.0, 0.5] for i in range(6)}})
    scores["filter_03"] = 0.0   # dead
    scores.to_csv(out / "filter_scores_seed42_fold0.csv", index=False)
    return models, out


def test_gate_mask_sidecar_round_trip(tmp_path, monkeypatch):
    models, _ = _mask_setup(tmp_path, monkeypatch)
    sidecar = gr.write_gate_mask(42, 0, pilot=True)
    saved = json.loads((models / "gate_mask_seed42_fold0.json").read_text(encoding="utf-8"))
    assert saved == sidecar and saved["ranking_rule"] == "group_turns" and saved["ranking"] == [4, 1, 5, 0, 2, 3]
    assert saved["keep"]["4"] == {"keep_idx": [0, 1, 4, 5], "gate_mask": [1, 1, 0, 0, 1, 1]}
    assert set(saved["sha256"]) == {"phase_a_seed42_fold0.pt", "experiments/filter_importance_seed42_fold0.csv"}
    assert gr.load_gate_mask(42, 0, 4, pilot=True).tolist() == [0, 1, 4, 5]
    assert gr.load_gate_mask(42, 0, 2, pilot=True).tolist() == [1, 4]


def test_gate_mask_refuses_a_changed_checkpoint(tmp_path, monkeypatch):
    models, _ = _mask_setup(tmp_path, monkeypatch)
    gr.write_gate_mask(42, 0, pilot=True)
    (models / "phase_a_seed42_fold0.pt").write_bytes(b"retrained")   # a new CNN: its filter ids mean other things
    with pytest.raises(ValueError, match="changed since the gate mask was made"):
        gr.load_gate_mask(42, 0, 4, pilot=True)


def test_gate_mask_refuses_a_changed_ranking(tmp_path, monkeypatch):
    _, out = _mask_setup(tmp_path, monkeypatch)
    gr.write_gate_mask(42, 0, pilot=True)
    pd.DataFrame({"rank": range(6), "filter": range(6)}).to_csv(out / "filter_importance_seed42_fold0.csv", index=False)
    with pytest.raises(ValueError, match="changed since the gate mask was made"):
        gr.load_gate_mask(42, 0, 4, pilot=True)


def test_gate_mask_refuses_an_unknown_k_and_an_edited_mask(tmp_path, monkeypatch):
    models, _ = _mask_setup(tmp_path, monkeypatch)
    gr.write_gate_mask(42, 0, pilot=True)
    with pytest.raises(ValueError, match="no gate mask for k = 3"):
        gr.load_gate_mask(42, 0, 3, pilot=True)
    path = models / "gate_mask_seed42_fold0.json"
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["keep"]["2"]["gate_mask"] = [1, 0, 0, 0, 0, 1]
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(AssertionError, match="gate mask and keep_idx differ"):
        gr.load_gate_mask(42, 0, 2, pilot=True)


def test_gate_mask_with_the_single_rule(tmp_path, monkeypatch):
    _mask_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(gr, "RANKING_RULE", "single")
    sidecar = gr.write_gate_mask(42, 0, pilot=True)
    # Step 2 importance alone, highest first; the dead filter 3 last although its importance is the highest.
    assert sidecar["ranking_rule"] == "single" and sidecar["ranking"] == [1, 5, 4, 0, 2, 3]
    assert gr.load_gate_mask(42, 0, 2, pilot=True).tolist() == [1, 5]
    assert len(sidecar["sha256"]) == 3   # the checkpoint, importance and filter scores


def test_ranking_rule_must_be_known():
    with pytest.raises(ValueError, match="unknown ranking rule"):
        gr.ranking_inputs(42, 0, "whole_groups")


def _compare_detail(single_loss: float, seeds=(42,), folds=(0, 1)):
    rows = []
    for s in seeds:
        for f in folds:
            for k in (48, 16):
                rows += [{"seed": s, "fold": f, "k": k, "order": "group_turns", "val_log_loss_refit": 0.20},
                         {"seed": s, "fold": f, "k": k, "order": "single", "val_log_loss_refit": single_loss},
                         {"seed": s, "fold": f, "k": k, "order": "random", "val_log_loss_refit": 0.10}]
    return pd.DataFrame(rows)


def test_ranking_decision_follows_the_plan_rule():
    assert gr.ranking_decision(_compare_detail(0.19))["rule"] == "single"        # single lower: switch
    kept = gr.ranking_decision(_compare_detail(0.21))
    assert kept["rule"] == "group_turns" and kept["single_lower_in"] == "0/4"   # random's 0.10 plays no part
    assert gr.ranking_decision(_compare_detail(0.20))["rule"] == "group_turns"  # a tie keeps the rule


def test_lock_is_a_dry_run_on_the_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "out_dir", lambda dataset: tmp_path)
    _compare_detail(0.19).to_csv(tmp_path / "ranking_compare.csv", index=False)
    gr.run_lock(pilot=True, seeds=[42], n_folds=2)
    assert not (tmp_path / "ranking_rule_decision.json").exists() and gr.locked_rule(pilot=True) is None


def test_lock_happens_once_and_the_mask_step_obeys_it(tmp_path, monkeypatch):
    _mask_setup(tmp_path, monkeypatch)
    out = tmp_path / "experiments"
    _compare_detail(0.19).to_csv(out / "ranking_compare.csv", index=False)
    gr.run_lock(pilot=False, seeds=[42], n_folds=2)
    saved = json.loads((out / "ranking_rule_decision.json").read_text(encoding="utf-8"))
    assert saved["rule"] == "single" and saved["rule_before"] == "group_turns" and gr.locked_rule() == "single"
    with pytest.raises(SystemExit, match="already locked"):
        gr.run_lock(pilot=False, seeds=[42], n_folds=2)
    with pytest.raises(ValueError, match="locked to single"):              # RANKING_RULE is still group_turns
        gr.write_gate_mask(42, 0, pilot=False)
    monkeypatch.setattr(gr, "RANKING_RULE", "single")
    assert gr.write_gate_mask(42, 0, pilot=False)["ranking_rule"] == "single"


def test_lock_needs_every_seed_and_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "out_dir", lambda dataset: tmp_path)
    _compare_detail(0.19, folds=(0,)).to_csv(tmp_path / "ranking_compare.csv", index=False)
    with pytest.raises(AssertionError, match="did not cover every seed and fold"):
        gr.run_lock(pilot=False, seeds=[42], n_folds=2)
