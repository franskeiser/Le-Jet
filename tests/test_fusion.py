"""
Tests for the C-prune fused model (docs/c_prune_architecture.md Phase 6):
src/features/model/fusion_classifier.PrunedFusionClassifier and src/unified/fusion.py.

Run with:  python -m pytest tests/test_fusion.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.model.cnn_encoder import TextEncoder, prune_encoder
from src.features.model.embedding import build_torch_embedding_layer
from src.features.model.fusion_classifier import PrunedFusionClassifier
from src.unified import fusion as fu
from src.unified.link_lane import LANE

VOCAB = ["<PAD>", "<URL>", "verify", "now", "account", "locked", "your", "statement", "is", "ready"]
KEEP = [0, 5, 9, 33, 63]


def _model(keep=KEEP):
    word_to_idx = {w: i for i, w in enumerate(VOCAB)}
    matrix = np.random.default_rng(0).normal(size=(len(VOCAB), 8)).astype(np.float32)
    matrix[0] = 0.0
    embedding = build_torch_embedding_layer(matrix, word_to_idx)
    embedding.weight.requires_grad = False
    torch.manual_seed(0)
    full = TextEncoder(embedding, 64)
    return PrunedFusionClassifier(prune_encoder(full, keep), keep, url_dim=40), word_to_idx


def _trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_section_5_3_totals():
    # The table of section 5.3 (300-d fastText, kernel 3, 40 link numbers, Dense 64).
    totals = {k: sum(fu.expected_trainable(k, 300, 3).values()) for k in (64, 48, 32, 16)}
    assert totals == {64: 64_514, 48: 49_074, 32: 33_634, 16: 18_194}


def test_pruned_fusion_layers_and_trainable_parameters():
    model, _ = _model()
    assert model.encoder.temp_head is None and not model.encoder.embedding.weight.requires_grad
    assert model.dense1.in_features == 5 + 40 and model.dense1.out_features == 64 and model.dense2.out_features == 2
    assert _trainable(model) == sum(fu.expected_trainable(5, 8, 3).values())
    # keep_idx is saved with the weights, the temporary head is not.
    state = model.state_dict()
    assert state["keep_idx"].tolist() == KEEP and not any(k.startswith("encoder.temp_head") for k in state)


def test_pruned_fusion_forward_reads_the_kept_filters_then_the_link_numbers():
    model, _ = _model()
    model.eval()
    torch.manual_seed(1)
    ids, link = torch.randint(0, len(VOCAB), (6, 10)), torch.randn(6, 40)
    with torch.no_grad():
        fused = torch.cat([model.encoder.encode(ids), link], dim=1)
        expected = model.dense2(torch.relu(model.dense1(fused)))
        assert torch.equal(model(ids, link), expected)
        proba = model.predict_proba(ids, link)
    assert proba.shape == (6, 2) and torch.allclose(proba.sum(dim=1), torch.ones(6))


def test_weight_decay_reaches_dense_64_weights_only():
    model, _ = _model()
    groups = model.parameter_groups(0.1)
    assert groups[0]["params"] == [model.dense1.weight] and groups[0]["weight_decay"] == 0.1
    listed = [p for g in groups for p in g["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(listed) == len(trainable) and all(any(p is q for q in listed) for p in trainable)
    # One Adam step with a zero loss: only the decayed weights move (L2 pulls them toward 0).
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam(groups, lr=0.01)
    ids, link = torch.randint(0, len(VOCAB), (4, 10)), torch.randn(4, 40)
    (model(ids, link).sum() * 0.0).backward()
    optimizer.step()
    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert moved == {"dense1.weight"}
    assert model.dense1.weight.abs().sum() < before["dense1.weight"].abs().sum()


def test_keep_idx_must_match_the_encoder():
    model, _ = _model()
    with pytest.raises(ValueError, match="keep_idx has 4 filters, the encoder 5"):
        PrunedFusionClassifier(model.encoder, KEEP[:4], url_dim=40)


def _lane(ids):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"message_id": ids, **{c: rng.normal(size=len(ids)) for c in LANE}})


def test_fusion_inputs_line_up_each_message_with_its_own_link_numbers():
    _, word_to_idx = _model()
    part = pd.DataFrame({"message_id": ["m1", "m2", "m3"], "label": ["smishing", "legitimate", "smishing"],
                         "sms_text": ["verify now https://a.xyz", "your statement is ready https://b.com",
                                      "account locked https://c.xyz"]})
    lane = _lane(["m3", "m1", "m2"])                       # another order on purpose
    ids, link, labels = fu.fusion_inputs(part, lane, word_to_idx, seq_len=6)
    assert ids.shape == (3, 6) and link.shape == (3, 40) and labels.tolist() == [1, 0, 1]
    expected = lane.set_index("message_id").loc[["m1", "m2", "m3"], LANE].to_numpy(dtype=np.float32)
    assert np.array_equal(link.numpy(), expected)


@pytest.mark.parametrize("lane_ids", [["m1", "m2"], ["m1", "m2", "m3", "m4"], ["m1", "m2", "m9"]])
def test_fusion_inputs_refuse_missing_or_extra_link_rows(lane_ids):
    _, word_to_idx = _model()
    part = pd.DataFrame({"message_id": ["m1", "m2", "m3"], "label": ["smishing"] * 3, "sms_text": ["verify now"] * 3})
    with pytest.raises(ValueError, match="do not match one to one"):
        fu.fusion_inputs(part, _lane(lane_ids), word_to_idx, seq_len=4)
