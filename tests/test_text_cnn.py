"""
Tests for src/unified/text_cnn.py (docs/c_prune_architecture.md Phase 4).

Run with:  python -m pytest tests/test_text_cnn.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.model.cnn_encoder import TextEncoder
from src.features.model.embedding import build_torch_embedding_layer
from src.unified import text_cnn as tc

VOCAB = ["<PAD>", "<URL>", "verify", "now", "account", "locked", "your", "statement", "is", "ready"]


def _setup():
    word_to_idx = {w: i for i, w in enumerate(VOCAB)}
    matrix = np.random.default_rng(0).normal(size=(len(VOCAB), 8)).astype(np.float32)
    matrix[0] = 0.0
    embedding = build_torch_embedding_layer(matrix, word_to_idx)
    embedding.weight.requires_grad = False
    return embedding, word_to_idx


def _parts():
    smish = "verify now your account is locked https://a.xyz"
    legit = "your statement is ready https://b.com"
    part = lambda n: pd.DataFrame({"sms_text": [smish, legit] * n, "label": ["smishing", "legitimate"] * n})
    return {"train": part(20), "val": part(5)}


def test_train_fold_keeps_embedding_frozen_and_saves_best_epoch(tmp_path):
    embedding, word_to_idx = _setup()
    before = embedding.weight.detach().clone()
    checkpoint = tmp_path / "phase_a.pt"
    metrics = tc.train_fold(_parts(), embedding, word_to_idx, seq_len=8, seed=42, checkpoint=checkpoint, epochs=3)
    assert torch.equal(embedding.weight, before)
    assert checkpoint.exists() and metrics["val_f1"] == 1.0
    # Only Conv (8*64*3 + 64) and the temporary head (64*2 + 2) train.
    assert metrics["trainable_parameters"] == 8 * 64 * 3 + 64 + 64 * 2 + 2
    state = torch.load(checkpoint)
    assert torch.equal(state["embedding.weight"], before)


def test_train_fold_is_repeatable(tmp_path):
    states = []
    for name in ("a.pt", "b.pt"):
        embedding, word_to_idx = _setup()
        tc.train_fold(_parts(), embedding, word_to_idx, seq_len=8, seed=7, checkpoint=tmp_path / name, epochs=2)
        states.append(torch.load(tmp_path / name))
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])


def _saved_fold(tmp_path, monkeypatch):
    """A trained CNN, its embedding folder and sidecar, all under a temporary project root."""
    monkeypatch.setattr(tc, "ROOT", tmp_path)
    emb_dir = tmp_path / "models" / "embeddings_crowd" / "seed42_fold0"
    emb_dir.mkdir(parents=True)
    embedding, word_to_idx = _setup()
    (emb_dir / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    np.save(emb_dir / "embedding_matrix.npy", embedding.weight.detach().numpy())
    checkpoint = tmp_path / "phase_a_seed42_fold0.pt"
    parts = _parts()
    metrics = tc.train_fold(parts, embedding, word_to_idx, seq_len=8, seed=42, checkpoint=checkpoint, epochs=2)
    metrics.update(seq_len_l=8, vocab_size=len(VOCAB))
    tc.write_sidecar(checkpoint, emb_dir, 42, 0, metrics)
    return checkpoint, parts, metrics


def test_sidecar_round_trip_repeats_val_metrics(tmp_path, monkeypatch):
    checkpoint, parts, metrics = _saved_fold(tmp_path, monkeypatch)
    cnn, word_to_idx, sidecar = tc.load_phase_a(checkpoint)
    assert not cnn.training and not cnn.embedding.weight.requires_grad
    assert sidecar["seq_len_l"] == 8 and sidecar["num_filters"] == 64
    again = tc.evaluate(cnn, *tc.tensors(parts["val"], word_to_idx, sidecar["seq_len_l"]))
    assert again == pytest.approx({k: metrics[k] for k in tc.VAL_METRICS})


def test_load_refuses_a_changed_vocabulary(tmp_path, monkeypatch):
    checkpoint, _, _ = _saved_fold(tmp_path, monkeypatch)
    vocab = tmp_path / "models" / "embeddings_crowd" / "seed42_fold0" / "vocab.json"
    vocab.write_text(json.dumps(VOCAB[::-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="changed since the CNN was trained"):
        tc.load_phase_a(checkpoint)


def test_unknown_words_are_dropped_not_an_error():
    _, word_to_idx = _setup()
    ids, labels = tc.tensors(pd.DataFrame({"sms_text": ["verify bagong salita now"], "label": ["smishing"]}),
                             word_to_idx, seq_len=4)
    assert ids.tolist() == [[word_to_idx["verify"], word_to_idx["now"], 0, 0]] and labels.tolist() == [1]
