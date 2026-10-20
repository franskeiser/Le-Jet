"""Tests for src/unified/cnn_baseline.py (plan docs/film_architecture_plan.md, step 3.6).

They need the (git-ignored) synthetic pilot files and are skipped where those are missing. Training and sidecars are written to a
temporary folder, never to the real models/ or experiments/.

Run with:  python -m pytest tests/test_cnn_baseline.py
"""
import json
import shutil

import pandas as pd
import pytest
import torch

from src.config import CROWD_PILOT_PATH
from src.unified import cnn_baseline as cb
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc


def pilot_available():
    dataset = tc.dataset_path(True)
    return ((tc.ROOT / CROWD_PILOT_PATH).exists() and (ll.out_dir(dataset) / "oof_seed42_fold0.csv").exists()
            and (tc.embeddings_dir(dataset, 42, 0) / "vocab.json").exists())


pytestmark = pytest.mark.skipif(not pilot_available(), reason="the synthetic pilot files are not on this machine")


@pytest.fixture
def tmp_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "model_dir", lambda dataset: tmp_path / "models" / dataset.stem)
    monkeypatch.setattr(fm, "out_dir", lambda dataset, step: tmp_path / "experiments" / dataset.stem / step)
    return tmp_path


def test_build_baseline_has_the_planned_size_and_its_own_trainable_embedding():
    a, _, _ = cb.build_baseline(42, 0, pilot=True)
    b, _, _ = cb.build_baseline(42, 0, pilot=True)
    without, with_embedding = cb.trainable_counts(a)
    assert without == cb.EXPECTED_TRAINABLE == 61954 and with_embedding == 61954 + a.embedding.weight.numel()
    assert a.embedding.weight.requires_grad and a.embedding is not b.embedding
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))     # same seed, same weights


def test_training_one_cell_writes_a_checkpoint_and_a_sidecar_that_reloads_exactly(tmp_outputs):
    row, log = cb.train_cell(42, 0, pilot=True, epochs=3)
    dataset = tc.dataset_path(True)
    checkpoint = fm.checkpoint_path(dataset, "cnn", 42, 0)
    assert checkpoint.exists() and checkpoint.with_suffix(".json").exists() and checkpoint.parent == tmp_outputs / "models" / dataset.stem
    sidecar = fm.read_sidecar(checkpoint)
    assert sidecar["kind"] == "cnn" and sidecar["trainable_without_embedding"] == 61954 and sidecar["freeze_embedding"] is False
    assert set(sidecar["sha256"]) == {"vocab.json", "embedding_matrix.npy", checkpoint.name}
    assert sidecar["recipe"]["lr"] == 1e-3 and sidecar["recipe"]["patience"] == 3 and sidecar["recipe"]["label_smoothing"] == 0.1
    assert row["epochs_run"] <= 3 and len(log) == row["epochs_run"] and (log["seed"] == 42).all()

    model, word_to_idx, seq_len, _ = cb.load_baseline(42, 0, pilot=True)
    val = cb.part_inputs(42, 0, "val", word_to_idx, seq_len, pilot=True)
    again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
    assert all(again[k] == sidecar["val"][k] == row[k] for k in fm.VAL_METRICS)                          # a gap of exactly 0
    assert model.embedding.weight.requires_grad and not model.training


def test_the_baseline_uses_the_same_rows_as_e2(tmp_outputs):
    from src.unified.film import build_film, part_inputs

    row, _ = cb.train_cell(42, 0, pilot=True, epochs=1)
    film_model, vocab, seq_len = build_film(42, 0, pilot=True)
    train = part_inputs(42, 0, "train", vocab, seq_len, pilot=True)
    val = part_inputs(42, 0, "val", vocab, seq_len, pilot=True)
    assert row["n_train"] == len(train.labels) and row["n_val"] == len(val.labels) and row["vocab"] == len(vocab)


def test_a_changed_checkpoint_or_vocabulary_file_is_refused(tmp_outputs):
    cb.train_cell(42, 0, pilot=True, epochs=1)
    dataset = tc.dataset_path(True)
    checkpoint = fm.checkpoint_path(dataset, "cnn", 42, 0)
    good = checkpoint.read_bytes()
    checkpoint.write_bytes(good + b"x")
    with pytest.raises(ValueError, match="changed since"):
        cb.load_baseline(42, 0, pilot=True)
    checkpoint.write_bytes(good)
    cb.load_baseline(42, 0, pilot=True)                                       # restored: loads again


def test_a_wrong_recorded_metric_is_caught_by_check(tmp_outputs):
    cb.train_cell(42, 0, pilot=True, epochs=2)
    summary = cb.run_check(pilot=True, seeds=[42], folds=[0])
    assert bool(summary["ok"].iloc[0]) and summary["max_gap"].iloc[0] == 0.0
    checkpoint = fm.checkpoint_path(tc.dataset_path(True), "cnn", 42, 0)
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    sidecar["val"]["val_f1"] += 0.01
    checkpoint.with_suffix(".json").write_text(json.dumps(sidecar))
    broken = cb.run_check(pilot=True, seeds=[42], folds=[0])
    assert not bool(broken["ok"].iloc[0]) and broken["max_gap"].iloc[0] > 0
    assert (tmp_outputs / "experiments" / "crowd_pilot_synthetic" / "cnn_baseline" / "check_summary.csv").exists()


def test_run_train_writes_the_summary_and_the_log(tmp_outputs):
    summary = cb.run_train(pilot=True, seeds=[42], folds=[1], epochs=2)
    out = tmp_outputs / "experiments" / "crowd_pilot_synthetic" / "cnn_baseline"
    assert len(summary) == 1 and (pd.read_csv(out / "train_summary.csv")["fold"] == 1).all()
    assert {"epoch", "train_loss", "val_f1", "seed", "fold"} <= set(pd.read_csv(out / "train_log.csv").columns)
