"""Tests for the training, save, load and check steps of src/unified/film.py (plan docs/film_architecture_plan.md, steps 3.2 to 3.5).

They need the (git-ignored) synthetic pilot files and are skipped where those are missing. Everything is written to a temporary
folder, never to the real models/ or experiments/.

Run with:  python -m pytest tests/test_film_train.py
"""
import json

import pandas as pd
import pytest
import torch

from src.config import CROWD_PILOT_PATH
from src.unified import cnn_baseline as cb
from src.unified import film
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc


def pilot_available():
    dataset = tc.dataset_path(True)
    return ((tc.ROOT / CROWD_PILOT_PATH).exists() and (ll.out_dir(dataset) / "oof_seed42_fold0.csv").exists()
            and film.link_rf_path(dataset, 42, 0).exists() and (tc.embeddings_dir(dataset, 42, 0) / "vocab.json").exists())


pytestmark = pytest.mark.skipif(not pilot_available(), reason="the synthetic pilot files are not on this machine")
DATASET = tc.dataset_path(True)


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "model_dir", lambda dataset: tmp_path / "models" / dataset.stem)
    monkeypatch.setattr(fm, "out_dir", lambda dataset, step: tmp_path / "experiments" / dataset.stem / step)
    return tmp_path


@pytest.fixture
def baseline(outputs):
    cb.train_cell(42, 0, pilot=True, epochs=2)                      # the warm-start source, in the temporary folder
    return outputs


# ------------------------------------------------------------------ 3.2 warm start
def test_the_warm_start_copies_the_embedding_and_conv_and_nothing_else(baseline):
    model, vocab, seq_len = film.build_film(42, 0, pilot=True)
    untouched = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(("generator.", "dense.", "output."))}
    source = film.warm_start_from_baseline(model, DATASET, 42, 0)
    base_state = torch.load(fm.checkpoint_path(DATASET, "cnn", 42, 0))
    assert torch.equal(model.embedding.weight, base_state["embedding.weight"])
    assert torch.equal(model.conv.weight, base_state["conv.weight"]) and torch.equal(model.conv.bias, base_state["conv.bias"])
    for name, value in untouched.items():                               # generator at zero, dense and output as seeded
        assert torch.equal(model.state_dict()[name], value)
    assert (model.generator.weight == 0).all() and not torch.equal(model.dense.weight, base_state["dense.weight"])
    assert source["checkpoint"] == "cnn_seed42_fold0.pt" and len(source["sha256"]) == 64


def test_at_epoch_zero_e2_gives_the_baselines_text_numbers_exactly_and_without_the_warm_start_it_does_not(baseline):
    model, vocab, seq_len = film.build_film(42, 0, pilot=True)
    assert not film.start_matches_baseline(model, 42, 0, True, vocab, seq_len)             # random conv: not the baseline
    film.warm_start_from_baseline(model, DATASET, 42, 0)
    assert film.start_matches_baseline(model, 42, 0, True, vocab, seq_len)                  # exactly the baseline's 64 numbers


def test_the_warm_start_needs_the_baseline_and_refuses_a_changed_one(outputs):
    model, _, _ = film.build_film(42, 0, pilot=True)
    with pytest.raises(FileNotFoundError, match="train the baseline first"):
        film.warm_start_from_baseline(model, DATASET, 42, 0)
    cb.train_cell(42, 0, pilot=True, epochs=1)
    checkpoint = fm.checkpoint_path(DATASET, "cnn", 42, 0)
    good = checkpoint.read_bytes()
    checkpoint.write_bytes(good + b"x")
    with pytest.raises(ValueError, match="changed since"):
        film.warm_start_from_baseline(model, DATASET, 42, 0)


# ------------------------------------------------------------------ 3.3 and 3.4 training
def test_training_one_cell_saves_the_weights_a_sidecar_and_the_gamma_beta_statistics(baseline):
    row, log, filters = film.train_cell(42, 0, pilot=True, epochs=3)
    checkpoint = fm.checkpoint_path(DATASET, "film", 42, 0)
    assert checkpoint.exists() and checkpoint.parent == baseline / "models" / DATASET.stem
    sidecar = fm.read_sidecar(checkpoint)
    assert sidecar["kind"] == "film" and sidecar["trainable_without_embedding"] == 70658
    assert sidecar["options"] == {"gamma_mode": "bounded", "mask_padding": False, "average_after_relu": False, "use_beta": True,
                                  "freeze_embedding": False}
    assert set(sidecar["sha256"]) == {"vocab.json", "embedding_matrix.npy", checkpoint.name, "link_rf"}
    assert sidecar["warm_start"]["checkpoint"] == "cnn_seed42_fold0.pt"
    recipe = sidecar["recipe"]
    assert (recipe["lr"], recipe["patience"], recipe["label_smoothing"], recipe["p_page_hidden"], recipe["p_text_hidden"]) == (1e-3, 3, 0.1, 0.25, 0.10)
    assert recipe["dense_weight_decay"] == recipe["generator_weight_decay"] == 1e-4
    assert row["epochs_run"] <= 3 and len(log) == row["epochs_run"] and row["train_seconds"] > 0
    # step 3.4: the statistics per epoch and per filter
    assert {"gamma_dev_mean", "gamma_dev_max", "beta_abs_mean", "beta_abs_max", "filters_moved", "lift_abs_mean",
            "no_phrase_share"} <= set(log.columns)
    assert len(filters) == 64 and (filters["seed"] == 42).all() and {"gamma_mean", "beta_mean", "lift_mean"} <= set(filters.columns)
    assert row["beta_abs_mean"] == pytest.approx(log.iloc[row["best_epoch"]]["beta_abs_mean"])
    assert row["trainable_with_embedding"] == 70658 + row["vocab"] * 300


def test_the_training_is_reproducible(baseline):
    a, log_a, _ = film.train_cell(42, 0, pilot=True, epochs=3)
    b, log_b, _ = film.train_cell(42, 0, pilot=True, epochs=3)
    assert all(a[k] == b[k] for k in fm.VAL_METRICS) and (log_a["train_loss"] == log_b["train_loss"]).all()


def test_training_from_scratch_is_a_separate_run_without_the_warm_start(outputs):
    row, _, _ = film.train_cell(42, 0, pilot=True, warm_start=False, epochs=2)        # no baseline needed
    checkpoint = fm.checkpoint_path(DATASET, "film_scratch", 42, 0)
    assert checkpoint.exists() and fm.read_sidecar(checkpoint)["warm_start"] is None and row["kind"] == "film_scratch"
    assert not fm.checkpoint_path(DATASET, "film", 42, 0).exists()


def test_the_options_are_trained_and_recorded(baseline):
    film.train_cell(42, 0, pilot=True, epochs=2, gamma_mode="free", use_beta=False)
    sidecar = fm.read_sidecar(fm.checkpoint_path(DATASET, "film", 42, 0))
    assert sidecar["options"]["gamma_mode"] == "free" and sidecar["options"]["use_beta"] is False
    assert sidecar["trainable_without_embedding"] == 66306
    model, _, _, _ = film.load_film(42, 0, pilot=True)
    assert model.gamma_mode == "free" and model.use_beta is False


def test_run_train_writes_the_summary_log_and_filter_tables(baseline):
    summary = film.run_train(pilot=True, seeds=[42], folds=[0], epochs=2)
    out = baseline / "experiments" / DATASET.stem / "train"
    assert len(summary) == 1
    assert {"val_f1", "best_epoch", "train_seconds", "filters_moved"} <= set(pd.read_csv(out / "train_summary.csv").columns)
    assert len(pd.read_csv(out / "filter_stats.csv")) == 64 and "epoch" in pd.read_csv(out / "train_log.csv").columns


# ------------------------------------------------------------------ 3.5 load and check
def test_a_trained_model_reloads_exactly_from_its_sidecar(baseline):
    row, _, _ = film.train_cell(42, 0, pilot=True, epochs=3)
    model, vocab, seq_len, sidecar = film.load_film(42, 0, pilot=True)
    val = film.part_inputs(42, 0, "val", vocab, seq_len, pilot=True)
    again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
    assert all(again[k] == sidecar["val"][k] == row[k] for k in fm.VAL_METRICS)
    assert not model.training and model.embedding.weight.requires_grad


def test_a_changed_checkpoint_or_link_forest_is_refused(baseline, monkeypatch):
    film.train_cell(42, 0, pilot=True, epochs=1)
    checkpoint = fm.checkpoint_path(DATASET, "film", 42, 0)
    good = checkpoint.read_bytes()
    checkpoint.write_bytes(good + b"x")
    with pytest.raises(ValueError, match="changed since"):
        film.load_film(42, 0, pilot=True)
    checkpoint.write_bytes(good)
    film.load_film(42, 0, pilot=True)
    other = baseline / "other_forest.joblib"                                   # the Link RF is part of what the model is served with
    other.write_bytes(b"a different forest")
    monkeypatch.setattr(film, "link_rf_path", lambda dataset, seed, fold: other)
    with pytest.raises(ValueError, match="changed since"):
        film.load_film(42, 0, pilot=True)


def test_check_passes_and_catches_a_wrong_recorded_metric_and_a_broken_app_path(baseline, monkeypatch):
    film.train_cell(42, 0, pilot=True, epochs=2)
    ok = film.run_check(pilot=True, seeds=[42], folds=[0])
    assert bool(ok["ok"].iloc[0]) and ok["max_gap"].iloc[0] == 0.0 and bool(ok["app_path_ok"].iloc[0])
    assert (baseline / "experiments" / DATASET.stem / "check" / "check_summary.csv").exists()

    real = film.predict_messages
    def wrong(model, link_rf, word_to_idx, seq_len, rows, hide_page=False):
        out = real(model, link_rf, word_to_idx, seq_len, rows, hide_page)
        out["p_smishing"] = 1 - out["p_smishing"]
        return out
    monkeypatch.setattr(film, "predict_messages", wrong)
    broken_path = film.run_check(pilot=True, seeds=[42], folds=[0])
    assert not bool(broken_path["app_path_ok"].iloc[0]) and not bool(broken_path["ok"].iloc[0])
    monkeypatch.setattr(film, "predict_messages", real)

    checkpoint = fm.checkpoint_path(DATASET, "film", 42, 0)
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    sidecar["val"]["val_f1"] += 0.01
    checkpoint.with_suffix(".json").write_text(json.dumps(sidecar))
    bad = film.run_check(pilot=True, seeds=[42], folds=[0])
    assert not bool(bad["ok"].iloc[0]) and bad["max_gap"].iloc[0] > 0
