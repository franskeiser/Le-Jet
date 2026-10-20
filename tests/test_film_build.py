"""Tests for src/unified/film.py (plan docs/film_architecture_plan.md, step 3.1).

Most tests use a tiny random forest and vocabulary and run anywhere. Those that need the (git-ignored) synthetic pilot files
are skipped where they are missing.

Run with:  python -m pytest tests/test_film_build.py
"""
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from src.config import CROWD_PILOT_PATH
from src.features.model.film_cnn import FiLMTextCNN
from src.features.model.rf_extractor import p_and_u
from src.unified import film
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc
from src.unified.prepare_web_37 import FEATURES_37

VOCAB = {"<PAD>": 0, "verify": 1, "account": 2, "locked": 3, "<URL>": 4, "your": 5, "parcel": 6}


# ------------------------------------------------------------------ the size formula
def test_expected_trainable_matches_the_plan():
    assert film.expected_trainable() == 70658 == film.EXPECTED_TRAINABLE
    assert film.expected_trainable(use_beta=False) == 66306
    assert film.expected_trainable(num_filters=32) < 70658


def test_trainable_counts_separate_the_embedding():
    emb = torch.nn.Embedding(len(VOCAB), 300, padding_idx=0)
    trainable = film.trainable_counts(FiLMTextCNN(emb))
    assert trainable == (70658, 70658 + len(VOCAB) * 300)
    assert film.trainable_counts(FiLMTextCNN(torch.nn.Embedding(len(VOCAB), 300, padding_idx=0), freeze_embedding=True)) == (70658, 70658)


# ------------------------------------------------------------------ predict_messages on a small forest
@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.random((60, len(FEATURES_37))), columns=FEATURES_37)
    features[ll.PAGE] = features[ll.PAGE] * 10                                     # page columns carry information
    labels = (features.iloc[:, 0] + features[ll.PAGE[0]] / 10 > 1).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(features, labels)
    rows = features.copy()
    rows.insert(0, "sms_text", ["verify your account locked <URL>", "your parcel <URL>"] * 30)
    rows["page_fetched"] = [1, 0, 1, 1, 0, 1] * 10
    rows.index = [f"r{i}" for i in range(len(rows))]
    torch.manual_seed(0)
    model = FiLMTextCNN(torch.nn.Embedding(len(VOCAB), 300, padding_idx=0))
    with torch.no_grad():                                                           # a non-zero generator, so the link matters
        model.generator.weight.copy_(torch.randn_like(model.generator.weight) * 0.5)
    return model, rf, rows


def test_predict_messages_returns_what_the_generator_read_and_the_verdict(setup):
    model, rf, rows = setup
    out = film.predict_messages(model, rf, VOCAB, 12, rows.iloc[:20])
    assert list(out.columns) == ["P", "u", "page_missing", "p_smishing", "pred"] and list(out.index) == list(rows.index[:20])
    p, u = p_and_u(rf, rows.iloc[:20][FEATURES_37])
    assert np.allclose(out["P"], p) and np.allclose(out["u"], u)
    assert list(out["page_missing"]) == [0, 1, 0, 0, 1, 0] * 3 + [0, 1]
    assert ((out["p_smishing"] >= 0) & (out["p_smishing"] <= 1)).all()
    assert list(out["pred"]) == list((out["p_smishing"] >= 0.5).astype(int))


def test_predict_messages_gives_the_same_logits_as_the_model_fed_those_numbers(setup):
    from src.features.model.fusion_data import encode_batch_tolerant

    model, rf, rows = setup
    part = rows.iloc[:15]
    out = film.predict_messages(model, rf, VOCAB, 12, part)
    ids = encode_batch_tolerant(part["sms_text"].tolist(), VOCAB, 12)
    link = torch.tensor(out[["P", "u", "page_missing"]].to_numpy(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        expected = torch.softmax(model(ids, link), dim=1)[:, 1].numpy()
    assert np.allclose(out["p_smishing"], expected, atol=1e-6)


def test_hiding_the_page_blanks_the_page_columns_and_sets_page_missing(setup):
    model, rf, rows = setup
    part = rows.iloc[:12]
    hidden = film.predict_messages(model, rf, VOCAB, 12, part, hide_page=True)
    assert (hidden["page_missing"] == 1).all()
    p, u = p_and_u(rf, ll.blank_page(part)[FEATURES_37])
    assert np.allclose(hidden["P"], p) and np.allclose(hidden["u"], u)
    assert not np.allclose(hidden["P"], film.predict_messages(model, rf, VOCAB, 12, part)["P"])   # the page mattered to the forest
    assert part["page_fetched"].tolist() == rows.iloc[:12]["page_fetched"].tolist()               # the caller's rows are not changed


def test_predict_messages_leaves_the_model_in_the_mode_it_was_in(setup):
    model, rf, rows = setup
    model.train()
    film.predict_messages(model, rf, VOCAB, 12, rows.iloc[:5])
    assert model.training
    model.eval()
    film.predict_messages(model, rf, VOCAB, 12, rows.iloc[:5])
    assert not model.training


def test_the_text_changes_the_verdict_for_the_same_link_numbers(setup):
    model, rf, rows = setup
    one = rows.iloc[[0, 0]].copy()
    one["sms_text"] = ["verify your account locked <URL>", "your parcel <URL>"]
    out = film.predict_messages(model, rf, VOCAB, 12, one)
    assert out["P"].iloc[0] == out["P"].iloc[1] and out["p_smishing"].iloc[0] != out["p_smishing"].iloc[1]


# ------------------------------------------------------------------ on the pilot files
def pilot_available():
    root = tc.ROOT
    dataset = tc.dataset_path(True)
    return ((root / CROWD_PILOT_PATH).exists() and (ll.out_dir(dataset) / "oof_seed42_fold0.csv").exists()
            and film.link_rf_path(dataset, 42, 0).exists() and (tc.embeddings_dir(dataset, 42, 0) / "vocab.json").exists())


needs_pilot = pytest.mark.skipif(not pilot_available(), reason="the synthetic pilot files are not on this machine")


@needs_pilot
def test_build_film_is_deterministic_per_seed_and_differs_between_seeds():
    a, _, _ = film.build_film(42, 0, pilot=True)
    b, _, _ = film.build_film(42, 0, pilot=True)
    c, _, _ = film.build_film(123, 0, pilot=True)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not torch.equal(a.conv.weight, c.conv.weight)
    assert film.trainable_counts(a)[0] == 70658 and a.embedding.weight.requires_grad
    assert a.embedding is not b.embedding                                  # every model gets its own embedding layer


@needs_pilot
def test_part_inputs_line_the_text_up_with_its_link_numbers():
    model, vocab, seq_len = film.build_film(42, 0, pilot=True)
    for part in ("train", "val", "test"):
        data = film.part_inputs(42, 0, part, vocab, seq_len, pilot=True)
        n = len(data.message_id)
        assert tuple(data.token_ids.shape) == (n, seq_len) and tuple(data.labels.shape) == (n,)
        assert tuple(data.link_full.shape) == tuple(data.link_blank.shape) == (n, 3)
        assert (data.link_blank[:, 2] == 1).all()
        assert set(data.labels.tolist()) <= {0, 1}
    with torch.no_grad():
        assert tuple(model(data.token_ids, data.link_full).shape) == (n, 2)


@needs_pilot
def test_one_cell_passes_every_check_and_a_wrong_expectation_is_reported(monkeypatch):
    row = film.check_cell(42, 0, pilot=True)
    assert row["ok"] and row["trainable_without_embedding"] == 70658 and row["n_val"] > 0
    checks = ["params", "params_with_embedding", "same_seed_same_weights", "groups", "generator_at_zero", "forward",
              "app_path_link_numbers", "app_path_logits"]
    assert all(row[c] for c in checks)
    monkeypatch.setattr(film, "EXPECTED_TRAINABLE", 1)
    broken = film.check_cell(42, 0, pilot=True)
    assert not broken["ok"] and not broken["params"] and broken["forward"]


@needs_pilot
def test_run_build_writes_its_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "out_dir", lambda dataset, step: tmp_path / step)
    summary = film.run_build(pilot=True, seeds=[42], folds=[0])
    written = pd.read_csv(tmp_path / "build" / "build_summary.csv")
    assert len(summary) == len(written) == 1 and bool(written["ok"].iloc[0])
