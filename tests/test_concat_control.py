"""Tests for the concatenation control: src/features/model/concat_cnn.py and src/unified/concat_control.py (plan D12).

The model tests run anywhere. The training tests use the (git-ignored) synthetic pilot, write to a temporary folder and are skipped
where the pilot files are missing.

Run with:  python -m pytest tests/test_concat_control.py
"""
import pytest
import torch
import torch.nn.functional as F

from src.config import CROWD_PILOT_PATH
from src.features.model.concat_cnn import ConcatTextCNN
from src.features.model.film_cnn import FiLMTextCNN
from src.unified import cnn_baseline as cb
from src.unified import concat_control as cc
from src.unified import film
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc

FRESH = 100


def embedding(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Embedding(FRESH, 300, padding_idx=0)


def batch(n=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, FRESH, (n, 50), generator=g), torch.rand(n, 3, generator=g)


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p is not model.embedding.weight)


# ------------------------------------------------------------------ the model
def test_the_control_has_62146_trainable_parameters_without_the_embedding():
    model = ConcatTextCNN(embedding())
    assert trainable(model) == 62146 == cc.expected_trainable(3)
    parts = {n: sum(p.numel() for p in layer.parameters()) for n, layer in
             (("conv", model.conv), ("dense", model.dense), ("output", model.output))}
    assert parts == {"conv": 57664, "dense": 4352, "output": 130}              # dense reads 64 text numbers + 3 link numbers
    assert trainable(FiLMTextCNN(embedding())) - trainable(model) == 70658 - 62146


def test_the_link_numbers_are_attached_before_dense_64_and_change_the_output():
    model = ConcatTextCNN(embedding()).eval()
    ids, link = batch()
    with torch.no_grad():
        logits = model(ids, link)
        assert tuple(logits.shape) == (8, 2)
        assert not torch.equal(logits, model(ids, 1 - link))                      # the link reaches the verdict
        text = model.encode(ids)
        manual = model.output(F.relu(model.dense(torch.cat([text, link], dim=1))))
        assert torch.equal(logits, manual)                                         # exactly: [64 text numbers, P, u, page_missing]


def test_hiding_the_text_zeroes_the_text_numbers_but_not_the_link():
    model = ConcatTextCNN(embedding()).eval()
    ids, link = batch()
    with torch.no_grad():
        assert (model.encode(ids, text_keep=torch.zeros(8)) == 0).all()
        # with the text hidden the verdict can still depend on the link, and not on the words
        a, b = model(ids, link, text_keep=torch.zeros(8)), model(batch(seed=2)[0], link, text_keep=torch.zeros(8))
        assert torch.equal(a, b) and not torch.equal(a, model(ids, 1 - link, text_keep=torch.zeros(8)))


def test_a_wrong_link_shape_is_refused_and_the_embedding_option_works():
    model = ConcatTextCNN(embedding())
    ids, link = batch()
    with pytest.raises(ValueError, match="link must be"):
        model(ids, link[:, :2])
    with pytest.raises(ValueError, match="link must be"):
        model(ids, link[:5])
    assert model.embedding.weight.requires_grad and not ConcatTextCNN(embedding(), freeze_embedding=True).embedding.weight.requires_grad


@pytest.mark.parametrize("freeze", [False, True])
def test_parameter_groups_cover_every_trainable_parameter_once(freeze):
    model = ConcatTextCNN(embedding(), freeze_embedding=freeze)
    groups = model.parameter_groups(1e-4)
    listed = [p for g in groups for p in g["params"]]
    assert len(listed) == len({id(p) for p in listed}) and {id(p) for p in listed} == {id(p) for p in model.parameters() if p.requires_grad}
    decay = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    assert decay[id(model.dense.weight)] == 1e-4 and decay[id(model.conv.weight)] == 0.0
    torch.optim.Adam(groups, lr=1e-3)


def test_the_baseline_embedding_and_conv_load_into_the_control_and_nothing_else():
    baseline = cb.PlainTextCNN(embedding())
    model = ConcatTextCNN(embedding(seed=5))
    result = model.load_state_dict({k: v for k, v in baseline.state_dict().items() if k.startswith(("embedding.", "conv."))}, strict=False)
    assert result.unexpected_keys == [] and {k.split(".")[0] for k in result.missing_keys} == {"dense", "output"}
    assert torch.equal(model.conv.weight, baseline.conv.weight)


# ------------------------------------------------------------------ training on the pilot
def pilot_available():
    dataset = tc.dataset_path(True)
    return ((tc.ROOT / CROWD_PILOT_PATH).exists() and (ll.out_dir(dataset) / "oof_seed42_fold0.csv").exists()
            and film.link_rf_path(dataset, 42, 0).exists() and (tc.embeddings_dir(dataset, 42, 0) / "vocab.json").exists())


needs_pilot = pytest.mark.skipif(not pilot_available(), reason="the synthetic pilot files are not on this machine")
DATASET = tc.dataset_path(True)


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "model_dir", lambda dataset: tmp_path / "models" / dataset.stem)
    monkeypatch.setattr(fm, "out_dir", lambda dataset, step: tmp_path / "experiments" / dataset.stem / step)
    return tmp_path


@needs_pilot
def test_the_warm_started_control_is_trained_saved_and_reloads_exactly(outputs):
    cb.train_cell(42, 0, pilot=True, epochs=1)                                      # the warm-start source, in the temporary folder
    row, _ = cc.train_cell(42, 0, pilot=True, epochs=2)
    checkpoint = fm.checkpoint_path(DATASET, "concat", 42, 0)
    sidecar = fm.read_sidecar(checkpoint)
    assert sidecar["kind"] == "concat" and sidecar["trainable_without_embedding"] == cc.EXPECTED_TRAINABLE
    assert sidecar["warm_start"]["checkpoint"] == "cnn_seed42_fold0.pt" and set(sidecar["sha256"]) == {"vocab.json", "embedding_matrix.npy", checkpoint.name, "link_rf"}
    assert sidecar["recipe"]["p_page_hidden"] == 0.25 and sidecar["recipe"]["p_text_hidden"] == 0.10 and sidecar["recipe"]["lr"] == 1e-3
    model, vocab, seq_len, _ = cc.load_control(42, 0, pilot=True)
    val = film.part_inputs(42, 0, "val", vocab, seq_len, pilot=True)
    again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
    assert all(again[k] == sidecar["val"][k] == row[k] for k in fm.VAL_METRICS)
    summary = cc.run_check(pilot=True, seeds=[42], folds=[0])
    assert bool(summary["ok"].iloc[0]) and summary["max_gap"].iloc[0] == 0.0


@needs_pilot
def test_the_from_scratch_control_is_a_separate_run_and_needs_no_baseline(outputs):
    row, _ = cc.train_cell(42, 0, pilot=True, warm_start=False, epochs=2)           # no baseline trained: not needed
    checkpoint = fm.checkpoint_path(DATASET, "concat_scratch", 42, 0)
    assert checkpoint.exists() and fm.read_sidecar(checkpoint)["warm_start"] is None
    assert not fm.checkpoint_path(DATASET, "concat", 42, 0).exists()
    model, _, _, sidecar = cc.load_control(42, 0, pilot=True, warm_start=False)
    assert sidecar["kind"] == "concat_scratch" and model.dense.in_features == 67
    assert (outputs / "experiments" / DATASET.stem / "concat_scratch" / "train_summary.csv").exists() is False   # only run_train writes it
    cc.run_train(pilot=True, seeds=[42], folds=[0], warm_start=False, epochs=1)
    assert (outputs / "experiments" / DATASET.stem / "concat_scratch" / "train_summary.csv").exists()


@needs_pilot
def test_a_missing_or_changed_baseline_stops_the_warm_started_control(outputs):
    with pytest.raises(FileNotFoundError, match="train the baseline first"):
        cc.train_cell(42, 0, pilot=True, epochs=1)
    cb.train_cell(42, 0, pilot=True, epochs=1)
    checkpoint = fm.checkpoint_path(DATASET, "cnn", 42, 0)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed since"):
        cc.train_cell(42, 0, pilot=True, epochs=1)
