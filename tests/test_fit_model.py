"""Tests for src/unified/fit_model.py, the shared training loop (plan docs/film_architecture_plan.md, steps 3.3, 3.4, 3.6).

The tests use tiny synthetic tasks, so they run in seconds and need no project data.

Run with:  python -m pytest tests/test_fit_model.py
"""
import json

import numpy as np
import pytest
import torch

from src.features.model.film_cnn import FiLMTextCNN
from src.features.model.plain_cnn import PlainTextCNN
from src.unified import fit_model as fm

VOCAB, LENGTH = 20, 12


def embedding(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Embedding(VOCAB, 300, padding_idx=0)


def text_task(n, seed):
    """The label is whether word 5 is in the message: the text carries it, the link says nothing."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, VOCAB, (n, LENGTH), generator=g)
    ids[ids == 5] = 6
    labels = (torch.arange(n) % 2).long()
    position = torch.randint(0, LENGTH, (n,), generator=g)
    ids[torch.arange(n)[labels == 1], position[labels == 1]] = 5
    link = torch.rand(n, 3, generator=g)
    return ids, link, torch.where(torch.ones(n, 3, dtype=torch.bool), link, link), labels


def link_task(n, seed):
    """Neutral wording: the text is random, the label is carried by the link alone (P high for smishing)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, VOCAB, (n, LENGTH), generator=g)
    labels = (torch.arange(n) % 2).long()
    p = (0.15 + 0.7 * labels.float() + 0.1 * torch.randn(n, generator=g)).clamp(0, 1)
    full = torch.stack([p, torch.full((n,), 0.1), torch.zeros(n)], dim=1)
    blank = full.clone()
    blank[:, 2] = 1.0                                                      # page hidden: the same P, page_missing = 1
    return ids, full, blank, labels


def split(task, n_train=240, n_val=80, seed=0):
    ids, full, blank, y = task(n_train + n_val, seed)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    tr, va = order[:n_train], order[n_train:]
    return (ids[tr], full[tr], blank[tr], y[tr]), (ids[va], full[va], y[va])


# ------------------------------------------------------------------ the loop
def test_a_text_model_learns_a_text_task_and_the_best_epoch_is_saved(tmp_path):
    train, val = split(text_task)
    model = PlainTextCNN(embedding())
    result = fm.fit(model, train, val, seed=1, checkpoint=tmp_path / "m.pt", lr=1e-2, p_page_hidden=0.0, p_text_hidden=0.0)
    assert result.metrics["val_balanced_accuracy"] > 0.9
    saved = torch.load(tmp_path / "m.pt")
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())          # the best epoch was loaded back
    assert fm.evaluate(model, *val)["val_balanced_accuracy"] == result.metrics["val_balanced_accuracy"]
    assert list(result.log["epoch"]) == list(range(result.epochs_run)) and result.seconds > 0
    assert result.filters is None                                                          # the baseline has no gamma and beta
    assert set(fm.VAL_METRICS) <= set(result.log.columns)


def test_the_same_seed_gives_the_same_training_and_another_seed_does_not(tmp_path):
    train, val = split(text_task)

    def run(seed):
        return fm.fit(PlainTextCNN(embedding()), train, val, seed=seed, checkpoint=tmp_path / f"s{seed}.pt", lr=1e-2,
                      p_page_hidden=0.0, p_text_hidden=0.0, epochs=4, patience=10)

    a, b, c = run(1), run(1), run(2)
    assert np.allclose(a.log["train_loss"], b.log["train_loss"]) and a.metrics == b.metrics
    assert not np.allclose(a.log["train_loss"], c.log["train_loss"])


def test_early_stopping_waits_for_patience_epochs_after_the_best(tmp_path):
    train, val = split(text_task)
    result = fm.fit(PlainTextCNN(embedding()), train, val, seed=1, checkpoint=tmp_path / "e.pt", lr=1e-2, patience=2,
                    p_page_hidden=0.0, p_text_hidden=0.0, epochs=30)
    assert result.epochs_run < 30 and result.epochs_run == result.best_epoch + 1 + 2


def test_it_stops_after_the_epoch_limit_when_it_keeps_improving(tmp_path):
    train, val = split(text_task)
    result = fm.fit(PlainTextCNN(embedding()), train, val, seed=1, checkpoint=tmp_path / "l.pt", lr=1e-5, patience=50,
                    p_page_hidden=0.0, p_text_hidden=0.0, epochs=3)
    assert result.epochs_run == 3 and len(result.log) == 3


class Spy(PlainTextCNN):
    """Records the link numbers and text_keep each training batch receives."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = []

    def forward(self, token_ids, link=None, text_keep=None):
        if self.training:
            self.seen.append((link.clone(), text_keep.clone()))
        return super().forward(token_ids, link, text_keep)


@pytest.mark.parametrize("p_page,p_text,expect_blank,expect_keep", [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0)])
def test_modality_dropout_swaps_the_link_and_hides_the_text(tmp_path, p_page, p_text, expect_blank, expect_keep):
    ids, full, blank, y = link_task(60, 0)
    full = torch.full_like(full, 0.1)
    blank = torch.full_like(blank, 0.9)                                      # distinguishable: 0.1 = full, 0.9 = page hidden
    train, val = (ids, full, blank, y), (ids[:20], full[:20], y[:20])
    spy = Spy(embedding())
    fm.fit(spy, train, val, seed=1, checkpoint=tmp_path / "d.pt", epochs=1, p_page_hidden=p_page, p_text_hidden=p_text)
    links = torch.cat([l for l, _ in spy.seen])
    keeps = torch.cat([k for _, k in spy.seen])
    assert (links == (0.9 if expect_blank else 0.1)).all() and (keeps == expect_keep).all()


def test_the_rates_are_followed_on_average_and_validation_has_no_dropout(tmp_path):
    ids, full, blank, y = link_task(400, 0)
    full, blank = torch.full_like(full, 0.1), torch.full_like(blank, 0.9)
    spy = Spy(embedding())
    fm.fit(spy, (ids, full, blank, y), (ids[:30], full[:30], y[:30]), seed=1, checkpoint=tmp_path / "r.pt", epochs=1, patience=5)
    links, keeps = torch.cat([l for l, _ in spy.seen]), torch.cat([k for _, k in spy.seen])
    assert abs(float((links[:, 0] == 0.9).float().mean()) - fm.P_PAGE_HIDDEN) < 0.07             # about 25% of pages hidden
    assert abs(float((keeps == 0).float().mean()) - fm.P_TEXT_HIDDEN) < 0.05                       # about 10% of text hidden
    n_train_calls = len(spy.seen)
    fm.evaluate(spy, ids[:30], full[:30], y[:30])
    assert len(spy.seen) == n_train_calls                                                           # evaluation is eval mode: nothing recorded


# ------------------------------------------------------------------ E2 against the baseline
def test_e2_can_use_the_link_when_the_words_say_nothing_and_the_baseline_cannot(tmp_path):
    """The central mechanism: the link reaches the verdict only through gamma and beta, and that is enough to learn from it.

    Uses the plan's own recipe (lr 1e-3, up to 30 epochs, patience 3) on 800 training messages. With only 240 the generator, which
    starts at zero, learns the link too slowly under that recipe (about 0.66 balanced accuracy in a probe): see the plan, part G.
    """
    train, val = split(link_task, n_train=800)
    kwargs = dict(seed=1, p_page_hidden=0.0, p_text_hidden=0.0)
    baseline = fm.fit(PlainTextCNN(embedding()), train, val, checkpoint=tmp_path / "b.pt", **kwargs)
    e2 = fm.fit(FiLMTextCNN(embedding()), train, val, checkpoint=tmp_path / "f.pt", **kwargs)
    assert baseline.metrics["val_balanced_accuracy"] < 0.7            # the text is noise, so the baseline stays near chance
    assert e2.metrics["val_balanced_accuracy"] > 0.9                   # E2 reads the link through the filters
    assert e2.filters is not None and len(e2.filters) == 64 and e2.log["filters_moved"].max() > 0


def test_film_stats_are_zero_at_the_start_and_move_after_training(tmp_path):
    train, val = split(link_task, n_train=800)
    model = FiLMTextCNN(embedding())
    summary, table = fm.film_stats(model, val[0], val[1])
    assert summary["gamma_dev_mean"] == 0 and summary["beta_abs_mean"] == 0 and summary["filters_moved"] == 0
    assert (table["lift_abs_mean"] == 0).all() and len(table) == 64
    fm.fit(model, train, val, seed=1, checkpoint=tmp_path / "s.pt", p_page_hidden=0.0, p_text_hidden=0.0)
    after, table = fm.film_stats(model, val[0], val[1])
    assert after["beta_abs_mean"] > 0 and after["filters_moved"] > 0 and after["lift_abs_mean"] > 0
    assert set(table.columns) >= {"filter", "gamma_mean", "gamma_std", "beta_mean", "beta_std", "lift_mean", "no_phrase_share"}
    assert 0 <= after["gamma_dev_mean"] <= 1.0 and after["gamma_dev_max"] <= 1.0                  # bounded gamma: |gamma - 1| <= 1


def test_a_class_missing_from_the_training_labels_is_refused(tmp_path):
    ids, full, blank, y = link_task(40, 0)
    with pytest.raises(AssertionError, match="class is missing"):
        fm.fit(PlainTextCNN(embedding()), (ids, full, blank, torch.zeros_like(y)), (ids, full, y), seed=1,
               checkpoint=tmp_path / "x.pt", epochs=1)


# ------------------------------------------------------------------ sidecars
def test_sidecar_round_trip_and_changed_files_are_caught(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"abc")
    digests = fm.hashes({"w": f})
    fm.check_hashes(digests, {"w": f})
    f.write_bytes(b"abd")
    with pytest.raises(ValueError, match="changed since"):
        fm.check_hashes(digests, {"w": f})
    ckpt = tmp_path / "m.pt"
    fm.write_sidecar(ckpt, {"seed": 42, "val": {"f1": np.float32(0.5)}})
    assert fm.read_sidecar(ckpt) == {"seed": 42, "val": {"f1": 0.5}} and json.loads(ckpt.with_suffix(".json").read_text())["seed"] == 42


def test_paths_follow_the_dataset_name(tmp_path):
    from pathlib import Path

    d = Path("data/processed/crowd_pilot_synthetic.csv")
    assert fm.checkpoint_path(d, "cnn", 42, 3).name == "cnn_seed42_fold3.pt"
    assert fm.checkpoint_path(d, "film", 123, 0).parent.name == "crowd_pilot_synthetic"
    assert fm.out_dir(d, "train").parts[-3:] == ("film", "crowd_pilot_synthetic", "train")
