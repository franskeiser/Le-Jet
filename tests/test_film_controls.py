"""Tests for src/unified/film_controls.py (plan docs/film_architecture_plan.md, steps 5.9 and 5.10).

The logic (the shuffle, the shared-wording subset, the tables) is tested on small made-up data, so it runs anywhere. One test at the end
runs the whole thing on the synthetic pilot and is skipped where its files are missing.

Run with:  python -m pytest tests/test_film_controls.py
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.unified import film_compare as fc
from src.unified import film_controls as ctl
from src.unified import text_cnn as tc


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------------------------ the shuffle
def link_matrix():
    # [P, u, page_missing, has_url]: rows 0-5 have a link, rows 6-7 do not (their numbers are zeros, as the lane writes them)
    link = torch.tensor([[0.1, 0.01, 0, 1], [0.2, 0.02, 0, 1], [0.3, 0.03, 1, 1], [0.4, 0.04, 0, 1], [0.5, 0.05, 0, 1],
                         [0.6, 0.06, 1, 1], [0.0, 0.0, 1, 0], [0.0, 0.0, 1, 0]])
    return link


def test_the_shuffle_keeps_has_url_with_its_message_and_leaves_no_link_rows_alone():
    link = link_matrix()
    out = ctl.shuffle_link(link, gen())
    assert torch.equal(out[:, -1], link[:, -1])                                 # has_url did not move
    assert torch.equal(out[6:], link[6:])                                       # the two no-link messages are untouched


def test_the_shuffle_hands_the_link_rows_to_other_messages_without_inventing_or_losing_any():
    link = link_matrix()
    out = ctl.shuffle_link(link, gen(3))
    assert sorted(map(tuple, out[:6, :-1].tolist())) == sorted(map(tuple, link[:6, :-1].tolist()))      # the same six rows, permuted
    assert not torch.equal(out[:6], link[:6])                                    # something moved
    again = ctl.shuffle_link(link, gen(3))
    assert torch.equal(out, again)                                               # the same generator gives the same permutation
    assert not torch.equal(out, ctl.shuffle_link(link, gen(4)))                  # another one gives another
    assert torch.equal(link, link_matrix())                                      # the input is not changed


def test_the_shuffle_is_the_same_permutation_for_links_of_different_widths():
    # every model gets the same wrong links: the permutation depends only on how many messages have a link
    link = link_matrix()
    wide = torch.cat([torch.arange(8.0).unsqueeze(1).repeat(1, 37), link[:, -1:]], dim=1)      # 37 numbers + has_url
    narrow, big = ctl.shuffle_link(link, gen(7)), ctl.shuffle_link(wide, gen(7))
    moved_from = big[:6, 0].long()                                               # row i received row moved_from[i] (its first column)
    assert torch.equal(narrow[:6, 0], link[moved_from, 0])


def test_a_single_message_with_a_link_cannot_be_shuffled_and_is_left_as_it_is():
    link = link_matrix()
    link[1:6, -1] = 0
    assert torch.equal(ctl.shuffle_link(link, gen()), link)


# ------------------------------------------------------------------ the shared-wording subset
def test_a_template_is_shared_wording_only_if_both_classes_use_it_enough():
    templates = pd.Series(["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 2)
    labels = pd.Series(["smishing"] * 5 + ["legitimate"] * 5 +           # a: 50 / 50
                       ["smishing"] * 10 +                                # b: one class only
                       ["smishing"] * 9 + ["legitimate"] * 1 +            # c: 10% rarer class
                       ["smishing", "legitimate"])                        # d: 50 / 50, two messages
    assert ctl.shared_wording_templates(templates, labels, 0.2) == {"a", "d"}
    assert ctl.shared_wording_templates(templates, labels, 0.05) == {"a", "c", "d"}
    assert ctl.shared_wording_templates(templates, labels, 0.6) == set()


def test_a_dataset_with_one_class_has_no_shared_wording():
    assert ctl.shared_wording_templates(pd.Series(["a", "b"]), pd.Series(["smishing", "smishing"])) == set()


# ------------------------------------------------------------------ the share of the link's benefit that is lost
def test_the_share_of_the_benefit_lost_is_zero_for_an_ignored_link_and_one_for_a_link_that_is_all_there_is():
    assert ctl.link_benefit_lost(0.95, 0.95, 0.80) == 0.0                         # shuffling changed nothing: the link was not used
    assert ctl.link_benefit_lost(0.95, 0.80, 0.80) == pytest.approx(1.0)          # back to the text-only level: all of the benefit
    assert ctl.link_benefit_lost(0.95, 0.875, 0.80) == pytest.approx(0.5)
    assert ctl.link_benefit_lost(0.95, 0.70, 0.80) > 1.0                          # a wrong link can mislead below the text-only level
    assert np.isnan(ctl.link_benefit_lost(0.80, 0.70, 0.80))                      # no benefit to lose


def predictions(rows):
    """rows: (seed, model, message_id, label, pred) -> the frame film_controls.collect writes (complete condition only)."""
    return pd.DataFrame([{"seed": s, "fold": 0, "message_id": m, "label": y, "model": model, "condition": "complete",
                          "p_smishing": float(p), "pred": p, "has_url": 1, "page_live": 1} for s, model, m, y, p in rows])


def test_the_shuffle_table_reports_accuracy_before_and_after_and_the_share_lost():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    ids = [f"m{i}" for i in range(8)]
    kept = predictions([(42, fc.BASELINE, i, y, 0) for i, y in zip(ids, labels)]                       # always "legitimate": accuracy 0.5
                       + [(42, fc.E2_SCRATCH, i, y, y) for i, y in zip(ids, labels)])                   # perfect with the right link
    shuffled = pd.concat([
        pd.DataFrame({"seed": 42, "fold": 0, "message_id": ids, "label": labels, "model": fc.BASELINE, "shuffle": k, "pred": 0})
        for k in range(2)] + [
        pd.DataFrame({"seed": 42, "fold": 0, "message_id": ids, "label": labels, "model": fc.E2_SCRATCH, "shuffle": k,
                      "pred": [1 - y if k == 0 else 0 for y in labels]}) for k in range(2)])             # wrong all the time / chance
    per_seed, summary = ctl.shuffle_tables(kept, shuffled)
    e2 = per_seed[per_seed["model"] == fc.E2_SCRATCH].iloc[0]
    assert e2["accuracy_right"] == 1.0 and e2["accuracy_shuffled"] == pytest.approx((0.0 + 0.5) / 2)
    assert e2["link_benefit_lost"] == pytest.approx((1.0 - 0.25) / (1.0 - 0.5))
    base = per_seed[per_seed["model"] == fc.BASELINE].iloc[0]
    assert base["accuracy_right"] == base["accuracy_shuffled"] == 0.5 and np.isnan(base["link_benefit_lost"])    # the floor, no benefit
    assert set(summary["model"]) == {fc.BASELINE, fc.E2_SCRATCH}


# ------------------------------------------------------------------ the shared-wording tables
def test_the_shared_wording_table_splits_the_messages_and_runs_mcnemar_on_the_subset():
    ids = [f"m{i}" for i in range(40)]
    labels = [i % 2 for i in range(40)]
    rows = []
    for model, right in ((fc.BASELINE, 0.5), (fc.E2_SCRATCH, 1.0), (fc.CONTROL_SCRATCH, 0.75)):
        for k, (i, y) in enumerate(zip(ids, labels)):
            correct = (k % 4) < right * 4                                          # a deterministic share right
            rows.append((42, model, i, y, y if correct else 1 - y))
    kept = predictions(rows)
    subset = set(ids[:20])
    summary, mc = ctl.shared_tables(kept, subset)
    shared = summary[(summary["subset"] == "shared wording") & (summary["model"] == fc.E2_SCRATCH)].iloc[0]
    rest = summary[(summary["subset"] == "the rest") & (summary["model"] == fc.E2_SCRATCH)].iloc[0]
    assert shared["n_mean"] == 20 and rest["n_mean"] == 20 and shared["accuracy_mean"] == 1.0
    assert set(mc["model_b"]) == {fc.BASELINE, fc.CONTROL_SCRATCH} and (mc["n"] == 20).all()
    vs_base = mc[mc["model_b"] == fc.BASELINE].iloc[0]
    assert vs_base["b"] == 10 and vs_base["c"] == 0 and vs_base["p_holm"] >= vs_base["p_exact"]


# ------------------------------------------------------------------ guards
def test_the_real_data_is_not_read_without_the_flag():
    with pytest.raises(PermissionError, match="--read-test"):
        ctl.run_controls(pilot=False, read_test=False)
    with pytest.raises(SystemExit) as stopped:
        ctl.main([])
    assert stopped.value.code == 2


# ------------------------------------------------------------------ the whole thing on the pilot
def pilot_models_exist():
    dataset = tc.dataset_path(True)
    from src.unified import fit_model as fm
    return all(fm.checkpoint_path(dataset, kind, 42, 0).exists() for kind in ("cnn", "film_scratch"))


@pytest.mark.skipif(not pilot_models_exist(), reason="the trained pilot models are not on this machine")
def test_a_pilot_run_writes_every_table_and_the_text_only_model_ignores_the_shuffle(tmp_path, monkeypatch):
    from src.unified import fit_model as fm
    monkeypatch.setattr(fm, "out_dir", lambda dataset, step: tmp_path / step)
    r = ctl.run_controls(pilot=True, seeds=[42], folds=[0], shuffles=2)
    per_seed = r["shuffle_per_seed"]
    base = per_seed[per_seed["model"] == fc.BASELINE].iloc[0]
    assert base["accuracy_right"] == base["accuracy_shuffled"]                    # it never reads the link
    e2 = per_seed[per_seed["model"] == fc.E2_SCRATCH].iloc[0]
    assert e2["accuracy_shuffled"] < e2["accuracy_right"]                          # E2 does read it
    for name in ("shuffle_summary.csv", "shuffle_per_seed.csv", "shared_wording_summary.csv", "shared_wording_mcnemar.csv",
                 "shared_wording_ids.csv", "test_predictions.csv", "test_reads.log"):
        assert (tmp_path / "controls" / name).exists()
    assert "read_test_flag=False" in (tmp_path / "controls" / "test_reads.log").read_text()
