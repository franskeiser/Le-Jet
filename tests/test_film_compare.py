"""Tests for src/unified/film_compare.py (plan docs/film_architecture_plan.md, Phase 4: RQ1 and RQ3).

The statistics tests run anywhere. The last two use the (git-ignored) synthetic pilot and its trained models, read-only, and write to a
temporary folder; they are skipped where those are missing.

Run with:  python -m pytest tests/test_film_compare.py
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.unified import film_compare as fc
from src.unified import fit_model as fm
from src.unified import text_cnn as tc
from src.unified.film_data import PartData


# ------------------------------------------------------------------ statistics
def test_scores_use_smishing_as_the_positive_class():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])                 # 2 caught, 1 missed, 1 false alarm, 4 passed
    s = fc.scores(y, pred)
    assert s["accuracy"] == pytest.approx(6 / 8) and s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3) and s["f1"] == pytest.approx(2 / 3)


def test_scores_do_not_fail_when_nothing_is_predicted_positive():
    s = fc.scores(np.array([1, 0, 1, 0]), np.array([0, 0, 0, 0]))
    assert s["precision"] == 0 and s["recall"] == 0 and s["f1"] == 0 and s["accuracy"] == 0.5


def test_mcnemar_counts_the_disagreements_and_gives_the_exact_p_value():
    y = np.ones(14, dtype=int)
    a = np.array([1] * 10 + [0] * 4)
    b = np.array([0] * 10 + [0] * 4)                          # A right where B is wrong 10 times; both wrong 4 times
    r = fc.mcnemar(y, a, b)
    assert (r["b"], r["c"]) == (10, 0) and r["p_exact"] == pytest.approx(2 * 0.5 ** 10)
    assert fc.mcnemar(y, b, a)["p_exact"] == pytest.approx(r["p_exact"])                       # symmetric
    assert fc.mcnemar(y, a, a) == {"b": 0, "c": 0, "p_exact": 1.0}                            # identical models: no evidence
    even = fc.mcnemar(np.ones(8, dtype=int), np.array([1, 1, 1, 1, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    assert (even["b"], even["c"]) == (4, 4) and even["p_exact"] == pytest.approx(1.0)


def test_holm_correction_is_step_down_monotone_and_capped():
    assert fc.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert fc.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert fc.holm([0.2]) == [0.2]
    adjusted = fc.holm([0.001, 0.2, 0.0005, 0.04])
    assert all(a >= p for a, p in zip(adjusted, [0.001, 0.2, 0.0005, 0.04])) and all(0 <= a <= 1 for a in adjusted)


def synthetic_predictions():
    rng = np.random.default_rng(0)
    rows = []
    for seed in (1, 2, 3):
        y = rng.integers(0, 2, 60)
        for model, flip in ((fc.BASELINE, 0.30), (fc.E2, 0.10)):
            for condition in fc.CONDITIONS:
                pred = np.where(rng.random(60) < flip, 1 - y, y)
                rows.append(pd.DataFrame({"seed": seed, "fold": 0, "message_id": [f"m{i}" for i in range(60)], "label": y,
                                          "model": model, "condition": condition, "p_smishing": pred.astype(float), "pred": pred}))
    return pd.concat(rows, ignore_index=True)


def test_metric_tables_give_per_seed_values_and_the_mean_and_std_over_seeds():
    predictions = synthetic_predictions()
    per_seed, summary = fc.metrics_tables(predictions)
    assert len(per_seed) == 3 * 2 * 2 and set(per_seed["n"]) == {60}
    row = summary[(summary["condition"] == "complete") & (summary["model"] == fc.E2)].iloc[0]
    values = per_seed[(per_seed["condition"] == "complete") & (per_seed["model"] == fc.E2)]["accuracy"]
    assert row["accuracy_mean"] == pytest.approx(values.mean()) and row["accuracy_std"] == pytest.approx(values.std(ddof=1))
    assert row["seeds"] == 3 and {f"{m}_mean" for m in fc.METRICS} <= set(summary.columns)


def test_the_mcnemar_table_has_one_row_per_seed_and_pair_and_only_for_the_models_present():
    table = fc.mcnemar_table(synthetic_predictions())
    assert len(table) == 3 and set(table["model_a"]) == {fc.E2} and set(table["model_b"]) == {fc.BASELINE}
    assert (table["p_holm"] >= table["p_exact"]).all() and table["significant_0.05"].dtype == bool
    full = synthetic_predictions()
    extra = full[full["model"] == fc.E2].assign(model=fc.CONTROL)
    bigger = fc.mcnemar_table(pd.concat([full, extra], ignore_index=True))
    assert len(bigger) == 3 * 3                                              # E2 v baseline, E2 v control, control v baseline


# ------------------------------------------------------------------ guards on the way predictions are made
class Fake(torch.nn.Module):
    def forward(self, token_ids, link):
        return torch.stack([torch.zeros(len(token_ids)), token_ids[:, 0].float() - 5], dim=1)


def fake_part(ids):
    n = len(ids)
    return PartData(message_id=np.array(ids), token_ids=torch.arange(n).reshape(n, 1) + 3, labels=torch.tensor([0, 1] * (n // 2)),
                    link_full=torch.zeros(n, 3), link_blank=torch.ones(n, 3))


def test_a_message_tested_twice_is_refused(monkeypatch):
    monkeypatch.setattr(fc, "LOADERS", {fc.BASELINE: ("cnn", lambda s, f, p: (Fake(), {"a": 1}, 5))})
    monkeypatch.setattr(fc, "part_inputs", lambda *a, **k: fake_part(["m0", "m1", "m2", "m3"]))
    with pytest.raises(AssertionError, match="tested twice"):
        fc.test_predictions(True, [42], [0, 1], [fc.BASELINE])                # two folds returning the same four messages


def test_models_with_different_vocabularies_are_refused(monkeypatch):
    monkeypatch.setattr(fc, "LOADERS", {fc.BASELINE: ("cnn", lambda s, f, p: (Fake(), {"a": 1}, 5)),
                                        fc.E2: ("film", lambda s, f, p: (Fake(), {"b": 1}, 5))})
    monkeypatch.setattr(fc, "part_inputs", lambda *a, **k: fake_part(["m0", "m1"]))
    with pytest.raises(AssertionError, match="vocabulary"):
        fc.test_predictions(True, [42], [0], [fc.BASELINE, fc.E2])


def test_the_link_is_read_by_condition_and_the_baseline_does_not_use_it(monkeypatch):
    seen = []

    class Spy(torch.nn.Module):
        def forward(self, token_ids, link):
            seen.append(float(link.mean()))
            return torch.zeros(len(token_ids), 2)

    part = fake_part(["m0", "m1"])
    fc.predict(Spy(), "x", part, "complete")
    fc.predict(Spy(), "x", part, "page hidden")
    assert seen == [0.0, 1.0]                                                  # full-page numbers, then the page-hidden numbers


# ------------------------------------------------------------------ the test read
def test_the_real_data_is_not_read_without_the_flag(monkeypatch):
    with pytest.raises(PermissionError, match="--read-test"):
        fc.run_compare(pilot=False, read_test=False)
    with pytest.raises(SystemExit) as stopped:
        fc.main([])
    assert stopped.value.code == 2


def pilot_models_exist():
    dataset = tc.dataset_path(True)
    return all(fm.checkpoint_path(dataset, kind, 42, 0).exists() for kind in ("cnn", "film"))


@pytest.mark.skipif(not pilot_models_exist(), reason="the trained pilot models are not on this machine")
def test_a_pilot_run_scores_every_message_once_writes_its_tables_and_logs_each_read(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "out_dir", lambda dataset, step: tmp_path / step)
    result = fc.run_compare(pilot=True, seeds=[42], folds=[0, 1])
    predictions = result["predictions"]
    assert fc.BASELINE in result["models"] and fc.E2 in result["models"]
    for (model, condition), g in predictions.groupby(["model", "condition"]):
        assert g["message_id"].is_unique and set(g["fold"]) == {0, 1}
    ids = {m: set(g["message_id"]) for m, g in predictions[predictions["condition"] == "complete"].groupby("model")}
    assert len({frozenset(v) for v in ids.values()}) == 1                      # every model scored the same messages
    base = predictions[(predictions["model"] == fc.BASELINE)]
    assert (base[base["condition"] == "complete"].set_index("message_id")["p_smishing"]
            == base[base["condition"] == "page hidden"].set_index("message_id")["p_smishing"]).all()   # the baseline has no link
    out = tmp_path / "compare"
    assert {"test_predictions.csv", "test_metrics.csv", "test_summary.csv", "test_mcnemar.csv", "test_reads.log"} <= {p.name for p in out.iterdir()}
    fc.run_compare(pilot=True, seeds=[42], folds=[0])
    assert len((out / "test_reads.log").read_text().strip().splitlines()) == 2        # every read is logged
    assert "seeds=[42]" in (out / "test_reads.log").read_text()
