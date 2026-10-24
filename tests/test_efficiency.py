"""Tests for src/unified/efficiency.py (SOP 2, plan docs/film_architecture_plan.md, steps 4.3a to 4.3d).

The measuring tools (the memory monitor, the job order, the summary) are tested on their own. The per-message pipelines and a whole
measurement use the (git-ignored) synthetic pilot files and are skipped where they are missing. Everything is written to a temporary folder.

Run with:  python -m pytest tests/test_efficiency.py
"""
import time

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import CROWD_PILOT_PATH
from src.data.split_crowd import load_fold
from src.unified import efficiency as eff
from src.unified import film
from src.unified import film_compare as fc
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc
from src.unified.film_data import part_inputs


# ------------------------------------------------------------------ the tools, no pilot files needed
def test_every_model_seed_and_run_is_planned_once_and_run_1_comes_before_run_2():
    models, seeds = list(eff.MODELS), [42, 123, 2024]
    order = eff.job_order(models, seeds)
    assert len(order) == len(set(order)) == len(models) * len(seeds) * 3 == 63
    assert [job[2] for job in order] == sorted(job[2] for job in order)                  # all run-1 jobs, then run 2, then run 3
    assert order == eff.job_order(models, seeds)                                         # the same order every time
    assert order != eff.job_order(models, seeds, rng_seed=1)
    assert len({job[0] for job in order[:len(models) * len(seeds)]}) == len(models)       # run 1 covers every model
    first = {eff.job_order(models, seeds, rng_seed=k)[0][0] for k in range(30)}
    assert len(first) > 1                                                                # no model always goes first


def test_the_instruments_five_models_are_there_and_the_extras_are_marked():
    assert {k for k, v in eff.MODELS.items() if not v["extra"]} == {"proposed", "cnn_only", "rf_only", "concat", "voting"}
    assert {k for k, v in eff.MODELS.items() if v["extra"]} == {"proposed_contrib", "raw_cnn"}


def test_the_memory_monitor_sees_an_allocation_made_while_it_runs():
    import psutil
    before = psutil.Process().memory_info().rss
    with eff.PeakRSS() as monitor:
        block = np.ones(8_000_000)                                  # 64 MB, touched, so it is resident
        time.sleep(0.05)
        assert block.sum() == 8_000_000
    assert monitor.peak - before > 40 * eff.MB
    assert monitor.peak >= before


def test_append_row_writes_a_header_once(tmp_path):
    path = tmp_path / "log.csv"
    eff.append_row(path, {"a": 1, "b": 2})
    eff.append_row(path, {"a": 3, "b": 4})
    assert path.read_text().splitlines() == ["a,b", "1,2", "3,4"]


def test_pm_formats_a_mean_and_sd_and_a_missing_value():
    assert eff.pm(31.4, 0.2, 1) == "31.4 ± 0.2" and eff.pm(np.nan, np.nan) == "-" and eff.pm(2.0, np.nan) == "2.00 ± 0.00"


def test_the_summary_adds_the_files_of_a_model_averages_the_runs_and_takes_the_sd_over_the_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(eff, "out_dir", lambda dataset: tmp_path)
    sizes = [("proposed", s, c, v) for s, parts in ((42, (1.0, 2.0, 3.0)), (123, (2.0, 2.0, 4.0)), (2024, (3.0, 2.0, 5.0)))
             for c, v in zip(("weights", "vocabulary", "Link RF"), parts)]
    pd.DataFrame([{"model": m, "seed": s, "fold": 0, "component": c, "file": c, "size_mb": v} for m, s, c, v in sizes]).to_csv(
        tmp_path / "form_b_model_size.csv", index=False)
    memory = [{"model": "proposed", "seed": s, "fold": 0, "run": r, "n": 10, "rss_baseline_mb": 0, "rss_peak_mb": 0, "peak_memory_mb": 10.0 * s / 42 + r}
              for s in (42, 123, 2024) for r in (1, 2, 3)]
    pd.DataFrame(memory).to_csv(tmp_path / "form_c_peak_memory.csv", index=False)
    runtime = [{"model": "proposed", "seed": s, "fold": 0, "run": r, "n": 10, "mean_ms": 10.0 + r, "mean_link_ms": 12.0 + r,
                "mean_nolink_ms": 1.0} for s in (42, 123, 2024) for r in (1, 2, 3)]
    pd.DataFrame(runtime).to_csv(tmp_path / "form_d_runtime.csv", index=False)
    summary = eff.summarize(pilot=True)
    row = summary.loc["proposed"]
    assert row["size_mb_mean"] == pytest.approx(np.mean([6.0, 8.0, 10.0])) and row["size_mb_std"] == pytest.approx(2.0)     # sum of files, then SD (n-1)
    assert row["mean_ms_mean"] == pytest.approx(12.0) and row["mean_ms_std"] == pytest.approx(0.0)                           # runs averaged per seed
    assert row["peak_memory_mb_mean"] == pytest.approx(np.mean([10.0 * s / 42 + 2 for s in (42, 123, 2024)]))
    assert (tmp_path / "form_e_summary.csv").exists() and "Proposed (CNN–RF + FiLM)" in (tmp_path / "form_e_summary.md").read_text(encoding="utf-8")


def test_a_resumed_run_does_only_the_jobs_that_have_both_a_memory_and_a_runtime_row(tmp_path):
    assert eff.completed_jobs(tmp_path) == set()                                   # no logs yet: nothing is done
    row = lambda model, seed, run: {"model": model, "seed": seed, "fold": 0, "run": run}
    pd.DataFrame([row("proposed", 42, 1), row("proposed", 42, 2), row("cnn_only", 42, 1)]).to_csv(tmp_path / "form_c_peak_memory.csv", index=False)
    pd.DataFrame([row("proposed", 42, 1), row("cnn_only", 42, 1)]).to_csv(tmp_path / "form_d_runtime.csv", index=False)
    done = eff.completed_jobs(tmp_path)
    assert done == {("proposed", 42, 1), ("cnn_only", 42, 1)}                      # run 2 of proposed died between the two files: redo it
    todo = [job for job in eff.job_order(["proposed", "cnn_only"], [42], (1, 2)) if job not in done]
    assert sorted(todo) == [("cnn_only", 42, 2), ("proposed", 42, 2)]


# ------------------------------------------------------------------ the pilot
def pilot_available():
    dataset = tc.dataset_path(True)
    return ((tc.ROOT / CROWD_PILOT_PATH).exists() and film.link_rf_path(dataset, 42, 0).exists()
            and (tc.embeddings_dir(dataset, 42, 0) / "vocab.json").exists()
            and all(fm.checkpoint_path(dataset, kind, 42, 0).exists() for kind in ("cnn", "film_scratch", "concat_scratch")))


needs_pilot = pytest.mark.skipif(not pilot_available(), reason="the trained pilot models are not on this machine")
DATASET = tc.dataset_path(True)


@needs_pilot
def test_the_files_counted_for_each_model_are_the_ones_a_deployment_reads():
    names = lambda m: [c for c, _ in eff.component_files(m, DATASET, 42, 0)]
    assert names("proposed") == names("proposed_contrib") == names("concat") == ["weights", "configuration", "vocabulary", "Link RF"]
    assert names("cnn_only") == ["weights", "configuration", "vocabulary"]
    assert names("rf_only") == ["Link RF"]
    assert names("voting") == ["TF-IDF vectorizer + KNN/RF/ExtraTrees vote"]
    for model in ("proposed", "cnn_only", "rf_only", "concat"):
        assert all(path.exists() for _, path in eff.component_files(model, DATASET, 42, 0))
    assert not any("embedding_matrix" in path.name or path.suffix == ".csv" for m in eff.MODELS
                   for _, path in eff.component_files(m, DATASET, 42, 0))                     # the starting matrix and the data are not counted


@needs_pilot
def test_a_model_is_loaded_from_its_sidecar_and_weights_without_the_starting_embedding_matrix():
    net, word_to_idx, seq_len, sidecar = eff.deployed_torch_model("proposed", DATASET, 42, 0)
    reference, ref_vocab, ref_len, _ = film.load_film(42, 0, pilot=True, warm_start=False)
    assert word_to_idx == ref_vocab and seq_len == ref_len and not net.training
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), reference.state_dict().values()))


@needs_pilot
def test_the_one_message_pipelines_give_the_same_labels_as_the_evaluation_pipeline():
    eff.import_all_libraries()
    n = 80
    messages = eff.load_messages(True, 42, 0, limit=n)
    batch = {}
    for model, name in (("proposed", fc.E2_SCRATCH), ("cnn_only", fc.BASELINE), ("concat", fc.CONTROL_SCRATCH)):
        net, vocab, length = fc.LOADERS[name][1](42, 0, True)
        test = part_inputs(42, 0, "test", vocab, length, True)
        test = test._replace(token_ids=test.token_ids[:n], link_full=test.link_full[:n])
        batch[model] = (fc.predict(net, name, test, "complete") >= 0.5).astype(int)
    for model, expected in batch.items():
        classify = eff.build_classifier(model, True, 42, 0)
        assert [classify(m) for m in messages] == list(expected), model
    # the forest alone, scored like film.predict_messages' forest, and E2 with the 37 contributions computed as well
    rf = film.load_link_rf(DATASET, 42, 0)
    rf_only = eff.build_classifier("rf_only", True, 42, 0)
    p, _ = __import__("src.features.model.rf_extractor", fromlist=["p_and_u"]).p_and_u(rf, pd.concat([m["features"] for m in messages]))
    assert [rf_only(m) for m in messages] == list((p >= 0.5).astype(int))
    contrib = eff.build_classifier("proposed_contrib", True, 42, 0)
    assert [contrib(m) for m in messages] == list(batch["proposed"])                # the contributions are computed, the verdict does not change


@needs_pilot
def test_a_message_with_no_link_skips_the_forest_and_gets_the_text_only_treatment_of_the_bypass():
    eff.import_all_libraries()
    message = eff.load_messages(True, 42, 0, limit=1)[0]
    no_link = {**message, "has_url": 0, "page_missing": 1}
    net, vocab, length = film.load_film(42, 0, pilot=True, warm_start=False)[:3]
    rf = film.load_link_rf(DATASET, 42, 0)
    rows = load_fold(42, 0, pilot=True)["test"].iloc[:1].copy()
    rows["has_url"], rows["page_fetched"] = 0, 0
    expected = film.predict_messages(net, rf, vocab, length, rows)["pred"].iloc[0]
    classify = eff.build_classifier("proposed", True, 42, 0)
    assert classify(no_link) == expected
    # it is also much faster than a message with a link, because the forest is not run
    start = time.perf_counter()
    for _ in range(5):
        classify(no_link)
    quick = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(5):
        classify(message)
    assert quick < (time.perf_counter() - start)


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "out_dir", lambda dataset, step: tmp_path / "experiments" / dataset.stem / step)
    return tmp_path / "experiments" / DATASET.stem / "efficiency"


@needs_pilot
def test_one_measurement_writes_size_memory_and_runtime_rows(outputs):
    for model in ("cnn_only", "proposed"):
        result = eff.measure(model, True, 42, 0, run=1, limit=30)
        assert result["memory"]["n"] == result["runtime"]["n"] == 30 and result["memory"]["peak_memory_mb"] >= 0
        assert result["runtime"]["mean_ms"] > 0 and result["runtime"]["sd_ms"] >= 0
    sizes = pd.read_csv(outputs / "form_b_model_size.csv")
    assert set(sizes["component"]) == {"weights", "configuration", "vocabulary", "Link RF"} and (sizes["size_mb"] > 0).all()
    assert len(pd.read_csv(outputs / "form_c_peak_memory.csv")) == len(pd.read_csv(outputs / "form_d_runtime.csv")) == 2
    eff.measure("cnn_only", True, 42, 0, run=2, limit=30)                              # run 2: no second size row
    assert len(pd.read_csv(outputs / "form_b_model_size.csv")) == len(sizes)


@needs_pilot
def test_rf_only_is_measured_on_the_messages_with_a_link_only(outputs):
    result = eff.measure("rf_only", True, 42, 0, run=1, limit=25)
    assert result["runtime"]["n"] <= 25 and result["runtime"]["mean_nolink_ms"] == ""


@needs_pilot
def test_the_descriptive_details_count_the_forest_and_the_network(outputs):
    row = eff.describe("proposed", True, 42, 0)
    assert row["parameters_trainable_without_embedding"] == 70658 and row["parameters_all"] > 70658
    assert row["forest_trees"] == 100 and row["forest_nodes"] > row["forest_trees"]
    assert (outputs / "descriptive_details.csv").exists()


@needs_pilot
def test_the_voting_model_is_saved_and_the_saved_one_predicts_like_the_fitted_one(tmp_path, monkeypatch):
    pytest.importorskip("imblearn")
    pytest.importorskip("nltk")
    pytest.importorskip("urlextract")
    from src.unified import voting_baseline as vb
    monkeypatch.setattr(fm, "model_dir", lambda dataset: tmp_path / "models" / dataset.stem)
    eff.save_voting(True, seeds=[42], folds=[0])
    assert eff.voting_path(DATASET, 42, 0).exists()
    parts = load_fold(42, 0, pilot=True)
    expected = vb.fit_and_score(parts["train"], parts["test"].iloc[:60], 42)[1]
    eff.import_all_libraries()
    classify = eff.build_classifier("voting", True, 42, 0)
    assert [classify(m) for m in eff.load_messages(True, 42, 0, limit=60)] == list(expected)


def test_the_environment_form_records_the_machine_and_what_can_be_checked():
    form = eff.environment(pilot=True)
    for key in ("cpu", "ram_gb", "os", "python", "torch", "scikit_learn", "psutil", "numpy", "measured_at"):
        assert form[key]
    assert form["checked_by_the_script"]["gpu_disabled"] is True and form["checked_by_the_script"]["forests_n_jobs"] == 1
    assert form["to_tick_by_hand"]
