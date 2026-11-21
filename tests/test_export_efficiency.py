"""Tests for scripts/export_efficiency.py: the file the frontend reads is made from the measurement's own files, nothing typed by hand.

Run with:  python -m pytest tests/test_export_efficiency.py
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_efficiency.py"
spec = importlib.util.spec_from_file_location("export_efficiency", SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

MODELS = ["proposed", "cnn_only", "rf_only", "concat"]


def make_files(root: Path, dataset: str = "crowd_relevant_synthetic", models=MODELS):
    eff = root / "experiments" / "film" / dataset / "efficiency"
    compare = root / "experiments" / "film" / dataset / "compare"
    eff.mkdir(parents=True)
    compare.mkdir(parents=True)
    rows = {}
    for i, m in enumerate(models):
        rows[m] = {"size_mb_mean": 5.0 + i, "size_mb_std": 0.1, "peak_memory_mb_mean": 20.0 + i, "peak_memory_mb_std": 0.5,
                   "mean_ms_mean": 1.0 + i, "mean_ms_std": 0.05, "mean_link_ms_mean": 2.0 + i, "mean_link_ms_std": 0.06,
                   "mean_nolink_ms_mean": float("nan") if m == "rf_only" else 0.5, "mean_nolink_ms_std": float("nan") if m == "rf_only" else 0.01}
    pd.DataFrame(rows).T.rename_axis("model").to_csv(eff / "form_e_summary.csv")
    pd.DataFrame([{"model": m, "seed": s, "fold": 0, "run": r, "n": 100 + i} for i, m in enumerate(models) for s in (42, 123) for r in (1, 2, 3)]).to_csv(
        eff / "form_d_runtime.csv", index=False)
    (eff / "form_a_environment.json").write_text(json.dumps({"measured_at": "2026-10-21T05:00:00", "cpu": "Test CPU", "cores_logical": 8,
                                                              "ram_gb": 16.0, "os": "Windows", "python": "3.13", "torch": "2.14",
                                                              "scikit_learn": "1.8"}), encoding="utf-8")
    pd.DataFrame([{"model": "proposed", "seed": 42, "fold": 0, "parameters_all": 1000, "parameters_trainable_without_embedding": 70658, "forest_trees": 100},
                  {"model": "proposed", "seed": 123, "fold": 0, "parameters_all": 1000, "parameters_trainable_without_embedding": 70658, "forest_trees": 100},
                  {"model": "rf_only", "seed": 42, "fold": 0, "forest_trees": 100}]).to_csv(eff / "descriptive_details.csv", index=False)
    pd.DataFrame([{"condition": "complete", "model": "E2 (from scratch)", "f1_mean": 0.953, "f1_std": 0.001},
                  {"condition": "page hidden", "model": "E2 (from scratch)", "f1_mean": 0.914, "f1_std": 0.002},
                  {"condition": "complete", "model": "CNN-only (baseline)", "f1_mean": 0.865, "f1_std": 0.002},
                  {"condition": "complete", "model": "Concatenation (from scratch)", "f1_mean": 0.954, "f1_std": 0.002}]).to_csv(compare / "test_summary.csv", index=False)
    pd.DataFrame([{"condition": "complete", "model": "RF-only (Link RF)", "f1_mean": 0.858, "f1_std": 0.001}]).to_csv(compare / "rf_only_summary.csv", index=False)


def test_the_export_has_every_model_in_the_display_order_with_the_proposed_one_last(tmp_path):
    make_files(tmp_path)
    data = exp.build("crowd_relevant_synthetic", root=tmp_path)
    assert data["order"] == ["cnn_only", "rf_only", "concat", "proposed"]
    assert data["models"]["proposed"]["proposed"] is True and not data["models"]["concat"]["proposed"]
    assert data["seeds"] == [42, 123] and data["runs_per_seed"] == 3 and data["n_test_messages"] == 103
    assert data["environment"]["cpu"] == "Test CPU" and data["measured_at"] == "2026-10-21T05:00:00"
    json.dumps(data)                                                           # plain JSON: no NaN, no numpy types


def test_the_numbers_are_the_measurements_and_a_missing_one_is_null(tmp_path):
    make_files(tmp_path)
    m = exp.build("crowd_relevant_synthetic", root=tmp_path)["models"]
    assert m["proposed"]["size_mb"] == {"mean": 5.0, "sd": 0.1}
    assert m["proposed"]["runtime_ms"]["with_link"] == {"mean": 2.0, "sd": 0.06}
    assert m["rf_only"]["runtime_ms"]["no_link"] is None                      # the forest cannot judge a message with no link
    assert m["proposed"]["parameters_trainable"] == 70658 and m["proposed"]["forest_trees"] == 100
    assert m["cnn_only"]["parameters_trainable"] is None and m["cnn_only"]["forest_trees"] is None


def test_f1_comes_from_the_comparison_on_complete_pages_for_the_right_model(tmp_path):
    make_files(tmp_path)
    m = exp.build("crowd_relevant_synthetic", root=tmp_path)["models"]
    assert m["proposed"]["f1"] == {"mean": 0.953, "sd": 0.001}                  # not the page-hidden 0.914
    assert m["cnn_only"]["f1"]["mean"] == 0.865 and m["concat"]["f1"]["mean"] == 0.954
    assert m["rf_only"]["f1"]["mean"] == 0.858                                 # from rf_only_summary.csv


def test_a_synthetic_dataset_is_a_dry_run_and_anything_else_is_final_unless_told(tmp_path):
    make_files(tmp_path, "crowd_relevant_synthetic")
    make_files(tmp_path, "crowd_smishing_v1")
    assert exp.build("crowd_relevant_synthetic", root=tmp_path)["status"] == "dry_run"
    assert exp.build("crowd_smishing_v1", root=tmp_path)["status"] == "final"
    assert exp.build("crowd_smishing_v1", status="dry_run", root=tmp_path)["status"] == "dry_run"


def test_the_map_of_served_models_starts_empty_and_survives_a_new_export(tmp_path):
    make_files(tmp_path)
    assert exp.build("crowd_relevant_synthetic", root=tmp_path)["live_model_ids"] == {}
    kept = exp.build("crowd_relevant_synthetic", previous={"live_model_ids": {"fusion_gated": "proposed"}}, root=tmp_path)
    assert kept["live_model_ids"] == {"fusion_gated": "proposed"}


def test_main_writes_the_file_and_keeps_the_map_when_it_runs_again(tmp_path):
    make_files(tmp_path)
    out = tmp_path / "out" / "efficiencyResults.json"
    args = ["--dataset", "crowd_relevant_synthetic", "--out", str(out), "--root", str(tmp_path)]
    exp.main(args)
    first = json.loads(out.read_text(encoding="utf-8"))
    assert first["status"] == "dry_run" and len(first["models"]) == 4
    first["live_model_ids"] = {"fusion_gated": "proposed"}
    out.write_text(json.dumps(first), encoding="utf-8")
    exp.main(args)
    assert json.loads(out.read_text(encoding="utf-8"))["live_model_ids"] == {"fusion_gated": "proposed"}


def test_live_model_says_which_measured_model_the_backend_serves_and_refuses_an_unknown_one(tmp_path):
    make_files(tmp_path)
    out = tmp_path / "out" / "efficiencyResults.json"
    args = ["--dataset", "crowd_relevant_synthetic", "--out", str(out), "--root", str(tmp_path)]
    exp.main(args + ["--live-model", "film_cnn=proposed"])
    assert json.loads(out.read_text(encoding="utf-8"))["live_model_ids"] == {"film_cnn": "proposed"}
    exp.main(args)                                                             # a later export keeps it
    assert json.loads(out.read_text(encoding="utf-8"))["live_model_ids"] == {"film_cnn": "proposed"}
    import pytest
    with pytest.raises(SystemExit):
        exp.main(args + ["--live-model", "film_cnn=not_a_model"])
