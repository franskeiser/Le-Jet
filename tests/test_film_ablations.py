"""Tests for src/unified/film_ablations.py (plan docs/film_architecture_plan.md, steps 5.3 and 5.6).

The parameter counts and the McNemar table are tested on their own; the link tensors, training, reloading and the comparison use the
(git-ignored) synthetic pilot files and are skipped where they are missing. Everything is written to a temporary folder.

Run with:  python -m pytest tests/test_film_ablations.py
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import CROWD_PILOT_PATH
from src.unified import cnn_baseline as cb
from src.unified import concat_control as cc
from src.unified import film
from src.unified import film_ablations as fa
from src.unified import film_compare as fc
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc
from src.unified.film_data import part_inputs


# ------------------------------------------------------------------ no pilot files needed
def test_the_parameter_counts_follow_from_what_each_generator_reads():
    assert fa.expected_trainable("no_average") == 57664 + 3 * 128 + 128 + 4160 + 130 == 62466       # -8,192 against E2's 70,658
    assert fa.expected_trainable("contributions") == 70658 + (37 - 3) * 128 == 75010               # +4,352
    assert fa.expected_trainable("no_u") == 70658 - 128 == 70530
    assert film.expected_trainable() == 70658                                                       # the main E2 is unchanged


def test_each_variant_has_a_label_and_a_checkpoint_kind_of_its_own():
    assert len({fa.label_of(v) for v in fa.VARIANTS}) == 3 and len({fa.kind_of(v) for v in fa.VARIANTS}) == 3
    assert all(fa.kind_of(v).startswith("film_scratch_") for v in fa.VARIANTS)
    assert fa.kind_of("no_u") != "film_scratch"                                                     # never overwrites the main E2


def toy_predictions():
    rows = []
    labels = [i % 2 for i in range(40)]
    for model, right in ((fc.E2_SCRATCH, 1.0), (fc.CONTROL_SCRATCH, 0.75), (fc.BASELINE, 0.5), (fa.label_of("no_average"), 1.0)):
        for k, y in enumerate(labels):
            correct = (k % 4) < right * 4
            rows.append({"seed": 42, "message_id": f"m{k}", "label": y, "model": model, "condition": "complete",
                         "pred": y if correct else 1 - y})
    return pd.DataFrame(rows)


def test_mcnemar_compares_the_main_e2_with_every_other_model_and_corrects_over_the_list():
    mc = fa.mcnemar_table(toy_predictions(), ["no_average"])
    assert list(mc["model_b"]) == [fc.CONTROL_SCRATCH, fc.BASELINE, fa.label_of("no_average")]
    same = mc[mc["model_b"] == fa.label_of("no_average")].iloc[0]
    assert same["b"] == 0 and same["c"] == 0 and same["p_exact"] == 1.0 and not same["significant_0.05"]      # identical predictions
    base = mc[mc["model_b"] == fc.BASELINE].iloc[0]
    assert base["b"] == 20 and base["c"] == 0 and base["p_holm"] >= base["p_exact"]            # right on all 40, the baseline on 20


# ------------------------------------------------------------------ the pilot
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
def test_each_variant_gets_the_link_numbers_it_is_meant_to_read():
    _, vocab, seq_len = film.build_film(42, 0, pilot=True)
    base = part_inputs(42, 0, "val", vocab, seq_len, pilot=True)
    pu = fa.variant_part("no_average", 42, 0, "val", vocab, seq_len, pilot=True)
    assert torch.equal(pu.link_full, base.link_full) and torch.equal(pu.link_blank, base.link_blank)           # same numbers, other generator
    no_u = fa.variant_part("no_u", 42, 0, "val", vocab, seq_len, pilot=True)
    assert tuple(no_u.link_full.shape) == (len(base.labels), 3)
    assert torch.equal(no_u.link_full[:, 0], base.link_full[:, 0]) and torch.equal(no_u.link_full[:, 1], base.link_full[:, 2])
    assert torch.equal(no_u.link_full[:, 2], base.link_full[:, 3])                                              # [P, page_missing, has_url]
    contrib = fa.variant_part("contributions", 42, 0, "val", vocab, seq_len, pilot=True)
    assert tuple(contrib.link_full.shape) == tuple(contrib.link_blank.shape) == (len(base.labels), 38)
    assert torch.equal(contrib.link_full[:, 37], base.link_full[:, 3]) and torch.equal(contrib.token_ids, base.token_ids)
    # P = bias + the sum of the contributions: what the generator reads is the forest's own breakdown of P
    lane_full, _ = ll.load_lane(42, 0, pilot=True)["val"]
    aligned = pd.DataFrame({"message_id": base.message_id}).merge(lane_full, on="message_id", validate="one_to_one")
    assert np.allclose(contrib.link_full[:, :37].numpy(), aligned[ll.CONTRIBUTIONS].to_numpy(dtype=np.float32))
    assert not torch.equal(contrib.link_full[:, :37], contrib.link_blank[:, :37])                               # a hidden page changes them


@needs_pilot
def test_training_one_cell_per_variant_saves_a_sidecar_that_reloads_to_the_same_validation_numbers(outputs):
    for variant in fa.VARIANTS:
        row, log = fa.train_cell(variant, 42, 0, pilot=True, epochs=2)
        checkpoint = fm.checkpoint_path(DATASET, fa.kind_of(variant), 42, 0)
        sidecar = fm.read_sidecar(checkpoint)
        assert sidecar["variant"] == variant and sidecar["trainable_without_embedding"] == fa.expected_trainable(variant)
        assert row["trainable_without_embedding"] == fa.expected_trainable(variant) and len(log) == row["epochs_run"]
        assert set(sidecar["sha256"]) == {"vocab.json", "embedding_matrix.npy", checkpoint.name, "link_rf"}
        assert sidecar["warm_start"] is None and sidecar["options"]["bypass_no_link"] is True
        model, vocab, seq_len, _ = fa.load_variant(variant, 42, 0, pilot=True)
        val = fa.variant_part(variant, 42, 0, "val", vocab, seq_len, pilot=True)
        again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
        assert all(again[k] == sidecar["val"][k] == row[k] for k in fm.VAL_METRICS)
    summary = fa.run_check(pilot=True, seeds=[42], folds=[0])
    assert len(summary) == 3 and summary["ok"].all() and (summary["max_gap"] == 0).all()


@needs_pilot
def test_the_variants_only_differ_from_the_main_e2_in_what_the_generator_reads(outputs):
    fa.train_cell("no_average", 42, 0, pilot=True, epochs=1)
    model, _, _, sidecar = fa.load_variant("no_average", 42, 0, pilot=True)
    assert model.use_average is False and model.generator.in_features == 3 and sidecar["options"]["use_average"] is False
    assert sidecar["options"]["gamma_mode"] == "bounded" and sidecar["options"]["freeze_embedding"] is False
    fa.train_cell("contributions", 42, 0, pilot=True, epochs=1)
    model, _, _, sidecar = fa.load_variant("contributions", 42, 0, pilot=True)
    assert model.use_average is True and model.generator.in_features == 37 + 64 and sidecar["link_dim"] == 37


@needs_pilot
def test_a_changed_weights_file_is_refused(outputs):
    fa.train_cell("no_u", 42, 0, pilot=True, epochs=1)
    checkpoint = fm.checkpoint_path(DATASET, fa.kind_of("no_u"), 42, 0)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed since"):
        fa.load_variant("no_u", 42, 0, pilot=True)


@needs_pilot
def test_the_real_data_is_not_read_without_the_flag():
    with pytest.raises(PermissionError, match="--read-test"):
        fa.run_compare(pilot=False, read_test=False)
    with pytest.raises(SystemExit) as stopped:
        fa.main(["compare"])
    assert stopped.value.code == 2


@needs_pilot
def test_a_comparison_scores_every_message_once_and_logs_the_read(outputs):
    cb.train_cell(42, 0, pilot=True, epochs=1)
    cc.train_cell(42, 0, pilot=True, warm_start=False, epochs=1)
    film.train_cell(42, 0, pilot=True, warm_start=False, epochs=1)
    for variant in fa.VARIANTS:
        fa.train_cell(variant, 42, 0, pilot=True, epochs=1)
    result = fa.run_compare(pilot=True, seeds=[42], folds=[0])
    predictions = result["predictions"]
    assert set(predictions["model"]) == {fc.BASELINE, fc.CONTROL_SCRATCH, fc.E2_SCRATCH} | {fa.label_of(v) for v in fa.VARIANTS}
    for (model, condition), g in predictions.groupby(["model", "condition"]):
        assert g["message_id"].is_unique
    ids = {m: frozenset(g["message_id"]) for m, g in predictions[predictions["condition"] == "complete"].groupby("model")}
    assert len(set(ids.values())) == 1                                                       # every model scored the same messages
    assert set(result["val"]["model"]) == set(ids) - {fc.BASELINE}
    log = (outputs / "experiments" / DATASET.stem / "ablations" / "test_reads.log").read_text()
    assert "read_test_flag=False" in log and "no_u" not in log                                # labels are logged, not variant keys
    assert not result["mcnemar"].empty
