"""Tests for the configurations adopted 2026-11-25 after the validation-only tuning (docs/validation_only_improvement_plan.md, Phases
E-G): several Conv widths for the three neural models (conv_blocks.MultiWidthConv), layer norm before FiLM and CNN-only's URL branch
in the Proposed model, the tuned presets, and resumable cell-by-cell training (cell_runs.py).

Run with:  python -m pytest tests/test_tuned_models.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.model import conv_blocks
from src.features.model.concat_cnn import ConcatTextCNN
from src.features.model.film_cnn import FiLMTextCNN
from src.features.model.raw_cnn import BRANCH_WIDTH, N_FEATURES, RAW_WIDTH, RawFeatureCNN, scale_raw_link
from src.unified import cell_runs
from src.unified import concat_control as cc
from src.unified import film
from src.unified import raw_cnn_control as rc

VOCAB = 100


def embedding(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Embedding(VOCAB, 300, padding_idx=0)


def text(n=6, seed=1, length=50, words=12):
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(n, length, dtype=torch.long)
    ids[:, :words] = torch.randint(1, VOCAB, (n, words), generator=g)          # short messages padded at the end, as in the data
    return ids


def raw_link(n=6, seed=2, has_url=None):
    g = torch.Generator().manual_seed(seed)
    has_url = torch.ones(n) if has_url is None else torch.as_tensor(has_url, dtype=torch.float32)
    return torch.cat([torch.rand(n, N_FEATURES, generator=g) * 50, torch.zeros(n, 1), has_url.unsqueeze(1)], dim=1)


def film_link(n=6, seed=3, has_url=None):
    g = torch.Generator().manual_seed(seed)
    has_url = torch.ones(n) if has_url is None else torch.as_tensor(has_url, dtype=torch.float32)
    return torch.cat([torch.rand(n, 2, generator=g), torch.zeros(n, 1), has_url.unsqueeze(1)], dim=1)


def tuned_film(seed=0, branch=True, layer_norm=True):
    model = FiLMTextCNN(embedding(seed), num_filters=192, gamma_mode="free", text_dropout="input")
    stats = (torch.randn(N_FEATURES) * 0.1, torch.rand(N_FEATURES) + 0.5)
    return film.apply_architecture(model, [3, 4, 5], 64, layer_norm, 64 if branch else 0, *stats)


# ------------------------------------------------------------------ the multi-width Conv
def test_one_width_is_exactly_the_old_conv():
    old = torch.nn.Conv1d(300, 16, 3)
    multi = conv_blocks.MultiWidthConv(300, (3,), 16)
    multi.convs[0].load_state_dict(old.state_dict())
    x = torch.randn(4, 300, 50)
    assert torch.equal(multi(x), old(x))


def test_several_widths_stack_with_the_same_positions_and_right_padding_is_padding():
    multi = conv_blocks.MultiWidthConv(300, (2, 3, 4), 8)
    x = torch.randn(4, 300, 50)
    y = multi(x)
    assert tuple(y.shape) == (4, 24, 49)                                     # 50 - smallest + 1 positions, 3 x 8 filters
    assert torch.allclose(y[:, 8:16, :48], multi.convs[1](x))               # width 3 on the real positions is a plain Conv
    padded = torch.cat([x, torch.zeros(4, 300, 2)], dim=2)                   # width 4: two extra zero columns = padding tokens
    assert torch.allclose(y[:, 16:, :], multi.convs[2](padded))


def test_use_widths_refuses_a_wrong_filter_count_and_a_second_call():
    model = RawFeatureCNN(embedding(), num_filters=64)
    with pytest.raises(ValueError, match="num_filters"):
        model.use_widths([3, 4, 5], 64)
    model = RawFeatureCNN(embedding(), num_filters=192)
    model.use_widths([3, 4, 5], 64)
    with pytest.raises(ValueError, match="once"):
        model.use_widths([3, 4, 5], 64)
    assert conv_blocks.filter_widths(model).tolist() == [3] * 64 + [4] * 64 + [5] * 64


def test_the_tuned_baselines_have_the_expected_size():
    k = RawFeatureCNN(embedding(), num_filters=192)
    k.use_widths([3, 4, 5], 64)
    c = ConcatTextCNN(embedding(), num_filters=192, link_dim=4)
    c.use_widths([3, 4, 5], 64)
    own = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad and p is not m.embedding.weight)
    assert own(k) == rc.expected_trainable(widths=[3, 4, 5], per_width=64) == 250050
    assert own(c) == cc.expected_trainable(4, widths=[3, 4, 5], per_width=64) == 243330
    for m, link in ((k, raw_link()), (c, film_link())):
        assert tuple(m.eval()(text(), link).shape) == (6, 2)


# ------------------------------------------------------------------ the tuned Proposed model
def test_defaults_leave_the_film_model_unchanged():
    model = FiLMTextCNN(embedding())
    keys = set(model.state_dict())
    film.apply_architecture(model)                                            # nothing asked for: nothing changes
    assert set(model.state_dict()) == keys and isinstance(model.conv, torch.nn.Conv1d) and model.link_width() == 4
    assert film.architecture_of(model) == {"widths": None, "per_width": None, "layer_norm": False, "url_branch_units": 0}


def test_tuned_film_starts_as_plain_text_and_reads_the_43_column_link():
    model = tuned_film().eval()
    assert model.link_width() == 43
    link = torch.cat([film_link(), raw_link()], dim=1)
    x, average = model.conv_features(text())
    gamma, beta = model.film_parameters(link, average)                        # the generator reads only [P, u, page_missing, has_url]
    assert bool((gamma == 1).all() and (beta == 0).all())
    assert tuple(model(text(), link).shape) == (6, 2)
    with pytest.raises(ValueError, match="43"):
        model(text(), film_link())


def test_layer_norm_normalizes_each_position_across_the_filters():
    model = tuned_film(branch=False)
    with torch.no_grad():
        x, _ = model.conv_features(text(words=12))
    assert x.shape[1] == 192
    assert float(x.mean(dim=1).abs().max()) < 1e-4
    words = x[:, :, :10]                    # windows over real words: spread 1 (pure padding windows are near-constant, so the
    assert torch.allclose(words.std(dim=1, unbiased=False), torch.ones_like(words[:, 0]), atol=1e-2)   # norm's eps keeps them < 1)
    assert bool((x.std(dim=1, unbiased=False) <= 1 + 1e-4).all())


def test_the_url_branch_reads_exactly_cnn_only_numbers():
    model = tuned_film()
    raw = raw_link(has_url=[1, 1, 0, 1, 0, 1])
    cnn_only = RawFeatureCNN(embedding(), model.feature_mean, model.feature_std)
    assert torch.equal(scale_raw_link(raw, model.feature_mean, model.feature_std), cnn_only.scaled_link(raw))
    assert scale_raw_link(raw, model.feature_mean, model.feature_std).shape[1] == BRANCH_WIDTH
    assert bool((scale_raw_link(raw, model.feature_mean, model.feature_std)[[2, 4]] == 0).all())     # no link: all zeros


def test_with_film_at_zero_the_branch_model_is_cnn_only_network():
    """Without layer norm and with the generator at zero, FiLM + URL branch computes exactly CNN-only's network (same weights)."""
    model = tuned_film(branch=True, layer_norm=False).eval()
    k = RawFeatureCNN(embedding(), model.feature_mean, model.feature_std, num_filters=192)
    k.use_widths([3, 4, 5], 64)
    k.load_state_dict({n: t for n, t in model.state_dict().items() if n in k.state_dict()})
    raw = raw_link()
    assert torch.allclose(model(text(), torch.cat([film_link(), raw], dim=1)), k.eval()(text(), raw), atol=1e-6)


def test_the_link_changes_the_verdict_through_film_and_the_branch():
    model = tuned_film()
    torch.nn.init.normal_(model.generator_layers()[-1].weight, std=0.1)
    model.eval()
    link = torch.cat([film_link(), raw_link()], dim=1)
    other_rf = link.clone()
    other_rf[:, :2] = 1 - other_rf[:, :2]                                   # only P and u change: FiLM must react
    other_raw = link.clone()
    other_raw[:, 4:4 + N_FEATURES] *= 3                                     # only the raw features change: the branch must react
    base = model(text(), link)
    assert not torch.allclose(base, model(text(), other_rf)) and not torch.allclose(base, model(text(), other_raw))


def test_neutralising_column_3_switches_film_off_and_keeps_the_branch():
    """Step 6's 'FiLM neutralised': has_url of FiLM (column 3) = 0 gives gamma = 1, beta = 0; the URL branch keeps its columns."""
    model = tuned_film()
    torch.nn.init.normal_(model.generator_layers()[-1].weight, std=0.1)
    link = torch.cat([film_link(), raw_link()], dim=1)
    off = link.clone()
    off[:, 3] = 0
    gamma, beta = model.film_parameters(off, model.conv_features(text())[1])
    assert bool((gamma == 1).all() and (beta == 0).all())
    assert torch.equal(off[:, 4:], link[:, 4:])


def test_parameter_groups_cover_every_trainable_parameter_once():
    model = tuned_film()
    ids = [id(p) for g in model.parameter_groups(1e-4) for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters() if p.requires_grad) and len(ids) == len(set(ids))


def test_rebuilding_from_the_recorded_architecture_gives_the_same_model():
    model = tuned_film()
    torch.nn.init.normal_(model.generator_layers()[-1].weight, std=0.1)
    again = FiLMTextCNN(embedding(5), num_filters=192, gamma_mode="free", text_dropout="input")
    film.apply_architecture(again, **film.architecture_of(model))             # placeholder mean / std, restored with the weights
    again.load_state_dict(model.state_dict())
    link = torch.cat([film_link(), raw_link()], dim=1)
    assert torch.equal(model.eval()(text(), link), again.eval()(text(), link))


def test_explain_masks_each_filter_window_to_its_own_width():
    model = tuned_film().eval()
    e = model.explain(text(), torch.cat([film_link(), raw_link()], dim=1))
    assert tuple(e.window_ids.shape) == (6, 192, 5)
    assert bool((e.window_ids[:, :64, 3:] == 0).all())                       # width-3 filters: the 4th and 5th slots are padding
    one = FiLMTextCNN(embedding()).eval()                                    # the one-width model keeps its (batch, filters, 3) windows
    assert tuple(one.explain(text(), film_link()).window_ids.shape) == (6, 64, 3)


def test_order_and_combinations_are_refused():
    model = FiLMTextCNN(embedding(), num_filters=192)
    model.use_layer_norm()
    with pytest.raises(ValueError, match="before"):
        model.use_widths([3, 4, 5], 64)
    with pytest.raises(ValueError, match="mask_padding"):
        FiLMTextCNN(embedding(), num_filters=192, mask_padding=True).use_widths([3, 4, 5], 64)
    with pytest.raises(ValueError, match="head_link"):
        FiLMTextCNN(embedding(), head_link=True).add_url_branch(torch.zeros(N_FEATURES), torch.ones(N_FEATURES))


def test_presets_are_the_validated_configurations():
    assert film.TUNED == {"architecture": {"widths": [3, 4, 5], "per_width": 64, "layer_norm": True, "url_branch_units": 64},
                          "options": {"gamma_mode": "free"},
                          "recipe": {"lr": 3e-3, "patience": 8, "p_page_hidden": 0.10, "p_text_hidden": 0.10}}
    assert rc.TUNED == {"architecture": {"widths": [3, 4, 5], "per_width": 64}, "recipe": {"lr": 3e-3, "patience": 8, "p_page_hidden": 0.10}}
    assert cc.TUNED == {"architecture": {"widths": [3, 4, 5], "per_width": 64}, "recipe": {"patience": 10, "p_text_hidden": 0.20}}
    assert RAW_WIDTH == 39


# ------------------------------------------------------------------ resumable runs
def test_cells_are_saved_skipped_and_assembled_in_order(tmp_path: Path):
    sig = {"model": "x", "recipe": {"lr": 0.003}}
    for seed, fold in ((123, 1), (42, 0), (42, 1)):
        cell_runs.save_cell(tmp_path, seed, fold, {"train_summary": pd.DataFrame([{"seed": seed, "fold": fold}])}, sig)
    assert cell_runs.cell_done(tmp_path, 42, 0, sig) and not cell_runs.cell_done(tmp_path, 42, 0, {**sig, "recipe": {"lr": 0.001}})
    assert not cell_runs.cell_done(tmp_path, 123, 0, sig)
    tables, finished = cell_runs.assemble(tmp_path, [42, 123], [0, 1], ["train_summary"], sig)
    assert finished == 3
    assert list(zip(tables["train_summary"].seed, tables["train_summary"].fold)) == [(42, 0), (42, 1), (123, 1)]
    assert (tmp_path / "train_summary.csv").exists()


def test_a_budget_stops_before_a_cell_that_would_not_fit():
    assert cell_runs.Budget(None).fits()
    budget = cell_runs.Budget(0.01)                                          # 0.6 seconds
    assert not budget.fits(first_guess=10.0)
    budget.add(0.01)
    assert budget.fits()


# ------------------------------------------------------------------ the real dataset (skipped where its files are absent)
def real_data():
    try:
        from src.unified import text_cnn as tc
        return tc.dataset_path(False).exists() and film.link_rf_path(tc.dataset_path(False), 42, 0).exists()
    except Exception:
        return False


needs_data = pytest.mark.skipif(not real_data(), reason="the final dataset and its link lane are not on this machine")


@needs_data
def test_inputs_for_joins_the_usual_link_and_cnn_only_raw_link_row_by_row():
    model, w2i, seq_len = film.build_film(42, 0, False, None, film.TUNED["architecture"], text_dropout=film.TEXT_DROPOUT,
                                          **film.TUNED["options"])
    data = film.inputs_for(model, 42, 0, "val", w2i, seq_len)
    base = film.part_inputs(42, 0, "val", w2i, seq_len)
    raw = rc.raw_part_inputs(42, 0, "val", w2i, seq_len)
    assert data.link_full.shape[1] == 43 and data.link_blank.shape[1] == 43
    assert torch.equal(data.link_full[:, :4], base.link_full) and torch.equal(data.link_full[:, 4:], raw.link_full)
    assert torch.equal(data.link_blank[:, :4], base.link_blank) and torch.equal(data.link_blank[:, 4:], raw.link_blank)
    assert np.array_equal(data.message_id, base.message_id)
    plain, w2i_plain, _ = film.build_film(42, 0)
    assert film.inputs_for(plain, 42, 0, "val", w2i_plain, seq_len).link_full.shape[1] == 4


@needs_data
def test_the_app_path_scores_the_tuned_model_like_the_pipeline():
    from src.data.split_crowd import load_fold
    model, w2i, seq_len = film.build_film(42, 0, False, None, film.TUNED["architecture"], text_dropout=film.TEXT_DROPOUT,
                                          **film.TUNED["options"])
    torch.nn.init.normal_(model.generator_layers()[-1].weight, std=0.05)
    model.eval()
    rows = load_fold(42, 0)["val"]
    data = film.inputs_for(model, 42, 0, "val", w2i, seq_len)
    scored = film.predict_messages(model, film.load_link_rf(film.tc.dataset_path(False), 42, 0), w2i, seq_len, rows)
    with torch.no_grad():
        direct = torch.softmax(model(data.token_ids, data.link_full), dim=1)[:, 1].numpy()
    assert np.allclose(scored["p_smishing"].to_numpy(), direct, atol=1e-5)
