"""Tests for the CNN-only baseline with the same inputs as E2 (text + the 37 raw URL / page features, no Link RF):
src/features/model/raw_cnn.py and src/unified/raw_cnn_control.py.

Run with:  python -m pytest tests/test_raw_cnn.py
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.features.model.raw_cnn import BRANCH_WIDTH, FLAGGED, LOG_FEATURES, N_FEATURES, RAW_WIDTH, RawFeatureCNN, log_columns
from src.unified import raw_cnn_control as rc
from src.unified import link_lane as ll
from src.unified.prepare_web_37 import FEATURES_37

FRESH = 100


def embedding(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Embedding(FRESH, 300, padding_idx=0)


def batch(n=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, FRESH, (n, 50), generator=g)
    link = torch.cat([torch.rand(n, N_FEATURES, generator=g) * 100, torch.zeros(n, 1), torch.ones(n, 1)], dim=1)
    return ids, link


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p is not model.embedding.weight)


# ------------------------------------------------------------------ the model
def test_the_model_has_68930_trainable_parameters_without_the_embedding():
    model = RawFeatureCNN(embedding())
    assert trainable(model) == 68930 == rc.EXPECTED_TRAINABLE == rc.expected_trainable()
    parts = {n: sum(p.numel() for p in layer.parameters()) for n, layer in
             (("conv", model.conv), ("link_branch", model.link_branch), ("dense", model.dense), ("output", model.output))}
    assert parts == {"conv": 57664, "link_branch": 2880, "dense": 8256, "output": 130}
    assert RAW_WIDTH == N_FEATURES + 2 == 39 and len(FEATURES_37) == N_FEATURES
    assert BRANCH_WIDTH == 37 + 1 + len(FLAGGED) + 1 == 44            # prepared 37, page_missing, 5 flags, has_url


def test_the_features_reach_the_verdict_and_there_is_no_forest_in_the_model():
    model = RawFeatureCNN(embedding()).eval()
    ids, link = batch()
    other = link.clone()
    other[:, :N_FEATURES] = link[:, :N_FEATURES].flip(1)
    with torch.no_grad():
        logits = model(ids, link)
        assert tuple(logits.shape) == (8, 2)
        assert not torch.equal(logits, model(ids, other))
    assert not any("forest" in n or "generator" in n for n, _ in model.named_modules())


def test_the_features_are_logged_standardised_with_the_given_mean_and_std_and_clipped():
    logged = [FEATURES_37.index(f) for f in LOG_FEATURES]
    plain = [i for i in range(N_FEATURES) if i not in logged]
    mean, std = torch.full((N_FEATURES,), 10.0), torch.full((N_FEATURES,), 2.0)
    model = RawFeatureCNN(embedding(), mean, std)
    link = torch.zeros(3, RAW_WIDTH)
    link[:, N_FEATURES + 1] = 1
    link[0, :N_FEATURES] = 12.0           # plain columns: one std above the mean
    link[1, :N_FEATURES] = 10_000.0       # an enormous value
    link[2, :N_FEATURES] = 10.0
    scaled = model.scaled_link(link)
    assert torch.allclose(scaled[0, plain], torch.ones(len(plain)))
    assert torch.allclose(scaled[0, logged], torch.full((len(logged),), (np.log1p(12.0) - 10.0) / 2.0))   # log(1 + x) first
    assert (scaled[1, plain] == 5.0).all()                            # clipped
    assert (scaled[2, plain] == 0).all()
    assert "feature_mean" in model.state_dict() and "feature_std" in model.state_dict()       # saved with the weights


def test_a_missing_cell_becomes_average_and_raises_its_flag():
    model = RawFeatureCNN(embedding(), torch.full((N_FEATURES,), 3.0), torch.full((N_FEATURES,), 2.0))
    link = torch.full((2, RAW_WIDTH), 7.0)
    link[:, N_FEATURES] = 0                                            # page loaded
    link[:, N_FEATURES + 1] = 1
    for f in FLAGGED:
        link[1, FEATURES_37.index(f)] = -1                             # missing WHOIS / rank / redirects on a loaded page
    out = model.scaled_link(link)
    flagged = [FEATURES_37.index(f) for f in FLAGGED]
    assert out.shape == (2, BRANCH_WIDTH)
    assert (out[1, flagged] == 0).all() and (out[0, flagged] != 0).all()      # missing -> 0 (the average), not a made-up number
    flags = out[:, N_FEATURES + 1:N_FEATURES + 1 + len(FLAGGED)]
    assert flags[0].sum() == 0 and (flags[1] == 1).all()
    assert out[:, N_FEATURES].tolist() == [0, 0] and out[:, -1].tolist() == [1, 1]   # page_missing, has_url passed on


def test_a_message_with_no_link_gets_an_all_zero_url_input():
    model = RawFeatureCNN(embedding(), torch.rand(N_FEATURES), torch.rand(N_FEATURES) + 1)
    link = torch.full((1, RAW_WIDTH), -1.0)
    link[0, N_FEATURES] = 1
    link[0, N_FEATURES + 1] = 0
    assert (model.scaled_link(link) == 0).all()


def test_a_message_with_no_link_ignores_its_feature_cells_even_nan():
    model = RawFeatureCNN(embedding(), torch.rand(N_FEATURES), torch.rand(N_FEATURES) + 1).eval()
    ids, link = batch()
    link[:, N_FEATURES + 1] = 0                                        # has_url = 0, page_missing = 0 or 1 as given
    a = link.clone()
    b = link.clone()
    a[:, :N_FEATURES] = float("nan")
    b[:, :N_FEATURES] = 7.0
    with torch.no_grad():
        pa, pb = model(ids, a), model(ids, b)
    assert torch.isfinite(pa).all() and torch.equal(pa, pb)


def test_a_bad_width_is_refused():
    model = RawFeatureCNN(embedding())
    ids, link = batch()
    with pytest.raises(ValueError, match="link must be"):
        model(ids, link[:, :38])


def test_the_parameter_groups_decay_only_dense_64_and_cover_every_parameter_once():
    model = RawFeatureCNN(embedding())
    groups = model.parameter_groups(1e-3)
    assert groups[0]["params"] == [model.dense.weight] and groups[0]["weight_decay"] == 1e-3
    listed = [id(p) for g in groups for p in g["params"]]
    assert len(listed) == len(set(listed)) == len([p for p in model.parameters() if p.requires_grad])


def test_text_keep_hides_the_text_numbers_but_not_the_link_branch():
    model = RawFeatureCNN(embedding()).eval()
    ids, link = batch(4)
    with torch.no_grad():
        assert (model.encode(ids, torch.zeros(4)) == 0).all()
        assert not torch.equal(model(ids, link, torch.zeros(4)), model(ids, link * 0.5, torch.zeros(4)))   # still reads the features


# ------------------------------------------------------------------ the inputs
def rows(n=6, has_url=None):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n, len(FEATURES_37))) * 10, columns=FEATURES_37)
    df["page_fetched"] = 1
    df["message_id"] = [f"m{i}" for i in range(n)]
    if has_url is not None:
        df["has_url"] = has_url
        df.loc[df["has_url"] == 0, FEATURES_37] = np.nan
        df.loc[df["has_url"] == 0, "page_fetched"] = 0
    return df


def test_raw_link_matrix_is_features_then_page_missing_then_has_url_and_zeros_a_no_link_row():
    df = rows(4, has_url=[1, 1, 0, 1])
    m = rc.raw_link_matrix(df)
    assert m.shape == (4, 39) and m.dtype == np.float32 and np.isfinite(m).all()
    assert np.allclose(m[[0, 1, 3], :N_FEATURES], df.loc[[0, 1, 3], FEATURES_37].to_numpy())
    assert (m[2, :N_FEATURES] == 0).all()
    assert list(m[:, N_FEATURES + 1]) == [1, 1, 0, 1] and list(m[:, N_FEATURES]) == [0, 0, 1, 0]
    assert rc.raw_link_matrix(rows(3))[:, N_FEATURES + 1].tolist() == [1, 1, 1]        # no has_url column: a link in every message


def test_the_hidden_page_version_blanks_the_13_page_columns_and_keeps_the_rest():
    df = rows(4)
    full, blank = rc.raw_link_matrix(df), rc.raw_link_matrix(ll.blank_page(df))
    page = [FEATURES_37.index(c) for c in ll.PAGE]
    assert len(page) == 13 and (blank[:, page] == -1).all() and (blank[:, N_FEATURES] == 1).all()
    rest = [i for i in range(N_FEATURES) if i not in page]
    assert np.array_equal(full[:, rest], blank[:, rest])                  # URL and outside columns stay, as for a real dead page


def test_a_missing_feature_in_a_message_with_a_link_is_refused():
    df = rows(3, has_url=[1, 1, 1])
    df.loc[1, FEATURES_37[0]] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        rc.raw_link_matrix(df)


def test_feature_stats_use_only_the_messages_with_a_link_and_a_constant_column_gets_std_1():
    df = rows(40, has_url=[1, 1, 1, 0] * 10)
    df.loc[df["has_url"] == 1, FEATURES_37[3]] = 5.0                       # constant over the linked messages
    mean, std = rc.feature_stats(df)
    linked = df[df["has_url"] == 1]
    prepared = pd.DataFrame(log_columns(linked[FEATURES_37].to_numpy(dtype=np.float64)), columns=FEATURES_37)
    assert np.allclose(mean, prepared.mean().to_numpy(), atol=1e-5) and np.isfinite(mean).all()
    assert std[3] == 1.0
    assert np.allclose(np.delete(std, 3), np.delete(prepared.std(ddof=0).to_numpy(), 3), atol=1e-4)


def test_feature_stats_leave_the_missing_cells_out():
    df = rows(20, has_url=[1] * 20)
    col = FEATURES_37.index("domain_age")
    real = df["domain_age"].copy()
    df.loc[:9, "domain_age"] = -1                                        # half missing
    mean, std = rc.feature_stats(df)
    kept = np.log1p(real[10:].to_numpy())                                # domain_age is logged
    assert mean[col] == pytest.approx(kept.mean(), abs=1e-5) and std[col] == pytest.approx(kept.std(), abs=1e-4)
