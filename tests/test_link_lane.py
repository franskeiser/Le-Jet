"""
Tests for src/unified/link_lane.py (docs/c_prune_architecture.md Phase 3).

Run with:  python -m pytest tests/test_link_lane.py
"""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from src.features.model.rf_extractor import PARAM_GRID
from src.unified import link_lane as ll
from src.unified.prepare_web_37 import FEATURES_37, OUTSIDE, PAGE


def _train(n_domains=30, per_domain=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_domains):
        smishing = d % 2 == 1
        for _ in range(per_domain):
            # length_url separates the classes; the other features are noise.
            feats = {f: float(rng.normal()) for f in FEATURES_37}
            feats["length_url"] = 60.0 + rng.normal() if smishing else 20.0 + rng.normal()
            rows.append({"label": "smishing" if smishing else "legitimate", "link_domain": f"d{d}.xyz",
                         "template_id": f"t{len(rows)}", "page_fetched": 1, **feats})
    df = pd.DataFrame(rows)
    # Every 5th domain's page was not reached: its 13 page columns are -1.
    dead = df["link_domain"].isin([f"d{d}.xyz" for d in range(0, n_domains, 5)])
    df.loc[dead, PAGE] = -1
    df.loc[dead, "page_fetched"] = 0
    return df


def test_tune_returns_a_grid_candidate_and_every_score():
    params, grid = ll.tune(_train(), "domain", seed=42)
    assert set(params) == set(PARAM_GRID)
    assert len(grid) == np.prod([len(v) for v in PARAM_GRID.values()])
    assert grid["f1"].max() == 1.0          # length_url alone separates the classes
    best = grid.sort_values(["f1", "log_loss"], ascending=[False, True]).iloc[0]
    assert {k: str(v) for k, v in params.items()} == {k: best[k] for k in PARAM_GRID}


PARAMS = {"n_estimators": 50, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}


def test_score_layout_and_contributions_add_up():
    train = _train()
    rf = ll.fit_rf(train[FEATURES_37], ll.labels(train), 42, **PARAMS)
    out = ll.score(rf, train)
    assert list(out.columns) == ll.OUTPUTS and len(ll.OUTPUTS) == 39
    _, bias = ll.path_contributions(rf, train[FEATURES_37].to_numpy(np.float32))
    np.testing.assert_allclose(bias + out[ll.CONTRIBUTIONS].sum(axis=1), out["P"], atol=1e-5)


def test_forest_bias_matches_path_contributions():
    train = _train()
    rf = ll.fit_rf(train[FEATURES_37], ll.labels(train), 42, **PARAMS)
    _, bias = ll.path_contributions(rf, train[FEATURES_37].to_numpy(np.float32))
    assert ll.forest_bias(rf) == pytest.approx(bias)
    out = ll.score(rf, train)
    np.testing.assert_allclose(out["P"] - out[ll.CONTRIBUTIONS].sum(axis=1), ll.forest_bias(rf), atol=1e-5)


def test_oof_scores_every_row_once_in_order():
    train = _train().sample(frac=1, random_state=1)        # shuffled index must come back unchanged
    oof, blank = ll.oof_outputs(train, "domain", seed=42, params=PARAMS)
    assert oof.index.equals(train.index) and blank.index.equals(train.index)
    assert list(oof.columns) == ll.LANE and len(ll.LANE) == 40 and "has_url" not in ll.LANE
    assert oof["P"].between(0, 1).all() and (oof["u"] >= 0).all()
    assert (oof["page_missing"] == 1 - train["page_fetched"]).all() and (blank["page_missing"] == 1).all()
    # A row whose page was already missing is the same in both versions (same forest).
    dead = train["page_fetched"] == 0
    pd.testing.assert_frame_equal(oof[dead], blank[dead])


def test_oof_never_scores_a_row_with_a_forest_that_saw_its_domain():
    # Labels are random per domain and every domain has a unique feature fingerprint, so a forest
    # that saw a domain recognises it; an out-of-fold forest can only guess.
    rng = np.random.default_rng(3)
    rows = []
    for d in range(60):
        label = "smishing" if rng.random() < 0.5 else "legitimate"
        fingerprint = rng.normal(size=len(FEATURES_37)) * 10
        for _ in range(5):
            rows.append({"label": label, "link_domain": f"d{d}.xyz", "template_id": f"t{len(rows)}",
                         "page_fetched": 1, **dict(zip(FEATURES_37, fingerprint))})
    train = pd.DataFrame(rows)
    y = ll.labels(train)
    in_sample = ll.score(ll.fit_rf(train[FEATURES_37], y, 42, **PARAMS), train)["P"]
    oof = ll.oof_outputs(train, "domain", seed=42, params=PARAMS)[0]["P"]
    assert ((in_sample >= 0.5) == y).mean() > 0.95
    assert ((oof >= 0.5) == y).mean() < 0.75


def test_final_fits_on_train_and_scores_only_val_and_test():
    df = _train(n_domains=30)
    df["message_id"] = [f"pilot_{i:06d}" for i in range(len(df))]
    domain = df["link_domain"].str.extract(r"d(\d+)")[0].astype(int)
    parts = {"train": df[domain < 20], "val": df[(domain >= 20) & (domain < 25)], "test": df[domain >= 25]}
    rf, scored, blank = ll.final_outputs(parts, seed=42, params=PARAMS)
    assert rf.n_features_in_ == 37 and len(rf.estimators_) == PARAMS["n_estimators"]
    assert list(scored.columns) == list(blank.columns) == ["message_id", "part"] + ll.LANE
    assert scored["message_id"].equals(blank["message_id"]) and (blank["page_missing"] == 1).all()
    assert scored.groupby("part").size().to_dict() == {"test": len(parts["test"]), "val": len(parts["val"])}
    assert not set(scored["message_id"]) & set(parts["train"]["message_id"])
    # Same forest as scoring the rows directly.
    np.testing.assert_allclose(scored.loc[scored["part"] == "val", "P"].to_numpy(),
                               ll.score(rf, parts["val"])["P"].to_numpy())


def test_blank_page_hides_only_the_13_page_columns():
    rows = _train(n_domains=4)
    blank = ll.blank_page(rows)
    assert (blank[PAGE] == -1).all().all() and (blank["page_fetched"] == 0).all()
    pd.testing.assert_frame_equal(blank[OUTSIDE], rows[OUTSIDE])
    url = [f for f in FEATURES_37 if f not in PAGE + OUTSIDE]
    assert len(PAGE) == 13 and len(url) == 20
    pd.testing.assert_frame_equal(blank[url], rows[url])
    assert (rows["page_fetched"] == 1).any()          # the input is not changed in place


def test_tune_is_repeatable():
    a, grid_a = ll.tune(_train(), "domain", seed=7)
    b, grid_b = ll.tune(_train(), "domain", seed=7)
    assert a == b
    pd.testing.assert_frame_equal(grid_a, grid_b)
