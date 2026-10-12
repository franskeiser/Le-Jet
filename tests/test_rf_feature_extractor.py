import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features.model.rf_extractor import VARIANTS, p_and_u
from src.features.model.rf_feature_extractor import (
    CONTRIBUTION_COLUMNS, FULL_FEATURES, extract_vector, extractor_gates, path_contributions, route_vector,
)


def toy_bundle(variant: str, n: int = 600, seed: int = 0) -> tuple[dict, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = VARIANTS[variant]
    X = pd.DataFrame(rng.normal(size=(n, len(features))), columns=features)
    y = ((X[features[0]] + 0.5 * X[features[-1]] + rng.normal(scale=0.5, size=n)) > 0).astype(int).to_numpy()
    rf = RandomForestClassifier(n_estimators=30, min_samples_leaf=2, random_state=seed).fit(X, y)
    return {"model": rf, "features": features}, X, y


def test_contributions_add_up_to_p():
    bundle, X, _ = toy_bundle("url_only")
    contribution, bias = path_contributions(bundle["model"], X.to_numpy(np.float32))
    p, _ = p_and_u(bundle["model"], X)
    np.testing.assert_allclose(bias + contribution.sum(axis=1), p, atol=1e-5)


def test_extract_vector_layout_and_zero_for_missing_features():
    bundle, X, _ = toy_bundle("url_only")
    X["has_url"] = 1
    out = extract_vector(bundle, X)
    assert list(out.columns) == ["P", "u"] + CONTRIBUTION_COLUMNS
    page_slots = [f"c_{f}" for f in FULL_FEATURES if f not in bundle["features"]]
    assert (out[page_slots] == 0).all().all()  # url_only never saw them: "no evidence"
    assert out["P"].between(0, 1).all()


def test_no_link_and_unknown_rows_are_minus_one_and_zero():
    bundle, X, _ = toy_bundle("url_only")
    X["has_url"] = 1
    X.loc[0, "has_url"] = 0                      # no link
    X.loc[1, bundle["features"]] = -1            # link but nothing known
    out = extract_vector(bundle, X)
    for row in (0, 1):
        assert out.loc[row, ["P", "u"]].tolist() == [-1.0, -1.0]
        assert (out.loc[row, CONTRIBUTION_COLUMNS] == 0).all()


def test_route_vector_uses_full_only_where_page_fetched():
    url_only, _, _ = toy_bundle("url_only")
    full, X, _ = toy_bundle("full", seed=1)
    X["has_url"] = 1
    X["page_fetched"] = [1, 0] * (len(X) // 2)
    out = route_vector(url_only, full, X)
    page_slots = [f"c_{f}" for f in FULL_FEATURES if f not in url_only["features"]]
    assert (out.loc[X["page_fetched"] == 0, page_slots] == 0).all().all()
    assert (out.loc[X["page_fetched"] == 1, page_slots] != 0).any().any()


def test_extractor_gates_range():
    gates = extractor_gates(pd.Series(np.arange(len(FULL_FEATURES), dtype=float), index=FULL_FEATURES))
    assert gates.min() == pytest.approx(0.1) and gates.max() == pytest.approx(3.0) and len(gates) == 31
