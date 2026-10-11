"""
Tests for the RF feature extractor (src/features/model/rf_extractor.py): the grouped split
never leaks a domain, page masking only touches webpage features, and extract_p_u returns
-1 for messages without a link.

Run with:  pytest tests/test_rf_extractor.py
"""

import numpy as np
import pandas as pd

from src.features.model import rf_extractor as rx


def _toy_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, len(rx.VARIANTS["full"]))), columns=rx.VARIANTS["full"])
    df[rx.PAGE_AVAILABLE] = 1
    df["label"] = rng.integers(0, 2, n)
    df["domain"] = [f"site{i % 60}.com" for i in range(n)]
    return df


def test_registrable_domain():
    assert rx.registrable_domain("http://a.b.example.com/x") == "example.com"
    assert rx.registrable_domain("https://news.bbc.co.uk/") == "bbc.co.uk"
    assert rx.registrable_domain("www.gcash.com.ph/login") == "gcash.com.ph"


def test_grouped_split_has_no_shared_domains():
    df = _toy_frame()
    train, val, test = rx.grouped_split(df, seed=42)
    parts = [set(df.iloc[idx]["domain"]) for idx in (train, val, test)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert len(train) + len(val) + len(test) == len(df)


def test_mask_page_features_only_touches_page():
    df = _toy_frame()
    masked = rx.mask_page_features(df, 1.0, seed=0)
    assert (masked[rx.WEBPAGE_FEATURES] == -1).all().all()
    assert (masked[rx.PAGE_AVAILABLE] == 0).all()
    untouched = rx.URL_TEXT_FEATURES + [rx.PAGE_RANK] + rx.EXTERNAL_FEATURES
    pd.testing.assert_frame_equal(masked[untouched], df[untouched])


def test_extract_p_u_sentinel_and_ranges():
    df = _toy_frame()
    features = rx.VARIANTS["full"]
    rf = rx.fit_rf(df[features], df["label"].to_numpy(), seed=0, n_estimators=20)
    bundle = {"model": rf, "features": features}

    sms = df.head(10).copy()
    sms["has_url"] = [1, 0] * 5
    out = rx.extract_p_u(bundle, sms)

    no_link = sms["has_url"] == 0
    assert (out.loc[no_link, ["P", "u"]] == -1).all().all()
    assert out.loc[~no_link, "P"].between(0, 1).all()
    assert out.loc[~no_link, "u"].between(0, 0.5).all()
    # P is the forest's own probability.
    expected = rf.predict_proba(sms.loc[~no_link, features])[:, 1]
    np.testing.assert_allclose(out.loc[~no_link, "P"], expected, atol=1e-6)
