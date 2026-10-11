"""
Tests for the RF feature extractor (src/features/model/rf_extractor.py): the grouped split
never leaks a domain, extract_p_u returns -1 for messages without a link, and route_p_u
sends each link to the right extractor.

Run with:  pytest tests/test_rf_extractor.py
"""

import numpy as np
import pandas as pd

from src.features.model import rf_extractor as rx


def _toy_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, len(rx.VARIANTS["full"]))), columns=rx.VARIANTS["full"])
    df["label"] = rng.integers(0, 2, n)
    df["domain"] = [f"site{i % 60}.com" for i in range(n)]
    return df


def _bundles(df: pd.DataFrame) -> dict:
    y = df["label"].to_numpy()
    return {
        variant: {"model": rx.fit_rf(df[features], y, seed=0, n_estimators=20), "features": features}
        for variant, features in rx.VARIANTS.items()
    }


def test_registrable_domain():
    assert rx.registrable_domain("http://a.b.example.com/x") == "example.com"
    assert rx.registrable_domain("https://news.bbc.co.uk/") == "bbc.co.uk"
    assert rx.registrable_domain("www.gcash.com.ph/login") == "gcash.com.ph"
    # Free-hosting subdomains are their own domain, not the platform's.
    assert rx.registrable_domain("https://evil.blogspot.com/") == "evil.blogspot.com"


def test_grouped_split_has_no_shared_domains():
    df = _toy_frame()
    train, val, test = rx.grouped_split(df, seed=42)
    parts = [set(df.iloc[idx]["domain"]) for idx in (train, val, test)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert len(train) + len(val) + len(test) == len(df)


def test_url_only_never_uses_lookups():
    # Old SMS links must not get any value looked up today (Phase 0 time shortcut).
    lookups = {rx.PAGE_RANK, *rx.WEBPAGE_FEATURES, *rx.EXTERNAL_FEATURES}
    assert not lookups & set(rx.VARIANTS["url_only"])
    assert len(rx.VARIANTS["url_only"]) == 14


def test_extract_p_u_sentinel_and_ranges():
    df = _toy_frame()
    bundle = _bundles(df)["full"]
    sms = df.head(10).copy()
    sms["has_url"] = [1, 0] * 5
    out = rx.extract_p_u(bundle, sms)

    no_link = sms["has_url"] == 0
    assert (out.loc[no_link, ["P", "u"]] == -1).all().all()

    # A link that exists but has no known features (<URL> placeholder) is not scored.
    redacted = sms.head(1).copy()
    redacted["has_url"] = 1
    redacted[bundle["features"]] = -1
    assert (rx.extract_p_u(bundle, redacted)[["P", "u"]] == -1).all().all()
    assert out.loc[~no_link, "P"].between(0, 1).all()
    assert out.loc[~no_link, "u"].between(0, 0.5).all()
    # P is the forest's own probability.
    expected = bundle["model"].predict_proba(sms.loc[~no_link, bundle["features"]])[:, 1]
    np.testing.assert_allclose(out.loc[~no_link, "P"], expected, atol=1e-6)


def test_route_p_u_picks_extractor_per_link():
    df = _toy_frame()
    bundles = _bundles(df)
    sms = df.head(6).copy()
    sms["has_url"] = [1, 1, 1, 0, 1, 1]
    sms[rx.PAGE_FETCHED] = [1, 0, 1, 1, 0, 1]
    # A failed page carries no page data at all; url_only must not need it.
    sms.loc[sms[rx.PAGE_FETCHED] == 0, rx.WEBPAGE_FEATURES] = np.nan

    out = rx.route_p_u(bundles["url_only"], bundles["full"], sms)

    assert out.index.equals(sms.index)
    assert out["extractor"].tolist() == ["full", "url_only", "full", "url_only", "url_only", "full"]
    assert (out.loc[sms["has_url"] == 0, ["P", "u"]] == -1).all().all()
    full_rows = out["extractor"] == "full"
    expected = rx.extract_p_u(bundles["full"], sms.loc[full_rows])
    np.testing.assert_allclose(out.loc[full_rows, "P"], expected["P"])
