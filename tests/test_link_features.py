"""
Tests for src/features/link_features.py: link finding in SMS text, URL normalization, and
the parity test -- the 14 features recomputed from the public web set's `url` column must
match the values the RF extractor was trained on (plan section 16, Level 0a).

Run with:  pytest tests/test_link_features.py
"""

from pathlib import Path

import pandas as pd
import pytest

from src.features import link_features as lf
from src.features.model.rf_extractor import URL_TEXT_FEATURES

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "processed" / "web_phishing_clean.csv"

# Minimum share of the 11,430 rows each recomputed feature must match.
PARITY = {name: 0.999 for name in lf.LINK_TEXT_FEATURES}
PARITY["random_domain"] = 0.90  # original input unknown; best reproduction is ~92%


def test_feature_list_matches_extractor():
    assert lf.LINK_TEXT_FEATURES == URL_TEXT_FEATURES


@pytest.mark.parametrize("text, expected", [
    ("Verify now at gcash-verify.xyz/login.", ["gcash-verify.xyz"]),
    ("Visit https://www.paymaya.com/deals to learn more", ["paymaya.com"]),
    ("hxxps://bdo-secure[.]top/a", ["bdo-secure.top"]),
    ("email us at help@gcash.com", []),        # email address, not a link
    ("see you tomorrow.call me", []),          # prose joined by a period
    ("ok.so what now", []),
    ("k.then", []),                            # not a real domain ending
])
def test_extract_links(text, expected):
    assert [link["domain"] for link in lf.extract_links(text)] == expected


@pytest.mark.parametrize("raw, expected", [
    ("gcash-verify.xyz/login.", "http://gcash-verify.xyz/login"),
    ("https://bit.ly/3abc", "https://bit.ly/3abc"),
    ("hxxps://bdo-secure[.]top/a", "https://bdo-secure.top/a"),
    ("www.usdtwtrc.com", "http://www.usdtwtrc.com"),
])
def test_normalize_link(raw, expected):
    assert lf.normalize_link(raw) == expected


def test_features_of_one_link():
    f = lf.link_text_features("https://login.gcash-verify.xyz/account/verify")
    assert f["https_presence"] == 1
    assert f["number_of_hyphens"] == 1
    assert f["length_hostname"] == len("login.gcash-verify.xyz")
    assert f["path_length"] == len("/account/verify")
    assert f["suspicious_keyword_count"] >= 2  # login, account, verify
    assert lf.no_link_features() == {name: -1 for name in lf.LINK_TEXT_FEATURES}


@pytest.mark.skipif(not CLEAN_CSV.exists(), reason="public web set not present")
def test_parity_with_public_web_set():
    df = pd.read_csv(CLEAN_CSV)
    recomputed = pd.DataFrame([lf.link_text_features(u) for u in df["url"]])
    for name, minimum in PARITY.items():
        share = (recomputed[name] == df[name]).mean()
        assert share >= minimum, f"{name}: {share:.4f} of rows match (need {minimum})"
