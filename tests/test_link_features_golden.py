"""
Golden values for the 14 URL-text features (WEB_INSPECTION_PLAN.md section 16, Level 0a).

The full parity test (tests/test_link_features.py) needs pandas and the public dataset, which the
analysis VM does not have. This file carries the same safeguard in a form that runs ANYWHERE:
the features of 30 fixed links, computed on the host, must come out identical wherever the code
runs -- above all inside the analysis VM. `random_domain` depends on the `nostril` version, so a
different version there would silently change a model input; this test catches that.

Run with:  python -m pytest tests/test_link_features_golden.py
"""

from __future__ import annotations

import json
import os

import pytest

from src.features import link_features as lf

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "fixtures", "link_features_golden.json"), encoding="utf-8") as fh:
    GOLDEN = json.load(fh)
NAMES = GOLDEN["features"]


def test_golden_feature_names_match_the_code():
    assert NAMES == lf.LINK_TEXT_FEATURES


@pytest.mark.parametrize("row", GOLDEN["rows"], ids=[r[0][:50] for r in GOLDEN["rows"]])
def test_features_match_the_golden_values(row):
    url, expected = row[0], dict(zip(NAMES, row[1:]))
    assert lf.link_text_features(lf.normalize_link(url)) == expected


def test_nostril_is_the_pinned_original_not_the_pypi_namesake():
    # the original project flags random-looking host names; the PyPI "nostril" does not exist in
    # this form. 'asdkjqwe-zxcmnbv.com' must be random, 'example.com' must not.
    assert lf.random_domain("https://asdkjqwe-zxcmnbv.com/") == 1
    assert lf.random_domain("https://example.com/") == 0
