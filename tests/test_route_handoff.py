"""
The hand-off from the inspector to the model (WEB_INSPECTION_PLAN.md section 16, Level 0:
"feed a fixture row ... to route_p_u(url_only, full, frame)").

The frozen extractors (models/rf_extractor/*.joblib) are git-ignored and are not on every
machine, so this test trains two TINY forests on the public set, with exactly the column lists
of the real extractors. That checks the CONTRACT (column names, order, routing, handling of
-1) without needing the real models. A missing or misspelled column must FAIL, never be
skipped.

Run with:  python -m pytest tests/test_route_handoff.py
"""

from __future__ import annotations

import asyncio
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from lejet import schema
from lejet.pipeline import Pipeline
from lejet.stages.dns_stage import RawAnswer
from lejet.stages.http_stage import Hop, HttpResult
from lejet.stages.rdap_stage import Bootstrap
from src.features.model import rf_extractor as rf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(ROOT, "data", "processed", "web_phishing_clean.csv")
pytestmark = pytest.mark.skipif(not os.path.exists(CSV), reason="public set not present")

import test_pipeline_local as tpl                       # reuse the fakes


@pytest.fixture(scope="module")
def bundles():
    df = pd.read_csv(CSV).sample(n=3000, random_state=1)
    y = df["status"].map(rf.LABEL_TO_IDX).to_numpy()
    out = {}
    for variant, cols in rf.VARIANTS.items():
        model = rf.fit_rf(df[cols], y, 1, n_estimators=25, max_depth=8)
        out[variant] = {"model": model, "features": cols, "variant": variant, "seed": 1,
                        "label_to_idx": rf.LABEL_TO_IDX}
    return out


def inspect_rows():
    async def fetched(norm):
        return tpl.ok_result(norm.normalized_url)

    async def challenge(norm):
        r = tpl.ok_result(norm.normalized_url)
        r.fail_reason = "challenge_page"
        r.analysis = {"features": {}, "html_bytes": 500, "script_count": 1, "near_empty": 1, "is_html": True,
                      "challenge": True, "parked": False, "redirect_url": "", "errors": []}
        return r
    cases = [("https://shop.example.com/", fetched), ("https://walled.example.com/", challenge),
             ("https://gone.example.org/login", None), ("http://10.0.0.1/admin", None)]
    res = tpl.FakeResolver({**tpl.resolves("shop.example.com", "walled.example.com")})
    out = []
    for raw, fn in cases:
        async def go(raw=raw, fn=fn):
            async with Pipeline(tpl.settings(), resolver=res, rdap_fetch=tpl.FakeRdap(), bootstrap=tpl.BOOTSTRAP) as p:
                if fn is not None:
                    p.http = tpl.FakeHttp(fn)
                return await p.inspect(tpl.uid(raw), raw, "b1")
        out.append(asyncio.run(go()).row)
    return out


def to_frame(rows, page_rank=3):
    frame = pd.DataFrame([{**{k: r[k] for k in schema.VM_FEATURES}, "has_url": 1,
                           "page_fetched": r["page_fetched"], "page_rank": page_rank} for r in rows])
    return frame


def test_rows_route_to_the_right_extractor(bundles):
    rows = inspect_rows()
    assert [r["page_fetched"] for r in rows] == [1, 0, 0, 0]
    out = rf.route_p_u(bundles["url_only"], bundles["full"], to_frame(rows))
    assert list(out["extractor"]) == ["full", "url_only", "url_only", "url_only"]
    for p in out["P"]:
        assert 0.0 <= p <= 1.0
    for u in out["u"]:
        assert u >= 0.0


def test_pipeline_columns_satisfy_both_extractors(bundles):
    rows = inspect_rows()
    frame = to_frame(rows)
    for variant in ("url_only", "full"):
        missing = [c for c in bundles[variant]["features"] if c not in frame.columns]
        assert missing == [], (variant, missing)
    out = rf.extract_p_u(bundles["full"], frame.assign(page_fetched=1))        # even rows with -1 values
    assert (out["P"].between(0, 1)).all()


def test_a_challenge_page_never_reaches_the_full_extractor(bundles):
    rows = inspect_rows()
    challenge_row = rows[1]
    assert challenge_row["fail_reason"] == "challenge_page" and challenge_row["page_fetched"] == 0
    out = rf.route_p_u(bundles["url_only"], bundles["full"], to_frame([challenge_row]))
    assert out["extractor"].iloc[0] == "url_only"


def test_a_missing_or_misspelled_column_fails_loudly(bundles):
    frame = to_frame(inspect_rows())
    with pytest.raises(KeyError):
        rf.route_p_u(bundles["url_only"], bundles["full"], frame.rename(columns={"nb_hyperlinks": "nb_hyperlink"}))
    with pytest.raises(KeyError):
        rf.route_p_u(bundles["url_only"], bundles["full"], frame.drop(columns=["page_rank"]))
    with pytest.raises(KeyError):
        rf.route_p_u(bundles["url_only"], bundles["full"], frame.drop(columns=["page_fetched"]))


def test_dns_direction_is_the_datasets_not_the_intuitive_one():
    rows = inspect_rows()
    gone = rows[2]
    assert gone["fail_reason"] == "dns_nxdomain" and gone["dns_record"] == 1       # no record = abnormal = 1
    df = pd.read_csv(CSV)
    assert df["dns_record"].mean() < 0.05                                            # the public set: mostly 0
