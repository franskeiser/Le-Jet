"""Tests for the synthetic-data defense demo (src/demo). Skipped until the demo is built:
    python -m src.demo.synthetic && python -m src.demo.train_demo
"""
import pandas as pd
import pytest

from src.demo.synthetic import DEMO_DATA, DEMO_MODEL_DIR, PAGE_COLUMNS, url_matches_text

built = DEMO_DATA.exists() and (DEMO_MODEL_DIR / "meta.json").exists()
needs_demo = pytest.mark.skipif(not built, reason="demo not built (python -m src.demo.synthetic; python -m src.demo.train_demo)")


def test_url_matches_text():
    assert url_matches_text("Hi, see http://example.com/a/b now", "http://example.com/a/b")
    assert not url_matches_text("Hi, no link here", "http://example.com/a/b")
    assert not url_matches_text("Hi, see http://other.com", "http://example.com")


@needs_demo
def test_synthetic_data_is_consistent():
    df = pd.read_csv(DEMO_DATA, keep_default_na=False)
    assert set(df["split"]) == {"train", "rf_fit", "test"}
    domains = df[df["has_url"] == 1].groupby("domain")["split"].nunique()
    assert (domains == 1).all(), "a domain is in two splits"
    unreachable = df[(df["has_url"] == 1) & (df["page_fetched"] == 0)]
    assert (unreachable[PAGE_COLUMNS] == -1).all().all()
    assert (df[df["has_url"] == 0][PAGE_COLUMNS] == -1).all().all()


@needs_demo
def test_analyze_covers_each_path():
    from src.demo.model import DemoModel

    model = DemoModel()
    no_link = model.analyze("Your OTP is 123456. Do not share it with anyone.")
    assert no_link["link"] is None and no_link["extractor_used"] is None and no_link["extractors"] == {}

    stored = next(iter(model.pages))
    seen = model.analyze(f"Please see the attached notice {stored}")
    assert seen["page_inspected"] and seen["extractor_used"] == "full" and set(seen["extractors"]) == {"url_only", "full"}

    off = model.analyze(f"Please see the attached notice {stored}", inspect=False)
    assert not off["page_inspected"] and off["extractor_used"] == "url_only"

    unknown = model.analyze("Please see the attached notice http://never-seen-before-domain.example.com/x")
    assert unknown["extractor_used"] == "url_only"
    for r in (no_link, seen, off, unknown):
        assert 0.0 <= r["prob_smishing"] <= 1.0 and r["verdict"] in ("smishing", "legitimate")
    # the forest's per-feature breakdown: empty with no link; with a page it can include page features
    assert no_link["rf_breakdown"] == []
    if model.recipe == "rf_vec":
        assert seen["rf_breakdown"] and all(set(b) == {"feature", "value", "points"} for b in seen["rf_breakdown"])
        assert not any(b["points"] == 0 for b in seen["rf_breakdown"])
        # with inspection off the url_only forest scores the link: it never reads page features
        assert not any(b["feature"] in PAGE_COLUMNS for b in off["rf_breakdown"])


@needs_demo
def test_server_handler_rejects_bad_input():
    import json
    import threading
    from http.server import HTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from src.demo.app import make_handler
    from src.demo.model import DemoModel

    server = HTTPServer(("127.0.0.1", 0), make_handler(DemoModel()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert urlopen(base + "/").status == 200
        ok = json.loads(urlopen(Request(base + "/api/analyze", json.dumps({"text": "hello there"}).encode(),
                                        {"Content-Type": "application/json"})).read())
        assert "verdict" in ok
        for payload in (b"garbage", json.dumps({"text": "  "}).encode()):
            with pytest.raises(HTTPError) as err:
                urlopen(Request(base + "/api/analyze", payload, {"Content-Type": "application/json"}))
            assert err.value.code == 400
    finally:
        server.shutdown()
