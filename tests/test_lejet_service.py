"""Tests for the serving layer (src/serving/lejet_service.py and api.py).

The service needs the trained final model (models/fusion/rf_vec, models/rf_extractor); tests that need it are
skipped when those files are absent. The link store is built from a tiny fake table in a temp folder, so no lab
data and no network are used. The service must never open a link, whatever the message says.
"""
from __future__ import annotations

import json
import socket
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest

from src.config import ARTIFACTS_ROOT
from src.features.link_features import normalize_link
from src.serving import api
from src.serving import lejet_service as ls

MODELS_PRESENT = all((ARTIFACTS_ROOT / "models" / p).exists() for p in (
    "fusion/rf_vec/phase_c_seed42.pt", "fusion/rf_vec/gates_seed42.csv", "rf_extractor/full_seed42.joblib"))
needs_models = pytest.mark.skipif(not MODELS_PRESENT, reason="final model not trained on this machine")

LINK = "https://shop-verify.example-store.xyz/login"


def fake_store(tmp_path, fetched=True):
    """One inspected link: all 31 extractor inputs, a screenshot file, keyed like the real live set."""
    shots = tmp_path / "shots" / "b1"
    shots.mkdir(parents=True)
    (shots / "0123456789abcdef.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 20)
    row = {"link_id": "0123456789abcdef", "key": ls.link_key(normalize_link(LINK)), "page_fetched": int(fetched),
           "fetch_status": "live" if fetched else "dead", "fail_reason": "none" if fetched else "dns_nxdomain",
           "final_url": "hxxps://landing[.]example[.]top/" if fetched else "", "final_domain": "example.top",
           "n_redirects": 1 if fetched else -1, "visited_at": "2026-10-13T01:00:00Z", "batch_id": "b1"}
    for name in ls.PAGE_COLUMNS:
        row[name] = 3 if fetched else -1
    csv = tmp_path / "live_links.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)
    return ls.LinkStore(csv, [tmp_path / "shots"])


def test_percent_is_floored_never_rounded_up_and_capped():
    assert ls.percent_floor(0.91999) == 91.99
    assert ls.percent_floor(1.0) == 99.99 and ls.percent_floor(0.99999) == 99.99
    assert ls.percent_floor(-0.2) == 0.0


def test_risk_levels():
    assert (ls.risk_level(0.95), ls.risk_level(0.6), ls.risk_level(0.1)) == ("high", "medium", "low")


def test_url_reasons_use_the_prefixes_the_frontend_maps():
    reasons = ls.url_reasons({"https_presence": 0, "shortened_url_indicator": 1, "has_ip_address": 1,
                              "number_of_hyphens": 4, "subdomain_count": 3, "suspicious_keyword_count": 2,
                              "suspicious_url_token_count": 1, "special_char_count": 9}, "x")
    assert reasons == ["Link does not use HTTPS", "Link uses a shortened-URL service", "Link uses a raw IP address",
                       "Domain contains multiple hyphens", "Link has an unusually high number of subdomains",
                       "Link contains suspicious keywords", "Link path contains suspicious wording",
                       "Link contains an unusual number of special characters"]
    assert ls.url_reasons({"https_presence": 1}, "x") == []


def test_store_lookup_is_by_hash_and_shots_only_by_16_hex_id(tmp_path):
    store = fake_store(tmp_path)
    assert store.get(normalize_link(LINK))["link_id"] == "0123456789abcdef"
    assert store.get(normalize_link("https://other.example.org/")) is None
    assert store.shot_path("0123456789abcdef") is not None
    for bad in ("../../etc/passwd", "0123456789ABCDEF", "short", "", None, "0123456789abcdef.png"):
        assert store.shot_path(bad) is None


def test_missing_store_file_gives_an_empty_store(tmp_path):
    store = ls.LinkStore(tmp_path / "nope.csv", [tmp_path])
    assert len(store) == 0 and store.get("https://a.example.com/") is None


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    if not MODELS_PRESENT:
        pytest.skip("final model not trained on this machine")
    return ls.LejetService(seed=42, store=fake_store(tmp_path_factory.mktemp("store")))


@needs_models
class TestWithModel:

    def test_contract_fields_the_frontend_reads(self, service):
        out = service.predict(f"Your account is locked. Verify now: {LINK}")
        for key in ("label", "confidence_percent", "detected_url", "smishing_probability", "legitimate_probability",
                    "risk_level", "model_id", "source", "scanned_at", "urls", "safe_browsing", "text_reasons",
                    "text_reason_spans", "cnn_evidence", "url_evidence", "url_reasons", "cleaned_text",
                    "url_features", "global_key_indicators", "jain_route", "inference_ms", "scope", "inspection"):
            assert key in out, key
        assert out["label"] in ("smishing", "legitimate")
        assert abs(out["smishing_probability"] + out["legitimate_probability"] - 100) < 0.05
        assert out["model_id"] == "fusion_gated" and out["url_features"]["has_url"] == 1
        json.dumps(out)                                              # everything must be JSON-serialisable

    def test_timings_say_where_the_time_went_and_add_up(self, service):
        out = service.predict(f"Your account is locked. Verify now: {LINK}")
        timings = out["timings"]
        assert set(timings) == {"link_forest_ms", "text_prep_ms", "model_ms", "explanation_ms", "total_ms", "has_link"}
        assert timings["has_link"] is True and timings["model_ms"] == out["inference_ms"]
        assert all(timings[k] >= 0 for k in ("link_forest_ms", "text_prep_ms", "model_ms", "explanation_ms"))
        parts = timings["link_forest_ms"] + timings["text_prep_ms"] + timings["model_ms"] + timings["explanation_ms"]
        assert timings["total_ms"] >= parts - 0.05                     # the call also did the routing and the scope check
        assert service.predict("See you at lunch tomorrow")["timings"]["has_link"] is False
        assert service.predict("Verify now " + LINK, model_id="cnn_only_baseline")["timings"]["explanation_ms"] < timings["explanation_ms"]

    def test_a_stored_link_is_routed_to_the_full_extractor(self, service):
        out = service.predict(f"Verify now {LINK}")
        ins = out["inspection"]
        assert ins["state"] == "inspected" and ins["extractor"] == "full" and ins["has_screenshot"]
        assert ins["final_url"].startswith("hxxps://") and "://" not in ins["final_url"].replace("hxxps://", "")
        assert set(ins["features"]) == set(ls.PAGE_COLUMNS) and "full" in ins["extractors"]

    def test_an_unknown_link_uses_url_only_and_is_never_fetched(self, service, monkeypatch):
        def no_network(*a, **k):
            raise AssertionError("the service must never open a socket")
        monkeypatch.setattr(socket, "create_connection", no_network)
        monkeypatch.setattr(socket, "getaddrinfo", no_network)
        out = service.predict("Claim your prize at https://never-seen-before.example.org/win")
        assert out["inspection"]["state"] == "not_inspected" and out["inspection"]["extractor"] == "url_only"
        assert "features" not in out["inspection"]

    def test_no_link_is_judged_on_text(self, service):
        out = service.predict("Your OTP is 123456. Do not share it with anyone.")
        assert out["detected_url"] is None and out["inspection"]["state"] == "no_link"
        assert out["url_reasons"] == [] and out["urls"] == []

    def test_the_cnn_baseline_has_no_url_evidence_and_unknown_models_are_refused(self, service):
        out = service.predict(f"Verify now {LINK}", "cnn_only_baseline")
        assert out["model_id"] == "cnn_only_baseline" and out["url_evidence"] is None
        assert out["global_key_indicators"] == []
        with pytest.raises(ValueError, match="unknown_model"):
            service.predict("hello", "jain_replication")

    def test_the_same_message_gives_the_same_answer(self, service):
        a, b = service.predict(f"Verify now {LINK}"), service.predict(f"Verify now {LINK}")
        assert a["smishing_probability"] == b["smishing_probability"]

    def test_long_messages_are_truncated_not_rejected(self, service):
        out = service.predict("hello world " * 1000)
        assert out["label"] in ("smishing", "legitimate")

    def test_http_layer(self, service):
        from src.serving.live_inspect import LiveInspector
        server = ThreadingHTTPServer(("127.0.0.1", 0), api.make_handler(service, LiveInspector(service.store, enabled=False)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            def call(method, path, body=None, headers=None):
                conn = HTTPConnection("127.0.0.1", port, timeout=30)
                conn.request(method, path, body=json.dumps(body) if body is not None else None,
                             headers=headers or {"Content-Type": "application/json"})
                r = conn.getresponse()
                return r.status, dict(r.getheaders()), r.read()

            status, headers, body = call("POST", "/predict", {"message": f"Verify now {LINK}"},
                                         {"Content-Type": "application/json", "Origin": "http://localhost:5173"})
            assert status == 200 and headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
            assert json.loads(body)["inspection"]["state"] == "inspected"
            status, headers, _ = call("POST", "/predict", {"message": "hi"},
                                      {"Content-Type": "application/json", "Origin": "http://evil.example"})
            assert status == 200 and "Access-Control-Allow-Origin" not in headers      # other sites get no CORS
            assert call("POST", "/predict", {"message": "  "})[0] == 400
            assert call("POST", "/predict", {"nope": 1})[0] == 400
            status, _, body = call("POST", "/predict", {"message": "hi", "model_id": "jain_replication"})
            assert status == 400 and json.loads(body) == {"error": "unknown_model"}
            status, headers, body = call("GET", "/evidence/0123456789abcdef.png")
            assert status == 200 and headers["Content-Type"] == "image/png" and body.startswith(b"\x89PNG")
            assert call("GET", "/evidence/ffffffffffffffff.png")[0] == 404
            assert call("GET", "/evidence/../../secret.png")[0] == 404
            status, _, body = call("GET", "/health")
            assert status == 200 and json.loads(body)["live_lab"]["ready"] is False        # lab switched off here
            assert call("POST", "/predict", {"message": "x" * 30_000})[0] == 413
            assert call("POST", "/inspect", {"message": "no link in this text"})[0] == 400
            status, _, body = call("POST", "/inspect", {"message": "see https://brand-new.example.com/x"})
            assert status == 503 and "off" in json.loads(body)["error"]                     # nothing was opened
            status, _, body = call("POST", "/inspect", {"message": f"Verify now {LINK}"})   # in the store, but every link is inspected
            assert status == 503
            assert call("GET", "/inspect/0123456789ab")[0] == 404
            assert call("GET", "/inspect/not-a-job")[0] == 404
        finally:
            server.shutdown()


def test_a_second_api_cannot_share_the_port():
    """On Windows the standard server would let two copies share one port and split the requests between them."""
    first = api.Server(("127.0.0.1", 0), api.BaseHTTPRequestHandler)
    try:
        port = first.server_address[1]
        with pytest.raises(OSError):
            api.Server(("127.0.0.1", port), api.BaseHTTPRequestHandler)
    finally:
        first.server_close()
