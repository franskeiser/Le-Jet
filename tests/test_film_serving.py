"""Tests for serving the FiLM model (src/serving/film_model.py, and the FiLM model inside lejet_service.py / api.py).

The feature tests need no trained model. The tests that load the FiLM model are skipped when its files are absent (models/film/<dataset>/ and the
fold's vocabulary); they use the dataset the pilot runs use (tc.dataset_path(pilot=True)). The link store is a tiny fake table in a
temp folder: no lab data and no network.

Run with:  python -m pytest tests/test_film_serving.py
"""
from __future__ import annotations

import json
import socket
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from src.features.link_features import normalize_link
from src.serving import api
from src.serving import lejet_service as ls
from src.serving.film_model import FilmModel, _has
from src.unified import text_cnn as tc
from src.unified.prepare_web_37 import FEATURES_37, OUTSIDE, PAGE, URL_FEATURES

LINK = "https://shop-verify.example-store.xyz/login"
NORMALISED = normalize_link(LINK)
DATASET = tc.dataset_path(True)
needs_film = pytest.mark.skipif(not FilmModel.available(DATASET), reason="the FiLM model is not trained for the pilot dataset on this machine")
needs_old_models = pytest.mark.skipif(not ls.ARTIFACTS_ROOT.joinpath("models/fusion/rf_vec/phase_c_seed42.pt").exists(),
                                      reason="earlier design not trained on this machine")


def lab_row(fetched=True, **overrides):
    """What the lab kept for one link: page_fetched, the 13 page columns, the 4 outside columns, the redirect count."""
    row = {"link_id": "0123456789abcdef", "key": ls.link_key(NORMALISED), "page_fetched": int(fetched), "fetch_status": "live",
           "fail_reason": "none", "final_url": "", "final_domain": "", "n_redirects": 2 if fetched else -1,
           "visited_at": "2026-10-21T01:00:00Z", "batch_id": "b1"}
    row.update({name: 7.0 if fetched else -1.0 for name in PAGE})
    row.update({"whois_registered_domain": 0, "domain_registration_length": 365, "domain_age": 900, "page_rank": 4})
    for name in ls.PAGE_COLUMNS:
        row.setdefault(name, 3 if fetched else -1)
    row.update(overrides)
    return row


def store_of(tmp_path, row):
    csv = tmp_path / "live_links.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)
    return ls.LinkStore(csv, [tmp_path / "shots"])


# ------------------------------------------------------------------ the 37 features (no model needed)
def test_a_link_the_lab_never_saw_has_only_its_spelling():
    frame, page_used = FilmModel.link_features(NORMALISED, None)
    assert list(frame.columns) == FEATURES_37 and len(frame) == 1 and page_used is False
    assert (frame.iloc[0][PAGE + OUTSIDE] == -1).all()                          # no page, no lookups
    assert frame.iloc[0]["nb_redirection"] == 0
    assert frame.iloc[0]["length_url"] == len(NORMALISED)
    assert (frame.iloc[0][URL_FEATURES] >= 0).all()


def test_a_complete_lab_result_fills_the_page_and_the_lookups():
    frame, page_used = FilmModel.link_features(NORMALISED, lab_row(safe_anchor=12.5))
    assert page_used is True
    assert (frame.iloc[0][[n for n in PAGE if n != "safe_anchor"]] == 7.0).all()
    assert frame.iloc[0]["safe_anchor"] == 12.5 and frame.iloc[0]["page_rank"] == 4 and frame.iloc[0]["domain_age"] == 900
    assert frame.iloc[0]["nb_redirection"] == 2


def test_a_lab_result_without_safe_anchor_keeps_the_lookups_but_not_the_page():
    """Lab results from before 2026-10-14 have no safe_anchor: the FiLM model treats the page as hidden (ll.blank_page keeps the 4 lookups)."""
    row = lab_row()
    del row["safe_anchor"]
    frame, page_used = FilmModel.link_features(NORMALISED, row)
    assert page_used is False
    assert (frame.iloc[0][PAGE] == -1).all() and frame.iloc[0]["nb_redirection"] == 0
    assert frame.iloc[0]["domain_age"] == 900 and frame.iloc[0]["page_rank"] == 4 and frame.iloc[0]["domain_registration_length"] == 365


def test_an_empty_safe_anchor_cell_counts_as_missing_not_as_a_number():
    """Rows written before the column existed are empty ("" or NaN) in the same csv as newer rows."""
    for empty in ("", float("nan"), None):
        _, page_used = FilmModel.link_features(NORMALISED, lab_row(safe_anchor=empty))
        assert page_used is False
    assert _has({"a": "3"}, "a") and not _has({"a": ""}, "a") and not _has({}, "a") and not _has(None, "a")


def test_a_page_that_did_not_load_is_hidden_but_its_domain_lookups_are_kept():
    frame, page_used = FilmModel.link_features(NORMALISED, lab_row(fetched=False, safe_anchor=-1))
    assert page_used is False
    assert (frame.iloc[0][PAGE] == -1).all() and frame.iloc[0]["domain_age"] == 900


# ------------------------------------------------------------------ the FiLM model inside the service
@pytest.fixture(scope="module")
def film():
    if not FilmModel.available(DATASET):
        pytest.skip("the FiLM model is not trained for the pilot dataset on this machine")
    return FilmModel(DATASET)


class _Slim(ls.LejetService):
    """The service without the earlier design's models: enough for the FiLM model's path (tests of the FiLM model alone should not need the other files)."""

    def __init__(self, film, store):
        self.seed, self.film, self.store = 42, film, store
        self.lock = threading.Lock()
        self.model_ids = (ls.FILM_ID,)
        self.columns, self.word_to_idx = [], film.word_to_idx


@needs_film
def test_the_film_model_is_the_default_model_and_answers_with_what_the_frontend_reads(film, tmp_path):
    service = _Slim(film, store_of(tmp_path, lab_row(safe_anchor=5.0)))
    out = service.predict(f"Your account is locked. Verify now: {LINK}")
    for key in ("label", "confidence_percent", "detected_url", "smishing_probability", "legitimate_probability", "risk_level",
                "model_id", "source", "scanned_at", "urls", "text_reasons", "text_reason_spans", "cnn_evidence", "url_evidence",
                "link_effect", "model_info", "url_reasons", "cleaned_text", "url_features", "inference_ms", "timings", "scope",
                "inspection"):
        assert key in out, key
    assert out["model_id"] == "film_cnn" and out["source"] == "film_model"
    assert out["model_info"]["kind"] == "film_scratch" and out["model_info"]["forest_trees"] == 100
    assert out["model_info"]["synthetic"] is ("synthetic" in DATASET.stem)
    assert abs(out["smishing_probability"] + out["legitimate_probability"] - 100) < 0.05
    assert out["timings"]["has_link"] is True and out["timings"]["model_ms"] == out["inference_ms"]
    json.dumps(out)


@needs_film
def test_the_page_the_lab_saw_reaches_the_model_and_a_message_with_no_link_bypasses_the_generator(film, tmp_path):
    seen = _Slim(film, store_of(tmp_path, lab_row(safe_anchor=5.0))).predict(f"Verify now {LINK}")
    assert seen["link_effect"]["page_missing"] == 0 and seen["link_effect"]["has_url"] == 1
    assert seen["inspection"]["state"] == "inspected" and "Random Forest" in seen["inspection"]["note"]
    inputs = seen["link_inputs"]
    assert inputs["page_used"] is True and inputs["lookups_used"] is True and len(inputs["features"]) == 37
    assert [f["group"] for f in inputs["features"]].count("url") == 20 and [f["group"] for f in inputs["features"]].count("page") == 13
    by_name = {f["name"]: f["value"] for f in inputs["features"]}
    assert by_name["safe_anchor"] == 5.0 and by_name["domain_age"] == 900 and by_name["nb_redirection"] == 2
    unseen = _Slim(film, ls.LinkStore(tmp_path / "none.csv", [tmp_path])).predict(f"Verify now {LINK}")
    assert unseen["link_effect"]["page_missing"] == 1 and "Link RF" in unseen["inspection"]["note"]
    assert unseen["link_inputs"]["page_used"] is False and unseen["link_inputs"]["lookups_used"] is False
    assert all(f["value"] == -1 for f in unseen["link_inputs"]["features"] if f["group"] != "url")
    none = _Slim(film, ls.LinkStore(tmp_path / "none.csv", [tmp_path])).predict("Your OTP is 123456. Do not share it with anyone.")
    assert none["detected_url"] is None and none["inspection"]["state"] == "no_link" and none["timings"]["has_link"] is False
    assert none["link_inputs"] is None
    assert none["link_effect"]["has_url"] == 0 and none["link_effect"]["mean_abs_gamma_shift"] == 0.0
    assert none["link_effect"]["mean_abs_beta"] == 0.0 and none["link_effect"]["filters_raised"] == 0


@needs_film
def test_an_old_lab_result_without_safe_anchor_says_why_the_film_model_ignored_the_page(film, tmp_path):
    row = lab_row()
    del row["safe_anchor"]
    out = _Slim(film, store_of(tmp_path, row)).predict(f"Verify now {LINK}")
    assert out["inspection"]["state"] == "inspected" and out["link_effect"]["page_missing"] == 1
    assert "safe_anchor" in out["inspection"]["note"]


@needs_film
def test_the_service_never_opens_a_link_when_the_film_model_answers(film, tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("the service must never open a socket")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    out = _Slim(film, ls.LinkStore(tmp_path / "none.csv", [tmp_path])).predict("Claim your prize at https://never-seen.example.org/win")
    assert out["inspection"]["state"] == "not_inspected"


@needs_film
def test_unknown_models_are_refused_and_the_same_message_gives_the_same_answer(film, tmp_path):
    service = _Slim(film, ls.LinkStore(tmp_path / "none.csv", [tmp_path]))
    with pytest.raises(ValueError, match="unknown_model"):
        service.predict("hello", "jain_replication")
    a, b = service.predict(f"Verify now {LINK}"), service.predict(f"Verify now {LINK}")
    assert a["smishing_probability"] == b["smishing_probability"] and a["text_reasons"] == b["text_reasons"]


@needs_film
def test_the_explanation_names_phrases_of_the_message(film, tmp_path):
    service = _Slim(film, ls.LinkStore(tmp_path / "none.csv", [tmp_path]))
    out = service.predict(f"URGENT: your bank account is locked. Verify now to avoid suspension: {LINK}")
    assert out["cnn_evidence"]["n_filters"] == 64 and out["url_evidence"] is None
    assert all(isinstance(r, str) and r for r in out["text_reasons"]) and len(out["text_reasons"]) == len(out["text_reason_spans"])
    skipped = service.predict(f"Verify now {LINK}", explain=False)
    assert skipped["text_reasons"] == [] and skipped["timings"]["explanation_ms"] < out["timings"]["explanation_ms"] + 5


# ------------------------------------------------------------------ the served path equals the evaluation
@needs_film
def test_served_labels_equal_the_evaluations_on_the_folds_test_messages(film):
    """The same check as scripts/check_film_serving.py, on the first 150 test messages: one message at a time through FilmModel against
    film.predict_messages."""
    from src.data.split_crowd import load_fold
    from src.unified import film as evaluation
    rows = load_fold(42, 0, pilot=True)["test"].iloc[:150].reset_index(drop=True)
    reference = evaluation.predict_messages(film.net, film.rf, film.word_to_idx, film.seq_len, rows)
    has_url = rows["has_url"].astype(int).to_numpy() if "has_url" in rows.columns else np.ones(len(rows), dtype=int)
    served = []
    for i in range(len(rows)):
        p = u = 0.0
        if has_url[i]:
            p, u = film.link_numbers(rows.iloc[[i]][FEATURES_37])
        link = film.link_tensor(p, u, 1 - int(rows["page_fetched"].iloc[i]), has_url[i])
        served.append(film.p_smishing(film.token_ids(str(rows["sms_text"].iloc[i])), link))
    served = np.array(served)
    assert ((served >= 0.5).astype(int) == reference["pred"].to_numpy()).all()
    assert np.abs(served - reference["p_smishing"].to_numpy()).max() < 1e-5


# ------------------------------------------------------------------ the HTTP layer
@needs_film
def test_health_lists_the_film_model_first_and_a_request_without_a_model_gets_it(film, tmp_path):
    from src.serving.live_inspect import LiveInspector
    service = _Slim(film, ls.LinkStore(tmp_path / "none.csv", [tmp_path]))
    server = ThreadingHTTPServer(("127.0.0.1", 0), api.make_handler(service, LiveInspector(service.store, enabled=False)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        def call(method, path, body=None):
            conn = HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request(method, path, body=json.dumps(body) if body is not None else None, headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            return r.status, json.loads(r.read())

        status, health = call("GET", "/health")
        assert status == 200 and health["models"][0] == "film_cnn" and health["film"]["kind"] == "film_scratch"
        status, answer = call("POST", "/predict", {"message": f"Verify now {LINK}"})
        assert status == 200 and answer["model_id"] == "film_cnn"
        status, answer = call("POST", "/predict", {"message": "hi", "model_id": "fusion_gated"})
        assert status == 400 and answer == {"error": "unknown_model"}                     # _Slim serves the FiLM model only
    finally:
        server.shutdown()


@needs_film
@needs_old_models
def test_the_earlier_models_are_still_served_next_to_the_film_model(film, tmp_path):
    service = ls.LejetService(seed=42, store=ls.LinkStore(tmp_path / "none.csv", [tmp_path]), film=film)
    assert service.model_ids == ("film_cnn", "fusion_gated", "cnn_only_baseline")
    assert service.predict(f"Verify now {LINK}")["model_id"] == "film_cnn"
    old = service.predict(f"Verify now {LINK}", "fusion_gated")
    assert old["model_id"] == "fusion_gated" and old["source"] == "fusion_model" and old["model_info"] is None
    assert old["inspection"]["extractor"] == "url_only" and "url_only extractor" in old["inspection"]["note"]
