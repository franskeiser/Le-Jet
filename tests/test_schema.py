"""
Tests for lejet.schema: the single definition of a row (plan 13.10, 14.0, 15.2).

The column lists must equal the ones the frozen RF extractors read (rf_extractor.py), because
a misspelled or reordered column "fails loudly or, worse, silently becomes missing".

Run with:  python -m pytest tests/test_schema.py
"""

from __future__ import annotations

import json
import math

import pytest

from lejet import safety, schema
from lejet.features import html_features


def test_column_lists_match_the_frozen_extractors():
    pytest.importorskip("sklearn")          # not installed inside the analysis VM
    from src.features.model import rf_extractor as rf

    assert schema.URL_TEXT_FEATURES == rf.URL_TEXT_FEATURES
    assert schema.WEBPAGE_FEATURES == rf.WEBPAGE_FEATURES
    assert schema.EXTERNAL_FEATURES == rf.EXTERNAL_FEATURES
    assert schema.PAGE_RANK == rf.PAGE_RANK
    # the 'full' variant is exactly: 14 URL-text + page_rank + 12 webpage + 4 external
    full = rf.VARIANTS["full"]
    assert full == schema.URL_TEXT_FEATURES + [schema.PAGE_RANK] + schema.WEBPAGE_FEATURES + schema.EXTERNAL_FEATURES
    assert len(full) == 31
    assert len(schema.VM_FEATURES) == 30
    assert "subdomain_count" in schema.URL_TEXT_FEATURES and "nb_subdomains" not in schema.FULL_FEATURES   # same for nb_subdomains


def test_html_feature_list_matches_schema():
    assert html_features.WEBPAGE_FEATURES == schema.WEBPAGE_FEATURES


def test_new_row_has_every_key_at_its_unknown_default():
    row = schema.new_row("a" * 16, "https://x.test/", "b001")
    for name in schema.VM_FEATURES:
        assert row[name] == -1, name
    assert set(schema.BOOKKEEPING_DEFAULTS) <= set(row)
    assert row["url_id"] == "a" * 16 and row["batch_id"] == "b001"
    assert row["has_url"] == 1 and row["page_fetched"] == 0
    assert row["fail_reason"] == "none" and row["fetch_ok"] == 0


def test_fail_reasons_cover_the_plan_taxonomy():
    plan = {"rejected_url", "port_blocked", "dns_nxdomain", "dns_timeout", "dns_servfail",
            "dns_no_answer", "connect_refused", "connect_timeout", "tls_error", "http_4xx",
            "http_5xx", "too_many_redirects", "redirect_to_blocked", "non_html", "too_large",
            "challenge_page", "parked_or_default_page", "render_timeout", "render_crash",
            "blocked_or_empty", "internal_error", "none"}
    assert plan <= set(schema.FAIL_REASONS)
    assert "read_timeout" in schema.FAIL_REASONS          # the one documented addition


def test_status_for():
    assert schema.status_for("none") == "done"
    assert schema.status_for("dns_nxdomain") == "failed:dns_nxdomain"


def test_url_id_validation():
    assert schema.is_valid_url_id("0123456789abcdef")
    for bad in ("", "xyz", "0123456789ABCDEF", "0123456789abcde", "0123456789abcdefg", None):
        assert not schema.is_valid_url_id(bad)


def test_clean_row_neutralizes_hostile_strings_and_numbers():
    row = schema.new_row("a" * 16, "x", "b")
    row["final_url"] = "http://e.test/\x1b[31m‮=cmd|'/C calc'!A0" + "A" * 5000
    row["hosting_platform"] = "pages.dev\x00\x07"
    row["domain_age"] = float("nan")
    row["nb_hyperlinks"] = float("inf")
    row["ratio_intHyperlinks"] = 0.123456789
    row["login_form"] = True
    row["script_count"] = "not a number"
    row["fail_reason"] = "totally_invented"
    clean = schema.clean_row(row)
    assert "\x1b" not in clean["final_url"] and "‮" not in clean["final_url"]
    assert len(clean["final_url"]) <= 1024
    assert "\x00" not in clean["hosting_platform"] and "\x07" not in clean["hosting_platform"]
    assert clean["domain_age"] == -1 and clean["nb_hyperlinks"] == -1
    assert clean["ratio_intHyperlinks"] == 0.123457
    assert clean["login_form"] == 1
    assert clean["script_count"] == -1
    assert clean["fail_reason"] == "internal_error"
    for v in clean.values():
        assert not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def test_export_record_shape_and_defanging():
    row = schema.new_row("a" * 16, "https://evil.example.com/login", "b001")
    row.update(fetch_ok=1, rdap_ok=1, html_available=1, page_fetched=1, html_bytes=1234,
               script_count=3, stage_ms_dns=5, stage_ms_rdap=6, stage_ms_http=7, stage_ms_html=8,
               duration_ms=26, url_length=30, nb_hyperlinks=4, domain_age=-1, dns_record=0,
               reg_domain="example.com", final_url="https://www.other.com/landing",
               final_reg_domain="other.com", hop_count=1, final_differs=1)
    rec = schema.export_record(row)
    assert rec["url"] == "hxxps://evil[.]example[.]com/login"
    assert rec["reg_domain"] == "example[.]com"
    assert rec["final_url"] == "hxxps://www[.]other[.]com/landing"
    assert rec["final_reg_domain"] == "other[.]com"
    assert rec["fetch_ok"] is True and rec["rdap_ok"] is True and rec["html_available"] is True
    assert rec["stage_ms"] == {"dns": 5, "rdap": 6, "http": 7, "html": 8}
    assert list(rec["features"]) == schema.VM_FEATURES            # exactly the 30 inputs, in order
    assert "page_rank" not in rec["features"]
    assert rec["features"]["nb_hyperlinks"] == 4
    assert set(schema.EXTRA_EXPORT_FIELDS) <= set(rec)
    # a record is plain JSON with no raw control characters
    text = json.dumps(rec, allow_nan=False)
    assert json.loads(text) == rec


def test_export_record_never_contains_a_clickable_url():
    row = schema.new_row("a" * 16, "http://phish.test/a?b=c", "b")
    row["final_url"] = "https://phish2.test/x"
    text = json.dumps(schema.export_record(row))
    assert "http://" not in text and "https://" not in text


# ---------------------------------------------------------------------------------------
# the exported public-set columns (73 of the set's 87)
# ---------------------------------------------------------------------------------------
def test_dataset_columns_are_the_public_sets_columns_in_order_and_features_full_is_a_subset():
    import csv
    from pathlib import Path
    assert len(schema.DATASET_COLUMNS) == 87 and len(set(schema.DATASET_COLUMNS)) == 87
    raw = Path(__file__).resolve().parents[1] / "data" / "raw" / "web-phishing" / "train-1.csv"
    if raw.exists():                                   # only on the host
        header = next(csv.reader(open(raw, encoding="utf-8")))
        assert header[1:-1] == schema.DATASET_COLUMNS  # between `url` and `status`
    assert len(schema.FULL_FEATURES) == 73 and len(schema.DROPPED_FROM_FULL) == 14
    assert schema.FULL_FEATURES == [n for n in schema.DATASET_COLUMNS if n not in schema.DROPPED_FROM_FULL]
    assert set(schema.DROPPED_FROM_FULL) == {
        "ratio_intRedirection", "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors", "google_index",
        "nb_or", "nb_hyphens", "dns_record", "web_traffic",
        "nb_subdomains", "ratio_extHyperlinks", "ratio_nullHyperlinks", "submit_email", "sfh"}


def test_dropping_a_column_from_the_table_does_not_touch_the_model_inputs():
    # nb_hyphens and dns_record are not exported in features_full, but they are still model inputs
    assert "number_of_hyphens" in schema.URL_TEXT_FEATURES and "dns_record" in schema.EXTERNAL_FEATURES
    assert "nb_hyphens" not in schema.FULL_FEATURES and "dns_record" not in schema.FULL_FEATURES
    assert len(schema.VM_FEATURES) == 30
    assert "subdomain_count" in schema.URL_TEXT_FEATURES and "nb_subdomains" not in schema.FULL_FEATURES   # same for nb_subdomains


def test_every_full_column_has_exactly_one_source():
    full = set(schema.FULL_FEATURES)
    from_model, host, stored = set(schema.FULL_FROM_MODEL_INPUTS), set(schema.FULL_NOT_ON_VM), set(schema.HF_STORED)
    assert not (from_model & stored) and not (host & stored) and not (from_model & host)
    assert from_model | host | stored == full
    assert host == {"page_rank"}
    assert from_model <= set(schema.WEBPAGE_FEATURES) | set(schema.EXTERNAL_FEATURES)


def test_a_new_row_has_every_hf_column_unknown_and_the_export_has_all_73():
    row = schema.new_row("0" * 16, "http://a.example/", "b")
    assert all(row[f"hf_{n}"] == -1 for n in schema.HF_STORED)
    full = schema.export_record(row)["features_full"]
    assert list(full) == schema.FULL_FEATURES and set(full.values()) == {-1}
