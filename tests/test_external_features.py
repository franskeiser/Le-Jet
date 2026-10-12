"""
Tests for the RDAP-derived features (WEB_INSPECTION_PLAN.md sections 13.4b, 14.0.4): recorded
RDAP JSON samples, including missing, 404 and hostile answers. No network.

Direction reminder (same as the public dataset): whois_registered_domain is 0 when a record
exists and 1 when there is NO registration (the abnormal case); -1 when the lookup failed.

Run with:  python -m pytest tests/test_external_features.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from lejet.features import external_features as ef

NOW = int(datetime(2026, 10, 11, tzinfo=timezone.utc).timestamp())


def ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def rdap(events) -> bytes:
    return json.dumps({"objectClassName": "domain", "ldhName": "example.test", "events": events}).encode()


SAMPLE_OK = rdap([
    {"eventAction": "registration", "eventDate": "2015-03-01T00:00:00Z"},
    {"eventAction": "last changed", "eventDate": "2026-01-05T10:00:00Z"},
    {"eventAction": "expiration", "eventDate": "2027-03-01T00:00:00Z"},
])


def feats(status, body, now=NOW):
    return ef.features_from_record(ef.record_from_response(status, body), now)


def test_normal_record():
    f = feats(200, SAMPLE_OK)
    assert f["rdap_ok"] == 1
    assert f["whois_registered_domain"] == 0                      # record exists = normal
    assert f["domain_age"] == (NOW - ts("2015-03-01T00:00:00Z")) // 86400
    assert f["domain_registration_length"] == (ts("2027-03-01T00:00:00Z") - NOW) // 86400
    assert f["days_to_expiry"] == f["domain_registration_length"]     # the same number


def test_registration_length_is_days_until_expiry_not_total_span():
    # the public set's median registration length (242) is far below its median age (3993)
    f = feats(200, SAMPLE_OK)
    assert f["domain_registration_length"] < f["domain_age"]


def test_404_means_not_registered():
    f = feats(404, b'{"errorCode": 404}')
    assert f["rdap_ok"] == 1
    assert f["whois_registered_domain"] == 1                      # NO registration = abnormal
    assert f["domain_age"] == -1 and f["domain_registration_length"] == -1


@pytest.mark.parametrize("status", [0, 400, 403, 429, 500, 502, 503])
def test_failed_lookups_are_minus_one_and_not_ok(status):
    f = feats(status, b"whatever")
    assert f["rdap_ok"] == 0
    assert f["whois_registered_domain"] == -1
    assert f["domain_age"] == -1 and f["domain_registration_length"] == -1


def test_record_without_events_is_registered_but_has_no_dates():
    f = feats(200, b'{"objectClassName": "domain"}')
    assert f["rdap_ok"] == 1 and f["whois_registered_domain"] == 0
    assert f["domain_age"] == -1 and f["domain_registration_length"] == -1


def test_expired_domain_never_stores_a_negative_duration():
    body = rdap([{"eventAction": "registration", "eventDate": "2010-01-01T00:00:00Z"},
                 {"eventAction": "expiration", "eventDate": "2025-01-01T00:00:00Z"}])
    f = feats(200, body)
    assert f["domain_age"] > 0
    assert f["domain_registration_length"] == -1 and f["days_to_expiry"] == -1


def test_registration_in_the_future_is_unknown_not_negative():
    body = rdap([{"eventAction": "registration", "eventDate": "2030-01-01T00:00:00Z"}])
    assert feats(200, body)["domain_age"] == -1


@pytest.mark.parametrize("body", [
    b"", b"not json", b"[1, 2, 3]", b'"a string"', b"null", b"\xff\xfe\x00bad",
    b'{"events": "nope"}', b'{"events": [1, 2, {"eventAction": 5}]}',
    b'{"events": [{"eventAction": "registration", "eventDate": "yesterday-ish"}]}',
])
def test_hostile_or_malformed_bodies_do_not_raise(body):
    f = feats(200, body)
    assert set(f) == {"whois_registered_domain", "domain_registration_length", "domain_age",
                      "days_to_expiry", "rdap_ok"}
    assert f["domain_age"] == -1 or f["rdap_ok"] == 1


def test_deeply_nested_json_does_not_crash():
    body = b"[" * 100000 + b"]" * 100000
    assert feats(200, body)["rdap_ok"] == 0


def test_huge_event_list_is_bounded():
    events = [{"eventAction": "noise", "eventDate": "2020-01-01T00:00:00Z"}] * 100000 + [
        {"eventAction": "registration", "eventDate": "2015-03-01T00:00:00Z"}]
    f = feats(200, rdap(events))
    assert f["rdap_ok"] == 1 and f["domain_age"] == -1      # the real event is beyond the cap


def test_date_formats():
    for stamp in ("2015-03-01T00:00:00Z", "2015-03-01T00:00:00+00:00", "2015-03-01T08:00:00+08:00",
                  "2015-03-01T00:00:00.123Z", "2015-03-01"):
        assert ef._parse_dt(stamp) is not None, stamp
    for bad in ("", None, 5, "x" * 100, "2015-13-45T00:00:00Z"):
        assert ef._parse_dt(bad) is None


def test_not_looked_up_defaults():
    assert ef.not_looked_up() == {"whois_registered_domain": -1, "domain_registration_length": -1,
                                  "domain_age": -1, "days_to_expiry": -1, "rdap_ok": 0}


def test_record_roundtrips_through_recomputation_with_a_later_now():
    rec = ef.record_from_response(200, SAMPLE_OK)
    a = ef.features_from_record(rec, NOW)
    b = ef.features_from_record(rec, NOW + 86400 * 100)
    assert b["domain_age"] == a["domain_age"] + 100
    assert b["domain_registration_length"] == a["domain_registration_length"] - 100
