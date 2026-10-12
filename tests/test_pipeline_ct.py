"""
The certificate-age columns in the pipeline (plan section 13.4b, addition of 2026-10-12):

  ct_first_cert_days            days since the oldest public TLS certificate (crt.sh), -1 = unknown
  ct_ok                         the crt.sh lookup completed
  registration_unavailable_tld  1 = the TLD has no RDAP service, so a -1 age means "unknown"

The stage runs only where RDAP gave nothing, never for free-hosting hosts, and only when it is
switched on (`Settings.ct_enabled`, or a fetcher is injected). No network: fakes everywhere.

Run with:  python -m pytest tests/test_pipeline_ct.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from lejet import schema
from lejet.pipeline import Pipeline
from test_pipeline_local import BOOTSTRAP, FakeRdap, FakeResolver, settings, uid

FIRST_CERT = "2024-01-01T00:00:00"
CT_BODY = json.dumps([{"not_before": "2025-05-05T00:00:00"}, {"not_before": FIRST_CERT}]).encode()


class CountingCt:
    def __init__(self, status=200, payload=CT_BODY):
        self.status, self.payload, self.urls = status, payload, []

    async def __call__(self, url):
        self.urls.append(url)
        return self.status, self.payload


def expected_days():
    first = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - first).days


def inspect(raw, *, ct=None, rdap=None, **kw):
    ct_fetch = ct
    rdap = rdap or FakeRdap()
    kw.setdefault("ct_min_interval", 0.0)
    kw.setdefault("ct_backoff", 0.0)

    async def go():
        async with Pipeline(settings(**kw), resolver=FakeResolver(), rdap_fetch=rdap, bootstrap=BOOTSTRAP,
                            ct_fetch=ct_fetch) as p:
            return await p.inspect(uid(raw), raw, "b001")
    return asyncio.run(go()).row


NO_RDAP_HOST = "http://www.alpha.kz/login"          # .kz is not in the fake bootstrap: no RDAP server


def test_a_tld_without_rdap_gets_certificate_age_and_the_flag():
    ct = CountingCt()
    row = inspect(NO_RDAP_HOST, ct=ct)
    assert row["registration_unavailable_tld"] == 1
    assert row["rdap_ok"] == 0 and row["domain_age"] == -1 and row["whois_registered_domain"] == -1
    assert row["ct_ok"] == 1 and abs(row["ct_first_cert_days"] - expected_days()) <= 1
    assert ct.urls and ct.urls[0].startswith("https://crt.sh/?q=alpha.kz")


def test_certificate_age_never_leaks_into_the_rdap_features():
    row = inspect(NO_RDAP_HOST, ct=CountingCt())
    assert row["domain_age"] == -1 and row["domain_registration_length"] == -1 and row["days_to_expiry"] == -1


def test_a_domain_with_rdap_never_asks_crt_sh():
    ct = CountingCt()
    row = inspect("http://www.alpha.com/", ct=ct)
    assert ct.urls == []
    assert row["rdap_ok"] == 1 and row["domain_age"] > 0
    assert (row["ct_ok"], row["ct_first_cert_days"], row["registration_unavailable_tld"]) == (0, -1, 0)


def test_a_failing_rdap_server_triggers_crt_sh_but_is_not_flagged_unavailable():
    ct = CountingCt()
    row = inspect("http://www.alpha.com/", ct=ct, rdap=FakeRdap(default=(503, b"")))
    assert ct.urls
    assert row["rdap_ok"] == 0 and row["registration_unavailable_tld"] == 0
    assert row["ct_ok"] == 1 and row["ct_first_cert_days"] > 0


def test_ph_style_flag_is_set_even_when_the_stage_is_off():
    row = inspect(NO_RDAP_HOST)                       # no ct_fetch, ct_enabled False
    assert row["registration_unavailable_tld"] == 1
    assert (row["ct_ok"], row["ct_first_cert_days"]) == (0, -1)


def test_free_hosting_hosts_never_ask_crt_sh():
    ct = CountingCt()
    row = inspect("http://myshop.pages.dev/", ct=ct)
    assert ct.urls == [] and row["rdap_skipped"] == 1
    assert (row["ct_ok"], row["ct_first_cert_days"], row["registration_unavailable_tld"]) == (0, -1, 0)


def test_a_crt_sh_outage_leaves_unknown_values_and_the_rest_of_the_row_intact():
    row = inspect(NO_RDAP_HOST, ct=CountingCt(status=503, payload=b""))
    assert row["ct_ok"] == 0 and row["ct_first_cert_days"] == -1 and row["registration_unavailable_tld"] == 1
    assert row["fail_reason"] != "internal_error"
    assert set(schema.VM_FEATURES) <= set(row)


def test_a_domain_that_never_had_a_certificate_is_ok_with_minus_one():
    row = inspect(NO_RDAP_HOST, ct=CountingCt(payload=b"[]"))
    assert row["ct_ok"] == 1 and row["ct_first_cert_days"] == -1


def test_the_start_of_run_probe_is_off_by_default_and_asks_crt_sh_once_when_on():
    ct = CountingCt()
    inspect("http://www.alpha.com/", ct=ct)                      # probe off: nothing asked (RDAP answered)
    assert ct.urls == []
    ct2 = CountingCt(payload=b"[]")
    inspect("http://www.alpha.com/", ct=ct2, ct_probe=True)
    assert len(ct2.urls) == 1 and "probe.lejet.invalid" in ct2.urls[0]


def test_a_dead_crt_sh_found_by_the_probe_costs_each_row_nothing():
    ct = CountingCt(status=502, payload=b"")
    rows = [inspect(f"http://www.alpha{i}.kz/", ct=ct, ct_probe=True) for i in range(4)]
    assert all(r["ct_ok"] == 0 and r["ct_first_cert_days"] == -1 for r in rows)
    assert all(r["registration_unavailable_tld"] == 1 for r in rows)
    assert all("probe.lejet.invalid" in u for u in ct.urls)      # crt.sh was asked about nothing else


def test_the_new_columns_are_in_every_row_and_in_the_export():
    row = inspect(NO_RDAP_HOST, ct=CountingCt())
    rec = schema.export_record(row)
    for name in ("ct_first_cert_days", "ct_ok", "registration_unavailable_tld"):
        assert name in row and name in rec and name in schema.EXTRA_EXPORT_FIELDS
        assert name not in schema.VM_FEATURES            # they are NOT model inputs
    blank = schema.new_row("0" * 16, "http://x.test/", "b")
    assert (blank["ct_first_cert_days"], blank["ct_ok"], blank["registration_unavailable_tld"]) == (-1, 0, 0)

