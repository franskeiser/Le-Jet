"""
Tests for Stage 1b, RDAP (WEB_INSPECTION_PLAN.md section 13.4b): bootstrap parsing, caching,
rate limiting, retries and failure handling, with a fake fetcher and a fake clock. No network.

Run with:  python -m pytest tests/test_rdap_stage.py
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lejet.features.external_features import features_from_record
from lejet.stages.rdap_stage import Bootstrap, RdapClient

IANA_SAMPLE = {
    "version": "1.0",
    "services": [
        [["com", "net"], ["https://rdap.verisign.com/com/v1/", "http://insecure.example/"]],
        [["ph"], ["https://rdap.dot.ph/"]],
        [["xyz"], ["https://rdap.centralnic.com/xyz"]],            # no trailing slash
        [["onlyhttp"], ["http://plain.example/"]],                  # not https: unusable
        ["broken"], 5, [[1, 2], "oops"],                            # junk entries are skipped
    ],
}
RECORD_200 = json.dumps({"events": [
    {"eventAction": "registration", "eventDate": "2015-03-01T00:00:00Z"},
    {"eventAction": "expiration", "eventDate": "2030-03-01T00:00:00Z"}]}).encode()


class FakeTime:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    async def sleep(self, secs):
        self.sleeps.append(secs)
        self.now += secs


class FakeFetch:
    def __init__(self, responses=None):
        self.responses = list(responses or [(200, RECORD_200)])
        self.urls = []

    async def __call__(self, url):
        self.urls.append(url)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def make(fetch=None, interval=1.0):
    t = FakeTime()
    f = fetch or FakeFetch()
    client = RdapClient(Bootstrap(IANA_SAMPLE), f, min_interval=interval, retry_backoff=1.5,
                        clock=t.clock, sleep=t.sleep)
    return client, f, t


def run(coro):
    return asyncio.run(coro)


def test_bootstrap_maps_tlds_to_https_bases():
    b = Bootstrap(IANA_SAMPLE)
    assert b.base_url("com") == "https://rdap.verisign.com/com/v1/"
    assert b.base_url("NET") == "https://rdap.verisign.com/com/v1/"
    assert b.base_url("ph") == "https://rdap.dot.ph/"
    assert b.base_url("xyz") == "https://rdap.centralnic.com/xyz/"      # slash added
    assert b.base_url("onlyhttp") is None and b.base_url("nosuch") is None
    assert len(b) >= 4


def test_bootstrap_file_loading(tmp_path):
    p = tmp_path / "dns.json"
    p.write_text(json.dumps(IANA_SAMPLE))
    assert Bootstrap.from_file(str(p)).base_url("ph") == "https://rdap.dot.ph/"
    assert len(Bootstrap.from_file(str(tmp_path / "missing.json"))) == 0
    p.write_text("not json at all")
    assert len(Bootstrap.from_file(str(p))) == 0


def test_successful_lookup_builds_the_right_url_and_features():
    client, fetch, _ = make()
    out = run(client.lookup("example.com", "com"))
    assert fetch.urls == ["https://rdap.verisign.com/com/v1/domain/example.com"]
    assert out.record.ok and out.record.registered == 0
    f = features_from_record(out.record)
    assert f["whois_registered_domain"] == 0 and f["domain_age"] > 0


def test_404_is_a_completed_lookup_of_an_unregistered_domain():
    client, _, _ = make(FakeFetch([(404, b"{}")]))
    out = run(client.lookup("surely-not-registered-xyz.com", "com"))
    assert out.record.ok and out.record.registered == 1


def test_results_are_cached_per_registrable_domain():
    client, fetch, _ = make()
    run(client.lookup("example.com", "com"))
    again = run(client.lookup("EXAMPLE.com.", "com"))
    assert again.from_cache and len(fetch.urls) == 1


def test_failures_are_not_cached():
    client, fetch, _ = make(FakeFetch([(500, b""), (500, b""), (200, RECORD_200)]))
    first = run(client.lookup("example.com", "com"))
    assert not first.record.ok
    second = run(client.lookup("example.com", "com"))
    assert second.record.ok and not second.from_cache


def test_concurrent_lookups_of_one_domain_share_one_request():
    client, fetch, _ = make()

    async def go():
        return await asyncio.gather(*[client.lookup("example.com", "com") for _ in range(5)])
    results = run(go())
    assert all(r.record.ok for r in results)
    assert len(fetch.urls) == 1


def test_rate_limit_one_request_per_second_per_server():
    client, fetch, t = make(interval=1.0)
    run(client.lookup("one.com", "com"))
    run(client.lookup("two.com", "com"))
    run(client.lookup("three.com", "com"))
    assert len(fetch.urls) == 3
    assert len(t.sleeps) == 2 and all(s >= 0.99 for s in t.sleeps)


def test_different_servers_do_not_throttle_each_other():
    client, _, t = make()
    run(client.lookup("one.com", "com"))
    run(client.lookup("one.ph", "ph"))
    assert t.sleeps == []


@pytest.mark.parametrize("bad_status", [429, 500, 503])
def test_one_retry_with_backoff_on_429_and_5xx(bad_status):
    client, fetch, t = make(FakeFetch([(bad_status, b""), (200, RECORD_200)]))
    out = run(client.lookup("example.com", "com"))
    assert out.record.ok and len(fetch.urls) == 2
    assert 1.5 in t.sleeps


def test_gives_up_after_one_retry():
    client, fetch, _ = make(FakeFetch([(503, b"")]))
    out = run(client.lookup("example.com", "com"))
    assert not out.record.ok and len(fetch.urls) == 2


def test_no_retry_on_a_definite_answer():
    client, fetch, _ = make(FakeFetch([(404, b"")]))
    run(client.lookup("example.com", "com"))
    assert len(fetch.urls) == 1


def test_fetch_exceptions_are_failures_not_crashes():
    client, fetch, _ = make(FakeFetch([RuntimeError("network down")]))
    out = run(client.lookup("example.com", "com"))
    assert not out.record.ok
    assert len(fetch.urls) == 2                       # retried once, then gave up


def test_tld_without_an_rdap_server():
    client, fetch, _ = make()
    out = run(client.lookup("example.nosuchtld", "nosuchtld"))
    assert not out.record.ok and out.note == "no_rdap_server" and fetch.urls == []


@pytest.mark.parametrize("domain", ["", "nodots", "bad domain.com", "a..com", "-bad.com",
                                    "evil.com/../../x", "x.com?y=1", "http://x.com", "ex ample.com",
                                    "a" * 300 + ".com", "x.com#frag", "x.com\r\nHost: evil",
                                    "example.com\n", "example.com\r", "\nexample.com"])
def test_bad_domains_never_reach_the_network(domain):
    client, fetch, _ = make()
    out = run(client.lookup(domain, "com"))
    assert not out.record.ok and out.note == "bad_domain" and fetch.urls == []


def test_cache_file_roundtrip(tmp_path):
    client, _, _ = make()
    run(client.lookup("example.com", "com"))
    path = str(tmp_path / "rdap_cache.json")
    assert client.save_cache(path) == 1
    client2, fetch2, _ = make()
    assert client2.load_cache(path) == 1
    out = run(client2.lookup("example.com", "com"))
    assert out.from_cache and fetch2.urls == []


def test_cache_file_never_stores_failures_and_survives_junk(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "good.com": {"ok": True, "registered": 0, "reg_ts": 1, "exp_ts": 2},
        "failed.com": {"ok": False, "registered": -1},
        "../evil": {"ok": True, "registered": 0},
        "broken.com": {"ok": True},
        "junk.com": "text",
    }))
    client, _, _ = make()
    assert client.load_cache(str(path)) == 1
    (tmp_path / "d.json").write_text("not json")
    assert client.load_cache(str(tmp_path / "d.json")) == 0
    assert client.load_cache(str(tmp_path / "missing.json")) == 0
