"""
End-to-end tests of the inspection pipeline (WEB_INSPECTION_PLAN.md sections 13.1, 13.2b, 13.2c,
13.8, 13.9, 16 Level 0), with NO internet:

  * DNS and RDAP are fakes;
  * real fetching is done against the local web server on 127.0.0.1 (test mode);
  * the final-URL logic (redirects to other domains, free hosting, popular domains) uses a fake
    HTTP stage so that no test ever contacts a real host.

Run with:  python -m pytest tests/test_pipeline_local.py
"""

from __future__ import annotations

import asyncio
import json

import pytest

from fixtures.pages import PAGES
from lejet import schema, urlnorm
from lejet.pipeline import Pipeline
from lejet.settings import Settings
from lejet.stages.dns_stage import RawAnswer
from lejet.stages.http_stage import Hop, HttpResult
from lejet.stages.rdap_stage import Bootstrap
from lejet.storage import Storage

BOOTSTRAP = Bootstrap({"services": [[[t], [f"https://rdap.test/{t}/"]] for t in
                                    ("com", "org", "net", "xyz", "ph", "io", "dev", "app", "ly")]})
RECORD_OK = json.dumps({"events": [
    {"eventAction": "registration", "eventDate": "2015-03-01T00:00:00Z"},
    {"eventAction": "expiration", "eventDate": "2099-03-01T00:00:00Z"}]}).encode()
PUBLIC_IP = "93.184.216.34"
HTML_FEATS = {"nb_hyperlinks": 7, "ratio_intHyperlinks": 0.5, "nb_extCSS": 1, "login_form": 1,
              "external_favicon": 0, "links_in_tags": 50.0, "ratio_intMedia": 25.0, "ratio_extMedia": 75.0,
              "iframe": 0, "empty_title": 0, "domain_in_title": 1, "domain_with_copyright": 0}


class FakeResolver:
    def __init__(self, table=None, default="nxdomain"):
        self.table, self.default, self.calls = table or {}, default, []

    async def query(self, name, rdtype):
        self.calls.append((name, rdtype))
        return self.table.get((name, rdtype)) or RawAnswer(self.default)


def resolves(*hosts):
    return {(h, "A"): RawAnswer("ok", (PUBLIC_IP,), 300) for h in hosts}


class FakeRdap:
    def __init__(self, status_by_domain=None, default=(200, RECORD_OK)):
        self.by = status_by_domain or {}
        self.default = default
        self.domains = []

    async def __call__(self, url):
        domain = url.rsplit("/domain/", 1)[1]
        self.domains.append(domain)
        return self.by.get(domain, self.default)


class FakeHttp:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    async def fetch(self, norm):
        self.calls.append(norm.normalized_url)
        return await self.fn(norm)


def ok_result(final_url, redirects=0, **extra):
    r = HttpResult(final_url=final_url, final_status=200, chain_complete=True, scheme_used="https",
                   redirects=redirects, elapsed_ms=30, html_ms=10)
    r.hops = [Hop(i, 302 if i < redirects else 200, "h.test", "", 5) for i in range(redirects + 1)]
    r.analysis = {"features": dict(HTML_FEATS), "html_bytes": 4321, "script_count": 4, "near_empty": 0,
                  "is_html": True, "redirect_url": "", "challenge": False, "parked": False, "errors": []}
    for k, v in extra.items():
        setattr(r, k, v)
    return r


def settings(**kw):
    base = dict(allow_loopback=False, parse_workers=0, rate=1000.0, jitter=0.0, concurrency=20,
                rdap_min_interval=0.0, hop_deadline=8, url_deadline=15, connect_timeout=3)
    base.update(kw)
    return Settings(**base)


def uid(raw):
    return urlnorm.compute_url_id(raw)


def inspect_all(raws, resolver=None, rdap=None, http=None, popular=None, **kw):
    """Run Pipeline.inspect over `raws` and return the Inspection objects (and the fakes)."""
    resolver = resolver or FakeResolver()
    rdap = rdap or FakeRdap()

    async def go():
        async with Pipeline(settings(**kw), resolver=resolver, rdap_fetch=rdap, bootstrap=BOOTSTRAP,
                            popular_domains=popular) as p:
            if http is not None:
                p.http = http
            return [await p.inspect(uid(r), r, "b001") for r in raws]
    return asyncio.run(go()), resolver, rdap


def one(raw, **kw):
    results, resolver, rdap = inspect_all([raw], **kw)
    return results[0], resolver, rdap


def link_features(raw):
    from src.features import link_features as lf
    return lf.link_text_features(lf.normalize_link(raw))


# ---------------------------------------------------------------------------------------
# Stage 0: rejected URLs (no network is touched)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, reason", [
    ("http://10.0.0.1/admin", "rejected_url"),
    ("http://127.0.0.1/", "rejected_url"),
    ("http://169.254.169.254/latest/meta-data/", "rejected_url"),
    ("http://2130706433/", "rejected_url"),
    ("http://0x7f.0.0.1/", "rejected_url"),
    ("http://[::1]/", "rejected_url"),
    ("http://localhost/", "rejected_url"),
    ("http://user:pass@example.com/", "rejected_url"),
    ("ftp://example.com/file", "rejected_url"),
    ("javascript:alert(1)", "rejected_url"),
    ("http://exa mple.com/", "rejected_url"),
    ("http://example.com:8080/", "port_blocked"),
    ("https://example.com:22/", "port_blocked"),
])
def test_rejected_urls_get_a_row_with_the_text_features_and_nothing_else(raw, reason):
    insp, resolver, rdap = one(raw)
    row = insp.row
    assert row["fail_reason"] == reason
    assert row["fetch_ok"] == 0 and row["page_fetched"] == 0 and row["stage_reached"] == 0
    assert resolver.calls == [] and rdap.domains == []             # nothing left the process
    for name in schema.WEBPAGE_FEATURES + schema.EXTERNAL_FEATURES:
        assert row[name] == -1, name
    # the 14 URL-text features are pure string operations, so a rejected URL still has them
    expected = link_features(raw)
    for name in schema.URL_TEXT_FEATURES:
        assert row[name] == expected[name], name


def test_overlong_and_junk_urls_are_rejected_and_the_row_still_has_every_key():
    for raw in ("http://" + "a" * 3000 + ".com/", "\x00\x01\x02", "   ", "http://"):
        insp, _, _ = one(raw)
        assert insp.row["fail_reason"] == "rejected_url"
        assert set(schema.new_row("x" * 16, raw, "b").keys()) <= set(insp.row)
        for name in schema.VM_FEATURES:
            assert name in insp.row


def test_url_text_features_are_computed_on_the_original_link_not_the_normalized_one():
    raw = "gcash-verify.xyz/login"                                  # schemeless SMS link
    insp, _, _ = one(raw)                                           # (NXDOMAIN in the fake DNS)
    expected = link_features(raw)
    for name in schema.URL_TEXT_FEATURES:
        assert insp.row[name] == expected[name], name
    assert insp.row["https_presence"] == 0                          # HTTPS was not stated by the SMS


# ---------------------------------------------------------------------------------------
# Stage 1 + 1b: DNS and RDAP
# ---------------------------------------------------------------------------------------
def test_nxdomain_row_still_gets_dns_record_and_rdap():
    insp, _, rdap = one("https://gone.example.org/login", resolver=FakeResolver(),
                        rdap=FakeRdap({"example.org": (200, RECORD_OK)}))
    row = insp.row
    assert row["fail_reason"] == "dns_nxdomain" and row["dns_ok"] == 0
    assert row["stage_reached"] == 1 and row["chain_complete"] == 0
    assert row["dns_record"] == 1                                   # 1 = no DNS record (abnormal)
    assert row["rdap_ok"] == 1 and row["whois_registered_domain"] == 0
    assert row["domain_age"] > 0 and row["domain_registration_length"] > 0
    assert rdap.domains == ["example.org"]
    assert row["final_differs"] == 0
    assert row["orig_dns_record"] == row["dns_record"] and row["orig_domain_age"] == row["domain_age"]
    for name in schema.WEBPAGE_FEATURES:
        assert row[name] == -1


def test_an_unregistered_domain_gets_whois_registered_domain_one():
    insp, _, _ = one("http://never-registered.xyz/", rdap=FakeRdap({"never-registered.xyz": (404, b"")}))
    row = insp.row
    assert row["rdap_ok"] == 1 and row["whois_registered_domain"] == 1       # 1 = NOT registered
    assert row["domain_age"] == -1 and row["domain_registration_length"] == -1


def test_resolver_timeout_is_minus_one_and_a_dns_timeout_failure():
    insp, _, _ = one("http://slow.example.com/", resolver=FakeResolver(default="timeout"))
    assert insp.row["fail_reason"] == "dns_timeout" and insp.row["dns_record"] == -1


def test_rdap_failure_is_rdap_ok_zero_not_a_crash():
    insp, _, _ = one("http://down.example.com/", rdap=FakeRdap(default=(503, b"")))
    row = insp.row
    assert row["rdap_ok"] == 0 and row["whois_registered_domain"] == -1
    assert row["fail_reason"] == "dns_nxdomain"                      # the page failure is independent


def test_dns_rebinding_answer_is_refused_before_any_http():
    http = FakeHttp(lambda n: (_ for _ in ()).throw(AssertionError("must not fetch")))
    resolver = FakeResolver({("rebind.example.com", "A"): RawAnswer("ok", ("10.0.0.5",), 60)})
    insp, _, _ = one("http://rebind.example.com/", resolver=resolver, http=http)
    assert insp.row["fail_reason"] == "dns_no_answer" and http.calls == []


# ---------------------------------------------------------------------------------------
# free hosting and path-hosted platforms (plan 13.2c): RDAP is never asked
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, platform, path_hosted", [
    ("https://evil.pages.dev/login", "pages.dev", 0),
    ("https://x.blogspot.com/", "blogspot.com", 0),
    ("https://a.b.github.io/x", "github.io", 0),
    ("https://docs.google.com/forms/d/e/abc/viewform", "docs.google.com/forms", 1),
])
def test_flagged_hosts_never_trigger_an_rdap_request(raw, platform, path_hosted):
    insp, _, rdap = one(raw, resolver=FakeResolver())             # NXDOMAIN in the fake DNS
    row = insp.row
    assert rdap.domains == [], raw
    assert row["rdap_skipped"] == 1 and row["rdap_ok"] == 0
    assert row["whois_registered_domain"] == -1 and row["domain_age"] == -1
    assert row["domain_registration_length"] == -1
    assert row["hosting_platform"] == platform
    assert row["platform_path_hosting"] == path_hosted
    assert row["free_hosting_subdomain"] == (0 if path_hosted else 1)
    assert row["dns_record"] in (0, 1)                             # DNS is still computed


def test_ordinary_domain_is_not_flagged_and_rdap_runs():
    insp, _, rdap = one("https://login.gcash-verify.xyz/", rdap=FakeRdap())
    assert rdap.domains == ["gcash-verify.xyz"]
    assert insp.row["rdap_skipped"] == 0 and insp.row["free_hosting_subdomain"] == 0


# ---------------------------------------------------------------------------------------
# the FINAL url (plan 13.2b) with a fake HTTP stage
# ---------------------------------------------------------------------------------------
def test_no_redirect_means_one_rdap_lookup_and_orig_equals_final():
    async def fn(norm):
        return ok_result(norm.normalized_url)
    http = FakeHttp(fn)
    insp, _, rdap = one("https://shop.example.com/", resolver=FakeResolver(resolves("shop.example.com")),
                        http=http)
    row = insp.row
    assert row["fail_reason"] == "none" and row["fetch_ok"] == 1
    assert row["html_available"] == 1 and row["page_fetched"] == 1 and row["stage_reached"] == 5
    assert rdap.domains == ["example.com"]                          # one lookup only
    assert row["final_differs"] == 0 and row["hop_count"] == 0 and row["chain_complete"] == 1
    for name in schema.EXTERNAL_FEATURES:
        assert row[name] == row["orig_" + name]
    for name, value in HTML_FEATS.items():
        assert row[name] == value
    assert row["html_bytes"] == 4321 and row["script_count"] == 4 and row["near_empty"] == 0


def test_redirect_to_another_domain_describes_the_final_domain_and_keeps_the_original():
    async def fn(norm):
        return ok_result("https://www.real-bank.com/login", redirects=2)
    rdap = FakeRdap({"bit.ly": (200, json.dumps({"events": [
        {"eventAction": "registration", "eventDate": "2010-01-01T00:00:00Z"},
        {"eventAction": "expiration", "eventDate": "2099-01-01T00:00:00Z"}]}).encode()),
        "real-bank.com": (404, b"")})
    resolver = FakeResolver({**resolves("bit.ly", "www.real-bank.com")})
    insp, _, rdap = one("https://bit.ly/abc", resolver=resolver, rdap=rdap, http=FakeHttp(fn))
    row = insp.row
    assert sorted(rdap.domains) == ["bit.ly", "real-bank.com"]
    assert row["final_differs"] == 1 and row["hop_count"] == 2
    assert row["final_reg_domain"] == "real-bank.com" and row["final_url"] == "https://www.real-bank.com/login"
    # main columns describe the FINAL domain (not registered -> 1) ...
    assert row["whois_registered_domain"] == 1 and row["domain_age"] == -1
    # ... the orig_* columns keep the shortener's own values
    assert row["orig_whois_registered_domain"] == 0 and row["orig_domain_age"] > 3000
    assert row["reg_domain"] == "bit.ly"
    # the 14 URL-text features still describe the ORIGINAL link
    assert row["url_length"] == len("https://bit.ly/abc")
    assert row["shortened_url_indicator"] == 1


def test_redirect_to_a_free_hosting_site_skips_rdap_for_the_final_domain_only():
    async def fn(norm):
        return ok_result("https://evil.pages.dev/x", redirects=1)
    insp, _, rdap = one("https://bit.ly/abc", resolver=FakeResolver(resolves("bit.ly", "evil.pages.dev")),
                        http=FakeHttp(fn))
    row = insp.row
    assert rdap.domains == ["bit.ly"]
    assert row["rdap_skipped"] == 1 and row["free_hosting_subdomain"] == 1 and row["hosting_platform"] == "pages.dev"
    assert row["whois_registered_domain"] == -1 and row["orig_whois_registered_domain"] == 0
    assert row["final_reg_domain"] == "evil.pages.dev" and row["final_differs"] == 1


def test_final_is_popular_flag_and_the_path_hosting_exception():
    async def to_google(norm):
        return ok_result("https://www.google.com/", redirects=1)

    async def to_forms(norm):
        return ok_result("https://docs.google.com/forms/d/x/viewform", redirects=1)
    popular = {"google.com"}
    res = resolves("cloak.example.net", "www.google.com", "docs.google.com")
    a, _, _ = one("https://cloak.example.net/", resolver=FakeResolver(res), http=FakeHttp(to_google), popular=popular)
    b, _, _ = one("https://cloak.example.net/", resolver=FakeResolver(res), http=FakeHttp(to_forms), popular=popular)
    assert a.row["final_is_popular"] == 1                            # a cloaking redirect to a big site
    assert b.row["final_is_popular"] == 0                            # a phishing FORM on a Google domain
    assert b.row["platform_path_hosting"] == 1


def test_chain_that_broke_keeps_the_last_hop_as_final():
    async def fn(norm):
        r = HttpResult(final_url="https://step1.example.com/a", final_status=302, chain_complete=False,
                       redirects=1, fail_reason="redirect_to_blocked", scheme_used="https", elapsed_ms=10)
        r.hops = [Hop(0, 302, "step1.example.com", "10.0.0.1", 5)]
        return r
    insp, _, _ = one("https://step1.example.com/a", resolver=FakeResolver(resolves("step1.example.com")),
                     http=FakeHttp(fn))
    row = insp.row
    assert row["fail_reason"] == "redirect_to_blocked" and row["chain_complete"] == 0
    assert row["fetch_ok"] == 0 and row["page_fetched"] == 0 and row["html_available"] == 0
    assert row["final_url"] == "https://step1.example.com/a"
    assert [h.status for h in insp.hops] == [302]
    for name in schema.WEBPAGE_FEATURES:
        assert row[name] == -1


def test_challenge_page_row_keeps_helpers_but_routes_to_url_only():
    async def fn(norm):
        r = ok_result(norm.normalized_url)
        r.fail_reason = "challenge_page"
        r.analysis = {"features": {}, "html_bytes": 900, "script_count": 3, "near_empty": 1, "is_html": True,
                      "challenge": True, "parked": False, "redirect_url": "", "errors": []}
        return r
    insp, _, _ = one("https://walled.example.com/", resolver=FakeResolver(resolves("walled.example.com")),
                     http=FakeHttp(fn))
    row = insp.row
    assert row["fail_reason"] == "challenge_page"
    assert row["page_fetched"] == 0 and row["html_available"] == 0       # must route to url_only
    assert row["html_bytes"] == 900 and row["near_empty"] == 1
    for name in schema.WEBPAGE_FEATURES:
        assert row[name] == -1


def test_an_exception_in_a_stage_becomes_an_internal_error_row_not_a_crash():
    async def fn(norm):
        raise RuntimeError("kaboom")
    insp, _, _ = one("https://x.example.com/", resolver=FakeResolver(resolves("x.example.com")), http=FakeHttp(fn))
    assert insp.row["fail_reason"] == "internal_error"
    assert any(e[1] == "RuntimeError" for e in insp.errors)
    assert insp.row["fetch_ok"] == 0


# ---------------------------------------------------------------------------------------
# real fetching against the local server (loopback test mode)
# ---------------------------------------------------------------------------------------
def test_local_fixture_end_to_end(server):
    raw = server.url("/page/html_all_internal")
    insp, resolver, rdap = one(raw, allow_loopback=True)
    row = insp.row
    assert row["fail_reason"] == "none" and row["fetch_ok"] == 1 and row["page_fetched"] == 1
    # Judged against the page's OWN host (127.0.0.1): the 6 relative links are internal, the 4
    # absolute links to example.com are external.
    assert row["nb_hyperlinks"] == 10 and row["ratio_intHyperlinks"] == pytest.approx(0.6)
    assert row["html_bytes"] == len(PAGES["html_all_internal"])
    expected = link_features(raw)
    for name in schema.URL_TEXT_FEATURES:
        assert row[name] == expected[name], name
    assert resolver.calls == [] and rdap.domains == []               # loopback never touches DNS/RDAP
    assert row["scheme_used"] == "http" and row["final_status"] == 200


def test_local_cross_host_redirect_sets_the_final_columns(server):
    insp, _, _ = one(server.url("/cross-host"), allow_loopback=True)
    row = insp.row
    assert row["fail_reason"] == "none" and row["hop_count"] == 1
    assert row["final_differs"] == 1 and "localhost" in row["final_url"]
    assert [h.status for h in insp.hops] == [302, 200]


def test_every_url_yields_exactly_one_row_with_every_key(server):
    paths = ["/page/plain", "/404", "/500", "/json", "/empty", "/redirect/2", "/loop", "/to-private",
             "/page/challenge", "/page/parked", "/page/html_login_empty_action"]
    raws = [server.url(p) for p in paths] + ["ftp://x.com/", "http://10.0.0.1/", "http://example.com:8080/"]
    results, _, _ = inspect_all(raws, allow_loopback=True)
    assert len(results) == len(raws)
    template = schema.new_row("a" * 16, "x", "b")
    for insp in results:
        assert set(template) <= set(insp.row)
        assert insp.row["fail_reason"] in schema.FAIL_REASONS
        for name in schema.VM_FEATURES:
            assert isinstance(insp.row[name], (int, float)), name
        assert insp.row["fetch_ok"] == (1 if insp.row["fail_reason"] == "none" else 0)
        assert insp.row["page_fetched"] == insp.row["html_available"]
        if insp.row["page_fetched"] == 0:
            assert all(insp.row[n] == -1 for n in schema.WEBPAGE_FEATURES)
    by_path = {p: r for p, r in zip(paths, results)}
    assert by_path["/404"].row["fail_reason"] == "http_4xx"
    assert by_path["/500"].row["fail_reason"] == "http_5xx"
    assert by_path["/loop"].row["fail_reason"] == "too_many_redirects"
    assert by_path["/to-private"].row["fail_reason"] == "redirect_to_blocked"
    assert by_path["/page/html_login_empty_action"].row["login_form"] == 1


# ---------------------------------------------------------------------------------------
# a whole batch: persistence, resume, concurrency, order
# ---------------------------------------------------------------------------------------
def run_batch(raws, db, http=None, resolver=None, rdap=None, batch="b001", **kw):
    items = [(uid(r), r) for r in raws]

    async def go():
        store = Storage(db)
        try:
            async with Pipeline(settings(**kw), resolver=resolver or FakeResolver(), rdap_fetch=rdap or FakeRdap(),
                                bootstrap=BOOTSTRAP) as p:
                if http is not None:
                    p.http = http
                return await p.run(items, batch, store), store.done_ids()
        finally:
            store.close()
    return asyncio.run(go())


def test_batch_persists_every_url_and_resumes(server, tmp_path):
    db = str(tmp_path / "b.sqlite")
    raws = [server.url(p) for p in ("/page/plain", "/404", "/page/parked", "/redirect/2")] + ["ftp://x.com/"]
    counts, done = run_batch(raws, db, allow_loopback=True)
    assert counts == {"done": 2, "failed:http_4xx": 1, "failed:parked_or_default_page": 1, "failed:rejected_url": 1}
    assert len(done) == 5
    hits_before = len(server.hits)
    counts2, _ = run_batch(raws, db, allow_loopback=True)                    # resume: nothing left to do
    assert counts2 == counts and len(server.hits) == hits_before

    store = Storage(db)
    try:
        rows = list(store.rows("b001"))
        assert len(rows) == 5 and rows == sorted(rows, key=lambda r: r["url_id"])
        redirect_row = next(r for r in rows if r["url"].endswith("/redirect/2"))
        assert [h[1] for h in store.hops_of(redirect_row["url_id"])] == [302, 302, 200]
        assert store.status_counts("b001") == counts
    finally:
        store.close()


def test_internal_error_rows_are_retried_on_resume(tmp_path):
    db = str(tmp_path / "b.sqlite")
    raw = "https://flaky.example.com/"
    calls = {"n": 0}

    async def fn(norm):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first time fails")
        return ok_result(norm.normalized_url)
    http = FakeHttp(fn)
    res = FakeResolver(resolves("flaky.example.com"))
    c1, _ = run_batch([raw], db, http=http, resolver=res)
    assert c1 == {"failed:internal_error": 1}
    c2, _ = run_batch([raw], db, http=http, resolver=res)
    assert c2 == {"done": 1}


def test_concurrency_limit_is_respected(tmp_path):
    state = {"now": 0, "max": 0}

    async def fn(norm):
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.15)
        state["now"] -= 1
        return ok_result(norm.normalized_url)
    raws = [f"https://site{i}.example.com/" for i in range(12)]
    res = FakeResolver({**resolves(*[f"site{i}.example.com" for i in range(12)])})
    run_batch(raws, str(tmp_path / "c.sqlite"), http=FakeHttp(fn), resolver=res, concurrency=3)
    assert state["max"] == 3


def test_start_rate_is_limited(tmp_path):
    starts = []

    async def fn(norm):
        starts.append(asyncio.get_running_loop().time())
        return ok_result(norm.normalized_url)
    raws = [f"https://r{i}.example.com/" for i in range(6)]
    res = FakeResolver({**resolves(*[f"r{i}.example.com" for i in range(6)])})
    run_batch(raws, str(tmp_path / "r.sqlite"), http=FakeHttp(fn), resolver=res, rate=10.0)
    gaps = [b - a for a, b in zip(sorted(starts), sorted(starts)[1:])]
    assert min(gaps) >= 0.08                                         # <= ~10 starts per second


def test_order_is_a_seeded_shuffle(tmp_path):
    def order_for(seed):
        seen = []

        async def fn(norm):
            seen.append(norm.normalized_url)
            return ok_result(norm.normalized_url)
        raws = [f"https://o{i}.example.com/" for i in range(25)]
        res = FakeResolver({**resolves(*[f"o{i}.example.com" for i in range(25)])})
        run_batch(raws, str(tmp_path / f"o{seed}_{len(seen)}_{id(seen)}.sqlite"), http=FakeHttp(fn),
                  resolver=res, seed=seed, concurrency=1)
        return seen
    a1, a2, b = order_for(7), order_for(7), order_for(8)
    assert a1 == a2 and a1 != b
    assert a1 != sorted(a1)                                          # it really is shuffled


def test_a_failing_url_does_not_stop_the_others(tmp_path):
    async def fn(norm):
        if "bad" in norm.normalized_url:
            raise RuntimeError("only this one fails")
        return ok_result(norm.normalized_url)
    raws = ["https://good1.example.com/", "https://bad.example.com/", "https://good2.example.com/"]
    res = FakeResolver({**resolves("good1.example.com", "bad.example.com", "good2.example.com")})
    counts, _ = run_batch(raws, str(tmp_path / "f.sqlite"), http=FakeHttp(fn), resolver=res)
    assert counts == {"done": 2, "failed:internal_error": 1}


def test_stored_rows_are_sanitized(tmp_path):
    async def fn(norm):
        return ok_result("https://evil.example.com/\x1b[31m‮=cmd|calc")
    res = FakeResolver({**resolves("start.example.com", "evil.example.com")})
    db = str(tmp_path / "s.sqlite")
    run_batch(["https://start.example.com/"], db, http=FakeHttp(fn), resolver=res)
    store = Storage(db)
    try:
        row = next(store.rows())
        assert "\x1b" not in row["final_url"] and "‮" not in row["final_url"]
    finally:
        store.close()
