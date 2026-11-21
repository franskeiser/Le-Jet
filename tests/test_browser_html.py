"""
Tests for the browser fallback (src/lejet/browser_html.py + its hook in Pipeline._inspect): a site that refuses the plain
fetch is read through a (fake) browser and gets its page features; every other failure stays exactly as it was. No internet,
no browser: the browser is a fake and the HTTP stage is the pipeline tests' fake.

Run with:  python -m pytest tests/test_browser_html.py
"""
from __future__ import annotations

import asyncio
import types

import pytest

from lejet import browser_html as bh
from lejet.browser_html import BrowserHtml, BrowserPage, needs_browser
from lejet.pipeline import Pipeline
from test_pipeline_local import (BOOTSTRAP, FakeHttp, FakeRdap, FakeResolver, ok_result, resolves, settings, uid)

PAGE = ("<html><head><title>Example Bank</title><link rel='icon' href='/f.ico'></head><body>"
        "<a href='/a'>a</a><a href='/b'>b</a><a href='https://other.test/c'>c</a><form><input type='password'></form>"
        "<p>&copy; Example Bank</p></body></html>")


@pytest.mark.parametrize("reason, status, expected", [
    ("http_4xx", 403, True), ("http_4xx", 401, True), ("http_4xx", 406, True), ("http_4xx", 429, True),
    ("http_5xx", 503, True), ("challenge_page", 200, True), ("blocked_or_empty", 200, True),
    ("http_4xx", 404, False), ("http_4xx", 410, False), ("http_4xx", 400, False), ("http_5xx", 500, False),
    ("dns_nxdomain", -1, False), ("connect_timeout", -1, False), ("none", 200, False), ("parked_or_default_page", 200, False),
])
def test_which_failures_may_a_browser_get_past(reason, status, expected):
    assert needs_browser(reason, status) is expected


class FakeRunner:
    def __init__(self, html=PAGE, status=200, ok=True, page_url="https://www.example.com/home"):
        self.res = types.SimpleNamespace(ok=ok, html=html if ok else "", http_status=status, page_url=page_url)
        self.calls = []

    async def shoot(self, url_id, url):
        self.calls.append(url)
        return self.res


def run(coro):
    return asyncio.run(coro)


def test_a_good_browser_answer_is_returned():
    b = BrowserHtml("/tmp", runner=FakeRunner())
    page = run(b.fetch("https://www.example.com/"))
    assert isinstance(page, BrowserPage) and "Example Bank" in page.html and page.status == 200
    assert (b.used, b.got) == (1, 1)


@pytest.mark.parametrize("runner", [FakeRunner(status=403), FakeRunner(status=503), FakeRunner(ok=False), FakeRunner(html=" ")])
def test_a_browser_that_is_refused_too_gives_nothing(runner):
    assert run(BrowserHtml("/tmp", runner=runner).fetch("https://www.example.com/")) is None


def test_the_page_limit_and_the_time_budget_stop_the_browser():
    b = BrowserHtml("/tmp", runner=FakeRunner(), max_pages=2)
    assert [run(b.fetch(f"https://e.example/{i}")) is not None for i in range(4)] == [True, True, False, False]
    now = [0.0]
    c = BrowserHtml("/tmp", runner=FakeRunner(), budget_s=10, clock=lambda: now[0])
    assert run(c.fetch("https://e.example/a")) is not None
    now[0] = 11.0
    assert run(c.fetch("https://e.example/b")) is None


def inspect_with(fallback, http_fn, raw="https://www.example.com/"):
    async def go():
        async with Pipeline(settings(), resolver=FakeResolver(resolves("www.example.com")), rdap_fetch=FakeRdap(),
                            bootstrap=BOOTSTRAP, popular_domains=None) as p:
            p.http = FakeHttp(http_fn)
            if fallback is not None:
                p.html_fallback = fallback
            return await p.inspect(uid(raw), raw, "b001")
    return run(go())


def refused(status=403, reason="http_4xx"):
    async def fn(norm):
        r = ok_result(norm.normalized_url)
        r.fail_reason, r.final_status, r.analysis = reason, status, None
        return r
    return fn


def test_a_refused_site_gets_its_page_features_from_the_browser():
    runner = FakeRunner()
    b = BrowserHtml("/tmp", runner=runner)
    insp = inspect_with(b.fetch, refused())
    row = insp.row
    assert row["fail_reason"] == "none" and row["page_fetched"] == 1 and row["html_available"] == 1
    assert row["nb_hyperlinks"] > 0 and row["empty_title"] == 0 and row["final_status"] == 200
    assert ("http", "note", "browser_fallback") in insp.errors
    assert row["final_url"] == "https://www.example.com/home"


def test_without_the_fallback_the_row_is_exactly_as_before():
    row = inspect_with(None, refused()).row
    assert row["fail_reason"] == "http_4xx" and row["page_fetched"] == 0 and row["nb_hyperlinks"] == -1


def test_a_page_that_is_really_gone_never_goes_to_the_browser():
    runner = FakeRunner()
    row = inspect_with(BrowserHtml("/tmp", runner=runner).fetch, refused(404)).row
    assert runner.calls == [] and row["fail_reason"] == "http_4xx" and row["page_fetched"] == 0


def test_a_browser_that_is_refused_leaves_the_row_untouched():
    row = inspect_with(BrowserHtml("/tmp", runner=FakeRunner(status=403)).fetch, refused()).row
    assert row["fail_reason"] == "http_4xx" and row["page_fetched"] == 0


def test_a_challenge_page_from_the_browser_is_not_accepted():
    challenge = "<html><head><title>Just a moment...</title></head><body>Checking your browser before accessing</body></html>"
    insp = inspect_with(BrowserHtml("/tmp", runner=FakeRunner(html=challenge)).fetch, refused())
    assert insp.row["fail_reason"] == "http_4xx" and insp.row["page_fetched"] == 0


def test_a_crash_in_the_fallback_never_breaks_the_row():
    async def boom(url):
        raise RuntimeError("browser died")
    insp = inspect_with(boom, refused())
    assert insp.row["fail_reason"] == "http_4xx" and insp.row["page_fetched"] == 0
    assert any(e[2].startswith("browser_fallback_error") for e in insp.errors)
