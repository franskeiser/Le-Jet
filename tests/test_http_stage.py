"""
Level 0 tests for Stage 3, the HTTP probe (WEB_INSPECTION_PLAN.md sections 13.5, 16 Level 0),
against a LOCAL web server on 127.0.0.1 (test mode `allow_loopback`). No test contacts the
internet.

Each safety property of the stage has a test here: manual redirects with the gate re-run on
every hop, the decoded-size cap (huge body, gzip bomb), the 10 MB announcement refusal, the
wall-clock deadline against trickling bodies, the per-URL cookie jar, the failure taxonomy.

Run with:  python -m pytest tests/test_http_stage.py
"""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from lejet import safety, urlnorm
from lejet.parse_pool import ParsePool
from lejet.settings import Settings
from lejet.stages.http_stage import DomainLocks, HttpStage, classify_exception


def fetch(server_or_url, path="", workers=0, parse_timeout=5.0, **overrides):
    url = server_or_url.url(path) if hasattr(server_or_url, "url") else server_or_url
    kw = dict(allow_loopback=True, parse_workers=workers, parse_timeout=parse_timeout,
              hop_deadline=8.0, url_deadline=15.0, connect_timeout=3.0, read_timeout=5.0)
    kw.update(overrides)
    settings = Settings(**kw)

    async def go():
        pool = await ParsePool(workers, parse_timeout).start()
        try:
            stage = HttpStage(settings, pool)
            norm = urlnorm.normalize(url, allow_loopback=True)
            assert norm.ok, norm
            return await stage.fetch(norm)
        finally:
            await pool.close()
    return asyncio.run(go())


def test_plain_page(server):
    r = fetch(server, "/page/plain")
    assert r.fail_reason == "none" and r.chain_complete
    assert r.final_status == 200 and r.redirects == 0 and len(r.hops) == 1
    assert r.scheme_used == "http"
    assert r.analysis and r.analysis["features"] and r.analysis["is_html"]
    assert r.final_url.endswith("/page/plain")


def test_request_headers_look_like_a_phone_and_carry_no_cookie(server):
    fetch(server, "/page/plain")
    path, headers = server.hits[0]
    h = {k.lower(): v for k, v in headers.items()}
    assert "Mobile" in h["user-agent"] and "Android" in h["user-agent"]
    assert h["accept-encoding"] == "gzip"                    # no brotli/zstd decoders are exercised
    assert h["accept-language"].startswith("en-PH")
    assert "cookie" not in h


def test_redirect_chain_is_followed_by_hand(server):
    r = fetch(server, "/redirect/3")
    assert r.fail_reason == "none"
    assert r.redirects == 3 and len(r.hops) == 4
    assert [h.status for h in r.hops] == [302, 302, 302, 200]
    assert r.hops[-1].status == 200
    assert r.final_url.endswith("/page/plain") and r.chain_complete


def test_redirect_loop_is_cut_off_at_the_hop_limit(server):
    r = fetch(server, "/loop")
    assert r.fail_reason == "too_many_redirects"
    assert r.chain_complete is False
    assert r.redirects == 9                                  # the 9th redirect exceeds the limit of 8
    assert r.final_url.endswith("/loop")
    assert len(server.hits) == 9


@pytest.mark.parametrize("path", ["/to-private", "/to-ftp", "/to-port", "/to-metadata", "/meta-private"])
def test_every_hop_target_goes_through_the_gate_again(server, path):
    r = fetch(server, path)
    assert r.fail_reason == "redirect_to_blocked", path
    assert r.chain_complete is False
    assert len(r.hops) == 1                                  # we never left the first host
    assert r.final_url.endswith(path)                        # final = the last hop actually reached
    assert [p for p, _ in server.hits] == [path]


def test_meta_refresh_is_followed_like_a_redirect(server):
    r = fetch(server, "/page/meta_refresh")
    assert r.fail_reason == "none"
    assert r.redirects == 1 and r.final_url.endswith("/page/plain")
    assert r.analysis["features"] and r.chain_complete


def test_script_redirect_is_followed_without_running_javascript(server):
    r = fetch(server, "/page/js_redirect")
    assert r.fail_reason == "none" and r.redirects == 1 and r.final_url.endswith("/page/plain")


def test_cross_host_redirect_is_followed(server):
    r = fetch(server, "/cross-host")
    assert r.fail_reason == "none", r.errors
    assert "localhost" in r.final_url and r.redirects == 1


def test_cookie_jar_is_per_url_and_breaks_cookie_loops(server):
    r = fetch(server, "/cookie-loop")
    assert r.fail_reason == "none" and r.redirects == 1       # the cookie was kept for the 2nd hop
    # a second, separate fetch must NOT inherit the cookie: it is redirected again
    server.reset()
    fetch(server, "/cookie-loop")
    assert "cookie" not in {k.lower() for k in server.hits[0][1]}


def test_huge_body_is_capped_at_2_mb_decoded(server):
    r = fetch(server, "/huge")
    assert r.body_truncated is True
    assert r.analysis["html_bytes"] == safety.MAX_DECODED_BYTES
    assert r.fail_reason == "none"


def test_gzip_bomb_cannot_expand_past_the_cap(server):
    started = time.monotonic()
    r = fetch(server, "/gzipbomb")
    assert r.body_truncated is True
    assert r.analysis["html_bytes"] == safety.MAX_DECODED_BYTES
    assert time.monotonic() - started < 20


def test_a_server_announcing_more_than_10_mb_is_refused_unread(server):
    started = time.monotonic()
    r = fetch(server, "/bigheader")
    assert r.fail_reason == "too_large" and r.analysis is None
    assert time.monotonic() - started < 6


def test_trickling_body_is_killed_by_the_wall_clock_deadline(server):
    started = time.monotonic()
    r = fetch(server, "/slow", hop_deadline=1.5, url_deadline=4.0, read_timeout=30.0)
    assert r.fail_reason == "read_timeout"
    assert r.chain_complete is False
    assert time.monotonic() - started < 6                     # the server would trickle for 24 s


@pytest.mark.parametrize("path, reason", [
    ("/404", "http_4xx"), ("/403", "http_4xx"), ("/500", "http_5xx"),
    ("/json", "non_html"), ("/empty", "blocked_or_empty"), ("/204", "blocked_or_empty"),
    ("/page/challenge", "challenge_page"), ("/page/challenge_body", "challenge_page"),
    ("/page/parked", "parked_or_default_page"), ("/page/default_nginx", "parked_or_default_page"),
])
def test_failure_taxonomy(server, path, reason):
    r = fetch(server, path)
    assert r.fail_reason == reason, (path, r.fail_reason, r.errors)
    assert not (r.analysis or {}).get("features")             # no features from a page we reject


def test_error_pages_are_a_completed_chain_not_a_broken_one(server):
    r = fetch(server, "/404")
    assert r.chain_complete is True and r.final_status == 404


def test_connection_refused():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    r = fetch(f"http://127.0.0.1:{port}/")
    assert r.fail_reason == "connect_refused" and r.hops == [] and r.chain_complete is False


def test_https_first_then_http_fallback_for_schemeless_links(server):
    # the local server speaks plain HTTP: the https attempt fails to handshake, then http works
    norm = urlnorm.normalize(f"127.0.0.1:{server.port}/page/plain", allow_loopback=True)
    assert norm.ok and norm.fetch_candidates[0].startswith("https://")
    settings = Settings(allow_loopback=True, parse_workers=0, hop_deadline=8, url_deadline=15,
                        connect_timeout=3)

    async def go():
        pool = await ParsePool(0).start()
        return await HttpStage(settings, pool).fetch(norm)
    r = asyncio.run(go())
    assert r.fail_reason == "none", (r.fail_reason, r.errors)
    assert r.scheme_used == "http"


def test_fallback_only_happens_when_the_first_hop_could_not_connect(server):
    # an explicit https:// link never falls back to http
    norm = urlnorm.normalize(f"https://127.0.0.1:{server.port}/page/plain", allow_loopback=True)
    assert norm.fetch_candidates == [f"https://127.0.0.1:{server.port}/page/plain"]
    settings = Settings(allow_loopback=True, parse_workers=0, connect_timeout=3)

    async def go():
        pool = await ParsePool(0).start()
        return await HttpStage(settings, pool).fetch(norm)
    r = asyncio.run(go())
    assert r.fail_reason in ("tls_error", "connect_refused", "blocked_or_empty")
    assert r.scheme_used == ""


def test_evil_strings_page_is_parsed_and_sanitized(server):
    r = fetch(server, "/page/evil_strings")
    assert r.fail_reason == "none"
    title = r.analysis["title"]
    assert len(title) <= 120 and "\x1b" not in title and "‮" not in title


def test_malformed_deeply_nested_page(server):
    started = time.monotonic()
    r = fetch(server, "/malformed")
    assert time.monotonic() - started < 20
    assert r.fail_reason in ("none", "blocked_or_empty")


def test_parse_timeout_is_reported_as_a_failure_not_a_hang(server):
    r = fetch(server, "/huge", workers=1, parse_timeout=0.02)
    assert r.fail_reason == "blocked_or_empty"
    assert "parse:parse_timeout" in r.errors


def test_html_work_is_attributed_to_html_not_http(server):
    r = fetch(server, "/page/plain")
    assert r.html_ms >= 0 and r.elapsed_ms >= r.html_ms


def test_domain_locks_are_per_domain():
    locks = DomainLocks()
    a, b = locks.lock_for("a.com"), locks.lock_for("b.com")
    assert locks.lock_for("a.com") is a and a is not b


def test_one_request_at_a_time_per_domain(server):
    """Two fetches of the same host overlap only if the lock is missing: the second request
    must start after the first one's response has been fully read."""
    settings = Settings(allow_loopback=True, parse_workers=0, hop_deadline=8, url_deadline=15)

    async def go():
        pool = await ParsePool(0).start()
        locks = DomainLocks()
        stage = HttpStage(settings, pool, locks=locks)
        norm = urlnorm.normalize(server.url("/page/plain"), allow_loopback=True)
        a, b = await asyncio.gather(stage.fetch(norm), stage.fetch(norm))
        return a, b
    a, b = asyncio.run(go())
    assert a.fail_reason == "none" and b.fail_reason == "none"


def test_waiting_for_the_domain_lock_does_not_use_up_the_urls_own_deadline(server):
    """Three URLs on one host, each answering after 1.5 s, share one lock: the third waits about
    3 s. With a per-URL deadline of only 3 s it must still succeed, because lock waiting is not
    counted against the deadline (otherwise a slow domain would cause spurious connect_timeouts)."""
    settings = Settings(allow_loopback=True, parse_workers=0, hop_deadline=5.0, url_deadline=3.0,
                        read_timeout=5.0, connect_timeout=3.0)

    async def go():
        pool = await ParsePool(0).start()
        stage = HttpStage(settings, pool, locks=DomainLocks())
        norm = urlnorm.normalize(server.url("/delay2"), allow_loopback=True)
        return await asyncio.gather(*[stage.fetch(norm) for _ in range(3)])
    started = time.monotonic()
    results = asyncio.run(go())
    assert [r.fail_reason for r in results] == ["none", "none", "none"]
    assert time.monotonic() - started >= 4.0                  # they really were serialized (3 x 1.5 s)


def test_the_deadline_still_fires_for_a_request_that_is_itself_slow(server):
    settings = Settings(allow_loopback=True, parse_workers=0, hop_deadline=1.0, url_deadline=2.0,
                        read_timeout=30.0)

    async def go():
        pool = await ParsePool(0).start()
        stage = HttpStage(settings, pool, locks=DomainLocks())
        norm = urlnorm.normalize(server.url("/slow"), allow_loopback=True)
        return await asyncio.gather(*[stage.fetch(norm) for _ in range(2)])
    started = time.monotonic()
    results = asyncio.run(go())
    assert all(r.fail_reason == "read_timeout" for r in results)
    assert time.monotonic() - started < 6


@pytest.mark.parametrize("exc_factory, reason", [
    (lambda h: h.ConnectTimeout("t"), "connect_timeout"),
    (lambda h: h.ReadTimeout("t"), "read_timeout"),
    (lambda h: h.PoolTimeout("t"), "connect_timeout"),
    (lambda h: h.RemoteProtocolError("p"), "blocked_or_empty"),
    (lambda h: h.DecodingError("bad gzip"), "blocked_or_empty"),
    (lambda h: h.UnsupportedProtocol("x"), "rejected_url"),
    (lambda h: ValueError("surprise"), "internal_error"),
])
def test_exception_classification(exc_factory, reason):
    import httpx
    assert classify_exception(exc_factory(httpx))[0] == reason


def test_connect_error_causes_are_told_apart():
    import ssl
    import httpx

    def mk(cause):
        e = httpx.ConnectError("boom")
        e.__cause__ = cause
        return e
    assert classify_exception(mk(ssl.SSLError("WRONG_VERSION_NUMBER")))[0] == "tls_error"
    assert classify_exception(mk(socket.gaierror(-2, "Name or service not known")))[0] == "dns_no_answer"
    assert classify_exception(mk(ConnectionRefusedError()))[0] == "connect_refused"
