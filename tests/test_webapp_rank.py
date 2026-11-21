"""
Tests for webapp/rank.py (the website's automatic page_rank look-up) against a FAKE Open PageRank built into httpx.
No internet, no real key. Skipped where httpx is not installed (the website's own requirements).

Run with:  python -m pytest tests/test_webapp_rank.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest

httpx = pytest.importorskip("httpx")

from lejet import page_rank as shared   # noqa: E402
from webapp import rank                   # noqa: E402

KEY = "TESTKEY-abcdef123456"
RANKS = {"example.com": 7, "bit.ly": 9, "unrated.example.org": None}


class Service:
    """A fake service: records requests, answers like the real one."""

    def __init__(self):
        self.requests = []
        self.mode = "ok"

    def __call__(self, request: "httpx.Request") -> "httpx.Response":
        self.requests.append(request)
        domains = request.url.params.get_list("domains[]")
        if request.headers.get("API-OPR") != KEY:
            return httpx.Response(403, text="forbidden")
        if self.mode == "500":
            return httpx.Response(500)
        if self.mode == "timeout":
            raise httpx.ReadTimeout("slow")
        if self.mode == "notjson":
            return httpx.Response(200, text="<html>nope</html>")
        if self.mode == "shape":
            return httpx.Response(200, json={"hello": 1})
        if self.mode == "scale":
            return httpx.Response(200, json={"response": [{"domain": d, "status_code": 200, "page_rank_integer": 55} for d in domains]})
        if self.mode == "huge":
            return httpx.Response(200, content=b"x" * (shared.MAX_RESPONSE + 10))
        items = []
        for d in domains:
            if d in RANKS:
                items.append({"domain": d, "status_code": 200, "page_rank_integer": RANKS[d]})
            else:                                                       # what the real service sends for an unknown domain
                items.append({"domain": d, "status_code": 404, "error": "Domain not found", "page_rank_integer": 0})
        return httpx.Response(200, json={"response": items})


@pytest.fixture()
def svc():
    return Service()


def client(svc, tmp_path=None, key=KEY, **kw):
    cache = str(tmp_path / "rank_cache.json") if tmp_path is not None else None
    kw.setdefault("retries", 0)                                              # the helper does not retry: older assertions stay exact
    return rank.RankClient(key, cache, base_url="http://127.0.0.1:9/getPageRank", transport=httpx.MockTransport(svc),
                           min_interval=0, retry_delay=0, **kw)


def run(coro):
    return asyncio.run(coro)


def test_ranks_follow_the_labs_rules(svc):
    c = client(svc)
    assert run(c.lookup("example.com")) == 7
    assert run(c.lookup("unrated.example.org")) == 0                       # answered, no rank: unrated = 0
    assert run(c.lookup("never-heard-of-it.top")) == 0                     # "Domain not found" is unrated too, not a failure
    assert run(c.lookup("example[.]com")) == 7                             # a defanged domain is accepted
    assert [r.url.params.get_list("domains[]") for r in svc.requests] == [["example.com"], ["unrated.example.org"], ["never-heard-of-it.top"]]
    assert all(r.headers["API-OPR"] == KEY for r in svc.requests)


@pytest.mark.parametrize("mode", ["500", "timeout", "notjson", "shape", "scale", "huge"])
def test_a_failed_lookup_is_minus_one_never_zero_and_is_not_cached(svc, mode, tmp_path):
    svc.mode = mode
    c = client(svc, tmp_path)
    assert run(c.lookup("example.com")) == -1
    assert c.last_error and KEY not in c.last_error
    svc.mode = "ok"                                                         # the service recovers: the domain is asked again
    assert run(c.lookup("example.com")) == 7 and c.last_error == ""
    assert len(svc.requests) == 2


def test_a_bad_key_gives_minus_one_and_says_why(svc):
    c = client(svc, key="WRONGKEY-abcdef123456")
    assert run(c.lookup("example.com")) == -1 and c.last_error == "HTTP 403"


def test_answers_are_cached_in_memory_and_on_disk(svc, tmp_path):
    c = client(svc, tmp_path)
    assert run(c.lookup("example.com")) == 7 and run(c.lookup("example.com")) == 7
    assert len(svc.requests) == 1
    again = client(svc, tmp_path)                                           # a restart keeps the answers
    assert run(again.lookup("example.com")) == 7 and len(svc.requests) == 1
    on_disk = json.loads((tmp_path / "rank_cache.json").read_text())
    assert on_disk == {"example.com": 7} and KEY not in json.dumps(on_disk)


def test_a_damaged_cache_file_is_ignored(svc, tmp_path):
    (tmp_path / "rank_cache.json").write_text(json.dumps({"example.com": 99, "ok.example.org": 4, "bad domain": 3, "x.test": True}))
    c = client(svc, tmp_path)
    assert run(c.lookup("ok.example.org")) == 4 and len(svc.requests) == 0   # the valid entry is used
    assert run(c.lookup("example.com")) == 7 and len(svc.requests) == 1      # the out-of-range one was dropped
    (tmp_path / "rank_cache.json").write_text("{not json")
    assert client(svc, tmp_path)._cache == {}


def test_only_real_domains_are_asked_about(svc):
    c = client(svc)
    for bad in ("", "localhost", "192.168.1.1", "a b.com", "https://example.com/x", "-.com"):
        assert run(c.lookup(bad)) == -1
    assert svc.requests == []


def test_without_a_key_it_is_off_and_asks_nothing(svc):
    c = client(svc, key=None)
    assert c.enabled is False and run(c.lookup("example.com")) == -1 and svc.requests == []


def test_requests_are_spaced_out(svc):
    c = rank.RankClient(KEY, None, base_url="http://127.0.0.1:9/x", transport=httpx.MockTransport(svc), min_interval=0.25)
    t0 = time.monotonic()
    run(asyncio.wait_for(_two(c), 5))
    assert time.monotonic() - t0 >= 0.24                                    # the second request waited


async def _two(c):
    await asyncio.gather(c.lookup("example.com"), c.lookup("bit.ly"))


def test_the_key_never_shows_up_in_logs_repr_or_errors(svc, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    c = client(svc, tmp_path)
    run(c.lookup("example.com"))
    svc.mode = "500"
    run(c.lookup("bit.ly"))
    svc.mode = "timeout"
    run(c.lookup("unrated.example.org"))
    blob = repr(c) + c.last_error + caplog.text + json.dumps(c._cache)
    assert KEY not in blob


def test_key_file_reading(tmp_path):
    f = tmp_path / "k"
    f.write_text(KEY + "\n", encoding="ascii")
    assert rank.read_key(str(f)) == KEY
    f.write_text("OPENPAGERANK_API_KEY=" + KEY + "\n", encoding="ascii")        # the line from the lab's .env is fine too
    assert rank.read_key(str(f)) == KEY
    for bad in ("short", "has space in it 12345", "quote'd-key-1234567", "é" * 12, "x" * 5000, ""):
        f.write_bytes(bad.encode("utf-8"))
        assert rank.read_key(str(f)) is None
    assert rank.read_key(str(tmp_path / "missing")) is None


def test_from_file_without_a_key_is_off_and_logs_no_secret(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    c = rank.RankClient.from_file(str(tmp_path / "missing"), None)
    assert c.enabled is False and "OFF" in caplog.text
    (tmp_path / "k").write_text(KEY, encoding="ascii")
    on = rank.RankClient.from_file(str(tmp_path / "k"), None)
    assert on.enabled is True and KEY not in caplog.text


def test_plain_http_to_a_real_host_is_refused():
    with pytest.raises(ValueError):
        rank.RankClient(KEY, None, base_url="http://openpagerank.com/api/v1.0/getPageRank")
    assert rank.RankClient(KEY, None, base_url="https://openpagerank.com/api/v1.0/getPageRank").enabled


# ---------------------------------------------------------------------------------------------- back-off, logging, visibility
@pytest.mark.parametrize("status", [401, 403, 429])
def test_after_bad_key_or_no_quota_it_stops_asking_for_a_while(status, tmp_path):
    hits = []

    def refuse(request):
        hits.append(request)
        return httpx.Response(status)

    c = rank.RankClient(KEY, None, base_url="http://127.0.0.1:9/x", transport=httpx.MockTransport(refuse), min_interval=0, pause_seconds=0.4)
    assert run(c.lookup("example.com")) == -1 and c.last_error == f"HTTP {status}"
    assert run(c.lookup("bit.ly")) == -1 and run(c.lookup("example.org")) == -1
    assert len(hits) == 1                                                  # the others were not even sent
    time.sleep(0.45)
    assert run(c.lookup("bit.ly")) == -1 and len(hits) == 2                # the pause is over: it asks again


def test_a_server_error_does_not_pause_the_next_lookup(svc):
    svc.mode = "500"
    c = client(svc)
    run(c.lookup("example.com"))
    svc.mode = "ok"
    assert run(c.lookup("example.com")) == 7 and len(svc.requests) == 2    # only 401/403/429 trigger the pause


def test_failures_are_logged_without_the_key_and_not_repeated_every_time(svc, caplog):
    caplog.set_level(logging.WARNING)
    svc.mode = "500"
    c = client(svc)
    for d in ("example.com", "bit.ly", "unrated.example.org"):
        run(c.lookup(d))
    lines = [r.getMessage() for r in caplog.records if "page_rank look-up failed" in r.getMessage()]
    assert len(lines) == 1 and "HTTP 500" in lines[0] and "example.com" in lines[0]      # one line, the reason, the domain
    assert KEY not in caplog.text


# ---------------------------------------------------------------------------------------------- one more try for a slow service
def _flaky(first_failures, failure):
    """A service whose first N answers fail (`failure` is an exception to raise or an HTTP status), then it answers."""
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) <= first_failures:
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure)
        return httpx.Response(200, json={"response": [{"domain": "example.com", "status_code": 200, "page_rank_integer": 7}]})

    return seen, httpx.MockTransport(handler)


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), httpx.ConnectError("down"), 502, 503])
def test_a_timeout_or_server_error_gets_one_more_try(failure):
    seen, transport = _flaky(1, failure)
    c = rank.RankClient(KEY, None, base_url="http://127.0.0.1:9/x", transport=transport, min_interval=0, retry_delay=0, retries=1)
    assert run(c.lookup("example.com")) == 7 and len(seen) == 2 and c.last_error == ""


def test_two_failures_in_a_row_still_end_as_minus_one():
    seen, transport = _flaky(5, httpx.ReadTimeout("slow"))
    c = rank.RankClient(KEY, None, base_url="http://127.0.0.1:9/x", transport=transport, min_interval=0, retry_delay=0, retries=1)
    assert run(c.lookup("example.com")) == -1 and len(seen) == 2 and c.last_error == "ReadTimeout"


@pytest.mark.parametrize("failure", [403, 429, 401, 404])
def test_a_refusal_is_not_retried(failure):
    seen, transport = _flaky(5, failure)
    c = rank.RankClient(KEY, None, base_url="http://127.0.0.1:9/x", transport=transport, min_interval=0, retry_delay=0, retries=3)
    assert run(c.lookup("example.com")) == -1 and len(seen) == 1


def test_an_unusable_answer_is_not_retried():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text="<html>not the api</html>")

    c = rank.RankClient(KEY, None, base_url="http://127.0.0.1:9/x", transport=httpx.MockTransport(handler), min_interval=0, retry_delay=0, retries=3)
    assert run(c.lookup("example.com")) == -1 and len(seen) == 1


# ---------------------------------------------------------------------------------------------- the service's redirect
OLD = "https://openpagerank.com/api/v1.0/getPageRank"
NEW = "https://openpagerank.keywordseverywhere.com/api/v1.0/getPageRank"


def _moved(location, final_ok=True, status=301):
    """openpagerank.com redirects to `location`; whatever host serves the answer is recorded, with whether it got the key."""
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers.get("API-OPR"), str(request.url)))
        if request.url.host == "openpagerank.com":
            return httpx.Response(status, headers={"Location": location})
        if not final_ok:
            return httpx.Response(200, json={"response": []})
        return httpx.Response(200, json={"response": [{"domain": "example.com", "status_code": 200, "page_rank_integer": 7}]})

    return seen, httpx.MockTransport(handler)


def _moved_client(transport):
    return rank.RankClient(KEY, None, base_url=OLD, transport=transport, min_interval=0, retries=0)


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_the_real_services_redirect_is_followed_with_the_key_and_the_query(status):
    seen, transport = _moved(NEW + "?domains%5B%5D=example.com", status=status)
    c = _moved_client(transport)
    assert run(c.lookup("example.com")) == 7 and c.last_error == ""
    assert [h for h, _, _ in seen] == ["openpagerank.com", "openpagerank.keywordseverywhere.com"]
    assert all(k == KEY for _, k, _ in seen)                            # the key goes to the two trusted hosts
    assert "domains%5B%5D=example.com" in seen[1][2]                    # the query survived the redirect


def test_a_relative_redirect_stays_on_the_same_trusted_host():
    seen, transport = _moved("/api/v2/getPageRank?domains%5B%5D=example.com")
    c = rank.RankClient(KEY, None, base_url="https://openpagerank.keywordseverywhere.com/old", transport=httpx.MockTransport(
        lambda r: httpx.Response(301, headers={"Location": "/api/v1.0/getPageRank?domains%5B%5D=example.com"}) if r.url.path == "/old"
        else httpx.Response(200, json={"response": [{"domain": "example.com", "status_code": 200, "page_rank_integer": 4}]})), min_interval=0, retries=0)
    assert run(c.lookup("example.com")) == 4


@pytest.mark.parametrize("location", [
    "https://evil.example.net/api/v1.0/getPageRank?domains%5B%5D=example.com",               # some other host
    "https://openpagerank.keywordseverywhere.com.evil.net/x",                                 # a look-alike
    "http://openpagerank.keywordseverywhere.com/api/v1.0/getPageRank",                        # a downgrade to plain http
    "https://openpagerank.keywordseverywhere.com:8443/api/v1.0/getPageRank",                  # a strange port
    "https://user:pw@openpagerank.keywordseverywhere.com/api/v1.0/getPageRank",               # credentials in the address
    "",                                                                                       # no Location at all
])
def test_a_redirect_anywhere_else_is_refused_and_the_key_is_not_sent(location):
    seen, transport = _moved(location)
    c = _moved_client(transport)
    assert run(c.lookup("example.com")) == -1
    assert c.last_error == "redirected to an address that is not trusted"
    assert [h for h, _, _ in seen] == ["openpagerank.com"]               # nothing was sent to the other address
    assert KEY not in c.last_error


def test_a_redirect_loop_is_cut_off():
    n = []

    def loop(request):
        n.append(1)
        return httpx.Response(301, headers={"Location": "https://openpagerank.com/api/v1.0/getPageRank?domains%5B%5D=example.com"})

    c = _moved_client(httpx.MockTransport(loop))
    assert run(c.lookup("example.com")) == -1 and c.last_error == "too many redirects"
    assert len(n) == rank.MAX_REDIRECTS + 1


def test_a_redirect_is_not_a_reason_to_pause_or_retry():
    seen, transport = _moved("https://evil.example.net/x")
    c = rank.RankClient(KEY, None, base_url=OLD, transport=transport, min_interval=0, retries=3, retry_delay=0)
    assert run(c.lookup("example.com")) == -1 and len(seen) == 1         # not retried
    seen2, transport2 = _moved(NEW + "?domains%5B%5D=example.com")
    c._client = httpx.AsyncClient(transport=transport2)                  # the next lookup is not paused either
    assert run(c.lookup("example.com")) == 7
