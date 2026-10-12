"""
Tests for Stage 1c, the certificate-transparency age lookup (src/lejet/stages/ct_stage.py).
A fake fetcher stands in for crt.sh: no network, no real waiting.

Run with:  python -m pytest tests/test_ct_stage.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from lejet import safety
from lejet.stages import ct_stage
from lejet.stages.ct_stage import CtClient, CtRecord

NOW = int(datetime(2026, 10, 12, tzinfo=timezone.utc).timestamp())
DAY = 86400


def body(*not_befores, extra=()):
    entries = [{"id": i, "not_before": nb, "common_name": "x.test", "issuer_name": "CA"}
               for i, nb in enumerate(not_befores)]
    return json.dumps(list(extra) + entries).encode()


class FakeCt:
    """url -> list of (status, body) answers, served in order (the last one repeats)."""

    def __init__(self, table):
        self.table = {k: list(v) for k, v in table.items()}
        self.urls = []

    async def __call__(self, url):
        self.urls.append(url)
        answers = self.table.get(url, [(404, b"")])
        return answers.pop(0) if len(answers) > 1 else answers[0]


def client(fetch, **kw):
    async def nosleep(_):
        return None
    kw.setdefault("min_interval", 0.0)
    kw.setdefault("retry_backoff", 0.0)
    return CtClient(fetch, sleep=nosleep, **kw)


def run(coro):
    return asyncio.run(coro)


EXACT = "https://crt.sh/?q=alpha.com.ph&output=json"
WILD = "https://crt.sh/?q=%25.alpha.com.ph&output=json"


# ---------------------------------------------------------------------------------------
# parsing a crt.sh answer (third-party data: it must never crash)
# ---------------------------------------------------------------------------------------
def test_oldest_certificate_wins():
    rec = ct_stage.parse_response(200, body("2025-06-01T00:00:00", "2024-01-15T08:30:00", "2025-01-01T00:00:00"), NOW)
    assert rec.ok and rec.first_ts == int(datetime(2024, 1, 15, 8, 30, tzinfo=timezone.utc).timestamp())


def test_age_in_days_is_computed_from_the_oldest_certificate():
    rec = CtRecord(True, NOW - 10 * DAY - 5)
    assert ct_stage.age_days(rec, NOW) == 10


def test_empty_list_is_a_completed_lookup_with_no_certificate():
    rec = ct_stage.parse_response(200, b"[]", NOW)
    assert rec.ok and rec.first_ts is None and ct_stage.age_days(rec, NOW) == -1


@pytest.mark.parametrize("status, payload", [
    (500, b"[]"), (429, b""), (0, b""), (404, b"[]"),
    (200, b"not json"), (200, b'{"a": 1}'), (200, b'"text"'), (200, b"null"),
])
def test_bad_answers_are_a_failed_lookup(status, payload):
    rec = ct_stage.parse_response(status, payload, NOW)
    assert not rec.ok and ct_stage.age_days(rec, NOW) == -1


def test_absurdly_nested_json_is_a_failed_lookup_not_a_crash():
    rec = ct_stage.parse_response(200, b"[" * 100000, NOW)
    assert not rec.ok


def test_junk_entries_and_impossible_dates_are_ignored():
    junk = [1, "x", None, [], {"not_before": 5}, {"not_before": None}, {"not_before": "garbage"},
            {"not_before": "1969-01-01T00:00:00"},          # before certificates existed
            {"not_before": "2099-01-01T00:00:00"},          # in the future
            {"not_before": "A" * 10000}]
    rec = ct_stage.parse_response(200, body("2023-05-05T00:00:00", extra=junk), NOW)
    assert rec.ok and rec.first_ts == int(datetime(2023, 5, 5, tzinfo=timezone.utc).timestamp())


def test_only_junk_means_ok_with_no_certificate():
    rec = ct_stage.parse_response(200, json.dumps([{"not_before": "garbage"}, 7]).encode(), NOW)
    assert rec.ok and rec.first_ts is None


def test_a_timestamp_with_timezone_and_fraction_is_understood():
    rec = ct_stage.parse_response(200, body("2024-03-01T10:00:00.123+02:00"), NOW)
    assert rec.ok and rec.first_ts == int(datetime(2024, 3, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp())


# ---------------------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------------------
def test_exact_query_is_enough_when_it_finds_certificates():
    fetch = FakeCt({EXACT: [(200, body("2024-01-01T00:00:00"))]})
    out = run(client(fetch).lookup("alpha.com.ph"))
    assert out.record.ok and out.record.first_ts and fetch.urls == [EXACT]


def test_wildcard_query_is_tried_only_when_the_exact_one_is_empty():
    fetch = FakeCt({EXACT: [(200, b"[]")], WILD: [(200, body("2022-02-02T00:00:00"))]})
    out = run(client(fetch).lookup("alpha.com.ph"))
    assert out.record.ok and out.record.first_ts and fetch.urls == [EXACT, WILD]


def test_no_certificate_anywhere_is_ok_but_unknown():
    fetch = FakeCt({EXACT: [(200, b"[]")], WILD: [(200, b"[]")]})
    out = run(client(fetch).lookup("alpha.com.ph"))
    assert out.record.ok and out.record.first_ts is None


def test_one_retry_on_a_server_error_then_success():
    fetch = FakeCt({EXACT: [(502, b""), (200, body("2024-01-01T00:00:00"))]})
    c = client(fetch)
    out = run(c.lookup("alpha.com.ph"))
    assert out.record.ok and c.requests_made == 2


def test_two_server_errors_are_a_failed_lookup_and_not_cached():
    fetch = FakeCt({EXACT: [(503, b"")]})
    c = client(fetch)
    out = run(c.lookup("alpha.com.ph"))
    assert not out.record.ok and out.note == "fetch_failed" and c.requests_made == 2
    run(c.lookup("alpha.com.ph"))
    assert c.requests_made == 4                              # failures are not cached


def test_a_fetcher_that_raises_is_a_failed_lookup_not_a_crash():
    async def boom(url):
        raise RuntimeError("network down")
    out = run(client(boom).lookup("alpha.com.ph"))
    assert not out.record.ok


def test_cache_answers_the_second_question_without_a_request():
    fetch = FakeCt({EXACT: [(200, body("2024-01-01T00:00:00"))]})
    c = client(fetch)

    async def go():
        a = await c.lookup("alpha.com.ph")
        b = await c.lookup("ALPHA.com.ph.")
        return a, b
    a, b = run(go())
    assert not a.from_cache and b.from_cache and len(fetch.urls) == 1


def test_two_simultaneous_questions_make_one_request():
    fetch = FakeCt({EXACT: [(200, body("2024-01-01T00:00:00"))]})
    c = client(fetch)

    async def go():
        return await asyncio.gather(c.lookup("alpha.com.ph"), c.lookup("alpha.com.ph"))
    a, b = run(go())
    assert a.record == b.record and len(fetch.urls) == 1


@pytest.mark.parametrize("bad", ["", "nodot", "a b.com", "-x.com", "x..com", "x.com/../etc", "x.com?q=1", "x" * 300 + ".com"])
def test_bad_domains_never_reach_the_network(bad):
    fetch = FakeCt({})
    out = run(client(fetch).lookup(bad))
    assert not out.record.ok and out.note == "bad_domain" and fetch.urls == []


def test_requests_are_spaced_by_the_minimum_interval():
    fetch = FakeCt({"https://crt.sh/?q=a.com.ph&output=json": [(200, body("2024-01-01T00:00:00"))],
                    "https://crt.sh/?q=b.com.ph&output=json": [(200, body("2024-01-01T00:00:00"))]})
    now = [100.0]
    slept = []

    async def sleep(s):
        slept.append(s)
        now[0] += s
    c = CtClient(fetch, min_interval=2.0, retry_backoff=0.0, clock=lambda: now[0], sleep=sleep)

    async def go():
        await c.lookup("a.com.ph")
        await c.lookup("b.com.ph")
    run(go())
    assert slept and abs(sum(slept) - 2.0) < 1e-6


def test_cache_file_round_trip_and_hostile_cache_file(tmp_path):
    fetch = FakeCt({EXACT: [(200, body("2024-01-01T00:00:00"))]})
    c = client(fetch)
    run(c.lookup("alpha.com.ph"))
    path = tmp_path / "ct.json"
    assert c.save_cache(str(path)) == 1
    c2 = client(FakeCt({}))
    assert c2.load_cache(str(path)) == 1
    assert run(c2.lookup("alpha.com.ph")).from_cache

    evil = {"ok.com.ph": {"first_ts": 1700000000}, "bad domain": {"first_ts": 1}, "x.com": {"first_ts": "no"},
            "y.com": {"first_ts": 5}, "z.com": {"first_ts": True}, "w.com": 7, "v.com": {"first_ts": None}}
    path.write_text(json.dumps(evil))
    c3 = client(FakeCt({}))
    assert c3.load_cache(str(path)) == 2                     # ok.com.ph and v.com only
    path.write_text("{ not json")
    assert client(FakeCt({})).load_cache(str(path)) == 0
    assert client(FakeCt({})).load_cache(str(tmp_path / "missing.json")) == 0


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def breaker_client(fetch, clock, **kw):
    async def sleep(s):
        clock.now += s
    return CtClient(fetch, min_interval=0.0, retry_backoff=0.0, clock=clock, sleep=sleep, **kw)


def test_three_failures_in_a_row_pause_crt_sh_so_the_run_stays_fast():
    fetch = FakeCt({})
    fetch.table = {f"https://crt.sh/?q=d{i}.com.ph&output=json": [(502, b"")] for i in range(10)}
    clock = Clock()
    c = breaker_client(fetch, clock)

    async def go():
        return [await c.lookup(f"d{i}.com.ph") for i in range(6)]
    outs = run(go())
    assert [o.note for o in outs[:3]] == ["fetch_failed"] * 3
    assert [o.note for o in outs[3:]] == ["ct_paused"] * 3
    assert c.paused_skips == 3
    assert len(fetch.urls) == 6                         # 3 lookups x (try + retry); nothing after the trip


def test_crt_sh_is_tried_again_after_the_pause():
    fetch = FakeCt({"https://crt.sh/?q=a.com.ph&output=json": [(502, b"")],
                    "https://crt.sh/?q=b.com.ph&output=json": [(502, b"")],
                    "https://crt.sh/?q=c.com.ph&output=json": [(502, b"")],
                    "https://crt.sh/?q=d.com.ph&output=json": [(200, body("2024-01-01T00:00:00"))]})
    clock = Clock()
    c = breaker_client(fetch, clock, breaker_pause=300.0)

    async def go():
        for d in ("a.com.ph", "b.com.ph", "c.com.ph"):
            await c.lookup(d)
        paused = await c.lookup("d.com.ph")
        clock.now += 301
        again = await c.lookup("d.com.ph")
        return paused, again
    paused, again = run(go())
    assert paused.note == "ct_paused" and again.record.ok and again.record.first_ts


def test_a_success_resets_the_failure_streak():
    table = {"https://crt.sh/?q=a.com.ph&output=json": [(502, b"")],
             "https://crt.sh/?q=b.com.ph&output=json": [(502, b"")],
             "https://crt.sh/?q=ok.com.ph&output=json": [(200, body("2024-01-01T00:00:00"))],
             "https://crt.sh/?q=c.com.ph&output=json": [(502, b"")],
             "https://crt.sh/?q=d.com.ph&output=json": [(502, b"")]}
    fetch = FakeCt(table)
    c = breaker_client(fetch, Clock())

    async def go():
        return [(await c.lookup(d)).note for d in ("a.com.ph", "b.com.ph", "ok.com.ph", "c.com.ph", "d.com.ph")]
    notes = run(go())
    assert "ct_paused" not in notes                     # two failures, a success, two failures: never 3 in a row


def test_lookups_queued_behind_a_dead_service_give_up_immediately_once_it_trips():
    """The real situation: many rows ask at once while crt.sh is down. Only 3 requests may be made."""
    async def always_down(url):
        raise TimeoutError("crt.sh is not answering")
    clock = Clock()

    async def sleep(s):
        clock.now += s
        await asyncio.sleep(0)
    c = CtClient(always_down, min_interval=2.0, retry_backoff=0.0, clock=clock, sleep=sleep)

    async def go():
        return await asyncio.gather(*[c.lookup(f"d{i}.com.ph") for i in range(10)])
    outs = run(go())
    notes = sorted(o.note for o in outs)
    assert notes.count("fetch_failed") == 3 and notes.count("ct_paused") == 7
    assert c.requests_made == 3                            # no retry after a timeout, none after the trip


def test_a_timeout_is_not_retried_but_a_server_error_is():
    async def down(url):
        raise TimeoutError()
    c1 = client(down)
    run(c1.lookup("alpha.com.ph"))
    assert c1.requests_made == 1
    c2 = client(FakeCt({EXACT: [(503, b"")]}))
    run(c2.lookup("alpha.com.ph"))
    assert c2.requests_made == 2


PROBE = "https://crt.sh/?q=probe.lejet.invalid&output=json"


def test_a_healthy_probe_leaves_crt_sh_open():
    fetch = FakeCt({PROBE: [(200, b"[]")], EXACT: [(200, body("2024-01-01T00:00:00"))]})
    c = breaker_client(fetch, Clock())

    async def go():
        healthy = await c.probe()
        return healthy, await c.lookup("alpha.com.ph")
    healthy, out = run(go())
    assert healthy is True and out.record.ok and fetch.urls[0] == PROBE


@pytest.mark.parametrize("answer", [(502, b""), (200, b"not json"), (200, b"{}"), (429, b"")])
def test_a_failing_probe_pauses_every_lookup_without_one_more_request(answer):
    fetch = FakeCt({PROBE: [answer], EXACT: [(200, body("2024-01-01T00:00:00"))]})
    c = breaker_client(fetch, Clock())

    async def go():
        healthy = await c.probe()
        return healthy, [await c.lookup(f"d{i}.com.ph") for i in range(5)]
    healthy, outs = run(go())
    assert healthy is False
    assert [o.note for o in outs] == ["ct_paused"] * 5
    assert all(u == PROBE for u in fetch.urls)             # only the probe (and its one 429/5xx retry) went out


def test_a_dead_service_is_found_by_the_probe_in_one_timeout():
    async def down(url):
        raise TimeoutError()
    clock = Clock()
    c = breaker_client(down, clock)
    assert run(c.probe()) is False and c.requests_made == 1


def test_the_probe_does_not_pollute_the_cache(tmp_path):
    fetch = FakeCt({PROBE: [(200, b"[]")]})
    c = breaker_client(fetch, Clock())
    run(c.probe())
    assert c.save_cache(str(tmp_path / "c.json")) == 0


def test_cached_answers_are_served_even_while_paused():
    fetch = FakeCt({"https://crt.sh/?q=ok.com.ph&output=json": [(200, body("2024-01-01T00:00:00"))]})
    clock = Clock()
    c = breaker_client(fetch, clock)
    run(c.lookup("ok.com.ph"))
    c._paused_until = clock.now + 999
    out = run(c.lookup("ok.com.ph"))
    assert out.from_cache and out.record.ok


def test_the_response_cap_is_two_megabytes():
    assert safety.MAX_CT_BYTES == 2 * 1024 * 1024


def test_real_fetcher_refuses_plain_http():
    async def go():
        f = ct_stage.HttpxCtFetcher()
        try:
            return await f("http://crt.sh/?q=x.com&output=json")
        finally:
            await f.aclose()
    assert run(go()) == (0, b"")
