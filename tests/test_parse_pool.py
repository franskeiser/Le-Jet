"""
Tests for the HTML parse pool (WEB_INSPECTION_PLAN.md section 13.5): parsing runs OUTSIDE the
event loop in worker processes with a hard time limit and kill-on-timeout.

These tests start real `spawn`ed worker processes. No network.

Run with:  python -m pytest tests/test_parse_pool.py
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import time

from fixtures.pages import BASE_URL, PAGES, big_html
from lejet import parse_pool
from lejet.parse_pool import ParsePool


def args(body, **kw):
    d = dict(body=body, content_type="text/html", base_url=BASE_URL, want_features=True)
    d.update(kw)
    return d


def run(coro):
    return asyncio.run(coro)


def test_pool_analyzes_many_pages_in_parallel_workers():
    async def go():
        pool = await ParsePool(workers=2, timeout=30).start()
        try:
            names = ["plain", "html_all_internal", "html_mixed_links", "html_media_mix", "challenge", "meta_refresh"]
            return names, await asyncio.gather(*[pool.analyze(**args(PAGES[n])) for n in names])
        finally:
            await pool.close()
    names, results = run(go())
    by = dict(zip(names, results))
    assert by["html_all_internal"]["features"]["nb_hyperlinks"] == 10
    assert by["html_mixed_links"]["features"]["ratio_intHyperlinks"] == 0.7
    assert by["html_media_mix"]["features"]["ratio_intMedia"] > 66
    assert by["challenge"]["challenge"] is True
    assert by["meta_refresh"]["redirect_kind"] == "meta"
    assert all("rid" in r for r in results)


def test_inline_mode_needs_no_processes():
    async def go():
        pool = await ParsePool(workers=0).start()
        out = await pool.analyze(**args(PAGES["plain"]))
        await pool.close()
        return out
    assert run(go())["features"]


def test_a_task_over_the_time_limit_is_killed_and_the_worker_replaced(monkeypatch):
    monkeypatch.setenv("LEJET_TEST_PARSE_DELAY", "3")      # every task takes >= 3 s in the workers
    async def go():
        pool = await ParsePool(workers=1, timeout=0.5).start()
        try:
            slow = await pool.analyze(**args(PAGES["plain"]))
            restarts_after_timeout = pool.restarts
            pool.timeout = 30
            fine = await pool.analyze(**args(PAGES["plain"]))          # served by the NEW worker
            return slow, restarts_after_timeout, fine
        finally:
            await pool.close()
    slow, restarts, fine = run(go())
    assert slow == {"error": "parse_timeout"}
    assert restarts == 1
    assert fine["features"] and "error" not in fine


def test_cancelling_a_caller_retires_the_busy_worker():
    async def go():
        pool = await ParsePool(workers=1, timeout=30).start()
        try:
            task = asyncio.create_task(pool.analyze(**args(big_html(1_900_000))))
            await asyncio.sleep(0.02)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            pool.timeout = 30
            out = await asyncio.wait_for(pool.analyze(**args(PAGES["plain"])), 30)
            return pool.restarts, out
        finally:
            await pool.close()
    restarts, out = run(go())
    assert restarts >= 1
    assert out["features"]


def test_a_late_answer_from_an_abandoned_call_is_never_mistaken_for_the_next_result(monkeypatch):
    monkeypatch.setenv("LEJET_TEST_PARSE_DELAY", "1.5")    # every task takes >= 1.5 s in the worker
    ctx = mp.get_context("spawn")
    worker = parse_pool._Worker(ctx)
    try:
        first = worker.call(args(PAGES["plain"]), 0.3)                 # gives up: worker still busy
        assert first is None
        second = worker.call(args(PAGES["html_all_internal"]), 60)     # must get ITS OWN answer
        assert second is not None
        assert second["features"]["nb_hyperlinks"] == 10
    finally:
        worker.stop()


def test_worker_survives_a_hostile_payload_error():
    async def go():
        pool = await ParsePool(workers=1, timeout=30).start()
        try:
            bad = await pool.analyze(body="this is a str, not bytes", content_type="text/html",
                                     base_url=BASE_URL, want_features=True)
            good = await pool.analyze(**args(PAGES["plain"]))
            return bad, good, pool.restarts
        finally:
            await pool.close()
    bad, good, restarts = run(go())
    assert "error" in bad
    assert good["features"] and restarts == 0


def test_close_is_clean_and_fast():
    async def go():
        pool = await ParsePool(workers=2, timeout=5).start()
        started = time.monotonic()
        await pool.close()
        return time.monotonic() - started
    assert run(go()) < 10
