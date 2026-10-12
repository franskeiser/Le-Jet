"""
Real-browser tests of the screenshot stage. They need Playwright AND an installed Chromium, so they
run inside the analysis VM after infra/analysis/install_browser.sh, and are SKIPPED everywhere else
(your PC, or the VM before the browser is installed). The pages come from the local fixture server
(tests/conftest.py); no test touches the internet.

Run (in the VM):  python -I -m pytest tests/test_shot_browser.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import tarfile

import pytest

pytest.importorskip("playwright.async_api")

from lejet import pngcheck, schema                                   # noqa: E402
from lejet.stages import shot_stage                                  # noqa: E402
from lejet.stages.shot_stage import ShotRunner                       # noqa: E402


@pytest.fixture(scope="module")
def chromium_works():
    """Skip the whole module unless Chromium really launches with its sandbox on."""
    from lejet import browser_selftest
    try:
        info = asyncio.run(browser_selftest._run())
    except Exception as exc:                                         # noqa: BLE001
        pytest.skip(f"Chromium is not available here: {type(exc).__name__}")
    if not (info["launched"] and info["png_ok"]):
        pytest.skip("Chromium did not produce a valid screenshot here")
    return info


def run(coro):
    return asyncio.run(coro)


def shoot_one(tmp_path, url, **kw):
    async def go():
        async with ShotRunner(str(tmp_path), allow_loopback=True, **kw) as runner:
            return await runner.shoot("0123456789abcdef", url)
    return run(go())


def test_the_sandbox_is_really_on(chromium_works):
    assert chromium_works["sandbox_on"] is True and chromium_works["processes"] >= 1


def test_a_normal_page_gives_a_valid_png_of_the_right_size(chromium_works, server, tmp_path):
    res = shoot_one(tmp_path, server.url("/page/plain"))
    assert res.ok and res.error == "none"
    data = (tmp_path / "0123456789abcdef.png").read_bytes()
    ok, why, w, h = pngcheck.check_png(data)
    assert ok, why
    assert (w, h) == (1280, 800) and res.width == 1280 and res.height == 800 and res.bytes == len(data)


def test_the_screenshot_really_shows_the_page(chromium_works, server, tmp_path):
    """Two different pages must give two different pictures (not a blank or a cached image)."""
    a = shoot_one(tmp_path, server.url("/page/plain"))
    first = (tmp_path / "0123456789abcdef.png").read_bytes()
    b = shoot_one(tmp_path, server.url("/page/html_media_mix"))
    second = (tmp_path / "0123456789abcdef.png").read_bytes()
    assert a.ok and b.ok and a.sha256 != b.sha256 and first != second


def test_a_redirect_to_a_private_address_is_blocked(chromium_works, server, tmp_path):
    res = shoot_one(tmp_path, server.url("/to-private"))
    assert not res.ok and res.error == "navigation_error"
    assert res.blocked_requests >= 1 and res.ms < 10000               # refused fast, and counted
    assert not (tmp_path / "0123456789abcdef.png").exists()
    assert not any(h[0] == "/secret" for h in server.hits)             # nothing ever asked the private host


def test_a_meta_refresh_to_a_private_address_is_blocked_and_counted(chromium_works, server, tmp_path):
    res = shoot_one(tmp_path, server.url("/meta-private"))
    assert res.blocked_requests >= 1


def test_a_redirect_to_the_metadata_address_is_blocked(chromium_works, server, tmp_path):
    res = shoot_one(tmp_path, server.url("/to-metadata"))
    assert not res.ok and res.error == "navigation_error"


def test_a_redirect_to_ftp_or_a_odd_port_is_blocked(chromium_works, server, tmp_path):
    for path in ("/to-ftp", "/to-port"):
        res = shoot_one(tmp_path, server.url(path))
        assert not res.ok, path


def test_a_page_that_never_finishes_still_gets_a_screenshot_or_a_clean_failure(chromium_works, server, tmp_path):
    res = shoot_one(tmp_path, server.url("/slow"), page_deadline=40.0)
    assert res.error in ("none", "page_deadline", "goto_timeout", "navigation_error", "screenshot_error")
    if res.ok:
        assert pngcheck.check_png((tmp_path / "0123456789abcdef.png").read_bytes())[0]


def test_the_deadline_is_enforced(chromium_works, server, tmp_path):
    res = shoot_one(tmp_path, server.url("/slow"), page_deadline=3.0)
    assert not res.ok and res.error == "page_deadline"
    assert res.ms < 15000


def test_dialogs_popups_downloads_and_private_requests_do_not_hang_or_get_through(chromium_works, server, tmp_path):
    res = shoot_one(tmp_path, server.url("/popups"))
    assert res.ok and res.ms < 30000                                 # alert/confirm/prompt did not freeze it
    assert res.blocked_requests >= 1                                 # fetch / WebSocket to 10.0.0.5 refused
    assert not any(f.endswith(".bin") for f in os.listdir(tmp_path))  # the automatic download went nowhere


def test_a_crashed_browser_is_replaced_for_the_next_page(chromium_works, server, tmp_path):
    async def go():
        async with ShotRunner(str(tmp_path), allow_loopback=True) as runner:
            first = await runner.shoot("0123456789abcdef", server.url("/page/plain"))
            await runner._browser.close()                              # simulate a dead browser
            second = await runner.shoot("fedcba9876543210", server.url("/page/plain"))
            return first, second, runner.restarts
    first, second, restarts = run(go())
    assert first.ok and second.ok and restarts == 1


def test_many_pages_with_two_at_a_time(chromium_works, server, tmp_path):
    items = [(f"{i:016x}", server.url("/page/plain")) for i in range(5)]

    async def go():
        async with ShotRunner(str(tmp_path), allow_loopback=True, concurrency=2) as runner:
            return await runner.run(items)
    results = run(go())
    assert [r.url_id for r in results] == [u for u, _ in items] and all(r.ok for r in results)


def test_cli_screenshots_end_to_end(chromium_works, server, tmp_path):
    """db with two rows (one fetched page, one failure) -> tar + sha256 with exactly one screenshot."""
    from lejet import cli
    from lejet.storage import Storage
    db = str(tmp_path / "b1.sqlite")
    store = Storage(db)
    store.start_run("b1", "0" * 64, 2, {})
    good = schema.new_row("0123456789abcdef", server.url("/page/plain"), "b1")
    good.update(fail_reason="none", html_available=1, final_url=server.url("/page/plain"))
    blocked = schema.new_row("fedcba9876543210", server.url("/404"), "b1")      # the plain fetch got a 404 page
    blocked.update(fail_reason="http_4xx", html_available=0, final_url=server.url("/404"))
    gone = schema.new_row("aaaaaaaaaaaaaaaa", "http://no-such-name.invalid/", "b1")   # DNS: nothing to show
    gone.update(fail_reason="dns_nxdomain", html_available=0)
    for r in (good, blocked, gone):
        store.save(r)
    store.close()
    out = tmp_path / "out"
    rc = cli.main(["screenshots", "--db", db, "--out-dir", str(out), "--batch-id", "b1", "--allow-loopback"])
    assert rc == 0
    tar_path = out / "b1.shots.tar"
    assert tar_path.exists() and (out / "b1.shots.tar.sha256").exists()
    with tarfile.open(tar_path) as tar:
        # the page the bot could not load is photographed too; the name that does not exist is not attempted
        assert sorted(tar.getnames()) == ["0123456789abcdef.png", "fedcba9876543210.png", "shots.jsonl"]
        lines = [json.loads(l) for l in tar.extractfile("shots.jsonl").read().decode().splitlines()]
    assert len(lines) == 2 and all(l["ok"] for l in lines)
