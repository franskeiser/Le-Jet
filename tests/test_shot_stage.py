"""
Tests for the pure parts of the screenshot stage (src/lejet/stages/shot_stage.py): which rows get a
screenshot, which requests the browser may make, the sandbox guard, PNG writing, the tar archive and
the work queue. No browser is started here (the real-Chromium test is tests/test_shot_browser.py).

Run with:  python -m pytest tests/test_shot_stage.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tarfile

import pytest

from lejet import pngcheck
from lejet.stages import shot_stage
from lejet.stages.shot_stage import ShotResult, ShotRunner
from test_pngcheck import png


# ---------------------------------------------------------------------------------------
# which rows get a screenshot
# ---------------------------------------------------------------------------------------
def row(**kw):
    base = {"fail_reason": "none", "html_available": 1, "final_url": "https://example.com/"}
    base.update(kw)
    return base


def test_only_fetched_pages_with_a_final_url_are_selected():
    assert shot_stage.selectable(row())
    assert not shot_stage.selectable(row(fail_reason="http_4xx"))
    assert not shot_stage.selectable(row(fail_reason="challenge_page"))
    assert not shot_stage.selectable(row(html_available=0))
    assert not shot_stage.selectable(row(final_url=""))
    assert not shot_stage.selectable(row(final_url=None))
    assert not shot_stage.selectable(row(final_url=5))
    assert not shot_stage.selectable({})


@pytest.mark.parametrize("reason", ["rejected_url", "port_blocked", "redirect_to_blocked", "dns_nxdomain", "dns_no_answer"])
def test_urls_with_nothing_to_show_or_refused_by_the_gate_are_never_opened(reason):
    r = row(fail_reason=reason, html_available=0, final_url="", url="https://example.com/")
    assert shot_stage.capture_target(r) is None


def test_a_page_the_plain_fetch_loaded_is_opened_at_its_final_url():
    assert shot_stage.capture_target(row()) == ("https://example.com/", False)


@pytest.mark.parametrize("reason", ["http_4xx", "http_5xx", "challenge_page", "parked_or_default_page",
                                    "blocked_or_empty", "non_html", "too_large"])
def test_a_site_that_answered_the_bot_is_opened_where_it_answered(reason):
    r = row(fail_reason=reason, html_available=0, final_url="https://www.example.org/blocked", url="https://x.test/")
    assert shot_stage.capture_target(r) == ("https://www.example.org/blocked", False)


@pytest.mark.parametrize("reason", ["too_many_redirects", "connect_timeout", "connect_refused", "read_timeout",
                                    "dns_timeout", "dns_servfail"])
def test_other_failures_are_retried_from_the_original_url(reason):
    r = row(fail_reason=reason, html_available=0, final_url="https://loop.test/last-hop", url="http://orig.test/path")
    target = shot_stage.capture_target(r)
    assert target is not None and "orig.test" in target[0] and target[1] is False


def test_a_tls_failure_is_opened_with_certificate_errors_tolerated():
    r = row(fail_reason="tls_error", html_available=0, final_url="", url="https://badcert.test/")
    url, ignore_tls = shot_stage.capture_target(r)
    assert "badcert.test" in url and ignore_tls is True


def test_schemeless_raw_urls_get_a_scheme_from_the_normalizer():
    r = row(fail_reason="connect_timeout", html_available=0, final_url="", url="evil-login.example.xyz/verify")
    url, _ = shot_stage.capture_target(r)
    assert url.startswith(("https://", "http://")) and "evil-login.example.xyz" in url


@pytest.mark.parametrize("raw", ["http://10.0.0.5/", "file:///etc/passwd", "ftp://x.test/", "", None, 5,
                                 "http://example.com:8080/"])
def test_an_unsafe_original_url_is_never_opened_even_after_a_failure(raw):
    r = row(fail_reason="connect_timeout", html_available=0, final_url="", url=raw)
    assert shot_stage.capture_target(r) is None


def test_nothing_unsafe_is_ever_offered_as_a_final_url():
    # an HTTP answer's final_url is trusted only if it looks like a web URL; the filter still re-checks it
    r = row(fail_reason="http_4xx", html_available=0, final_url="javascript:alert(1)", url="https://ok.test/")
    target = shot_stage.capture_target(r)
    assert target is not None and target[0].startswith("https://ok.test")


# ---------------------------------------------------------------------------------------
# the request filter (the SSRF bouncer for everything the page asks for)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "https://example.com/", "http://example.com/a?b=c", "https://sub.example.co.uk/x#y",
    "https://example.com:443/", "http://example.com:80/", "wss://example.com/socket",
    "ws://example.com/socket", "data:image/png;base64,AAAA", "blob:https://example.com/uuid", "about:blank",
])
def test_ordinary_requests_are_allowed(url):
    assert shot_stage.request_allowed(url)[0], url


@pytest.mark.parametrize("url, why", [
    ("http://10.0.0.5/admin", "private address"),
    ("http://192.168.1.1/", "private address"),
    ("http://172.16.0.1/", "private address"),
    ("http://169.254.169.254/latest/meta-data/", "cloud metadata"),
    ("http://127.0.0.1/", "loopback"),
    ("http://localhost/", "loopback name"),
    ("http://[::1]/", "ipv6 loopback"),
    ("http://2130706433/", "decimal-encoded 127.0.0.1"),
    ("http://0x7f.0.0.1/", "hex-encoded 127.0.0.1"),
    ("http://10.77.0.1/", "the gateway itself"),
    ("http://example.com:8080/", "port 8080"),
    ("https://example.com:22/", "port 22"),
    ("http://user:pass@example.com/", "credentials in the URL"),
    ("ftp://example.com/file", "ftp"),
    ("file:///etc/passwd", "file"),
    ("javascript:alert(1)", "javascript"),
    ("chrome://settings", "chrome internal"),
    ("view-source:https://example.com", "view-source"),
    ("ws://10.0.0.5/socket", "websocket to a private address"),
    ("wss://example.com:9999/socket", "websocket on an odd port"),
    ("", "empty"),
    ("   ", "blank"),
])
def test_dangerous_requests_are_blocked(url, why):
    assert not shot_stage.request_allowed(url)[0], why


def test_loopback_is_allowed_only_in_test_mode_and_only_for_the_two_names():
    assert shot_stage.request_allowed("http://127.0.0.1:5555/x", allow_loopback=True)[0]
    assert shot_stage.request_allowed("http://localhost:5555/x", allow_loopback=True)[0]
    assert not shot_stage.request_allowed("http://10.0.0.5:5555/x", allow_loopback=True)[0]
    assert not shot_stage.request_allowed("http://127.0.0.2:5555/x", allow_loopback=True)[0]
    assert not shot_stage.request_allowed("http://127.0.0.1:5555/x", allow_loopback=False)[0]


def test_the_filter_never_raises_on_junk():
    for junk in (None, 5, "http://", "http://[", "http://%00", "\x00", "http://a b/", "https://" + "a" * 5000):
        try:
            ok = shot_stage.request_allowed(junk)[0]
        except Exception as exc:                                    # noqa: BLE001
            pytest.fail(f"raised {exc!r} for {junk!r}")
        assert ok is False


# ---------------------------------------------------------------------------------------
# the sandbox must never be switched off
# ---------------------------------------------------------------------------------------
def test_the_built_in_chromium_arguments_keep_the_sandbox_on():
    assert shot_stage.chromium_args_ok(shot_stage.CHROMIUM_ARGS)
    for a in shot_stage.CHROMIUM_ARGS:
        assert "sandbox" not in a and "single-process" not in a


@pytest.mark.parametrize("bad", ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu-sandbox",
                                 "--disable-seccomp-filter-sandbox", "--single-process", "--no-sandbox=1"])
def test_any_sandbox_off_switch_is_detected(bad):
    assert not shot_stage.chromium_args_ok(["--mute-audio", bad])


def test_a_runner_cannot_be_built_with_the_sandbox_off(monkeypatch, tmp_path):
    monkeypatch.setattr(shot_stage, "CHROMIUM_ARGS", ["--no-sandbox"])
    with pytest.raises(AssertionError):
        ShotRunner(str(tmp_path))


# ---------------------------------------------------------------------------------------
# PNG writing and the tar archive
# ---------------------------------------------------------------------------------------
def test_a_valid_png_is_written_atomically(tmp_path):
    path = str(tmp_path / "0123456789abcdef.png")
    ok, err, w, h = shot_stage.write_png(path, png(10, 4))
    assert (ok, err, w, h) == (True, "none", 10, 4)
    assert open(path, "rb").read() == png(10, 4)
    assert not os.path.exists(path + ".part")


@pytest.mark.parametrize("data", [b"", b"<html>not a png</html>", png()[:-5], png() + b"x"])
def test_an_invalid_png_is_never_written(tmp_path, data):
    path = str(tmp_path / "a.png")
    ok, err, _, _ = shot_stage.write_png(path, data)
    assert not ok and err == "png_invalid"
    assert not os.path.exists(path) and not os.path.exists(path + ".part")


def results_and_files(tmp_path):
    good = ShotResult("0123456789abcdef", ok=True, error="none", bytes=len(png(8, 8)), sha256="x", width=8, height=8)
    bad = ShotResult("fedcba9876543210", ok=False, error="goto_timeout")
    (tmp_path / f"{good.url_id}.png").write_bytes(png(8, 8))
    return [good, bad]


def test_the_tar_holds_only_successful_pngs_and_the_manifest(tmp_path):
    results = results_and_files(tmp_path)
    out = str(tmp_path / "b1.shots.tar")
    assert shot_stage.pack_tar(str(tmp_path), results, out) == 1
    with tarfile.open(out) as tar:
        names = sorted(tar.getnames())
        assert names == ["0123456789abcdef.png", "shots.jsonl"]
        assert all(m.isfile() and m.uid == 0 and m.gid == 0 and m.mtime == 0 for m in tar.getmembers())
        lines = [json.loads(l) for l in tar.extractfile("shots.jsonl").read().decode().splitlines()]
        assert [l["url_id"] for l in lines] == ["0123456789abcdef", "fedcba9876543210"]
        assert lines[1]["error"] == "goto_timeout" and lines[1]["ok"] is False
        data = tar.extractfile("0123456789abcdef.png").read()
        assert pngcheck.check_png(data)[0]


def test_the_tar_is_deterministic(tmp_path):
    results = results_and_files(tmp_path)
    a, b = str(tmp_path / "a.tar"), str(tmp_path / "b.tar")
    shot_stage.pack_tar(str(tmp_path), results, a)
    shot_stage.pack_tar(str(tmp_path), results, b)
    assert open(a, "rb").read() == open(b, "rb").read()


def test_error_codes_are_a_closed_list():
    assert "none" in shot_stage.ERRORS and len(set(shot_stage.ERRORS)) == len(shot_stage.ERRORS)
    for r in (ShotResult("a" * 16),):
        assert r.error in shot_stage.ERRORS


# ---------------------------------------------------------------------------------------
# the work queue (a fake `shoot`: no browser)
# ---------------------------------------------------------------------------------------
class FakeRunner(ShotRunner):
    def __init__(self, tmp_path, concurrency, delay=0.01):
        super().__init__(str(tmp_path), concurrency=concurrency)
        self.delay, self.active, self.peak, self.seen = delay, 0, 0, []

    async def shoot(self, url_id, url, ignore_tls=False):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        self.seen.append(url_id)
        return ShotResult(url_id, ok=not url.endswith("/bad"), error="none" if not url.endswith("/bad") else "goto_timeout")


def test_every_item_gets_exactly_one_result_in_input_order(tmp_path):
    items = [(f"{i:016x}", f"https://e{i}.test/" + ("bad" if i % 3 == 0 else "ok")) for i in range(12)]
    runner = FakeRunner(tmp_path, concurrency=3)
    results = asyncio.run(runner.run(items))
    assert [r.url_id for r in results] == [u for u, _ in items]
    assert sorted(runner.seen) == sorted(u for u, _ in items)
    assert [r.ok for r in results] == [i % 3 != 0 for i in range(12)]


def test_concurrency_limit_is_respected(tmp_path):
    items = [(f"{i:016x}", "https://x.test/ok") for i in range(10)]
    r1, r3 = FakeRunner(tmp_path, 1), FakeRunner(tmp_path, 3)
    asyncio.run(r1.run(items))
    asyncio.run(r3.run(items))
    assert r1.peak == 1 and r3.peak == 3


def test_progress_callback_reaches_the_total(tmp_path):
    items = [(f"{i:016x}", "https://x.test/ok") for i in range(4)]
    seen = []
    asyncio.run(FakeRunner(tmp_path, 2).run(items, lambda d, t: seen.append((d, t))))
    assert seen[-1] == (4, 4) and len(seen) == 4


def test_an_empty_batch_is_fine(tmp_path):
    assert asyncio.run(FakeRunner(tmp_path, 2).run([])) == []


def test_blocked_start_urls_never_open_a_browser(tmp_path):
    runner = ShotRunner(str(tmp_path))
    for url in ("http://10.0.0.5/", "file:///etc/passwd", "ftp://example.com/", "http://example.com:8080/"):
        res = asyncio.run(runner.shoot("0123456789abcdef", url))
        assert res.error == "blocked_start" and not res.ok
