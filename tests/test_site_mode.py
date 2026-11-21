"""
Tests for the "whole site" mode of the screenshot stage (used by the live tool on the cloud VM):
which same-site links are followed, the retry ladder, reason codes, the extended manifest line, the tar with
linked pages, and that the strict host validator accepts exactly that and nothing more.
No browser is started here.

Run with:  python -m pytest tests/test_site_mode.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os

import pytest

from lejet.stages import shot_stage
from lejet.stages.shot_stage import ShotResult, ShotRunner
from test_pngcheck import png

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("validate_shots", os.path.join(HERE, "..", "scripts", "validate_shots.py"))
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)

A = "0123456789abcdef"


# ---------------------------------------------------------------------------------------
# links worth following
# ---------------------------------------------------------------------------------------
def test_only_same_site_pages_are_followed_and_at_most_four():
    base = "https://www.shop.example/home"
    hrefs = ["/a", "/b", "https://shop.example/c", "https://evil.example/x", "/d", "/e", "/f"]
    got = shot_stage.pick_links(base, hrefs)
    assert got == ["https://www.shop.example/a", "https://www.shop.example/b", "https://shop.example/c",
                   "https://www.shop.example/d"]


@pytest.mark.parametrize("href", ["mailto:a@b.c", "tel:+63917", "sms:+63917", "javascript:alert(1)", "#top",
                                  "/files/app.apk", "/doc.pdf", "/logout", "/user/sign-out", "/unsubscribe?x=1",
                                  "ftp://shop.example/x", "", "   ", None, 5, "/" + "a" * 3000])
def test_unwanted_links_are_skipped(href):
    assert shot_stage.pick_links("https://shop.example/", [href]) == []


def test_the_landing_page_itself_and_duplicates_are_not_followed_twice():
    got = shot_stage.pick_links("https://shop.example/home?x=1",
                                ["/home?x=1", "/p#one", "/p#two", "https://shop.example/p"])
    assert got == ["https://shop.example/p"]


def test_links_to_private_addresses_are_refused_even_on_the_same_host():
    assert shot_stage.pick_links("http://10.0.0.5/", ["/a"]) == []
    assert shot_stage.pick_links("https://shop.example/", ["http://169.254.169.254/latest"]) == []


# ---------------------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------------------
def test_the_ladder_starts_with_the_normal_visit_then_gets_more_patient():
    steps = shot_stage.ladder_steps("https://shop.example/x?a=1", True)
    assert steps[0] == ("https://shop.example/x?a=1", True, False)
    assert steps[1] == ("https://shop.example/x?a=1", False, True)               # other device, longer waits
    assert ("http://shop.example/x?a=1", True, True) in steps                    # other scheme
    assert ("https://www.shop.example/x?a=1", True, True) in steps               # www toggle
    assert len(steps) <= shot_stage.LADDER_MAX_ATTEMPTS and len(set(steps)) == len(steps)


def test_the_ladder_does_not_invent_www_for_an_ip_address():
    steps = shot_stage.ladder_steps("http://203.0.113.9/p", False)
    assert all("www." not in s[0] for s in steps)


@pytest.mark.parametrize("text, refused, want", [
    ("net::ERR_NAME_NOT_RESOLVED at http://x", False, "nxdomain"),
    ("net::ERR_CONNECTION_REFUSED", False, "refused"),
    ("net::ERR_CONNECTION_RESET", False, "reset"),
    ("net::ERR_EMPTY_RESPONSE", False, "reset"),
    ("Timeout 20000ms exceeded", False, "timeout"),
    ("net::ERR_CERT_AUTHORITY_INVALID", False, "tls"),
    ("net::ERR_CONNECTION_TIMED_OUT", False, "timeout"),
    ("net::ERR_TUNNEL_CONNECTION_FAILED at https://x/", False, "unreachable"),
    ("net::ERR_ADDRESS_UNREACHABLE", False, "unreachable"),
    ("something odd", False, "other"),
    ("net::ERR_NAME_NOT_RESOLVED", True, "blocked"),
])
def test_navigation_errors_become_a_short_closed_reason(text, refused, want):
    got = shot_stage.classify_nav_error(text, refused)
    assert got == want and got in shot_stage.REASONS


class FakeRunner(ShotRunner):
    """A runner whose attempts come from a script instead of a browser."""

    def __init__(self, script, **kw):
        super().__init__("unused", **kw)
        self.script, self.calls = list(script), []

    async def _attempt(self, url_id, url, ignore_tls, mobile=None, slow=False):
        self.calls.append((url, mobile, slow))
        ok, reason, size, text = self.script.pop(0)
        return ShotResult(url_id, ok=ok, error="none" if ok else "navigation_error", bytes=size,
                          text_chars=text, reason=reason)


def run(coro):
    return asyncio.run(coro)


def test_a_site_that_opens_on_the_first_try_is_live():
    r = FakeRunner([(True, "ok", 50_000, 400)], ladder=True, mobile=True)
    res = run(r.shoot(A, "https://shop.example/"))
    assert (res.reach, res.attempts, res.reason, res.extended) == ("live", 1, "ok", True) and len(r.calls) == 1


def test_a_site_that_needs_the_other_profile_is_live_retry():
    r = FakeRunner([(False, "timeout", 0, -1), (True, "ok", 60_000, 300)], ladder=True, mobile=True)
    res = run(r.shoot(A, "https://shop.example/"))
    assert (res.reach, res.attempts) == ("live_retry", 2)
    assert r.calls[1][1:] == (False, True)                                      # desktop, patient


def test_a_blank_white_page_is_not_good_enough_and_the_next_rung_is_tried():
    r = FakeRunner([(True, "ok", 4_000, 0), (True, "ok", 80_000, 500)], ladder=True, mobile=True)
    res = run(r.shoot(A, "https://shop.example/"))
    assert res.reach == "live_retry" and res.bytes == 80_000


def test_a_name_that_does_not_exist_stops_the_ladder_at_once():
    r = FakeRunner([(False, "nxdomain", 0, -1)], ladder=True, mobile=True)
    res = run(r.shoot(A, "https://gone.example/"))
    assert (res.reach, res.reason, res.attempts, res.ok) == ("dead", "nxdomain", 1, False) and len(r.calls) == 1


def test_when_every_rung_fails_the_result_is_dead_with_a_reason():
    script = [(False, "timeout", 0, -1)] * shot_stage.LADDER_MAX_ATTEMPTS
    r = FakeRunner(script, ladder=True, mobile=True)
    res = run(r.shoot(A, "https://shop.example/"))
    assert res.reach == "dead" and res.reason in shot_stage.REASONS and not res.ok
    assert res.attempts == len(r.calls) <= shot_stage.LADDER_MAX_ATTEMPTS


def test_without_the_new_options_nothing_changes_for_the_dataset_lab():
    r = ShotRunner("unused")
    assert not r.extended and not r.full_page and r.max_pages == 0 and not r.ladder
    d = ShotResult(A).as_dict()
    assert set(d) == set(vs.EXPECTED_KEYS)                                       # exactly the old manifest line


def test_max_pages_is_capped():
    assert ShotRunner("x", max_pages=99).max_pages == shot_stage.MAX_LINKED_PAGES
    assert ShotRunner("x", max_pages=-3).max_pages == 0


# ---------------------------------------------------------------------------------------
# the archive with linked pages, through the strict validator
# ---------------------------------------------------------------------------------------
def _site_result(tmp_path, pages=2):
    main = png(8, 8)
    (tmp_path / f"{A}.png").write_bytes(main)
    res = ShotResult(A, ok=True, error="none", bytes=len(main), sha256=hashlib.sha256(main).hexdigest(), width=8,
                     height=8, extended=True, reach="live", reason="ok", attempts=1, text_chars=50)
    for n in range(1, pages + 1):
        data = png(8 + n, 8)
        (tmp_path / f"{A}_p{n}.png").write_bytes(data)
        res.pages.append({"n": n, "url": f"hxxps://shop[.]example/p{n}", "title": f"Page {n}", "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest(), "width": 8 + n, "height": 8,
                          "text_chars": 10, "truncated": False})
    return res


def _pack(tmp_path, res):
    tar = str(tmp_path / "b1.shots.tar")
    shot_stage.pack_tar(str(tmp_path), [res], tar)
    open(tar + ".sha256", "w").write(f"{hashlib.sha256(open(tar, 'rb').read()).hexdigest()}  b1.shots.tar\n")
    return tar


def test_a_whole_site_archive_passes_the_strict_validator(tmp_path):
    tar = _pack(tmp_path, _site_result(tmp_path))
    problems, pngs, manifest = vs.validate(tar, tar + ".sha256", {A})
    assert problems == []
    assert sorted(pngs) == [f"{A}.png", f"{A}_p1.png", f"{A}_p2.png"]
    assert json.loads(manifest.decode().splitlines()[0])["reach"] == "live"


def test_a_page_png_that_the_manifest_does_not_list_is_refused(tmp_path):
    res = _site_result(tmp_path)
    res.pages.pop()                                                             # p2.png exists on disk but is not listed
    (tmp_path / f"{A}_p2.png").write_bytes(png(10, 8))
    tar = _pack(tmp_path, res)
    problems, pngs, _ = vs.validate(tar, tar + ".sha256", {A})
    assert f"{A}_p2.png" not in pngs                                            # ...so the unlisted one never arrives


def test_a_listed_page_whose_png_differs_from_the_manifest_is_refused(tmp_path):
    res = _site_result(tmp_path)
    res.pages[0]["sha256"] = "0" * 64
    tar = _pack(tmp_path, res)
    problems, pngs, _ = vs.validate(tar, tar + ".sha256", {A})
    assert problems and pngs == {}


@pytest.mark.parametrize("change", [
    {"reach": "archived_wayback"}, {"reach": "somewhere"}, {"reason": "its-complicated"}, {"attempts": 999},
    {"pages": [{"n": 1}]}, {"pages": "x"},
])
def test_extended_manifest_values_outside_the_closed_lists_are_refused(change):
    line = _site_result_line(**change)
    assert vs.check_manifest_line(line) is not None


def _site_result_line(**kw):
    rec = {"url_id": A, "ok": False, "error": "navigation_error", "bytes": 0, "sha256": "", "width": 0, "height": 0,
           "partial": False, "blocked_requests": 0, "ms": 5, "final_url": "", "title": "", "text_chars": -1,
           "rescued": False, "reach": "dead", "reason": "nxdomain", "truncated": False, "attempts": 1, "pages": []}
    rec.update(kw)
    return rec


def test_a_valid_extended_line_is_accepted_and_a_half_extended_one_is_not():
    assert vs.check_manifest_line(_site_result_line()) is None
    half = _site_result_line()
    del half["reach"]
    assert vs.check_manifest_line(half) is not None


def test_more_than_four_linked_pages_are_refused():
    pages = [{"n": n, "url": "", "title": "", "bytes": 1, "sha256": "a" * 64, "width": 1, "height": 1,
              "text_chars": 0, "truncated": False} for n in range(1, 6)]
    assert vs.check_manifest_line(_site_result_line(pages=pages)) is not None
