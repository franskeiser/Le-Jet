"""
The four content features of the public 87-feature set that the original 12 webpage features do not cover and that the
lab exports (lejet.features.hannousse_html): popup_window, safe_anchor, onmouseover, right_clic. They are measured by
analyze_page in the same walk as the original twelve.

No network: every page is a small fixture with a hand-checked expected value. The public set contains no pages, so
(unlike the URL features) these cannot be compared with it row by row; these tests pin the definitions.
"""
from __future__ import annotations

import pytest

from fixtures.pages import BASE_URL, PAGES, anchors, page
from lejet import schema
from lejet.features import hannousse_html as X
from lejet.features import html_features as hf


def extra(body: bytes) -> dict:
    out = hf.analyze_page(body, "text/html; charset=utf-8", BASE_URL)
    assert out["features"], out
    return out["features_extra"]


def test_names_and_counts():
    assert len(X.CONTENT_FEATURES) == 16 and len(set(X.CONTENT_FEATURES)) == 16
    assert X.CONTENT_FEATURES == [n for n in schema.FULL_FEATURES if n in X.CONTENT_FEATURES]     # public set's order
    assert X.EXTRA_CONTENT_FEATURES == ["popup_window", "safe_anchor", "onmouseover", "right_clic"]
    assert set(X.EXTRA_CONTENT_FEATURES) | set(hf.WEBPAGE_FEATURES) == set(X.CONTENT_FEATURES)
    assert not set(X.EXTRA_CONTENT_FEATURES) & set(hf.WEBPAGE_FEATURES)


def test_the_original_twelve_are_unchanged_by_collecting_the_facts():
    for name in ("plain", "html_mixed_links", "html_ext_css", "html_media_mix", "html_login_ext_action"):
        with_facts = hf.analyze_page(PAGES[name], "text/html", BASE_URL)["features"]
        root = hf.parse_document(hf.decode_html(PAGES[name], "text/html"))
        without = hf.extract_features(root, BASE_URL, "example.com", "www.example.com", "example", [])[0]
        assert with_facts == without


def test_only_the_four_exported_features_are_computed():
    e = extra(PAGES["html_mixed_links"])
    assert sorted(e) == sorted(X.EXTRA_CONTENT_FEATURES)
    for name in ("ratio_extHyperlinks", "ratio_nullHyperlinks", "submit_email", "sfh", "ratio_intRedirection",
                 "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors"):
        assert name not in e and name not in schema.FULL_FEATURES and name in schema.DROPPED_FROM_FULL


def test_a_page_without_anchors_has_zero_safe_anchor():
    assert extra(PAGES["plain"])["safe_anchor"] == 0.0


def test_safe_anchor_is_the_percentage_of_anchors_to_hash_javascript_or_mailto():
    body = page("t", anchors(["/a", "#", "javascript:void(0)", "MAILTO:x@y.z", "/b#top", "https://o.org/"]))
    assert extra(body)["safe_anchor"] == pytest.approx(100 * 4 / 6)      # '#', javascript, mailto and '/b#top'
    assert extra(page("t", anchors(["#", "#x"])))["safe_anchor"] == 100.0
    assert extra(page("t", anchors(["/a", "/b"])))["safe_anchor"] == 0.0


def test_popup_window_needs_a_prompt_or_a_window_open_with_a_text_field():
    assert extra(page("t", "<script>var a = prompt('your password?');</script>"))["popup_window"] == 1
    assert extra(page("t", "<script>window.open('x.html')</script><input type=text name=q>"))["popup_window"] == 1
    assert extra(page("t", "<script>window.open('x.html')</script><p>no field</p>"))["popup_window"] == 0
    assert extra(page("t", "<script>alert('hi')</script>"))["popup_window"] == 0
    assert extra(page("t", '<script src="https://x.org/prompt(.js"></script>'))["popup_window"] == 0   # external: not read


def test_onmouseover_and_right_click():
    assert extra(page("t", '<a href="/x" onmouseover="window.status=\'x\'">a</a>'))["onmouseover"] == 1
    assert extra(page("t", '<a href="/x">a</a>'))["onmouseover"] == 0
    assert extra(page("t", "<script>document.onmousedown=function(){ if (event.button == 2) return false; }</script>"))["right_clic"] == 1
    assert extra(page("t", '<div oncontextmenu="if(event.button===2){return false}">x</div>'))["right_clic"] == 1
    assert extra(page("t", "<script>var a = 1;</script>"))["right_clic"] == 0


def test_hostile_script_text_cannot_make_it_slow():
    import time
    # 10 MB in ONE script is right at the HTML parser's own text-node limit: whether the parser keeps or drops that
    # text depends on the libxml2 build (Windows kept it, the Linux VM drops it). What must hold everywhere: it is
    # fast, it does not crash, and the answer is a valid flag.
    big = "<script>" + ("prompt(" * 5 + "x" * 10_000_000) + "</script>"
    start = time.monotonic()
    e = extra(page("t", big))
    assert time.monotonic() - start < 3
    assert e["popup_window"] in (0, 1)
    # a script well under every parser limit is read, and only its first 32 KB are searched
    start = time.monotonic()
    assert extra(page("t", "<script>prompt(" + "x" * 1_000_000 + "</script>"))["popup_window"] == 1
    assert extra(page("t", "<script>" + "x" * 100_000 + "prompt(</script>"))["popup_window"] == 0   # beyond the cap
    assert time.monotonic() - start < 3


def test_no_features_means_no_extra_features_for_a_challenge_page():
    out = hf.analyze_page(b"<html><head><title>Just a moment...</title></head><body>x</body></html>", "text/html", BASE_URL)
    assert out["features"] == {} and out["features_extra"] == {}
