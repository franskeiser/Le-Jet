"""
Level 0 tests for the 12 static HTML features, the redirect hints and the page verdicts
(WEB_INSPECTION_PLAN.md sections 13.5b, 14.0.3, 16). No network: every page is a fixture with a
known expected value (tests/fixtures/pages.py).

Run with:  python -m pytest tests/test_html_features.py
"""

from __future__ import annotations

import time

import pytest

from fixtures.pages import BASE_URL, PAGES, anchors, big_html, malformed, page
from lejet.features import html_features as hf

WEBPAGE_FEATURES = hf.WEBPAGE_FEATURES


def analyze(name_or_bytes, base=BASE_URL, ctype="text/html; charset=utf-8"):
    body = PAGES[name_or_bytes] if isinstance(name_or_bytes, str) else name_or_bytes
    return hf.analyze_page(body, ctype, base)


def feats(name_or_bytes, **kw):
    out = analyze(name_or_bytes, **kw)
    assert out["features"], f"no features for {name_or_bytes!r}: {out}"
    return out["features"]


# ---------------------------------------------------------------------------------------
# link classification
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("href, expected", [
    ("/a", "internal"), ("b.html", "internal"), ("../c", "internal"), ("?q=1", "internal"),
    ("https://www.example.com/x", "internal"), ("https://shop.example.com/", "internal"),
    ("//cdn.example.com/x.js", "internal"), ("mailto:me@example.org", "internal"),
    ("tel:+639171234567", "internal"), ("data:image/png;base64,AAAA", "internal"),
    ("https://other.org/", "external"), ("//third.io/z", "external"),
    ("http://example.org/", "external"), ("https://example.com.evil.net/", "external"),
    ("https://evil-example.com/", "external"), ("http://127.0.0.1/x", "external"),
    ("", "null"), ("#", "null"), ("about:blank", "null"), ("javascript:void(0)", "null"),
    ("JavaScript:alert(1)", "null"), ("  #  ", "null"), (None, "null"),
])
def test_classify_link(href, expected):
    assert hf.classify_link(href, BASE_URL, "example.com", "www.example.com") == expected


def test_classify_link_with_ip_base_compares_hosts():
    assert hf.classify_link("/x", "http://203.0.114.7/a", "", "203.0.114.7") == "internal"
    assert hf.classify_link("http://203.0.114.7/y", "http://203.0.114.7/a", "", "203.0.114.7") == "internal"
    assert hf.classify_link("http://203.0.114.8/y", "http://203.0.114.7/a", "", "203.0.114.7") == "external"


def test_classify_link_private_suffix_is_its_own_domain():
    base = "https://evil.pages.dev/"
    assert hf.classify_link("https://other.pages.dev/x", base, "evil.pages.dev", "evil.pages.dev") == "external"
    assert hf.classify_link("https://evil.pages.dev/x", base, "evil.pages.dev", "evil.pages.dev") == "internal"


# ---------------------------------------------------------------------------------------
# the 12 features
# ---------------------------------------------------------------------------------------
def test_all_features_present_and_typed():
    f = feats("plain")
    assert list(f) == WEBPAGE_FEATURES
    for name, value in f.items():
        assert isinstance(value, (int, float)), name


def test_nb_hyperlinks_and_internal_ratio_all_internal():
    f = feats("html_all_internal")
    assert f["nb_hyperlinks"] == 10
    assert f["ratio_intHyperlinks"] == 1.0


def test_mixed_links_null_links_count_as_internal():
    f = feats("html_mixed_links")
    assert f["nb_hyperlinks"] == 10
    # 5 internal + 2 null (counted as internal, documented) out of 10
    assert f["ratio_intHyperlinks"] == pytest.approx(0.7)


def test_no_links_is_zero_not_nan():
    f = feats(page("T", "<p>just text just text just text just text just text</p>"))
    assert f["nb_hyperlinks"] == 0
    assert f["ratio_intHyperlinks"] == 0.0
    assert f["links_in_tags"] == 0.0
    assert f["ratio_intMedia"] == 0.0 and f["ratio_extMedia"] == 0.0


def test_external_css_counted():
    assert feats("html_ext_css")["nb_extCSS"] == 2


@pytest.mark.parametrize("name, expected", [
    ("html_login_ext_action", 1),       # form posts to another domain
    ("html_login_empty_action", 1),     # action=""
    ("html_login_hash_action", 1),      # action="#"
    ("html_login_blank_action", 1),     # action="about:blank"
    ("html_login_php_action", 1),       # internal bare name.php (the dataset's rule)
    ("html_login_normal_action", 0),    # ordinary internal form
    ("html_form_no_action", 0),         # forms without an action attribute are not counted
    ("plain", 0),
])
def test_login_form(name, expected):
    assert feats(name)["login_form"] == expected


@pytest.mark.parametrize("name, expected", [
    ("html_ext_favicon", 1), ("html_int_favicon", 0), ("html_icon_in_body", 0), ("plain", 0),
])
def test_external_favicon(name, expected):
    assert feats(name)["external_favicon"] == expected


def test_media_ratios_are_percentages():
    f = feats("html_media_mix")
    assert f["ratio_intMedia"] == pytest.approx(66.666667, abs=1e-5)   # 4 of 6
    assert f["ratio_extMedia"] == pytest.approx(33.333333, abs=1e-5)   # 2 of 6


@pytest.mark.parametrize("name, expected", [
    ("html_hidden_iframe", 1), ("html_display_none_iframe", 1), ("html_visible_iframe", 0), ("plain", 0),
])
def test_hidden_iframe(name, expected):
    assert feats(name)["iframe"] == expected


@pytest.mark.parametrize("name, empty, domain_in_title", [
    ("html_no_title", 1, 1),             # no <title>: empty, and the domain is not in it
    ("html_empty_title", 1, 1),
    ("html_domain_in_title", 0, 0),      # direction as in the public set: 0 = domain IS in the title
    ("html_domain_not_in_title", 0, 1),
])
def test_title_features(name, empty, domain_in_title):
    f = feats(name)
    assert f["empty_title"] == empty
    assert f["domain_in_title"] == domain_in_title


@pytest.mark.parametrize("name, expected", [
    ("html_copyright_domain", 0),        # the domain name sits next to the mark
    ("html_copyright_other", 1),         # a mark exists, the domain name is not near it
    ("html_no_copyright", 0),            # no mark at all
    ("html_copyright_far", 1),           # the domain is mentioned, but more than 50 characters away
])
def test_domain_with_copyright(name, expected):
    assert feats(name)["domain_with_copyright"] == expected


def test_links_in_tags_share():
    assert feats("html_meta_script_link_links")["links_in_tags"] == pytest.approx(75.0)


def test_title_label_uses_dataset_extractor_not_private_suffixes():
    # evil.pages.dev: the dataset's 'domain' is "pages" (private suffixes off), so a title that
    # contains "evil" but not "pages" still counts as "domain not in title" (1)
    out = hf.analyze_page(page("Evil site", "<p>x</p>"), "text/html", "https://evil.pages.dev/")
    assert out["features"]["domain_in_title"] == 1
    out = hf.analyze_page(page("Hosted on pages", "<p>x</p>"), "text/html", "https://evil.pages.dev/")
    assert out["features"]["domain_in_title"] == 0


def test_features_judge_internal_against_the_pages_own_domain():
    body = page("T", anchors(["https://evil.net/a", "https://evil.net/b", "/c"]))
    on_example = hf.analyze_page(body, "text/html", "https://www.example.com/")["features"]
    on_evil = hf.analyze_page(body, "text/html", "https://evil.net/")["features"]
    assert on_example["ratio_intHyperlinks"] == pytest.approx(1 / 3, abs=1e-5)
    assert on_evil["ratio_intHyperlinks"] == 1.0


# ---------------------------------------------------------------------------------------
# helpers and near-empty pages
# ---------------------------------------------------------------------------------------
def test_js_shell_is_flagged_near_empty():
    out = analyze("html_empty")
    assert out["is_html"] and out["features"]
    assert out["script_count"] == 2
    assert out["near_empty"] == 1
    assert out["html_bytes"] == len(PAGES["html_empty"])


def test_normal_page_is_not_near_empty():
    out = analyze("plain")
    assert out["near_empty"] == 0
    assert out["script_count"] == 0


# ---------------------------------------------------------------------------------------
# redirect hints
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name, kind, target", [
    ("meta_refresh", "meta", "/page/plain"),
    ("meta_refresh_abs", "meta", "https://www.other.com/landing"),
    ("js_redirect", "js", "/page/plain"),
    ("js_replace", "js", "https://www.other.com/x"),
    ("js_assign", "js", "https://www.other.com/y"),
])
def test_redirect_hints(name, kind, target):
    out = analyze(name)
    assert out["redirect_kind"] == kind
    assert out["redirect_url"] == target
    assert out["features"] == {}        # a redirect shell is not the page we want features for


def test_slow_meta_refresh_is_not_followed():
    out = analyze("meta_refresh_slow")
    assert out["redirect_kind"] == ""
    assert out["features"]


def test_script_redirect_ignored_on_big_pages():
    big = b"<html><body><script>window.location.href='/x';</script>" + b"<p>text</p>" * 5000 + b"</body></html>"
    assert len(big) > hf.JS_REDIRECT_MAX_BODY
    out = analyze(big)
    assert out["redirect_kind"] == ""
    assert out["features"]


def test_meta_refresh_without_target_is_not_a_redirect():
    body = b"<html><head><meta http-equiv='refresh' content='5'></head><body>stay here, plenty of text, plenty of text, plenty</body></html>"
    assert analyze(body)["redirect_kind"] == ""


# ---------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["challenge", "challenge_body"])
def test_challenge_pages(name):
    out = analyze(name)
    assert out["challenge"] is True
    assert out["features"] == {}


@pytest.mark.parametrize("name", ["parked", "default_nginx", "directory_listing"])
def test_parked_and_default_pages(name):
    out = analyze(name)
    assert out["parked"] is True and out["challenge"] is False
    assert out["features"] == {}


@pytest.mark.parametrize("name", ["plain", "html_all_internal", "html_login_ext_action", "html_empty"])
def test_normal_pages_are_not_flagged(name):
    out = analyze(name)
    assert out["challenge"] is False and out["parked"] is False


# ---------------------------------------------------------------------------------------
# non-HTML, encodings, hostile input
# ---------------------------------------------------------------------------------------
def test_json_is_not_html():
    out = hf.analyze_page(b'{"a": 1}', "application/json", BASE_URL)
    assert out["is_html"] is False and out["features"] == {}


def test_sniffing_without_a_content_type():
    assert hf.analyze_page(b"<!doctype html><html><body>hi</body></html>", None, BASE_URL)["is_html"]
    assert not hf.analyze_page(b"\x89PNG\r\n\x1a\n\x00\x00", None, BASE_URL)["is_html"]
    assert not hf.analyze_page(b"<html></html>", "image/png", BASE_URL)["is_html"]


def test_declared_charset_is_used():
    out = analyze("latin1", ctype="text/html")      # charset comes from <meta>
    assert out["title"] == "Caf\xe9 Example"
    assert out["features"]["domain_in_title"] == 0
    out2 = hf.analyze_page("<title>Caf\xe9 Example</title><p>x</p>".encode("latin-1"),
                           "text/html; charset=ISO-8859-1", BASE_URL)
    assert out2["title"] == "Caf\xe9 Example"


def test_bad_charset_falls_back_to_utf8():
    out = hf.analyze_page(b"<title>ok</title><p>x</p>", "text/html; charset=nonsense-9", BASE_URL)
    assert out["title"] == "ok"


def test_evil_strings_do_not_leak_into_the_title():
    out = analyze("evil_strings")
    assert out["is_html"] and out["features"]
    title = out["title"]
    assert len(title) <= 120
    assert "\x1b" not in title and "‮" not in title and "‬" not in title
    assert out["features"]["empty_title"] == 0


def test_malformed_and_deeply_nested_html_does_not_crash():
    started = time.monotonic()
    out = hf.analyze_page(malformed(3000), "text/html", BASE_URL)
    assert time.monotonic() - started < 10
    assert out["is_html"]
    if out["features"]:
        assert list(out["features"]) == WEBPAGE_FEATURES


def test_huge_page_hits_the_element_cap_and_stays_fast():
    body = big_html(2 * 1024 * 1024)
    started = time.monotonic()
    out = hf.analyze_page(body, "text/html", BASE_URL)
    assert time.monotonic() - started < 15
    assert "element_cap_hit" in out["errors"]
    assert out["features"]


@pytest.mark.parametrize("body", [b"", b"   \n  ", b"<", b"\x00\x00\x00"])
def test_empty_or_garbage_documents(body):
    out = hf.analyze_page(body, "text/html", BASE_URL)
    assert out["features"] == {}
    assert out["challenge"] is False and out["parked"] is False


def test_extractor_error_gives_minus_one_and_is_logged(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(hf, "classify_link", boom)
    out = analyze(page("T", anchors(["/a", "/b"]) + "<p>text text text text text text text text text</p>"))
    assert out["features"] == {name: -1 for name in WEBPAGE_FEATURES}
    assert any(e.startswith("extractor_error:walk") for e in out["errors"])


def test_want_features_false_skips_extraction():
    out = hf.analyze_page(PAGES["plain"], "text/html", BASE_URL, want_features=False)
    assert out["features"] == {}
    assert out["is_html"]
