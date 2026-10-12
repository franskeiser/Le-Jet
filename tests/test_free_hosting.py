"""
Free-hosting and registrable-domain tests (WEB_INSPECTION_PLAN.md section 13.2c, item 8).
No network.

Run with:  python -m pytest tests/test_free_hosting.py
"""

from __future__ import annotations

import pytest

from lejet import domains


@pytest.mark.parametrize("host, registrable, flagged", [
    ("evil.pages.dev", "evil.pages.dev", True),
    ("a.b.github.io", "b.github.io", True),
    ("x.blogspot.com", "x.blogspot.com", True),
    ("www.example.co.uk", "example.co.uk", False),
    ("login.gcash-verify.xyz", "gcash-verify.xyz", False),
    ("shop.example.com", "example.com", False),
    ("my-site.netlify.app", "my-site.netlify.app", True),
    ("abc123.ngrok-free.app", "abc123.ngrok-free.app", True),
    ("EVIL.PAGES.DEV.", "evil.pages.dev", True),                 # case and trailing dot
])
def test_registrable_domain_and_flag(host, registrable, flagged):
    assert domains.registrable_domain(host) == registrable
    info = domains.classify_hosting(host)
    assert bool(info.free_hosting_subdomain) is flagged
    assert info.skip_rdap is flagged


def test_flagged_host_names_its_platform():
    assert domains.classify_hosting("evil.pages.dev").hosting_platform == "pages.dev"
    assert domains.classify_hosting("a.b.github.io").hosting_platform == "github.io"
    assert domains.classify_hosting("x.example.com").hosting_platform == ""


def test_the_platform_itself_is_not_a_free_subdomain():
    # "pages.dev" is the platform, not a site on it
    assert domains.classify_hosting("pages.dev").free_hosting_subdomain == 0
    assert domains.classify_hosting("github.io").free_hosting_subdomain == 0


def test_curated_list_catches_hosts_the_psl_does_not():
    # "000webhostapp.com" and the byethost* wildcard come from free_hosting_suffixes.txt
    assert domains.classify_hosting("shop.000webhostapp.com").free_hosting_subdomain == 1
    info = domains.classify_hosting("site.byethost7.com")
    assert info.free_hosting_subdomain == 1 and info.hosting_platform == "byethost*"
    assert domains.classify_hosting("byethost7.com").free_hosting_subdomain == 0


@pytest.mark.parametrize("host, path, platform", [
    ("docs.google.com", "/forms/d/e/abc/viewform", "docs.google.com/forms"),
    ("forms.gle", "/AbCdEf", "forms.gle"),
    ("sites.google.com", "/view/anything", "sites.google.com"),
    ("linktr.ee", "/someone", "linktr.ee"),
    ("www.facebook.com", "/page123", "www.facebook.com"),
])
def test_path_hosted_platforms(host, path, platform):
    info = domains.classify_hosting(host, path)
    assert info.platform_path_hosting == 1
    assert info.free_hosting_subdomain == 0
    assert info.hosting_platform == platform
    assert info.skip_rdap


def test_path_hosting_needs_the_right_path_prefix():
    # docs.google.com serves other things than forms (documents, sheets): not flagged
    assert domains.classify_hosting("docs.google.com", "/document/d/xyz").platform_path_hosting == 0


def test_ip_and_odd_hosts_are_never_flagged():
    for host in ("127.0.0.1", "203.0.114.7", "::1", "localhost", ""):
        info = domains.classify_hosting(host)
        assert info.free_hosting_subdomain == 0 and info.platform_path_hosting == 0


def test_ip_literals_have_no_registrable_domain():
    assert domains.registrable_domain("203.0.114.7") == ""
    assert domains.registrable_domain("[2001:db8::1]") == ""
    assert domains.registrable_domain("localhost") == ""
    assert domains.registrable_domain("") == ""


def test_dataset_domain_label_ignores_private_suffixes():
    assert domains.domain_label("login.gcash-verify.xyz") == "gcash-verify"
    assert domains.domain_label("evil.pages.dev") == "pages"
    assert domains.domain_label("www.example.co.uk") == "example"
    assert domains.domain_label("203.0.114.7") == ""


@pytest.mark.parametrize("host, tld", [
    ("example.com", "com"), ("www.example.co.uk", "uk"), ("gov.ph", "ph"),
    ("evil.pages.dev", "dev"),                         # private suffixes are ignored for RDAP
    ("203.0.114.7", ""), ("", ""),
])
def test_registry_tld(host, tld):
    assert domains.registry_tld(host) == tld


def test_clean_host():
    assert domains.clean_host("  WWW.Example.COM. ") == "www.example.com"
    assert domains.clean_host("[::1]") == "::1"
    assert domains.host_is_ip("[::1]") and domains.host_is_ip("10.0.0.1")
    assert not domains.host_is_ip("example.com")
