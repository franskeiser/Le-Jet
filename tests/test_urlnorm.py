"""
Tests for lejet.urlnorm (Stage 0). Verifies parsing, normalization, the
schemeless https-then-http fallback, and that the safety gate is actually
wired in (unsafe hosts are refused here, not just by the firewall later).

Run with:  pytest tests/test_urlnorm.py
Or plain:  python -I tests/test_urlnorm.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lejet import urlnorm  # noqa: E402


# (raw_url, expected_ok, expected_fail_reason_or_None)
CASES = [
    # legitimate / fetchable
    ("https://example.com/a", True, None),
    ("http://example.com/", True, None),
    ("https://example.com:443/x", True, None),        # explicit allowed port
    ("https://xn--80ak6aa92e.com/", True, None),       # already-punycode host
    ("gcash-verify.xyz/login", True, None),            # schemeless SMS URL
    ("gcash-verify.xyz/login).", True, None),          # trailing junk stripped
    ("login.gcash-verify.xyz/a", True, None),          # phishing name is valid
    # impossible port -> distinct reason (not "dead")
    ("http://example.com:8080/", False, "port_blocked"),
    ("http://example.com:8443/", False, "port_blocked"),
    # structurally rejected
    ("ftp://example.com/x", False, "rejected_url"),
    ("javascript:alert(1)", False, "rejected_url"),
    ("mailto:a@b.com", False, "rejected_url"),
    ("http://user:pass@example.com/", False, "rejected_url"),
    ("https://exa mple.com/", False, "rejected_url"),  # whitespace in URL
    ("https://" + "x" * 3000 + ".com/", False, "rejected_url"),  # too long
    ("", False, "rejected_url"),
    # SSRF -- the safety gate must refuse these at Stage 0
    ("http://127.0.0.1/", False, "rejected_url"),
    ("http://2130706433/", False, "rejected_url"),     # decimal loopback
    ("http://0x7f000001/", False, "rejected_url"),     # hex loopback
    ("http://169.254.169.254/latest/meta-data/", False, "rejected_url"),
    ("https://localhost/", False, "rejected_url"),
    ("http://10.77.0.1/", False, "rejected_url"),      # the gateway VM
    ("http://[::1]/", False, "rejected_url"),          # IPv6 loopback
]


def test_cases():
    failures = []
    for raw, want_ok, want_reason in CASES:
        r = urlnorm.normalize(raw)
        if r.ok != want_ok:
            failures.append(f"{raw!r}: ok={r.ok} (reason={r.fail_reason!r}), want ok={want_ok}")
        elif not want_ok and r.fail_reason != want_reason:
            failures.append(f"{raw!r}: reason={r.fail_reason!r}, want {want_reason!r}")
    assert not failures, "urlnorm regressions:\n" + "\n".join(failures)


def test_schemeless_fallback_order():
    r = urlnorm.normalize("gcash-verify.xyz/login")
    assert r.ok and r.scheme_present is False
    assert r.fetch_candidates[0].startswith("https://")
    assert r.fetch_candidates[1].startswith("http://")
    assert r.normalized_url == r.fetch_candidates[0]
    assert r.host == "gcash-verify.xyz"


def test_raw_url_is_preserved():
    raw = "gcash-verify.xyz/login)."          # junk kept in raw, stripped in fetch
    r = urlnorm.normalize(raw)
    assert r.raw_url == raw                    # features rely on the exact string
    assert r.normalized_url == "https://gcash-verify.xyz/login"


def test_url_id_stable_and_distinct():
    a = urlnorm.compute_url_id("https://a.example/")
    b = urlnorm.compute_url_id("https://b.example/")
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)
    assert a != b
    # normalize() reports the same id as the helper
    assert urlnorm.normalize("https://a.example/").url_id == a


def test_punycode_flag():
    assert urlnorm.normalize("https://xn--80ak6aa92e.com/").punycode is True
    assert urlnorm.normalize("https://example.com/").punycode is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nAll {len(fns)} test functions passed.")
