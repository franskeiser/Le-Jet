"""
Tests for lejet.safety -- the safety gate and sanitizers.

The IP table is the important part: it is the regression guard for the
obfuscated-loopback bypass (review item S8). Every row that disguises a
blocked address MUST be rejected.

Run with:  pytest tests/test_safety.py
Or plain:  python -I tests/test_safety.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lejet import safety  # noqa: E402


# (host, must_be_allowed) -- False means the gate must REJECT it.
HOST_CASES = [
    # --- obfuscated loopback / private: all must be REJECTED ---
    ("2130706433", False),        # decimal 127.0.0.1
    ("0x7f000001", False),        # hex 127.0.0.1
    ("0177.0.0.1", False),        # octal first octet
    ("127.1", False),             # short form
    ("0x7f.0.0.1", False),        # mixed hex
    ("017700000001", False),      # full octal
    ("127.0.0.1", False),         # plain loopback
    ("[::1]", False),             # IPv6 loopback literal
    ("[::ffff:127.0.0.1]", False),# IPv4-mapped IPv6 loopback
    ("[::ffff:7f00:1]", False),   # same, hex form
    ("169.254.169.254", False),   # cloud metadata
    ("2852039166", False),        # decimal 169.254.169.254
    ("10.77.0.1", False),         # the gateway VM
    ("10.77.0.2", False),         # the analysis VM
    ("192.168.1.1", False),       # home LAN
    ("172.16.5.4", False),        # private
    ("100.64.0.1", False),        # CGNAT
    ("0.0.0.0", False),           # unspecified -> localhost on connect
    ("localhost", False),
    ("LOCALHOST.", False),        # case + trailing dot
    ("foo.localhost", False),
    ("printer.local", False),
    ("db.internal", False),
    ("fe80::1", False),           # link-local v6 (no brackets)
    # --- legitimate public targets: all must be ALLOWED ---
    ("example.com", True),
    ("gcash-verify.xyz", True),   # a phishing name is still a valid target
    ("login.gcash-verify.xyz", True),
    ("8.8.8.8", True),            # public IP
    ("1.1.1.1", True),
    ("[2606:4700:4700::1111]", True),  # public IPv6 (Cloudflare)
    ("xn--80ak6aa92e.com", True),      # already-punycode host
]


def test_host_gate():
    failures = []
    for host, want_allowed in HOST_CASES:
        verdict = safety.check_host(host)
        if verdict.allowed != want_allowed:
            failures.append(
                f"{host!r}: got allowed={verdict.allowed} "
                f"(canonical={verdict.canonical!r}), want {want_allowed}"
            )
    assert not failures, "host-gate regressions:\n" + "\n".join(failures)


def test_resolved_ip_backstop():
    # A public name that (via DNS rebinding) resolves to a private IP must be
    # caught at the resolved-IP layer.
    assert safety.is_ip_allowed("93.184.216.34") is True      # example.com
    assert safety.is_ip_allowed("127.0.0.1") is False
    assert safety.is_ip_allowed("169.254.169.254") is False
    assert safety.is_ip_allowed("not-an-ip") is False


def test_port_gate():
    assert safety.check_port(None) is True
    assert safety.check_port(80) is True
    assert safety.check_port(443) is True
    assert safety.check_port(8080) is False
    assert safety.check_port(8443) is False


def test_size_caps():
    assert safety.content_length_ok(None) is True
    assert safety.content_length_ok(1_000) is True
    assert safety.content_length_ok(safety.MAX_CONTENT_LENGTH + 1) is False

    budget = safety.ByteBudget(limit=10)
    budget.add(b"12345")
    budget.add(b"12345")          # exactly at limit, fine
    try:
        budget.add(b"x")          # one over -> raises
    except safety.BodyTooLarge:
        pass
    else:
        raise AssertionError("ByteBudget did not raise past the cap")


def test_sanitizers():
    # control chars, bidi override and ANSI escape all stripped
    dirty = "a\x00b\x1b[31mc‮d"
    assert safety.strip_dangerous_chars(dirty) == "abcd"

    # CSV formula injection neutralized
    assert safety.csv_safe("=1+2") == "'=1+2"
    assert safety.csv_safe("+ping") == "'+ping"
    assert safety.csv_safe("@SUM(A1)") == "'@SUM(A1)"
    assert safety.csv_safe("normal") == "normal"

    # length cap
    assert len(safety.clean_field("x" * 999, max_len=120)) == 120

    # defang
    d = safety.defang_url("https://evil.example.com/login")
    assert "hxxps://" in d and "[.]" in d and "://" in d
    assert "http" not in d.replace("hxxp", "")


def test_defang_neutralises_links_embedded_in_a_url():
    """A redirect page carries whole links inside its query string; none may stay clickable."""
    import re
    url = ("https://accounts.google.com/v3/signin/rejected?continue=https://accounts.google.com/"
           "&followup=http://example.org/x&next=HTTPS://Evil.example.net/")
    d = safety.defang_url(url)
    assert not re.search(r"(?i)\bhttps?://", d)
    assert d.startswith("hxxps://accounts[.]google[.]com/")
    assert d.lower().count("hxxps://") == 3 and d.lower().count("hxxp://") == 1
    # the lab's output validator must accept it
    assert "://" in d


def test_defang_embedded_is_idempotent_and_leaves_dots_alone():
    once = safety.defang_url("https://a.example.com/?u=https://b.example.org/")
    assert safety.defang_embedded_urls(once) == once                     # already safe: unchanged
    assert safety.defang_embedded_urls("see https://x.example.com/a.b") == "see hxxps://x.example.com/a.b"
    assert safety.defang_embedded_urls("no link here") == "no link here"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nAll {len(fns)} test functions passed.")
