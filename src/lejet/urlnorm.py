"""
urlnorm.py -- Stage 0: parse, normalize, and refuse unsafe URLs.

This is the FIRST thing that happens to every URL, before any network
contact (plan section 13.2). It does three jobs:

  1. Parse the messy raw link an SMS extractor produced.
  2. Produce a clean, fetchable form (add a scheme, strip trailing junk).
  3. Refuse anything unsafe or impossible -- by calling the safety gate
     in lejet.safety.

Two representations are kept for every URL (important, plan section 13.2):

  * raw_url        -- the string EXACTLY as given. The 55 URL-string
                      features are computed on this, for dataset parity.
                      urlnorm never mutates it.
  * normalized_url -- a tidy form used ONLY to fetch (and for url_id).

No third-party dependencies are required to run this module; tldextract is
used opportunistically for the registrable domain and degrades to None if
it is not installed, so the module stays testable on the host.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlsplit, urlunsplit

from lejet import safety

# A scheme token at the very start, e.g. "http:", "javascript:", "mailto:".
_SCHEME_RE = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*):")
# Any whitespace or control character anywhere -> a malformed URL.
_WS_CTRL_RE = re.compile(r"[\s\x00-\x1f\x7f]")
# Trailing punctuation SMS extractors commonly glue on (plan section 13.2).
_TRAILING_JUNK = ".,);]}>\"'"

ALLOWED_SCHEMES = ("http", "https")


@dataclass(frozen=True)
class NormResult:
    ok: bool
    fail_reason: str                 # "" when ok; else a fail_reason code
    raw_url: str                     # exactly as given (features use this)
    url_id: str                      # sha256(raw_url)[:16]
    normalized_url: Optional[str] = None   # primary fetch target
    fetch_candidates: List[str] = field(default_factory=list)  # https, then http
    host: Optional[str] = None       # host as parsed (pre-IDNA)
    host_ascii: Optional[str] = None # IDNA/ASCII host used in normalized_url
    port: Optional[int] = None       # explicit port, if any
    scheme_present: bool = False     # was a scheme in the raw URL?
    punycode: bool = False           # does the ASCII host contain xn--?
    registrable_domain: Optional[str] = None


def compute_url_id(raw_url: str) -> str:
    """Stable id for joining results back to labels.

    NOTE: in production the HOST computes this and pushes `url_id<TAB>raw_url`
    lines; the VM only echoes the id back (plan section 13.2, review V14). This
    helper exists so both sides compute it identically.
    """
    return hashlib.sha256(raw_url.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _reject(raw_url: str, reason: str) -> NormResult:
    return NormResult(
        ok=False, fail_reason=reason, raw_url=raw_url, url_id=compute_url_id(raw_url)
    )


# tldextract is optional: construct it offline (no PSL download, no cache),
# with private suffixes ON so evil.pages.dev is its own registrable domain
# (plan section 13.2c). Returns None if tldextract is not installed.
_extractor = None
_extractor_tried = False


def registrable_domain(host: str) -> Optional[str]:
    global _extractor, _extractor_tried
    if not _extractor_tried:
        _extractor_tried = True
        try:
            import tldextract  # type: ignore

            _extractor = tldextract.TLDExtract(
                suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
            )
        except Exception:
            _extractor = None
    if _extractor is None:
        return None
    try:
        ext = _extractor(host)
        return ext.top_domain_under_public_suffix or None
    except Exception:
        return None


LOOPBACK_TEST_HOSTS = ("localhost", "127.0.0.1")


def normalize(raw_url: str, allow_loopback: bool = False) -> NormResult:
    """Parse, normalize and safety-check one raw URL. Never raises.

    `allow_loopback` is TEST MODE ONLY (plan section 13.2, `--allow-loopback`): it lets
    `localhost` / `127.0.0.1` pass the host gate and any port through, so the Level 0
    fixtures can be served from a local web server. Real runs never set it.
    """
    if raw_url is None:
        return _reject("", "rejected_url")

    core = raw_url.strip()
    if not core:
        return _reject(raw_url, "rejected_url")

    # Length and stray whitespace/control characters.
    if len(core) > safety.MAX_URL_LEN:
        return _reject(raw_url, "rejected_url")
    if _WS_CTRL_RE.search(core):
        return _reject(raw_url, "rejected_url")

    # Strip trailing punctuation (affects the fetch copy only; raw is untouched).
    core = core.rstrip(_TRAILING_JUNK)
    if not core:
        return _reject(raw_url, "rejected_url")

    # Scheme handling.
    m = _SCHEME_RE.match(core)
    scheme_present = False
    if m:
        scheme = m.group("scheme").lower()
        if scheme not in ALLOWED_SCHEMES:
            return _reject(raw_url, "rejected_url")      # ftp:, javascript:, ...
        scheme_present = True
        parse_target = core
    else:
        # Schemeless (very common in SMS): parse with a stand-in so urlsplit
        # fills netloc; the real scheme is decided by fetch order below.
        parse_target = "https://" + core

    try:
        parts = urlsplit(parse_target)
    except ValueError:
        return _reject(raw_url, "rejected_url")

    # Credentials in the authority (user:pass@host) are refused.
    if parts.username or parts.password:
        return _reject(raw_url, "rejected_url")

    host = parts.hostname
    if not host:
        return _reject(raw_url, "rejected_url")

    # Explicit port: urlsplit raises ValueError on a bad port.
    try:
        port = parts.port
    except ValueError:
        return _reject(raw_url, "rejected_url")

    # --- SAFETY GATE (the SSRF bouncer) --------------------------------------
    # Test mode only: exactly "localhost" / "127.0.0.1" (nothing else) skip the gate.
    loopback_ok = allow_loopback and host.lower() in LOOPBACK_TEST_HOSTS
    if not loopback_ok:
        # Pass the host as parsed so safety does its own NFKC/IP canonicalization.
        verdict = safety.check_host(host)
        if not verdict.allowed:
            return _reject(raw_url, verdict.reason or "rejected_url")

        # Port must be one the gateway firewall permits; otherwise it can never
        # succeed, so record a distinct reason (not "dead").
        if not safety.check_port(port):
            return _reject(raw_url, "port_blocked")

    # IDNA / punycode conversion for the host used in the normalized URL.
    try:
        host.encode("ascii")
        host_ascii = host.lower()
    except UnicodeEncodeError:
        try:
            host_ascii = host.encode("idna").decode("ascii")
        except (UnicodeError, UnicodeDecodeError):
            return _reject(raw_url, "rejected_url")
    punycode = "xn--" in host_ascii.lower()

    # Rebuild a clean authority (keep an explicit allowed port if present).
    netloc = host_ascii if port is None else f"{host_ascii}:{port}"
    path = parts.path or ("/" if scheme_present else "")
    query = parts.query

    def _build(scheme: str) -> str:
        return urlunsplit((scheme, netloc, path or "/", query, ""))

    if scheme_present:
        normalized = _build(parts.scheme.lower())
        candidates = [normalized]
    else:
        # Try HTTPS first, then HTTP once (many phishing sites are HTTP-only).
        https_url = _build("https")
        http_url = _build("http")
        normalized = https_url
        candidates = [https_url, http_url]

    return NormResult(
        ok=True,
        fail_reason="",
        raw_url=raw_url,
        url_id=compute_url_id(raw_url),
        normalized_url=normalized,
        fetch_candidates=candidates,
        host=host,
        host_ascii=host_ascii,
        port=port,
        scheme_present=scheme_present,
        punycode=punycode,
        registrable_domain=registrable_domain(host_ascii),
    )
