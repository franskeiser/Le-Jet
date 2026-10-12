"""
domains.py -- registrable domains and free-hosting detection (plan sections 13.2, 13.2c).

Two public-suffix views of the same host, both OFFLINE (bundled snapshot, no download, no
cache directory -- see the "tldextract gotcha" in plan 13.2):

  * private suffixes ON  -> `registrable_domain()`: evil.pages.dev is its own domain. Used for
    internal/external link classification, politeness, deduplication and the RDAP decision.
  * private suffixes OFF -> `domain_label()`: the dataset's own extractor, used for the
    `domain_in_title` / `domain_with_copyright` features so they mean what they meant in
    training (the same split `src/features/link_features.py` documents).
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import tldextract

_PRIVATE = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)
_PUBLIC = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

HOSTING_FILE = os.path.join(os.path.dirname(__file__), "features", "free_hosting_suffixes.txt")


def clean_host(host: str) -> str:
    """Lower-case, strip IPv6 brackets and trailing dots."""
    h = (host or "").strip().lower()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    return h.rstrip(".")


def host_is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(clean_host(host))
        return True
    except ValueError:
        return False


def registrable_domain(host: str) -> str:
    """Registrable domain with private suffixes ON ('' for IPs, single labels, bad input)."""
    h = clean_host(host)
    if not h or host_is_ip(h):
        return ""
    try:
        return _PRIVATE(h).top_domain_under_public_suffix or ""
    except Exception:
        return ""


def domain_label(host: str) -> str:
    """The dataset's 'domain' part: the label left of the (public) suffix, private OFF.
    'login.gcash-verify.xyz' -> 'gcash-verify'; 'evil.pages.dev' -> 'pages'."""
    h = clean_host(host)
    if not h or host_is_ip(h):
        return ""
    try:
        return _PUBLIC(h).domain or ""
    except Exception:
        return ""


def public_suffix(host: str) -> str:
    """Private-suffix-ON suffix ('' if none)."""
    h = clean_host(host)
    if not h or host_is_ip(h):
        return ""
    try:
        return _PRIVATE(h).suffix or ""
    except Exception:
        return ""


def registry_tld(host: str) -> str:
    """The TLD to look up in the IANA RDAP bootstrap: the last label of the PUBLIC suffix
    (co.uk -> uk). Private suffixes are ignored on purpose."""
    h = clean_host(host)
    if not h or host_is_ip(h):
        return ""
    try:
        suffix = _PUBLIC(h).suffix
    except Exception:
        return ""
    return suffix.rsplit(".", 1)[-1] if suffix else ""


# ---------------------------------------------------------------------------
# Free-hosting / shared platforms (plan 13.2c)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HostingInfo:
    free_hosting_subdomain: int = 0
    hosting_platform: str = ""
    platform_path_hosting: int = 0

    @property
    def skip_rdap(self) -> bool:
        return bool(self.free_hosting_subdomain or self.platform_path_hosting)


@lru_cache(maxsize=1)
def _load_rules(path: str = HOSTING_FILE) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    suffixes: List[str] = []
    prefixes: List[str] = []
    paths: List[Tuple[str, str]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = []
    for line in lines:
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        if line.startswith("path:"):
            spec = line[5:]
            host, _, rest = spec.partition("/")
            paths.append((host, "/" + rest))
        elif line.endswith("*"):
            prefixes.append(line[:-1])
        else:
            suffixes.append(line)
    return tuple(suffixes), tuple(prefixes), tuple(paths)


def classify_hosting(host: str, path: str = "") -> HostingInfo:
    """Flag free-hosting subdomains and path-hosted platforms for one host (+ URL path)."""
    h = clean_host(host)
    if not h:
        return HostingInfo()
    suffixes, prefixes, paths = _load_rules()
    path = path or "/"

    # Path-hosted platforms: the host is shared, the content is in the path.
    for p_host, p_prefix in paths:
        if h == p_host and (p_prefix == "/" or path.lower().startswith(p_prefix)):
            return HostingInfo(0, p_host + (p_prefix if p_prefix != "/" else ""), 1)

    if host_is_ip(h):
        return HostingInfo()

    # Curated suffix list (longest match wins).
    best = ""
    for suf in suffixes:
        if h.endswith("." + suf) and len(suf) > len(best):
            best = suf
    if best:
        return HostingInfo(1, best, 0)

    # Wildcard label-prefix rules such as byethost*.
    if prefixes:
        labels = h.split(".")
        for pref in prefixes:
            if any(lab.startswith(pref) for lab in labels[1:]):
                return HostingInfo(1, pref + "*", 0)

    # The Public Suffix List's private section: the host sits under a private suffix
    # (pages.dev, github.io, blogspot.com, ...) AND has a label of its own beyond it.
    try:
        priv, pub = _PRIVATE(h), _PUBLIC(h)
    except Exception:
        return HostingInfo()
    if priv.suffix and priv.suffix != pub.suffix and priv.domain:
        return HostingInfo(1, priv.suffix, 0)
    return HostingInfo()
