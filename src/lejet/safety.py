"""
safety.py -- the single chokepoint for "is this safe?" in Le-Jet.

This module does NOT fetch anything and has NO third-party dependencies
(standard library only), so it can be imported and tested anywhere,
including on the host with `python -I`.

It answers three questions for the rest of the pipeline:

  1. "Is this target address allowed?"  -> host/IP safety gate (anti-SSRF)
     Called by Stage 0 (urlnorm) AND re-run on every redirect hop (Stage 3),
     and again by the live /inspect service (plan section 24), which may run
     WITHOUT the two-VM firewall -- so here these checks are the only defense.

  2. "Is this response too big?"        -> size caps (anti-resource-exhaustion)

  3. "Is this string safe to keep?"     -> sanitizers (anti-injection)

See WEB_INSPECTION_PLAN.md threat-model controls A5 (DNS/IP), A9 (resources),
A10 (poisoned output), and review items S8 / S4.

Design note on the IP gate
---------------------------
A hostile page can hand us an address that *looks* harmless as text but
resolves to our own machine or LAN, e.g. http://2130706433/ == 127.0.0.1.
Testing showed that `ipaddress.ip_address()` ALONE rejects every obfuscated
IPv4 form (decimal/hex/octal/short) as "not an IP" and would let them through
as a hostname; `socket.inet_aton()` alone misses IPv4-mapped IPv6. So we
canonicalize with BOTH a strict parser and a relaxed, getaddrinfo-style parser,
after Unicode (NFKC) folding and trailing-dot stripping, and reject if ANY
interpretation lands in a blocked range.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Size caps (control A9). Count DECODED bytes, not compressed, so a
# decompression bomb cannot expand past the cap.
# ---------------------------------------------------------------------------
MAX_URL_LEN = 2048            # reject longer URLs outright
MAX_DECODED_BYTES = 2 * 1024 * 1024    # read at most 2 MB of HTML body
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # abort if the server announces > 10 MB
MAX_RDAP_BYTES = 256 * 1024            # RDAP JSON response cap
MAX_STR_FIELD = 512           # default cap for any stored free-text field
MAX_TITLE = 120               # <title> and similar

# Ports the gateway firewall permits. Anything else can never succeed, so we
# reject it up front with a distinct reason instead of recording it as "dead".
ALLOWED_PORTS = frozenset({80, 443})

# ---------------------------------------------------------------------------
# Blocked address space -- ONE source of truth (review item: three hand-kept
# copies in the plan drift apart). nftables/unbound configs should be
# generated from this list, not maintained separately.
#
# We keep the explicit list AND rely on ipaddress's own classification flags
# below, so a new reserved range added to Python is covered even if someone
# forgets to add it here.
# ---------------------------------------------------------------------------
_BLOCKED_V4 = [
    "0.0.0.0/8",         # "this host" / unspecified -- connect() treats as localhost
    "10.0.0.0/8",        # private
    "100.64.0.0/10",     # CGNAT / shared address space
    "127.0.0.0/8",       # loopback
    "169.254.0.0/16",    # link-local  (includes 169.254.169.254 cloud metadata)
    "172.16.0.0/12",     # private
    "192.0.0.0/24",      # IETF protocol assignments
    "192.0.2.0/24",      # TEST-NET-1
    "192.168.0.0/16",    # private
    "198.18.0.0/15",     # benchmarking
    "198.51.100.0/24",   # TEST-NET-2
    "203.0.113.0/24",    # TEST-NET-3
    "224.0.0.0/3",       # multicast + reserved (240.0.0.0/4)
]
_BLOCKED_V6 = [
    "::/128",            # unspecified
    "::1/128",           # loopback
    "::ffff:0:0/96",     # IPv4-mapped (covered again via the mapped v4 check)
    "64:ff9b::/96",      # NAT64 (can embed private v4)
    "100::/64",          # discard-only
    "2001:db8::/32",     # documentation
    "fc00::/7",          # unique local
    "fe80::/10",         # link-local
    "ff00::/8",          # multicast
]
_BLOCKED_NETS = [ipaddress.ip_network(c) for c in _BLOCKED_V4 + _BLOCKED_V6]

# Hostnames that are unsafe regardless of what they resolve to, plus the lab's
# own machines. Matched case-insensitively, with and without a trailing dot.
_BLOCKED_NAME_SUFFIXES = (
    ".localhost", ".local", ".internal", ".lan", ".home.arpa",
)
_BLOCKED_NAMES = {
    "localhost",
    "lejet-gw", "lejet-an",          # the lab VMs' own hostnames
}
# The lab's own internal IPs (string form is also caught by _BLOCKED_V4 via
# 10.0.0.0/8, but we keep them explicit for clarity/tests).
LAB_INTERNAL_IPS = {"10.77.0.1", "10.77.0.2", "10.0.2.2", "10.0.2.3"}


# ---------------------------------------------------------------------------
# IP parsing
# ---------------------------------------------------------------------------
def _parse_ipv4_relaxed(text: str) -> Optional[ipaddress.IPv4Address]:
    """
    Parse the obfuscated IPv4 forms a resolver (inet_aton / glibc getaddrinfo)
    accepts but `ipaddress.ip_address()` rejects: 1-4 dotted parts, each in
    decimal, octal (leading 0) or hex (leading 0x); the final part may be a
    big number that fills the remaining low octets.

    Returns None if `text` is not a numeric IPv4 under these rules.
    """
    parts = text.split(".")
    if not 1 <= len(parts) <= 4:
        return None

    values = []
    for p in parts:
        if p == "":
            return None
        try:
            if p.lower().startswith("0x"):
                v = int(p, 16)
            elif p[0] == "0" and len(p) > 1:
                v = int(p, 8)          # ValueError on e.g. "08"
            else:
                v = int(p, 10)
        except ValueError:
            return None
        if v < 0:
            return None
        values.append(v)

    # Each leading part is one octet (< 256); the last part fills the rest.
    *lead, last = values
    if any(v > 0xFF for v in lead):
        return None
    max_last = 1 << (8 * (4 - len(lead)))
    if last >= max_last:
        return None
    acc = 0
    for v in lead:
        acc = (acc << 8) | v
    acc = (acc << (8 * (4 - len(lead)))) | last
    return ipaddress.IPv4Address(acc)


def _as_ip(host: str) -> Optional[ipaddress._BaseAddress]:
    """
    Interpret `host` as an IP address under ANY representation a resolver
    might accept, or return None if it is a genuine name.
    """
    h = host.strip()
    # IPv6 literal in URL brackets: [::1]
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
        # strip a zone id if present (fe80::1%eth0)
        h = h.split("%", 1)[0]
    try:
        return ipaddress.ip_address(h)      # clean v4, v6, and v4-mapped v6
    except ValueError:
        pass
    v4 = _parse_ipv4_relaxed(h)             # decimal/hex/octal/short v4
    if v4 is not None:
        return v4
    return None


def is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """True if this concrete IP must never be connected to."""
    # ipaddress's own flags first (future-proof against new reserved ranges).
    if (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
        return True
    # IPv4-mapped / 6to4 / Teredo: unwrap to the embedded v4 and re-check.
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped or ip.sixtofour
        if mapped is not None and is_blocked_ip(mapped):
            return True
    # Explicit list (covers CGNAT and anything the flags miss).
    return any(ip in net for net in _BLOCKED_NETS)


def is_ip_allowed(ip_text: str) -> bool:
    """
    Check an already-resolved IP (string) before connecting to it.
    Use this on every A/AAAA answer (control A5 rebinding backstop).
    """
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return not is_blocked_ip(ip)


def _normalize_host(host: str) -> str:
    """Fold confusables and trailing dots so look-alike hosts can't dodge
    the name/IP checks. NFKC turns e.g. circled digits into ASCII digits."""
    h = unicodedata.normalize("NFKC", host).strip().strip(".")
    return h.lower()


# ---------------------------------------------------------------------------
# The host safety gate (the SSRF bouncer)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HostVerdict:
    allowed: bool
    reason: str          # "" when allowed; a fail_reason code otherwise
    is_ip: bool
    canonical: str       # canonical IP string, or the normalized name


def check_host(host: str, extra_blocked: Iterable[str] = ()) -> HostVerdict:
    """
    Decide whether `host` (the host part of a URL, or a redirect target's host)
    may be contacted. Does NOT resolve DNS -- name hosts pass this gate and are
    then checked again at the resolved-IP level by is_ip_allowed().

    fail_reason codes returned: "rejected_url" (unsafe host).
    """
    if not host:
        return HostVerdict(False, "rejected_url", False, "")

    ip = _as_ip(host)
    if ip is not None:
        canon = str(ip.ipv4_mapped) if (
            isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped
        ) else str(ip)
        if is_blocked_ip(ip):
            return HostVerdict(False, "rejected_url", True, canon)
        return HostVerdict(True, "", True, canon)

    name = _normalize_host(host)
    extra = {_normalize_host(x) for x in extra_blocked}
    if name in _BLOCKED_NAMES or name in extra or name in LAB_INTERNAL_IPS:
        return HostVerdict(False, "rejected_url", False, name)
    if any(name == s.lstrip(".") or name.endswith(s) for s in _BLOCKED_NAME_SUFFIXES):
        return HostVerdict(False, "rejected_url", False, name)
    return HostVerdict(True, "", False, name)


def check_port(port: Optional[int]) -> bool:
    """True if a connection to this port can be allowed (None = default 80/443)."""
    return port is None or port in ALLOWED_PORTS


# ---------------------------------------------------------------------------
# Size caps (control A9)
# ---------------------------------------------------------------------------
def content_length_ok(content_length: Optional[int]) -> bool:
    """False if the server announces a body larger than we will ever read."""
    if content_length is None:
        return True            # chunked/unknown -> rely on the streaming budget
    return 0 <= content_length <= MAX_CONTENT_LENGTH


class ByteBudget:
    """
    Accumulate decoded body bytes and stop at the cap. Use while streaming:

        budget = ByteBudget()
        for chunk in resp.iter_bytes():      # already gzip-decoded
            budget.add(chunk)                # raises BodyTooLarge past 2 MB
    """
    __slots__ = ("limit", "seen")

    def __init__(self, limit: int = MAX_DECODED_BYTES) -> None:
        self.limit = limit
        self.seen = 0

    def add(self, chunk: bytes) -> None:
        self.seen += len(chunk)
        if self.seen > self.limit:
            raise BodyTooLarge(self.seen, self.limit)

    @property
    def exceeded(self) -> bool:
        return self.seen > self.limit


class BodyTooLarge(Exception):
    """Raised by ByteBudget when the decoded body passes the cap."""

    def __init__(self, seen: int, limit: int) -> None:
        super().__init__(f"body exceeded {limit} bytes (saw at least {seen})")
        self.seen = seen
        self.limit = limit


# ---------------------------------------------------------------------------
# Sanitizers (control A10) -- make attacker-controlled strings inert before
# they are stored in the VM or exported to the host.
# ---------------------------------------------------------------------------
# C0 controls (except we never keep tab/newline/cr either), DEL, C1 controls,
# and Unicode bidi overrides/isolates that can reorder text visually.
_CONTROL_RE = re.compile(
    r"[\x00-\x1f\x7f\x80-\x9f‎‏‪-‮⁦-⁩]"
)
# ANSI/CSI escape sequences (terminal injection).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# Leading characters that make a spreadsheet treat a cell as a formula.
_FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r", "\n")


def strip_dangerous_chars(s: str) -> str:
    """Remove control chars, bidi overrides and ANSI escapes."""
    if not isinstance(s, str):
        return ""
    s = _ANSI_RE.sub("", s)
    return _CONTROL_RE.sub("", s)


def clean_field(s: Optional[str], max_len: int = MAX_STR_FIELD) -> str:
    """Full cleanup for any free-text value we keep: strip dangerous chars,
    collapse surrounding whitespace, and cap length."""
    if s is None:
        return ""
    out = strip_dangerous_chars(str(s)).strip()
    if len(out) > max_len:
        out = out[:max_len]
    return out


def csv_safe(s: str) -> str:
    """Prefix a single quote if the value could be read as a spreadsheet
    formula. Apply ONLY when writing CSV (not needed for Parquet/JSON)."""
    if s and s[0] in _FORMULA_LEADERS:
        return "'" + s
    return s


def defang_url(url: str) -> str:
    """Make a URL un-clickable for anything that leaves the VM or appears in
    a document/spreadsheet: http(s) -> hxxp(s), '.' -> '[.]'."""
    s = strip_dangerous_chars(str(url))
    s = re.sub(r"^https?", lambda m: m.group(0).replace("http", "hxxp"), s, flags=re.I)
    return s.replace(".", "[.]")
