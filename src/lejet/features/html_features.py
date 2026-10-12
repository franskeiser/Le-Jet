"""
html_features.py -- static HTML analysis (plan sections 13.5, 13.5b, 14.0.3).

`analyze_page()` is a PURE function over the bytes the HTTP stage captured. It never
executes anything and never fetches a linked resource. It returns:

  * the 12 webpage features (or {} when the page is not a usable page),
  * a redirect hint (<meta refresh> or a trivial script redirect found by pattern),
  * "challenge page" / "parked or default page" verdicts,
  * helpers: html_bytes, script_count, near_empty.

It runs inside a worker process with a hard time limit (lejet.parse_pool), because parsing
2 MB of hostile HTML must never be able to freeze the event loop.

Definitions follow the dataset's own extractor (Hannousse & Yahiouche, 2021) as closely as
documented in docs/html_feature_definitions.md. Every ambiguity that had to be resolved is
listed there. Scales match the cleaned public set: ratio_intHyperlinks is 0-1,
links_in_tags / ratio_intMedia / ratio_extMedia are 0-100.
"""

from __future__ import annotations

import codecs
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

from lxml import etree
from lxml import html as lhtml

from lejet import domains, safety

MAX_ELEMENTS = 20_000          # plan 13.5b: count at most 20,000 elements
MAX_TEXT_CHARS = 2_000_000
JS_REDIRECT_MAX_BODY = 32 * 1024   # script redirects are only trusted on tiny "shell" pages
JS_SCRIPT_TEXT_CAP = 32 * 1024
META_REFRESH_MAX_DELAY = 10.0

# The 12 features, in the order of rf_extractor.WEBPAGE_FEATURES.
WEBPAGE_FEATURES = [
    "nb_hyperlinks", "ratio_intHyperlinks", "nb_extCSS", "login_form", "external_favicon",
    "links_in_tags", "ratio_intMedia", "ratio_extMedia", "iframe", "empty_title",
    "domain_in_title", "domain_with_copyright",
]

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")
_PHP_ACTION_RE = re.compile(r"^[a-zA-Z0-9_]+\.php")
_CHARSET_HEADER_RE = re.compile(r"charset\s*=\s*[\"']?([A-Za-z0-9_\-:.]+)", re.I)
_CHARSET_META_RE = re.compile(rb"charset\s*=\s*[\"']?([A-Za-z0-9_\-:.]+)", re.I)
_META_REFRESH_RE = re.compile(
    r"^\s*(?P<delay>\d+(?:\.\d+)?)\s*[;,]?\s*(?:url\s*=\s*)?[\"']?(?P<url>[^\"']*?)[\"']?\s*$", re.I
)
# Trivial script redirects with a STRING LITERAL target (linear-time patterns, plan 13.5).
_JS_REDIRECT_RES = (
    re.compile(r"(?:window\.|document\.|top\.|self\.)?location(?:\.href)?\s*=\s*[\"']([^\"'\\\r\n]{1,2048})[\"']"),
    re.compile(r"location\.(?:replace|assign)\(\s*[\"']([^\"'\\\r\n]{1,2048})[\"']\s*\)"),
)
_COPYRIGHT_RE = re.compile("[©™®]")   # (c) (tm) (r)
_ICON_REL = {"icon", "shortcut", "apple-touch-icon"}

_CHALLENGE_TITLES = (
    "just a moment", "attention required", "checking your browser", "are you a robot",
    "verify you are human", "ddos protection", "security check", "one more step",
)
_CHALLENGE_BODY = (
    "enable javascript and cookies to continue", "cf-browser-verification",
    "challenge-platform", "/cdn-cgi/challenge", "verifying you are human",
    "needs to review the security of your connection", "g-recaptcha", "h-captcha",
)
_PARKED_TITLES = (
    "welcome to nginx", "apache2 ubuntu default page", "apache2 debian default page",
    "test page for the apache", "it works!", "index of /", "iis windows server",
    "default web site page", "domain default page", "web server's default page",
    "default page", "under construction", "coming soon", "parked", "domain for sale",
    "this domain is for sale", "account suspended", "suspended",
)
_PARKED_BODY = (
    "domain is for sale", "this domain is for sale", "buy this domain", "domain may be for sale",
    "this web page is parked", "domain parking", "sedo domain parking", "parked free",
    "account suspended", "this account has been suspended", "default web page",
)


# ---------------------------------------------------------------------------
# decoding / parsing
# ---------------------------------------------------------------------------
def _is_known_codec(name: str) -> bool:
    try:
        codecs.lookup(name)
        return True
    except LookupError:
        return False


def decode_html(body: bytes, content_type: Optional[str] = None) -> str:
    """Decode with the declared charset (header, then <meta>), else UTF-8; never raises."""
    charset = None
    if content_type:
        m = _CHARSET_HEADER_RE.search(content_type)
        if m:
            charset = m.group(1)
    if not charset:
        m2 = _CHARSET_META_RE.search(body[:4096])
        if m2:
            charset = m2.group(1).decode("ascii", "ignore")
    if not charset or not _is_known_codec(charset):
        charset = "utf-8"
    return body.decode(charset, errors="replace")


def looks_like_html(body: bytes, content_type: Optional[str]) -> bool:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if "html" in ct:
        return True
    if ct:
        return False          # a declared non-HTML type is believed (json, image, pdf, ...)
    head = body[:1024].lstrip().lower()
    return any(tok in head for tok in (b"<!doctype html", b"<html", b"<head", b"<body", b"<title", b"<meta"))


_TAG_START_RE = re.compile(r"<[A-Za-z!?/]")


def parse_document(text: str):
    """Parse leniently. Returns the lxml root, or None if there is nothing to parse."""
    if not text.strip():
        return None
    if not _TAG_START_RE.search(text):
        # No tag-like text at all ("<", NUL bytes, ...). Different libxml2 builds either raise
        # or return an empty page for this; make it the same everywhere.
        return None
    parser = lhtml.HTMLParser(recover=True, remove_comments=True, remove_pis=True,
                              encoding="utf-8")
    try:
        return lhtml.document_fromstring(text.encode("utf-8", "replace"), parser=parser)
    except (etree.ParserError, etree.XMLSyntaxError, ValueError):
        return None


# ---------------------------------------------------------------------------
# link classification
# ---------------------------------------------------------------------------
def classify_link(href: Optional[str], base_url: str, base_reg: str, base_host: str) -> str:
    """'internal' | 'external' | 'null' (plan 13.5b).

    null     = empty, '#', 'about:blank' or a javascript: URL
    internal = relative, a non-http(s) scheme (mailto:, tel:, data:, ...), or the same
               registrable domain as the page (private suffixes ON)
    external = an http(s) URL on a different registrable domain
    """
    h = (href or "").strip()
    low = h.lower()
    if low in ("", "#", "about:blank") or low.startswith("javascript:"):
        return "null"
    m = _SCHEME_RE.match(h)
    if m and m.group(1).lower() not in ("http", "https"):
        return "internal"
    try:
        host = urlsplit(urljoin(base_url, h)).hostname
    except ValueError:
        return "null"
    if not host:
        return "null"
    host = domains.clean_host(host)
    if base_reg:
        return "internal" if domains.registrable_domain(host) == base_reg else "external"
    return "internal" if host == base_host else "external"


class _Counts:
    """internal / external / null counters for one link group."""
    __slots__ = ("internal", "external", "null")

    def __init__(self) -> None:
        self.internal = 0
        self.external = 0
        self.null = 0

    def add(self, kind: str) -> None:
        if kind == "external":
            self.external += 1
        elif kind == "null":
            self.null += 1
        else:
            self.internal += 1

    @property
    def total(self) -> int:
        return self.internal + self.external + self.null

    @property
    def internal_like(self) -> int:
        """Null links count as internal (same page): the public set has
        ratio_nullHyperlinks == 0 in every row, so its extractor never separated them."""
        return self.internal + self.null


def _pct(part: int, whole: int) -> float:
    return 0.0 if whole == 0 else round(100.0 * part / whole, 6)


def _zero_size(value: Optional[str]) -> bool:
    v = (value or "").strip().lower().replace("px", "")
    try:
        return float(v) == 0.0
    except ValueError:
        return False


def _iframe_hidden(el) -> bool:
    style = re.sub(r"\s+", "", (el.get("style") or "").lower())
    if "display:none" in style or "visibility:hidden" in style:
        return True
    if el.get("hidden") is not None:
        return True
    zero_w = _zero_size(el.get("width")) or "width:0" in style and "width:0." not in style
    zero_h = _zero_size(el.get("height")) or "height:0" in style and "height:0." not in style
    return bool(zero_w and zero_h)


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
def extract_features(root, base_url: str, base_reg: str, base_host: str,
                     label: str, errors: List[str]) -> Tuple[Dict[str, Any], int]:
    """The 12 features for one parsed page. Returns (features, script_count).

    Each group is wrapped separately: a failure makes only its features -1 and is logged
    in `errors` (plan 13.5b: extractor_error)."""
    feats: Dict[str, Any] = {name: -1 for name in WEBPAGE_FEATURES}
    script_count = 0

    groups = {k: _Counts() for k in ("href", "link", "media", "form", "css", "favicon")}
    form_actions: List[Tuple[str, str]] = []     # (kind, action) for forms with an action attribute
    favicon_external = False
    hidden_iframe = False
    title_text: Optional[str] = None
    capped = False

    try:
        for n, el in enumerate(root.iter()):
            if n >= MAX_ELEMENTS:
                capped = True
                break
            tag = el.tag
            if not isinstance(tag, str):
                continue
            tag = tag.lower()

            if tag == "a" or tag == "area":
                href = el.get("href")
                if href is not None:
                    groups["href"].add(classify_link(href, base_url, base_reg, base_host))
            elif tag == "link":
                href = el.get("href")
                if href is not None:
                    kind = classify_link(href, base_url, base_reg, base_host)
                    rel = (el.get("rel") or "").lower().split()
                    groups["link"].add(kind)
                    if "stylesheet" in rel:
                        groups["css"].add(kind)
                    parent = el.getparent()
                    if (any(tok in _ICON_REL for tok in rel) and parent is not None
                            and isinstance(parent.tag, str) and parent.tag.lower() == "head"):
                        groups["favicon"].add(kind)
                        if kind == "external":
                            favicon_external = True
            elif tag == "script":
                script_count += 1
                src = el.get("src")
                if src is not None:
                    groups["link"].add(classify_link(src, base_url, base_reg, base_host))
            elif tag == "meta":
                content = (el.get("content") or "").strip()
                if content.lower().startswith(("http://", "https://")):
                    groups["link"].add(classify_link(content, base_url, base_reg, base_host))
            elif tag in ("img", "audio", "video", "embed"):
                src = el.get("src")
                if src is not None:
                    groups["media"].add(classify_link(src, base_url, base_reg, base_host))
            elif tag == "iframe":
                src = el.get("src")
                if src is not None:
                    groups["media"].add(classify_link(src, base_url, base_reg, base_host))
                if _iframe_hidden(el):
                    hidden_iframe = True
            elif tag == "form":
                action = el.get("action")
                if action is not None:        # forms without an action attribute are not counted
                    kind = classify_link(action, base_url, base_reg, base_host)
                    groups["form"].add(kind)
                    form_actions.append((kind, action.strip()))
            elif tag == "title" and title_text is None:
                title_text = " ".join((el.text_content() or "").split())
    except Exception as exc:                                  # noqa: BLE001 - logged, never raised
        errors.append(f"extractor_error:walk:{type(exc).__name__}")
        return feats, script_count

    if capped:
        errors.append("element_cap_hit")

    # --- hyperlink counts (one pass over the six groups, like the dataset's extractor:
    #     stylesheets and icons are links too, so they are counted in more than one group)
    try:
        total = sum(g.total for g in groups.values())
        internal = sum(g.internal_like for g in groups.values())
        feats["nb_hyperlinks"] = total
        feats["ratio_intHyperlinks"] = 0.0 if total == 0 else round(internal / total, 6)
        feats["nb_extCSS"] = groups["css"].external
        feats["external_favicon"] = 1 if favicon_external else 0
        link = groups["link"]
        feats["links_in_tags"] = _pct(link.internal_like, link.total)
        media = groups["media"]
        feats["ratio_intMedia"] = _pct(media.internal_like, media.total)
        feats["ratio_extMedia"] = _pct(media.external, media.total)
    except Exception as exc:                                  # noqa: BLE001
        errors.append(f"extractor_error:links:{type(exc).__name__}")

    try:
        # login_form: a form whose action is empty/'#'/about:blank/javascript or on another
        # domain, or an internal action that is a bare "name.php" (the dataset's rule).
        flagged = any(kind in ("external", "null") for kind, _ in form_actions)
        if not flagged:
            flagged = any(_PHP_ACTION_RE.match(action) for kind, action in form_actions
                          if kind == "internal")
        feats["login_form"] = 1 if flagged else 0
    except Exception as exc:                                  # noqa: BLE001
        errors.append(f"extractor_error:login_form:{type(exc).__name__}")

    try:
        feats["iframe"] = 1 if hidden_iframe else 0
    except Exception as exc:                                  # noqa: BLE001
        errors.append(f"extractor_error:iframe:{type(exc).__name__}")

    try:
        feats["empty_title"] = 1 if not title_text else 0
        # Same direction as the public set: 1 = the domain name is NOT in the title.
        feats["domain_in_title"] = 0 if (label and title_text and label.lower() in title_text.lower()) else 1
    except Exception as exc:                                  # noqa: BLE001
        errors.append(f"extractor_error:title:{type(exc).__name__}")

    try:
        text = visible_text(root)
        feats["domain_with_copyright"] = _domain_with_copyright(text, label)
    except Exception as exc:                                  # noqa: BLE001
        errors.append(f"extractor_error:copyright:{type(exc).__name__}")

    return feats, script_count


def visible_text(root) -> str:
    """Text outside <script>/<style>/<noscript>. NOTE: this removes those elements from
    `root`, so call it after everything that needs them."""
    etree.strip_elements(root, "script", "style", "noscript", with_tail=False)
    text = " ".join(" ".join(root.itertext()).split())
    return text[:MAX_TEXT_CHARS]


def _domain_with_copyright(text: str, label: str) -> int:
    """1 = a copyright/trademark mark exists and the domain name is NOT within 50 characters
    of the first one; 0 = no mark, or the domain name is next to it (the dataset's rule)."""
    m = _COPYRIGHT_RE.search(text)
    if not m:
        return 0
    window = text[max(0, m.start() - 50): m.start() + 50].lower()
    return 0 if (label and label.lower() in window) else 1


# ---------------------------------------------------------------------------
# redirect hints and page verdicts
# ---------------------------------------------------------------------------
def meta_refresh_target(root) -> str:
    for el in root.iter("meta"):
        if (el.get("http-equiv") or "").strip().lower() != "refresh":
            continue
        m = _META_REFRESH_RE.match(el.get("content") or "")
        if m and m.group("url").strip() and float(m.group("delay")) <= META_REFRESH_MAX_DELAY:
            return m.group("url").strip()
    return ""


def script_redirect_target(root, body_len: int) -> str:
    """A trivial `location = "..."` redirect, only on tiny shell pages (see module docs)."""
    if body_len > JS_REDIRECT_MAX_BODY:
        return ""
    pieces: List[str] = []
    used = 0
    for el in root.iter("script"):
        txt = el.text or ""
        used += len(txt)
        if used > JS_SCRIPT_TEXT_CAP:
            break
        pieces.append(txt)
    blob = "\n".join(pieces)
    for rx in _JS_REDIRECT_RES:
        m = rx.search(blob)
        if m:
            target = m.group(1).strip()
            if target and target != "#" and not target.lower().startswith("javascript:"):
                return target
    return ""


def is_challenge_page(title: str, raw_head: str, text: str) -> bool:
    t = title.lower()
    if any(mark in t for mark in _CHALLENGE_TITLES):
        return True
    return len(text) < 1500 and any(mark in raw_head for mark in _CHALLENGE_BODY)


def is_parked_page(title: str, text: str) -> bool:
    t = title.lower().strip()
    if any(t == mark or (len(mark) > 6 and t.startswith(mark)) for mark in _PARKED_TITLES):
        return True
    snippet = text[:3000].lower()
    return len(text) < 3000 and any(mark in snippet for mark in _PARKED_BODY)


# ---------------------------------------------------------------------------
# the one entry point
# ---------------------------------------------------------------------------
def analyze_page(body: bytes, content_type: Optional[str], base_url: str,
                 want_features: bool = True) -> Dict[str, Any]:
    """Analyse one fetched page. `base_url` is the URL of THIS page (a hop of the chain);
    internal/external is judged against its registrable domain."""
    out: Dict[str, Any] = {
        "is_html": False, "redirect_kind": "", "redirect_url": "",
        "challenge": False, "parked": False, "features": {},
        "html_bytes": len(body), "script_count": -1, "near_empty": -1,
        "title": "", "errors": [],
    }
    errors: List[str] = out["errors"]

    if not looks_like_html(body, content_type):
        return out
    out["is_html"] = True

    try:
        host = domains.clean_host(urlsplit(base_url).hostname or "")
    except ValueError:
        host = ""
    base_reg = domains.registrable_domain(host)
    label = domains.domain_label(host)

    text = decode_html(body, content_type)
    root = parse_document(text)
    if root is None:
        errors.append("unparseable_or_empty")
        out["script_count"] = 0
        out["near_empty"] = 1
        return out

    # redirect hints first: a redirect shell is not the page we want features for
    target = meta_refresh_target(root)
    kind = "meta" if target else ""
    if not target:
        target = script_redirect_target(root, len(body))
        kind = "js" if target else ""
    if target:
        out["redirect_kind"], out["redirect_url"] = kind, target[:2048]

    feats, script_count = extract_features(root, base_url, base_reg, host, label, errors)
    out["script_count"] = script_count

    # title (cleaned) and plain text for the verdicts (visible_text() mutates `root`)
    title_el = root.find(".//title")
    title = " ".join((title_el.text_content() or "").split()) if title_el is not None else ""
    out["title"] = safety.clean_field(title, max_len=safety.MAX_TITLE)
    raw_head = text[:20000].lower()
    plain = visible_text(root)
    out["near_empty"] = 1 if len(plain) < 50 else 0

    if not target:
        out["challenge"] = is_challenge_page(title, raw_head, plain)
        out["parked"] = (not out["challenge"]) and is_parked_page(title, plain)
        if want_features and not out["challenge"] and not out["parked"]:
            out["features"] = feats
    return out
