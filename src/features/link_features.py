"""Find links in SMS text and compute the 14 link-text features the url_only extractor uses.

The extractor was trained on the public web set, so every feature here must be measured
exactly as in that set:
  * 5 features were computed by us -- reused from src/data/clean_web_phishing.py.
  * 9 came pre-computed in the dataset (Hannousse & Yahiouche's extractor). They are
    re-implemented below and checked against all 11,430 rows (tests/test_link_features.py):
        url_length 99.95%, number_of_hyphens 100%, number_of_slashes 100%,
        length_hostname 100%, subdomain_count 99.99%, has_ip_address 100%,
        shortened_url_indicator 100%, tld_in_path 100%, random_domain ~92% (see below).
    Their quirks are copied on purpose (e.g. subdomain_count counts the dots in the whole
    URL, and an IP host with no domain ending gets tld_in_path = 1).

No network access: everything comes from the link string.
"""
import re
from urllib.parse import urlparse, urlsplit

import tldextract

from src.data.clean_web_phishing import (
    path_length, special_char_count, suspicious_keyword_count, suspicious_url_token_count,
)

# Same order as rf_extractor.URL_TEXT_FEATURES.
LINK_TEXT_FEATURES = [
    "url_length", "https_presence", "number_of_hyphens", "number_of_slashes",
    "subdomain_count", "special_char_count", "suspicious_url_token_count",
    "shortened_url_indicator", "has_ip_address", "suspicious_keyword_count",
    "path_length", "random_domain", "length_hostname", "tld_in_path",
]
NO_LINK = -1

# ---------------------------------------------------------------------------------------
# Finding links in SMS text
# ---------------------------------------------------------------------------------------

# Offline public suffix list (bundled snapshot). Private suffixes ON for grouping and
# lookups, so x.blogspot.com is its own domain (same rule as lejet.urlnorm).
_EXTRACT_PRIVATE = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
# Private suffixes OFF for the features, matching the dataset's extractor.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# Candidate link: optional (possibly defanged) scheme, optional www., host labels joined by
# "." or "[.]", then an optional path.
_LINK_RE = re.compile(
    r"(?i)(?P<scheme>h(?:tt|xx)ps?://)?(?P<www>www\.)?"
    r"(?P<host>(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.|\[\.\]|\(\.\)))+[a-z]{2,24})"
    r"(?P<path>/[^\s<>\"']*)?"
)
# Short TLDs that are also common English words; a bare "ok.so" or "that.is" without a
# scheme, www. or path is almost always prose, not a link.
_AMBIGUOUS_BARE = {
    "am", "as", "at", "be", "by", "do", "go", "hi", "in", "is", "it", "me", "my", "no",
    "of", "ok", "on", "or", "so", "to", "tv", "up", "us", "we",
}
# Word-like gTLDs that catch two words joined by a period ("tomorrow.call", "day.Click").
# A bare match with one of these is kept only if its name has a digit or hyphen
# ("my-bdo-advisory.click"), which prose almost never does.
_PROSE_TLDS = {
    "call", "click", "download", "google", "how", "hsbc", "ice", "like", "love", "meet",
    "now", "onion", "pa", "search", "today", "xxx", "you",
}
# Shorteners hide the real destination; their own domain lookup says nothing about the scam.
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "rb.gy",
    "cutt.ly", "shorturl.at", "tiny.cc", "s.id", "rebrand.ly", "t.ly", "bit.do", "v.gd",
    "shorturl.asia", "u.to", "qr.ae", "lnkd.in",
}
# Trailing punctuation SMS text glues onto a link ("visit gcash.com.").
_TRAILING_JUNK = ".,);]}>\"'!?"


def extract_links(text: str) -> list[dict]:
    links = []
    for m in _LINK_RE.finditer(text):
        if m.start() > 0 and text[m.start() - 1] == "@":
            continue  # the domain of an email address, not a link
        host = re.sub(r"\[\.\]|\(\.\)", ".", m.group("host")).lower()
        parts = _EXTRACT_PRIVATE(host)
        if not parts.suffix or not parts.domain:
            continue  # not a real domain ending (e.g. "k.then")
        explicit = m.group("scheme") or m.group("www") or m.group("path")
        # Very short bare names are mostly prose ("6.House", "W.PLUS"), except under a
        # two-part ending like edu.ph or com.ph ("up.edu.ph"), which prose rarely produces.
        too_short = len(parts.domain) < 3 and "." not in parts.suffix
        if not explicit and (parts.suffix in _AMBIGUOUS_BARE or too_short):
            continue
        if not explicit and parts.suffix in _PROSE_TLDS and parts.domain.isalpha():
            continue
        raw = m.group(0)
        links.append({
            "raw": raw, "host": host,
            "domain": parts.top_domain_under_public_suffix,
            # Where the link sits in the text, minus trailing punctuation, so the CNN's
            # <URL> masking replaces exactly the links has_url counts.
            "start": m.start(), "end": m.start() + len(raw.rstrip(_TRAILING_JUNK)),
        })
    return links


def normalize_link(raw: str) -> str:
    # The dataset's URLs all start with http:// or https://, so SMS links get the same
    # shape before any feature is measured (plan section 4, step 6). A link without a
    # scheme gets http:// -- https_presence is then 0, since HTTPS was not stated.
    url = raw.rstrip(_TRAILING_JUNK)
    url = re.sub(r"\[\.\]|\(\.\)", ".", url)
    url = re.sub(r"(?i)^hxxp", "http", url)
    if not re.match(r"(?i)^https?://", url):
        url = "http://" + url
    return url

# ---------------------------------------------------------------------------------------
# The 9 dataset-defined features (re-implemented from the dataset's extractor)
# ---------------------------------------------------------------------------------------

_IP_RE = re.compile(
    r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\."
    r"([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\/)|"  # IPv4
    r"((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)|"  # hex IPv4
    r"(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|"  # IPv6
    r"[0-9a-fA-F]{7}"  # as in the original: any 7 hex characters
)
_SHORTENER_RE = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|"
    r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|"
    r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
    r"tr\.im|link\.zip\.net"
)


def _netloc(url: str) -> str:
    try:
        return urlparse(url).netloc
    except ValueError:
        return ""


def _hostname(url: str) -> str:
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


def subdomain_count(url: str) -> int:
    # Dataset definition: dots in the WHOLE URL, capped at 3 (not the hostname's labels).
    dots = url.count(".")
    return 1 if dots == 1 else 2 if dots == 2 else 3


def has_ip_address(url: str) -> int:
    return int(bool(_IP_RE.search(url)))


def shortened_url_indicator(url: str) -> int:
    return int(bool(_SHORTENER_RE.search(url)))


def tld_in_path(url: str) -> int:
    # Dataset definition: the host's suffix appears anywhere after the host. With no suffix
    # (an IP host) the original checks for an empty string, which always matches -> 1.
    netloc = _netloc(url)
    suffix = _EXTRACT(netloc).suffix
    rest = url.split(netloc, 1)[-1] if netloc else url
    return int(suffix.lower() in rest.lower())


def random_domain(url: str) -> int:
    # Dataset definition: the nostril "nonsense" test (github.com/casics/nostril) on the
    # hostname. Only ~92% of dataset values are reproducible -- the original's exact input
    # is unknown. nostril refuses very short strings; those count as not random.
    from nostril import nonsense
    try:
        return int(nonsense(_hostname(url)))
    except ValueError:
        return 0


def link_text_features(url: str) -> dict:
    """The 14 features for one normalized URL (see normalize_link)."""
    return {
        "url_length": len(url),
        "https_presence": int(url.lower().startswith("https://")),
        "number_of_hyphens": url.count("-"),
        "number_of_slashes": url.count("/"),
        "subdomain_count": subdomain_count(url),
        "special_char_count": special_char_count(url),
        "suspicious_url_token_count": suspicious_url_token_count(url),
        "shortened_url_indicator": shortened_url_indicator(url),
        "has_ip_address": has_ip_address(url),
        "suspicious_keyword_count": suspicious_keyword_count(url),
        "path_length": path_length(url),
        "random_domain": random_domain(url),
        "length_hostname": len(_hostname(url)),
        "tld_in_path": tld_in_path(url),
    }


def no_link_features() -> dict:
    return {name: NO_LINK for name in LINK_TEXT_FEATURES}
