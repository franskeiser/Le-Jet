"""
hannousse_url.py -- the 56 URL-based features of the public web-phishing set (Hannousse & Yahiouche,
2021, "Towards benchmark datasets for machine learning based website phishing detection", Table 2),
under the dataset's own column names and in its column order.

Pure functions of the link text (plus, for two features, the redirect chain the HTTP stage already
recorded). No network, no file access, nothing is executed or fetched. Every definition below was
checked against the 11,430 URLs of the public set (tests/test_hannousse_url.py and
docs/hannousse_87_features.md list the match rate of every feature).

Values are the dataset's: counts are integers, ratios are floats, flags are 0/1. `-1` is never
produced here: a link is always measurable. The caller uses -1 when there was no link at all.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urlparse

import tldextract

from lejet.features import hannousse_lists as L

# The 56 names, in the dataset's column order (columns 2..57 of the public CSV).
URL_FEATURES: List[str] = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and", "nb_or",
    "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon", "nb_comma",
    "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash", "http_in_path",
    "https_token", "ratio_digits_url", "ratio_digits_host", "punycode", "port", "tld_in_path",
    "tld_in_subdomain", "abnormal_subdomain", "nb_subdomains", "prefix_suffix", "random_domain",
    "shortening_service", "path_extension", "nb_redirection", "nb_external_redirection",
    "length_words_raw", "char_repeat", "shortest_words_raw", "shortest_word_host", "shortest_word_path",
    "longest_words_raw", "longest_word_host", "longest_word_path", "avg_words_raw", "avg_word_host",
    "avg_word_path", "phish_hints", "domain_in_brand", "brand_in_subdomain", "brand_in_path",
    "suspecious_tld", "statistical_report",
]
assert len(URL_FEATURES) == 56

# Offline public-suffix view with private suffixes OFF, like the dataset's extractor.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_SPLIT = re.compile(r"[-./?=@&%:_]")
_IP_RE = re.compile(
    r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\."
    r"([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\/)|"
    r"((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)|"
    r"(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|"
    r"[0-9a-fA-F]{7}"
)
_PORT_RE = re.compile(
    r"^[a-z][a-z0-9+\-.]*://([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\]):([0-9]+)")
_ABNORMAL_FIRST_RE = re.compile(r"^(\d|w{1,3}[\d-])")
_PREFIX_SUFFIX_RE = re.compile(r"https?://[^\-]+-[^\-]+/")


def _parts(url: str):
    try:
        p = urlparse(url)
    except ValueError:
        p = urlparse("")
    try:
        ext = _EXTRACT(url)
    except Exception:                                       # noqa: BLE001 - never let a odd link raise
        ext = _EXTRACT("")
    return p, ext


def _abnormal_subdomain(hostname: str, subdomain: str, tld: str) -> int:
    """A raw-IP host, or a first sub-domain label that starts with a digit or looks like www1 / ww2 / www-x
    (the paper's 'w[w]?[0-9]*' pattern). Plain 'www' is normal. Fitted on the public set: 99.6% of its rows."""
    if not tld and re.fullmatch(r"[\d.]+", hostname or ""):
        return 1
    first = subdomain.split(".")[0] if subdomain else ""
    return int(bool(first) and bool(_ABNORMAL_FIRST_RE.match(first)))


def _words(text: str) -> List[str]:
    return [w for w in _SPLIT.split(text.lower()) if w]


def _stats(words: Sequence[str]):
    if not words:
        return 0, 0, 0.0
    lens = [len(w) for w in words]
    return min(lens), max(lens), mean(lens)


def _char_repeat(words: Iterable[str]) -> int:
    """Runs of 2, 3, 4 and 5 identical characters inside a word, all windows counted."""
    total = 0
    for word in words:
        for size in (2, 3, 4, 5):
            for i in range(len(word) - size + 1):
                if len(set(word[i:i + size])) == 1:
                    total += 1
    return total


def _random_domain(hostname: str) -> int:
    from nostril import nonsense
    try:
        return int(nonsense(hostname))
    except ValueError:
        return 0


def url_features(url: str) -> Dict[str, float]:
    """The 56 features of one normalised link (it starts with http:// or https://)."""
    p, ext = _parts(url)
    hostname = (p.hostname or "")
    path = p.path or ""
    domain, subdomain, tld = ext.domain.lower(), ext.subdomain.lower(), ext.suffix.lower()
    low = url.lower()

    # "path" for the word statistics is everything after the host, query included; for a raw-IP link (no domain
    # ending) the dataset's extractor used the link without its scheme (the address is counted a second time).
    netloc = p.netloc
    rest = url.split(netloc, 1)[-1] if netloc else url
    word_path = rest if tld else re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", url)
    w_domain, w_sub, w_path = _words(domain), _words(subdomain), _words(word_path)
    w_host = w_domain + w_sub
    w_raw = w_host + w_path
    s_raw, l_raw, a_raw = _stats(w_raw)
    s_host, l_host, a_host = _stats(w_host)
    s_path, l_path, a_path = _stats(w_path)

    dslash = [m.start() for m in re.finditer("//", url)]
    host_digits = sum(c.isdigit() for c in hostname)
    url_digits = sum(c.isdigit() for c in url)

    out: Dict[str, float] = {
        "length_url": len(url),
        "length_hostname": len(hostname),
        "ip": int(bool(_IP_RE.search(url))),
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": url.count("@"),
        "nb_qm": url.count("?"),
        "nb_and": url.count("&"),
        "nb_or": url.count("|"),
        "nb_eq": url.count("="),
        "nb_underscore": url.count("_"),
        "nb_tilde": url.count("~"),
        "nb_percent": url.count("%"),
        "nb_slash": url.count("/"),
        "nb_star": url.count("*"),
        "nb_colon": url.count(":"),
        "nb_comma": url.count(","),
        "nb_semicolumn": url.count(";"),
        "nb_dollar": url.count("$"),
        "nb_space": url.count(" ") + url.count("%20"),
        "nb_www": sum("www" in w for w in w_raw),
        "nb_com": sum("com" in w for w in w_raw),
        "nb_dslash": int(bool(dslash) and dslash[-1] > 6),
        "http_in_path": rest.count("http"),
        "https_token": 0 if p.scheme == "https" else 1,
        "ratio_digits_url": url_digits / len(url) if url else 0.0,
        "ratio_digits_host": host_digits / len(hostname) if hostname else 0.0,
        "punycode": int("xn--" in low),
        "port": int(bool(_PORT_RE.search(low))),
        "tld_in_path": int(tld in rest.lower()),
        "tld_in_subdomain": int(tld in subdomain),
        "abnormal_subdomain": _abnormal_subdomain(hostname, subdomain, tld),
        "nb_subdomains": {1: 1, 2: 2}.get(url.count("."), 3),
        "prefix_suffix": int(bool(_PREFIX_SUFFIX_RE.search(url))),
        "random_domain": _random_domain(hostname),
        "shortening_service": int(bool(L.SHORTENER_RE.search(url))),
        "path_extension": int(path.lower().endswith((".txt", ".exe", ".js"))),
        "nb_redirection": 0,                    # filled from the recorded redirect chain, see redirection_features()
        "nb_external_redirection": 0,
        "length_words_raw": len(w_raw),
        "char_repeat": _char_repeat(w_raw),
        "shortest_words_raw": s_raw, "shortest_word_host": s_host, "shortest_word_path": s_path,
        "longest_words_raw": l_raw, "longest_word_host": l_host, "longest_word_path": l_path,
        "avg_words_raw": a_raw, "avg_word_host": a_host, "avg_word_path": a_path,
        "phish_hints": sum(low.count(h) for h in L.PHISH_HINTS),
        "domain_in_brand": int(domain in L.BRANDS),
        "brand_in_subdomain": int(any(f".{b}." in f".{subdomain}." and b not in domain for b in L.BRANDS)),
        "brand_in_path": int(any(f".{b}." in path.lower() and b not in domain for b in L.BRANDS)),
        "suspecious_tld": int(tld in L.SUSPICIOUS_TLDS),
        "statistical_report": 0,
    }
    # statistical_report: 0, 1 (matches a top phishing host pattern) or 2 (its address was on the published list of
    # top phishing IPs). The dataset resolved the address in 2020; here the table reconstructed from the public
    # set is looked up by registrable domain (or raw IP), then the host patterns.
    key = hostname.lower() if not tld else f"{domain}.{tld}"
    out["statistical_report"] = L.STAT_HOSTS.get(key, int(bool(L.STAT_DOMAIN_RE.search(url))))
    return out


def redirection_features(original_url: str, redirect_hosts: Sequence[str]) -> Dict[str, int]:
    """nb_redirection / nb_external_redirection from the redirect chain the HTTP stage recorded.

    `redirect_hosts` are the hosts of the responses that REDIRECTED (every hop but the last), in order. The
    dataset's extractor counted `response.history` the same way: all redirects, and those whose URL does not
    contain the original domain name."""
    domain = _EXTRACT(original_url).domain.lower()
    hosts = [h.lower() for h in redirect_hosts]
    return {
        "nb_redirection": len(hosts),
        "nb_external_redirection": sum(1 for h in hosts if domain not in h),
    }
