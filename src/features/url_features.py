from urllib.parse import urlparse
import re
import tldextract

from src.features.text_preprocessing import primary_url

def _ensure_scheme(url: str) -> str:
    if '://' not in url:
        return 'http://' + url
    return url

def _parse(url: str):
    url = _ensure_scheme(url)
    parsed = urlparse(url)
    hostname = (parsed.hostname or '').lower()
    return url, parsed, hostname

def url_length(url: str) -> int:
    return len(url)

def https_presence(parsed) -> int:
    return 1 if parsed.scheme.lower() == 'https' else 0

def number_of_hyphens(url: str) -> int:
    return url.count('-')

def number_of_slashes(url: str) -> int:
    return url.count('/')

SUSPICIOUS_KEYWORDS = [
    'verify', 'secure', 'login', 'update', 'confirm', 'account',
    'suspend', 'urgent', 'claim', 'password', 'signin', 'banking',
]

def suspicious_keyword_count(url: str) -> int:
    lowered = url.lower()
    return sum(lowered.count(kw) for kw in SUSPICIOUS_KEYWORDS)

SHORTENER_DOMAINS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'rebrand.ly', 'cutt.ly', 'shorturl.at', 'tiny.cc',
}

def shortened_url_indicator(hostname: str) -> int:
    return 1 if hostname in SHORTENER_DOMAINS else 0

import ipaddress

def has_ip_address(hostname: str) -> int:
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0

IPV4_RE = re.compile(r'^(25[0-5]|2[0-4]\d|1?\d{1,2})(\.(25[0-5]|2[0-4]\d|1?\d{1,2})){3}$')

def subdomain_count(hostname: str) -> int:
    if not hostname or has_ip_address(hostname):
        return 0
    ext = tldextract.extract(hostname)
    if not ext.subdomain:
        return 0
    return len(ext.subdomain.split('.'))

def special_char_count(url: str) -> int:
    return sum(1 for c in url if not c.isalnum() and c not in ('.', '-', '/'))

def suspicious_url_token_count(parsed) -> int:
    target = (parsed.path + '?' + parsed.query).lower()
    return sum(target.count(kw) for kw in SUSPICIOUS_KEYWORDS)

# Explainability helpers -- NOT part of the model's feature vector, so adding/
# changing these has no effect on training or on what predict.py feeds the
# model. They exist only so app/app.py can tell the user WHICH keywords from
# SUSPICIOUS_KEYWORDS actually matched (same scope as the two count functions
# above -- suspicious_keyword_count over the whole URL, suspicious_url_token_count
# over path+query only) instead of just reporting a count.
def suspicious_keyword_matches(url: str) -> list[str]:
    lowered = url.lower()
    return [kw for kw in SUSPICIOUS_KEYWORDS if kw in lowered]

def suspicious_url_token_matches(parsed) -> list[str]:
    target = (parsed.path + '?' + parsed.query).lower()
    return [kw for kw in SUSPICIOUS_KEYWORDS if kw in target]

def get_suspicious_keyword_matches(raw_url: str) -> dict:
    url, parsed, _ = _parse(raw_url)
    return {
        'keyword_matches': suspicious_keyword_matches(url),
        'token_matches': suspicious_url_token_matches(parsed),
    }

def extract_url_features(raw_url: str) -> dict:
    url, parsed, hostname = _parse(raw_url)
    return {
        'url_length': url_length(url),
        'https_presence': https_presence(parsed),
        'number_of_hyphens': number_of_hyphens(url),
        'number_of_slashes': number_of_slashes(url),
        'suspicious_keyword_count': suspicious_keyword_count(url),
        'shortened_url_indicator': shortened_url_indicator(hostname),
        'has_ip_address': has_ip_address(hostname),
        'subdomain_count': subdomain_count(hostname),
        'special_char_count': special_char_count(url),
        'suspicious_url_token_count': suspicious_url_token_count(parsed),
    }

# Sentinel-value convention: count-based features are
# honestly 0 with no URL present; measurement/indicator features get -1 so
# a no-URL row can never be mistaken for "URL present, scored low."
def no_url_features() -> dict:
    return {
        'url_length': -1,
        'https_presence': -1,
        'number_of_hyphens': 0,
        'number_of_slashes': 0,
        'suspicious_keyword_count': 0,
        'shortened_url_indicator': -1,
        'has_ip_address': -1,
        'subdomain_count': 0,
        'special_char_count': 0,
        'suspicious_url_token_count': 0,
        'has_url': 0,
    }

def get_url_feature_vector(text: str) -> dict:
    url = primary_url(text)
    if url is None:
        return no_url_features()
    features = extract_url_features(url)
    features['has_url'] = 1
    return features

