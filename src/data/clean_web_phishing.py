"""Clean the merged web-phishing dataset down to the selected feature set.

Decisions and column mapping: docs/web_phishing_cleaning.md

Usage:
    python src/data/clean_web_phishing.py [input_csv] [output_csv]
"""
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "data" / "interim" / "hussein-train-and-test-merged.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "web_phishing_clean.csv"

# Special characters counted in the URL (scheme "://" excluded).
SPECIAL_CHARS = set("@?&%=_~*:,;$!#+|^ ")

# Path/file tokens common in phishing kits (same list behind the original phish_hints).
SUSPICIOUS_URL_TOKENS = [
    "wp", "login", "includes", "admin", "content", "site", "images", "js",
    "alibaba", "css", "myaccount", "dropbox", "themes", "plugins", "signin", "view",
]

# Social-engineering words used to lure users.
SUSPICIOUS_KEYWORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account",
    "secure", "security", "update", "bank", "banking", "confirm", "password",
    "webscr", "ebayisapi", "paypal", "billing", "unlock", "suspend", "wallet",
    "authenticate", "validation", "recover", "support", "free", "bonus", "lucky",
]

# Output name -> source column in the interim CSV (renamed to our names).
RENAME = {
    "length_url": "url_length",
    "nb_hyphens": "number_of_hyphens",
    "nb_slash": "number_of_slashes",
    "nb_subdomains": "subdomain_count",
    "shortening_service": "shortened_url_indicator",
    "ip": "has_ip_address",
}

URL_FEATURES = [
    "url_length", "https_presence", "number_of_hyphens", "number_of_slashes",
    "subdomain_count", "special_char_count", "suspicious_url_token_count",
    "shortened_url_indicator", "has_ip_address", "suspicious_keyword_count", "has_url",
    "page_rank", "path_length", "random_domain", "length_hostname", "tld_in_path",
]
# ratio_nullHyperlinks and sfh are not kept: they are 0 in every row of the source data.
WEBPAGE_FEATURES = [
    "nb_hyperlinks", "ratio_intHyperlinks", "nb_extCSS", "login_form",
    "external_favicon", "links_in_tags", "ratio_intMedia", "ratio_extMedia",
    "iframe", "empty_title", "domain_in_title", "domain_with_copyright",
]
EXTERNAL_FEATURES = [
    "whois_registered_domain", "domain_registration_length", "domain_age", "dns_record",
]
OUTPUT_COLUMNS = ["url"] + URL_FEATURES + WEBPAGE_FEATURES + EXTERNAL_FEATURES + ["status"]


def strip_scheme(url: str) -> str:
    return re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)


def special_char_count(url: str) -> int:
    return sum(ch in SPECIAL_CHARS for ch in strip_scheme(url))


def suspicious_url_token_count(url: str) -> int:
    lowered = url.lower()
    return sum(lowered.count(tok) for tok in SUSPICIOUS_URL_TOKENS)


def suspicious_keyword_count(url: str) -> int:
    words = re.split(r"[^a-z0-9-]+", url.lower())
    text = " " + " ".join(words) + " "
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text)


def path_length(url: str) -> int:
    try:
        return len(urlsplit(url).path)
    except ValueError:
        return len(strip_scheme(url).partition("/")[2]) + 1


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    url = df["url"].astype(str).str.strip()
    df["https_presence"] = url.str.lower().str.startswith("https://").astype(int)
    df["special_char_count"] = url.map(special_char_count)
    df["suspicious_url_token_count"] = url.map(suspicious_url_token_count)
    df["suspicious_keyword_count"] = url.map(suspicious_keyword_count)
    df["path_length"] = url.map(path_length)
    df["has_url"] = 1

    # Negative values are WHOIS lookup failures; mark them all as -1.
    for col in ["domain_age", "domain_registration_length"]:
        df.loc[df[col] < 0, col] = -1

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    raw = pd.read_csv(src)
    out = clean(raw)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)

    print(f"Read  {len(raw):>6} rows x {raw.shape[1]} cols from {src}")
    print(f"Wrote {len(out):>6} rows x {out.shape[1]} cols to   {dst}")
    print(out["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
