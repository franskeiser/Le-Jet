"""
schema.py -- the single definition of one output row (plan sections 13.10, 14.0, 15.2).

Everything that describes "what a row looks like" lives here, so the pipeline, the
storage layer, the exporter and the tests cannot drift apart. Standard library only.

A row is a plain dict. `new_row()` returns one with EVERY key present and every value at
its "unknown" default (-1 for numeric features), because the plan requires that every URL
yields exactly one row with every feature key (never omit a key).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict

from lejet import safety

# ---------------------------------------------------------------------------
# The model-input columns. These names/orders must equal
# src/features/model/rf_extractor.py (tests/test_schema.py checks it on the host).
# ---------------------------------------------------------------------------
URL_TEXT_FEATURES = [
    "url_length", "https_presence", "number_of_hyphens", "number_of_slashes",
    "subdomain_count", "special_char_count", "suspicious_url_token_count",
    "shortened_url_indicator", "has_ip_address", "suspicious_keyword_count",
    "path_length", "random_domain", "length_hostname", "tld_in_path",
]
PAGE_RANK = "page_rank"   # host-side lookup; absent from the VM export (plan 13.1, 15.2)
WEBPAGE_FEATURES = [
    "nb_hyperlinks", "ratio_intHyperlinks", "nb_extCSS", "login_form", "external_favicon",
    "links_in_tags", "ratio_intMedia", "ratio_extMedia", "iframe", "empty_title",
    "domain_in_title", "domain_with_copyright",
]
EXTERNAL_FEATURES = [
    "whois_registered_domain", "domain_registration_length", "domain_age", "dns_record",
]
# The 30 model inputs the VM itself produces (31 minus page_rank).
VM_FEATURES = URL_TEXT_FEATURES + WEBPAGE_FEATURES + EXTERNAL_FEATURES

# ---------------------------------------------------------------------------
# The public web-phishing set's columns (Hannousse & Yahiouche, 2021) that the lab exports next to the 31 model
# inputs, as `features_full`. The set has 87; fourteen are left out on purpose (DROPPED_FROM_FULL), so
# `features_full` has 73, in the public set's order and under its names. The model inputs (`features`) are NOT affected: two of the
# dropped columns (nb_hyphens, dns_record) are still model inputs under their cleaned names.
#   * URL-based (53)     -> lejet.features.hannousse_url   (computed from the original link, stored as hf_<name>)
#   * content (4 extra)  -> lejet.features.hannousse_html  (stored as hf_<name>)
#   * 12 content + 3 external -> the same values as the model inputs of the same name (no second copy)
#   * page_rank          -> -1 on the VM; the host adds it (scripts/add_full_features.py)
# ---------------------------------------------------------------------------
DATASET_COLUMNS = [
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
    "nb_hyperlinks", "ratio_intHyperlinks", "ratio_extHyperlinks", "ratio_nullHyperlinks", "nb_extCSS",
    "ratio_intRedirection", "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors", "login_form",
    "external_favicon", "links_in_tags", "submit_email", "ratio_intMedia", "ratio_extMedia", "sfh", "iframe",
    "popup_window", "safe_anchor", "onmouseover", "right_clic", "empty_title", "domain_in_title",
    "domain_with_copyright",
    "whois_registered_domain", "domain_registration_length", "domain_age", "web_traffic", "dns_record",
    "google_index", "page_rank",
]
assert len(DATASET_COLUMNS) == 87 and len(set(DATASET_COLUMNS)) == 87
# Columns left out of features_full, and why (the user's decision of 2026-10-14).
DROPPED_FROM_FULL = {
    # cannot be captured by the lab
    "ratio_intRedirection": "needs a request to every hyperlink of the page; the lab never fetches linked resources",
    "ratio_extRedirection": "needs a request to every hyperlink of the page; the lab never fetches linked resources",
    "ratio_intErrors": "needs a request to every hyperlink of the page; the lab never fetches linked resources",
    "ratio_extErrors": "needs a request to every hyperlink of the page; the lab never fetches linked resources",
    "google_index": "needs one Google search per link (against Google's terms)",
    # not needed
    "nb_or": "always 0 in the public set",
    "nb_hyphens": "weak (correlation -0.10); still a model input as number_of_hyphens",
    "dns_record": "weak (correlation 0.12); still a model input",
    "web_traffic": "weak (correlation 0.06); the Alexa rank it held no longer exists",
    # exact duplicates or no information (analysis of the public set's training split, 2026-10-14)
    "nb_subdomains": "an exact function of nb_dots (1, 2 or 3); still a model input as subdomain_count",
    "ratio_extHyperlinks": "exactly 1 - ratio_intHyperlinks in the lab (null links count as internal)",
    "ratio_nullHyperlinks": "0 in every row of the public set, so it cannot be evaluated; dropped to keep the table lean",
    "submit_email": "0 in every row of the public set, so it cannot be evaluated; dropped to keep the table lean",
    "sfh": "0 in every row of the public set, so it cannot be evaluated; dropped to keep the table lean",
}
FULL_FEATURES = [n for n in DATASET_COLUMNS if n not in DROPPED_FROM_FULL]
assert len(FULL_FEATURES) == 73
# Columns of `features_full` that take their value from the model-input column of the same name.
FULL_FROM_MODEL_INPUTS = [n for n in FULL_FEATURES if n in WEBPAGE_FEATURES or n in EXTERNAL_FEATURES]
# Columns the VM cannot fill: -1 here (the host adds page_rank; see scripts/add_full_features.py).
FULL_NOT_ON_VM = ["page_rank"]
# Columns stored in the row as hf_<name> (everything else is read from the model-input column or is -1 on the VM).
HF_STORED = [n for n in FULL_FEATURES if n not in FULL_FROM_MODEL_INPUTS and n not in FULL_NOT_ON_VM]

# ---------------------------------------------------------------------------
# fail_reason (plan 13.10). `read_timeout` is an addition: the plan's list has no value for
# "connected but the server stalled / trickled the body", and calling that "dead" would
# be wrong.
# ---------------------------------------------------------------------------
FAIL_REASONS = (
    "none",
    "rejected_url", "port_blocked",
    "dns_nxdomain", "dns_timeout", "dns_servfail", "dns_no_answer",
    "connect_refused", "connect_timeout", "read_timeout", "tls_error",
    "http_4xx", "http_5xx", "too_many_redirects", "redirect_to_blocked",
    "non_html", "too_large", "challenge_page", "parked_or_default_page",
    "render_timeout", "render_crash", "blocked_or_empty", "internal_error",
)

# Bookkeeping columns and their "unknown" defaults. (type of default = type of column.)
BOOKKEEPING_DEFAULTS: Dict[str, Any] = {
    "url_id": "",
    "batch_id": "",
    "fetch_ts": "",
    "url": "",                      # raw URL in the VM database; defanged on export
    "has_url": 1,
    "page_fetched": 0,              # == html_available; routing flag, never a model input
    "fetch_ok": 0,
    "fail_reason": "none",
    "rdap_ok": 0,
    "dns_ok": 0,
    "html_available": 0,
    "html_bytes": -1,
    "script_count": -1,
    "near_empty": -1,
    "body_truncated": 0,
    "stage_reached": 0,
    "stage_ms_dns": 0,
    "stage_ms_rdap": 0,
    "stage_ms_http": 0,
    "stage_ms_html": 0,
    "duration_ms": 0,
    "days_to_expiry": -1,
    "free_hosting_subdomain": 0,
    "hosting_platform": "",
    "platform_path_hosting": 0,
    "rdap_skipped": 0,
    "reg_domain": "",
    "punycode": 0,
    "final_url": "",
    "final_reg_domain": "",
    "final_differs": 0,
    "final_status": -1,
    "hop_count": 0,
    "chain_complete": 0,
    "scheme_used": "",
    "final_is_popular": 0,
    "orig_whois_registered_domain": -1,
    "orig_domain_registration_length": -1,
    "orig_domain_age": -1,
    "orig_dns_record": -1,
    # certificate-transparency age (crt.sh), looked up only where RDAP gave nothing (e.g. .ph).
    # NOT the registration date; kept in its own columns, never mixed into domain_age.
    "ct_first_cert_days": -1,       # days since the oldest public TLS certificate; -1 = unknown/none
    "ct_ok": 0,                     # the crt.sh lookup completed
    "registration_unavailable_tld": 0,   # 1 = this TLD has no RDAP service, so -1 age means "unknown"
}

# Extra fields (beyond the 30 features) that go into the export record.
EXTRA_EXPORT_FIELDS = [
    "dns_ok", "near_empty", "body_truncated", "stage_reached", "days_to_expiry",
    "free_hosting_subdomain", "hosting_platform", "platform_path_hosting", "rdap_skipped",
    "reg_domain", "punycode", "final_url", "final_reg_domain", "final_differs",
    "final_status", "hop_count", "chain_complete", "scheme_used", "final_is_popular",
    "orig_whois_registered_domain", "orig_domain_registration_length",
    "orig_domain_age", "orig_dns_record",
    "ct_first_cert_days", "ct_ok", "registration_unavailable_tld",
]
_DEFANGED_FIELDS = ("url", "final_url", "reg_domain", "final_reg_domain")
_STRING_FIELDS = {"url_id", "batch_id", "fetch_ts", "url", "fail_reason", "hosting_platform",
                  "reg_domain", "final_url", "final_reg_domain", "scheme_used"}
_MAX_URL_FIELD = 1024
_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_row(url_id: str, raw_url: str, batch_id: str) -> Dict[str, Any]:
    """A row with every key present and every value at its unknown default."""
    row: Dict[str, Any] = {name: -1 for name in VM_FEATURES}
    row.update({f"hf_{name}": -1 for name in HF_STORED})
    row.update(BOOKKEEPING_DEFAULTS)
    row["url_id"] = url_id
    row["batch_id"] = batch_id
    row["url"] = raw_url
    row["fetch_ts"] = utc_now_iso()
    return row


def status_for(fail_reason: str) -> str:
    """Terminal state stored per URL: 'done' or 'failed:<reason>' (plan 13.8)."""
    return "done" if fail_reason == "none" else f"failed:{fail_reason}"


def is_valid_url_id(url_id: str) -> bool:
    return bool(_HEX16.match(url_id or ""))


def clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Make every stored value inert: strip control/ANSI/bidi characters from strings, cap
    lengths, replace non-finite or non-numeric junk in numeric columns by -1."""
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if key in _STRING_FIELDS:
            limit = _MAX_URL_FIELD if key in ("url", "final_url") else safety.MAX_STR_FIELD
            out[key] = safety.clean_field(value, max_len=limit)
        elif isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, int):
            out[key] = value
        elif isinstance(value, float):
            out[key] = round(value, 6) if value == value and abs(value) != float("inf") else -1
        else:
            out[key] = -1
    if out.get("fail_reason") not in FAIL_REASONS:
        out["fail_reason"] = "internal_error"
    return out


def full_features(r: Dict[str, Any]) -> Dict[str, Any]:
    """The 87 public-set columns of one cleaned row, in the public set's order (-1 = not measured / unknown)."""
    out: Dict[str, Any] = {}
    for name in FULL_FEATURES:
        if name in FULL_FROM_MODEL_INPUTS:
            out[name] = r.get(name, -1)
        elif name in FULL_NOT_ON_VM:
            out[name] = -1
        else:
            out[name] = r.get(f"hf_{name}", -1)
    return out


def _defang_domain(domain: str) -> str:
    return safety.strip_dangerous_chars(domain).replace(".", "[.]")


def export_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """The JSONL record for one URL (plan 15.2). URLs and domains are defanged; the
    `features` object holds the 30 VM-side model inputs (page_rank is added on the host)."""
    r = clean_row(row)
    features = {name: r.get(name, -1) for name in VM_FEATURES}
    rec: Dict[str, Any] = {
        "url_id": r["url_id"],
        "batch_id": r["batch_id"],
        "fetch_ts": r["fetch_ts"],
        "url": safety.defang_url(r["url"]),
        "has_url": 1,
        "page_fetched": r["page_fetched"],
        "fetch_ok": bool(r["fetch_ok"]),
        "fail_reason": r["fail_reason"],
        "rdap_ok": bool(r["rdap_ok"]),
        "html_available": bool(r["html_available"]),
        "html_bytes": r["html_bytes"],
        "script_count": r["script_count"],
        "stage_ms": {
            "dns": r["stage_ms_dns"], "rdap": r["stage_ms_rdap"],
            "http": r["stage_ms_http"], "html": r["stage_ms_html"],
        },
        "duration_ms": r["duration_ms"],
        "features": features,
        "features_full": full_features(r),
    }
    for name in EXTRA_EXPORT_FIELDS:
        value = r.get(name, BOOKKEEPING_DEFAULTS.get(name))
        if name in _DEFANGED_FIELDS:
            value = (safety.defang_url(value) if name.endswith("url") else _defang_domain(value))
        rec[name] = value
    return rec
