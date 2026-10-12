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
    }
    for name in EXTRA_EXPORT_FIELDS:
        value = r.get(name, BOOKKEEPING_DEFAULTS.get(name))
        if name in _DEFANGED_FIELDS:
            value = (safety.defang_url(value) if name.endswith("url") else _defang_domain(value))
        rec[name] = value
    return rec
