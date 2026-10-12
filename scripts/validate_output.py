#!/usr/bin/env python
"""
validate_output.py -- the HOST-SIDE validator for what leaves the analysis VM
(WEB_INSPECTION_PLAN.md sections 12.5, 13.2, 15.2). Run it with `python -I`:

    python -I scripts/validate_output.py D:\\LeJetData\\quarantine\\b0007\\b0007.jsonl.gz ^
        --input D:\\LeJetData\\input\\urls_push.txt --clean-dir D:\\LeJetData\\clean

Everything in the file is treated as HOSTILE data (the VM may have been compromised by a
phishing page): it is checked strictly and only numbers, booleans and short sanitized strings
survive. Nothing is ever executed, opened in a browser or interpreted as a path.

Checks (any failure => exit code 1 and NOTHING is written to the clean directory):
  * size caps on the compressed file, on the decompressed stream (zip bombs) and on every line;
  * the sha256 sidecar, if present, matches;
  * every line is strict JSON (NaN / Infinity rejected), one object, no duplicate url_id;
  * exactly the expected keys, no more and no less, with the expected types;
  * features: finite numbers, -1 or within the documented range;
  * strings: no control / ANSI / bidi characters, length caps, no clickable URL;
  * spreadsheet formula leaders in any string field;
  * fail_reason is one of the known values; indicator columns agree with each other;
  * row count equals the input file (or --expected-rows), and every input url_id is present.

Output (only on success): <clean-dir>/<batch>.clean.jsonl and, with --csv, a CSV made safe for
spreadsheets (formula characters neutralised).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))        # `python -I` ignores PYTHONPATH

from lejet import safety, schema                            # noqa: E402

MAX_COMPRESSED = 300 * 1024 * 1024
MAX_DECOMPRESSED = 2 * 1024 * 1024 * 1024
MAX_LINE = 16 * 1024
MAX_PROBLEMS_SHOWN = 20

_URL_ID = re.compile(r"^[0-9a-f]{16}$")
_BAD_CHARS = re.compile("[\x00-\x1f\x7f\x80-\x9f\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_FORMULA = ("=", "+", "-", "@", "\t", "\r", "\n")

# Value ranges of the model inputs (-1 is always allowed: unknown).
_RANGE_01 = {"ratio_intHyperlinks"}
_RANGE_PCT = {"links_in_tags", "ratio_intMedia", "ratio_extMedia"}
_BINARY = {"https_presence", "shortened_url_indicator", "has_ip_address", "random_domain", "tld_in_path",
           "login_form", "external_favicon", "iframe", "empty_title", "domain_in_title",
           "domain_with_copyright", "whois_registered_domain", "dns_record"}

_STR_FIELDS = {"url_id": 16, "batch_id": 64, "fetch_ts": 20, "url": 1100, "fail_reason": 40,
               "hosting_platform": 100, "reg_domain": 300, "final_url": 1100, "final_reg_domain": 300,
               "scheme_used": 8}
_INT_EXTRAS = ["dns_ok", "near_empty", "body_truncated", "stage_reached", "days_to_expiry",
               "free_hosting_subdomain", "platform_path_hosting", "rdap_skipped", "punycode",
               "final_differs", "final_status", "hop_count", "chain_complete", "final_is_popular",
               "orig_whois_registered_domain", "orig_domain_registration_length", "orig_domain_age",
               "orig_dns_record", "ct_first_cert_days", "ct_ok", "registration_unavailable_tld"]

EXPECTED_KEYS = {
    "url_id", "batch_id", "fetch_ts", "url", "has_url", "page_fetched", "fetch_ok", "fail_reason",
    "rdap_ok", "html_available", "html_bytes", "script_count", "stage_ms", "duration_ms", "features",
} | set(schema.EXTRA_EXPORT_FIELDS)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))


def check_string(name: str, v: Any, limit: int) -> Optional[str]:
    if not isinstance(v, str):
        return f"{name}: not a string"
    if len(v) > limit:
        return f"{name}: longer than {limit}"
    if _BAD_CHARS.search(v):
        return f"{name}: control/ANSI/bidi character"
    if v.startswith(_FORMULA) and name not in ("url",):
        return f"{name}: starts with a spreadsheet formula character"
    if re.search(r"(?i)\bhttps?://", v):
        return f"{name}: contains a clickable URL (must be defanged)"
    return None


def check_record(rec: Any) -> List[str]:
    """All problems of one parsed JSONL record (empty list = valid)."""
    if not isinstance(rec, dict):
        return ["record is not an object"]
    errs: List[str] = []
    keys = set(rec)
    if keys != EXPECTED_KEYS:
        extra, missing = sorted(keys - EXPECTED_KEYS), sorted(EXPECTED_KEYS - keys)
        if extra:
            errs.append(f"unexpected keys: {extra[:5]}")
        if missing:
            errs.append(f"missing keys: {missing[:5]}")
        return errs

    for name, limit in _STR_FIELDS.items():
        e = check_string(name, rec[name], limit)
        if e:
            errs.append(e)
    if isinstance(rec["url_id"], str) and not _URL_ID.match(rec["url_id"]):
        errs.append("url_id is not 16 lower-case hex characters")
    if isinstance(rec["fetch_ts"], str) and not _TS.match(rec["fetch_ts"]):
        errs.append("fetch_ts is not an ISO UTC timestamp")
    if rec["fail_reason"] not in schema.FAIL_REASONS:
        errs.append(f"unknown fail_reason {str(rec['fail_reason'])[:30]!r}")
    if isinstance(rec["url"], str) and rec["url"] and not rec["url"].startswith(("hxxp://", "hxxps://")) \
            and "[.]" not in rec["url"] and "." in rec["url"]:
        errs.append("url is not defanged")

    for name in ("fetch_ok", "rdap_ok", "html_available"):
        if not isinstance(rec[name], bool):
            errs.append(f"{name}: not a boolean")
    for name in ("has_url", "page_fetched", "html_bytes", "script_count", "duration_ms"):
        if not _is_int(rec[name]):
            errs.append(f"{name}: not an integer")
    if rec["has_url"] != 1:
        errs.append("has_url must be 1")

    stage = rec["stage_ms"]
    if not (isinstance(stage, dict) and set(stage) == {"dns", "rdap", "http", "html"}
            and all(_is_int(v) and 0 <= v < 10_000_000 for v in stage.values())):
        errs.append("stage_ms: wrong shape or values")

    feats = rec["features"]
    if not isinstance(feats, dict) or list(feats) != schema.VM_FEATURES:
        errs.append("features: wrong keys or order")
    else:
        for name, v in feats.items():
            if not _is_num(v):
                errs.append(f"features.{name}: not a finite number")
                continue
            if v < -1:
                errs.append(f"features.{name}: below -1")
            elif name in _RANGE_01 and v > 1:
                errs.append(f"features.{name}: above 1")
            elif name in _RANGE_PCT and v > 100:
                errs.append(f"features.{name}: above 100")
            elif name in _BINARY and v not in (-1, 0, 1):
                errs.append(f"features.{name}: not 0, 1 or -1")
            elif v > 10**9:
                errs.append(f"features.{name}: implausibly large")

    for name in _INT_EXTRAS:
        v = rec.get(name)
        if not _is_int(v) or v < -1 or v > 10**9:
            errs.append(f"{name}: not a plausible integer")
    for name in ("html_bytes", "script_count", "duration_ms"):
        if _is_int(rec[name]) and not -1 <= rec[name] <= 10**10:
            errs.append(f"{name}: out of range")

    # indicator columns must agree with each other (plan 13.10)
    if isinstance(rec["fetch_ok"], bool) and isinstance(rec["fail_reason"], str):
        if rec["fetch_ok"] != (rec["fail_reason"] == "none"):
            errs.append("fetch_ok disagrees with fail_reason")
    if isinstance(rec["html_available"], bool) and rec["page_fetched"] != int(rec["html_available"]):
        errs.append("page_fetched disagrees with html_available")
    if isinstance(feats, dict) and rec.get("html_available") is False:
        if any(feats.get(n) != -1 for n in schema.WEBPAGE_FEATURES):
            errs.append("webpage features present although html_available is false")
    if isinstance(rec["rdap_ok"], bool) and rec["rdap_ok"] is False and isinstance(feats, dict):
        if feats.get("domain_age") != -1 or feats.get("domain_registration_length") != -1:
            errs.append("RDAP features present although rdap_ok is false")
    return errs


def read_lines_bounded(path: str) -> Tuple[List[bytes], List[str]]:
    """Decompress with hard caps. Returns (lines, problems)."""
    problems: List[str] = []
    if os.path.getsize(path) > MAX_COMPRESSED:
        return [], [f"compressed file larger than {MAX_COMPRESSED} bytes"]
    lines: List[bytes] = []
    total = 0
    try:
        with gzip.open(path, "rb") as fh:
            buf = b""
            while True:
                chunk = fh.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DECOMPRESSED:
                    return [], [f"decompressed data exceeds {MAX_DECOMPRESSED} bytes (zip bomb?)"]
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line, buf = buf[:nl], buf[nl + 1:]
                    lines.append(line)
                if len(buf) > MAX_LINE:
                    return lines, [f"line {len(lines) + 1} longer than {MAX_LINE} bytes"]
            if buf:
                problems.append("last line has no newline")
                lines.append(buf)
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        return lines, [f"not a readable gzip file: {type(exc).__name__}"]
    for n, line in enumerate(lines, 1):
        if len(line) > MAX_LINE:
            problems.append(f"line {n} longer than {MAX_LINE} bytes")
    return lines, problems


def _reject_constant(name: str):
    raise ValueError(f"non-finite number {name}")


def validate(path: str, input_ids: Optional[List[str]] = None, expected_rows: Optional[int] = None,
             sha256_file: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Returns (records, problems). Records are only trustworthy when `problems` is empty."""
    problems: List[str] = []

    if sha256_file:
        try:
            expected = open(sha256_file, "r", encoding="ascii").read().split()[0].lower()
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                problems.append("sha256 does not match the sidecar file")
        except (OSError, IndexError, UnicodeDecodeError):
            problems.append("cannot read the sha256 sidecar")

    lines, line_problems = read_lines_bounded(path)
    problems += line_problems
    records: List[Dict[str, Any]] = []
    seen = set()
    for n, raw in enumerate(lines, 1):
        if len(raw) > MAX_LINE:
            continue
        try:
            text = raw.decode("ascii")                       # the exporter escapes everything to ASCII
        except UnicodeDecodeError:
            problems.append(f"line {n}: not ASCII")
            continue
        try:
            rec = json.loads(text, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            problems.append(f"line {n}: invalid JSON ({str(exc)[:60]})")
            continue
        errs = check_record(rec)
        uid = rec.get("url_id") if isinstance(rec, dict) else None
        if isinstance(uid, str):
            if uid in seen:
                errs.append("duplicate url_id")
            seen.add(uid)
        if errs:
            problems += [f"line {n}: {e}" for e in errs]
        else:
            records.append(rec)

    if expected_rows is not None and len(lines) != expected_rows:
        problems.append(f"row count {len(lines)} != expected {expected_rows}")
    if input_ids is not None:
        want = set(input_ids)
        missing, extra = want - seen, seen - want
        if len(lines) != len(want):
            problems.append(f"row count {len(lines)} != input count {len(want)}")
        if missing:
            problems.append(f"{len(missing)} input url_id(s) missing from the output")
        if extra:
            problems.append(f"{len(extra)} url_id(s) in the output that were never pushed")
    return records, problems


def read_input_ids(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            ids.append(line.split("\t", 1)[0].strip())
    return ids


def write_clean(records: List[Dict[str, Any]], clean_dir: str, batch: str, csv_path: Optional[str]) -> str:
    os.makedirs(clean_dir, exist_ok=True)
    out = os.path.join(clean_dir, f"{batch}.clean.jsonl")
    with open(out, "w", encoding="ascii", newline="\n") as fh:
        for rec in sorted(records, key=lambda r: r["url_id"]):
            fh.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n")
    if csv_path:
        cols = (["url_id", "batch_id", "fetch_ts", "url", "page_fetched", "fetch_ok", "fail_reason",
                 "rdap_ok", "html_available", "html_bytes", "script_count", "duration_ms"]
                + schema.VM_FEATURES + [n for n in schema.EXTRA_EXPORT_FIELDS])
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for rec in sorted(records, key=lambda r: r["url_id"]):
                row = []
                for c in cols:
                    v = rec["features"][c] if c in rec["features"] else rec.get(c, "")
                    row.append(safety.csv_safe(v) if isinstance(v, str) else v)
                w.writerow(row)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsonl_gz")
    ap.add_argument("--input", help="the pushed url_id<TAB>url file: every id must come back, nothing else")
    ap.add_argument("--expected-rows", type=int)
    ap.add_argument("--sha256-file", help="sidecar (default: <jsonl_gz>.sha256 if it exists)")
    ap.add_argument("--clean-dir", help="write <batch>.clean.jsonl here when everything passes")
    ap.add_argument("--csv", help="also write a spreadsheet-safe CSV (needs --clean-dir)")
    a = ap.parse_args(argv)

    sidecar = a.sha256_file or (a.jsonl_gz + ".sha256" if os.path.exists(a.jsonl_gz + ".sha256") else None)
    ids = read_input_ids(a.input) if a.input else None
    records, problems = validate(a.jsonl_gz, ids, a.expected_rows, sidecar)

    if problems:
        print(f"VALIDATION FAILED: {len(problems)} problem(s)")
        for p in problems[:MAX_PROBLEMS_SHOWN]:
            print("  -", p)
        if len(problems) > MAX_PROBLEMS_SHOWN:
            print(f"  ... and {len(problems) - MAX_PROBLEMS_SHOWN} more")
        return 1
    print(f"VALIDATION OK: {len(records)} rows" + (", sha256 verified" if sidecar else ", no sha256 sidecar"))
    if a.clean_dir:
        batch = records[0]["batch_id"] if records else os.path.basename(a.jsonl_gz).split(".")[0]
        batch = re.sub(r"[^A-Za-z0-9_-]", "_", batch)[:40]
        out = write_clean(records, a.clean_dir, batch, a.csv)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
