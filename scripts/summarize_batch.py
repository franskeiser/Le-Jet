#!/usr/bin/env python
"""
summarize_batch.py -- a plain-language report for one validated batch. Run on the HOST:

    python scripts/summarize_batch.py D:\\LeJetData\\clean\\b0002.clean.jsonl [--out D:\\LeJetData\\reports\\b0002.md]

Reads only the validated, defanged output (never anything from the VM directly) and prints:
  * how many URLs finished, and why the others failed;
  * per top-level domain: how many rows, how many got registration data (RDAP), how many got a
    certificate age (crt.sh) and how many are marked "no RDAP service for this TLD";
  * the rows whose page could not be fetched.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List


def tld_of(defanged_domain: str) -> str:
    d = (defanged_domain or "").replace("[.]", ".").strip().lower().rstrip(".")
    return d.rsplit(".", 1)[-1] if "." in d else (d or "?")


def load(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_report(rows: List[Dict[str, Any]], batch: str) -> str:
    n = len(rows)
    done = [r for r in rows if r.get("fail_reason") == "none"]
    reasons = Counter(r.get("fail_reason", "?") for r in rows if r.get("fail_reason") != "none")

    out: List[str] = []
    out.append(f"# Batch {batch}: summary")
    out.append("")
    out.append(f"- URLs in the batch: **{n}**")
    out.append(f"- Page fetched and all features computed: **{len(done)}** ({_pct(len(done), n)})")
    if reasons:
        out.append("- Not fetched, by reason: " + ", ".join(f"{k} x{v}" for k, v in reasons.most_common()))
    out.append("")

    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[tld_of(r.get("final_reg_domain") or r.get("reg_domain") or "")].append(r)
    out.append("## Registration data and certificate age, per TLD")
    out.append("")
    out.append("| TLD | rows | registration data (RDAP) | certificate age (crt.sh) | TLD has no RDAP | page fetched |")
    out.append("|---|---|---|---|---|---|")
    for tld, rs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rdap = sum(1 for r in rs if r.get("rdap_ok"))
        ct = sum(1 for r in rs if (r.get("ct_first_cert_days", -1) or -1) >= 0)
        unavailable = sum(1 for r in rs if r.get("registration_unavailable_tld"))
        fetched = sum(1 for r in rs if r.get("fail_reason") == "none")
        out.append(f"| .{tld} | {len(rs)} | {rdap} | {ct} | {unavailable} | {fetched} |")
    out.append("")

    unavailable_rows = [r for r in rows if r.get("registration_unavailable_tld")]
    if unavailable_rows:
        out.append("## Rows on TLDs without RDAP")
        out.append("")
        out.append("| domain | certificate age (days) | page |")
        out.append("|---|---|---|")
        for r in unavailable_rows:
            days = r.get("ct_first_cert_days", -1)
            out.append(f"| {r.get('final_reg_domain') or r.get('reg_domain')} | "
                       f"{days if days is not None and days >= 0 else 'unknown'} | {r.get('fail_reason')} |")
        out.append("")

    failed = [r for r in rows if r.get("fail_reason") != "none"]
    if failed:
        out.append("## Rows without a page")
        out.append("")
        out.append("| url | reason | status | hops |")
        out.append("|---|---|---|---|")
        for r in failed:
            out.append(f"| {r.get('url')} | {r.get('fail_reason')} | {r.get('final_status')} | {r.get('hop_count')} |")
        out.append("")
    return "\n".join(out)


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.0f}%" if b else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clean_file")
    ap.add_argument("--out", default=None, help="also write the report to this .md file")
    args = ap.parse_args()
    rows = load(args.clean_file)
    batch = rows[0].get("batch_id", "?") if rows else "?"
    text = build_report(rows, batch)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
        print(f"\nreport written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
