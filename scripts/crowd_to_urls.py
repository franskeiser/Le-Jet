#!/usr/bin/env python
"""
crowd_to_urls.py -- pull the first link out of each crowdsourced SMS and write the URL list the
inspector takes (docs/c_prune_architecture.md section 4.2, docs/inspector_runbook.md step 4).
Run on the HOST:

    python scripts/crowd_to_urls.py D:\\LeJetData\\crowd\\raw\\submissions_2026-11-03.csv --out D:\\LeJetData\\input\\urls.txt

Several submission files may be given at once. The link is found with the same finder and
normalised the same way as the app (src.features.link_features), so the inspector visits exactly
the URL the model later looks up. Only the FIRST link of a message is used (collection rule 7).

Rows are skipped, and counted in the report, when:
  * consent is not 1 (the builder drops them, so they are not inspected either);
  * sms_text has no link (the dataset is URL messages only: fix the row);
  * it is still the template's example row (fake GCash message).

Outputs:
  <out>              one normalised URL per line, duplicates removed -> input of make_push_file.py
  <out>.report.json  counts, plus the file and row number of every row without a link

<out> holds REAL phishing links: never open it in a browser. make_push_file.py writes the
defanged copy that is safe to look at.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from src.features.link_features import extract_links, normalize_link   # noqa: E402

COLUMNS = ["date_submitted", "sms_text", "label", "category", "language", "sender_type",
           "date_received", "label_source", "label_verified_by", "consent"]
EXAMPLE_TEXT = "Your GCash account will be locked today. Verify here: https://gcash-verify-ph.xyz"


def first_link(text: str) -> Optional[str]:
    links = extract_links(text)
    return normalize_link(links[0]["raw"]) if links else None


def read_rows(path: str) -> List[Dict[str, str]]:
    # utf-8-sig: Excel's "CSV UTF-8" puts a byte-order mark before the first header.
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        return list(reader)


def build(files: List[Tuple[str, List[Dict[str, str]]]]) -> Tuple[List[str], Dict[str, object]]:
    urls: List[str] = []
    seen = set()
    report: Dict[str, object] = {"rows": 0, "kept": 0, "duplicates": 0, "no_consent": 0,
                                 "example_row": 0, "no_link": 0, "no_link_rows": []}
    for path, rows in files:
        # Row 1 is the header, so the first message is row 2 (the row number Excel shows).
        for rownum, row in enumerate(rows, start=2):
            report["rows"] += 1                                  # type: ignore[operator]
            text = (row.get("sms_text") or "").strip()
            if text == EXAMPLE_TEXT:
                report["example_row"] += 1                       # type: ignore[operator]
                continue
            if (row.get("consent") or "").strip() != "1":
                report["no_consent"] += 1                        # type: ignore[operator]
                continue
            url = first_link(text)
            if url is None:
                report["no_link"] += 1                           # type: ignore[operator]
                report["no_link_rows"].append(f"{os.path.basename(path)}:{rownum}")  # type: ignore[union-attr]
                continue
            if url in seen:
                report["duplicates"] += 1                        # type: ignore[operator]
                continue
            seen.add(url)
            urls.append(url)
    report["kept"] = len(urls)
    return urls, report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="one or more submissions_<date>.csv files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    files = [(p, read_rows(p)) for p in a.inputs]
    urls, report = build(files)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for url in urls:
            fh.write(url + "\n")
    with open(a.out + ".report.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))
    if report["example_row"]:
        print("note: the template's example row was found and skipped; delete it from your file")
    if report["no_link"]:
        print("warning: some rows have no link, fix them: " + ", ".join(report["no_link_rows"]))  # type: ignore[arg-type]
    print(f"wrote {len(urls)} URLs to {a.out} (real links: never open this file in a browser)")
    return 0 if urls else 1


if __name__ == "__main__":
    sys.exit(main())
