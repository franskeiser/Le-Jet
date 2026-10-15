#!/usr/bin/env python
r"""
add_full_features.py -- HOST-SIDE: complete the feature table of one validated batch with `page_rank`.

The lab (the analysis VM) measures 72 of the 73 exported public-set columns itself. `page_rank` is a third-party
look-up (Open PageRank), done on the host so the key never enters the VM:

    python -I scripts/lookup_page_rank.py D:\LeJetData\clean\b0001.clean.jsonl ^
        --out D:\LeJetData\clean\b0001.model_input.csv --env-file D:\LeJetData\secrets\.env
    python -I scripts/add_full_features.py D:\LeJetData\clean\b0001.clean.jsonl ^
        --out D:\LeJetData\clean\b0001.full_features.csv ^
        --page-rank-csv D:\LeJetData\clean\b0001.model_input.csv

  page_rank   taken from the output of scripts/lookup_page_rank.py (--page-rank-csv, matched by url_id);
              -1 without it or when that lookup failed (a failed lookup is never turned into 0).

Fourteen public-set columns are not exported at all (schema.DROPPED_FROM_FULL, with the reason for each): the ones
the lab cannot capture, the weak ones, and the exact duplicates or never-measured ones.

Reads the validated .clean.jsonl (so it has been through scripts/validate_output.py); writes a spreadsheet-safe CSV
with the columns url_id, batch_id, url (defanged), page_fetched, fail_reason + the 73 exported columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from lejet import safety, schema                            # noqa: E402

HEAD = ["url_id", "batch_id", "url", "page_fetched", "fail_reason"]


def read_page_rank(path: str) -> Dict[str, int]:
    """url_id -> page_rank from lookup_page_rank.py's CSV; failures (-1) are kept as -1."""
    out: Dict[str, int] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["url_id"]] = int(float(row["page_rank"]))
            except (KeyError, ValueError):
                continue
    return out


def read_clean(path: str) -> List[dict]:
    with open(path, encoding="ascii") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def complete(rec: dict, page_rank: Optional[Dict[str, int]]) -> Dict[str, float]:
    full = dict(rec.get("features_full") or {n: -1 for n in schema.FULL_FEATURES})
    if page_rank is not None:
        full["page_rank"] = page_rank.get(rec["url_id"], -1)
    return full


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clean_jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--page-rank-csv")
    a = ap.parse_args(argv)

    records = read_clean(a.clean_jsonl)
    page_rank = read_page_rank(a.page_rank_csv) if a.page_rank_csv else None
    missing = [r["url_id"] for r in records if "features_full" not in r]
    if missing:
        print(f"{len(missing)} row(s) come from a batch run before the extra feature export existed; their columns stay -1")
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEAD + schema.FULL_FEATURES)
        for rec in sorted(records, key=lambda r: r["url_id"]):
            full = complete(rec, page_rank)
            w.writerow([safety.csv_safe(rec[h]) if isinstance(rec[h], str) else rec[h] for h in HEAD]
                       + [full.get(n, -1) for n in schema.FULL_FEATURES])
    print(f"wrote {a.out}: {len(records)} rows x {len(schema.FULL_FEATURES)} columns"
          + ("" if page_rank is not None else "; page_rank not measured (no --page-rank-csv)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
