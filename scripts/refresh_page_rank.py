#!/usr/bin/env python
r"""
refresh_page_rank.py -- HOST-SIDE: get `page_rank` for every page the lab managed to load, in one step.

    python scripts/refresh_page_rank.py

Reads every D:\LeJetData\clean\feat*.clean.jsonl (newest result per link), keeps only the links whose page LOADED
(the others never enter the feature tables, so their domains are not sent to the service), and runs
scripts/lookup_page_rank.py on them. The answers are cached in D:\LeJetData\clean\page_rank_cache.json, so a domain
is only ever asked for once. Output: D:\LeJetData\clean\lookup_fetched.model_input.csv (url_id, page_rank, ...), which
scripts/build_feature_dataset.py --page-rank-csv reads. `scripts/run_parts.ps1` calls this before it builds the tables.

The Open PageRank key is read from D:\LeJetData\secrets\.env (OPENPAGERANK_API_KEY=...) and is never printed. If that file
is missing, nothing is looked up and `page_rank` stays -1 (not measured); this script then exits 0 so a run is not stopped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lookup_page_rank                                       # noqa: E402

CLEAN_DIR = r"D:\LeJetData\clean"
ENV_FILE = r"D:\LeJetData\secrets\.env"


def fetched_records(clean_dir: str) -> List[dict]:
    """The newest record of every link whose page loaded, over all feat*.clean.jsonl (the combined *_all files are skipped)."""
    best: Dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(clean_dir, "feat*.clean.jsonl"))):
        if "_all" in os.path.basename(path):
            continue
        with open(path, encoding="ascii") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec["url_id"] not in best or rec["fetch_ts"] >= best[rec["url_id"]]["fetch_ts"]:
                        best[rec["url_id"]] = rec
    return [r for r in best.values() if r["page_fetched"]]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clean-dir", default=CLEAN_DIR)
    ap.add_argument("--env-file", default=ENV_FILE)
    a = ap.parse_args(argv)

    if not os.path.exists(a.env_file):
        print(f"page_rank: {a.env_file} not found, so nothing is looked up (page_rank stays -1).")
        return 0
    recs = fetched_records(a.clean_dir)
    if not recs:
        print("page_rank: no loaded pages yet.")
        return 0
    src = os.path.join(a.clean_dir, "lookup_input_fetched.jsonl")          # not feat*: it must not count as a lab result
    with open(src, "w", encoding="ascii") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    out = os.path.join(a.clean_dir, "lookup_fetched.model_input.csv")
    rc = lookup_page_rank.main([src, "--out", out, "--env-file", a.env_file,
                                "--cache", os.path.join(a.clean_dir, "page_rank_cache.json")])
    if rc != 0:
        print("page_rank: some lookups failed; those links keep -1 (a failure is never turned into 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
