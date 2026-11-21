#!/usr/bin/env python
r"""
fill_page_rank.py -- HOST-SIDE: add `page_rank` to the CSV downloaded from the collection website.

    python scripts/fill_page_rank.py D:\path\lejet_web_submissions.csv

Why this is a separate step: `page_rank` is a third-party look-up (Open PageRank). Like the lab's own pipeline, the website
cannot measure it, because the API key must never be put on a machine that visits hostile pages. The website's CSV therefore
has page_rank = -1 ("not measured") in every row, and this script fills it in on YOUR PC, reusing the lab's look-up code
(scripts/lookup_page_rank.py) and the same rules:

  * only rows whose page was fetched (page_fetched = 1) are looked up. A page that was not reached has all its page features
    blanked (-1), page_rank included, and its domain is not sent to the service. --all looks up every row instead;
  * the domain asked for is `link_domain` (the final registrable domain);
  * a domain the service answers for but has no rank for -> 0 (unrated is a real value in the training data);
  * a failed look-up (network, quota, bad key) -> stays -1, NEVER 0, so an outage cannot look like "unrated";
  * the input file is never changed: the result goes to <input>.with_page_rank.csv (or --out). Every other cell is copied
    exactly as it was.

The key is read from D:\LeJetData\secrets\.env (OPENPAGERANK_API_KEY=...), the environment variable of the same name, or
--env-file / --key-file, and is never printed or written. Answers are cached (default D:\LeJetData\clean\page_rank_cache.json,
the same cache scripts/refresh_page_rank.py uses), so a domain is only ever asked for once.

Exit code 0 = done, 3 = done but some domains could not be looked up (they stay -1; run again later).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
from typing import Dict, List, Optional, Sequence, Set

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import lookup_page_rank as lpr                                   # noqa: E402

ENV_FILE = r"D:\LeJetData\secrets\.env"
CACHE_FILE = r"D:\LeJetData\clean\page_rank_cache.json"
NEEDED = ("link_domain", "page_fetched", "page_rank")
MISSING = "-1"


def read_rows(path: str):
    # utf-8-sig: a CSV that went through Excel may start with a byte-order mark
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        absent = [c for c in NEEDED if c not in header]
        if absent:
            raise SystemExit(f"{path}: missing column(s) {', '.join(absent)}: is this the website's CSV?")
        return header, list(reader)


def row_domain(row: Dict[str, str]) -> str:
    dom = lpr.refang_domain(row.get("link_domain", ""))
    return dom if lpr._DOMAIN.fullmatch(dom) else ""


def wanted(rows: Sequence[Dict[str, str]], everything: bool) -> List[str]:
    """The domains to ask for: rows whose page was fetched (or all rows with --all), in order, without repeats."""
    out = [row_domain(r) for r in rows if everything or str(r.get("page_fetched", "")).strip() == "1"]
    return list(dict.fromkeys(d for d in out if d))


def apply_ranks(rows: Sequence[Dict[str, str]], ranks: Dict[str, int], everything: bool) -> int:
    """Write the answers into the rows that were asked about. Returns how many rows got a value."""
    filled = 0
    for r in rows:
        if not everything and str(r.get("page_fetched", "")).strip() != "1":
            continue
        rank = ranks.get(row_domain(r))
        if rank is not None:
            r["page_rank"] = str(rank)
            filled += 1
    return filled


def load_cache(path: Optional[str]) -> Dict[str, int]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return {k: v for k, v in raw.items() if lpr._DOMAIN.fullmatch(k) and isinstance(v, int) and 0 <= v <= 10}
    except (OSError, ValueError):
        return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv_file", help="the CSV downloaded from the website")
    ap.add_argument("--out", help="result file (default: <input>.with_page_rank.csv)")
    ap.add_argument("--all", action="store_true", help="also look up rows whose page was not fetched")
    ap.add_argument("--env-file", default=ENV_FILE, help="a .env file holding OPENPAGERANK_API_KEY=... (default: %(default)s)")
    ap.add_argument("--key-file", help="a file holding only the key")
    ap.add_argument("--cache", default=CACHE_FILE, help="JSON cache of answers, never holds the key (default: %(default)s)")
    ap.add_argument("--base-url", default=lpr.DEFAULT_URL)
    ap.add_argument("--pause", type=float, default=0.5)
    a = ap.parse_args(argv)

    parsed = urllib.parse.urlsplit(a.base_url)
    if parsed.scheme != "https" and parsed.hostname not in ("127.0.0.1", "localhost"):
        raise SystemExit("the API base URL must be https (http only for localhost tests)")
    out_path = a.out or a.csv_file + ".with_page_rank.csv"
    if os.path.abspath(out_path) == os.path.abspath(a.csv_file):
        raise SystemExit("--out must be a different file: the input is never changed")

    header, rows = read_rows(a.csv_file)
    domains = wanted(rows, a.all)
    cache = load_cache(a.cache)
    failed_batches = 0
    if domains:
        env_file = a.env_file if os.path.exists(a.env_file) else None      # no file there: load_key says what is missing
        key = lpr.load_key(a.key_file, env_file)
        got, failed_batches = lpr.lookup(domains, key, a.base_url, cache, pause=a.pause)
        cache.update(got)
        if a.cache:
            os.makedirs(os.path.dirname(os.path.abspath(a.cache)), exist_ok=True)
            with open(a.cache, "w", encoding="utf-8") as fh:
                json.dump(dict(sorted(cache.items())), fh)
    filled = apply_ranks(rows, cache, a.all)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    still = sum(1 for r in rows if r["page_rank"] == MISSING and (a.all or str(r.get("page_fetched", "")).strip() == "1"))
    print(f"{len(rows)} rows: {len(domains)} domains asked about, page_rank filled in {filled} rows; "
          f"{still} looked-up rows still -1 (failed batches: {failed_batches}) -> {out_path}")
    return 0 if failed_batches == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
