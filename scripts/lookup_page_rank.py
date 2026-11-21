#!/usr/bin/env python
"""
lookup_page_rank.py -- the HOST-SIDE `page_rank` lookup (WEB_INSPECTION_PLAN.md sections 13.1,
14.0.2). Run on the host AFTER a batch was validated; the API key never goes into the VM:

    python -I scripts/lookup_page_rank.py D:\\LeJetData\\clean\\b0001.clean.jsonl ^
        --out D:\\LeJetData\\clean\\b0001.model_input.csv ^
        --env-file D:\\LeJetData\\secrets\\.env

The key is read from the environment variable OPENPAGERANK_API_KEY, or from --env-file (a `.env` file
with the line OPENPAGERANK_API_KEY=...), or from --key-file (a file holding only the key), and is NEVER
printed, logged or written anywhere. Keep the file outside the repository and outside OneDrive.

What it does:
  * takes the FINAL registrable domain of every row (plan 13.2b) and asks Open PageRank for it
    (<= 100 domains per request, one request at a time, results cached per domain);
  * a domain the service answers for but has no rank for -> 0 (as in the training column, where
    unrated domains are 0: a real value, not "missing");
  * the lookup failed (network, quota, bad key, answer not an integer 0-10) -> -1 and
    page_rank_ok = 0. Failures are NEVER filled with 0: an outage must not look like "unrated";
  * writes a CSV with exactly the columns route_p_u() needs: the 30 VM features, page_rank,
    has_url, page_fetched, plus url_id and the indicator / hosting columns for the analysis.

Rows on free-hosting platforms get the PLATFORM's rank (high and misleading, plan 14.0.2); they are
flagged by `free_hosting_subdomain` / `platform_path_hosting` so results can be reported with and
without them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from lejet import page_rank as shared                       # noqa: E402  (the answer rules, shared with the website)
from lejet import schema                                    # noqa: E402

DEFAULT_URL = shared.DEFAULT_URL
BATCH = shared.BATCH
MAX_RESPONSE = shared.MAX_RESPONSE
_DOMAIN = shared.DOMAIN_RE
refang_domain = shared.refang_domain


ENV_NAME = "OPENPAGERANK_API_KEY"
_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def read_env_key(path: str) -> str:
    """The value of OPENPAGERANK_API_KEY in a `.env` file: `NAME=value` per line, `#` comments and blank
    lines allowed, no quotes, no spaces around `=`. Other names in the file are ignored. Error messages
    never contain the value."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(4097)
        if len(raw) > 4096:
            raise SystemExit(f"the .env file is unexpectedly large: {path}")
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"cannot read the .env file ({type(exc).__name__}): {path}") from None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if m and m.group(1) == ENV_NAME:
            value = m.group(2).strip()
            if value[:1] in ("'", '"'):
                raise SystemExit(f"remove the quotes around the value of {ENV_NAME} in the .env file")
            return value
    raise SystemExit(f"{ENV_NAME}=... was not found in the .env file (write it as one line, no spaces around '=')")


def load_key(key_file: Optional[str], env_file: Optional[str] = None) -> str:
    """Order: the environment variable, then --env-file, then --key-file."""
    key = os.environ.get(ENV_NAME, "").strip()
    if not key and env_file:
        key = read_env_key(env_file)
    if not key and key_file:
        with open(key_file, "r", encoding="ascii", errors="strict") as fh:
            key = fh.read().strip()
    if not key:
        raise SystemExit(f"no API key: use --env-file, --key-file or set {ENV_NAME}")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{8,128}", key):
        raise SystemExit("the API key has an unexpected format")
    return key


def fetch_batch(domains: List[str], key: str, base_url: str, timeout: float = 15.0) -> Dict[str, int]:
    """Ask for up to 100 domains. Returns {domain: rank 0..10} for the domains that were answered
    (0 when the service has no rank for a domain it answered for). Raises RuntimeError on failure."""
    query = urllib.parse.urlencode([("domains[]", d) for d in domains])
    req = urllib.request.Request(f"{base_url}?{query}", headers={"API-OPR": key, "Accept": "application/json",
                                                                   "User-Agent": "Le-Jet-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(MAX_RESPONSE + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from None            # never include headers (the key)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"network error: {type(exc).__name__}") from None
    return shared.parse_answer(body, domains)


def lookup(domains: List[str], key: str, base_url: str, cache: Dict[str, int],
           pause: float = 0.5, retries: int = 1) -> Tuple[Dict[str, int], int]:
    """Returns ({domain: rank} for every success, number of failed batches)."""
    result: Dict[str, int] = {}
    todo = [d for d in dict.fromkeys(domains) if d not in cache]
    failed_batches = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for attempt in range(retries + 1):
            try:
                got = fetch_batch(chunk, key, base_url)
                result.update(got)
                break
            except RuntimeError:
                if attempt == retries:
                    failed_batches += 1
                else:
                    time.sleep(pause * 3)
        time.sleep(pause)
    return result, failed_batches


def read_clean(path: str) -> List[dict]:
    with open(path, "r", encoding="ascii") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clean_jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-file", help="a .env file holding OPENPAGERANK_API_KEY=... (keep it outside the repo)")
    ap.add_argument("--key-file", help="a file holding only the key")
    ap.add_argument("--cache", help="JSON cache file (loaded and saved); never holds the key")
    ap.add_argument("--base-url", default=DEFAULT_URL)
    ap.add_argument("--pause", type=float, default=0.5)
    a = ap.parse_args(argv)

    parsed = urllib.parse.urlsplit(a.base_url)
    if parsed.scheme != "https" and parsed.hostname not in ("127.0.0.1", "localhost"):
        raise SystemExit("the API base URL must be https (http only for localhost tests)")

    key = load_key(a.key_file, a.env_file)
    records = read_clean(a.clean_jsonl)

    cache: Dict[str, int] = {}
    if a.cache and os.path.exists(a.cache):
        try:
            raw = json.load(open(a.cache, encoding="utf-8"))
            cache = {k: v for k, v in raw.items() if _DOMAIN.fullmatch(k) and isinstance(v, int) and 0 <= v <= 10}
        except (OSError, ValueError):
            cache = {}

    wanted: List[str] = []
    for rec in records:
        dom = refang_domain(rec.get("final_reg_domain", ""))
        if _DOMAIN.fullmatch(dom):
            wanted.append(dom)
    got, failed_batches = lookup(wanted, key, a.base_url, cache, pause=a.pause)
    cache.update(got)
    if a.cache:
        with open(a.cache, "w", encoding="utf-8") as fh:
            json.dump(dict(sorted(cache.items())), fh)

    cols = (["url_id", "has_url", "page_fetched"] + schema.URL_TEXT_FEATURES + ["page_rank"]
            + schema.WEBPAGE_FEATURES + schema.EXTERNAL_FEATURES
            + ["page_rank_ok", "fail_reason", "final_reg_domain", "free_hosting_subdomain",
               "platform_path_hosting", "rdap_ok", "html_available"])
    ok_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for rec in sorted(records, key=lambda r: r["url_id"]):
            dom = refang_domain(rec.get("final_reg_domain", ""))
            rank = cache.get(dom)
            ok = 1 if rank is not None else 0
            ok_count += ok
            feats = rec["features"]
            row = {"url_id": rec["url_id"], "has_url": 1, "page_fetched": rec["page_fetched"],
                   "page_rank": rank if ok else -1, "page_rank_ok": ok, "fail_reason": rec["fail_reason"],
                   "final_reg_domain": rec["final_reg_domain"],
                   "free_hosting_subdomain": rec["free_hosting_subdomain"],
                   "platform_path_hosting": rec["platform_path_hosting"], "rdap_ok": int(rec["rdap_ok"]),
                   "html_available": int(rec["html_available"])}
            row.update({n: feats[n] for n in schema.VM_FEATURES})
            w.writerow([row[c] for c in cols])
    print(f"{len(records)} rows, page_rank_ok for {ok_count}, failed batches: {failed_batches} -> {a.out}")
    return 0 if failed_batches == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
