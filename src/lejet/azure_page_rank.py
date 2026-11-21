"""azure_page_rank.py -- the `page_rank` look-up (Open PageRank), run ON the Azure VM in its own locked-down unit.

    python -I -m lejet.azure_page_rank            (started only by the root helper /usr/local/sbin/lejet-azure-pagerank)

Why it is a separate unit and not part of the fetch: the API key must never be on the account that opens hostile pages
(`fetcher`). This program is NOT that account. The root helper starts it as a throw-away user (systemd DynamicUser) with:
  * the key and the domain list handed over as systemd credentials (files only this unit can read, in $CREDENTIALS_DIRECTORY),
  * a read-only file system, no access to /home or /var/lib/lejetweb, a memory cap and a time limit,
  * no page content at all: the domain list comes from the PC (taken from the batch the strict validator already accepted),
    so no byte that a hostile page produced is ever read here.

What it does: asks Open PageRank for up to 100 domains per request, with the answer rules of src/lejet/page_rank.py (the same
rules as the website and scripts/lookup_page_rank.py): a rank is 0..10, a domain the service answers for but has no rank for
is 0, a failure is simply ABSENT from the answer (the PC then keeps -1, never 0). The answer goes to standard output as one
JSON line {"ranks": {domain: rank}, "asked": N, "failed_batches": K}. The key is never printed, logged or put in an error.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List

from lejet import page_rank as shared

MAX_DOMAINS = 2000
MAX_FILE = 200_000
PAUSE = 0.5


def _read_credential(name: str, limit: int) -> str:
    base = os.environ.get("CREDENTIALS_DIRECTORY", "")
    if not base:
        raise SystemExit("no credentials directory (this program is started by lejet-azure-pagerank)")
    path = os.path.join(base, name)
    try:
        with open(path, "rb") as fh:
            raw = fh.read(limit + 1)
    except OSError as exc:
        raise SystemExit(f"cannot read credential '{name}' ({type(exc).__name__})") from None
    if len(raw) > limit:
        raise SystemExit(f"credential '{name}' is unexpectedly large")
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        raise SystemExit(f"credential '{name}' is not plain ASCII") from None


def load_key() -> str:
    key = _read_credential("key", 4096).strip()
    if key.startswith("OPENPAGERANK_API_KEY="):                  # the website's key file may be written this way
        key = key.split("=", 1)[1].strip()
    if not shared.KEY_RE.fullmatch(key):
        raise SystemExit("the API key has an unexpected format")
    return key


def load_domains() -> List[str]:
    seen: Dict[str, None] = {}
    for line in _read_credential("domains", MAX_FILE).splitlines():
        dom = shared.refang_domain(line)
        if dom and shared.DOMAIN_RE.fullmatch(dom):
            seen[dom] = None
    if len(seen) > MAX_DOMAINS:
        raise SystemExit(f"too many domains ({len(seen)}; the limit is {MAX_DOMAINS})")
    return list(seen)


def fetch_batch(domains: List[str], key: str, base_url: str = shared.DEFAULT_URL, timeout: float = 15.0) -> Dict[str, int]:
    query = urllib.parse.urlencode([("domains[]", d) for d in domains])
    req = urllib.request.Request(f"{base_url}?{query}", headers={
        "API-OPR": key, "Accept": "application/json", "User-Agent": "Le-Jet-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(shared.MAX_RESPONSE + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from None            # never include headers (the key)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"network error: {type(exc).__name__}") from None
    return shared.parse_answer(body, domains)


def lookup(domains: List[str], key: str, base_url: str = shared.DEFAULT_URL, pause: float = PAUSE,
           retries: int = 1, fetch=None):
    """({domain: rank} for every answered domain, number of failed batches)."""
    fetch = fetch or fetch_batch
    ranks: Dict[str, int] = {}
    failed = 0
    for i in range(0, len(domains), shared.BATCH):
        chunk = domains[i:i + shared.BATCH]
        for attempt in range(retries + 1):
            try:
                ranks.update(fetch(chunk, key, base_url))
                break
            except RuntimeError as exc:
                if attempt == retries:
                    failed += 1
                    print(f"batch of {len(chunk)} domains failed: {exc}", file=sys.stderr)
                else:
                    time.sleep(pause * 3)
        time.sleep(pause)
    return ranks, failed


def main() -> int:
    key = load_key()
    domains = load_domains()
    ranks, failed = lookup(domains, key)
    print(json.dumps({"ranks": dict(sorted(ranks.items())), "asked": len(domains), "failed_batches": failed},
                     separators=(",", ":")))
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
