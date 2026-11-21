#!/usr/bin/env python
r"""
azure_page_rank.py -- get `page_rank` from the AZURE LAB for batches that were already fetched there.

    python -I scripts/azure_page_rank.py D:\LeJetData\clean\hn01.clean.jsonl [more .clean.jsonl files]

What happens (the lab's counterpart of lookup_page_rank.py, which needs the key on this PC):
  1. from every .clean.jsonl (a batch the strict validator already accepted) it takes the final registrable domain of the rows
     whose page was FETCHED (a page that was not reached keeps -1, and its domain is not sent anywhere), minus the domains
     already answered in the cache;
  2. it copies those domain names (names only, no page content) to the VM and runs
     `sudo -n /usr/local/sbin/lejet-azure-pagerank <batch>` there. The VM looks them up in its own locked-down unit with the
     Open PageRank key that lives only on the VM (installed with infra/azure/set_rank_key.sh); `fetcher`, the account that opens
     hostile pages, cannot read that key;
  3. it pulls <batch>.page_rank.json back, checks every entry (a domain name and a whole number 0..10) and adds the answers to the
     cache (default D:\LeJetData\clean\page_rank_cache.json: domains and ranks, never a key);
  4. for every .clean.jsonl it writes <name>.model_input.csv (url_id, page_rank, page_rank_ok, ...), exactly the file that
     scripts/lookup_page_rank.py writes, via `lookup_page_rank.py --offline`. scripts/add_full_features.py, build_feature_dataset.py
     and run_parts.ps1 read it unchanged.

Rules kept from the other look-ups: a domain the service answers for but has no rank for is 0; a domain that could not be looked
up stays -1 (page_rank_ok = 0), NEVER 0. If the VM has no key installed it says "NO KEY", nothing is looked up, page_rank stays -1
and this script exits 0 (a run is never stopped by it). Exit code 3 = done, but some domains failed (run it again later).

Settings: environment LEJET_LAB_HOST = the ssh host alias (default lejet-azure).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import lookup_page_rank as lpr                                   # noqa: E402
from lejet import page_rank as shared                            # noqa: E402

CACHE_FILE = r"D:\LeJetData\clean\page_rank_cache.json"
MAX_PER_CALL = 2000
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
HOST_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
BATCH_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def load_cache(path: str) -> Dict[str, int]:
    if not path or not os.path.exists(path):
        return {}
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in raw.items()
            if shared.DOMAIN_RE.fullmatch(k) and isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 10}


def save_cache(path: str, cache: Dict[str, int]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(sorted(cache.items())), fh)


def fetched_domains(records: Sequence[dict]) -> List[str]:
    """The final registrable domains of the rows whose page loaded, in order, without repeats."""
    seen: Dict[str, None] = {}
    for rec in records:
        if not rec.get("page_fetched"):
            continue
        dom = shared.refang_domain(rec.get("final_reg_domain", ""))
        if shared.DOMAIN_RE.fullmatch(dom):
            seen[dom] = None
    return list(seen)


def parse_vm_answer(text: str, asked: Sequence[str]) -> Dict[str, int]:
    """The VM's JSON, checked entry by entry: only domains we asked for, whole numbers 0..10."""
    try:
        data = json.loads(text)
        ranks = data["ranks"]
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("the VM's answer is not the expected JSON") from None
    if not isinstance(ranks, dict):
        raise RuntimeError("the VM's answer is not the expected JSON")
    wanted = set(asked)
    return {d: r for d, r in ranks.items()
            if d in wanted and isinstance(r, int) and not isinstance(r, bool) and 0 <= r <= 10}


def run(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def ask_vm(domains: List[str], batch: str, host: str) -> Optional[Dict[str, int]]:
    """Look the domains up on the VM. None = no key installed there (nothing was asked). Raises RuntimeError on a real failure."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "domains.txt"), "w", encoding="ascii", newline="\n") as fh:
            fh.write("\n".join(domains) + "\n")
        r = run(["scp", *SSH_OPTS, "domains.txt", f"{host}:incoming/domains.txt"], cwd=tmp)
        if r.returncode != 0:
            raise RuntimeError("scp of the domain list failed (is the VM running?)")
        r = run(["ssh", *SSH_OPTS, host, f"sudo -n /usr/local/sbin/lejet-azure-pagerank {batch}"])
        out = (r.stdout or "") + (r.stderr or "")
        if "NO KEY" in out:
            print("   the VM has no Open PageRank key yet: page_rank stays -1 "
                  "(on the VM: sudo bash ~/lejet-src/infra/azure/set_rank_key.sh)")
            return None
        if r.returncode not in (0, 3):
            tail = " | ".join(out.strip().splitlines()[-3:])
            raise RuntimeError(f"the VM's look-up failed (exit {r.returncode}): {tail}")
        r2 = run(["scp", *SSH_OPTS, f"{host}:export/{batch}.page_rank.json", "."], cwd=tmp)
        path = os.path.join(tmp, f"{batch}.page_rank.json")
        if r2.returncode != 0 or not os.path.exists(path):
            raise RuntimeError("could not pull the answer file from the VM")
        return parse_vm_answer(open(path, encoding="utf-8").read(), domains)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clean_jsonl", nargs="+")
    ap.add_argument("--batch", help="name for the VM's answer file (default: derived from the first file)")
    ap.add_argument("--cache", default=CACHE_FILE)
    ap.add_argument("--host", default=os.environ.get("LEJET_LAB_HOST", "lejet-azure"))
    ap.add_argument("--no-model-input", action="store_true", help="only update the cache; do not write the .model_input.csv files")
    a = ap.parse_args(argv)

    if not HOST_RE.fullmatch(a.host):
        raise SystemExit("bad host alias")
    first = os.path.basename(a.clean_jsonl[0]).replace(".clean.jsonl", "")
    batch = a.batch or re.sub(r"[^A-Za-z0-9_-]", "_", first)[:32] + "_pr"
    if not BATCH_RE.fullmatch(batch):
        raise SystemExit("bad batch id (letters, digits, - and _ only, at most 40)")

    records: Dict[str, List[dict]] = {p: lpr.read_clean(p) for p in a.clean_jsonl}
    cache = load_cache(a.cache)
    todo: List[str] = []
    for recs in records.values():
        todo += [d for d in fetched_domains(recs) if d not in cache]
    todo = list(dict.fromkeys(todo))
    print(f"{len(todo)} domains to look up on the VM ({len(cache)} already in the cache)")

    failed = False
    for i in range(0, len(todo), MAX_PER_CALL):
        chunk = todo[i:i + MAX_PER_CALL]
        name = batch if i == 0 else f"{batch[:35]}_{i // MAX_PER_CALL}"
        try:
            got = ask_vm(chunk, name, a.host)
        except RuntimeError as exc:
            print(f"   {exc}", file=sys.stderr)
            failed = True
            break
        if got is None:                                  # no key on the VM: not an error
            break
        cache.update(got)
        save_cache(a.cache, cache)
        missing = [d for d in chunk if d not in got]
        if missing:
            print(f"   {len(missing)} domains were not answered (they stay -1; run this again later)")
            failed = True

    rc = 0
    if not a.no_model_input:
        for path in a.clean_jsonl:
            out = re.sub(r"\.clean\.jsonl$", "", path) + ".model_input.csv"
            r = lpr.main([path, "--out", out, "--offline", "--cache", a.cache])
            rc = rc or r
    return 3 if (failed or rc) else 0


if __name__ == "__main__":
    sys.exit(main())
