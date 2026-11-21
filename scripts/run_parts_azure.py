#!/usr/bin/env python
r"""
run_parts_azure.py -- HOST-SIDE: run the labelled feature input (the part files of scripts/build_labeled_input.py) through the AZURE lab,
FEATURES ONLY (no screenshots).

    python scripts/run_parts_azure.py [--stamp 20261112_031122] [--from 1] [--to 0] [--dry-run]

scripts/run_parts.ps1 uses the two VirtualBox VMs. This does the same for the Azure VM: each part file (about 120 links) is one call of
scripts/run_features_azure.ps1, which fetches every link and computes the 73 exported features on the VM (no browser, no pictures,
a few seconds per LINK instead of 30 to 90) and brings back the validated result as batch `feat<stamp>p<part>` in D:\LeJetData\clean,
the same name run_parts.ps1 uses, so scripts/build_feature_dataset.py reads it with no change:

    python scripts/build_feature_dataset.py --balance --lean

It never opens a link itself and prints no link. Safe to stop and run again: a part whose `<batch>.clean.jsonl` exists is skipped (a
part that failed is simply run again, a new VM batch each time). Stops if the VM's safety gate refuses (exit 3) or 3 parts in a row fail.
For the live tool's mode (one link, screenshots) see scripts/run_live_azure.ps1; it is not used here.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple

DATASETS = Path(r"D:\LeJetData\datasets")
CLEAN = Path(r"D:\LeJetData\clean")
INPUT = Path(r"D:\LeJetData\input")
QUARANTINE = Path(r"D:\LeJetData\quarantine")
LAB_SCRIPT = Path(__file__).resolve().parent / "run_features_azure.ps1"
PART = re.compile(r"^feature_input_(\d{8}_\d{6})_part(\d+)of(\d+)\.txt$")
MAX_PER_CALL = 400                      # the VM helper's cap


def newest_stamp(folder: Path) -> str:
    stamps = sorted({m.group(1) for f in folder.iterdir() if (m := PART.match(f.name))})
    if not stamps:
        raise SystemExit(f"no feature_input_*_part*.txt files in {folder}")
    return stamps[-1]


def batch_name(stamp: str, part: int) -> str:
    """The same name scripts/run_parts.ps1 gives: feat<YYYYMMDD_HHMMSS without the underscore>p<part>."""
    return f"feat{stamp.replace('_', '')}p{part}"


def parts(folder: Path, stamp: str) -> List[Tuple[int, Path]]:
    return sorted((int(m.group(2)), f) for f in folder.iterdir() if (m := PART.match(f.name)) and m.group(1) == stamp)


def links_of(path: Path) -> List[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def jobs(folder: Path, stamp: str, first: int = 1, last: int = 0) -> Iterator[Tuple[str, Path, int]]:
    """(batch, part file, number of links) for each part in range."""
    for n, f in parts(folder, stamp):
        if n < first or (last and n > last):
            continue
        k = len(links_of(f))
        if k > MAX_PER_CALL:
            raise SystemExit(f"{f.name} has {k} links; the VM takes {MAX_PER_CALL} per call: rebuild with --chunk {MAX_PER_CALL} or fewer")
        yield batch_name(stamp, n), f, k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", help="the group of part files (default: the newest)")
    ap.add_argument("--datasets", type=Path, default=DATASETS)
    ap.add_argument("--from", dest="first", type=int, default=1, help="first part to run")
    ap.add_argument("--to", dest="last", type=int, default=0, help="last part to run (0 = the last)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    stamp = a.stamp or newest_stamp(a.datasets)
    all_jobs = list(jobs(a.datasets, stamp, a.first, a.last))
    todo = [j for j in all_jobs if not (CLEAN / f"{j[0]}.clean.jsonl").exists()]
    print(f"stamp {stamp}: {len(all_jobs)} parts ({sum(j[2] for j in all_jobs)} links), "
          f"{len(all_jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    if a.dry_run:
        for b, f, k in todo:
            print(f"  would run {b}: {k} links from {f.name}")
        print("dry run: nothing started")
        return 0
    for d in (CLEAN, INPUT, QUARANTINE):
        d.mkdir(parents=True, exist_ok=True)
    failed_in_row = 0
    for i, (batch, f, k) in enumerate(todo, 1):
        t0 = time.time()
        p = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAB_SCRIPT), "-UrlFile", str(f),
                            "-Batch", batch, "-InputDir", str(INPUT), "-Quarantine", str(QUARANTINE), "-CleanDir", str(CLEAN)],
                           capture_output=True, text=True, timeout=3600, creationflags=0x08000000 if os.name == "nt" else 0)
        ok = p.returncode == 0 and (CLEAN / f"{batch}.clean.jsonl").exists()
        print(f"[{i}/{len(todo)}] {'ok  ' if ok else 'FAIL'} exit {p.returncode} {batch} {k} links ({time.time() - t0:.0f}s)", flush=True)
        if not ok:
            tail = [x for x in (p.stdout + p.stderr).splitlines() if x.strip()][-4:]
            for x in tail:
                print("      " + x[:160], flush=True)
        if p.returncode == 3:
            print("the VM's safety gate refused: stopping (nothing was opened)", flush=True)
            return 3
        failed_in_row = 0 if ok else failed_in_row + 1
        if failed_in_row >= 3:
            print("3 parts in a row failed: stopping; check `ssh lejet-azure true`, the VM and that the features helper is installed", flush=True)
            return 1
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
