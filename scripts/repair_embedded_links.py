"""Host-side repair for ONE known defect of an older VM build: a link carried inside a page address
(e.g. a sign-in page that redirects to ".../signin?continue=https://...") was left clickable in the lab output,
so the strict validator rejected the whole batch.

What this does, and nothing more: reads the quarantined export with the validator's own bounded reader,
parses each line as JSON, rewrites every remaining `http://` / `https://` inside any text field to
`hxxp://` / `hxxps://` (src/lejet/safety.defang_embedded_urls), and writes a NEW file + sha256 sidecar next to
the original (the original is never changed). It does not trust or validate anything: the result still has to
pass `validate_output.py` exactly as before. A line that is not plain JSON is a hard error.

Usage:
    python -I scripts\\repair_embedded_links.py <quarantine>\\benign01.jsonl.gz
    python -I scripts\\validate_output.py <quarantine>\\benign01.repaired.jsonl.gz --sha256-file ... --input ... --clean-dir ...
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import validate_output as vo          # noqa: E402  (bounded reader, JSON rules)
from lejet.safety import defang_embedded_urls  # noqa: E402


def neutralise(value):
    if isinstance(value, str):
        return defang_embedded_urls(value)
    if isinstance(value, list):
        return [neutralise(v) for v in value]
    if isinstance(value, dict):
        return {k: neutralise(v) for k, v in value.items()}
    return value


def repair(src: str, dst: str) -> int:
    lines, problems = vo.read_lines_bounded(src)
    if problems:
        raise SystemExit("cannot read the export: " + "; ".join(problems[:3]))
    changed = 0
    out = []
    for n, raw in enumerate(lines, 1):
        try:
            rec = json.loads(raw.decode("ascii"), parse_constant=vo._reject_constant)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise SystemExit(f"line {n}: not plain JSON ({type(exc).__name__}); refusing to repair")
        fixed = neutralise(rec)
        changed += fixed != rec
        out.append(json.dumps(fixed, ensure_ascii=True, separators=(",", ":")))
    data = ("\n".join(out) + "\n").encode("ascii")
    with gzip.open(dst, "wb") as fh:
        fh.write(data)
    digest = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    with open(dst + ".sha256", "w", encoding="ascii") as fh:
        fh.write(f"{digest}  {os.path.basename(dst)}\n")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsonl_gz")
    args = ap.parse_args()
    if not args.jsonl_gz.endswith(".jsonl.gz"):
        raise SystemExit("expected a .jsonl.gz export")
    dst = args.jsonl_gz[: -len(".jsonl.gz")] + ".repaired.jsonl.gz"
    if os.path.exists(dst):
        raise SystemExit(f"{dst} already exists")
    changed = repair(args.jsonl_gz, dst)
    print(f"{changed} record(s) had a clickable link inside a field; wrote {dst}")


if __name__ == "__main__":
    main()
