#!/usr/bin/env python
"""
make_push_file.py -- turn a list of URLs into the push file for the analysis VM
(WEB_INSPECTION_PLAN.md sections 13.2, 12.3). Run on the HOST:

    python scripts/make_push_file.py D:\\LeJetData\\input\\urls.txt --out D:\\LeJetData\\input\\urls_push.txt

Input: one URL per line (may be defanged: hxxps://evil[.]com). Blank lines and # comments are
skipped. Output files:

  <out>               url_id<TAB>raw_url     REFANGED, machine-consumed only (never open it in a browser)
  <out>.defanged.txt  url_id<TAB>defanged    the copy that is safe to view, e-mail or paste
  <out>.report.json   counts: input, kept, duplicates, rejected, with the reasons

`url_id = sha256(raw_url)[:16]` is computed HERE, not in the VM: the VM only echoes it back, which
keeps the join with the labels lossless and stops the VM from relabelling rows. Duplicate URLs
(same url_id) are dropped and counted, so a duplicate can never get two labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from lejet import safety, urlnorm                            # noqa: E402

_WS_CTRL = re.compile(r"[\s\x00-\x1f\x7f]")


def refang(text: str) -> str:
    """hxxps://evil[.]com -> https://evil.com (the same un-defanging link_features uses)."""
    s = text.strip()
    s = re.sub(r"\[\.\]|\(\.\)|\[dot\]", ".", s, flags=re.I)
    s = re.sub(r"(?i)^hxxp", "http", s)
    s = re.sub(r"\[://\]|\[:\]//", "://", s)
    return s


def build(lines: List[str]) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
    items: List[Tuple[str, str]] = []
    seen = set()
    report: Dict[str, object] = {"input_lines": len(lines), "kept": 0, "blank_or_comment": 0,
                                 "duplicates": 0, "rejected": 0, "rejected_reasons": {}}
    reasons: Dict[str, int] = {}

    def reject(reason: str) -> None:
        report["rejected"] += 1                              # type: ignore[operator]
        reasons[reason] = reasons.get(reason, 0) + 1

    for line in lines:
        raw = line.strip("\r\n")
        if not raw.strip() or raw.lstrip().startswith("#"):
            report["blank_or_comment"] += 1                  # type: ignore[operator]
            continue
        url = refang(raw)
        if _WS_CTRL.search(url):
            reject("whitespace_or_control_characters")
            continue
        if len(url) > safety.MAX_URL_LEN:
            reject("longer_than_2048")
            continue
        url_id = urlnorm.compute_url_id(url)
        if url_id in seen:
            report["duplicates"] += 1                        # type: ignore[operator]
            continue
        seen.add(url_id)
        items.append((url_id, url))
    items.sort()
    report["kept"] = len(items)
    report["rejected_reasons"] = dict(sorted(reasons.items()))
    return items, report


def write_outputs(items: List[Tuple[str, str]], report: Dict[str, object], out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for url_id, url in items:
            fh.write(f"{url_id}\t{url}\n")
    with open(out + ".defanged.txt", "w", encoding="utf-8", newline="\n") as fh:
        for url_id, url in items:
            fh.write(f"{url_id}\t{safety.defang_url(url)}\n")
    with open(out + ".report.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.input, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    items, report = build(lines)
    write_outputs(items, report, a.out)
    print(json.dumps(report, sort_keys=True))
    print(f"wrote {len(items)} lines to {a.out} (view the .defanged.txt copy, never the refanged one)")
    return 0 if items else 1


if __name__ == "__main__":
    sys.exit(main())
