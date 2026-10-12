#!/usr/bin/env python
"""
make_gallery.py -- a contact sheet of the validated screenshots of one batch.

    python scripts/make_gallery.py D:\\LeJetData\\clean\\b0005.clean.jsonl --shots-dir D:\\LeJetData\\clean\\shots\\b0005

Writes <shots-dir>/index.html. The page is deliberately inert:
  * a Content-Security-Policy that allows ONLY images from the same folder (no scripts, no network);
  * no <script>, no <a href>, no forms: the screenshots are pictures, not live pages;
  * every URL and domain shown is DEFANGED text taken from the validated clean file, HTML-escaped.
The images are PNG files that already passed the strict check twice (VM and host).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Any, Dict, List

CSP = "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'"


def build_html(rows: List[Dict[str, Any]], shot_ids: set, batch: str) -> str:
    cards = []
    for r in rows:
        uid = r["url_id"]
        if uid not in shot_ids:
            continue
        label = r.get("label", "")
        cards.append(
            "<figure>"
            f"<img src=\"{html.escape(uid)}.png\" width=\"320\" alt=\"screenshot {html.escape(uid)}\">"
            f"<figcaption><b>{html.escape(str(r.get('final_url') or r.get('url') or ''))}</b><br>"
            f"id {html.escape(uid)} &middot; HTTP {html.escape(str(r.get('final_status')))}"
            f"{' &middot; ' + html.escape(str(label)) if label else ''}"
            f"<br>plain fetch: {html.escape(str(r.get('fail_reason')))}"
            f"{' (the browser got further than the plain fetch)' if r.get('fail_reason') not in ('none', None) else ''}"
            "</figcaption></figure>"
        )
    missing = sum(1 for r in rows if r["url_id"] not in shot_ids)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<meta http-equiv=\"Content-Security-Policy\" content=\"{CSP}\">"
        f"<title>Batch {html.escape(batch)}: screenshots</title>"
        "<style>body{font-family:sans-serif;margin:16px;background:#f4f4f4}"
        "figure{display:inline-block;margin:8px;padding:8px;background:#fff;border:1px solid #ccc;"
        "vertical-align:top;width:336px}figcaption{font-size:12px;word-break:break-all;margin-top:6px}"
        "img{display:block;border:1px solid #999}</style></head><body>"
        f"<h2>Batch {html.escape(batch)}: {len(cards)} screenshots ({missing} pages without one)</h2>"
        "<p>URLs are defanged. These are pictures of pages taken in the isolated lab VM; the pages are "
        "not live and nothing here is clickable.</p>" + "".join(cards) + "</body></html>"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clean_jsonl")
    ap.add_argument("--shots-dir", required=True)
    a = ap.parse_args()
    rows = []
    with open(a.clean_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    shot_ids = {f[:-4] for f in os.listdir(a.shots_dir) if f.endswith(".png") and len(f) == 20}
    batch = rows[0].get("batch_id", "?") if rows else "?"
    page = build_html(rows, shot_ids, batch)
    out = os.path.join(a.shots_dir, "index.html")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print(f"gallery written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
