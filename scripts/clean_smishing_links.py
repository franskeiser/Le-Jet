#!/usr/bin/env python
"""
clean_smishing_links.py -- turn the team's spreadsheets of smishing links into a clean, de-duplicated
URL list for the lab (WEB_INSPECTION_PLAN.md sections 12.3, 13.2). Run on the HOST:

    set PYTHONPATH=D:\\LeJetData\\devlibs        (where openpyxl lives; it is not a repo dependency)
    python scripts/clean_smishing_links.py --out-dir D:\\LeJetData\\input --name smish_oct ^
        C:\\Users\\Franz\\Downloads\\asdasdasdas.xlsx C:\\Users\\Franz\\Downloads\\linkss.xlsx

Reads every cell that holds a link (the link column is found by its header or by content), and writes
  <name>.txt        one cleaned URL per line (refanged; feed it to scripts/make_push_file.py; NEVER open it)
  <name>.meta.json  per URL: where it came from (file, row) and what the team labelled it
and prints a plain-language count of what was removed and why.

Cleaning rules:
  * invisible characters, newlines, tabs, non-breaking spaces and trailing punctuation are stripped;
  * a cell with several links gives several URLs; header/label cells ("Links", "URL", ...) are skipped;
  * defanged text (hxxps://x[.]com) is refanged;
  * duplicates are merged (same host, path and query, scheme and a trailing "/" ignored): the first
    spelling is kept and every source row is remembered;
  * URLs the lab's safety gate would refuse (private addresses, odd ports, credentials) are kept in the
    metadata with a note but NOT written to the URL list;
  * the team's own label (Live / Dead / Unsure, any spelling) is normalised and kept as `manual`;
    reviewer names are NOT copied (not needed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from lejet import urlnorm                                       # noqa: E402

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_TRAILING = ".,;:!?)>\"'"
_HEADERS = {"links", "link", "url", "urls", "october messages", "verdict", "classification", "status",
            "reviewed by", "date sent", "messages", "message"}
# something that is plausibly a URL: optional scheme, a dotted host, optional path
_URLISH = re.compile(r"^(?:(?:https?|hxxps?)://)?(?:[^\s/@]+@)?[A-Za-z0-9._~%\-]+(?:\[\.\]|\.)[A-Za-z0-9\[\]._~%\-]+(?::\d+)?(?:[/?#]\S*)?$",
                     re.I)         # the optional `user@` part keeps the classic `https://trusted.com@evil.com` trick
_SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "cutt.ly", "rb.gy", "is.gd", "ow.ly", "shorturl.at", "s.id",
               "tiny.cc", "goo.gl", "rebrand.ly", "lnkd.in", "buff.ly", "t.ly", "shorturl.asia"}
_LABELS = {"live": "live", "alive": "live", "dead": "dead", "unsure": "unsure", "unknown": "unsure"}


def refang(text: str) -> str:
    s = re.sub(r"\[\.\]|\(\.\)|\[dot\]", ".", text, flags=re.I)
    s = re.sub(r"(?i)^hxxp", "http", s)
    return re.sub(r"\[://\]|\[:\]//", "://", s)


def clean_token(token: str) -> str:
    s = token.translate(_INVISIBLE).replace("\u00a0", " ").strip()
    while s and s[-1] in _TRAILING:
        s = s[:-1]
    return refang(s.lstrip("<(\"'"))


def links_in_cell(value: Any) -> List[str]:
    """Every link-looking piece of one cell, cleaned. Never raises."""
    if value is None:
        return []
    text = str(value).translate(_INVISIBLE).replace("\u00a0", " ")
    out: List[str] = []
    for token in text.split():
        t = clean_token(token)
        if not t or t.lower() in _HEADERS:
            continue
        if _URLISH.match(t) and not re.fullmatch(r"[\d.]+", t):
            out.append(t)
    return out


def canonical_key(url: str) -> str:
    """Same link spelled differently -> same key: scheme and a trailing '/' are ignored, host is lower case."""
    probe = url if re.match(r"(?i)^https?://", url) else "https://" + url
    try:
        p = urlsplit(probe)
        host = (p.hostname or "").lower()
        path = p.path.rstrip("/")
        return host + (f":{p.port}" if p.port else "") + path + (f"?{p.query}" if p.query else "")
    except ValueError:
        return url.lower()


def normalise_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _LABELS.get(str(value).strip().lower(), str(value).strip().lower()[:20] or None)


def clean(entries: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """entries: {'cell': value, 'source': 'file:row', 'manual': label-or-None}. Returns (unique records, stats)."""
    stats = {"cells": 0, "empty_or_header": 0, "links_found": 0, "duplicates": 0, "refused_by_gate": 0}
    seen: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for e in entries:
        stats["cells"] += 1
        links = links_in_cell(e.get("cell"))
        if not links:
            stats["empty_or_header"] += 1
            continue
        for raw in links:
            stats["links_found"] += 1
            key = canonical_key(raw)
            rec = seen.get(key)
            if rec is None:
                host = key.split("/")[0].split("?")[0].split(":")[0]
                rec = {"raw": raw, "key": key, "sources": [], "manual": [], "note": "",
                       "shortener": host in _SHORTENERS}
                norm = urlnorm.normalize(raw)
                if not norm.ok:
                    rec["note"] = f"refused by the safety gate: {norm.fail_reason}"
                    stats["refused_by_gate"] += 1
                seen[key] = rec
                order.append(key)
            else:
                stats["duplicates"] += 1
            rec["sources"].append(e["source"])
            label = normalise_label(e.get("manual"))
            if label and label not in rec["manual"]:
                rec["manual"].append(label)
    return [seen[k] for k in order], stats


def read_workbook(path: str) -> List[Dict[str, Any]]:
    """One entry per link cell. The link column is the one whose header says link/url, else column A; the
    team's label comes from a column headed Classification / Verdict in the same row."""
    import openpyxl                                            # lazy: host helper, not a repo dependency

    wb = openpyxl.load_workbook(path, data_only=True)
    entries: List[Dict[str, Any]] = []
    name = os.path.basename(path)
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        # header row = the first row among the first 3 that names a link/label column
        link_col, label_col, start = 0, None, 0
        for i, row in enumerate(rows[:3]):
            heads = [str(c).strip().lower() if c is not None else "" for c in row]
            for j, h in enumerate(heads):
                if h in ("url", "urls", "links", "link"):
                    link_col, start = j, i + 1
                if h in ("verdict", "classification"):
                    label_col = j
        for i in range(start, len(rows)):
            row = rows[i]
            cell = row[link_col] if link_col < len(row) else None
            manual = row[label_col] if label_col is not None and label_col < len(row) else None
            entries.append({"cell": cell, "source": f"{name}:{ws.title}:{i + 1}", "manual": manual})
    return entries


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workbooks", nargs="+")
    ap.add_argument("--out-dir", default=r"D:\LeJetData\input")
    ap.add_argument("--name", default="smishing_links")
    a = ap.parse_args(argv)

    entries: List[Dict[str, Any]] = []
    for path in a.workbooks:
        part = read_workbook(path)
        print(f"{os.path.basename(path)}: {len(part)} rows read")
        entries += part
    records, stats = clean(entries)

    os.makedirs(a.out_dir, exist_ok=True)
    urls = [r["raw"] for r in records if not r["note"]]
    with open(os.path.join(a.out_dir, f"{a.name}.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(urls) + "\n")
    with open(os.path.join(a.out_dir, f"{a.name}.meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, indent=1, ensure_ascii=True)

    print()
    print(f"cells read ................ {stats['cells']}")
    print(f"  empty or header cells ... {stats['empty_or_header']}")
    print(f"links found ............... {stats['links_found']}")
    print(f"  duplicates merged ....... {stats['duplicates']}")
    print(f"unique links .............. {len(records)}")
    print(f"  refused by the safety gate (kept in the metadata, not in the URL list): {stats['refused_by_gate']}")
    print(f"  link shorteners ......... {sum(1 for r in records if r['shortener'])}")
    manual: Dict[str, int] = {}
    for r in records:
        for m in r["manual"] or ["(no label)"]:
            manual[m] = manual.get(m, 0) + 1
    print("team labels on the unique links:", dict(sorted(manual.items())))
    print(f"wrote {len(urls)} URLs to {os.path.join(a.out_dir, a.name + '.txt')} (never open it; use the defanged copy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
