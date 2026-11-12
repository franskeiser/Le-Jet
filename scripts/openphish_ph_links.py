#!/usr/bin/env python
"""
openphish_ph_links.py -- Philippines-related phishing links from the OpenPhish academic repository (its git history).

    python scripts/openphish_ph_links.py [--repo C:\\Users\\Franz\\academic] [--out D:\\LeJetData\\openphish] [--defang]

The repository is a daily series of snapshots of feed.csv (about 600 phishing URLs each). This script takes ONE snapshot per day
(the newest commit of that day), reads every row, keeps each URL the first day it appears, and writes two files into --out:

  openphish_ph_strict.csv   links with a strong Philippines signal: a Philippine brand named by the feed (BDO, Land Bank, BPI, GCash,
                            Maya, Metrobank, ...), a .ph domain, or hosting in the Philippines. Use this one.
  openphish_ph_broad.csv    everything with ANY weak signal too (Tagalog page, a Philippine word in the URL, regional brands such as
                            Shopee/Grab). Noisy: Shopee scams of other countries, language-detector false alarms. Do not treat it
                            as labelled Philippine smishing.

URLs are written as real links unless --defang is given (then `hxxp://a[.]b`, safe to look at and paste). The files hold REAL phishing
addresses: they are only ever opened by the lab (scripts/openphish_check.py), never in your own browser, and the OpenPhish academic
terms (which you accepted) apply to them: keep them out of the repository and out of anything you publish.

Needs `git` on PATH. Reads the repository only; changes nothing in it.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_REPO = Path.home() / "academic"
DEFAULT_OUT = Path(r"D:\LeJetData\openphish")

# Brands that are Philippine by themselves (a Shopee or Grab scam may be aimed at any Southeast Asian country, so those are only "weak").
PH_ONLY_BRAND = re.compile(
    r"^(bdo unibank|land bank of the philippines|bank of the philippine|bpi\b|gcash|maya|paymaya|metrobank|unionbank|pnb|"
    r"philippine national bank|security bank|rcbc|chinabank|philhealth|sss\b|pag-?ibig|lto\b|meralco|pldt|globe|smart\b|pagcor|dito|jollibee)",
    re.I)
# Weak signals (used for the broad file only).
WEAK_BRAND = re.compile(
    r"gcash|maya|paymaya|bpi\b|\bbdo\b|metrobank|unionbank|landbank|\bpnb\b|security ?bank|rcbc|chinabank|globe|smart comm|philhealth|\bsss\b|"
    r"pag-?ibig|\blto\b|ltfrb|\bbir\b|\bnbi\b|grab|shopee|lazada|pagcor|philippine|\bph\b|jollibee|meralco|converge|pldt|dito", re.I)
WEAK_URL = re.compile(
    r"gcash|paymaya|bpi|bdo-|unionbank|landbank|philhealth|pag-?ibig|lto[-.]|ltfrb|meralco|pldt|\.ph(/|$|:)|\.com\.ph|philippin|pinoy|"
    r"pilipinas|pagcor", re.I)

COLUMNS = ["first_seen_day", "why", "signal", "brand", "sector", "country_code", "tld", "lang", "url"]


def defang(url: str) -> str:
    """http://a.b -> hxxp://a[.]b (safe to display; cannot be clicked)."""
    return re.sub(r"\.", "[.]", re.sub(r"^http", "hxxp", url.strip(), flags=re.I))


def refang(url: str) -> str:
    """The reverse of defang()."""
    return re.sub(r"\[\.\]|\(\.\)", ".", re.sub(r"^hxxp", "http", url.strip(), flags=re.I))


def weak_reasons(row: Dict[str, str]) -> List[str]:
    """Every Philippines signal this feed row has, strong or weak."""
    url = row.get("url") or ""
    lang = (row.get("page_language") or "").split(":")[0]
    why = []
    if row.get("country_code") == "PH":
        why.append("hosted_in_PH")
    if (row.get("tld") or "") == "ph":
        why.append("tld_ph")
    if lang in ("tl", "fil"):
        why.append("tagalog_page")
    if WEAK_BRAND.search(row.get("brand") or ""):
        why.append("ph_brand")
    if WEAK_URL.search(url):
        why.append("ph_words_in_url")
    return why


def strong_signal(row: Dict[str, str], why: Iterable[str]) -> str:
    """The strong signals only ('PH brand + .ph domain' ...), or '' when there is none."""
    why = list(why)
    sig = []
    if PH_ONLY_BRAND.search(row.get("brand") or ""):
        sig.append("PH brand")
    if "tld_ph" in why:
        sig.append(".ph domain")
    if "hosted_in_PH" in why:
        sig.append("hosted in PH")
    return " + ".join(sig)


def _git(repo: Path, *args: str, stdin: Optional[bytes] = None) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, check=True).stdout


def daily_snapshots(repo: Path) -> Dict[str, str]:
    """{day: newest commit of that day that touched feed.csv}"""
    by_day: Dict[str, str] = {}
    for line in _git(repo, "log", "--format=%H %ad", "--date=short", "--", "feed.csv").decode().splitlines():
        if line.strip():
            sha, day = line.split()
            by_day.setdefault(day, sha)               # newest first: the first one seen per day wins
    return by_day


def snapshot_rows(repo: Path, by_day: Dict[str, str]) -> Iterable[tuple]:
    """(day, row dict) for every row of every daily snapshot, oldest day first."""
    days = sorted(by_day)
    wanted = "".join(f"{by_day[d]}:feed.csv\n" for d in days).encode()
    buf = _git(repo, "cat-file", "--batch", stdin=wanted)
    pos = 0
    for day in days:
        nl = buf.index(b"\n", pos)
        head = buf[pos:nl].split()
        if len(head) < 3:                              # "<object> missing"
            pos = nl + 1
            continue
        size = int(head[2])
        body = buf[nl + 1: nl + 1 + size]
        pos = nl + 1 + size + 1
        for row in csv.DictReader(io.StringIO(body.decode("utf-8", "replace"))):
            yield day, row


def collect(repo: Path) -> List[Dict[str, str]]:
    """One record per URL (the day it first appeared) that has at least one Philippines signal."""
    by_day = daily_snapshots(repo)
    if not by_day:
        raise SystemExit(f"no feed.csv history found in {repo}")
    print(f"{len(by_day)} daily snapshots, {min(by_day)} .. {max(by_day)}", flush=True)
    found: Dict[str, Dict[str, str]] = {}
    scanned = 0
    for day, row in snapshot_rows(repo, by_day):
        scanned += 1
        url = row.get("url") or ""
        why = weak_reasons(row)
        if why and url not in found:
            found[url] = {"first_seen_day": day, "why": "+".join(why), "signal": strong_signal(row, why), "brand": row.get("brand", ""),
                          "sector": row.get("sector", ""), "country_code": row.get("country_code", ""), "tld": row.get("tld", ""),
                          "lang": (row.get("page_language") or "").split(":")[0], "url": url}
    print(f"{scanned} rows scanned, {len(found)} unique links with any Philippines signal", flush=True)
    return sorted(found.values(), key=lambda r: r["first_seen_day"], reverse=True)


def write(path: Path, rows: List[Dict[str, str]], defanged: bool) -> None:
    cols = [("url_defanged" if c == "url" and defanged else c) for c in COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([defang(r["url"]) if c == "url" and defanged else r[c] for c in COLUMNS])
    print(f"wrote {path} ({len(rows)} links)", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="the cloned OpenPhish academic repository")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="folder for the two CSV files (keep it OUTSIDE the project repository)")
    ap.add_argument("--defang", action="store_true", help="write hxxp://a[.]b instead of real links (files are then named *_defanged.csv)")
    a = ap.parse_args(argv)
    if not (a.repo / ".git").exists():
        print(f"{a.repo} is not a git clone of the OpenPhish academic repository", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    rows = collect(a.repo)
    strict = [r for r in rows if r["signal"]]
    suffix = "_defanged" if a.defang else ""
    write(a.out / f"openphish_ph_strict{suffix}.csv", strict, a.defang)
    write(a.out / f"openphish_ph_broad{suffix}.csv", rows, a.defang)
    print("strict by signal:", dict(Counter(r["signal"] for r in strict)))
    print("strict by brand :", dict(Counter(r["brand"] or "(none)" for r in strict).most_common(8)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
