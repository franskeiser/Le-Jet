#!/usr/bin/env python
"""
openphish_check.py -- which of the OpenPhish Philippines links are NOT dead yet? Opens them ONE AT A TIME in the Azure lab.

    python scripts/openphish_check.py [--limit 25] [--all-brands] [--again [--only-dead]] [--dry-run] [--out D:\\LeJetData\\openphish]

Links are taken newest first (the newest are the likeliest to be alive).

For each link from openphish_ph_strict.csv (made by scripts/openphish_ph_links.py) it runs scripts/run_live_azure.ps1: the link is opened
in the isolated Azure VM through the Philippine VPN exit, the result is pulled back and validated on this PC. THIS SCRIPT NEVER OPENS A
LINK ITSELF and never prints one except defanged. Needs the lab up: `ssh lejet-azure true` must work (see docs/azure_lab.md).

Writes into --out:
  openphish_ph_bank_checked_ALL.csv   every link: alive (yes / maybe / no / unknown), why, title, size, the lab's batch id
  openphish_ph_bank_NOT_DEAD.csv      only the links with alive = yes or maybe
  run\\                                the lab's working files (validated results and screenshots); scripts/openphish_score.py reads them

A re-run skips links that already have a result in the ALL file (use --again to check them anew), so an interrupted run can be resumed.
About 35 to 90 seconds per link (a dead domain is quick, an unreachable server is slow).

What "alive" means (judge() below, tested in tests/test_openphish_scripts.py): the page opened and shows content. A domain that no longer
exists, a parked page, a server that answers only with an error message (404 "Not Found", a Cloudflare tunnel error, a suspended-account
page), or one that cannot be reached at all is DEAD. A bot-protection page or an odd page the browser could show is "maybe".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_OUT = Path(r"D:\LeJetData\openphish")
LAB_SCRIPT = Path(__file__).resolve().parent / "run_live_azure.ps1"
BANKS = ("BDO Unibank", "Land Bank of the Philippines", "Bank of the Philippine Islands")
COLUMNS = ["alive", "why", "first_seen_day", "brand", "url", "final_url", "title", "page_text_start", "fail_reason", "html_fetched", "shot_ok",
           "reach", "reason", "text_chars", "shot_size", "pages", "checked_at", "lab_batch"]

# Phrases that mean "the page is gone, blocked or never was the phishing page": a takedown or removal notice, a suspended account, an
# expired or parked domain, a browser or host error, a web server's default page. They are looked for in the page TITLE and in the first
# 400 characters of the visible TEXT (the lab records those), so a page that says "has been removed" under a normal-looking title is dead too.
GONE_TEXT = re.compile(
    r"has been (removed|taken down|deleted|disabled|blocked|suspended|terminated|deactivated|closed|archived|reported)|"
    r"(was|were|is|are) (removed|taken down|deleted|disabled|blocked|suspended|deactivated)|"
    r"(page|site|website|domain|account|content|link|url|app|file|resource)\b.{0,30}\b(no longer|not|isn.?t|is not|aren.?t|are not|wasn.?t|was not) "
    r"(available|exist|exists|found|active|accessible|online|valid)|"
    r"(could not|couldn.?t|can.?t|cannot|unable to) (find|locate) (that|this|the|your)? ?(page|site|website|file|link|resource)|"
    r"no longer (available|exists?|active|accessible|online|valid|in service)|"
    r"(page|site|website|resource|file|link) (could not|can.?t|cannot|couldn.?t|does not|doesn.?t|did not) (be )?(found|exist|load)|"
    r"(page|site|website|file|resource|content|link)( was| is)? not found|\b404\b|page not found|"
    r"(site|website|page|domain|account) (is |has been )?(suspended|unavailable|disabled|expired|for sale|parked|blocked|terminated)|"
    r"domain (name )?(has )?expired|buy this domain|domain (is )?for sale|"
    r"suspected phishing|deceptive site|dangerous site|reported as (unsafe|phishing|deceptive|fraud)|phishing (site|page) (warning|detected|blocked)|"
    r"can.?t be reached|refused to connect|took too long to respond|"
    r"access denied|forbidden|not authori[sz]ed to (view|access)|"
    r"tunnel error|error 10\d\d|error 5\d\d|bad gateway|service unavailable|gateway time-?out|temporarily unavailable|"
    r"under maintenance|under construction|coming soon|"
    r"default web page|welcome to nginx|apache2 (ubuntu )?default|\bit works!|index of /|"
    r"violat\w+ (of )?(our |the )?(terms|polic)|abuse (report|complaint)|take-?down|taken down", re.I)
GONE_TEXT_MAX_CHARS = 1500     # a page with this much text is real content: a stray "forbidden" in its small print does not make it dead
ERROR_PAGE = GONE_TEXT         # (the old name, still used by the tests)


def gone_phrase(title: str, text_head: str, text_chars: int) -> str:
    """The phrase that says the page is gone ('' if none): in the title, or in the start of the text of a page with little text."""
    m = GONE_TEXT.search(title or "")
    if m:
        return m.group(0)
    if text_chars < GONE_TEXT_MAX_CHARS:                     # (-1, unknown, counts as little)
        m = GONE_TEXT.search(text_head or "")
        if m:
            return m.group(0)
    return ""


def defang(url: str) -> str:
    return re.sub(r"\.", "[.]", re.sub(r"^http", "hxxp", url.strip(), flags=re.I))


def judge(fail_reason: str, html_fetched: bool, shot: Optional[Dict[str, Any]], lab_ok: bool, has_result: bool = True) -> Tuple[str, str]:
    """(alive, why) from what the lab saw. alive is 'yes', 'maybe', 'no' or 'unknown' (the lab run itself failed)."""
    if not has_result and not lab_ok:
        return "unknown", "the lab run failed"
    shot = shot or {}
    title = str(shot.get("title") or "")
    head = str(shot.get("text_head") or "")
    chars = int(shot.get("text_chars", -1) if shot.get("text_chars") is not None else -1)
    shot_ok = bool(shot.get("ok")) and shot.get("reach") in ("live", "live_retry")

    if fail_reason in ("dns_nxdomain", "dns_no_answer"):
        return "no", "domain does not exist"
    if fail_reason == "parked_or_default_page":
        return "no", "parked or default page"
    # THE PAGE ITSELF SAYS IT IS GONE (removed, taken down, suspended, not found, expired, blocked, a server error ...): dead, even if it
    # loaded fine and even if the title looks normal
    said = gone_phrase(title, head, chars)
    if said and (shot_ok or title or head):
        return "no", f"the page says it is gone or blocked: \"{said[:50]}\""
    if fail_reason in ("http_4xx", "http_5xx"):                      # the server answers with an error status
        if chars < 100:
            return "no", f"server answers with an error page ({fail_reason}" + (f": {title[:40]}" if title else "") + ")"
        return "maybe", f"shows a page but the server says {fail_reason}"
    if fail_reason == "none" and (html_fetched or shot_ok):
        return "yes", "page loads"
    if fail_reason in ("challenge_page", "blocked_or_empty"):
        return "maybe", "bot-protection or empty page (the host answers)"
    if shot_ok and chars >= 100:
        return "maybe", f"the browser shows a page (the plain fetch said {fail_reason})"
    reason = shot.get("reason") or ""
    return "no", f"could not open ({fail_reason}{'/' + reason if reason else ''})"


def read_json_lines(path: Path) -> List[dict]:
    try:
        return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    except (OSError, ValueError):
        return []


def facts_for(run: Path, batch: str, lab_ok: bool) -> Tuple[str, str, Dict[str, Any]]:
    """Read the lab's validated output for one batch and judge it."""
    clean = read_json_lines(run / "clean" / f"{batch}.clean.jsonl")
    shots = read_json_lines(run / "clean" / "shots" / batch / "shots.jsonl")
    fetch = clean[0] if clean else {}
    s = shots[0] if shots else {}
    fail = fetch.get("fail_reason", "no_result" if not lab_ok else "?")
    alive, why = judge(fail, bool(fetch.get("html_available")), s, lab_ok, has_result=bool(clean))
    shot_ok = bool(s.get("ok")) and s.get("reach") in ("live", "live_retry")
    details = {"fail_reason": fail, "html_fetched": bool(fetch.get("html_available")), "shot_ok": shot_ok, "reach": s.get("reach", ""),
               "reason": s.get("reason", ""), "title": (s.get("title") or "")[:100], "text_chars": s.get("text_chars", -1),
               "page_text_start": (s.get("text_head") or "")[:150],
               "final_url": fetch.get("final_url") or s.get("final_url", ""),
               "shot_size": f'{s.get("width", 0)}x{s.get("height", 0)}' if shot_ok else "", "pages": len(s.get("pages", [])) if s else 0}
    return alive, why, details


def write_results(out: Path, rows: List[Dict[str, Any]]) -> None:
    for name, subset in (("openphish_ph_bank_checked_ALL.csv", rows),
                         ("openphish_ph_bank_NOT_DEAD.csv", [r for r in rows if r["alive"] in ("yes", "maybe")])):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(subset)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--links", type=Path, default=None, help="default: <out>/openphish_ph_strict.csv")
    ap.add_argument("--limit", type=int, default=25, help="how many links (default 25)")
    ap.add_argument("--all-brands", action="store_true", help="not only the Philippine banks (BDO, Land Bank, BPI) but every strict link")
    ap.add_argument("--again", action="store_true", help="check links again even if the ALL file already has a result")
    ap.add_argument("--only-dead", action="store_true", help="with --again: re-check only the links currently marked dead or unknown (skip the live ones)")
    ap.add_argument("--dry-run", action="store_true", help="only list the links (defanged) that would be checked")
    ap.add_argument("--lab-script", type=Path, default=LAB_SCRIPT)
    a = ap.parse_args(argv)

    links_csv = a.links or (a.out / "openphish_ph_strict.csv")
    if not links_csv.exists():
        print(f"{links_csv} not found: run scripts/openphish_ph_links.py first", file=sys.stderr)
        return 2
    run = a.out / "run"
    for sub in ("input", "q", "clean"):
        (run / sub).mkdir(parents=True, exist_ok=True)

    candidates = [r for r in csv.DictReader(open(links_csv, encoding="utf-8")) if a.all_brands or r["brand"] in BANKS]
    candidates.sort(key=lambda r: r["first_seen_day"], reverse=True)
    all_csv = a.out / "openphish_ph_bank_checked_ALL.csv"
    done = {r["url"]: r for r in csv.DictReader(open(all_csv, encoding="utf-8"))} if all_csv.exists() else {}
    if a.again and a.only_dead:
        candidates = [r for r in candidates if r["url"] in done and done[r["url"]]["alive"] in ("no", "unknown")]
    todo = [r for r in candidates if a.again or r["url"] not in done][: a.limit]
    print(f"{len(candidates)} candidate links, {len(done)} already checked, {len(todo)} to check now", flush=True)
    if a.dry_run:
        for r in todo:
            print(f'{r["first_seen_day"]}  {r["brand"][:12]:12}  {defang(r["url"])}')
        return 0

    results: Dict[str, Dict[str, Any]] = dict(done)               # earlier results are kept; a link checked again replaces its own row
    stamp = time.strftime("%m%d")
    existing = len([f for f in os.listdir(run / "clean") if f.endswith(".clean.jsonl")])
    for i, r in enumerate(todo, 1):
        batch = f"op{stamp}n{existing + i:02d}"                    # a new batch id per link: letters, digits only
        url_file = run / "input" / f"{batch}.txt"
        url_file.write_text(r["url"] + "\n", encoding="ascii", errors="replace")
        t0 = time.time()
        p = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(a.lab_script), "-UrlFile", str(url_file),
                            "-Batch", batch, "-InputDir", str(run / "input"), "-Quarantine", str(run / "q"), "-CleanDir", str(run / "clean")],
                           capture_output=True, text=True, timeout=600, creationflags=0x08000000 if os.name == "nt" else 0)
        alive, why, details = facts_for(run, batch, p.returncode == 0)
        print(f"[{i:02d}/{len(todo)}] {alive:7} {r['brand'][:6]:6} {defang(r['url'])[:58]:58} -> {why} ({time.time() - t0:.0f}s)", flush=True)
        results[r["url"]] = {"checked_at": time.strftime("%Y-%m-%d %H:%M"), "alive": alive, "why": why, "first_seen_day": r["first_seen_day"],
                             "brand": r["brand"], "url": r["url"], "lab_batch": batch, **details}
        write_results(a.out, list(results.values()))                # saved after every link: nothing is lost if the run is interrupted
    rows = list(results.values())
    live = [x for x in rows if x["alive"] in ("yes", "maybe")]
    print(f"done: {len(rows)} links on file, {len(live)} not dead -> {a.out / 'openphish_ph_bank_NOT_DEAD.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
