#!/usr/bin/env python
r"""
build_labeled_input.py -- HOST-SIDE: one labelled link list for the lab's feature extractor.

    python scripts/build_labeled_input.py

Keeps ONE table that grows, D:\LeJetData\datasets\feature_input.csv, with a label per link:

    url, status, source, trust, url_id, date_added

Each run adds only what is new: the smishing master links that are not in the table yet, and the SAME number of
new legitimate links, each stamped with the date it was added (`date_added`, your local date). Old rows are never changed or
removed, so the table is also the record of what was given to the lab and when. For the new rows it writes the lab
input files to run:

    D:\LeJetData\datasets\feature_input_<YYYYMMDD_HHMM>_partNofM.txt   one defanged link per line

and `scripts\run_parts.ps1` runs them. The first time, the newest per-day file (feature_input_YYYYMMDD.csv, the
older way of doing this) becomes the start of the table, every row dated that day.

It is also run automatically by the link hunter: after `hunt.py --import-lab` the master's links that the hunter's
own lab check found live are handed over with --only-live (see LeJetLinkHunter\README.md).

Where the legitimate links come from (the `source` column says which, per row):
  * legit_sms_real        links found in the REAL legitimate messages of the project's SMS dataset
                          (data/raw/sms/v2_2026-09-15.csv), but ONLY from senders in SMS_LEGIT_ALLOW: that
                          dataset's "legitimate" label is noisy (it holds lookalike links such as
                          gcash-10-year.lc). At most --per-domain per site, so one sender cannot dominate.
  * legit_curated_homepage  hand-picked lists of well-known legitimate sites: the one the project already uses
                          (D:\LeJetData\input\benign01_push.txt.defanged.txt) and a second one
                          (D:\LeJetData\input\legit_curated_extra.txt, written from general knowledge, nothing visited).
  Nothing is crawled or invented: no legitimate link is built by changing a smishing link.

Safety / labelling rules:
  * every URL in every output is DEFANGED (hxxps://a[.]com); the lab's push step un-defangs it itself;
  * a legitimate link on the same registrable domain as any master link is never used (no contradictory labels);
  * `status` is `phishing` (the public set's word) for master rows and `legitimate` for the rest. The master also
    holds feed links that are phishing but not proven to be SMS-delivered: the `trust` column keeps that
    distinction (`smishing-verified` = confirmed by you), so filter on it if you need confirmed smishing only;
  * the two classes are interleaved (each shuffled with a fixed seed) so EVERY part has both classes in the overall
    proportion: parts are run one after another and links die with time, so a part with one class only would
    bias the liveness.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from lejet import domains, safety, urlnorm                    # noqa: E402
import make_push_file                                         # noqa: E402

DEFAULT_MASTER = r"D:\LeJetLinkHunter\MASTER_smishing_links.csv"
DEFAULT_SMS = os.path.join(ROOT, "data", "raw", "sms", "v2_2026-09-15.csv")
DEFAULT_CURATED = [r"D:\LeJetData\input\benign01_push.txt.defanged.txt", r"D:\LeJetData\input\legit_curated_extra.txt"]
DEFAULT_OUT = r"D:\LeJetData\datasets"
DEFAULT_TRANCO_POOL = r"D:\LeJetData\input\legit_tranco_pool.txt"
POSITIVE, NEGATIVE = "phishing", "legitimate"
CUMULATIVE = "feature_input.csv"
COLUMNS_IN = ["url", "status", "source", "trust", "url_id", "date_added"]

# The SMS dataset's "legitimate" messages contain some links that are clearly NOT legitimate (for example
# gcash-10-year.lc and mybdo.onlinefunds-ph.com): its labels are noisy. A phishing link labelled legitimate would
# poison the feature table, so a real-SMS link is used only if its registrable domain is in this list of
# organisations that are recognisably the real sender (reviewed by hand, 2026-10-15; edit it if you disagree).
# Generic shorteners (bit.ly, tinyurl, onelink, page.link...) are left out: their destination is unknown.
SMS_LEGIT_ALLOW = {
    "coins.ph", "smart.com.ph", "globe.com.ph", "glbe.co", "tntph.com", "smrt.ph", "unionbankph.com", "gcash.com",
    "paymaya.com", "maya.ph", "landbank.com", "e.gov.ph", "tourism.gov.ph", "sec.gov.ph", "usc.edu", "nus.edu.sg",
    "adp.com", "binance.com", "wikipedia.com", "netflix.com", "icicibank.com", "moneycontrol.com", "fb.com",
    "fb.me", "m.me", "t.me", "lzd.co", "shp.ee", "landers.ph",
}


def defang(url: str) -> str:
    return safety.defang_url(url)


def reg_domain(url: str) -> str:
    m = re.match(r"(?i)^[a-z][a-z0-9+.\-]*://([^/?#]+)", url)
    host = (m.group(1) if m else url.split("/")[0]).split("@")[-1].split(":")[0]
    return domains.registrable_domain(host) or host.lower()


def read_master(path: str) -> List[Dict[str, str]]:
    """The master's rows; the newer of the sheet and its UNSAVED_COPY (written while Excel had it open)."""
    partial = os.path.splitext(path)[0] + "_UNSAVED_COPY.csv"
    use = path
    if os.path.exists(partial) and (not os.path.exists(path) or os.path.getmtime(partial) > os.path.getmtime(path)):
        use = partial
    with open(use, encoding="utf-8-sig", newline="") as fh:
        return [r for r in csv.DictReader(fh) if (r.get("link") or "").strip()]


def sms_legit_links(path: str, allow=None) -> List[str]:
    """Distinct links found in the legitimate messages of the SMS dataset, shaped like the lab will see them,
    from the allowed senders only (`allow`, default SMS_LEGIT_ALLOW)."""
    from src.features import link_features as lf
    allow = SMS_LEGIT_ALLOW if allow is None else allow
    seen, out = set(), []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("label") or "").strip().lower() != "legitimate":
                continue
            for found in lf.extract_links(row.get("sms") or ""):
                url = lf.normalize_link(found["raw"])
                key = url.lower()
                if reg_domain(url) not in allow:
                    continue
                if key not in seen:
                    seen.add(key)
                    out.append(url)
    return out


def curated_links(paths) -> List[str]:
    """Links of one or several curated files (`id<TAB>defanged` or one defanged link per line)."""
    out, seen = [], set()
    for path in ([paths] if isinstance(paths, str) else paths):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                url = make_push_file.refang(line.split("\t")[-1])
                if url.lower() not in seen:
                    seen.add(url.lower())
                    out.append(url)
    return out


def pick_legit(pool_a: List[str], pool_b: List[str], n: int, per_domain: int, rng: random.Random,
               banned_domains: set, banned_links: set,
               already_used: Optional[Dict[str, int]] = None,
               pool_c: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """n (url, source) pairs: real SMS links first (at most `per_domain` per site), then the curated lists, then the
    large Tranco-based pool (`pool_c`). `already_used` = how many links per site the table already holds (they count
    against the cap). If the pools are short the per-site cap is raised one step at a time until n is reached or
    nothing is left."""
    def usable(pool):
        items = [u for u in pool if u.lower() not in banned_links and reg_domain(u) not in banned_domains]
        rng.shuffle(items)
        return items
    a, b, c = usable(pool_a), usable(pool_b), usable(pool_c or [])
    chosen: List[Tuple[str, str]] = []
    per: Dict[str, int] = dict(already_used or {})
    taken = set()
    cap = max(1, per_domain)
    while len(chosen) < n:
        progressed = False
        for source, pool in (("legit_sms_real", a), ("legit_curated_homepage", b), ("legit_tranco", c)):
            for url in pool:
                if len(chosen) >= n:
                    break
                d = reg_domain(url)
                if url in taken or per.get(d, 0) >= cap:
                    continue
                taken.add(url)
                per[d] = per.get(d, 0) + 1
                chosen.append((url, source))
                progressed = True
        if len(chosen) >= n or (not progressed and all(url in taken for url in a + b + c)):
            break                                   # done, or both pools are used up
        cap += 1                                    # nothing more fits under the cap: allow one more per site
    return chosen


def link_id(url: str) -> str:
    """The id the lab will give this link (a hash of the real link)."""
    return urlnorm.compute_url_id(make_push_file.refang(defang(url)))


def build_new_rows(master_rows: List[Dict[str, str]], legit: List[Tuple[str, str]], seed: int,
                   date_added: str, have: set) -> List[Dict[str, str]]:
    """Rows for links that are not in the table yet, both classes interleaved, each stamped with its date."""
    rows: List[Dict[str, str]] = []
    seen = set(have)
    for r in master_rows:
        link = r["link"].strip()
        uid = link_id(link)
        if uid in seen:
            continue
        seen.add(uid)
        rows.append({"url": defang(link), "status": POSITIVE, "source": "smishing_master:" + (r.get("source") or ""),
                     "trust": r.get("trust") or "", "url_id": uid, "date_added": date_added})
    for url, source in legit:
        uid = link_id(url)
        if uid in seen:
            continue
        seen.add(uid)
        rows.append({"url": defang(url), "status": NEGATIVE, "source": source, "trust": "", "url_id": uid,
                     "date_added": date_added})
    return interleave(rows, seed)


def interleave(rows: List[Dict[str, str]], seed: int) -> List[Dict[str, str]]:
    """Each class shuffled (fixed seed), then merged so that ANY stretch of the list has the classes in the overall
    proportion. A plain shuffle can leave a whole lab part with one class only."""
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        groups.setdefault(r["status"], []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    total = {k: len(v) for k, v in groups.items()}
    taken = {k: 0 for k in groups}
    out: List[Dict[str, str]] = []
    while len(out) < len(rows):
        # the class that is furthest behind its share goes next (ties: alphabetical, so it is deterministic)
        k = min((k for k in groups if taken[k] < total[k]), key=lambda k: (taken[k] / total[k], k))
        out.append(groups[k][taken[k]])
        taken[k] += 1
    return out


TRUST_ORDER = {"smishing-verified": 0, "human-verified": 0, "urlscan-malicious": 1}


def group_key(url: str) -> str:
    """The hosting platform of a link (pages.dev, github.io, vercel.app...) or else its registrable domain.

    A phishing kit is often copied hundreds of times onto one free platform; links of one platform are near-copies of
    each other, so they are counted (and capped) together, not as hundreds of different sites."""
    m = re.match(r"(?i)^[a-z][a-z0-9+.\-]*://([^/?#]+)(/[^?#]*)?", url)
    host = (m.group(1) if m else url.split("/")[0]).split("@")[-1].split(":")[0]
    path = (m.group(2) if m else "") or "/"
    info = domains.classify_hosting(host, path)
    if (info.free_hosting_subdomain or info.platform_path_hosting) and info.hosting_platform:
        return info.hosting_platform
    return reg_domain(url)


def choose_new_phishing(candidates: List[Dict[str, str]], cap: int, already: Dict[str, int],
                        rng: random.Random) -> Tuple[List[Dict[str, str]], int]:
    """(ordered candidates, number held back by the cap).

    At most `cap` links per hosting platform in the whole table (`already` = what the table holds; 0 = no cap). The
    order takes one link from each platform in turn (confirmed links first inside a platform), so if only the first N
    are used the selection is as varied as possible."""
    by_group: Dict[str, List[Dict[str, str]]] = {}
    for r in candidates:
        by_group.setdefault(group_key(r["link"].strip()), []).append(r)
    held = 0
    for key, items in by_group.items():
        rng.shuffle(items)
        items.sort(key=lambda r: TRUST_ORDER.get(r.get("trust") or "", 2))        # stable: confirmed links first
        if cap > 0:
            room = max(0, cap - already.get(key, 0))
            held += max(0, len(items) - room)
            del items[room:]
    order = sorted(by_group, key=lambda k: (min((TRUST_ORDER.get(r.get("trust") or "", 2) for r in by_group[k]), default=2), k))
    out: List[Dict[str, str]] = []
    depth = 0
    while any(len(by_group[k]) > depth for k in order):
        out.extend(by_group[k][depth] for k in order if len(by_group[k]) > depth)
        depth += 1
    return out, held


def read_extra_links(path: str, limit: int = 0) -> Tuple[List[str], Dict[str, int]]:
    """The first `limit` rows (0 = all) of a list of phishing links: a CSV with a `url` column (PhishTank's
    verified_online.csv) or one link per line. Returns (usable links, why others were skipped). A link is usable if
    it is a plain http(s) address with no whitespace or control characters and at most 2048 long."""
    raw: List[str] = []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        head = fh.readline()
        fh.seek(0)
        cols = [c.strip().lower() for c in next(csv.reader([head]), [])]
        if "url" in cols:
            reader = csv.reader(fh)
            idx = cols.index("url")
            next(reader, None)
            for row in reader:
                if limit and len(raw) >= limit:
                    break
                raw.append(row[idx].strip() if len(row) > idx else "")
        else:
            for line in fh:
                if limit and len(raw) >= limit:
                    break
                if line.strip() and not line.lstrip().startswith("#"):
                    raw.append(line.strip())
    skipped: Dict[str, int] = {}
    links: List[str] = []
    for url in raw:
        if not re.match(r"(?i)^https?://", url):
            reason = "not an http(s) link"
        elif re.search(r"[\s\x00-\x1f\x7f]", url):
            reason = "whitespace or control characters"
        elif len(url) > 2048:
            reason = "longer than 2048"
        else:
            links.append(url)
            continue
        skipped[reason] = skipped.get(reason, 0) + 1
    return links, skipped


def build_extra_rows(links: List[str], source: str, trust: str, date_added: str, seen: set,
                     skipped: Dict[str, int]) -> List[Dict[str, str]]:
    """Phishing rows for links not in the table yet, in the order given (no legitimate partner, no platform cap:
    the user chose these links on purpose; scripts/build_feature_dataset.py still caps and balances the final table)."""
    rows: List[Dict[str, str]] = []
    for url in links:
        uid = link_id(url)
        if uid in seen:
            skipped["already in the table or the master"] = skipped.get("already in the table or the master", 0) + 1
            continue
        seen.add(uid)
        rows.append({"url": defang(url), "status": POSITIVE, "source": "extra_phishing:" + source, "trust": trust,
                     "url_id": uid, "date_added": date_added})
    return rows


def read_table(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r.setdefault("date_added", "")
    return rows


def load_table(out_dir: str) -> Tuple[List[Dict[str, str]], str]:
    """The cumulative table (feature_input.csv) and a note on where it came from.

    The newer of the table and its UNSAVED_COPY (written while Excel had it open) is used. If there is no table yet,
    the newest older per-day file (feature_input_YYYYMMDD.csv) becomes the start of it, every row stamped with that
    file's date."""
    path = os.path.join(out_dir, CUMULATIVE)
    partial = os.path.join(out_dir, "feature_input_UNSAVED_COPY.csv")
    if os.path.exists(partial) and (not os.path.exists(path) or os.path.getmtime(partial) > os.path.getmtime(path)):
        return read_table(partial), "the unsaved copy"
    if os.path.exists(path):
        return read_table(path), ""
    legacy = sorted(p for p in glob.glob(os.path.join(out_dir, "feature_input_*.csv"))
                    if re.search(r"feature_input_\d{8}\.csv$", p))
    if not legacy:
        return [], ""
    newest = legacy[-1]
    stamp = re.search(r"feature_input_(\d{4})(\d{2})(\d{2})\.csv$", newest).groups()
    date = "-".join(stamp)
    rows = read_table(newest)
    for r in rows:
        r["date_added"] = r["date_added"] or date
    return rows, f"started from {os.path.basename(newest)} (every row dated {date})"


def write_table(out_dir: str, rows: List[Dict[str, str]]) -> str:
    """Write feature_input.csv; if Excel has it open, write feature_input_UNSAVED_COPY.csv instead (the next run uses
    the newer of the two). Returns the path written."""
    path = os.path.join(out_dir, CUMULATIVE)
    partial = os.path.join(out_dir, "feature_input_UNSAVED_COPY.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS_IN)
        w.writeheader()
        w.writerows(rows)
    try:
        os.replace(tmp, path)
        if os.path.exists(partial):
            os.remove(partial)
        return path
    except PermissionError:
        os.replace(tmp, partial)
        print(f"\n  !! {CUMULATIVE} is open in another program (Excel?). Nothing is lost: the latest table is in\n"
              f"     {os.path.basename(partial)}. Close the table and run any command again; it will be merged.")
        return partial


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--sms", default=DEFAULT_SMS)
    ap.add_argument("--curated", nargs="+", default=DEFAULT_CURATED, help="curated legitimate-link files")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--per-domain", type=int, default=4, help="legitimate real-SMS links per site (default 4)")
    ap.add_argument("--platform-cap", type=int, default=60,
                    help="smishing links per hosting platform (pages.dev, github.io...) in the whole table (default 20; 0 = no cap)")
    ap.add_argument("--chunk", type=int, default=120, help="links per lab input file (default 120, about 3 minutes each)")
    ap.add_argument("--legit-ratio", type=float, default=1.0, metavar="R",
                    help="legitimate links per new phishing link (default 1.0). Legitimate pages load more often (about 59%%) "
                         "than phishing ones (about 33%%), so about 0.6 balances the pages that LOAD without wasting lab time")
    ap.add_argument("--add-legit", type=int, default=0, metavar="N",
                    help="add N MORE legitimate links on top of the partners for new smishing links (the final table is "
                         "balanced on pages that LOADED, and phishing pages load far less often than legitimate ones)")
    ap.add_argument("--tranco-pool", metavar="FILE", default=None,
                    help="the large pool made by scripts/build_legit_pool.py (default: " + DEFAULT_TRANCO_POOL + " if it exists)")
    ap.add_argument("--extra-phishing", metavar="FILE",
                    help="also add phishing links from a file (a CSV with a `url` column such as PhishTank's "
                         "verified_online.csv, or one link per line). They get no legitimate partner and no platform cap.")
    ap.add_argument("--limit", type=int, default=0, help="with --extra-phishing: only the first N rows of the file (0 = all)")
    ap.add_argument("--extra-source", default="", help="with --extra-phishing: the source name (default: the file name)")
    ap.add_argument("--extra-trust", choices=("human-verified", "feed-listed", "smishing-verified"), default="feed-listed",
                    help="with --extra-phishing: the trust label (human-verified for PhishTank)")
    ap.add_argument("--only-live", action="store_true",
                    help="only master links the hunter's own lab check found live (what the hunter hands over)")
    ap.add_argument("--date", help="the date to stamp on new rows (default: today, local time)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be added and write nothing")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)

    master = read_master(a.master)
    if a.only_live:
        master = [r for r in master if (r.get("lab_page_fetched") or "") == "1"]
    if not master:
        print("The master sheet has no " + ("live " if a.only_live else "") + "links.")
        return 1
    date_added = a.date or datetime.now().strftime("%Y-%m-%d")           # your local date, not UTC
    existing, note = load_table(a.out_dir)
    have = {r["url_id"] for r in existing}
    new_master = []
    seen = set(have)
    for r in master:
        uid = link_id(r["link"].strip())
        if uid not in seen:
            seen.add(uid)
            new_master.append(r)
    print(f"table: {len(existing)} rows" + (f" ({note})" if note else "") +
          f" | master links: {len(master)} | not in the table yet: {len(new_master)}")
    extra_rows: List[Dict[str, str]] = []
    if a.extra_phishing:
        if not os.path.exists(a.extra_phishing):
            print(f"Not found: {a.extra_phishing}")
            return 1
        links, skipped = read_extra_links(a.extra_phishing, a.limit)
        source = a.extra_source or os.path.splitext(os.path.basename(a.extra_phishing))[0]
        extra_rows = build_extra_rows(links, source, a.extra_trust, date_added, seen, skipped)
        print(f"extra phishing list {os.path.basename(a.extra_phishing)}" + (f" (first {a.limit} rows)" if a.limit else "") +
              f": {len(links)} usable links, {len(extra_rows)} new" + (f" | left out: {skipped}" if skipped else ""))
    if not new_master and not extra_rows and not a.add_legit:
        if note and not a.dry_run:
            write_table(a.out_dir, existing)             # the migration itself is worth saving
            print(f"saved {CUMULATIVE} ({len(existing)} rows).")
        print("Nothing new to add.")
        return 0

    rng = random.Random(a.seed)
    master_all = read_master(a.master)
    banned_domains = {reg_domain(r["link"].strip()) for r in master_all}
    banned_domains |= {reg_domain(make_push_file.refang(r["url"])) for r in existing if r["status"] == POSITIVE}
    banned_links = {r["link"].strip().lower() for r in master_all}
    banned_links |= {make_push_file.refang(r["url"]).lower() for r in existing}
    used_per_domain: Dict[str, int] = {}
    for r in existing:
        if r["status"] == NEGATIVE:
            d = reg_domain(make_push_file.refang(r["url"]))
            used_per_domain[d] = used_per_domain.get(d, 0) + 1
    pool_a, pool_b = sms_legit_links(a.sms), curated_links(a.curated)
    pool_c: List[str] = []
    tranco = a.tranco_pool if a.tranco_pool is not None else (DEFAULT_TRANCO_POOL if os.path.exists(DEFAULT_TRANCO_POOL) else "")
    if tranco:
        pool_c = curated_links(tranco)
        print(f"large legitimate pool {os.path.basename(tranco)}: {len(pool_c)} links")
    group_counts: Dict[str, int] = {}
    for r in existing:
        if r["status"] == POSITIVE:
            k = group_key(make_push_file.refang(r["url"]))
            group_counts[k] = group_counts.get(k, 0) + 1
    ordered, held = choose_new_phishing(new_master, a.platform_cap, group_counts, random.Random(a.seed))
    # legitimate links: first one partner for every new smishing link, then --add-legit more on top of that
    ratio = a.legit_ratio if a.legit_ratio > 0 else 1.0
    need = int(-(-len(ordered) * ratio // 1)) + a.add_legit          # ceil(ratio * smishing links) + extras
    legit = pick_legit(pool_a, pool_b, need, a.per_domain, rng, banned_domains, banned_links, used_per_domain, pool_c)
    # never more smishing links than the legitimate ones can partner (at `ratio` legitimate per smishing link)
    take = ordered[:min(len(ordered), int(len(legit) / ratio))]
    waiting = len(ordered) - len(take)
    new_rows = build_new_rows(take, legit, a.seed, date_added, have) + extra_rows
    n_pos = sum(r["status"] == POSITIVE for r in new_rows)
    n_neg = len(new_rows) - n_pos
    print(f"adding {len(new_rows)} rows dated {date_added}: {n_pos} {POSITIVE}, {n_neg} {NEGATIVE}")
    if a.add_legit and len(legit) - len(take) < a.add_legit:
        print(f"  note: asked for {a.add_legit} extra legitimate links, found only {max(0, len(legit) - len(take))}; "
              f"add a bigger pool with scripts\\build_legit_pool.py or --curated <file>.")
    if extra_rows and n_neg < n_pos:
        print(f"  note: the extra list has no legitimate partners, so the table now holds more {POSITIVE} than {NEGATIVE} "
              f"links; the final table is balanced and platform-capped afterwards (build_feature_dataset.py).")
    if held:
        print(f"  {held} smishing link(s) held back by the platform cap (at most {a.platform_cap} per hosting platform, "
              f"e.g. pages.dev); they stay in the master. --platform-cap 0 removes the cap.")
    if waiting:
        print(f"  !! {waiting} smishing link(s) are WAITING: there are no more legitimate links to match them (the pool of "
              f"unused legitimate links is empty).\n     They stay in the master and are added automatically once more "
              f"legitimate links exist. Give it more with:  --curated <a file of defanged legitimate links>\n"
              f"     (the two default lists stay in use).")
    if not new_rows:
        print("Nothing could be added.")
        return 0
    if a.dry_run:
        print("Dry run: nothing was written.")
        return 0

    os.makedirs(a.out_dir, exist_ok=True)
    table_path = write_table(a.out_dir, existing + new_rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_try = 1
    while glob.glob(os.path.join(a.out_dir, f"feature_input_{stamp}_part*of*.txt")):
        n_try += 1                       # never reuse a name: an earlier run's links may not have been run yet
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"-{n_try}"
    parts = [new_rows[i:i + a.chunk] for i in range(0, len(new_rows), a.chunk)]
    for n, part in enumerate(parts, 1):
        p = os.path.join(a.out_dir, f"feature_input_{stamp}_part{n}of{len(parts)}.txt")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(r["url"] + "\n" for r in part))
    print(f"\ntable:  {table_path} ({len(existing) + len(new_rows)} rows)")
    print(f"lab input for the new links: {len(parts)} file(s), feature_input_{stamp}_part*of{len(parts)}.txt")
    print("Run them in D:\\Le-Jet with ONE command (it picks up these files):")
    print("  powershell -ExecutionPolicy Bypass -File .\\scripts\\run_parts.ps1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
