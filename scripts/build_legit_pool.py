#!/usr/bin/env python
r"""
build_legit_pool.py -- HOST-SIDE: a large pool of legitimate-site links, sampled from the Tranco ranking.

    python scripts/build_legit_pool.py [--n 4000]

Input : a Tranco list on disk (rank,domain), by default the newest D:\LeJetData\input\tranco\tranco_top100k_*.csv
        (the public list from https://tranco-list.eu, downloaded once; the download date and sha256 are in the
        tranco_<date>_INFO.txt next to it. Cite the Tranco list in the thesis).
Output: D:\LeJetData\input\legit_tranco_pool.txt        one DEFANGED link per line (hxxps://a[.]com/), which
                                                         scripts\build_labeled_input.py uses as its large pool
        D:\LeJetData\input\legit_tranco_pool.csv        domain, rank, band (for your own analysis)

Nothing is visited: this reads a file and writes two files.

What is sampled, and why:
  * only the top 100,000 sites, in three rank bands (default 15% from 1-5,000, 35% from 5,001-20,000, 50% from
    20,001-100,000). The very top is dominated by giants that refuse the lab's plain fetch (HTTP 403); the middle ranks
    are ordinary sites and load far more often. Below 100,000 the ranking contains more parked, dead and dubious sites.
  * infrastructure names (CDNs, ad and analytics networks, cloud endpoints) are skipped: they have no homepage to load.
  * names that are clearly adult or gambling sites are skipped (crude keyword filter), and so are the TLDs the phishing
    feeds are full of (.top, .xyz, .icu...): a ranked site there is more often shady than not.
  * free-hosting platforms are skipped as a whole (pages.dev, github.io...): a platform's root page says nothing about
    the sites hosted on it.
A popularity ranking is NOT a list of safe sites. It is a sensible proxy for "legitimate" and the usual choice in the
literature (the project plan names Tranco for benign sites), but a few ranked sites can be shady; scripts\build_labeled_input.py
never uses a domain that also appears among the phishing links, and the lab's own result (does the page load, what is in it)
is recorded for every link.

The sample is reproducible: the same list and the same --seed give the same pool.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from lejet import domains, safety                             # noqa: E402

TRANCO_DIR = r"D:\LeJetData\input\tranco"
OUT_TXT = r"D:\LeJetData\input\legit_tranco_pool.txt"
OUT_CSV = r"D:\LeJetData\input\legit_tranco_pool.csv"
BANDS = [(1, 5_000, 0.15), (5_001, 20_000, 0.35), (20_001, 100_000, 0.50)]

# names that are infrastructure (content delivery, ads, analytics, cloud endpoints, DNS/CT plumbing): no homepage
_INFRA = re.compile(
    r"(akamai|cloudfront|cloudflare|fastly|edgekey|edgesuite|edgecast|cdn|gstatic|googleapis|googleusercontent|"
    r"googlesyndication|googletagmanager|google-analytics|doubleclick|adnxs|adsrvr|rubiconproject|criteo|scorecardresearch|"
    r"demdex|omtrdc|2o7|amazonaws|awsdns|azure|windows\.net|msedge|trafficmanager|office365|akadns|ytimg|fbcdn|twimg|"
    r"yimg|ggpht|apple-dns|icloud-content|mzstatic|cloudapp|herokudns|nsone|dnsmadeeasy|ultradns|root-servers|"
    r"in-addr|ip6|sentry|segment\.io|newrelic|nr-data|optimizely|hotjar|branch\.io|appsflyer|adjust\.com|crashlytics|"
    r"firebaseio|workers\.dev|vercel-dns|b-cdn|jsdelivr|unpkg|bootstrapcdn|fontawesome|typekit)",
    re.I)


# adult and gambling names: popular, but not what the "legitimate" class should hold (the phishing data is full of
# gambling lures, and an adult site is not a normal SMS destination); crude on purpose, it only trims the obvious ones
_UNWANTED = re.compile(r"(xxx|porn|(?:^|[.-])sex|sex(?:y|cam|chat|shop|toy|video|tube)|adultsite|adultfriend|escort|nude|hentai|"
                       r"fuck|milf|erotic|fetish|camgirl|livecam|"
                       r"casino|gambl|poker|betting|bet365|slots|jackpot|lotto|bingo|viagra|cialis)", re.I)


# top-level domains the phishing feeds are full of. A ranked site on one of these is more often shady than not, so they
# are left out of the "legitimate" pool to keep the label clean (this also makes `suspecious_tld` an easy signal; say so).
_ABUSED_TLDS = {"top", "xyz", "icu", "work", "link", "party", "review", "tk", "ml", "ga", "cf", "gq", "buzz", "cyou",
                "sbs", "cfd", "click", "rest", "monster", "bid", "loan", "win", "men", "stream", "download", "racing",
                "accountant", "date", "faith", "science", "cricket", "zip", "mov"}


def newest_tranco_file() -> str:
    files = sorted(glob.glob(os.path.join(TRANCO_DIR, "tranco_top100k_*.csv")))
    if not files:
        raise SystemExit(f"No Tranco list in {TRANCO_DIR}: download https://tranco-list.eu/top-1m.csv.zip first.")
    return files[-1]


def read_tranco(path: str, limit: int = 100_000) -> List[Tuple[int, str]]:
    """(rank, domain) pairs from a Tranco CSV; a header line is skipped; the first rank of a domain wins."""
    out, seen = [], set()
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or not row[0].strip().isdigit():
                continue
            rank, dom = int(row[0]), row[1].strip().lower()
            if rank > limit or dom in seen or not re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", dom):
                continue
            seen.add(dom)
            out.append((rank, dom))
    return out


def usable(dom: str) -> bool:
    """A domain with a homepage worth loading: not infrastructure, not a free-hosting platform's root."""
    if _INFRA.search(dom) or _UNWANTED.search(dom) or dom.rsplit(".", 1)[-1] in _ABUSED_TLDS:
        return False
    info = domains.classify_hosting(dom, "/")
    if info.free_hosting_subdomain or info.platform_path_hosting:
        return False
    # the bare platform root itself (pages.dev, github.io, blogspot.com...)
    return dom not in {s for s in _platform_roots()}


_ROOTS: Optional[set] = None


def _platform_roots() -> set:
    global _ROOTS
    if _ROOTS is None:
        roots = set()
        try:
            for line in open(domains.HOSTING_FILE, encoding="utf-8"):
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    roots.add(line.split("/")[0].lstrip("*."))
        except OSError:
            pass
        _ROOTS = roots
    return _ROOTS


def sample_pool(rows: List[Tuple[int, str]], n: int, seed: int,
                bands: List[Tuple[int, int, float]] = BANDS) -> List[Tuple[int, str, str]]:
    """n (rank, domain, band) triples spread over the rank bands by their weights (a short band gives its share to
    the others), without repeats, in a fixed random order."""
    rng = random.Random(seed)
    pools: List[Tuple[str, List[Tuple[int, str]]]] = []
    for lo, hi, _w in bands:
        items = [(r, d) for r, d in rows if lo <= r <= hi and usable(d)]
        rng.shuffle(items)
        pools.append((f"{lo}-{hi}", items))
    take = [min(len(items), int(round(n * w))) for (_, items), (_, _, w) in zip(pools, bands)]
    short = n - sum(take)
    for i, (_, items) in enumerate(pools):                    # fill a shortfall from the bands that still have domains
        extra = min(max(0, short), len(items) - take[i])
        take[i] += extra
        short -= extra
    out = [(r, d, name) for (name, items), k in zip(pools, take) for r, d in items[:k]]
    rng.shuffle(out)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tranco", help="a Tranco rank,domain CSV (default: the newest in " + TRANCO_DIR + ")")
    ap.add_argument("--n", type=int, default=4000, help="how many links (default 4000)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=OUT_TXT)
    a = ap.parse_args(argv)

    path = a.tranco or newest_tranco_file()
    rows = read_tranco(path)
    pool = sample_pool(rows, a.n, a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Legitimate-site pool sampled from {os.path.basename(path)} (Tranco), seed {a.seed}, "
                 f"{len(pool)} links. Defanged. Made by scripts/build_legit_pool.py\n")
        for _rank, dom, _band in pool:
            fh.write(safety.defang_url(f"https://{dom}/") + "\n")
    csv_path = os.path.splitext(a.out)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "rank", "band"])
        for rank, dom, band in pool:
            w.writerow([dom, rank, band])
    by_band: Dict[str, int] = {}
    for _r, _d, band in pool:
        by_band[band] = by_band.get(band, 0) + 1
    print(f"Tranco file: {path} ({len(rows)} usable rows read)")
    print(f"pool: {len(pool)} links by rank band {by_band}")
    print(f"wrote {a.out}\n      {csv_path}")
    if len(pool) < a.n:
        print(f"  note: only {len(pool)} of the {a.n} asked for were available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
