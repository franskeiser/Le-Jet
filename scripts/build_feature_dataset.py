#!/usr/bin/env python
r"""
build_feature_dataset.py -- HOST-SIDE: the labelled feature table of our own links, in the public set's format.

After the lab has run the files made by scripts/build_labeled_input.py:

    python scripts/build_feature_dataset.py --balance --lean

With no other arguments it reads the table D:\LeJetData\datasets\feature_input.csv (labels) and EVERY
feat*.clean.jsonl in D:\LeJetData\clean (the lab results of everything run so far). To use other files:
--labels <table> --clean <the .clean.jsonl files>.

It writes D:\LeJetData\datasets\feature_dataset.csv (rebuilt from everything each time) with EXACTLY the columns of
D:\LeJetData\datasets\train-1_73features.csv:

    url, <the 73 exported features, in the public set's order>, status

so the lab's output and the trimmed public set can be stacked, compared or fed to the same code.

How the rows are chosen:
  * the lab results (validated `.clean.jsonl`, so they already passed scripts/validate_output.py) are joined to the
    labels by `url_id`; a link seen in several batches keeps its newest result;
  * only links whose page was fetched are kept (`page_fetched = 1`), so the page columns are real; --all keeps every
    link (the page columns of an unfetched link are -1);
  * --balance then drops random rows of the larger class so both classes have the same number (fixed seed);
  * `page_rank` is -1 unless you pass --page-rank-csv (scripts/lookup_page_rank.py output).

`url` is the REAL link, exactly as the lab fetched it (like the public set). The file therefore contains live phishing
links: never open them on this computer, and do not share the file as it is; the lab VMs are the only place that
visits them. --defanged writes hxxps://a[.]com instead. (The real link is recovered exactly from the label file's
defanged one: the lab's url_id is a hash of it.)

--lean also writes <out>_lean.csv without the columns that carry no information in THIS data: those with one value in
every row, those whose value differs from the most common one in fewer than --min-variation (5%) of the rows (too
rare to learn from), and all but the strongest of every group of near-identical columns (|Spearman| >= 0.95).

Rows that came from a batch run before the 73-column export existed are refused (they have no `features_full`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from lejet import safety, schema                              # noqa: E402
import add_full_features                                      # noqa: E402
import make_push_file                                         # noqa: E402
import build_labeled_input as bli                             # noqa: E402

DEFAULT_OUT = r"D:\LeJetData\datasets"
DEFAULT_LABELS = os.path.join(DEFAULT_OUT, "feature_input.csv")
DEFAULT_CLEAN_DIR = r"D:\LeJetData\clean"
COLUMNS = ["url"] + schema.FULL_FEATURES + ["status"]


def read_labels(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    need = {"url", "status", "url_id"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"{path} is not a feature_input file (needs the columns {sorted(need)})")
    return {r["url_id"]: r for r in rows}


def load_results(paths: List[str]) -> Dict[str, dict]:
    """url_id -> the newest record over all the given clean.jsonl files."""
    best: Dict[str, dict] = {}
    for path in paths:
        for rec in add_full_features.read_clean(path):
            old = best.get(rec["url_id"])
            if old is None or rec["fetch_ts"] >= old["fetch_ts"]:
                best[rec["url_id"]] = rec
    return best


def build_rows(labels: Dict[str, Dict[str, str]], results: Dict[str, dict], page_rank: Optional[Dict[str, int]],
               keep_unfetched: bool, real_urls: bool = True) -> tuple:
    rows, stats = [], Counter()
    for url_id, rec in results.items():
        label = labels.get(url_id)
        if label is None:
            stats["not_in_labels"] += 1
            continue
        if "features_full" not in rec:
            raise SystemExit("a result has no `features_full`: it was made before the 73-column export. "
                             "Re-run that batch on the updated VM.")
        if not rec["page_fetched"] and not keep_unfetched:
            stats["page_not_fetched:" + label["status"]] += 1
            continue
        full = add_full_features.complete(rec, page_rank)
        url = make_push_file.refang(label["url"]) if real_urls else label["url"]
        rows.append([safety.csv_safe(url)] + [full[n] for n in schema.FULL_FEATURES] + [label["status"]])
    stats["no_result_yet"] = len(set(labels) - set(results))
    return rows, stats


ONE_VALUE = "one value in every row"


def lean_columns(rows: List[list], min_variation: float = 0.05) -> tuple:
    """(columns to drop, why) for THIS data: one value in every row, a value that differs from the most common one in
    fewer than `min_variation` of the rows (too rare to learn from), or a near-identical copy of a stronger column."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    df = pd.DataFrame(rows, columns=COLUMNS)
    y = (df["status"] == sorted(set(df["status"]))[-1]).astype(int).values      # 'phishing' sorts after 'legitimate'
    X = df[schema.FULL_FEATURES].astype(float)
    drop: Dict[str, str] = {}
    for c in schema.FULL_FEATURES:
        if X[c].nunique() == 1:
            drop[c] = ONE_VALUE
        elif min_variation > 0 and (1 - X[c].value_counts(normalize=True).iloc[0]) < min_variation:
            drop[c] = f"differs from the most common value in under {min_variation:.0%} of the rows"
    live = [c for c in schema.FULL_FEATURES if c not in drop]
    strength = {c: abs(roc_auc_score(y, X[c]) - 0.5) * 2 for c in live} if len(set(y)) == 2 else {c: 0.0 for c in live}
    corr = X[live].corr(method="spearman").abs().fillna(0)
    parent = {c: c for c in live}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if corr.loc[a, b] >= 0.95:
                parent[find(a)] = find(b)
    groups: Dict[str, List[str]] = {}
    for c in live:
        groups.setdefault(find(c), []).append(c)
    for g in groups.values():
        if len(g) > 1:
            keep = max(g, key=lambda c: (round(strength[c], 6), -schema.FULL_FEATURES.index(c)))
            for c in g:
                if c != keep:
                    drop[c] = f"near-identical to {keep}"
    return drop, strength


def write_csv(path: str, header: List[str], rows: List[list]) -> str:
    """Write a CSV. If the file is open in Excel (Windows refuses to overwrite it), write <name>_NEW_<time>.csv next to
    it and say so, instead of losing the run. Returns the path written."""
    try:
        fh = open(path, "w", encoding="utf-8", newline="")
    except PermissionError:
        alt = os.path.splitext(path)[0] + datetime.now().strftime("_NEW_%H%M%S") + ".csv"
        print(f"  !! {os.path.basename(path)} is open in another program (Excel?); writing {os.path.basename(alt)} instead. "
              f"Close the old one and rename this.")
        path, fh = alt, open(alt, "w", encoding="utf-8", newline="")
    with fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def cap_platforms(rows: List[list], cap: int, seed: int) -> tuple:
    """Keep at most `cap` phishing rows per hosting platform (pages.dev, github.io...) or site. A kit copied hundreds of
    times onto one free platform would otherwise teach the table the platform instead of the phishing. Legitimate
    rows are not touched. Returns (rows, number dropped); cap 0 = off."""
    if cap <= 0:
        return rows, 0
    rng = random.Random(seed)
    groups: Dict[str, List[list]] = {}
    for r in rows:
        if r[-1] == bli.POSITIVE:
            groups.setdefault(bli.group_key(make_push_file.refang(r[0])), []).append(r)
    drop = set()
    for key in sorted(groups):
        items = groups[key]
        if len(items) > cap:
            keep = rng.sample(items, cap)
            drop |= {id(r) for r in items} - {id(r) for r in keep}
    return [r for r in rows if id(r) not in drop], len(drop)


def balance(rows: List[list], seed: int) -> List[list]:
    by: Dict[str, List[list]] = {}
    for r in rows:
        by.setdefault(r[-1], []).append(r)
    if len(by) < 2:
        return rows
    n = min(len(v) for v in by.values())
    rng = random.Random(seed)
    out: List[list] = []
    for label in sorted(by):
        group = by[label][:]
        rng.shuffle(group)
        out.extend(group[:n])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", default=DEFAULT_LABELS, help="the table made by build_labeled_input.py (default: feature_input.csv)")
    ap.add_argument("--clean", nargs="*", help="the validated <batch>.clean.jsonl files (default: every feat*.clean.jsonl in "
                    + DEFAULT_CLEAN_DIR + ", so the table covers everything run so far)")
    ap.add_argument("--out", help=f"default: {DEFAULT_OUT}\\feature_dataset.csv (rebuilt from everything each time)")
    ap.add_argument("--page-rank-csv", help="scripts/lookup_page_rank.py output (adds page_rank)")
    ap.add_argument("--all", action="store_true", help="keep links whose page was not fetched too")
    ap.add_argument("--platform-cap", type=int, default=60,
                    help="keep at most this many phishing rows per hosting platform (default 20; 0 = no cap)")
    ap.add_argument("--target", type=int, default=2500, help="rows wanted per class, for the progress line (default 2500; 0 = off)")
    ap.add_argument("--balance", action="store_true", help="drop random rows of the larger class so both are equal")
    ap.add_argument("--defanged", action="store_true", help="write hxxps://a[.]com instead of the real link")
    ap.add_argument("--lean", action="store_true", help="also write <out>_lean.csv without the uninformative columns")
    ap.add_argument("--min-variation", type=float, default=0.05,
                    help="with --lean: drop columns whose value differs from the most common one in fewer than this "
                         "share of the rows (default 0.05; 0 = only columns with a single value)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)

    labels = read_labels(a.labels)
    clean = a.clean or sorted(p for p in glob.glob(os.path.join(DEFAULT_CLEAN_DIR, "feat*.clean.jsonl"))
                              if "_all" not in os.path.basename(p))
    if not clean:
        raise SystemExit("no lab results found: pass --clean <the .clean.jsonl files>")
    results = load_results(clean)
    page_rank = add_full_features.read_page_rank(a.page_rank_csv) if a.page_rank_csv else None
    rows, stats = build_rows(labels, results, page_rank, a.all, real_urls=not a.defanged)
    rows, capped = cap_platforms(rows, a.platform_cap, a.seed)
    before = Counter(r[-1] for r in rows)
    if a.balance:
        rows = balance(rows, a.seed)
    rows.sort(key=lambda r: (r[-1], r[0]))

    out = a.out or os.path.join(DEFAULT_OUT, "feature_dataset.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    out = write_csv(out, COLUMNS, rows)

    if a.lean:
        drop, _ = lean_columns(rows, a.min_variation)
        keep = [i for i, c in enumerate(COLUMNS) if c not in drop]
        lean_path = write_csv(os.path.splitext(a.out or os.path.join(DEFAULT_OUT, "feature_dataset.csv"))[0] + "_lean.csv",
                              [COLUMNS[i] for i in keep], [[r[i] for i in keep] for r in rows])
        print(f"WROTE {lean_path}: {len(rows)} rows x {len(keep)} columns ({len(drop)} dropped: "
              f"{sum(v == ONE_VALUE for v in drop.values())} constant, "
              f"{sum(v.startswith('differs') for v in drop.values())} almost constant, "
              f"{sum(v.startswith('near-identical') for v in drop.values())} near-identical)")
    print(f"links in the label file: {len(labels)} | lab results read: {len(results)}")
    print(f"not run in the lab yet: {stats['no_result_yet']} | results not in the label file: {stats['not_in_labels']}")
    dropped = {k.split(':', 1)[1]: v for k, v in stats.items() if k.startswith("page_not_fetched:")}
    if dropped:
        print(f"left out because the page was not fetched: {dropped}")
    if capped:
        print(f"left out by the platform cap (at most {a.platform_cap} phishing rows per hosting platform): {capped}")
    print(f"rows with features before balancing: {dict(before)}")
    goal = a.target
    if goal:
        pos, neg = before.get(bli.POSITIVE, 0), before.get(bli.NEGATIVE, 0)
        print(f"PROGRESS toward {goal} per class: phishing {pos} ({100 * pos // goal}%), legitimate {neg} ({100 * neg // goal}%) "
              f"-> a balanced table of {2 * min(pos, neg)} rows now, {2 * goal} when both reach {goal}.")
    print(f"WROTE {out}: {len(rows)} rows x {len(COLUMNS)} columns, status = {dict(Counter(r[-1] for r in rows))}")
    if not a.defanged:
        print("NOTE: the url column holds REAL phishing links. Never open them on this computer; keep the file private.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
