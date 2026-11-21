r"""Build the single final training dataset: the website CSV + ISDA, in the website's 50-column layout (+ source, source_id).

    .venv\Scripts\python.exe scripts/build_final_dataset.py --site <site_dataset.csv> --isda <isda_features_labeled.csv>
                              --out <final.csv> [--synthetic <synthetic_smishing_v3.csv>] [--legit-bulk <combined_messages.csv>]

What it does (every count is printed and written to <out>.summary.json):
  * website rows: the site's spreadsheet guard (a leading ' on text that starts with = + - @, and on a missing value "-1") is removed;
    the test row (the site's own address) and messages that are only a link are dropped;
  * ISDA rows: the 310 rows labelled only "Spam" (ads, job offers: not confirmed smishing) are left out; the rest become the same 50
    columns (language by the project's word-list detector, fetch_status by the website's own rule, link domain / template id /
    shortener flag by the website's own helpers). The ISDA features are used as they are (same lab extractor);
  * the same message (case and spacing ignored) is kept once; a message that appears with BOTH labels is dropped everywhere;
  * missing values are -1, never blank.
`source` says where a row came from (isda, web_group, web_synthetic, web_legit_lab) so the group can split or filter by it.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from webapp import dataset                                   # noqa: E402
from lejet import urlnorm                                    # noqa: E402

MISSING = dataset.MISSING
COLUMNS = dataset.COLUMNS + ["source", "source_id", "dup_group"]
NEAR_DUP_SIMILARITY = 0.90          # character 3-5-gram cosine similarity of the text with links and digits blanked
PRIORITY = {"web_group": 0, "isda": 1, "web_legit_lab": 2, "web_synthetic": 3}      # which copy of a repeated message is kept
GUARD = re.compile(r"^'(?=[=+\-@\s])")
LINK_LIKE = re.compile(r"(https?://\S+|\b[\w.-]+\.(?:com|ph|cc|ly|link|net|co|io|me|gov|onelink\.me)\b\S*)", re.I)
SITE_HOST = "lejet-x7k2q.japaneast.cloudapp.azure.com"
NO_ANSWER = ("dns_", "connect_", "tls_", "read_timeout")      # the page never answered: no final address


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("\ufeff", "")).strip().lower()


def unguard(t: str) -> str:
    """the spreadsheet guard (a leading ') and stray byte-order marks are not part of the message"""
    return GUARD.sub("", t.replace("﻿", ""))


def letters_outside_links(t: str) -> int:
    return len(re.sub(r"[^A-Za-z]", "", LINK_LIKE.sub("", t)))


def near_duplicate_groups(texts: list) -> list:
    """A group number per message; messages whose wording is at least NEAR_DUP_SIMILARITY alike (links and digits ignored,
    so 'PHP 789.00 ... 09****1057' and 'PHP 789.OO ... 09****9741' match) share a number, and chains of such pairs join into
    one group. A message with no near-copy is a group of its own. Split train / test BY THIS COLUMN, never row by row."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import connected_components

    def blank(t: str) -> str:
        t = LINK_LIKE.sub(" ", norm_text(t))
        return re.sub(r"\s+", " ", re.sub(r"\d+", "0", t)).strip()

    x = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit_transform([blank(t) for t in texts])
    graph = NearestNeighbors(radius=1 - NEAR_DUP_SIMILARITY, metric="cosine", algorithm="brute").fit(x).radius_neighbors_graph(x, mode="connectivity")
    return list(connected_components(graph, directed=False)[1])


def host_of(url: str) -> str:
    u = url if "://" in url else "http://" + url
    try:
        return (urlparse(u).hostname or "").lower()
    except ValueError:                       # a malformed link such as "http://[abc": no host
        return ""


def reg_domain(url: str) -> str:
    h = host_of(url)
    return (urlnorm.registrable_domain(h) or h) if h else ""


def language_of(text: str) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lang_detect                                       # noqa: E402  (scripts/lang_detect.py)
    return {"english": "en", "tagalog": "tl", "taglish": "taglish"}[lang_detect.classify3(text)]


def clean_site(site: pd.DataFrame, synthetic: set, bulk: set, stats) -> list:
    out = []
    for _, r in site.iterrows():
        row = {c: r[c] for c in dataset.COLUMNS}
        for c in ("sms_text", "link", "final_url", "link_domain"):
            v = row[c]
            if v == "'-1":
                row[c] = str(MISSING)
            elif GUARD.match(v):
                row[c] = v[1:]
        row["sms_text"] = text = unguard(row["sms_text"])
        if host_of(row["link"]) == SITE_HOST or SITE_HOST in text:
            stats["dropped: test row (the site's own address)"] += 1
            continue
        if row["link"] != str(MISSING) and letters_outside_links(text) < 3:
            stats["dropped: message is only a link"] += 1
            continue
        row["source"] = "web_synthetic" if text in synthetic else ("web_legit_lab" if text in bulk else "web_group")
        row["source_id"] = r["message_id"]
        out.append(row)
    return out


def isda_row(r: pd.Series, language: str) -> dict:
    link = r["link"] if r["has_link"] == "1" and r["link"] else ""
    row = {c: MISSING for c in dataset.COLUMNS}
    row.update(sms_text=unguard(r["message"]).strip(), label=r["label"], language=language,
               template_id=dataset.template_id(unguard(r["message"]).strip()), link_shortened=0, page_fetched=0)
    if link:
        status = r["fetch_status"]
        reason = "none" if status == "live" else status.split(":", 1)[-1]
        answered = status == "live" or not reason.startswith(NO_ANSWER)
        final = r["final_url"] if answered and r["final_url"] else ""
        differs = bool(final) and (reg_domain(final) or host_of(final)) != (reg_domain(link) or host_of(link))
        row.update(link=link, final_url=final or MISSING, link_domain=reg_domain(final or link) or MISSING,
                   link_shortened=dataset.is_shortened(link), page_fetched=int(r["page_fetched"]),
                   fetch_status=dataset.fetch_status(reason, differs))
        for f in dataset.FEATURES_37:
            row[f] = r[f]
    row["source"] = "isda"
    row["source_id"] = r["message_id"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--isda", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", default=r"D:\Downloads\synthetic_smishing_messages_v3.csv")
    ap.add_argument("--legit-bulk", default=r"D:\Downloads\combined_messages.csv")
    a = ap.parse_args()
    stats: collections.Counter = collections.Counter()

    synthetic = set(pd.read_csv(a.synthetic, keep_default_na=False, dtype=str).message_text)
    comb = pd.read_csv(a.legit_bulk, keep_default_na=False, dtype=str)
    bulk = set(comb[comb.label == "legitimate"].message)
    site = pd.read_csv(a.site, keep_default_na=False, dtype=str)
    rows = clean_site(site, synthetic, bulk, stats)
    stats["website rows kept"] = len(rows)

    isda = pd.read_csv(a.isda, keep_default_na=False, dtype=str)
    unconfirmed = isda.label == ""
    stats["dropped: ISDA 'Spam' rows not confirmed as smishing"] = int(unconfirmed.sum())
    isda = isda[~unconfirmed]
    n0 = len(isda)
    isda = isda[~isda.message.map(norm_text).duplicated()]
    stats["dropped: ISDA repeated messages"] = n0 - len(isda)
    for _, r in isda.iterrows():
        row = isda_row(r, language_of(r["message"]))
        if row["link"] != MISSING and letters_outside_links(row["sms_text"]) < 3:
            stats["dropped: message is only a link"] += 1
            continue
        rows.append(row)
    stats["rows before the cross-source duplicate check"] = len(rows)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["_k"] = df.sms_text.map(norm_text)
    both = df.groupby("_k").label.nunique()
    conflict = df._k.isin(both[both > 1].index)
    stats["dropped: same message with both labels"] = int(conflict.sum())
    df = df[~conflict]
    df = df.assign(_p=df.source.map(PRIORITY)).sort_values(["_p"], kind="stable")
    n1 = len(df)
    df = df[~df._k.duplicated()]
    stats["dropped: same message in more than one place"] = n1 - len(df)
    # near-duplicate groups: wording that is 90% alike once links and digits are ignored belongs to one group
    group = near_duplicate_groups(df.sms_text.tolist())
    df["dup_group"] = group
    mixed = df.groupby("dup_group").label.nunique()
    near_conflict = df.dup_group.isin(mixed[mixed > 1].index)
    stats["dropped: near-identical wording with both labels"] = int(near_conflict.sum())
    df = df[~near_conflict]
    df["_l"] = (df.label != "smishing").astype(int)                 # smishing on top, legitimate at the bottom
    df = df.sort_values(["_l", "_p"], kind="stable").drop(columns=["_k", "_p", "_l"]).reset_index(drop=True)
    df["dup_group"] = df.dup_group.map({g: f"g{n:05d}" for n, g in enumerate(df.dup_group.drop_duplicates(), 1)})
    sizes = df.groupby("dup_group").size()
    stats["near-duplicate groups with 2 or more rows"] = int((sizes > 1).sum())
    stats["rows inside those groups"] = int(sizes[sizes > 1].sum())
    df["message_id"] = [f"lj_{n:06d}" for n in range(1, len(df) + 1)]       # new ids; the old ones stay in source_id

    for c in COLUMNS:
        if df[c].astype(str).str.strip().eq("").any():
            raise SystemExit(f"blank cell in column {c}")
    df.to_csv(a.out, index=False, encoding="utf-8", lineterminator="\n")

    summary = {"built": datetime.datetime.now().isoformat(timespec="seconds"), "rows": len(df), "columns": len(df.columns),
               "label": df.label.value_counts().to_dict(), "language": df.language.value_counts().to_dict(),
               "source_by_label": {f"{s}|{l}": int(n) for (s, l), n in df.groupby(["source", "label"]).size().items()},
               "fetch_status_by_label": {f"{s}|{l}": int(n) for (s, l), n in df.groupby(["fetch_status", "label"]).size().items()},
               "steps": dict(stats)}
    Path(str(a.out) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
