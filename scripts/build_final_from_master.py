r"""Build the final training dataset FROM THE DATA LEAD'S LIST (the messages and labels are the lead's, nothing else is added or removed).

    .venv\Scripts\python.exe scripts/build_final_from_master.py --master "<final_combined_dataset.csv>" --site <site csv> --isda <isda csv>
                              --out data/final/lejet_final_dataset.csv [--urls-out <file>]

The lead's file has two columns (SMS_Text, Label). For every message in it:
  * it is found in the pool of rows that already have features (the website CSV and ALL of ISDA, nothing dropped), and its features,
    link and page result are taken from there; the LABEL is always the lead's;
  * if it is not in the pool, it is a NEW message: no link -> has_url 0 and every feature -1; a link -> its features come from the lab's
    cleaned records (D:\LeJetData\clean\*.clean.jsonl, + page_rank files). A link the lab has not seen yet is listed in --urls-out
    (run the lab on that file, then run this script again). The script refuses to build while any link is still missing.
Only the lead's file decides which messages exist: repeated messages (same text) are kept once; a text the lead gave BOTH labels is left out
(it cannot be both) and is listed in the summary, unless data/final/label_decisions.csv decides its label (then it is kept once with that label). Text damaged by a wrong encoding (a peso sign shown as "â±") is repaired.
Output: ONE file with all 56 columns (<out>) and <out>.summary.json. The training code reads only the 46 columns in build_final_dataset.TRAIN_COLUMNS, by name.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import build_final_dataset as B                       # noqa: E402
import financial_category                              # noqa: E402
import make_push_file, add_full_features              # noqa: E402
from webapp import dataset                            # noqa: E402
from lejet import urlnorm                             # noqa: E402

refang = make_push_file.refang
MISSING = dataset.MISSING
CLEAN = Path(r"D:\LeJetData\clean")
BLOCK_STATUS = (401, 403, 406, 429, 503)


def repair(text: str) -> str:
    """Undo UTF-8 text that was read as Windows-1252 or Latin-1 (a peso sign appears as 'â' plus an odd character); unchanged when it was fine."""
    raw = bytearray()
    for ch in text:
        try:
            raw += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) > 0xFF:
                return text                              # a real character outside both code pages: the text was fine
            raw.append(ord(ch))                          # a Latin-1 control character that Windows-1252 leaves undefined
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return text


def key_of(text: str) -> str:
    return B.norm_text(B.unguard(repair(text)))


def load_lab():
    best, ranks = {}, {}
    for p in glob.glob(str(CLEAN / "*.clean.jsonl")):
        if "_all" in os.path.basename(p):
            continue
        for line in open(p, encoding="ascii"):
            if line.strip():
                r = json.loads(line)
                if r["url_id"] not in best or r["fetch_ts"] >= best[r["url_id"]]["fetch_ts"]:
                    best[r["url_id"]] = r
    for p in glob.glob(str(CLEAN / "*.model_input.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("page_rank_ok") == "1":
                ranks[r["url_id"]] = int(r["page_rank"])
    return best, ranks


def lab_of(rec):
    return {"final_url": refang(rec.get("final_url") or "") or -1, "final_status": rec.get("final_status", -1),
            "final_reg_domain": refang(rec.get("final_reg_domain") or ""), "reg_domain": refang(rec.get("reg_domain") or ""),
            "page_fetched": rec["page_fetched"], "fail_reason": rec["fail_reason"], "final_differs": rec.get("final_differs")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--isda", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--urls-out", default=None, help="write the links the lab has not inspected yet to this file and stop")
    ap.add_argument("--decisions", default=str(ROOT / "data" / "final" / "label_decisions.csv"),
                    help="messages whose label we decided ourselves (see label_decisions.csv); an empty value means: use the lead's labels only")
    ap.add_argument("--synthetic", default=r"D:\Downloads\synthetic_smishing_messages_v3.csv")
    ap.add_argument("--legit-bulk", default=r"D:\Downloads\combined_messages.csv")
    a = ap.parse_args()
    stats: collections.Counter = collections.Counter()

    # ---- the lead's list
    m = pd.read_csv(a.master, keep_default_na=False, dtype=str)
    assert list(m.columns) == ["SMS_Text", "Label"], m.columns
    stats["lead's file: rows"] = len(m)
    m["label"] = m.Label.str.strip().str.lower()
    assert set(m.label) <= {"smishing", "legitimate"}, set(m.label)
    m["text"] = m.SMS_Text.map(lambda t: B.unguard(repair(t)).strip())
    m["key"] = m.SMS_Text.map(key_of)
    m["lead_row"] = range(2, len(m) + 2)
    m["lead_label"] = m.label
    if a.decisions and Path(a.decisions).exists():                # labels decided by us for messages the lead labelled inconsistently
        dec = pd.read_csv(a.decisions, keep_default_na=False, dtype=str)
        decided = {key_of(t): l for t, l in zip(dec.sms_text, dec.final_label)}
        assert set(decided.values()) <= {"smishing", "legitimate"}
        m["label"] = [decided.get(k, l) for k, l in zip(m.key, m.label)]
        stats["labels decided by us instead of the lead's (label_decisions.csv)"] = int((m.label != m.lead_label).sum())                       # the row number in the lead's file (header = row 1)
    stats["lead's file: texts repaired from wrong encoding"] = int((m.text != m.SMS_Text.map(lambda t: B.unguard(t).strip())).sum())
    both = m.groupby("key").label.nunique()
    conflict = sorted(both[both > 1].index)
    stats["lead's file: texts with BOTH labels (left out)"] = len(conflict)
    m = m[~m.key.isin(conflict)]
    n1 = len(m)
    m = m.drop_duplicates("key")
    stats["lead's file: repeated messages (kept once)"] = n1 - len(m)
    stats["messages in the final dataset"] = len(m)

    # ---- the pool: every row that already has features, nothing dropped
    synthetic = set(pd.read_csv(a.synthetic, keep_default_na=False, dtype=str).message_text)
    comb = pd.read_csv(a.legit_bulk, keep_default_na=False, dtype=str)
    bulk = set(comb[comb.label == "legitimate"].message)
    pool_rows = B.clean_site(pd.read_csv(a.site, keep_default_na=False, dtype=str), synthetic, bulk, collections.Counter(), drop_junk=False)
    isda = pd.read_csv(a.isda, keep_default_na=False, dtype=str)
    wanted = set(m.key)
    for _, r in isda.iterrows():
        if key_of(r["message"]) in wanted:                     # (language is only computed for the rows that are needed)
            pool_rows.append(B.isda_row(r, B.language_of(B.unguard(r["message"]))))
    pool = {}
    for row in sorted(pool_rows, key=lambda x: B.PRIORITY[x["source"]]):
        pool.setdefault(key_of(row["sms_text"]), row)

    best, ranks = load_lab()
    now = datetime.datetime.now(datetime.timezone.utc)
    rows, missing_links = [], []
    for _, r in m.iterrows():
        pooled = pool.get(r.key)
        if pooled is not None:
            row = dict(pooled)
            if row["label"] not in ("", r.label):
                stats["label taken from the lead, different from ours or ISDA's"] += 1
            if row["label"] == "":
                stats["label given by the lead to an ISDA row that had none"] += 1
            row.update(label=r.label, sms_text=r.text, template_id=dataset.template_id(r.text))
            stats[f"found with features: {row['source']}"] += 1
        else:
            found = dataset.first_link(r.text)
            lang = B.language_of(r.text)                      # already "en", "tl" or "taglish"
            if found is None:
                row = {c: MISSING for c in dataset.COLUMNS}
                row.update(sms_text=r.text, label=r.label, language=lang, link_shortened=0, template_id=dataset.template_id(r.text), page_fetched=0)
                stats["NEW message, no link"] += 1
            else:
                raw, norm = found
                uid = urlnorm.compute_url_id(refang(norm))
                rec = best.get(uid)
                if rec is None:
                    missing_links.append(raw)
                    continue
                full = add_full_features.complete(rec, {uid: ranks[uid]} if uid in ranks else None)
                row = dataset.build_row("", r.text, r.label, lang, raw, lab_of(rec), full, now)
                if row["fetch_status"] == "dead" and rec["fail_reason"] in ("http_4xx", "http_5xx") and rec.get("final_status") in BLOCK_STATUS:
                    row["fetch_status"] = "blocked"
                stats["NEW message, with link (features from the lab)"] += 1
            row["source"] = "data_lead"
            row["source_id"] = f"lead_row{r.lead_row}"
        rows.append(row)

    if missing_links:
        if a.urls_out:
            Path(a.urls_out).write_text("\n".join(dict.fromkeys(missing_links)) + "\n", encoding="utf-8")
            print(f"{len(set(missing_links))} link(s) of NEW messages have no lab record yet: written to {a.urls_out}. "
                  f"Run the lab on that file, then run this script again.")
            return 0
        print(f"{len(set(missing_links))} link(s) of NEW messages have no lab record yet; run with --urls-out to list them.", file=sys.stderr)
        return 2

    # ---- assemble
    df = pd.DataFrame(rows, columns=B.COLUMNS)
    df["dup_group"] = B.near_duplicate_groups(df.sms_text.tolist())
    mixed = df.groupby("dup_group").label.nunique()
    stats["near-duplicate groups that carry both labels (kept: they are the lead's)"] = int((mixed > 1).sum())
    df["_l"] = (df.label != "smishing").astype(int)
    df["_p"] = df.source.map({**B.PRIORITY, "data_lead": 4})
    df = df.sort_values(["_l", "_p"], kind="stable").drop(columns=["_l", "_p"]).reset_index(drop=True)
    df["dup_group"] = df.dup_group.map({g: f"g{n:05d}" for n, g in enumerate(df.dup_group.drop_duplicates(), 1)})
    sizes = df.groupby("dup_group").size()
    stats["near-duplicate groups with 2 or more rows"] = int((sizes > 1).sum())
    stats["rows inside those groups"] = int(sizes[sizes > 1].sum())
    df["message_id"] = [f"lj_{n:06d}" for n in range(1, len(df) + 1)]
    df["has_url"] = (df.link.astype(str) != str(MISSING)).astype(int)
    _cats = [financial_category.categorize(t) for t in df.sms_text]            # DRAFT by rules, not a human annotation
    df["category"] = [c[0] for c in _cats]
    df["categories_all"] = [c[1] or "none" for c in _cats]        # never blank
    for c in ["page_fetched", "link_shortened"] + dataset.FEATURES_37:          # (rows from the site CSV carry text numbers, ISDA rows real ones)
        df[c] = pd.to_numeric(df[c])
    for c in B.FULL_COLUMNS:
        if df[c].astype(str).str.strip().eq("").any():
            raise SystemExit(f"blank cell in column {c}")
    out = Path(a.out)
    df[B.FULL_COLUMNS].to_csv(out, index=False, encoding="utf-8", lineterminator="\n")      # ONE file, all 56 columns
    summary = {"built": datetime.datetime.now().isoformat(timespec="seconds"), "from": ("the data lead's list; the lead's labels" + (", except those decided in label_decisions.csv" if stats.get("labels decided by us instead of the lead's (label_decisions.csv)") else " (no label changed)")),
               "rows": len(df), "columns": len(B.FULL_COLUMNS), "columns_read_by_training": B.TRAIN_COLUMNS, "label": df.label.value_counts().to_dict(),
               "has_url": df.has_url.value_counts().to_dict(), "page_fetched": df.page_fetched.value_counts().to_dict(),
               "language": df.language.value_counts().to_dict(),
               "category_draft_by_label": {f"{c}|{l}": int(n) for (c, l), n in df.groupby(["category", "label"]).size().items()},
               "source_by_label": {f"{s}|{l}": int(n) for (s, l), n in df.groupby(["source", "label"]).size().items()},
               "conflicting_texts_left_out": conflict, "steps": dict(stats)}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "conflicting_texts_left_out"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
