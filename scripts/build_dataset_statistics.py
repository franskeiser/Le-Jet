r"""Dataset_Statistics.xlsx: the numbers of Chapter 4.1 and the guide's Step 1 table, from the ONE final dataset file.

    .venv\Scripts\python.exe scripts/build_dataset_statistics.py [--data data/final/lejet_final_dataset.csv] [--out data/final/Dataset_Statistics.xlsx]

Sheets: Dataset information, By source, By language, Link result by label, Near-duplicates, Checks, Categories (draft),
Smishing without category, Label decisions, Freeze candidate, Notes. Every number is computed from the file (none is typed in).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
from pathlib import Path

import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from webapp import dataset                                   # noqa: E402  (the 37 feature names)
ORDER = ["banking", "e-wallet", "lending", "payments_billing", "investment_crypto", "gambling", "none"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "final" / "lejet_final_dataset.csv"))
    ap.add_argument("--decisions", default=str(ROOT / "data" / "final" / "label_decisions.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "final" / "Dataset_Statistics.xlsx"))
    a = ap.parse_args()
    t = pd.read_csv(a.data, keep_default_na=False)
    sha = hashlib.sha256(open(a.data, "rb").read()).hexdigest()
    sm = t[t.label == "smishing"]
    n_none, n_none_syn = int((sm.category == "none").sum()), int(((sm.category == "none") & (sm.source == "web_synthetic")).sum())
    fetched = t[t.page_fetched == 1]

    checks = {"unique message_id": bool(t.message_id.is_unique),
              "every row has a label (smishing/legitimate)": bool(t.label.isin(["smishing", "legitimate"]).all()),
              "no blank cells": bool((t.astype(str) == "").sum().sum() == 0),
              "no exact repeated message": bool(t.sms_text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip().is_unique),
              "has_url=0 rows: every URL / page feature is -1": bool((t[t.has_url == 0][dataset.FEATURES_37] == -1).all().all()),
              "label_verified_by filled (second-person check)": bool((t.label_verified_by != -1).all())}

    def row(name, df):
        return {"Item": name, "Total": len(df), "Smishing": int((df.label == "smishing").sum()), "Legitimate": int((df.label == "legitimate").sum())}
    info = pd.DataFrame([row("Total retained messages", t), row("Messages with URLs", t[t.has_url == 1]), row("Messages without URLs", t[t.has_url == 0]),
                         row("Accessible webpages", t[t.page_fetched == 1]),
                         row("Inaccessible webpages (URL present, page not loaded)", t[(t.has_url == 1) & (t.page_fetched == 0)])])
    src = pd.crosstab(t.source, t.label).reset_index()
    src["Total"] = src.smishing + src.legitimate
    lang = pd.crosstab(t.language, t.label).reset_index().rename(columns={"language": "language (en=English, tl=Tagalog, taglish=mixed; detector estimate)"})
    fs = pd.crosstab(t.fetch_status.astype(str).replace({"-1": "no link"}), t.label).reset_index().rename(columns={"fetch_status": "link result"})
    g = t.groupby("dup_group").size()
    dup = pd.DataFrame([{"Item": "Groups of near-identical messages (2 or more)", "N": int((g > 1).sum())},
                        {"Item": "Messages inside those groups", "N": int(g[g > 1].sum())}, {"Item": "Largest group", "N": int(g.max())},
                        {"Item": "Groups carrying both labels", "N": int((t.groupby("dup_group").label.nunique() > 1).sum())},
                        {"Item": "Rule", "N": "90% character-3-to-5-gram similarity after links and digits are ignored; kept, not removed; split train/test by dup_group"}])
    chk = pd.DataFrame([{"Check": k, "Result": "PASS" if v else "NOT YET"} for k, v in checks.items()])
    cat = pd.crosstab(t.category, t.label).reindex(ORDER).fillna(0).astype(int)
    cat["Total"] = cat.sum(axis=1)
    cat["% of smishing"] = (cat.smishing / cat.smishing.sum() * 100).round(1)
    cat["% of legitimate"] = (cat.legitimate / cat.legitimate.sum() * 100).round(1)
    cat = cat.reset_index().rename(columns={"category": "category (DRAFT by rules)"})
    nf = pd.crosstab(sm.category == "none", sm.source).rename(index={True: "no financial category", False: "has a financial category"}).reset_index()
    nf = nf.rename(columns={"category": "smishing messages"})
    dec = pd.read_csv(a.decisions, keep_default_na=False) if Path(a.decisions).exists() else pd.DataFrame()
    freeze = pd.DataFrame([{"Item": "File", "Value": f"{Path(a.data).as_posix().split('Le-Jet/')[-1]} ({t.shape[1]} columns, one file)"},
                           {"Item": "SHA-256 (candidate; freeze only after Step 1 items 4-6)", "Value": sha}, {"Item": "Rows", "Value": len(t)},
                           {"Item": "Built", "Value": datetime.date.today().isoformat()},
                           {"Item": "Labels from", "Value": f"the data lead's list; {len(dec)} labels decided by us, see the Label decisions sheet"}])
    notes = pd.DataFrame({"Note": [
        "CATEGORIES ARE A DRAFT assigned by word and brand rules (scripts/financial_category.py), not by people: in a hand reading of 58 random messages about 80-90% of the smishing assignments were plausible; check a sample before using them in the thesis. A message can touch several categories (column categories_all); the sheet counts the main one.",
        f"Smishing with no financial category: {n_none:,} of {len(sm):,} ({n_none_syn:,} of them synthetic). Examples: Microsoft/Google/WhatsApp account lures, job offers, parcel and voucher notices. Chapter 3 says smishing samples must relate to one of the six financial subcategories, so the group must decide how to treat them.",
        "Inclusion and exclusion criteria (guide step 1, item 3): applied by the data lead; confirmed done by the group on 2026-11-21.",
        "Annotation and external-validation results are NOT in this file: not done or not recorded yet (label_verified_by is empty).",
        f"Synthetic messages (source = web_synthetic): {int((t.source == 'web_synthetic').sum()):,}, all smishing.",
        f"For loaded pages the mean page_rank is {fetched[fetched.label == 'legitimate'].page_rank.mean():.1f} (legitimate) vs {fetched[fetched.label == 'smishing'].page_rank.mean():.1f} (smishing); {t[t.label == 'smishing'].has_url.mean():.1%} of smishing has a link vs {t[t.label == 'legitimate'].has_url.mean():.1%} of legitimate: possible shortcuts, report with and without.",
        f"The 'blocked' pages ({int((t.fetch_status == 'blocked').sum())}) exist but refuse automated visitors; they are counted as inaccessible."]})
    out = Path(a.out)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        for name, frame in [("Dataset information", info), ("By source", src), ("By language", lang), ("Link result by label", fs),
                            ("Near-duplicates", dup), ("Checks", chk), ("Categories (draft)", cat), ("Smishing without category", nf),
                            ("Label decisions", dec), ("Freeze candidate", freeze), ("Notes", notes)]:
            frame.to_excel(w, sheet_name=name, index=False)
        for ws in w.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(12, max(len(str(c.value or "")) for c in col) + 2))
    print(info.to_string(index=False))
    print(chk.to_string(index=False))
    print("sha", sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
