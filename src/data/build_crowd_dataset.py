"""Build the C-prune study dataset in the 57-column layout of docs/c_prune_architecture.md section 4.3.

Two sources, one output layout:

    --pilot   data/processed/fresh_lean.csv (SYNTHETIC SMS text around real URLs with their 37 features)
              -> data/processed/crowd_pilot_synthetic.csv (+ .summary.json)
              A PIPELINE TEST ONLY: it lets Phases 2-7 be built and run before crowd data exists.
              Its scores are never study results (section 3.2: "Synthetic text: none").
    (crowd)   submissions_<date>.csv + the lab's <batch>.clean.jsonl -> crowd_smishing_v1.csv.
              Not built yet: it needs the first real inspected batch to fix the join.

Both apply the row rules of section 4.4 that the source can be checked against:
    * every row has a link (the dataset is URL messages only);
    * exact duplicates (same sms_text and same link): keep the first;
    * at most 20 rows of one link_domain per class: the rest are set aside (kept in a side file, not deleted);
    * missing values are -1, never blank, NaN or 0.

Usage:
    python -m src.data.build_crowd_dataset --pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.features.link_features import SHORTENERS, extract_links
from src.unified.prepare_web_37 import FEATURES_37

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PILOT_IN = PROCESSED / "fresh_lean.csv"
PILOT_OUT = PROCESSED / "crowd_pilot_synthetic.csv"

MESSAGE_COLUMNS = ["message_id", "sms_text", "label", "category", "language", "sender_type", "date_received",
                   "date_submitted", "label_source", "label_verified_by", "consent"]
LINK_COLUMNS = ["link", "final_url", "link_domain", "template_id", "link_shortened"]
INSPECTION_COLUMNS = ["inspected_at", "hours_to_inspect", "page_fetched", "fetch_status"]
COLUMNS = MESSAGE_COLUMNS + LINK_COLUMNS + INSPECTION_COLUMNS + list(FEATURES_37)
assert len(COLUMNS) == 57

MAX_PER_DOMAIN_PER_CLASS = 20
MISSING = -1

# Brand -> category, for the brands the synthetic generator uses (src/unified/prepare_lean_synthetic.BRANDS).
BRAND_CATEGORY = {
    "gcash": "e-wallet", "paymaya": "e-wallet", "maya": "e-wallet", "coins.ph": "e-wallet",
    "bdo": "bank", "bpi": "bank", "metrobank": "bank", "unionbank": "bank", "landbank": "bank",
    "security bank": "bank", "gotyme": "bank", "seabank": "bank",
    "home credit": "loan", "atome": "loan",
}
# Tagalog function words; any of them marks a message as Taglish (the generator writes no pure Tagalog).
_TAGALOG = re.compile(r"\b(ang|ng|sa|mo|ka|na|para|mag|nag|wala|dito|agad|salamat|lang)\b|\b(mag|nag|ma|i)-", re.I)


def category_for(text: str) -> str:
    low = text.lower()
    # Longest brand first, so "paymaya" wins over "maya".
    for brand in sorted(BRAND_CATEGORY, key=len, reverse=True):
        if brand in low:
            return BRAND_CATEGORY[brand]
    return "other_financial"


def language_for(text: str) -> str:
    return "taglish" if _TAGALOG.search(text) else "en"


def template_key(text: str) -> str:
    """The message with its links and numbers masked, lower-cased: near-identical texts share a key."""
    for link in reversed(extract_links(text)):
        text = text[:link["start"]] + " <url> " + text[link["end"]:]
    text = re.sub(r"\d[\d,.]*", "0", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def template_ids(texts: pd.Series) -> pd.Series:
    keys = texts.map(template_key)
    return keys.map(lambda k: "t" + hashlib.sha256(k.encode("utf-8")).hexdigest()[:8])


def is_shortened(link: str) -> int:
    links = extract_links(link)
    host = links[0]["host"] if links else ""
    return int(host.removeprefix("www.") in SHORTENERS)


def apply_row_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Section 4.4 rules. Returns (kept rows, rows set aside by the domain cap, counts)."""
    counts = {"input_rows": len(df)}
    has_link = df["link"].astype(str).str.strip().ne("") & df["sms_text"].map(lambda t: bool(extract_links(str(t))))
    counts["dropped_no_link"] = int((~has_link).sum())
    df = df[has_link]
    dup = df.duplicated(["sms_text", "link"], keep="first")
    counts["dropped_exact_duplicate"] = int(dup.sum())
    df = df[~dup]
    # Rows are already in submission order, so "the first 20" are the earliest.
    over = df.groupby(["link_domain", "label"]).cumcount() >= MAX_PER_DOMAIN_PER_CLASS
    counts["set_aside_domain_cap"] = int(over.sum())
    return df[~over].reset_index(drop=True), df[over].reset_index(drop=True), counts


def check(df: pd.DataFrame) -> None:
    assert list(df.columns) == COLUMNS, "columns out of order"
    assert not df.isna().any().any(), "blank values: missing must be -1"
    assert df["message_id"].is_unique
    assert set(df["label"]) <= {"smishing", "legitimate"}
    assert (df["consent"] == 1).all()
    assert (df.groupby(["link_domain", "label"]).size() <= MAX_PER_DOMAIN_PER_CLASS).all()


def summarize(df: pd.DataFrame, counts: Dict[str, int], source: str) -> dict:
    label_counts = df["label"].value_counts().to_dict()
    return {
        "source": source,
        "rows": len(df),
        "row_rules": counts,
        "label": label_counts,
        "legitimate_share": round(label_counts.get("legitimate", 0) / max(len(df), 1), 4),
        "category": df["category"].value_counts().to_dict(),
        "language": df["language"].value_counts().to_dict(),
        "domains": int(df["link_domain"].nunique()),
        "templates": int(df["template_id"].nunique()),
        "max_rows_per_domain_per_class": int(df.groupby(["link_domain", "label"]).size().max()),
        "page_fetched": df["page_fetched"].value_counts().to_dict(),
        "link_shortened": int(df["link_shortened"].sum()),
    }


def build_pilot(src: Path = PILOT_IN) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(src)
    out = pd.DataFrame({
        "message_id": [f"pilot_{i:06d}" for i in range(1, len(raw) + 1)],
        "sms_text": raw["sms_text"],
        "label": raw["label"],
        "category": raw["sms_text"].map(category_for),
        "language": raw["sms_text"].map(language_for),
        # The synthetic set has no sender, dates, reviewer or inspection time.
        "sender_type": "unknown",
        "date_received": MISSING,
        "date_submitted": MISSING,
        "label_source": MISSING,
        "label_verified_by": MISSING,
        "consent": 1,
        "link": raw["link"],
        "final_url": MISSING,
        "link_domain": raw["link_domain"],
        "template_id": template_ids(raw["sms_text"]),
        "link_shortened": raw["link"].map(is_shortened),
        "inspected_at": MISSING,
        "hours_to_inspect": MISSING,
        # Every row of fresh_lean has its 13 page features, so every page was fetched.
        "page_fetched": 1,
        "fetch_status": "live",
    })
    out = pd.concat([out, raw[FEATURES_37]], axis=1)[COLUMNS]
    kept, set_aside, counts = apply_row_rules(out)
    check(kept)
    return kept, set_aside, summarize(kept, counts, "SYNTHETIC PILOT (fresh_lean.csv): pipeline test only, not study results")


def write(kept: pd.DataFrame, set_aside: pd.DataFrame, summary: dict, out: Path) -> None:
    kept.to_csv(out, index=False, encoding="utf-8")
    set_aside.to_csv(out.with_suffix(".set_aside.csv"), index=False, encoding="utf-8")
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="build the synthetic pilot from fresh_lean.csv")
    a = ap.parse_args(argv)
    if not a.pilot:
        raise SystemExit("Only --pilot exists so far: the crowd build is written once the first real batch is inspected.")
    kept, set_aside, summary = build_pilot()
    write(kept, set_aside, summary, PILOT_OUT)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {PILOT_OUT} ({len(kept)} rows, {len(COLUMNS)} columns); {len(set_aside)} rows set aside")


if __name__ == "__main__":
    main()
