r"""Build the C-prune study dataset in the 57-column layout of docs/c_prune_architecture.md section 4.3.

Two sources, one output layout:

    --pilot   data/processed/fresh_lean.csv (SYNTHETIC SMS text around real URLs with their 37 features)
              -> data/processed/crowd_pilot_synthetic.csv (+ .summary.json)
              A PIPELINE TEST ONLY: it lets Phases 2-7 be built and run before crowd data exists.
              Its scores are never study results (section 3.2: "Synthetic text: none").
    --crowd   data/crowd/raw/submissions_*.xlsx (+ .csv) + the lab's D:\LeJetData\clean\crowd*.clean.jsonl
              (+ the <batch>.model_input.csv page_rank lookups) -> data/processed/crowd_smishing_draft.csv.
              A DRAFT, rebuilt from the raw files every collection day; message ids may still shift.
              When collection ends it is frozen once as crowd_smishing_v1.csv (section 4.2).
              Messages whose link the lab has not inspected yet, or whose label nobody has checked
              (label_verified_by empty), are left out and counted in the summary.

Both apply the row rules of section 4.4 that the source can be checked against:
    * every row has a link (the dataset is URL messages only);
    * exact duplicates (same sms_text and same link): keep the first;
    * at most 20 rows of one link_domain per class: the rest are set aside (kept in a side file, not deleted);
    * missing values are -1, never blank, NaN or 0.

Usage:
    python -m src.data.build_crowd_dataset --pilot
    python -m src.data.build_crowd_dataset --crowd
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.features.link_features import SHORTENERS, extract_links
from src.unified.prepare_web_37 import FEATURES_37

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PILOT_IN = PROCESSED / "fresh_lean.csv"
PILOT_OUT = PROCESSED / "crowd_pilot_synthetic.csv"
CROWD_RAW = ROOT / "data" / "crowd" / "raw"
CROWD_CLEAN = Path(r"D:\LeJetData\clean")
CROWD_OUT = PROCESSED / "crowd_smishing_draft.csv"
LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=8))   # Philippine time, as date_submitted is typed

sys.path.insert(0, str(ROOT / "scripts"))
import crowd_to_urls  # noqa: E402  (the same reader and first-link rule the lab's URL list was made with)

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


# Mobile numbers (09xx / +639xx / 639xx, spaced or dashed) and long digit runs (accounts, references,
# OTPs) outside the link. Names cannot be found reliably by a pattern: section 4.4 sends those back.
_PHONE = re.compile(r"(?<!\d)(?:\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
_LONG_DIGITS = re.compile(r"(?<!\d)\d(?:[\s-]?\d){7,}(?!\d)")


def mask_numbers(text: str) -> str:
    """Replace phone numbers with <NUM> and 8+ digit runs with <ACCT>; the links are left untouched."""
    out, pos = [], 0
    for link in extract_links(text) + [{"start": len(text), "end": len(text)}]:
        part = _PHONE.sub("<NUM>", text[pos:link["start"]])
        out.append(_LONG_DIGITS.sub("<ACCT>", part))
        out.append(text[link["start"]:link["end"]])
        pos = link["end"]
    return "".join(out)


def url_id(url: str) -> str:
    """The lab's join key: src/lejet/urlnorm.compute_url_id of the URL as pushed (make_push_file.py)."""
    return hashlib.sha256(url.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _refang(text: str) -> str:
    return re.sub(r"(?i)^hxxp", "http", str(text).replace("[.]", "."))


def fetch_status(rec: dict) -> str:
    """Section 4.3: live, redirected, dead, blocked or timeout, from the lab's fail_reason."""
    reason = rec.get("fail_reason", "internal_error")
    if reason == "none":
        return "redirected" if rec.get("final_differs") else "live"
    if "timeout" in reason:
        return "timeout"
    if reason in ("rejected_url", "port_blocked", "redirect_to_blocked", "challenge_page", "blocked_or_empty"):
        return "blocked"
    return "dead"


def load_lab(clean_dir: Path) -> Dict[str, dict]:
    """The newest lab record per url_id over every crowd*.clean.jsonl, with page_rank from <batch>.model_input.csv."""
    best: Dict[str, dict] = {}
    for path in sorted(clean_dir.glob("crowd*.clean.jsonl")):
        ranks: Dict[str, int] = {}
        lookup = path.with_name(path.name.replace(".clean.jsonl", ".model_input.csv"))
        if lookup.exists():
            ranks = dict(pd.read_csv(lookup, usecols=["url_id", "page_rank"], dtype={"url_id": str}).values)
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            rec["page_rank"] = int(ranks.get(rec["url_id"], MISSING))
            if rec["url_id"] not in best or rec["fetch_ts"] > best[rec["url_id"]]["fetch_ts"]:
                best[rec["url_id"]] = rec
    return best


def _local(ts: str) -> datetime.datetime:
    utc = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    return utc.astimezone(LOCAL_TZ)


def _hours_between(submitted: str, inspected: datetime.datetime) -> float:
    try:
        start = datetime.datetime.strptime(submitted, "%Y-%m-%d %H:%M").replace(tzinfo=LOCAL_TZ)
    except ValueError:
        return MISSING
    return round((inspected - start).total_seconds() / 3600, 2)


def _text(row: Dict[str, str], col: str):
    return (row.get(col) or "").strip() or MISSING


def build_crowd(raw_dir: Path = CROWD_RAW, clean_dir: Path = CROWD_CLEAN) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    files = sorted(p for p in glob.glob(str(raw_dir / "submissions_*.*"))
                   if p.lower().endswith((".xlsx", ".csv")) and not Path(p).name.startswith("~$"))
    lab = load_lab(clean_dir)
    picked, left_out = [], {"example_row": 0, "no_consent": 0, "not_verified": 0, "not_inspected": 0}
    for path in files:
        for r in crowd_to_urls.read_rows(path):
            text = (r.get("sms_text") or "").strip()
            if text == crowd_to_urls.EXAMPLE_TEXT:
                left_out["example_row"] += 1
            elif (r.get("consent") or "").strip() != "1":
                left_out["no_consent"] += 1
            elif not (r.get("label_verified_by") or "").strip():
                left_out["not_verified"] += 1
            else:
                link = crowd_to_urls.first_link(text)
                rec = lab.get(url_id(link)) if link else None
                if link and rec is None:
                    left_out["not_inspected"] += 1
                else:
                    picked.append((r, text, rec))
    picked.sort(key=lambda x: x[0]["date_submitted"])

    out = []
    for i, (r, text, rec) in enumerate(picked, start=1):
        links = extract_links(text)
        link = links[0]["raw"] if links else ""
        rec = rec or {}   # no link: the row rules drop the row, only the message columns matter
        inspected = _local(rec["fetch_ts"]) if rec.get("fetch_ts") else None
        submitted = (r.get("date_submitted") or "").strip()
        row = {
            "message_id": f"crowd_{i:06d}",
            "sms_text": mask_numbers(text),
            **{c: _text(r, c) for c in ["label", "category", "language", "sender_type", "date_received",
                                        "date_submitted", "label_source", "label_verified_by"]},
            "consent": 1,
            "link": link,
            "final_url": _refang(rec["final_url"]) if rec.get("final_status", -1) != -1 and rec.get("final_url")
            else MISSING,
            "link_domain": _refang(rec.get("final_reg_domain") or rec.get("reg_domain") or MISSING),
            "template_id": "",
            "link_shortened": is_shortened(link) if link else 0,
            "inspected_at": inspected.strftime("%Y-%m-%d %H:%M") if inspected else MISSING,
            "hours_to_inspect": _hours_between(submitted, inspected) if inspected else MISSING,
            "page_fetched": int(rec.get("page_fetched", 0)),
            "fetch_status": fetch_status(rec) if rec else "dead",
        }
        full = rec.get("features_full", {})
        for f in FEATURES_37:
            row[f] = rec.get("page_rank", MISSING) if f == "page_rank" else full.get(f, MISSING)
        out.append(row)
    df = pd.DataFrame(out, columns=COLUMNS)
    if len(df):
        df["template_id"] = template_ids(df["sms_text"])
    kept, set_aside, counts = apply_row_rules(df.fillna(MISSING))
    counts.update({f"left_out_{k}": v for k, v in left_out.items()})
    check(kept)
    summary = summarize(kept, counts, f"CROWD DRAFT from {len(files)} submission file(s); ids may still shift")
    summary["inspected_over_24h"] = int((kept["hours_to_inspect"] > 24).sum())
    return kept, set_aside, summary


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
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="build the synthetic pilot from fresh_lean.csv")
    mode.add_argument("--crowd", action="store_true", help="build the crowd draft from data/crowd/raw + the lab results")
    ap.add_argument("--clean-dir", type=Path, default=CROWD_CLEAN, help="where the lab's crowd*.clean.jsonl files are")
    a = ap.parse_args(argv)
    if a.pilot:
        (kept, set_aside, summary), out = build_pilot(), PILOT_OUT
    else:
        (kept, set_aside, summary), out = build_crowd(clean_dir=a.clean_dir), CROWD_OUT
    write(kept, set_aside, summary, out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out} ({len(kept)} rows, {len(COLUMNS)} columns); {len(set_aside)} rows set aside")


if __name__ == "__main__":
    main()
