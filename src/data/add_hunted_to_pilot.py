r"""Add the link hunter's live links (lab results) to the synthetic crowd pilot, with SYNTHETIC SMS text.

Two outputs:
    data/live/hunt_live_features.csv   every live hunted link: the lab's 78 columns + where the hunter found it
                                       (source, trust). Real URLs stay defanged (hxxps://a[.]com).
    data/processed/crowd_pilot_synthetic.csv (+ .set_aside.csv, .summary.json)
                                       the existing pilot with one new row per live hunted link, message ids continuing.

The hunted links come from phishing feeds, not from SMS: the text around them is invented, exactly like the rest of
the pilot (src/unified/prepare_lean_synthetic.py templates), and every one of them is labelled `smishing`, the same
way the pilot labels web-phishing links. It is a pipeline test only, never a study result.
Before the first change the original pilot is copied to crowd_pilot_synthetic.v1_*.csv (never overwritten).

Usage:
    python -m src.data.add_hunted_to_pilot [--clean-dir D:\LeJetData\clean] [--prefix hunt20261019_1512]
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract

from src.data import build_crowd_dataset as bcd
from src.unified.prepare_lean_synthetic import BRANDS, NEUTRAL, PHISH, brand_for  # noqa: F401  (BRANDS: same list)
from src.unified.prepare_web_37 import FEATURES_37

ROOT = Path(__file__).resolve().parents[2]
HUNTER = Path(r"D:\LeJetLinkHunter")
LIVE_OUT = ROOT / "data" / "live" / "hunt_live_features.csv"
SOURCE_NOTE = "SYNTHETIC PILOT (fresh_lean.csv + live hunted phishing links): pipeline test only, not study results"


def load_live(clean_dir: Path, prefix: str) -> pd.DataFrame:
    """Every live (page_fetched = 1) row of <prefix>p*.full_features.csv, one per url, with the hunter's source and trust."""
    files = sorted(glob.glob(str(clean_dir / f"{prefix}p*.full_features.csv")))
    if not files:
        raise SystemExit(f"no {prefix}p*.full_features.csv in {clean_dir}")
    lab = pd.concat([pd.read_csv(f, dtype={"url_id": str}) for f in files], ignore_index=True)
    live = lab[lab["page_fetched"] == 1].drop_duplicates("url").reset_index(drop=True)
    missing = [c for c in FEATURES_37 if c not in live.columns]
    assert not missing, f"lab files lack features: {missing}"
    master = HUNTER / "MASTER_smishing_links.csv"
    if master.exists():
        m = pd.read_csv(master, usecols=["link_defanged", "source", "trust"], encoding="utf-8-sig")
        live = live.merge(m.drop_duplicates("link_defanged").rename(columns={"link_defanged": "url"}), on="url", how="left")
    else:
        live["source"], live["trust"] = np.nan, np.nan
    live[["source", "trust"]] = live[["source", "trust"]].fillna("unknown")
    return live


def synthetic_rows(live: pd.DataFrame, first_id: int, neutral_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    links = live["url"].map(bcd._refang)
    texts = []
    for link in links:
        pool = NEUTRAL if rng.random() < neutral_rate else PHISH
        texts.append(str(rng.choice(pool)).format(b=brand_for(link, rng), u=link, n=f"{int(rng.integers(2, 80)) * 100:,}"))
    ext = links.map(tldextract.extract)
    out = pd.DataFrame({
        "message_id": [f"pilot_{i:06d}" for i in range(first_id, first_id + len(live))],
        "sms_text": texts,
        "label": "smishing",
        "category": [bcd.category_for(t) for t in texts],
        "language": [bcd.language_for(t) for t in texts],
        "sender_type": "unknown",
        "date_received": bcd.MISSING,
        "date_submitted": bcd.MISSING,
        "label_source": bcd.MISSING,
        "label_verified_by": bcd.MISSING,
        "consent": 1,
        "link": links.values,
        "final_url": bcd.MISSING,
        "link_domain": [e.registered_domain or e.domain for e in ext],
        "template_id": bcd.template_ids(pd.Series(texts)).values,
        "link_shortened": links.map(bcd.is_shortened).values,
        "inspected_at": bcd.MISSING,
        "hours_to_inspect": bcd.MISSING,
        "page_fetched": 1,
        "fetch_status": "live",
    })
    return pd.concat([out, live[FEATURES_37].reset_index(drop=True)], axis=1)[bcd.COLUMNS]


def add_to_pilot(pilot: pd.DataFrame, aside: pd.DataFrame, live: pd.DataFrame, neutral_rate: float = 0.25, seed: int = 42):
    """Returns (kept, set_aside, counts). Links already in the pilot are skipped; old rows keep their ids and order."""
    fresh = live[~live["url"].map(bcd._refang).isin(set(pilot["link"]) | set(aside["link"]))]
    first_id = int(pilot["message_id"].str.extract(r"(\d+)$")[0].astype(int).max()) + 1
    new = synthetic_rows(fresh, first_id, neutral_rate, seed)
    kept, over, counts = bcd.apply_row_rules(pd.concat([pilot, new], ignore_index=True))
    counts["hunted_live_in"] = len(live)
    counts["hunted_already_in_pilot"] = len(live) - len(fresh)
    return kept, pd.concat([aside, over], ignore_index=True), counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clean-dir", type=Path, default=bcd.CROWD_CLEAN)
    ap.add_argument("--prefix", default="hunt20261019_1512")
    ap.add_argument("--neutral-rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    live = load_live(a.clean_dir, a.prefix)
    LIVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    live.to_csv(LIVE_OUT, index=False, encoding="utf-8")
    print(f"wrote {LIVE_OUT} ({len(live)} live links)")

    out = bcd.PILOT_OUT
    aside_path = out.with_suffix(".set_aside.csv")
    for src in (out, aside_path, out.with_suffix(".summary.json")):   # first-run backup of the original pilot
        keep = src.with_name(src.name.replace("crowd_pilot_synthetic", "crowd_pilot_synthetic.v1_original", 1))
        if not keep.exists():
            shutil.copy2(src, keep)
    pilot = pd.read_csv(out, keep_default_na=False, na_values=[])
    aside = pd.read_csv(aside_path, keep_default_na=False, na_values=[])
    kept, set_aside, counts = add_to_pilot(pilot, aside, live, a.neutral_rate, a.seed)
    bcd.check(kept)
    summary = bcd.summarize(kept, counts, SOURCE_NOTE)
    bcd.write(kept, set_aside, summary, out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out} ({len(kept)} rows, {len(bcd.COLUMNS)} columns); {len(set_aside)} rows set aside")


if __name__ == "__main__":
    main()
