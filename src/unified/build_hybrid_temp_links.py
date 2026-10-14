"""TEMPORARY placeholder: give old SMS links page/domain features borrowed from the web set.

Why: the page and domain features (page_rank, HTML, WHOIS, DNS) cannot be computed for old SMS
links, because 82% of the smishing domains are dead (docs/decision_web_inspection.md). Until the
fresh inspected set exists, this builds a stand-in so the single-dataset pipeline can be wired up
and tested end to end.

How: every SMS row that has a real link keeps its own link-text features. It is paired with one
web-set page of the same class (smishing <-> phishing, legitimate <-> legitimate), drawn without
replacement, and receives that page's 17 page/domain features. Rows without a usable link
(no link, or a redacted <URL> placeholder) keep the page columns empty.

NOT STUDY DATA. The page features are independent of the SMS link they are attached to, so a
model trained on this file learns the web set's page signal and the SMS set's link signal
separately and scores will be optimistic. Do not report numbers from it. Every row carries
hybrid_page=1 and page_source_url so it can be found and dropped when the fresh set replaces it.

Usage (from the project root):
    python -m src.unified.build_hybrid_temp_links
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.model.rf_extractor import EXTERNAL_FEATURES, PAGE_RANK, WEBPAGE_FEATURES

ROOT = Path(__file__).resolve().parents[2]
SMS_PATH = ROOT / "data" / "processed" / "sms_link_features.csv"
WEB_PATH = ROOT / "data" / "processed" / "web_phishing_clean.csv"
OUT_PATH = ROOT / "data" / "processed" / "sms_link_features_hybrid_temp.csv"

PAGE_COLUMNS = [PAGE_RANK] + WEBPAGE_FEATURES + EXTERNAL_FEATURES
SMS_TO_WEB_LABEL = {"smishing": "phishing", "legitimate": "legitimate"}


def build(sms: pd.DataFrame, web: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = sms.copy()
    out[PAGE_COLUMNS] = np.nan
    out["page_source_url"] = pd.NA
    out["hybrid_page"] = 0

    has_link = out["link_status"] == "link"
    for sms_label, web_label in SMS_TO_WEB_LABEL.items():
        rows = out.index[has_link & (out["label"] == sms_label)]
        pool = web[web["status"] == web_label]
        if len(rows) > len(pool):
            raise ValueError(f"{sms_label}: {len(rows)} SMS links but only {len(pool)} {web_label} pages")
        picked = pool.iloc[rng.choice(len(pool), size=len(rows), replace=False)]
        out.loc[rows, PAGE_COLUMNS] = picked[PAGE_COLUMNS].to_numpy()
        out.loc[rows, "page_source_url"] = picked["url"].to_numpy()
        out.loc[rows, "hybrid_page"] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sms", type=Path, default=SMS_PATH)
    ap.add_argument("--web", type=Path, default=WEB_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = build(pd.read_csv(args.sms), pd.read_csv(args.web), args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n = int(out["hybrid_page"].sum())
    print(f"wrote {args.out} ({len(out)} rows, {n} with borrowed page features)")
    print(out[out["hybrid_page"] == 1].groupby("label").size().to_string())


if __name__ == "__main__":
    main()
