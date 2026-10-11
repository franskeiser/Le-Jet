"""Compute has_url and the 14 link-text features for every SMS message.

Input:  data/processed/sms_v2_balanced.csv (real v2 messages + synthetic legitimate ones)
Output: data/processed/sms_link_features.csv, one row per message, joined by message_id

Only the FIRST link in a message feeds the features (one link's worth of features per
message, as in SmishGuard). No network access: the features come from the link string.

link_status says what the URL branch can know about a message:
    link      -- a link was found; has_url = 1 and the 14 features are computed
    redacted  -- the text has the <URL> placeholder (the link was removed upstream) and no
                 other link; has_url = 1 but the features are unknown (-1)
    none      -- no link; has_url = 0 and the features are -1

Usage (from the project root):
    python -m src.data.build_sms_link_features
"""
import argparse
from pathlib import Path

import pandas as pd

from src.data.split_dataset import ID_COLUMN, ensure_message_id
from src.features.link_features import (
    LINK_TEXT_FEATURES, extract_links, link_text_features, no_link_features, normalize_link,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "data" / "processed" / "sms_v2_balanced.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "sms_link_features.csv"
PLACEHOLDER = "<URL>"


def message_features(text: str) -> dict:
    links = extract_links(text)
    if links:
        url = normalize_link(links[0]["raw"])
        return {"link_status": "link", "link": url, "link_domain": links[0]["domain"],
                "n_links": len(links), "has_url": 1, **link_text_features(url)}
    status = "redacted" if PLACEHOLDER in text else "none"
    return {"link_status": status, "link": "", "link_domain": "", "n_links": 0,
            "has_url": int(status == "redacted"), **no_link_features()}


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_message_id(df)
    features = pd.DataFrame([message_features(str(t)) for t in df["sms_text"]], index=df.index)
    keep = [ID_COLUMN, "label", "source", "template_id"]
    return pd.concat([df[keep], features], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = build(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Wrote {len(out)} rows x {out.shape[1]} cols -> {args.output}")
    print(pd.crosstab([out["source"], out["label"]], out["link_status"], margins=True).to_string())
    print(f"\nFeatures: {', '.join(LINK_TEXT_FEATURES)}")


if __name__ == "__main__":
    main()
