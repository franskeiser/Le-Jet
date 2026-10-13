"""Build the ONE dataset that trains the Random Forest extractor and the CNN together.

One row per SMS message: the text (CNN input), the link, the 14 link-text features, the 17
page/domain features, and how the page evidence was obtained. The same table, with the same
splits (data/splits), feeds both models (docs/unified_architecture.md).

Page evidence (`inspect_status`):
    loaded         the 17 page/domain columns hold real values
    failed         a link was inspected but the page could not be fetched; page columns are NaN
    not_attempted  old SMS data: the domain is mostly dead, so nothing is looked up; page columns are NaN
`page_origin` says where loaded values came from:
    hybrid_temp    TEMPORARY placeholder borrowed from the public web set
                   (src/data/build_hybrid_temp_links.py). Not study data.
    inspector      the lab inspector, on a fresh link (the real thing, once collected)

Usage (from the project root):
    python -m src.unified.build_unified_dataset
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH
from src.data.split_dataset import ID_COLUMN, ensure_message_id
from src.features.link_features import LINK_TEXT_FEATURES
from src.features.model.rf_extractor import EXTERNAL_FEATURES, PAGE_RANK, WEBPAGE_FEATURES

ROOT = Path(__file__).resolve().parents[2]
HYBRID_PATH = ROOT / "data" / "processed" / "sms_link_features_hybrid_temp.csv"
OUT_PATH = ROOT / "data" / "processed" / "unified_dataset.csv"

PAGE_COLUMNS = [PAGE_RANK] + WEBPAGE_FEATURES + EXTERNAL_FEATURES
LINK_COLUMNS = ["link_status", "link", "link_domain", "n_links", "has_url"]
COLUMNS = (
    [ID_COLUMN, "sms_text", "label", "source", "template_id"] + LINK_COLUMNS + LINK_TEXT_FEATURES
    + ["inspect_status", "page_origin"] + PAGE_COLUMNS + ["page_source_url"]
)


def build(sms: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    sms = ensure_message_id(sms)
    keep = [ID_COLUMN] + LINK_COLUMNS + LINK_TEXT_FEATURES + PAGE_COLUMNS + ["page_source_url", "hybrid_page"]
    merged = sms.merge(links[keep], on=ID_COLUMN, how="left", validate="one_to_one")
    if merged["has_url"].isna().any():
        raise ValueError("a message has no row in the link-feature table")

    hybrid = merged["hybrid_page"] == 1
    merged["inspect_status"] = hybrid.map({True: "loaded", False: "not_attempted"})
    merged["page_origin"] = hybrid.map({True: "hybrid_temp", False: "none"})
    return merged[COLUMNS]


def balance(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Down-sample to `size` rows, half smishing and half legitimate.

    Redacted-<URL> rows are dropped (docs/training_rules.md rule 1 excludes them anyway). Every row
    with a real link is kept, because only those reach the forest; each class is then filled up to
    size/2 with randomly drawn no-link messages. If a class has more link rows than its share, its
    link rows are down-sampled instead."""
    per_class = size // 2
    parts = []
    for label in ("smishing", "legitimate"):
        rows = df[(df["label"] == label) & (df["link_status"] != "redacted")]
        linked, unlinked = rows[rows["link_status"] == "link"], rows[rows["link_status"] != "link"]
        if len(linked) >= per_class:
            parts.append(linked.sample(per_class, random_state=seed))
            continue
        need = per_class - len(linked)
        if len(unlinked) < need:
            raise ValueError(f"{label}: only {len(rows)} usable rows, need {per_class}")
        parts.append(pd.concat([linked, unlinked.sample(need, random_state=seed)]))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sms", default=str(ROOT / DATASET_PATH))
    ap.add_argument("--links", type=Path, default=HYBRID_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--size", type=int, default=15000, help="balanced total rows (half per class); 0 keeps everything")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = build(pd.read_csv(args.sms), pd.read_csv(args.links, low_memory=False))
    if args.size:
        out = balance(out, args.size, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} rows")
    print(pd.crosstab([out["link_status"], out["inspect_status"], out["page_origin"]], out["label"]).to_string())


if __name__ == "__main__":
    main()
