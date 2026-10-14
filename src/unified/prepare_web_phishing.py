"""Make web_phishing_clean.csv usable by the features-only run: add the columns the split and forest code read
(message_id, label, link_status, link_domain, source) and write domain-grouped splits for every seed.

Real pages, real URL + page features, no SMS text. Rows of one registrable domain always land in the same part.

Usage:
    python -m src.unified.prepare_web_phishing
    python -m src.unified.run_features_only --data data/processed/web_phishing_linked.csv --splits-dir data/splits_web_phishing --out experiments/web_phishing
"""
from pathlib import Path

import pandas as pd
import tldextract

from src.config import SEEDS
from src.data.split_dataset import assert_disjoint, save_ids
from src.unified.make_splits import split_with_big_groups

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "processed" / "web_phishing_clean.csv"
OUT = ROOT / "data" / "processed" / "web_phishing_linked.csv"
SPLITS = ROOT / "data" / "splits_web_phishing"


def main() -> None:
    df = pd.read_csv(SRC, low_memory=False)
    extract = tldextract.TLDExtract(suffix_list_urls=())      # offline: bundled suffix list
    df["link_domain"] = df["url"].map(lambda u: extract(u).top_domain_under_public_suffix or u)
    df["message_id"] = [f"web_{i}" for i in range(len(df))]
    df["label"] = df["status"].map({"phishing": "smishing", "legitimate": "legitimate"})
    df["link_status"] = "link"
    df["source"] = "web"
    df["template_id"] = ""
    df.to_csv(OUT, index=False)
    print(f"{OUT}: {len(df)} rows, {df['link_domain'].nunique()} domains")

    for seed in SEEDS:
        train, rf_fit, test = split_with_big_groups(df, seed, 100)
        assert_disjoint(train, rf_fit, test)
        for name, part in [("train", train), ("rf_fit", rf_fit), ("test", test)]:
            save_ids(part, str(SPLITS / f"seed_{seed}" / f"{name}_ids.txt"))
        print(f"seed {seed}: train {len(train)}, rf_fit {len(rf_fit)}, test {len(test)} "
              f"({(test['label'] == 'smishing').mean():.2f} smishing)")


if __name__ == "__main__":
    main()
