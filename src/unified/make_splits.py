"""Grouped Training / RF-Fitting / Test split of the unified dataset, for every seed.

Same rule as src/data/split_dataset.py (rows of one domain or one synthetic template always land
in the same part; 75% / 10% / 15%), applied to the table the unified model actually trains on.
One change: groups of --big-group rows or more (three scam domains hold 44% of all smishing links)
always go to Training. Left to the random fold assignment they tip whichever part receives them,
and a test set that is mostly one scam domain says nothing about unseen domains. Writes data/splits_unified/seed_<s>/
{train,rf_fit,test}_ids.txt.

Usage (from the project root):
    python -m src.unified.make_splits
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import SEEDS
from src.data.split_dataset import assert_disjoint, save_ids, split_groups, split_report, three_way_split
from src.unified.build_unified_dataset import OUT_PATH

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits_unified"


def split_with_big_groups(df: pd.DataFrame, seed: int, big_group: int):
    groups = split_groups(df)
    big = groups.map(groups.value_counts()) >= big_group
    train, rf_fit, test = three_way_split(df[~big], "label", seed)
    return pd.concat([train, df[big]]), rf_fit, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=OUT_PATH)
    ap.add_argument("--out", type=Path, default=SPLITS_DIR)
    ap.add_argument("--big-group", type=int, default=100, help="groups this large always go to Training")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    df = pd.read_csv(args.data, low_memory=False, keep_default_na=False, na_values=[""])
    rows = []
    for seed in args.seeds:
        train, rf_fit, test = split_with_big_groups(df, seed, args.big_group)
        assert_disjoint(train, rf_fit, test)
        assert len(train) + len(rf_fit) + len(test) == len(df), "rows lost during split"
        for name, part in [("train", train), ("rf_fit", rf_fit), ("test", test)]:
            save_ids(part, str(args.out / f"seed_{seed}" / f"{name}_ids.txt"))
            rows.append({"seed": seed, **split_report(name, part, "label")})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
