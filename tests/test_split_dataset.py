"""
Tests for the grouped SMS split (src/data/split_dataset.py): no domain or synthetic template
spans two splits, and the sizes are about 75 / 10 / 15.

Run with:  pytest tests/test_split_dataset.py
"""

import numpy as np
import pandas as pd

from src.data.split_dataset import ID_COLUMN, split_groups, three_way_split


def _toy_sms(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kind = rng.choice(["real_link", "real_none", "synthetic"], size=n, p=[0.3, 0.5, 0.2])
    return pd.DataFrame({
        ID_COLUMN: [f"m{i}" for i in range(n)],
        "label": np.where(kind == "synthetic", "legitimate", rng.choice(["smishing", "legitimate"], size=n)),
        "source": np.where(kind == "synthetic", "synthetic_legit_url", "v2_2026-09-15"),
        "template_id": np.where(kind == "synthetic", [f"t{i % 30}" for i in range(n)], ""),
        "link_status": np.where(kind == "real_link", "link", "none"),
        "link_domain": np.where(kind == "real_link", [f"d{i % 80}.com" for i in range(n)], ""),
    })


def test_groups_never_span_two_splits():
    df = _toy_sms()
    groups = split_groups(df)
    train, rf_fit, test = three_way_split(df, "label", seed=42)
    seen = [set(groups[part.index]) for part in (train, rf_fit, test)]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    assert len(train) + len(rf_fit) + len(test) == len(df)


def test_split_sizes():
    df = _toy_sms()
    train, rf_fit, test = three_way_split(df, "label", seed=123)
    assert abs(len(test) / len(df) - 0.15) < 0.04
    assert abs(len(rf_fit) / len(df) - 0.10) < 0.04


def test_grouping_keys():
    df = _toy_sms(50)
    groups = split_groups(df)
    assert groups[df["source"] == "synthetic_legit_url"].str.startswith("template:").all()
    real_link = (df["source"] != "synthetic_legit_url") & (df["link_status"] == "link")
    assert groups[real_link].str.startswith("domain:").all()
