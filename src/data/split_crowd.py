"""Grouped 5-fold cross-validation splits for the C-prune study (docs/c_prune_architecture.md Phase 2).

For each seed (42, 123, 2024) and each of the 5 folds, every message gets one part:
    test   the fold itself: every message is tested exactly once per seed
    val    about 15% of the other 4 folds, for early stopping
    train  the rest
Rows of one group never sit in two parts. `--group-by` picks the group:
    domain+template   (the plan's rule) messages sharing a link_domain OR a template_id are joined into one
                      group, so no domain and no template appears in two parts. Refused when one joined group
                      is too big to fit in a fold (the synthetic pilot: one group holds ~73% of the rows).
    domain            link_domain only. Template overlap between parts is then measured and reported.

Per fold it also measures SEQ_LEN_L: the 95th percentile of token counts (the same cleaning as the CNN,
link -> <URL>) on the TRAIN part only.

Outputs, in <splits_dir>/<dataset name>/:
    seed<S>_fold<F>.csv   message_id,part
    split_report.json     the group rule, and per fold: sizes, smishing share, SEQ_LEN_L, overlaps

Usage:
    python -m src.data.split_crowd --pilot --group-by domain
    python -m src.data.split_crowd                                  # the real crowd dataset, strict rule
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    CROWD_DATASET_PATH, CROWD_N_FOLDS, CROWD_PILOT_PATH, CROWD_SPLITS_DIR, CROWD_VAL_FRACTION, SEEDS,
)
from src.features.text_preprocessing import text_tokens

ROOT = Path(__file__).resolve().parents[2]
GROUP_RULES = ("domain+template", "domain")
PARTS = ("train", "val", "test")


def joined_groups(df: pd.DataFrame) -> pd.Series:
    """Union-find over link_domain and template_id: rows linked by either share a group."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    domains = "d:" + df["link_domain"].astype(str)
    templates = "t:" + df["template_id"].astype(str)
    for d, t in zip(domains, templates):
        parent[find(d)] = find(t)
    return domains.map(find)


def make_groups(df: pd.DataFrame, rule: str, n_folds: int = CROWD_N_FOLDS) -> pd.Series:
    if rule == "domain":
        return "d:" + df["link_domain"].astype(str)
    if rule != "domain+template":
        raise ValueError(f"unknown group rule {rule!r}")
    groups = joined_groups(df)
    largest = groups.value_counts().max()
    # A fold holds about 1/n of the rows; a bigger group cannot be placed without breaking the folds.
    if largest > len(df) / n_folds:
        raise ValueError(
            f"one domain+template group holds {largest} of {len(df)} rows ({largest / len(df):.0%}), more than "
            f"one fold (1/{n_folds}). The folds cannot keep every domain and template apart: use --group-by domain "
            "and read the template overlap in the report.")
    return groups


def fold_parts(labels: pd.Series, groups: pd.Series, seed: int, n_folds: int = CROWD_N_FOLDS,
               val_fraction: float = CROWD_VAL_FRACTION) -> List[np.ndarray]:
    """One array of part names per fold, aligned with the rows."""
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # Validation = one of k grouped inner folds, k chosen so 1/k is close to val_fraction.
    k = max(2, round(1 / val_fraction))
    result = []
    for rest, test in outer.split(np.zeros(len(labels)), labels, groups):
        part = np.full(len(labels), "train", dtype=object)
        part[test] = "test"
        inner = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        _, val = next(inner.split(np.zeros(len(rest)), labels.iloc[rest], groups.iloc[rest]))
        part[rest[val]] = "val"
        result.append(part)
    return result


def seq_len(texts: pd.Series) -> int:
    """95th percentile of token counts, rounded up (src/config.py SEQ_LEN_L method)."""
    return int(math.ceil(np.percentile([len(text_tokens(str(t))) for t in texts], 95)))


def check_fold(df: pd.DataFrame, part: np.ndarray, groups: pd.Series) -> Dict[str, object]:
    """Asserts the guarantees and measures what the group rule does not guarantee."""
    by_part = {p: df[part == p] for p in PARTS}
    for p in PARTS:
        assert len(by_part[p]), f"empty {p} part"
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(groups[part == a]) & set(groups[part == b]), f"a group is in both {a} and {b}"
    train_templates = set(by_part["train"]["template_id"])
    train_domains = set(by_part["train"]["link_domain"])
    test = by_part["test"]
    return {
        "rows": {p: len(by_part[p]) for p in PARTS},
        "smishing_share": {p: round(float((by_part[p]["label"] == "smishing").mean()), 4) for p in PARTS},
        "seq_len_l": seq_len(by_part["train"]["sms_text"]),
        "test_rows_with_train_domain": int(test["link_domain"].isin(train_domains).sum()),
        "test_rows_with_train_template": int(test["template_id"].isin(train_templates).sum()),
    }


def run(dataset: Path, out_dir: Path, rule: str, seeds: List[int] = SEEDS,
        n_folds: int = CROWD_N_FOLDS) -> Tuple[dict, Dict[Tuple[int, int], pd.DataFrame]]:
    df = pd.read_csv(dataset, keep_default_na=False)
    assert df["message_id"].is_unique
    groups = make_groups(df, rule, n_folds)
    report: dict = {"dataset": dataset.name, "group_by": rule, "n_folds": n_folds, "rows": len(df),
                    "groups": int(groups.nunique()), "largest_group": int(groups.value_counts().max()), "folds": []}
    files = {}
    for seed in seeds:
        tested = np.zeros(len(df), dtype=int)
        for fold, part in enumerate(fold_parts(df["label"], groups, seed, n_folds)):
            tested += part == "test"
            report["folds"].append({"seed": seed, "fold": fold, **check_fold(df, part, groups)})
            files[(seed, fold)] = pd.DataFrame({"message_id": df["message_id"], "part": part})
        assert (tested == 1).all(), "a message is not tested exactly once"
    out_dir.mkdir(parents=True, exist_ok=True)
    for (seed, fold), frame in files.items():
        frame.to_csv(out_dir / f"seed{seed}_fold{fold}.csv", index=False)
    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, files


def load_fold(seed: int, fold: int, pilot: bool = False,
              splits_dir: str = CROWD_SPLITS_DIR) -> Dict[str, pd.DataFrame]:
    """{train, val, test} rows of one fold, for Phases 3-7."""
    dataset = ROOT / (CROWD_PILOT_PATH if pilot else CROWD_DATASET_PATH)
    df = pd.read_csv(dataset, keep_default_na=False)
    parts = pd.read_csv(ROOT / splits_dir / dataset.stem / f"seed{seed}_fold{fold}.csv")
    df = df.merge(parts, on="message_id", validate="one_to_one")
    return {p: df[df["part"] == p].drop(columns="part").reset_index(drop=True) for p in PARTS}


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="split the synthetic pilot (pipeline test only)")
    ap.add_argument("--group-by", choices=GROUP_RULES, default="domain+template")
    a = ap.parse_args(argv)
    dataset = ROOT / (CROWD_PILOT_PATH if a.pilot else CROWD_DATASET_PATH)
    out_dir = ROOT / CROWD_SPLITS_DIR / dataset.stem
    try:
        report, files = run(dataset, out_dir, a.group_by)
    except ValueError as err:
        raise SystemExit(str(err))
    folds = pd.DataFrame([{"seed": f["seed"], "fold": f["fold"], **{f"n_{p}": f["rows"][p] for p in PARTS},
                           "smish_test": f["smishing_share"]["test"], "seq_len_l": f["seq_len_l"],
                           "test_train_domain": f["test_rows_with_train_domain"],
                           "test_train_template": f["test_rows_with_train_template"]} for f in report["folds"]])
    pd.set_option("display.width", 200)
    print(f"{report['dataset']}: {report['rows']} rows, {report['groups']} groups by {report['group_by']} "
          f"(largest {report['largest_group']})")
    print(folds.to_string(index=False))
    print(f"wrote {len(files)} split files and split_report.json to {out_dir}")


if __name__ == "__main__":
    main()
