"""E2's link inputs: P, u and page_missing for every message of a fold (plan docs/film_architecture_plan.md, step 1.2).

E2 reads three numbers per message from the link lane (link_lane.load_lane), not the 40 the concatenation models read:
    P            mean smishing probability of the Link RF's trees
    u            spread of the trees' probabilities
    page_missing 1 if the page was not fetched
Each is given twice, scored by the SAME forest: `full` (the page as inspected) and `blank` (the page hidden, page_missing = 1,
used by modality dropout while training). Train rows carry out-of-fold values (no row is scored by a forest that saw it or
its domain); val and test rows carry the final forest's values. load_lane already splits them that way; this module only
lines them up with the text rows and checks them.

align() is the check: every message has exactly one row of link numbers, in the order of the text rows (as
fusion.fusion_inputs did on the pilot branch), the numbers are finite, P is a probability, u >= 0, and page_missing agrees
with the dataset's page_fetched (full) or is 1 everywhere (blank).

Usage:
    python -m src.unified.film_inputs --pilot       # check every seed x fold of the synthetic pilot, print a summary
    python -m src.unified.film_inputs               # the real crowd dataset
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import CROWD_N_FOLDS, SEEDS
from src.data.split_crowd import load_fold
from src.unified import link_lane as ll

E2_LINK = ["P", "u", "page_missing"]
PARTS = ("train", "val", "test")


@dataclass
class PartLink:
    """One part (train, val or test) of one fold: the link numbers, in the order of the text rows."""

    message_id: np.ndarray               # (n,) message ids, the text rows' order
    label: np.ndarray                    # (n,) 1 = smishing, 0 = legitimate
    full: Dict[str, np.ndarray]          # P, u, page_missing, each (n,): the page as inspected
    blank: Dict[str, np.ndarray]         # the same with the page hidden (page_missing = 1)

    def stack(self, which: str = "full") -> np.ndarray:
        """(n, 3) float32 array [P, u, page_missing] of `full` or `blank`."""
        numbers = self.full if which == "full" else self.blank
        return np.column_stack([numbers[c] for c in E2_LINK]).astype(np.float32)


def align(rows: pd.DataFrame, full: pd.DataFrame, blank: pd.DataFrame) -> PartLink:
    """Line the lane's frames up with the text `rows` of one part, or stop with the reason they do not match."""
    ids = rows["message_id"]
    if ids.duplicated().any():
        raise ValueError("a message id appears twice in the text rows")
    aligned = {}
    for name, lane in (("full", full), ("blank", blank)):
        if len(lane) != len(rows) or lane["message_id"].duplicated().any() or set(lane["message_id"]) != set(ids):
            raise ValueError(f"{name}: the link numbers and the text rows do not match one to one")
        link = rows[["message_id"]].merge(lane[["message_id"] + E2_LINK], on="message_id", how="left", validate="one_to_one")
        values = link[E2_LINK].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: a link number is missing or not finite")
        aligned[name] = {c: link[c].to_numpy(dtype=np.float64) for c in E2_LINK}

    for name, numbers in aligned.items():
        if (numbers["P"] < 0).any() or (numbers["P"] > 1).any():
            raise ValueError(f"{name}: P is outside [0, 1]")
        if (numbers["u"] < 0).any():
            raise ValueError(f"{name}: u is negative")
    if not np.array_equal(aligned["full"]["page_missing"], 1 - rows["page_fetched"].astype(int).to_numpy()):
        raise ValueError("full: page_missing disagrees with the dataset's page_fetched")
    if not (aligned["blank"]["page_missing"] == 1).all():
        raise ValueError("blank: page_missing must be 1 on every row")

    return PartLink(message_id=ids.to_numpy(), label=(rows["label"] == "smishing").to_numpy().astype(int),
                    full=aligned["full"], blank=aligned["blank"])


def link_inputs(seed: int, fold: int, pilot: bool = False) -> Dict[str, PartLink]:
    """{train, val, test} -> PartLink for one seed and fold. Train rows are out-of-fold; val and test use the final forest."""
    parts = load_fold(seed, fold, pilot=pilot)
    lane = ll.load_lane(seed, fold, pilot=pilot)
    return {part: align(parts[part], *lane[part]) for part in PARTS}


def summary_rows(seed: int, fold: int, inputs: Dict[str, PartLink]) -> List[dict]:
    return [{"seed": seed, "fold": fold, "part": part, "n": len(p.message_id), "smishing": round(float(p.label.mean()), 3),
             "page_missing": round(float(p.full["page_missing"].mean()), 3),
             "P_full": round(float(p.full["P"].mean()), 3), "P_blank": round(float(p.blank["P"].mean()), 3),
             "u_full": round(float(p.full["u"].mean()), 3)} for part, p in inputs.items()]


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a pipeline test only)")
    args = ap.parse_args(argv)
    rows, failures = [], []
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            try:
                rows += summary_rows(seed, fold, link_inputs(seed, fold, pilot=args.pilot))
            except Exception as error:
                failures.append((seed, fold, f"{type(error).__name__}: {error}"))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    done = len(SEEDS) * CROWD_N_FOLDS - len(failures)
    print(f"\n{done} of {len(SEEDS) * CROWD_N_FOLDS} seed x fold cells aligned and checked")
    for seed, fold, reason in failures:
        print(f"  seed {seed} fold {fold}: {reason}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
