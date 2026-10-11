"""Does training the head on a mix of P and u sources fix the missing-page mismatch?

Trains the demo fusion model three ways, each over several training seeds, and scores all of
them the same way: on test rows with P and u routed as in live serving (`full` where the page was
reached, `url_only` otherwise).

    url_only  the head sees only link-spelling P and u in training (all the old SMS data can give)
    routed    the head sees `full` P and u wherever a page was reached
    mixed     each read randomly uses `full` or `url_only` P and u (modality dropout)

PROTOTYPE ONLY: synthetic data, so the numbers show whether the training change does what it is
meant to, not how the system performs on real messages.

Usage (from the project root, after src.demo.synthetic):
    python -m src.demo.compare_pu_training
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.demo.synthetic import DEMO_MODEL_DIR, ROOT

MODES = ["url_only", "routed", "mixed"]
OUT_ROOT = ROOT / "models" / "demo_experiments"
REPORT_DIR = ROOT / "experiments" / "demo"


def run(mode: str, seed: int, hide_prob: float) -> dict:
    out_dir = OUT_ROOT / f"{mode}_s{seed}"
    subprocess.run(
        [sys.executable, "-m", "src.demo.train_demo", "--pu-training", mode, "--train-seed", str(seed),
         "--hide-prob", str(hide_prob), "--out-dir", str(out_dir), "--gate-repeats", "5"],
        check=True, stdout=subprocess.DEVNULL)
    return json.loads((out_dir / "meta.json").read_text())["test_metrics"]["fusion"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--hide-prob", type=float, default=0.5)
    args = parser.parse_args()

    rows = []
    for mode in MODES:
        for seed in args.seeds:
            for group, m in run(mode, seed, args.hide_prob).items():
                rows.append({"mode": mode, "seed": seed, "group": group, "n": m["n"], "accuracy": m["accuracy"]})
            print(f"done {mode} seed {seed}", flush=True)
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["group", "mode"])["accuracy"].agg(["mean", "std"]).round(4).unstack("mode")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(REPORT_DIR / "pu_training_per_seed.csv", index=False)
    summary.to_csv(REPORT_DIR / "pu_training_summary.csv")
    pd.set_option("display.width", 200)
    print(summary.to_string())


if __name__ == "__main__":
    main()
