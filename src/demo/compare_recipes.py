"""Which inputs should the head get besides the text? Compare the three recipes on the demo data.

    link_pu  14 raw link features + has_url + P + u     (results reused from compare_pu_training, mixed mode)
    pu_only  has_url + P + u                              (link features only inside the RF)
    rf_vec   has_url + P + u + 31 per-feature RF contributions

All three use the same CNN, the same mixed P/u training and the same synthetic data; only the recipe
differs. PROTOTYPE ONLY: synthetic data, so this checks how the designs behave, not real performance.

Usage (from the project root, after src.demo.synthetic):
    python -m src.demo.compare_recipes
"""
import argparse
import json
import subprocess
import sys

import pandas as pd

from src.demo.synthetic import ROOT

RECIPES = ["pu_only", "rf_vec"]
OUT_ROOT = ROOT / "models" / "demo_experiments"
REPORT_DIR = ROOT / "experiments" / "demo"


def run(recipe: str, seed: int) -> dict:
    out_dir = OUT_ROOT / f"{recipe}_s{seed}"
    subprocess.run([sys.executable, "-m", "src.demo.train_demo", "--recipe", recipe, "--pu-training", "mixed",
                    "--train-seed", str(seed), "--out-dir", str(out_dir), "--gate-repeats", "5"],
                   check=True, stdout=subprocess.DEVNULL)
    return json.loads((out_dir / "meta.json").read_text())["test_metrics"]["fusion"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    rows = []
    for recipe in RECIPES:
        for seed in args.seeds:
            for group, m in run(recipe, seed).items():
                rows.append({"recipe": recipe, "seed": seed, "group": group, "n": m["n"], "accuracy": m["accuracy"]})
            print(f"done {recipe} seed {seed}", flush=True)
    earlier = pd.read_csv(REPORT_DIR / "pu_training_per_seed.csv")
    earlier = earlier[(earlier["mode"] == "mixed") & earlier["seed"].isin(args.seeds)]
    rows += [{"recipe": "link_pu", "seed": r.seed, "group": r.group, "n": r.n, "accuracy": r.accuracy}
             for r in earlier.itertuples()]
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["group", "recipe"])["accuracy"].agg(["mean", "std"]).round(4).unstack("recipe")
    per_seed.to_csv(REPORT_DIR / "recipes_per_seed.csv", index=False)
    summary.to_csv(REPORT_DIR / "recipes_summary.csv")
    pd.set_option("display.width", 200)
    print(summary.to_string())


if __name__ == "__main__":
    main()
