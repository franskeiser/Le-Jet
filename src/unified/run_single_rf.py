"""One forest for every link versus two dedicated forests, on pages seen complete and with the page columns blanked.

Why: docs/rf_link_expert_architecture.md uses ONE forest that reads the 14 URL features and the 17 page
features, with the page columns NaN when the page was not reached. That only works if the forest, trained
on both cases, stays close to a forest built for each case. Three forests per seed, tuned by grouped CV on
the Training link rows (urf.tune):
    single    trained on every Training row twice: complete, and with the 17 page columns set to NaN;
    full      the 31 inputs, complete rows only;
    url_only  the 14 URL inputs only.
Each is scored on the same Test rows in two conditions: `complete` (all 31 features present) and `blanked`
(the 17 page columns NaN). full is not run on blanked rows (it was not built for them); it is scored there
too, to show what happens without the single forest's training, because it must read NaN anyway.
url_only ignores the page columns, so its two conditions are identical.

Same data, splits and seeds as run_features_only.py. Writes <out>/single_rf/{metrics_per_seed,metrics_summary}.csv.

Usage:
    python -m src.unified.run_single_rf --data data/processed/web_phishing_linked.csv --splits-dir data/splits_web_phishing --out experiments/web_phishing
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SEEDS
from src.features.model.rf_extractor import fit_rf
from src.features.model.run_experiment import score_group
from src.unified import unified_rf as urf
from src.unified.run_unified import load_part

FULL = urf.INPUTS["full"]
URL = urf.INPUTS["url_only"]
PAGE = [c for c in FULL if c not in URL]


def blank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[PAGE] = np.nan
    return out


def run_seed(df: pd.DataFrame, seed: int, args) -> tuple[list[dict], dict]:
    train = load_part(df, seed, "train", args.splits_dir)
    test = load_part(df, seed, "test", args.splits_dir)
    train = train[urf.link_mask(train)].reset_index(drop=True)
    test = test[urf.link_mask(test)].reset_index(drop=True)
    y_test = urf.labels(test)

    both = pd.concat([train, blank(train)], ignore_index=True)    # duplicates keep their domain, so CV folds stay grouped
    forests = {"single": (both, FULL), "full": (train, FULL), "url_only": (train, URL)}
    conditions = {"complete": test, "blanked": blank(test)}

    rows, params_used = [], {}
    for name, (data, inputs) in forests.items():
        params, _ = urf.tune(data, inputs, seed)
        rf = fit_rf(data[inputs], urf.labels(data), seed, **params)
        params_used[name] = {k: str(v) for k, v in params.items()}
        for cond, frame in conditions.items():
            pred = rf.predict(frame[inputs])
            rows.append({"seed": seed, "forest": name, "condition": cond, **score_group(y_test, pred)})
        print(f"seed {seed}: {name} done")
    return rows, {"seed": seed, "params": params_used, "n_train": len(train), "n_test": len(test)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    df = pd.read_csv(args.data, low_memory=False, keep_default_na=False, na_values=[""])
    rows, infos = [], []
    for seed in args.seeds:
        r, info = run_seed(df, seed, args)
        rows += r
        infos.append(info)

    out = args.out / "single_rf"
    out.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "metrics_per_seed.csv", index=False)
    metrics = ["accuracy", "precision", "recall", "f1"]
    summary = per_seed.groupby(["condition", "forest"])[metrics + ["n"]].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "metrics_summary.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "runs": infos}, indent=2))

    table = summary[["condition", "forest"]].copy()
    for c in metrics:
        table[c] = summary[f"{c}_mean"].round(4).astype(str) + " +-" + summary[f"{c}_std"].round(4).astype(str)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
