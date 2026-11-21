"""Time the Link RF two ways, per message (plan docs/film_architecture_plan.md, step 1.3).

    with_contributions  link_lane.score(): P, u AND the 37 per-feature contributions (the pilot lane's path)
    p_u_only            rf_extractor.p_and_u(): P and u alone (E2 reads only these, plus page_missing)

Both run on the same messages, one message at a time (batch 1), on CPU with one thread (rf.n_jobs = 1 and the numeric
libraries limited to 1 thread), after `warmup` messages that are not counted. For each message the two variants are timed
back to back, and which one goes first alternates, so a slow moment or a warm cache does not favour one of them.
The two must return the same P and u (asserted).

What is timed is the forest's scoring of one row of the 37 link/page features. The web inspection and the feature
extraction come before it and are the same for both, so they are not timed here.

Writes to experiments/film/<dataset>/link_rf_timing/:
    per_message.csv   seed, fold, message_id, n_trees, ms for each variant
    summary.csv       per seed and fold, and pooled: n, median / 95th percentile / mean ms for each variant, and the ratio
    forests.csv       per seed and fold: trees, total nodes, file size, forest settings
    machine.json      CPU, cores, OS, Python and library versions (the numbers mean nothing without them)

Usage:
    python -m src.unified.time_link_rf --pilot                       # fold 0 of each seed, 200 timed messages each
    python -m src.unified.time_link_rf --pilot --folds 0 1 2 3 4     # every fold
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # before numpy is imported: one thread, as the app runs the forest

import argparse
import json
import platform
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_ROOT, CROWD_N_FOLDS, SEEDS
from src.data.split_crowd import load_fold
from src.features.model.rf_extractor import p_and_u
from src.unified import link_lane as ll
from src.unified.prepare_web_37 import FEATURES_37
from src.unified.text_cnn import dataset_path

VARIANTS = ["with_contributions", "p_u_only"]


def out_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "experiments" / "film" / dataset.stem / "link_rf_timing"


def time_variants(rf, rows: pd.DataFrame, n: int, warmup: int, seed: int = 0) -> pd.DataFrame:
    """Per-message milliseconds of both variants on `n` timed messages (after `warmup` untimed ones) of `rows`."""
    if len(rows) < n + warmup:
        raise ValueError(f"need {n + warmup} messages, have {len(rows)}")
    rf.n_jobs = 1
    order = np.random.default_rng(seed).permutation(len(rows))[: n + warmup]
    messages = [rows.iloc[[i]] for i in order]                     # one-row frames, built before any timing

    def with_contributions(one):
        return ll.score(rf, one)[["P", "u"]].to_numpy()[0]

    def p_u_only(one):
        p, u = p_and_u(rf, one[FEATURES_37])
        return np.array([p[0], u[0]])

    functions = {"with_contributions": with_contributions, "p_u_only": p_u_only}
    out = []
    for position, one in enumerate(messages):
        timed, values = {}, {}
        for name in (VARIANTS if position % 2 == 0 else VARIANTS[::-1]):      # alternate who goes first
            start = time.perf_counter()
            values[name] = functions[name](one)
            timed[name] = (time.perf_counter() - start) * 1000.0
        if not np.allclose(values["with_contributions"], values["p_u_only"], atol=1e-9):
            raise AssertionError(f"P and u differ between the two paths for {one['message_id'].iloc[0]}")
        if position >= warmup:
            out.append({"message_id": one["message_id"].iloc[0], **{f"ms_{k}": timed[k] for k in VARIANTS}})
    return pd.DataFrame(out)


def summarize(frame: pd.DataFrame) -> dict:
    row = {"n": len(frame)}
    for name in VARIANTS:
        ms = frame[f"ms_{name}"].to_numpy()
        row.update({f"{name}_median": float(np.median(ms)), f"{name}_p95": float(np.percentile(ms, 95)),
                    f"{name}_mean": float(ms.mean())})
    row["median_ratio"] = row["with_contributions_median"] / row["p_u_only_median"]
    return row


def machine_info() -> dict:
    import sklearn

    info = {"platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return info


def run(pilot: bool, seeds: List[int], folds: List[int], n: int, warmup: int) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    out = out_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    per_message, forests, summaries = [], [], []
    for seed in seeds:
        for fold in folds:
            path = ll.model_dir(dataset) / f"link_rf_seed{seed}_fold{fold}.joblib"
            bundle = joblib.load(path)
            rf = bundle["model"]
            assert bundle["features"] == list(FEATURES_37)
            test = load_fold(seed, fold, pilot=pilot)["test"]
            timed = time_variants(rf, test, n, warmup, seed=seed)
            timed.insert(0, "fold", fold)
            timed.insert(0, "seed", seed)
            timed["n_trees"] = len(rf.estimators_)
            per_message.append(timed)
            forests.append({"seed": seed, "fold": fold, "trees": len(rf.estimators_),
                            "nodes": int(sum(t.tree_.node_count for t in rf.estimators_)),
                            "file_mb": round(path.stat().st_size / 2**20, 1), **{k: str(v) for k, v in bundle["params"].items()}})
            summaries.append({"seed": seed, "fold": fold, **summarize(timed)})
            s = summaries[-1]
            print(f"seed {seed} fold {fold}: {len(rf.estimators_)} trees | with contributions median "
                  f"{s['with_contributions_median']:.1f} ms | P and u only median {s['p_u_only_median']:.2f} ms | "
                  f"ratio {s['median_ratio']:.0f}x", flush=True)

    everything = pd.concat(per_message, ignore_index=True)
    everything.to_csv(out / "per_message.csv", index=False)
    pooled = pd.DataFrame(summaries + [{"seed": "all", "fold": "all", **summarize(everything)}])
    pooled.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(forests).to_csv(out / "forests.csv", index=False)
    (out / "machine.json").write_text(json.dumps({**machine_info(), "n_timed": n, "warmup": warmup, "batch_size": 1}, indent=2))
    return pooled


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a pipeline test only)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=[0], help=f"folds to time (0-{CROWD_N_FOLDS - 1})")
    ap.add_argument("--n", type=int, default=200, help="timed messages per forest")
    ap.add_argument("--warmup", type=int, default=50, help="messages run first and not counted")
    args = ap.parse_args(argv)
    pooled = run(args.pilot, args.seeds, args.folds, args.n, args.warmup)
    pd.set_option("display.width", 220)
    print("\n" + pooled.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
