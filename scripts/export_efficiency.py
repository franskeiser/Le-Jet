"""Export the SOP 2 measurement to the file the frontend reads (frontend/src/data/efficiencyResults.json).

Reads, for one dataset, what src/unified/efficiency.py and the SOP 1 comparison wrote under experiments/film/<dataset>/:
    efficiency/form_e_summary.csv       model size, peak memory, time per message: mean and SD over the seeds (Form E)
    efficiency/form_d_runtime.csv       the number of test messages timed (Form D)
    efficiency/form_a_environment.json  the computer and library versions (Form A)
    efficiency/descriptive_details.csv  parameter counts, forest trees and nodes
    compare/{test,rf_only,voting}_summary.csv   F1 on the test messages, complete pages (SOP 1)
and writes one JSON with nothing typed by hand. "status" is "dry_run" for a synthetic pilot (the numbers test the measuring tool and are
not study results) and "final" otherwise; --status overrides. "live_model_ids" maps a model id the backend serves to the measured model
that stands behind it; it is left empty (so the page never shows the study's size and memory for a different, older model) and is kept
as it is when the file is exported again.

Usage:
    python scripts/export_efficiency.py --dataset crowd_relevant_synthetic
    python scripts/export_efficiency.py --dataset crowd_smishing_v1 --status final
    python scripts/export_efficiency.py --dataset crowd_relevant_synthetic --out some/other/file.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.unified.efficiency import MODELS, RUNS  # noqa: E402

DEFAULT_OUT = ROOT / "frontend" / "src" / "data" / "efficiencyResults.json"

# the display order: the models the instrument names first, the extra rows, and the proposed model last (as the Research page does)
ORDER = ["cnn_only", "rf_only", "voting", "raw_cnn", "concat", "proposed_contrib", "proposed"]

# efficiency model key -> the model's name in the comparison tables (SOP 1). proposed_contrib has the FiLM model's verdicts: it only computes more.
F1_NAMES = {
    "proposed": "E2 (from scratch)", "proposed_contrib": "E2 (from scratch)", "cnn_only": "CNN-only (baseline)",
    "concat": "Concatenation (from scratch)", "raw_cnn": "CNN-only, text + URL features (from scratch)",
    "rf_only": "RF-only (Link RF)", "voting": "KNN-RF-ETC voting (text only)",
}


def number(value):
    """A float, or None for a missing / NaN value (JSON has no NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    return float(value)


def mean_sd(summary: pd.DataFrame, key: str, column: str):
    mean = number(summary.loc[key, f"{column}_mean"]) if f"{column}_mean" in summary.columns else None
    if mean is None:
        return None
    return {"mean": round(mean, 4), "sd": None if number(summary.loc[key, f"{column}_std"]) is None else round(number(summary.loc[key, f"{column}_std"]), 4)}


def f1_table(compare: Path) -> dict:
    """model name -> {"mean", "sd"} of F1 on complete pages, from whichever summary files exist."""
    out = {}
    for name in ("test_summary.csv", "rf_only_summary.csv", "voting_summary.csv"):
        path = compare / name
        if not path.exists():
            continue
        table = pd.read_csv(path)
        for _, row in table[table["condition"] == "complete"].iterrows():
            out[row["model"]] = {"mean": round(number(row["f1_mean"]), 4), "sd": None if number(row["f1_std"]) is None else round(number(row["f1_std"]), 4)}
    return out


def build(dataset: str, status: str | None = None, previous: dict | None = None, root: Path = ROOT) -> dict:
    base = root / "experiments" / "film" / dataset
    eff, compare = base / "efficiency", base / "compare"
    summary = pd.read_csv(eff / "form_e_summary.csv", index_col=0)
    runtime = pd.read_csv(eff / "form_d_runtime.csv")
    environment = json.loads((eff / "form_a_environment.json").read_text(encoding="utf-8")) if (eff / "form_a_environment.json").exists() else {}
    details = pd.read_csv(eff / "descriptive_details.csv").drop_duplicates("model").set_index("model") if (eff / "descriptive_details.csv").exists() else pd.DataFrame()
    f1 = f1_table(compare)
    models = {}
    for key in [k for k in ORDER if k in summary.index]:
        spec = MODELS[key]
        row = details.loc[key] if key in details.index else None
        models[key] = {
            "label": spec["label"], "extra": bool(spec["extra"]), "proposed": key == "proposed",
            "size_mb": mean_sd(summary, key, "size_mb"), "peak_memory_mb": mean_sd(summary, key, "peak_memory_mb"),
            "runtime_ms": {"all": mean_sd(summary, key, "mean_ms"), "with_link": mean_sd(summary, key, "mean_link_ms"),
                           "no_link": mean_sd(summary, key, "mean_nolink_ms")},
            "f1": f1.get(F1_NAMES.get(key)),
            "parameters_trainable": None if row is None or number(row.get("parameters_trainable_without_embedding")) is None else int(row["parameters_trainable_without_embedding"]),
            "forest_trees": None if row is None or number(row.get("forest_trees")) is None else int(row["forest_trees"]),
        }
    resolved = status or ("dry_run" if "synthetic" in dataset else "final")
    return {
        "status": resolved, "dataset": dataset, "fold": int(runtime["fold"].iloc[0]) if "fold" in runtime else 0,
        "seeds": sorted(int(s) for s in runtime["seed"].unique()), "runs_per_seed": int(runtime.groupby(["model", "seed"])["run"].nunique().max()),
        "n_test_messages": int(runtime["n"].max()), "measured_at": environment.get("measured_at"),
        "runs_completed": int(len(runtime)), "runs_planned": int(runtime["model"].nunique() * runtime["seed"].nunique() * len(RUNS)),
        "environment": {k: environment.get(k) for k in ("cpu", "cores_logical", "ram_gb", "os", "python", "torch", "scikit_learn")},
        "order": list(models), "models": models,
        "live_model_ids": (previous or {}).get("live_model_ids", {}),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="the dataset's name, e.g. crowd_relevant_synthetic")
    ap.add_argument("--status", choices=["dry_run", "final"], default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--root", type=Path, default=ROOT, help="the project folder that holds experiments/ (default: this one)")
    ap.add_argument("--live-model", action="append", default=[], metavar="BACKEND_ID=MEASURED_ID",
                    help="say that the model the backend serves under BACKEND_ID is the measured model MEASURED_ID (for example "
                         "film_cnn=proposed); repeatable. Only for a served model that really is the measured one (same files)")
    args = ap.parse_args(argv)
    previous = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else None
    data = build(args.dataset, args.status, previous, args.root)
    for pair in args.live_model:
        backend_id, _, measured = pair.partition("=")
        if not backend_id or measured not in data["models"]:
            ap.error(f"--live-model {pair!r}: the measured model must be one of {sorted(data['models'])}")
        data["live_model_ids"][backend_id] = measured
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(data['models'])} models, status {data['status']})")


if __name__ == "__main__":
    main()
