r"""Re-record the dataset fingerprint in the CNN-only sidecars after a change to columns that no model reads.

    .venv\Scripts\python.exe scripts/restamp_dataset_hash.py --old D:/LeJetData/dataset_backups/lejet_final_dataset_before_category_fix_20261122.csv

Why: the CNN-only sidecars (models/film/<dataset>/cnn_raw_*.json) hold the sha256 of the whole dataset file, so changing only the
category / categories_all columns (2026-11-22 category fix) makes raw_cnn_control.load_raw refuse the models. Retraining would give
the same weights but rewrite the recorded training times. This script instead:
  1. checks that the OLD file is exactly the file every sidecar was trained on (its sha256 equals the recorded one);
  2. checks that the old and the current file are identical in every column except the allowed ones (default category, categories_all),
     same rows in the same order;
  3. writes the current file's sha256 into each sidecar and records the change under "dataset_restamped" (date, reason, old sha256,
     columns allowed to differ).
Afterwards run `python -m src.unified.raw_cnn_control check`: every model must reproduce its recorded validation metrics exactly.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.unified import fit_model as fm                                  # noqa: E402
from src.unified import text_cnn as tc                                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", required=True, help="the dataset file the models were trained on (a backup copy)")
    ap.add_argument("--allowed", nargs="+", default=["category", "categories_all"], help="columns allowed to differ")
    ap.add_argument("--reason", default="2026-11-22 category fix: only category / categories_all changed (no model reads them)")
    a = ap.parse_args()
    dataset = tc.dataset_path(False)
    old_sha, new_sha = fm.sha256(Path(a.old)), fm.sha256(dataset)
    old = pd.read_csv(a.old, keep_default_na=False, dtype=str)
    new = pd.read_csv(dataset, keep_default_na=False, dtype=str)
    keep = [c for c in old.columns if c not in a.allowed]
    if list(old.columns) != list(new.columns) or not old[keep].equals(new[keep]):
        raise SystemExit("the files differ outside the allowed columns: retrain instead")
    sidecars = sorted(fm.model_dir(dataset).glob("cnn_raw_seed*_fold*.json"))
    if not sidecars:
        raise SystemExit("no CNN-only sidecars found")
    for path in sidecars:
        side = json.loads(path.read_text(encoding="utf-8"))
        recorded = side["sha256"]["dataset.csv"]
        if recorded == new_sha:
            print(f"{path.name}: already the current file")
            continue
        if recorded != old_sha:
            raise SystemExit(f"{path.name} was not trained on {a.old} (recorded {recorded[:12]}..., old file {old_sha[:12]}...)")
        side["sha256"]["dataset.csv"] = new_sha
        side.setdefault("dataset_restamped", []).append({"date": datetime.date.today().isoformat(), "reason": a.reason,
                                                         "old_sha256": old_sha, "columns_allowed_to_differ": a.allowed})
        path.write_text(json.dumps(side, indent=2, default=float), encoding="utf-8")
        print(f"{path.name}: restamped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
