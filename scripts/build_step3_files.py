r"""Step 3 of the experimental testing guide: the checks behind Cross_Validation_Splits.xlsx.

    .venv\Scripts\python.exe scripts/build_step3_files.py [--data data/final/lejet_final_dataset.csv]
                                                          [--splits data/splits_crowd/lejet_final_dataset]
                                                          [--out experiments/film/lejet_final_dataset/data_prep/Cross_Validation_Checks.xlsx]

The folds themselves are made by src/data/split_crowd.py (Stratified Group 5-fold, seeds 42/123/2024, grouping by
link_domain + template_id + dup_group joined into one group, 15% grouped validation taken from the training part).
This script does NOT make new folds. It reads the saved seed<S>_fold<F>.csv files, re-runs the split in memory to prove
the saved files are reproducible, runs the guide's "before proceeding" checks, and writes the workbook.

The per-message assignments (the guide's Step 3 file) are Cross_Validation_Splits.xlsx in the same folder, written by
scripts/report_vocabularies.py. This workbook does not repeat them, so there is one copy of the folds.
Sheets: Summary (the guide's seed table), Folds (sizes, smishing share, message conditions per part), Checks, Notes.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_N_FOLDS, CROWD_VAL_FRACTION, SEEDS   # noqa: E402
from src.data.split_crowd import PARTS, fold_parts, joined_groups   # noqa: E402


def condition(t: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(t.has_url == 0, "no_url", np.where(t.page_fetched == 1, "accessible", "inaccessible")), index=t.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "final" / "lejet_final_dataset.csv"))
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits_crowd" / "lejet_final_dataset"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "film" / "lejet_final_dataset" / "data_prep" / "Cross_Validation_Checks.xlsx"))
    a = ap.parse_args()
    t = pd.read_csv(a.data, keep_default_na=False)
    sha = hashlib.sha256(open(a.data, "rb").read()).hexdigest()
    groups = joined_groups(t)
    cond = condition(t)
    ids = set(t.message_id)

    assign = pd.DataFrame({"message_id": t.message_id, "label": t.label, "message_condition": cond,
                           "link_domain": t.link_domain, "template_id": t.template_id, "dup_group": t.dup_group,
                           "cv_group": pd.factorize(groups)[0] + 1})
    fold_rows, checks = [], {}
    reproducible, ids_ok, once_ok, group_ok, val_ok = True, True, True, True, True
    for s in SEEDS:
        recomputed = fold_parts(t.label, groups, s, CROWD_N_FOLDS, CROWD_VAL_FRACTION)
        test_fold = np.full(len(t), -1)
        for f in range(CROWD_N_FOLDS):
            p = pd.read_csv(Path(a.splits) / f"seed{s}_fold{f}.csv")
            ids_ok &= len(p) == len(t) and set(p.message_id) == ids
            p = t[["message_id"]].merge(p, on="message_id", how="left", validate="one_to_one")
            part = p.part.to_numpy()
            reproducible &= bool((part == recomputed[f]).all())
            assign[f"seed{s}_fold{f}"] = part
            test_fold[part == "test"] = f
            once_ok &= bool(((part == "test") & (test_fold != f)).sum() == 0)
            for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
                group_ok &= not (set(groups[part == x]) & set(groups[part == y]))
            val_ok &= bool((part == "val").sum() > 0)
            row = {"seed": s, "fold": f}
            for q in PARTS:
                m = part == q
                row[f"n_{q}"] = int(m.sum())
                row[f"smishing_{q}"] = int((t.label[m] == "smishing").sum())
                row[f"legitimate_{q}"] = int((t.label[m] == "legitimate").sum())
                row[f"smishing_share_{q}"] = round(float((t.label[m] == "smishing").mean()), 4)
            for c in ("no_url", "inaccessible", "accessible"):
                row[f"test_{c}"] = int(((part == "test") & (cond == c)).sum())
            tr = part == "train"
            te = part == "test"
            row["test_rows_sharing_domain_with_train"] = int(t.link_domain[te].isin(set(t.link_domain[tr]) - {"-1", -1}).sum())
            row["test_rows_sharing_template_with_train"] = int(t.template_id[te].isin(set(t.template_id[tr])).sum())
            row["test_rows_sharing_dup_group_with_train"] = int(t.dup_group[te].isin(set(t.dup_group[tr])).sum())
            fold_rows.append(row)
        once_ok &= bool((test_fold >= 0).all())
        assign[f"test_fold_seed{s}"] = test_fold
    folds = pd.DataFrame(fold_rows)

    checks["every split file holds every message_id exactly once"] = ids_ok
    checks["every message is in exactly one test fold per seed"] = once_ok
    checks["no group (domain / template / near-duplicate) is in two parts of a run"] = group_ok
    checks["no test row shares a domain, template or near-duplicate group with its training part"] = bool(
        (folds[[c for c in folds if c.startswith("test_rows_sharing")]] == 0).all().all())
    checks["validation part exists and comes from the training folds (never the test fold)"] = val_ok
    checks["saved split files equal a fresh re-run of src/data/split_crowd.py"] = reproducible
    checks["15 runs (3 seeds x 5 folds)"] = len(folds) == len(SEEDS) * CROWD_N_FOLDS
    share = folds[[f"smishing_share_{q}" for q in PARTS]]
    overall = float((t.label == "smishing").mean())
    checks[f"smishing share of every part within 1 point of the dataset's {overall:.4f}"] = bool(((share - overall).abs() <= 0.01).all().all())

    summary = pd.DataFrame({"Random Seed": [str(s) for s in SEEDS] + ["Total"],
                            "Number of Folds": [CROWD_N_FOLDS] * len(SEEDS) + [CROWD_N_FOLDS * len(SEEDS)]})
    chk = pd.DataFrame({"check": list(checks), "result": ["PASS" if v else "FAIL" for v in checks.values()]})
    gsize = groups.value_counts()
    notes = pd.DataFrame({"item": [
        "dataset file", "dataset sha256", "rows", "smishing / legitimate", "method", "seeds", "grouping rule",
        "number of groups", "largest group (rows)", "groups of one message", "validation part",
        "test part use", "split code", "built"], "value": [
        Path(a.data).name, sha, len(t), f"{int((t.label == 'smishing').sum())} / {int((t.label == 'legitimate').sum())}",
        "sklearn StratifiedGroupKFold, 5 splits, shuffle=True, random_state=seed",
        ", ".join(map(str, SEEDS)),
        "messages sharing a link_domain, a template_id or a dup_group (>= 90% same wording once links and digits are ignored) "
        "are joined into one group (union-find); a group never sits in two parts. Messages without a link are not grouped by domain.",
        int(gsize.size), int(gsize.max()), int((gsize == 1).sum()),
        f"one of {round(1 / CROWD_VAL_FRACTION)} grouped, stratified inner folds of the 4 training folds (about {CROWD_VAL_FRACTION:.0%}); "
        "used for early stopping and model selection only",
        "held out: never used for training, early stopping, thresholds or model selection",
        "src/data/split_crowd.py (folds), scripts/build_step3_files.py (this workbook)",
        datetime.datetime.now().isoformat(timespec="seconds")]})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        for name, frame in [("Summary", summary), ("Folds", folds), ("Checks", chk), ("Notes", notes)]:
            frame.to_excel(w, sheet_name=name, index=False)
    print(chk.to_string(index=False))
    print(folds[["seed", "fold", "n_train", "n_val", "n_test", "smishing_share_test", "test_no_url", "test_inaccessible", "test_accessible"]].to_string(index=False))
    print(f"wrote {out}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
