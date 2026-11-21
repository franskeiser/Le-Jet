r"""Step 5 of the experimental testing guide: out-of-fold predictions on the TEST folds, one row per message, model, seed.

    .venv\Scripts\python.exe scripts/build_step5_files.py --read-test [--dataset lejet_final_dataset]

This is the step that reads the test folds. Each test fold is scored by the models of its own run (trained in Step 4 on the other
folds), so with 5 folds every message is predicted exactly once per seed by models that never saw it or its group. Nothing is trained
or tuned here; metrics are Step 7 and McNemar's test is Step 10 (this script computes neither).

Models: the four of Step 4 (Proposed E2, CNN-only, Concatenation, KNN-RF-ETC) and the text-only CNN (Step 6's component check).
Predictions use the saved models: the neural ones through src/unified/film_compare.test_predictions (the code the comparison was tested
with), KNN-RF-ETC through its saved .joblib (src/unified/voting_baseline.py). Threshold 0.5 on P(smishing) for the neural models; the
routed classifier's own verdict for KNN-RF-ETC.

Two conditions are scored for the models that read the page: `complete` (the page as inspected: the Step 5 predictions) and
`page_hidden` (13 page features blanked, page_missing = 1: kept for Step 6's "webpage features hidden" check, in its own file).

Writes experiments/film/<dataset>/step5/:
  <Model>_Predictions.csv          the guide's columns, complete pages (one file per model, as the guide lists)
  Predictions.xlsx                 one sheet per model + Checks
  PageHidden_Predictions.csv       all models, page hidden (for Step 6)
and appends a line to experiments/film/<dataset>/compare/test_reads.log.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_N_FOLDS, SEEDS                              # noqa: E402
from src.data.split_crowd import load_fold                               # noqa: E402
from src.unified import film_compare as fc                               # noqa: E402
from src.unified import fit_model as fm                                  # noqa: E402
from src.unified import text_cnn as tc                                   # noqa: E402
from src.unified import voting_baseline as vb                            # noqa: E402

# film_compare name -> (guide name, file stem)
NEURAL = {fc.E2_SCRATCH: ("Proposed Model", "Proposed_Model"), fc.RAW_CNN: ("CNN-only", "CNN_only"),
          fc.CONTROL_SCRATCH: ("Concatenation Fusion", "Concatenation"), fc.BASELINE: ("Text-only CNN (Step 6 check)", "TextOnly_CNN")}
KNN = ("KNN-RF-ETC", "KNN_RF_ETC")
LABEL = {1: "smishing", 0: "legitimate"}
COLUMNS = ["Message ID", "Actual Label", "Predicted Label", "Predicted Probability", "Model", "Seed", "Fold", "Message Condition"]


def conditions_of(dataset_csv: Path) -> pd.Series:
    d = pd.read_csv(dataset_csv, keep_default_na=False)
    cond = np.where(d.has_url == 0, "no URL", np.where(d.page_fetched == 1, "accessible webpage", "inaccessible webpage"))
    return pd.Series(cond, index=d.message_id)


def guide_format(frame: pd.DataFrame, model: str, cond: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Message ID": frame["message_id"].to_numpy(), "Actual Label": frame["label"].map(LABEL).to_numpy(),
                         "Predicted Label": frame["pred"].map(LABEL).to_numpy(), "Predicted Probability": frame["p_smishing"].to_numpy(),
                         "Model": model, "Seed": frame["seed"].to_numpy(), "Fold": frame["fold"].to_numpy(),
                         "Message Condition": cond.loc[frame["message_id"]].to_numpy()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lejet_final_dataset")
    ap.add_argument("--read-test", action="store_true", help="required: this reads the test folds")
    a = ap.parse_args()
    if not a.read_test:
        print("Step 5 reads the test folds. Run it once everything is fixed, with --read-test.")
        return 2
    dataset = tc.dataset_path(False)
    assert dataset.stem == a.dataset
    folds = list(range(CROWD_N_FOLDS))
    fc.log_read(dataset, list(NEURAL) + [vb.VOTING], SEEDS, folds, True)

    neural = fc.test_predictions(False, SEEDS, folds, list(NEURAL))            # asserts: each message once per seed, same ids for all
    knn = vb.voting_predictions(False, SEEDS)                                  # the saved Step 4 models
    cond = conditions_of(dataset)
    ids = set(cond.index)

    out = fm.out_dir(dataset, "step5")
    out.mkdir(parents=True, exist_ok=True)
    tables, hidden, checks = {}, [], []
    for name, (guide, stem) in list(NEURAL.items()) + [(vb.VOTING, KNN)]:
        src = knn if name == vb.VOTING else neural[neural["model"] == name]
        complete = src[src["condition"] == "complete"]
        t = guide_format(complete, guide, cond)
        t.to_csv(out / f"{stem}_Predictions.csv", index=False)
        tables[guide] = t
        if name != vb.VOTING and name != fc.BASELINE:
            hidden.append(guide_format(src[src["condition"] == "page hidden"], guide, cond))
        for seed in SEEDS:
            s = t[t["Seed"] == seed]
            fold_ok = True
            for fold in folds:
                test_ids = set(load_fold(seed, fold)["test"]["message_id"])
                fold_ok &= set(s.loc[s["Fold"] == fold, "Message ID"]) == test_ids
            checks.append({"model": guide, "seed": seed, "rows": len(s), "every message exactly once": bool(s["Message ID"].is_unique and
                           set(s["Message ID"]) == ids), "fold matches Step 3": bool(fold_ok),
                           "probability in [0, 1]": bool(s["Predicted Probability"].between(0, 1).all()),
                           "no missing value": bool(s.notna().all().all())})
    check = pd.DataFrame(checks)
    same = {g: tuple(sorted(zip(t["Seed"], t["Message ID"]))) for g, t in tables.items()}
    check["same messages as every other model"] = len(set(same.values())) == 1
    flags = [c for c in check.columns if c not in ("model", "seed", "rows")]
    check["all passed"] = check[flags].all(axis=1)
    pd.concat(hidden, ignore_index=True).to_csv(out / "PageHidden_Predictions.csv", index=False)
    with pd.ExcelWriter(out / "Predictions.xlsx", engine="openpyxl") as w:
        check.to_excel(w, sheet_name="Checks", index=False)
        for guide, t in tables.items():
            t.to_excel(w, sheet_name=guide.replace("(", "").replace(")", "")[:31], index=False)
    pd.set_option("display.width", 220)
    print(check.to_string(index=False))
    print(f"wrote {len(tables)} prediction files, PageHidden_Predictions.csv and Predictions.xlsx to {out}")
    return 0 if check["all passed"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
