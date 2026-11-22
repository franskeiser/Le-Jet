r"""Step 7 of the experimental testing guide: confusion matrices and classification metrics of the four main models.

    .venv\Scripts\python.exe scripts/build_step7_files.py

Reads only the Step 5 prediction files (experiments/film/<dataset>/step5/*_Predictions.csv); nothing is trained or scored again.
Positive class = Smishing. Per model and seed (one out-of-fold test prediction per message, 6,714 on the current dataset):
    confusion matrix  TP (smishing -> smishing), FN (smishing -> legitimate), FP (legitimate -> smishing), TN (legitimate -> legitimate)
    Accuracy  = (TP + TN) / (TP + TN + FP + FN)      Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)                       F1-score  = 2 x Precision x Recall / (Precision + Recall)
computed from the counts with the guide's formulas (cross-checked against scikit-learn). Then mean and SD over the three seeds
(sample SD, n - 1 = 2). The pooled matrix (the three seeds' counts added) is given for display only; the metrics are per seed.

Writes experiments/film/<dataset>/step7/Classification_Performance_Results.xlsx:
    Summary              the guide's table: Model | Accuracy | Precision | Recall | F1-score as "mean ± SD"
    Per_Seed             the four metrics for every model and seed (the individual seed results)
    Confusion_Matrices   TP / FN / FP / TN for every model and seed, plus the pooled matrix
    Matrices_Layout      each model's matrix in the guide's 2 x 2 layout (per seed and pooled)
    Method               formulas, positive class, SD definition, source files
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import SEEDS                                    # noqa: E402
from src.unified import fit_model as fm                         # noqa: E402
from src.unified import text_cnn as tc                          # noqa: E402

MODELS = [("CNN-only", "CNN_only"), ("Concatenation Fusion", "Concatenation"), ("KNN-RF-ETC", "KNN_RF_ETC"),
          ("Proposed Model", "Proposed_Model")]                 # the guide's order
METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]


def counts(actual: pd.Series, predicted: pd.Series) -> dict:
    a, p = actual == "smishing", predicted == "smishing"
    return {"TP": int((a & p).sum()), "FN": int((a & ~p).sum()), "FP": int((~a & p).sum()), "TN": int((~a & ~p).sum())}


def metrics(c: dict) -> dict:
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"Accuracy": (tp + tn) / (tp + tn + fp + fn), "Precision": precision, "Recall": recall,
            "F1-score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def main() -> int:
    dataset = tc.dataset_path(False)
    step5 = fm.out_dir(dataset, "step5")
    per_seed, matrices, layout = [], [], []
    for model, stem in MODELS:
        t = pd.read_csv(step5 / f"{stem}_Predictions.csv")
        pooled = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        for seed in SEEDS:
            s = t[t.Seed == seed]
            assert len(s) == s["Message ID"].nunique(), f"{model} seed {seed}: a message appears twice"
            c = counts(s["Actual Label"], s["Predicted Label"])
            m = metrics(c)
            y, yhat = (s["Actual Label"] == "smishing").astype(int), (s["Predicted Label"] == "smishing").astype(int)
            sk = [accuracy_score(y, yhat), precision_score(y, yhat), recall_score(y, yhat), f1_score(y, yhat)]
            assert np.allclose([m[k] for k in METRICS], sk), f"{model} seed {seed}: formula and scikit-learn disagree"
            per_seed.append({"Model": model, "Seed": seed, "N": len(s), **m})
            matrices.append({"Model": model, "Seed": str(seed), "N": len(s), **c})
            for k in pooled:
                pooled[k] += c[k]
            layout += [{"Model": model, "Seed": str(seed), "Actual / Predicted": "Smishing", "Smishing": c["TP"], "Legitimate": c["FN"]},
                       {"Model": model, "Seed": str(seed), "Actual / Predicted": "Legitimate", "Smishing": c["FP"], "Legitimate": c["TN"]}]
        matrices.append({"Model": model, "Seed": "pooled (3 seeds)", "N": sum(pooled.values()), **pooled})
        layout += [{"Model": model, "Seed": "pooled (3 seeds)", "Actual / Predicted": "Smishing", "Smishing": pooled["TP"], "Legitimate": pooled["FN"]},
                   {"Model": model, "Seed": "pooled (3 seeds)", "Actual / Predicted": "Legitimate", "Smishing": pooled["FP"], "Legitimate": pooled["TN"]}]
    per_seed = pd.DataFrame(per_seed)
    agg = per_seed.groupby("Model", sort=False)[METRICS].agg(["mean", "std"])
    summary = pd.DataFrame({"Model": agg.index})
    for m in METRICS:
        summary[m] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(agg[(m, "mean")], agg[(m, "std")])]
    numeric = agg.copy()
    numeric.columns = [f"{m}_{s}" for m, s in numeric.columns]
    method = pd.DataFrame({"item": ["positive class", "Accuracy", "Precision", "Recall", "F1-score", "per seed", "mean ± SD",
                                    "pooled matrix", "source", "framework"], "value": [
        "Smishing", "(TP + TN) / (TP + TN + FP + FN)", "TP / (TP + FP)", "TP / (TP + FN)", "2 x Precision x Recall / (Precision + Recall)",
        f"each seed's {int(per_seed['N'].iloc[0]):,} out-of-fold test predictions (every message tested once per seed)",
        "mean and sample SD (n - 1) over seeds 42, 123, 2024",
        "the three seeds' counts added; shown for the figure only, the metrics are per seed",
        f"experiments/film/{dataset.stem}/step5/<Model>_Predictions.csv", "Stratified Group 5-fold x 3 seeds (Step 3)"]})

    out = fm.out_dir(dataset, "step7")
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "Classification_Performance_Results.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        numeric.reset_index().to_excel(w, sheet_name="Summary_Numeric", index=False)
        per_seed.to_excel(w, sheet_name="Per_Seed", index=False)
        pd.DataFrame(matrices).to_excel(w, sheet_name="Confusion_Matrices", index=False)
        pd.DataFrame(layout).to_excel(w, sheet_name="Matrices_Layout", index=False)
        method.to_excel(w, sheet_name="Method", index=False)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))
    print(per_seed.round(4).to_string(index=False))
    print(pd.DataFrame(matrices).to_string(index=False))
    print(f"wrote {out / 'Classification_Performance_Results.xlsx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
