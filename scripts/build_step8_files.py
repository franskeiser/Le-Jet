r"""Step 8 of the experimental testing guide: performance of the four main models by message condition.

    .venv\Scripts\python.exe scripts/build_step8_files.py

Reads only the Step 5 prediction files; nothing is trained or scored again. Conditions (from the dataset, the same for every model):
    No URL                 has_url = 0                                  (888 messages: 75 smishing, 813 legitimate)
    Inaccessible webpage   has_url = 1, page did not load               (2,204: 1,201 smishing, 1,003 legitimate)
    Accessible webpage     has_url = 1, page loaded                     (3,623: 1,716 smishing, 1,907 legitimate)
Per model, condition and seed: N, TP / FN / FP / TN, Accuracy, Precision, Recall, F1 (smishing = positive; the guide's formulas).
Aggregation method: the metrics are computed per seed on that condition's messages (each message tested once per seed), then
mean and sample SD (n - 1) over the three seeds. N is the same in every seed (every message is in every seed's predictions).

Writes experiments/film/<dataset>/step8/Message_Condition_Analysis.xlsx:
    Summary        the guide's table per model: Condition | Number of Messages | Accuracy | Precision | Recall | F1 (mean ± SD)
    Per_Seed       every model x condition x seed, with the confusion counts
    By_Model_Wide  accuracy and F1 side by side, model x condition (for reading across models)
    Method         conditions, aggregation, source
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import SEEDS                                    # noqa: E402
from src.unified import fit_model as fm                         # noqa: E402
from src.unified import text_cnn as tc                          # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from build_step7_files import MODELS, METRICS, counts, metrics  # noqa: E402

CONDITIONS = [("no URL", "No URL"), ("inaccessible webpage", "Inaccessible webpage"), ("accessible webpage", "Accessible webpage")]


def main() -> int:
    dataset = tc.dataset_path(False)
    step5 = fm.out_dir(dataset, "step5")
    rows = []
    for model, stem in MODELS:
        t = pd.read_csv(step5 / f"{stem}_Predictions.csv")
        for key, name in CONDITIONS:
            for seed in SEEDS:
                s = t[(t.Seed == seed) & (t["Message Condition"] == key)]
                c = counts(s["Actual Label"], s["Predicted Label"])
                rows.append({"Model": model, "Message Condition": name, "Seed": seed, "Number of Messages": len(s),
                             "Smishing": int((s["Actual Label"] == "smishing").sum()), **c, **metrics(c)})
    per_seed = pd.DataFrame(rows)
    n_msgs = len(pd.read_csv(step5 / f"{MODELS[0][1]}_Predictions.csv").query("Seed == @SEEDS[0]"))        # every message once per seed (6,714 on the current dataset)
    assert (per_seed.groupby(["Model", "Seed"])["Number of Messages"].sum() == n_msgs).all(), "the conditions do not cover every message"
    agg = per_seed.groupby(["Model", "Message Condition"], sort=False).agg(
        **{"Number of Messages": ("Number of Messages", "first"), "Smishing": ("Smishing", "first")},
        **{f"{m}_mean": (m, "mean") for m in METRICS}, **{f"{m}_sd": (m, "std") for m in METRICS}).reset_index()
    summary = agg[["Model", "Message Condition", "Number of Messages", "Smishing"]].copy()
    for m in METRICS:
        summary[m] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(agg[f"{m}_mean"], agg[f"{m}_sd"])]
    wide = agg.pivot(index="Model", columns="Message Condition", values=["Accuracy_mean", "F1-score_mean"])
    wide.columns = [f"{c} | {m.replace('_mean', '')}" for m, c in wide.columns]
    wide = wide[[f"{c} | {m}" for m in ("Accuracy", "F1-score") for _, c in CONDITIONS]].reindex([m for m, _ in MODELS]).reset_index()
    method = pd.DataFrame({"item": ["No URL", "Inaccessible webpage", "Accessible webpage", "metrics", "aggregation", "source"], "value": [
        "has_url = 0 (all 37 URL features -1; E2 bypasses FiLM)", "has_url = 1, page did not load (13 page features -1, page_missing = 1)",
        "has_url = 1, page loaded (all 37 features)", "Accuracy, Precision, Recall, F1 with smishing as the positive class (Step 7 formulas)",
        "per seed on the condition's messages, then mean and sample SD (n - 1) over seeds 42, 123, 2024",
        f"experiments/film/{dataset.stem}/step5/<Model>_Predictions.csv, column Message Condition"]})

    out = fm.out_dir(dataset, "step8")
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "Message_Condition_Analysis.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        agg.to_excel(w, sheet_name="Summary_Numeric", index=False)
        per_seed.to_excel(w, sheet_name="Per_Seed", index=False)
        wide.to_excel(w, sheet_name="By_Model_Wide", index=False)
        method.to_excel(w, sheet_name="Method", index=False)
    pd.set_option("display.width", 230)
    print(summary.to_string(index=False))
    pooled = per_seed.groupby(["Message Condition", "Model"], sort=False)[["TP", "FN", "FP", "TN"]].sum()
    print(pooled.to_string())
    print(f"wrote {out / 'Message_Condition_Analysis.xlsx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
