r"""Step 10 of the experimental testing guide: McNemar's test, the Proposed Model against each of the three baselines.

    .venv\Scripts\python.exe scripts/build_step10_files.py

Reads only the Step 5 prediction files (experiments/film/<dataset>/step5/*_Predictions.csv); nothing is trained or scored again.
For every baseline and every seed (3 comparisons x 3 seeds = nine tests), the same message is paired across the two models
(the same seed puts it in the same fold, so both models predicted it out of fold):
    b = messages the Proposed Model got right and the baseline got wrong
    c = messages the Proposed Model got wrong and the baseline got right
    chi-square = (|b - c| - 1)^2 / (b + c)        (continuity-corrected, 1 degree of freedom; 0 when b + c = 0)
    p-value    = upper tail of the chi-square distribution with 1 df
    decision   = p < 0.05: reject H0 (the two models' error patterns differ); otherwise fail to reject H0
Cross-check: the exact two-sided binomial p-value of b out of b + c at 0.5 is added (not used for the decision).
Multiple comparisons: the guide leaves the adjustment to the adviser (item 39). The uncorrected decision is the guide's; Holm and
Bonferroni adjusted p-values over the nine tests are reported beside it, so the adviser's answer only picks the column.

Writes experiments/film/<dataset>/step10/McNemar_Test_Results.xlsx:
    Summary         the nine tests: comparison, seed, b, c, chi-square, p-value, decision, which model has more correct disagreements
    Adjusted        the same tests with Bonferroni and Holm adjusted p-values and decisions
    Agreement       for each test the four cells of the paired table (both right, both wrong, b, c) and the accuracy of each model
    Method          formulas, alpha, source files
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import SEEDS                                    # noqa: E402
from src.unified import fit_model as fm                         # noqa: E402
from src.unified import text_cnn as tc                          # noqa: E402

PROPOSED = ("Proposed Model", "Proposed_Model")
BASELINES = [("CNN-only", "CNN_only"), ("Concatenation Fusion", "Concatenation"), ("KNN-RF-ETC", "KNN_RF_ETC")]   # the guide's order
ALPHA = 0.05


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """continuity-corrected chi-square and its p-value (1 df); no disagreements -> no evidence of a difference"""
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stat), float(chi2.sf(stat, 1))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> int:
    dataset = tc.dataset_path(False)
    step5 = fm.out_dir(dataset, "step5")
    prop = pd.read_csv(step5 / f"{PROPOSED[1]}_Predictions.csv")
    rows, cells = [], []
    for base, stem in BASELINES:
        other = pd.read_csv(step5 / f"{stem}_Predictions.csv")
        for seed in SEEDS:
            a = prop[prop.Seed == seed].set_index("Message ID")
            o = other[other.Seed == seed].set_index("Message ID")
            assert a.index.is_unique and o.index.is_unique, f"{base} seed {seed}: a message appears twice"
            assert set(a.index) == set(o.index), f"{base} seed {seed}: the two models were not tested on the same messages"
            o = o.loc[a.index]
            assert (a["Actual Label"] == o["Actual Label"]).all(), f"{base} seed {seed}: the true labels differ between the files"
            ok_p = (a["Predicted Label"] == a["Actual Label"]).to_numpy()
            ok_b = (o["Predicted Label"] == o["Actual Label"]).to_numpy()
            both_right, both_wrong = int((ok_p & ok_b).sum()), int((~ok_p & ~ok_b).sum())
            b, c = int((ok_p & ~ok_b).sum()), int((~ok_p & ok_b).sum())
            assert both_right + both_wrong + b + c == len(a)
            stat, p = mcnemar(b, c)
            exact = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
            more = "Proposed Model" if b > c else base if c > b else "equal"
            name = f"Proposed Model vs. {base}"
            rows.append({"Comparison": name, "Seed": seed, "N (paired messages)": len(a), "b (Proposed right, baseline wrong)": b,
                         "c (Proposed wrong, baseline right)": c, "Chi-square": stat, "p-value": p,
                         "Decision (alpha = 0.05)": "Reject H0" if p < ALPHA else "Fail to reject H0",
                         "More correct among disagreements": more, "Exact binomial p (cross-check)": exact})
            cells.append({"Comparison": name, "Seed": seed, "Both correct": both_right, "Both wrong": both_wrong,
                          "b": b, "c": c, "Proposed Model accuracy": ok_p.mean(), f"{base} accuracy": ok_b.mean()})
    summary = pd.DataFrame(rows)
    assert len(summary) == 9, "the guide asks for nine tests"
    p = summary["p-value"].to_numpy()
    adjusted = summary[["Comparison", "Seed", "p-value"]].rename(columns={"p-value": "p-value (uncorrected)"})
    adjusted["p-value (Bonferroni, 9 tests)"] = np.minimum(1.0, p * len(p))
    adjusted["p-value (Holm, 9 tests)"] = holm(p)
    for col, label in [("p-value (uncorrected)", "Decision (uncorrected)"), ("p-value (Bonferroni, 9 tests)", "Decision (Bonferroni)"),
                       ("p-value (Holm, 9 tests)", "Decision (Holm)")]:
        adjusted[label] = np.where(adjusted[col] < ALPHA, "Reject H0", "Fail to reject H0")
    method = pd.DataFrame({"item": ["comparisons", "pairing", "b", "c", "statistic", "p-value", "decision", "tests",
                                    "adjustment", "cross-check", "source"], "value": [
        "Proposed Model vs. CNN-only, vs. Concatenation Fusion, vs. KNN-RF-ETC",
        "same message, same seed (same fold): both models predicted it out of fold; true labels asserted identical",
        "messages the Proposed Model classified correctly and the baseline incorrectly",
        "messages the Proposed Model classified incorrectly and the baseline correctly",
        "chi-square = (|b - c| - 1)^2 / (b + c), continuity-corrected, 1 degree of freedom (0 when b + c = 0)",
        "upper tail of the chi-square distribution with 1 degree of freedom",
        "p < 0.05 reject H0, otherwise fail to reject H0 (alpha = 0.05)",
        "3 baselines x 3 seeds (42, 123, 2024) = 9",
        "not decided (guide item 39, adviser): the Summary sheet is uncorrected; Bonferroni and Holm over the 9 tests are on sheet Adjusted",
        "exact two-sided binomial test of b out of b + c at 0.5 (not used for the decision)",
        f"experiments/film/{dataset.stem}/step5/<Model>_Predictions.csv"]})

    out = fm.out_dir(dataset, "step10")
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "McNemar_Test_Results.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        adjusted.to_excel(w, sheet_name="Adjusted", index=False)
        pd.DataFrame(cells).to_excel(w, sheet_name="Agreement", index=False)
        method.to_excel(w, sheet_name="Method", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summary.round(4).to_string(index=False))
    print(adjusted.round(4).to_string(index=False))
    print(f"wrote {out / 'McNemar_Test_Results.xlsx'}")
    # the experiment papers (Appendix 1, 2, 4, 5): a new Markdown copy whenever the Step 5 predictions changed (one per retraining)
    from build_experiment_papers import main as experiment_papers
    experiment_papers([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
