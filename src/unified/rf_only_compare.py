"""RF-only baseline: the final Link RF's P alone (>= 0.5), no text model, scored on the same test messages as film_compare.

It reads the forest's saved test-row scores (link_lane final_*.csv) and the labels and message ids already written by
film_compare (compare/test_predictions.csv), so it needs no model checkpoint. Nothing is trained or tuned: the forest is the one
the Link RF lane already fitted, with the same 0.5 cut-off the other models use. Two conditions as in film_compare: `complete`
(page as inspected) and `page hidden` (the forest's score with the page blanked, page_missing = 1).

Writes experiments/film/<dataset>/compare/rf_only_{predictions,metrics,summary,mcnemar}.csv. McNemar (exact, complete pages) is RF-only
against every other model already in test_predictions.csv, Holm-corrected per seed over those pairs only, so film_compare's own
tables are left untouched.

Usage:
    python -m src.unified.rf_only_compare --pilot
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import CROWD_N_FOLDS, SEEDS
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc
from src.unified.film_compare import RF_ONLY, holm, mcnemar, metrics_tables


def rf_only_predictions(pilot: bool, reference: pd.DataFrame) -> pd.DataFrame:
    """RF-only rows for every (seed, fold) in `reference` (film_compare's test_predictions), checked against its ids and labels."""
    lane_dir = ll.out_dir(tc.dataset_path(pilot))
    rows = []
    ref = reference.drop_duplicates(["seed", "fold", "message_id"])[["seed", "fold", "message_id", "label"]]
    for (seed, fold), expected in ref.groupby(["seed", "fold"]):
        for condition, prefix in (("complete", "final"), ("page hidden", "final_blank")):
            scored = pd.read_csv(lane_dir / f"{prefix}_seed{seed}_fold{fold}.csv")
            test = scored[scored["part"] == "test"][["message_id", "P"]]
            merged = expected.merge(test, on="message_id", how="outer", validate="one_to_one", indicator=True)
            assert (merged["_merge"] == "both").all(), f"seed {seed} fold {fold} {condition}: forest rows and test rows differ"
            rows.append(pd.DataFrame({"seed": seed, "fold": fold, "message_id": merged["message_id"], "label": merged["label"],
                                      "model": RF_ONLY, "condition": condition, "p_smishing": merged["P"],
                                      "pred": (merged["P"] >= 0.5).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def mcnemar_vs_rf(reference: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([reference[reference["condition"] == "complete"], rf[rf["condition"] == "complete"]])
    rows = []
    for seed, g in both.groupby("seed"):
        wide = {m: gm.sort_values(["fold", "message_id"]) for m, gm in g.groupby("model")}
        others = [m for m in wide if m != RF_ONLY]
        results = [mcnemar(wide[RF_ONLY]["label"].to_numpy(), wide[m]["pred"].to_numpy(), wide[RF_ONLY]["pred"].to_numpy()) for m in others]
        adjusted = holm([r["p_exact"] for r in results])
        for m, r, p in zip(others, results, adjusted):     # b = the other model right and RF-only wrong
            rows.append({"seed": seed, "model_a": m, "model_b": RF_ONLY, **r, "p_holm": p, "significant_0.05": p < 0.05})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args(argv)
    out = fm.out_dir(tc.dataset_path(args.pilot), "compare")
    reference = pd.read_csv(out / "test_predictions.csv")
    rf = rf_only_predictions(args.pilot, reference)
    per_seed, summary = metrics_tables(rf)
    mc = mcnemar_vs_rf(reference, rf)
    rf.to_csv(out / "rf_only_predictions.csv", index=False)
    per_seed.to_csv(out / "rf_only_metrics.csv", index=False)
    summary.to_csv(out / "rf_only_summary.csv", index=False)
    mc.to_csv(out / "rf_only_mcnemar.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print(per_seed.round(4).to_string(index=False))
    print(mc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
