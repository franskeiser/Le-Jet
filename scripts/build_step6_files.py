r"""Step 6 of the experimental testing guide: the four component checks of the proposed model (E2), on the TEST folds.

    .venv\Scripts\python.exe scripts/build_step6_files.py --read-test

Same cross-validation framework as Steps 4-5 (Stratified Group 5-fold x seeds 42, 123, 2024; every message tested once per seed). Nothing is
trained: the checks change what the TRAINED E2 of each run receives at test time, except the text-only CNN, a separately trained model.

  Proposed model (reference)  E2 as in Step 5: the page as inspected. Asserted identical to step5/Proposed_Model_Predictions.csv.
  Text-only CNN               a CNN trained on the text alone (no link, no RF, no FiLM): src/unified/cnn_baseline.py, Step 4 checkpoints,
                              predictions from step5/TextOnly_CNN_Predictions.csv. Same embedding, Conv, head, recipe and folds as E2.
  FiLM neutralised            E2 with gamma = 1 and beta = 0 for EVERY message (every message is passed as has_url = 0, so the generator
                              is bypassed; asserted: gamma == 1 and beta == 0 for all messages). The text path is E2's own trained one.
  Shuffled RF outputs         within each test fold, the messages that have a link trade their Link RF numbers (P, u, page_missing) at
                              random; has_url stays with its message, no-link messages are untouched. 5 permutations per fold, seeded
                              1000*seed + 10*fold + k (film_controls.shuffle_link); metrics are averaged over the 5 per seed.
  Webpage features hidden     the 13 page features set to -1 and page_missing = 1 before the Link RF scores the message (the 4 outside
                              features and the 20 URL features stay), i.e. link_lane.blank_page; the Link RF's page-hidden numbers.

Metrics: accuracy, precision, recall, F1 (smishing = positive), per seed over the 6,715 messages, then mean and SD over the 3 seeds.
Writes experiments/film/<dataset>/step6/Component_Check_Results.xlsx (Summary, Per_Seed, By_Condition, Configuration) and
Component_Check_Predictions.csv (every message, check, seed; the shuffled check with its permutation number).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_N_FOLDS, SEEDS                                # noqa: E402
from src.unified import film                                               # noqa: E402
from src.unified import film_compare as fc                                 # noqa: E402
from src.unified import fit_model as fm                                    # noqa: E402
from src.unified import text_cnn as tc                                     # noqa: E402
from src.unified.film_controls import shuffle_link                         # noqa: E402
from src.unified.film_data import part_inputs                              # noqa: E402

SHUFFLES = 5
REF, TEXT, NEUTRAL, SHUFFLED, HIDDEN = ("Proposed model (reference)", "Text-only CNN", "FiLM neutralised", "Shuffled RF outputs",
                                        "Webpage features hidden")
ORDER = [REF, TEXT, NEUTRAL, SHUFFLED, HIDDEN]
WHAT = {REF: "nothing (E2 as in Step 5)", TEXT: "no URL conditioning at all: a CNN trained on text only",
        NEUTRAL: "gamma = 1, beta = 0 for every message (FiLM switched off in the trained E2)",
        SHUFFLED: "each linked message gets another linked message's RF outputs (5 permutations per fold)",
        HIDDEN: "13 webpage features hidden (-1, page_missing = 1) before the Link RF"}
PURPOSE = {REF: "the reference", TEXT: "text-based performance", NEUTRAL: "effect of disabling FiLM",
           SHUFFLED: "whether the right URL-message pairing matters", HIDDEN: "contribution of the webpage features"}
METRICS = ["accuracy", "precision", "recall", "f1"]


def probs(model, token_ids: torch.Tensor, link: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(token_ids, link), dim=1)[:, 1].numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-test", action="store_true", help="required: this reads the test folds")
    a = ap.parse_args()
    if not a.read_test:
        print("Step 6 reads the test folds: pass --read-test.")
        return 2
    dataset = tc.dataset_path(False)
    step5 = fm.out_dir(dataset, "step5")
    fc.log_read(dataset, [f"step6: E2 {REF}, {NEUTRAL}, {SHUFFLED}, {HIDDEN}"], SEEDS, list(range(CROWD_N_FOLDS)), True)
    cond = pd.read_csv(step5 / "Proposed_Model_Predictions.csv").drop_duplicates("Message ID").set_index("Message ID")["Message Condition"]

    rows = []
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            model, w2i, seq_len, _ = film.load_film(seed, fold, False, warm_start=False)
            t = part_inputs(seed, fold, "test", w2i, seq_len)
            meta = {"seed": seed, "fold": fold, "message_id": t.message_id, "label": t.labels.numpy()}
            off = t.link_full.clone()
            off[:, -1] = 0                                                  # has_url = 0 for every message -> generator bypassed
            gamma, beta = model.film_parameters(off, model.conv_features(t.token_ids)[1])
            assert bool((gamma == 1).all() and (beta == 0).all()), "FiLM neutralised: gamma / beta are not 1 / 0"
            for check, link in ((REF, t.link_full), (NEUTRAL, off), (HIDDEN, t.link_blank)):
                p = probs(model, t.token_ids, link)
                rows.append(pd.DataFrame({**meta, "check": check, "permutation": "", "p_smishing": p, "pred": (p >= 0.5).astype(int)}))
            for k in range(SHUFFLES):
                g = torch.Generator().manual_seed(1000 * seed + 10 * fold + k)
                p = probs(model, t.token_ids, shuffle_link(t.link_full, g))
                rows.append(pd.DataFrame({**meta, "check": SHUFFLED, "permutation": k, "p_smishing": p, "pred": (p >= 0.5).astype(int)}))
            print(f"seed {seed} fold {fold} done", flush=True)
    pred = pd.concat(rows, ignore_index=True)

    # the reference must equal Step 5's predictions exactly
    s5 = pd.read_csv(step5 / "Proposed_Model_Predictions.csv")
    ref = pred[pred.check == REF].merge(s5, left_on=["seed", "message_id"], right_on=["Seed", "Message ID"])
    assert len(ref) == len(s5) and np.allclose(ref.p_smishing, ref["Predicted Probability"], atol=1e-6), "reference differs from Step 5"
    # the text-only CNN from Step 5
    tx = pd.read_csv(step5 / "TextOnly_CNN_Predictions.csv")
    pred = pd.concat([pred, pd.DataFrame({"seed": tx.Seed, "fold": tx.Fold, "message_id": tx["Message ID"],
                                          "label": (tx["Actual Label"] == "smishing").astype(int), "check": TEXT, "permutation": "",
                                          "p_smishing": tx["Predicted Probability"],
                                          "pred": (tx["Predicted Label"] == "smishing").astype(int)})], ignore_index=True)
    pred["message_condition"] = cond.loc[pred.message_id].to_numpy()
    for (check, seed, perm), g in pred.groupby(["check", "seed", "permutation"]):
        assert g.message_id.is_unique and len(g) == len(cond), f"{check} seed {seed}: not every message exactly once"

    def metric_rows(frame: pd.DataFrame, keys: list) -> pd.DataFrame:
        out = []
        for key, g in frame.groupby(keys + ["permutation"], sort=False):
            out.append({**dict(zip(keys + ["permutation"], key)), "n": len(g), **fc.scores(g.label.to_numpy(), g.pred.to_numpy())})
        m = pd.DataFrame(out)
        return m.groupby(keys, sort=False)[["n"] + METRICS].mean().reset_index()      # the shuffled check: mean of its 5 permutations

    per_seed = metric_rows(pred, ["check", "seed"])
    per_seed["check"] = pd.Categorical(per_seed.check, ORDER)
    per_seed = per_seed.sort_values(["check", "seed"])
    summary = per_seed.groupby("check", observed=True)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    ref_row = summary[summary.check == REF].iloc[0]
    for m in METRICS:
        summary[f"{m}_change_vs_reference"] = summary[f"{m}_mean"] - ref_row[f"{m}_mean"]
    for m in METRICS:
        summary[f"{m} (mean ± SD)"] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    by_cond = metric_rows(pred, ["check", "message_condition", "seed"]).groupby(["check", "message_condition"], sort=False)[
        ["n"] + METRICS].mean().reset_index()
    by_cond["check"] = pd.Categorical(by_cond.check, ORDER)
    by_cond = by_cond.sort_values(["message_condition", "check"])
    config = pd.DataFrame({"component check": ORDER, "what changes": [WHAT[c] for c in ORDER], "purpose": [PURPOSE[c] for c in ORDER],
                           "trained models used": ["E2, Step 4", "text-only CNN, Step 4", "E2, Step 4", "E2, Step 4", "E2, Step 4"],
                           "framework": "Stratified Group 5-fold x seeds 42, 123, 2024; each message tested once per seed (Step 3)"})

    out = fm.out_dir(dataset, "step6")
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "Component_Check_Predictions.csv", index=False)
    with pd.ExcelWriter(out / "Component_Check_Results.xlsx", engine="openpyxl") as w:
        summary[["check"] + [f"{m} (mean ± SD)" for m in METRICS] + [f"{m}_change_vs_reference" for m in METRICS]
                + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]].to_excel(w, sheet_name="Summary", index=False)
        per_seed.to_excel(w, sheet_name="Per_Seed", index=False)
        by_cond.to_excel(w, sheet_name="By_Condition", index=False)
        config.to_excel(w, sheet_name="Configuration", index=False)
    pd.set_option("display.width", 220)
    print(summary[["check"] + [f"{m} (mean ± SD)" for m in METRICS]].to_string(index=False))
    print((summary.set_index("check")[[f"{m}_change_vs_reference" for m in METRICS]] * 100).round(2).to_string())
    print(by_cond.pivot(index="check", columns="message_condition", values="accuracy").round(4).to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
