r"""Phase A2-A4 of docs/validation_only_improvement_plan.md, on the VALIDATION predictions only (scripts/validation_predictions.py).

    .venv\Scripts\python.exe scripts/validation_study_analysis.py

Decision rules, written before the results were looked at (thresholds are judgement calls, stated here so they are not chosen afterwards):
  H1 (cut-off too low): supported if the Proposed model's best validation cut-off is above 0.5 in at least 11 of the 15 seed-fold runs
     AND its mean best cut-off is higher than Concatenation's and CNN-only's.
     Best cut-off = highest balanced accuracy on that fold's validation rows over 0.30, 0.32, ..., 0.70; ties go to the value nearest 0.5.
  H2 (Link RF over-scores big-brand pages): supported if, among legitimate validation messages with a link whose page rank is 10, the
     Link RF's own score rf_P is >= 0.5 for at least 20% of them.
  H3 (text-style shortcut): "short" = 3 words or fewer once the link is removed. Supported if the false-alarm rate gap
     (Proposed minus CNN-only, on legitimate validation messages) is at least 5 percentage points larger in the short group than in the longer group.
Everything here uses `val` rows only. Output: experiments/film/<dataset>/validation_study/{A2_cutoffs.csv, A3_link_rf.csv, A4_short_messages.csv, summary.txt}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.unified import fit_model as fm                                  # noqa: E402
from src.unified import text_cnn as tc                                   # noqa: E402
from src.unified import voting_baseline as vb                            # noqa: E402

GRID = np.round(np.arange(0.30, 0.7001, 0.02), 2)
NEURAL = ["Proposed Model", "CNN-only", "Concatenation Fusion"]
ORDER = NEURAL + ["KNN-RF-ETC"]


def best_cutoff(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    scores = np.array([balanced_accuracy_score(y, (p >= t).astype(int)) for t in GRID])
    top = np.flatnonzero(scores == scores.max())
    i = top[np.argmin(np.abs(GRID[top] - 0.5))]
    return float(GRID[i]), float(scores[i])


def main() -> int:
    dataset = tc.dataset_path(False)
    out = fm.out_dir(dataset, "validation_study")
    v = pd.read_csv(out / "validation_predictions.csv")
    d = pd.read_csv(dataset, keep_default_na=False, dtype=str).set_index("message_id")
    v = v.join(d[["source", "page_rank", "link_domain", "sms_text"]], on="message_id")
    v["pred05"] = (v.p_smishing >= 0.5).astype(int)
    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    # ---------------------------------------------------------------- A2
    rows = []
    for (m, seed, fold), g in v.groupby(["model", "seed", "fold"]):
        y, p = g.label.to_numpy(), g.p_smishing.to_numpy()
        t, b = best_cutoff(y, p)
        pr = (p >= 0.5).astype(int)
        rows.append({"model": m, "seed": seed, "fold": fold, "n_val": len(g), "false_alarms_at_0.5": int(((y == 0) & (pr == 1)).sum()),
                     "misses_at_0.5": int(((y == 1) & (pr == 0)).sum()), "bal_acc_at_0.5": balanced_accuracy_score(y, pr),
                     "best_cutoff": t, "bal_acc_at_best": b})
    a2 = pd.DataFrame(rows)
    a2.to_csv(out / "A2_cutoffs.csv", index=False)
    agg = a2.groupby("model").agg(false_alarms=("false_alarms_at_0.5", "mean"), misses=("misses_at_0.5", "mean"),
                                  bal_acc_05=("bal_acc_at_0.5", "mean"), mean_best_cutoff=("best_cutoff", "mean"),
                                  folds_best_above_05=("best_cutoff", lambda s: int((s > 0.5).sum())),
                                  folds_best_below_05=("best_cutoff", lambda s: int((s < 0.5).sum())),
                                  bal_acc_best=("bal_acc_at_best", "mean")).loc[ORDER]
    say("A2  CUT-OFF (validation rows, 15 seed-fold runs; 767 messages each)")
    say(agg.round(4).to_string())
    say("    (bal_acc_best is measured on the same rows the cut-off was picked on, so it is optimistic; KNN-RF-ETC's p is not a calibrated probability)")
    pm = a2[a2.model == "Proposed Model"].best_cutoff
    h1 = (pm > 0.5).sum() >= 11 and pm.mean() > a2[a2.model == "Concatenation Fusion"].best_cutoff.mean() and pm.mean() > a2[a2.model == "CNN-only"].best_cutoff.mean()
    say(f"    H1 rule: Proposed best cut-off > 0.5 in {(pm > 0.5).sum()}/15 runs, mean {pm.mean():.3f}  ->  H1 {'SUPPORTED' if h1 else 'NOT supported'}")

    # ---------------------------------------------------------------- A3
    leg = v[(v.label == 0) & (v.has_url == 1) & v.model.isin(NEURAL)].copy()
    pr = pd.to_numeric(leg.page_rank, errors="coerce")
    leg["page_rank_group"] = np.where(pr == 10, "page rank 10", np.where(pr >= 7, "page rank 7-9", "page rank 0-6 or unknown"))
    rf = leg[leg.model == "Proposed Model"]
    rfs = rf.groupby("page_rank_group").agg(n_rows=("message_id", "size"), rf_P_mean=("rf_P", "mean"), rf_flags_share=("rf_P", lambda s: (s >= 0.5).mean()))
    fa = leg.assign(fa=leg.pred05).pivot_table(index="page_rank_group", columns="model", values="fa", aggfunc="mean")[NEURAL]
    a3 = rfs.join(fa.add_prefix("false_alarm_rate_"))
    a3_src = rf.groupby("source").agg(n_rows=("message_id", "size"), rf_P_mean=("rf_P", "mean"), rf_flags_share=("rf_P", lambda s: (s >= 0.5).mean())).join(
        leg.assign(fa=leg.pred05).pivot_table(index="source", columns="model", values="fa", aggfunc="mean")[NEURAL].add_prefix("false_alarm_rate_"))
    a3.to_csv(out / "A3_link_rf.csv")
    say("\nA3  LINK RF on LEGITIMATE validation messages that have a link (rows summed over 15 runs)")
    say(a3.round(3).to_string())
    say("    by source:")
    say(a3_src.round(3).to_string())
    share10 = a3.loc["page rank 10", "rf_flags_share"] if "page rank 10" in a3.index else float("nan")
    say(f"    H2 rule: RF flags (P >= 0.5) {share10:.1%} of legitimate page-rank-10 messages (need >= 20%)  ->  H2 {'SUPPORTED' if share10 >= 0.20 else 'NOT supported'}")

    # ---------------------------------------------------------------- A4
    v["words_without_link"] = v.sms_text.map(lambda t: len(vb.strip_links(t).split()))
    v["group"] = np.where(v.words_without_link <= 3, "short (<= 3 words without the link)", "longer")
    l4 = v[(v.label == 0) & v.model.isin(NEURAL)]
    a4 = l4.assign(fa=l4.pred05).pivot_table(index="group", columns="model", values="fa", aggfunc="mean")[NEURAL]
    a4.insert(0, "n_rows_proposed", l4[l4.model == "Proposed Model"].groupby("group").size())
    a4.to_csv(out / "A4_short_messages.csv")
    say("\nA4  FALSE-ALARM RATE on legitimate validation messages, by message length")
    say(a4.round(4).to_string())
    gap = a4["Proposed Model"] - a4["CNN-only"]
    short = [i for i in gap.index if i.startswith("short")][0]
    diff = (gap[short] - gap["longer"]) * 100
    say(f"    H3 rule: gap Proposed - CNN-only is {gap[short]*100:.1f} pts for short, {gap['longer']*100:.1f} pts for longer; difference {diff:.1f} pts (need >= 5)  ->  H3 {'SUPPORTED' if diff >= 5 else 'NOT supported'}")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
