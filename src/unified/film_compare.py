"""Compare the E2 models with their baselines on the test folds (plan docs/film_architecture_plan.md, Phase 4: RQ1 and RQ3).

Models (the first is required, the others are scored when their checkpoints exist):
    CNN-only (baseline)        the text CNN, no link                                  cnn_baseline.py
    Concatenation (control)    the same CNN and Link RF numbers, attached before Dense 64   concat_control.py
    Concatenation (from scratch)  the control without the warm start (the fair partner of E2 from scratch)
    E2 (warm start)            the proposed model, warm-started from the baseline     film.py train
    E2 (from scratch)          E2 without the warm start (the D7 check run)           film.py train --warm-start none

Every model is scored on the TEST part of every fold: with 5-fold cross-validation each message is tested once per seed, by models
that never saw it (their weights, vocabulary and early stopping came from the other folds), so the models are paired on the same
messages (the ids are asserted identical). Two conditions: `complete` (the link numbers of the page as inspected) and `page hidden`
(the link numbers of the page-hidden version, page_missing = 1); the CNN-only baseline has no link and is the same in both.

RQ1  accuracy, precision, recall and F1 (smishing = positive class) per seed, then mean and standard deviation over the seeds.
RQ3  exact McNemar test per seed on the complete condition, b = the first model right and the second wrong, c = the reverse; the
     p-values of one seed are Holm-corrected over all the comparisons listed.

The test part is read ONCE, after everything is fixed. On the real data this refuses to run without --read-test and logs every read in
test_reads.log; the synthetic pilot may be read for a rehearsal (its numbers are never study results).

Writes experiments/film/<dataset>/compare/: test_predictions.csv, test_metrics.csv (per seed), test_summary.csv (mean +- std),
test_mcnemar.csv, test_reads.log.

Usage:
    python -m src.unified.film_compare --pilot
    python -m src.unified.film_compare --read-test          # the real crowd dataset
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import CROWD_N_FOLDS, SEEDS
from src.unified import cnn_baseline as cb
from src.unified import concat_control as cc
from src.unified import film
from src.unified import fit_model as fm
from src.unified import raw_cnn_control as rc
from src.unified import text_cnn as tc
from src.unified.film_data import part_inputs

BASELINE = "CNN-only (baseline)"
CONTROL = "Concatenation (control)"
CONTROL_SCRATCH = "Concatenation (from scratch)"
E2 = "E2 (warm start)"
E2_SCRATCH = "E2 (from scratch)"
E2_NOBYPASS = "E2 (from scratch, no has_url bypass)"      # the ablation: the generator also reads the numbers of a no-link message
RAW_CNN = "CNN-only, text + URL features (from scratch)"   # one network, the 37 raw features instead of the Link RF
RF_ONLY = "RF-only (Link RF)"          # scored by rf_only_compare.py from the saved forest scores, not here
CONDITIONS = ("complete", "page hidden")
METRICS = ["accuracy", "precision", "recall", "f1"]
COMPARISONS = [(E2, BASELINE), (E2, CONTROL), (CONTROL, BASELINE),                          # warm start family: the headline
               (E2_SCRATCH, CONTROL_SCRATCH), (CONTROL_SCRATCH, BASELINE), (E2_SCRATCH, BASELINE),   # from scratch family
               (E2_SCRATCH, E2), (CONTROL_SCRATCH, CONTROL),                                          # what the warm start does
               (E2_SCRATCH, E2_NOBYPASS),                                                             # what the has_url bypass adds
               (E2_SCRATCH, RAW_CNN), (CONTROL_SCRATCH, RAW_CNN), (RAW_CNN, BASELINE)]                # what the Link RF adds over raw features

# name -> (checkpoint kind, loader returning (model, word_to_idx, seq_len))
LOADERS: Dict[str, tuple[str, Callable]] = {
    BASELINE: ("cnn", lambda seed, fold, pilot: cb.load_baseline(seed, fold, pilot)[:3]),
    CONTROL: ("concat", lambda seed, fold, pilot: cc.load_control(seed, fold, pilot)[:3]),
    CONTROL_SCRATCH: ("concat_scratch", lambda seed, fold, pilot: cc.load_control(seed, fold, pilot, warm_start=False)[:3]),
    E2: ("film", lambda seed, fold, pilot: film.load_film(seed, fold, pilot, warm_start=True)[:3]),
    E2_SCRATCH: ("film_scratch", lambda seed, fold, pilot: film.load_film(seed, fold, pilot, warm_start=False)[:3]),
    RAW_CNN: ("cnn_raw", lambda seed, fold, pilot: rc.load_raw(seed, fold, pilot)[:3]),
    E2_NOBYPASS: ("film_scratch_nobypass", lambda seed, fold, pilot: film.load_film(seed, fold, pilot, warm_start=False, bypass=False)[:3]),
}
GROUPS = [("all messages", lambda d: d["has_url"] >= 0),
          ("with a link", lambda d: d["has_url"] == 1),
          ("with a link, page live", lambda d: (d["has_url"] == 1) & (d["page_live"] == 1)),
          ("with a link, page not fetched", lambda d: (d["has_url"] == 1) & (d["page_live"] == 0)),
          ("no link", lambda d: d["has_url"] == 0)]


# ------------------------------------------------------------------ statistics
def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    """Accuracy, precision, recall and F1 with smishing (1) as the positive class."""
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": precision, "recall": recall, "f1": f1}


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar: b = A right and B wrong, c = A wrong and B right; two-sided binomial test of b among b + c."""
    right_a, right_b = pred_a == y, pred_b == y
    b, c = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    return {"b": b, "c": c, "p_exact": 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)}


def holm(p_values: List[float]) -> List[float]:
    """Holm's step-down correction: sorted p_(i) * (m - i), made non-decreasing, capped at 1."""
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


# ------------------------------------------------------------------ the test read
def log_read(dataset: Path, models: List[str], seeds: List[int], folds: List[int], read_test: bool) -> None:
    out = fm.out_dir(dataset, "compare")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_reads.log", "a", encoding="utf-8") as log:
        log.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} dataset={dataset.name} seeds={seeds} folds={folds} "
                  f"read_test_flag={read_test} models={models}\n")


def predict(model, name: str, test, condition: str) -> np.ndarray:
    """P(smishing) of every test message. The baseline ignores the link; the others read the numbers of `condition`."""
    link = test.link_blank if condition == "page hidden" else test.link_full
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(test.token_ids, link), dim=1)[:, 1].numpy()


def test_predictions(pilot: bool, seeds: List[int], folds: List[int], names: List[str]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for fold in folds:
            loaded = {n: LOADERS[n][1](seed, fold, pilot) for n in names}
            base_vocab, base_len = loaded[BASELINE][1], loaded[BASELINE][2]
            assert all(v == base_vocab and l == base_len for _, v, l in loaded.values()), "the models disagree on the fold's vocabulary"
            test = part_inputs(seed, fold, "test", base_vocab, base_len, pilot)
            raw_test = rc.raw_part_inputs(seed, fold, "test", base_vocab, base_len, pilot) if RAW_CNN in loaded else None
            for name, (model, _, _) in loaded.items():
                for condition in CONDITIONS:
                    p = predict(model, name, raw_test if name == RAW_CNN else test, condition)
                    rows.append(pd.DataFrame({"seed": seed, "fold": fold, "message_id": test.message_id, "label": test.labels.numpy(),
                                              "model": name, "condition": condition, "p_smishing": p, "pred": (p >= 0.5).astype(int),
                                              "has_url": test.link_full[:, 3].numpy().astype(int),
                                              "page_live": (1 - test.link_full[:, 2]).numpy().astype(int)}))
    out = pd.concat(rows, ignore_index=True)
    for (seed, model, condition), group in out.groupby(["seed", "model", "condition"]):
        assert group["message_id"].is_unique, f"seed {seed}: a message was tested twice by {model}"
    for (seed, condition), group in out.groupby(["seed", "condition"]):
        id_sets = {m: tuple(sorted(g["message_id"])) for m, g in group.groupby("model")}
        assert len(set(id_sets.values())) == 1, f"seed {seed}: the models were not scored on the same messages"
    return out


def metrics_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-seed metrics, mean and standard deviation over the seeds) for every model and condition."""
    rows = []
    for (seed, model, condition), g in predictions.groupby(["seed", "model", "condition"], sort=False):
        rows.append({"seed": seed, "model": model, "condition": condition, "n": len(g), **scores(g["label"].to_numpy(), g["pred"].to_numpy())})
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["condition", "model"], sort=False)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["seeds"] = per_seed.groupby(["condition", "model"], sort=False)["seed"].nunique()
    return per_seed, summary.reset_index()


def group_metrics(predictions: pd.DataFrame, condition: str = "complete") -> pd.DataFrame:
    """Accuracy, precision, recall and F1 (mean and std over the seeds) per model, for the messages with a link, with a live page, with a
    page not fetched, and with no link. A group a model cannot speak for (a class missing) is left out of its mean, not scored 0."""
    rows = []
    part = predictions[predictions["condition"] == condition]
    for group, select in GROUPS:
        for (seed, model), g in part[select(part)].groupby(["seed", "model"], sort=False):
            rows.append({"group": group, "seed": seed, "model": model, "n": len(g), "smishing_share": round(float(g["label"].mean()), 3),
                         **scores(g["label"].to_numpy(), g["pred"].to_numpy())})
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["group", "model"], sort=False)[["n", "smishing_share"] + METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary.reset_index()


def mcnemar_table(predictions: pd.DataFrame) -> pd.DataFrame:
    complete = predictions[predictions["condition"] == "complete"]
    present = set(complete["model"])
    rows = []
    for seed, g in complete.groupby("seed"):
        pairs = [(a, b) for a, b in COMPARISONS if a in present and b in present]
        wide = {m: gm.sort_values("message_id") for m, gm in g.groupby("model")}
        results = [mcnemar(wide[a]["label"].to_numpy(), wide[a]["pred"].to_numpy(), wide[b]["pred"].to_numpy()) for a, b in pairs]
        adjusted = holm([r["p_exact"] for r in results])
        for (a, b), r, p in zip(pairs, results, adjusted):
            rows.append({"seed": seed, "model_a": a, "model_b": b, **r, "p_holm": p, "significant_0.05": p < 0.05})
    return pd.DataFrame(rows)


def run_compare(pilot: bool, read_test: bool = False, seeds: List[int] = SEEDS, folds: List[int] | None = None) -> dict:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    if not pilot and not read_test:
        raise PermissionError("the test folds of the real data are read once, after everything is fixed: pass --read-test")
    names = [BASELINE] + [n for n in (CONTROL, E2, CONTROL_SCRATCH, E2_SCRATCH, E2_NOBYPASS, RAW_CNN)
                          if all(fm.checkpoint_path(dataset, LOADERS[n][0], s, f).exists() for s in seeds for f in folds)]
    log_read(dataset, names, seeds, folds, read_test)
    predictions = test_predictions(pilot, seeds, folds, names)
    per_seed, summary = metrics_tables(predictions)
    mc = mcnemar_table(predictions)
    groups = group_metrics(predictions)
    out = fm.out_dir(dataset, "compare")
    predictions.to_csv(out / "test_predictions.csv", index=False)
    per_seed.to_csv(out / "test_metrics.csv", index=False)
    summary.to_csv(out / "test_summary.csv", index=False)
    mc.to_csv(out / "test_mcnemar.csv", index=False)
    groups.to_csv(out / "test_groups.csv", index=False)
    return {"predictions": predictions, "per_seed": per_seed, "summary": summary, "mcnemar": mc, "groups": groups, "models": names}


def format_summary(summary: pd.DataFrame) -> str:
    lines = []
    for condition in CONDITIONS:
        part = summary[summary["condition"] == condition]
        lines.append(f"\n{condition} pages (mean +- std over {int(part['seeds'].max())} seeds)")
        lines.append(f"{'model':<26}" + "".join(f"{m:>16}" for m in METRICS))
        for _, r in part.iterrows():
            lines.append(f"{r['model']:<26}" + "".join(f"{r[f'{m}_mean']:>9.3f}+-{r[f'{m}_std']:.3f}" for m in METRICS))
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a rehearsal; its numbers are not study results)")
    ap.add_argument("--read-test", action="store_true", help="required for the real data: reads the test folds")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    args = ap.parse_args(argv)
    try:
        result = run_compare(args.pilot, args.read_test, args.seeds, args.folds)
    except PermissionError as error:
        print(error)
        sys.exit(2)
    print(format_summary(result["summary"]))
    pd.set_option("display.width", 200)
    print("\nBy group (complete pages; mean over the seeds):")
    g = result["groups"]
    for group, _ in GROUPS:
        part = g[g["group"] == group]
        if part.empty:                                     # for example no "no link" messages in a dataset where every message has one
            print(f"\n{group}: no messages")
            continue
        print(f"\n{group} (n per seed about {part['n_mean'].iloc[0]:.0f}, smishing share {part['smishing_share_mean'].iloc[0]:.2f})")
        print(f"{'model':<40}" + "".join(f"{m:>10}" for m in METRICS))
        for _, r in part.iterrows():
            print(f"{r['model']:<40}" + "".join(f"{r[f'{m}_mean']:>10.3f}" for m in METRICS))
    print("\nMcNemar (exact, complete pages), Holm-corrected per seed over the comparisons listed:")
    print(result["mcnemar"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
