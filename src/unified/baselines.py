"""C-prune against its baselines (docs/c_prune_architecture.md Phase 7). For now: plain CNN and UNGATED.

Each baseline is built and fine-tuned like C-prune, so that only the architecture differs: the same Phase A CNN
with ALL 64 filters kept (nothing pruned), the same code (PrunedFusionClassifier with every filter kept), the same
recipe (fusion.fine_tune), the same splits, seeds and frozen embedding.
    plain_cnn  text (64 filters) -> Dense 64 -> 2                      (section 5.3: Dense 64 x 64 + 64)
               no link input; modality dropout off (the text is its only input, hiding it would hide everything)
    ungated    text (64 filters) + the 40 link numbers -> Dense 64 -> 2  (section 5.3: 64,514 trainable)
               = C-prune with k = 64: the same link lane, the same modality dropout (page 0.25, text 0.10)
Step <name> trains one baseline per seed and fold: models/c_prune/<dataset>/<name>_seed<S>_fold<F>.pt.

Step compare: every model on the TEST folds. Every message is tested exactly once per seed, by models that never
saw it (5-fold cross-validation), so the models are paired on the same messages (ids asserted identical).
    RQ1  accuracy, precision, recall, F1 (smishing = positive) per seed, then mean +- std over the seeds;
         C-prune at every k, and its chosen final k (fusion.final_k) for the tests
    RQ2  trainable parameters; latency per message (batch 1, CPU, 1 thread, 50 warm-up messages first) of each
         text model (stage 4) and of the Link RF (stage 3), median / 95th percentile / mean
    RQ3  exact McNemar per seed, C-prune (final k) against each baseline: b = C-prune right and the baseline
         wrong, c = the reverse; the p-values of one seed are Holm-corrected over the comparisons
The models with a link lane score the RAW test rows: the saved Link RF (the C-prune bundle's, the same forest for
every model) makes the 40 numbers, as in the app. The web inspection (stage 1) is not timed: the pilot has no live
inspection.
The test folds are read once, after everything is fixed (Phase 7): on the real data this step refuses to run
without --read-test, and each read is logged in test_reads.log. The pilot may be read for a rehearsal.

Writes experiments/c_prune/<dataset>/baselines/: <name>_train_summary.csv; test_predictions.csv,
test_metrics.csv (per seed), test_summary.csv (mean +- std), test_mcnemar.csv, test_latency.csv,
test_parameters.csv.

Usage:
    python -m src.unified.baselines --pilot plain_cnn
    python -m src.unified.baselines --pilot ungated
    python -m src.unified.baselines --pilot compare
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import ARTIFACTS_ROOT, CROWD_N_FOLDS, DENSE_UNITS, PRUNE_KEEP, SEEDS
from src.data.split_crowd import load_fold
from src.features.model.cnn_encoder import prune_encoder
from src.features.model.fusion_classifier import PrunedFusionClassifier
from src.unified.fusion import final_k, fine_tune, load_c_prune, train_inputs
from src.unified.link_lane import LANE, lane_numbers
from src.unified.text_cnn import dataset_path, load_phase_a, model_dir, tensors, trainable_parameters

WARM_UP = 50
BASELINES = {"plain_cnn": {"link": False, "label": "plain CNN"},
             "ungated": {"link": True, "label": "UNGATED"}}


def out_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "experiments" / "c_prune" / dataset.stem / "baselines"


def baseline_path(name: str, seed: int, fold: int, pilot: bool = False) -> Path:
    return model_dir(dataset_path(pilot)) / f"{name}_seed{seed}_fold{fold}.pt"


def build_baseline(name: str, seed: int, fold: int, pilot: bool = False) -> tuple:
    """(model, word_to_idx, Phase A sidecar): the Phase A CNN with every filter kept, with or without the link lane."""
    cnn, word_to_idx, sidecar = load_phase_a(model_dir(dataset_path(pilot)) / f"phase_a_seed{seed}_fold{fold}.pt")
    every = list(range(sidecar["num_filters"]))
    url_dim = len(LANE) if BASELINES[name]["link"] else 0
    torch.manual_seed(seed)
    model = PrunedFusionClassifier(prune_encoder(cnn, every), every, url_dim=url_dim, dense_units=DENSE_UNITS)
    return model, word_to_idx, sidecar


def no_link(n: int) -> torch.Tensor:
    return torch.zeros(n, 0)


def text_inputs(part: pd.DataFrame, word_to_idx: dict, seq_len: int) -> tuple:
    ids, labels = tensors(part, word_to_idx, seq_len)
    return ids, no_link(len(part)), labels


def load_baseline(name: str, seed: int, fold: int, pilot: bool = False) -> tuple:
    model, word_to_idx, sidecar = build_baseline(name, seed, fold, pilot)
    model.load_state_dict(torch.load(baseline_path(name, seed, fold, pilot), map_location="cpu"))
    return model.eval(), word_to_idx, sidecar


def run_baseline(name: str, pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    out = out_dir(dataset_path(pilot))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            model, word_to_idx, sidecar = build_baseline(name, seed, fold, pilot)
            seq_len = sidecar["seq_len_l"]
            if BASELINES[name]["link"]:
                train, val = train_inputs(seed, fold, word_to_idx, seq_len, pilot)
                dropout = {}                                  # the shared rates, as for C-prune
            else:
                parts = load_fold(seed, fold, pilot=pilot)
                ids, link, y = text_inputs(parts["train"], word_to_idx, seq_len)
                train, val = (ids, link, link, y), text_inputs(parts["val"], word_to_idx, seq_len)
                dropout = {"p_page_hidden": 0.0, "p_text_hidden": 0.0}
            start = time.perf_counter()
            metrics = fine_tune(model, train, val, seed, baseline_path(name, seed, fold, pilot), **dropout)
            seconds = time.perf_counter() - start
            rows.append({"seed": seed, "fold": fold, "trainable_parameters": trainable_parameters(model),
                         **{m: round(v, 4) if isinstance(v, float) else v for m, v in metrics.items()},
                         "train_seconds": round(seconds, 2)})
            print(f"seed {seed} fold {fold}: {BASELINES[name]['label']} val F1 {metrics['val_f1']:.4f}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / f"{name}_train_summary.csv", index=False)
    return summary


def test_predictions(seed: int, n_folds: int, pilot: bool) -> pd.DataFrame:
    """Every message of the dataset once (its test fold), with each model's P(smishing)."""
    frames = []
    for fold in range(n_folds):
        test = load_fold(seed, fold, pilot=pilot)["test"]
        frame = pd.DataFrame({"seed": seed, "fold": fold, "message_id": test["message_id"],
                              "y": (test["label"] == "smishing").astype(int)})
        link = None
        for k in PRUNE_KEEP:
            model, link_rf, word_to_idx, sidecar = load_c_prune(seed, fold, k, pilot)
            link = torch.tensor(lane_numbers(link_rf["model"], test)[LANE].to_numpy(), dtype=torch.float32)
            ids, _ = tensors(test, word_to_idx, sidecar["architecture"]["seq_len_l"])
            frame[f"p_c_prune_k{k}"] = model.predict_proba(ids, link)[:, 1].numpy()
        for name, spec in BASELINES.items():
            model, word_to_idx, sidecar = load_baseline(name, seed, fold, pilot)
            ids, _ = tensors(test, word_to_idx, sidecar["seq_len_l"])
            frame[f"p_{name}"] = model.predict_proba(ids, link if spec["link"] else no_link(len(test)))[:, 1].numpy()
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    assert out["message_id"].is_unique, "a message was tested twice in one seed"
    return out


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}


def mcnemar(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    """Exact McNemar: b = A right and B wrong, c = A wrong and B right; two-sided binomial test on b of b + c."""
    right_a, right_b = (p_a >= 0.5) == y, (p_b >= 0.5) == y
    b, c = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    return {"b": b, "c": c, "p_exact": 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue}


def holm(p_values: List[float]) -> List[float]:
    """Holm's step-down correction: sorted p_(i) * (m - i), made non-decreasing, capped at 1."""
    order = np.argsort(p_values)
    m, adjusted, running = len(p_values), np.empty(len(p_values)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


def time_per_message(fn, n_items: int) -> np.ndarray:
    """Milliseconds per call of fn(i), one message at a time, after WARM_UP untimed calls."""
    for i in range(min(WARM_UP, n_items)):
        fn(i)
    times = []
    for i in range(n_items):
        start = time.perf_counter()
        fn(i)
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def latency(seed: int, pilot: bool) -> List[dict]:
    """Stages 3 and 4 per message on fold 0's test messages, batch 1, one CPU thread."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        test = load_fold(seed, 0, pilot=pilot)["test"]
        model, link_rf, word_to_idx, sidecar = load_c_prune(seed, 0, final_k(pilot), pilot)
        link_rf["model"].set_params(n_jobs=1)
        ids, _ = tensors(test, word_to_idx, sidecar["architecture"]["seq_len_l"])
        link = torch.tensor(lane_numbers(link_rf["model"], test)[LANE].to_numpy(), dtype=torch.float32)
        timed = {("link_rf", "link_rf"): lambda i: lane_numbers(link_rf["model"], test.iloc[[i]]),
                 ("c_prune", "text_model"): lambda i: model(ids[i:i + 1], link[i:i + 1])}
        for name, spec in BASELINES.items():
            base, base_w2i, base_sidecar = load_baseline(name, seed, 0, pilot)
            base_ids, _ = tensors(test, base_w2i, base_sidecar["seq_len_l"])
            base_link = link if spec["link"] else no_link(len(test))
            timed[(name, "text_model")] = (lambda m, x, l: lambda i: m(x[i:i + 1], l[i:i + 1]))(base, base_ids, base_link)
        rows = []
        with torch.no_grad():
            for (name, stage), fn in timed.items():
                ms = time_per_message(fn, len(test))
                rows.append({"seed": seed, "model": name, "stage": stage, "messages": len(ms),
                             "median_ms": np.median(ms), "p95_ms": np.percentile(ms, 95), "mean_ms": ms.mean()})
        return rows
    finally:
        torch.set_num_threads(threads)


def parameters(seed: int, pilot: bool) -> List[dict]:
    rows = [{"seed": seed, "model": f"c_prune_k{k}", "trainable_parameters": load_c_prune(seed, 0, k, pilot)[3][
        "trainable_parameters"]} for k in PRUNE_KEEP]
    return rows + [{"seed": seed, "model": name, "trainable_parameters": trainable_parameters(
        load_baseline(name, seed, 0, pilot)[0])} for name in BASELINES]


def run_compare(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS, read_test: bool = False) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    out = out_dir(dataset)
    if not pilot and not read_test:
        raise SystemExit("this reads the TEST folds of the real data, once, after everything is fixed (Phase 7): "
                         "run it again with --read-test")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_reads.log", "a", encoding="utf-8") as log:
        log.write(f"{datetime.now().isoformat(timespec='seconds')} test folds read ({dataset.name})\n")
    chosen = final_k(pilot)
    predictions, metrics, tests, timing, sizes = [], [], [], [], []
    for seed in seeds:
        pred = test_predictions(seed, n_folds, pilot)
        predictions.append(pred)
        y = pred["y"].to_numpy()
        for name in [f"c_prune_k{k}" for k in PRUNE_KEEP] + list(BASELINES):
            metrics.append({"seed": seed, "model": name, "messages": len(y), **scores(y, pred[f"p_{name}"].to_numpy())})
        seed_tests = [{"seed": seed, "comparison": f"C-prune (k = {chosen}) vs {spec['label']}",
                       **mcnemar(y, pred[f"p_c_prune_k{chosen}"].to_numpy(), pred[f"p_{name}"].to_numpy())}
                      for name, spec in BASELINES.items()]
        for row, p_holm in zip(seed_tests, holm([r["p_exact"] for r in seed_tests])):
            row.update(p_holm=p_holm, significant=p_holm < 0.05)
        tests += seed_tests
        timing += latency(seed, pilot)
        sizes += parameters(seed, pilot)
        print(f"seed {seed}: {len(y)} test messages scored")
    pd.concat(predictions, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
    per_seed = pd.DataFrame(metrics)
    per_seed.to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "test_mcnemar.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "test_latency.csv", index=False)
    pd.DataFrame(sizes).to_csv(out / "test_parameters.csv", index=False)
    summary = per_seed.groupby("model", sort=False)[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.round(4).reset_index()
    summary.to_csv(out / "test_summary.csv", index=False)
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("--read-test", action="store_true", help="real data only: read the test folds (once, last)")
    ap.add_argument("step", choices=list(BASELINES) + ["compare"])
    a = ap.parse_args(argv)
    summary = run_compare(a.pilot, read_test=a.read_test) if a.step == "compare" else run_baseline(a.step, a.pilot)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
