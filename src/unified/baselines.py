"""C-prune against its baselines (docs/c_prune_architecture.md Phase 7). For now: the plain CNN (text only).

Step plain_cnn: the plain CNN baseline per seed and fold, built and fine-tuned like C-prune so that only the
architecture differs: the same Phase A CNN (all 64 filters), the same code (PrunedFusionClassifier with every
filter kept and no link input), the same recipe (fusion.fine_tune), the same splits and seeds.
    text (64 filters) -> Dense 64 (weight decay) -> ReLU -> dropout 0.5 -> 2        (section 5.3: 64 x 64 + 64)
Modality dropout is off: the text is its only input, so hiding it would hide everything.
Checkpoints: models/c_prune/<dataset>/plain_cnn_seed<S>_fold<F>.pt.

Step compare: both models on the TEST folds. Every message is tested exactly once per seed, by models that never
saw it (5-fold cross-validation), so the two models are paired on the same messages (ids asserted identical).
    RQ1  accuracy, precision, recall, F1 (smishing = positive) per seed, then mean +- std over the seeds;
         C-prune at its chosen final k (fusion.final_k) and at every k
    RQ2  trainable parameters; latency per message (batch 1, CPU, 1 thread, 50 warm-up messages first) of the
         text model (stage 4) and the Link RF (stage 3), median / 95th percentile / mean
    RQ3  exact McNemar per seed: b = C-prune right and plain CNN wrong, c = the reverse
C-prune scores the RAW test rows (fusion.predict_messages: the saved Link RF makes the 40 numbers), as the app
will. The web inspection (stage 1) is not timed here: the pilot has no live inspection.
The test folds are read once, after everything is fixed (Phase 7): on the real data this step refuses to run
without --read-test, and each read is logged in test_reads.log. The pilot may be read for a rehearsal.

Writes experiments/c_prune/<dataset>/baselines/: plain_cnn_train_summary.csv; test_predictions.csv,
test_metrics.csv (per seed), test_summary.csv (mean +- std), test_mcnemar.csv, test_latency.csv.

Usage:
    python -m src.unified.baselines --pilot plain_cnn
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
from src.unified.fusion import final_k, fine_tune, load_c_prune, predict_messages
from src.unified.link_lane import lane_numbers
from src.unified.text_cnn import dataset_path, load_phase_a, model_dir, tensors, trainable_parameters

WARM_UP = 50
MODELS = ["c_prune", "plain_cnn"]


def out_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "experiments" / "c_prune" / dataset.stem / "baselines"


def plain_cnn_path(seed: int, fold: int, pilot: bool = False) -> Path:
    return model_dir(dataset_path(pilot)) / f"plain_cnn_seed{seed}_fold{fold}.pt"


def build_plain_cnn(seed: int, fold: int, pilot: bool = False) -> tuple:
    """(plain CNN, word_to_idx, Phase A sidecar): the Phase A CNN, every filter kept, no link input."""
    cnn, word_to_idx, sidecar = load_phase_a(model_dir(dataset_path(pilot)) / f"phase_a_seed{seed}_fold{fold}.pt")
    every = list(range(sidecar["num_filters"]))
    torch.manual_seed(seed)
    return PrunedFusionClassifier(prune_encoder(cnn, every), every, url_dim=0, dense_units=DENSE_UNITS), word_to_idx, sidecar


def no_link(n: int) -> torch.Tensor:
    return torch.zeros(n, 0)


def text_inputs(part: pd.DataFrame, word_to_idx: dict, seq_len: int) -> tuple:
    ids, labels = tensors(part, word_to_idx, seq_len)
    return ids, no_link(len(part)), labels


def load_plain_cnn(seed: int, fold: int, pilot: bool = False) -> tuple:
    model, word_to_idx, sidecar = build_plain_cnn(seed, fold, pilot)
    model.load_state_dict(torch.load(plain_cnn_path(seed, fold, pilot), map_location="cpu"))
    return model.eval(), word_to_idx, sidecar


def run_plain_cnn(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    out = out_dir(dataset_path(pilot))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            model, word_to_idx, sidecar = build_plain_cnn(seed, fold, pilot)
            parts = load_fold(seed, fold, pilot=pilot)
            ids, link, y = text_inputs(parts["train"], word_to_idx, sidecar["seq_len_l"])
            val = text_inputs(parts["val"], word_to_idx, sidecar["seq_len_l"])
            start = time.perf_counter()
            metrics = fine_tune(model, (ids, link, link, y), val, seed, plain_cnn_path(seed, fold, pilot),
                                p_page_hidden=0.0, p_text_hidden=0.0)
            seconds = time.perf_counter() - start
            rows.append({"seed": seed, "fold": fold, "trainable_parameters": trainable_parameters(model),
                         **{m: round(v, 4) if isinstance(v, float) else v for m, v in metrics.items()},
                         "train_seconds": round(seconds, 2)})
            print(f"seed {seed} fold {fold}: plain CNN val F1 {metrics['val_f1']:.4f}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "plain_cnn_train_summary.csv", index=False)
    return summary


def test_predictions(seed: int, n_folds: int, pilot: bool) -> pd.DataFrame:
    """Every message of the dataset once (its test fold), with each model's P(smishing)."""
    frames = []
    for fold in range(n_folds):
        test = load_fold(seed, fold, pilot=pilot)["test"]
        frame = pd.DataFrame({"seed": seed, "fold": fold, "message_id": test["message_id"],
                              "y": (test["label"] == "smishing").astype(int)})
        for k in PRUNE_KEEP:
            model, link_rf, word_to_idx, sidecar = load_c_prune(seed, fold, k, pilot)
            frame[f"p_c_prune_k{k}"] = predict_messages(model, link_rf, word_to_idx,
                                                        sidecar["architecture"]["seq_len_l"], test)
        plain, word_to_idx, sidecar = load_plain_cnn(seed, fold, pilot)
        ids, link, _ = text_inputs(test, word_to_idx, sidecar["seq_len_l"])
        frame["p_plain_cnn"] = plain.predict_proba(ids, link)[:, 1].numpy()
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
        seq_len = sidecar["architecture"]["seq_len_l"]
        ids, _ = tensors(test, word_to_idx, seq_len)
        link = torch.tensor(lane_numbers(link_rf["model"], test).to_numpy(), dtype=torch.float32)
        plain, plain_w2i, plain_sidecar = load_plain_cnn(seed, 0, pilot)
        plain_ids, _ = tensors(test, plain_w2i, plain_sidecar["seq_len_l"])
        with torch.no_grad():
            timed = {
                ("c_prune", "link_rf"): lambda i: lane_numbers(link_rf["model"], test.iloc[[i]]),
                ("c_prune", "text_model"): lambda i: model(ids[i:i + 1], link[i:i + 1]),
                ("plain_cnn", "text_model"): lambda i: plain(plain_ids[i:i + 1], no_link(1)),
            }
            rows = []
            for (name, stage), fn in timed.items():
                ms = time_per_message(fn, len(test))
                rows.append({"seed": seed, "model": name, "stage": stage, "messages": len(ms),
                             "median_ms": np.median(ms), "p95_ms": np.percentile(ms, 95), "mean_ms": ms.mean()})
        return rows
    finally:
        torch.set_num_threads(threads)


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
    predictions, metrics, tests, timing = [], [], [], []
    for seed in seeds:
        pred = test_predictions(seed, n_folds, pilot)
        predictions.append(pred)
        y = pred["y"].to_numpy()
        columns = {f"c_prune_k{k}": f"p_c_prune_k{k}" for k in PRUNE_KEEP} | {"plain_cnn": "p_plain_cnn"}
        for name, column in columns.items():
            metrics.append({"seed": seed, "model": name, "messages": len(y), **scores(y, pred[column].to_numpy())})
        tests.append({"seed": seed, "comparison": f"C-prune (k = {chosen}) vs plain CNN",
                      **mcnemar(y, pred[f"p_c_prune_k{chosen}"].to_numpy(), pred["p_plain_cnn"].to_numpy())})
        timing += latency(seed, pilot)
        print(f"seed {seed}: {len(y)} test messages scored")
    pd.concat(predictions, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
    per_seed = pd.DataFrame(metrics)
    per_seed.to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "test_mcnemar.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "test_latency.csv", index=False)
    summary = per_seed.groupby("model", sort=False)[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.round(4).reset_index()
    summary.to_csv(out / "test_summary.csv", index=False)
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("--read-test", action="store_true", help="real data only: read the test folds (once, last)")
    ap.add_argument("step", choices=["plain_cnn", "compare"])
    a = ap.parse_args(argv)
    summary = run_plain_cnn(a.pilot) if a.step == "plain_cnn" else run_compare(a.pilot, read_test=a.read_test)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
