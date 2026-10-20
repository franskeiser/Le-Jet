"""The text-only baseline of the E2 study, retrained with a trainable embedding (plan docs/film_architecture_plan.md, step 3.6).

D6 changed on 2026-10-20: the embedding is fine-tuned for every model. The old baseline checkpoints were trained with a frozen
embedding, so E2 and the baseline would not have been treated the same. This module trains the baseline again, for every seed
and fold, with the same shared loop E2 uses (fit_model.fit) and the same recipe: Adam 1e-3, up to 30 epochs, patience 3 on
validation balanced accuracy, label smoothing 0.1, class weights, weight decay 1e-4 on Dense 64's weights.
It has no link and no modality dropout (it has one input), so it is passed 0 and 0.

Per seed and fold:
    model    src/features/model/plain_cnn.PlainTextCNN: embedding (from the fold's vocabulary and fastText matrix, fine-tuned),
             Conv1d 64 filters, Dense 64, Output: 61,954 trainable parameters without the embedding;
    data     the fold's train and validation messages (film.part_inputs), the same rows and order E2 reads;
    output   models/film/<dataset>/cnn_seed<S>_fold<F>.pt, and .json next to it (the sidecar: what it was built from, with
             sha256 of the vocabulary, the starting embedding matrix and the weights, the recipe, the validation metrics);
             experiments/film/<dataset>/cnn_baseline/{train_summary,train_log}.csv.

Step check rebuilds every model from its sidecar alone (hash-checked) and repeats its validation metrics, which must equal the
ones recorded at training time.

Usage:
    python -m src.unified.cnn_baseline train --pilot
    python -m src.unified.cnn_baseline check --pilot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.config import CROWD_N_FOLDS, DENSE_WEIGHT_DECAY, LABEL_SMOOTHING, SEEDS
from src.features.model.plain_cnn import PlainTextCNN
from src.unified import fit_model as fm
from src.unified import text_cnn as tc
from src.unified.film_data import part_inputs

KIND = "cnn"
EXPECTED_TRAINABLE = 61954             # section A5 (embedding not included)


def build_baseline(seed: int, fold: int, pilot: bool = False, **options) -> tuple[PlainTextCNN, Dict[str, int], int]:
    """(model, word_to_idx, SEQ_LEN_L) of one seed and fold, untrained, with a fresh embedding layer of its own."""
    embedding, word_to_idx, seq_len = tc.fold_text_setup(tc.dataset_path(pilot), seed, fold)
    tc.seed_everything(seed)
    return PlainTextCNN(embedding, **options), word_to_idx, seq_len


def trainable_counts(model: PlainTextCNN) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    own = model.embedding.weight.numel() if model.embedding.weight.requires_grad else 0
    return total - own, total


def train_cell(seed: int, fold: int, pilot: bool, epochs: int = fm.EPOCHS) -> tuple[dict, pd.DataFrame]:
    dataset = tc.dataset_path(pilot)
    model, word_to_idx, seq_len = build_baseline(seed, fold, pilot)
    train, val = (part_inputs(seed, fold, p, word_to_idx, seq_len, pilot) for p in ("train", "val"))
    checkpoint = fm.checkpoint_path(dataset, KIND, seed, fold)
    result = fm.fit(model, (train.token_ids, train.link_full, train.link_blank, train.labels),
                    (val.token_ids, val.link_full, val.labels), seed, checkpoint, epochs=epochs,
                    p_page_hidden=0.0, p_text_hidden=0.0)
    without, with_embedding = trainable_counts(model)
    emb_dir = tc.embeddings_dir(dataset, seed, fold)
    sidecar = fm.write_sidecar(checkpoint, {
        "kind": KIND, "seed": seed, "fold": fold, "num_filters": model.num_filters, "kernel_size": model.kernel_size,
        "freeze_embedding": model.freeze_embedding, "seq_len_l": seq_len, "vocab_size": len(word_to_idx),
        "embeddings_dir": str(emb_dir.relative_to(tc.ROOT)), "sha256": fm.hashes(fm.sidecar_files(dataset, seed, fold, checkpoint)),
        "recipe": {"optimizer": "adam", "lr": fm.LR, "epochs": epochs, "patience": fm.PATIENCE, "batch": fm.BATCH_SIZE,
                   "label_smoothing": LABEL_SMOOTHING, "class_weights": True, "dense_weight_decay": DENSE_WEIGHT_DECAY},
        "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
        "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "train_seconds": result.seconds,
        "val": {k: result.metrics[k] for k in fm.VAL_METRICS}})
    row = {"seed": seed, "fold": fold, "n_train": len(train.labels), "n_val": len(val.labels), "vocab": len(word_to_idx),
           "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
           "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "train_seconds": round(result.seconds, 3),
           **{k: result.metrics[k] for k in fm.VAL_METRICS}}
    log = result.log.assign(seed=seed, fold=fold)
    assert sidecar["trainable_without_embedding"] == EXPECTED_TRAINABLE
    return row, log


def run_train(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None, epochs: int = fm.EPOCHS) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows, logs = [], []
    for seed in seeds:
        for fold in folds:
            row, log = train_cell(seed, fold, pilot, epochs)
            rows.append(row)
            logs.append(log)
            print(f"seed {seed} fold {fold}: val F1 {row['val_f1']:.3f}, balanced accuracy {row['val_balanced_accuracy']:.3f} | "
                  f"best epoch {row['best_epoch']} of {row['epochs_run']} | {row['train_seconds']:.1f} s", flush=True)
    out = fm.out_dir(dataset, "cnn_baseline")
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "train_summary.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(out / "train_log.csv", index=False)
    return summary


def load_baseline(seed: int, fold: int, pilot: bool = False) -> tuple[PlainTextCNN, Dict[str, int], int, dict]:
    """(model in eval mode, word_to_idx, SEQ_LEN_L, sidecar), rebuilt from the sidecar alone; refuses a changed file."""
    dataset = tc.dataset_path(pilot)
    checkpoint = fm.checkpoint_path(dataset, KIND, seed, fold)
    sidecar = fm.read_sidecar(checkpoint)
    fm.check_hashes(sidecar["sha256"], fm.sidecar_files(dataset, seed, fold, checkpoint))
    embedding, word_to_idx = tc.frozen_embedding(tc.ROOT / sidecar["embeddings_dir"])
    model = PlainTextCNN(embedding, sidecar["num_filters"], sidecar["kernel_size"], freeze_embedding=sidecar["freeze_embedding"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval(), word_to_idx, sidecar["seq_len_l"], sidecar


def run_check(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for seed in seeds:
        for fold in folds:
            model, word_to_idx, seq_len, sidecar = load_baseline(seed, fold, pilot)
            val = part_inputs(seed, fold, "val", word_to_idx, seq_len, pilot)
            again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
            gap = max(abs(again[k] - sidecar["val"][k]) for k in fm.VAL_METRICS)
            rows.append({"seed": seed, "fold": fold, "val_f1": again["val_f1"], "max_gap": gap, "ok": gap == 0.0})
    summary = pd.DataFrame(rows)
    out = fm.out_dir(dataset, "cnn_baseline")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "check_summary.csv", index=False)
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["train", "check"])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a pipeline test only)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    args = ap.parse_args(argv)
    if args.step == "train":
        summary = run_train(args.pilot, args.seeds, args.folds)
        print(f"\n{len(summary)} baselines trained; mean val F1 {summary['val_f1'].mean():.3f} (std {summary['val_f1'].std():.3f})")
        return
    summary = run_check(args.pilot, args.seeds, args.folds)
    print(summary.to_string(index=False))
    print(f"\n{int(summary['ok'].sum())} of {len(summary)} rebuilt from their sidecars with a gap of exactly 0")
    sys.exit(0 if summary["ok"].all() else 1)


if __name__ == "__main__":
    main()
