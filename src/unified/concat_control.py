"""The concatenation control of the E2 study (plan docs/film_architecture_plan.md, D12): the same CNN and the same Link RF numbers as
E2, but P, u and page_missing are attached to the 64 text numbers before Dense 64. No generator, no FiLM.

Everything else is identical to E2, so only how the link reaches the verdict differs: the same folds and inputs (film_data.part_inputs),
the same shared loop and recipe (fit_model.fit, with the same modality dropout: page hidden 0.25, text hidden 0.10), and the same warm
start (the embedding and the Conv copied from the retrained text-only baseline, cnn_baseline.py). Dense 64 and Output are new and seeded.

Per seed and fold, writes models/film/<dataset>/concat_seed<S>_fold<F>.pt with its .json sidecar (hashes of the vocabulary, the starting
matrix, the weights and the Link RF; the recipe; the validation metrics) and experiments/film/<dataset>/concat/{train_summary,train_log}.csv.
Step check rebuilds every model from its sidecar alone and repeats its validation metrics, which must equal the recorded ones exactly.
The test part is never read here.

Usage:
    python -m src.unified.concat_control train --pilot      # after: python -m src.unified.cnn_baseline train --pilot
    python -m src.unified.concat_control check --pilot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from src.config import CROWD_N_FOLDS, DENSE_WEIGHT_DECAY, LABEL_SMOOTHING, SEEDS
from src.features.model.concat_cnn import ConcatTextCNN
from src.unified import film
from src.unified import fit_model as fm
from src.unified import text_cnn as tc
from src.unified.film_data import part_inputs

KIND = "concat"


def kind_of(warm_start: bool) -> str:
    return KIND if warm_start else KIND + "_scratch"
EXPECTED_TRAINABLE = 62146             # embedding not included


def build_control(seed: int, fold: int, pilot: bool = False, **options) -> tuple[ConcatTextCNN, Dict[str, int], int]:
    """(model, word_to_idx, SEQ_LEN_L) of one seed and fold, untrained, with a fresh embedding layer of its own."""
    embedding, word_to_idx, seq_len = tc.fold_text_setup(tc.dataset_path(pilot), seed, fold)
    tc.seed_everything(seed)
    return ConcatTextCNN(embedding, **options), word_to_idx, seq_len


def trainable_counts(model: ConcatTextCNN) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    own = model.embedding.weight.numel() if model.embedding.weight.requires_grad else 0
    return total - own, total


def train_cell(seed: int, fold: int, pilot: bool, warm_start: bool = True, epochs: int = fm.EPOCHS) -> tuple[dict, pd.DataFrame]:
    dataset = tc.dataset_path(pilot)
    kind = kind_of(warm_start)
    model, word_to_idx, seq_len = build_control(seed, fold, pilot)
    source = film.warm_start_from_baseline(model, dataset, seed, fold, new_layers={"dense", "output"}) if warm_start else None
    train, val = (part_inputs(seed, fold, p, word_to_idx, seq_len, pilot) for p in ("train", "val"))
    checkpoint = fm.checkpoint_path(dataset, kind, seed, fold)
    result = fm.fit(model, (train.token_ids, train.link_full, train.link_blank, train.labels),
                    (val.token_ids, val.link_full, val.labels), seed, checkpoint, epochs=epochs)
    without, with_embedding = trainable_counts(model)
    assert without == EXPECTED_TRAINABLE
    emb_dir = tc.embeddings_dir(dataset, seed, fold)
    fm.write_sidecar(checkpoint, {
        "kind": kind, "seed": seed, "fold": fold, "num_filters": model.num_filters, "kernel_size": model.kernel_size,
        "link_dim": model.link_dim, "dense_units": model.dense.out_features, "freeze_embedding": model.freeze_embedding,
        "seq_len_l": seq_len, "vocab_size": len(word_to_idx), "embeddings_dir": str(emb_dir.relative_to(tc.ROOT)),
        "sha256": fm.hashes(fm.sidecar_files(dataset, seed, fold, checkpoint, {"link_rf": film.link_rf_path(dataset, seed, fold)})),
        "warm_start": source,
        "recipe": {"optimizer": "adam", "lr": fm.LR, "epochs": epochs, "patience": fm.PATIENCE, "batch": fm.BATCH_SIZE,
                   "label_smoothing": LABEL_SMOOTHING, "class_weights": True, "dense_weight_decay": DENSE_WEIGHT_DECAY,
                   "p_page_hidden": fm.P_PAGE_HIDDEN, "p_text_hidden": fm.P_TEXT_HIDDEN},
        "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
        "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "train_seconds": result.seconds,
        "val": {k: result.metrics[k] for k in fm.VAL_METRICS}})
    row = {"seed": seed, "fold": fold, "n_train": len(train.labels), "n_val": len(val.labels), "vocab": len(word_to_idx),
           "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
           "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "train_seconds": round(result.seconds, 3),
           **{k: result.metrics[k] for k in fm.VAL_METRICS}}
    return row, result.log.assign(seed=seed, fold=fold)


def run_train(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None, warm_start: bool = True,
              epochs: int = fm.EPOCHS) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows, logs = [], []
    for seed in seeds:
        for fold in folds:
            row, log = train_cell(seed, fold, pilot, warm_start, epochs)
            rows.append(row)
            logs.append(log)
            print(f"seed {seed} fold {fold}: val F1 {row['val_f1']:.3f}, balanced accuracy {row['val_balanced_accuracy']:.3f} | "
                  f"best epoch {row['best_epoch']} of {row['epochs_run']} | {row['train_seconds']:.1f} s", flush=True)
    out = fm.out_dir(dataset, kind_of(warm_start))
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "train_summary.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(out / "train_log.csv", index=False)
    return summary


def load_control(seed: int, fold: int, pilot: bool = False, warm_start: bool = True) -> tuple[ConcatTextCNN, Dict[str, int], int, dict]:
    """(model in eval mode, word_to_idx, SEQ_LEN_L, sidecar), rebuilt from the sidecar alone; refuses a changed file."""
    dataset = tc.dataset_path(pilot)
    checkpoint = fm.checkpoint_path(dataset, kind_of(warm_start), seed, fold)
    sidecar = fm.read_sidecar(checkpoint)
    fm.check_hashes(sidecar["sha256"], fm.sidecar_files(dataset, seed, fold, checkpoint,
                                                        {"link_rf": film.link_rf_path(dataset, seed, fold)}))
    embedding, word_to_idx = tc.frozen_embedding(tc.ROOT / sidecar["embeddings_dir"])
    model = ConcatTextCNN(embedding, sidecar["num_filters"], sidecar["kernel_size"], sidecar["link_dim"], sidecar["dense_units"],
                          freeze_embedding=sidecar["freeze_embedding"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval(), word_to_idx, sidecar["seq_len_l"], sidecar


def run_check(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None, warm_start: bool = True) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for seed in seeds:
        for fold in folds:
            model, word_to_idx, seq_len, sidecar = load_control(seed, fold, pilot, warm_start)
            val = part_inputs(seed, fold, "val", word_to_idx, seq_len, pilot)
            again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
            gap = max(abs(again[k] - sidecar["val"][k]) for k in fm.VAL_METRICS)
            rows.append({"seed": seed, "fold": fold, "val_f1": again["val_f1"], "max_gap": gap, "ok": gap == 0.0})
    summary = pd.DataFrame(rows)
    out = fm.out_dir(dataset, kind_of(warm_start))
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "check_summary.csv", index=False)
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["train", "check"])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a pipeline test only)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--warm-start", choices=["baseline", "none"], default="baseline",
                    help="train: start from the retrained baseline (default) or from scratch (the D7a check run); check: which set")
    args = ap.parse_args(argv)
    warm = args.warm_start == "baseline"
    if args.step == "train":
        summary = run_train(args.pilot, args.seeds, args.folds, warm)
        print(f"\n{len(summary)} concatenation controls trained; mean val F1 {summary['val_f1'].mean():.3f} (std {summary['val_f1'].std():.3f})")
        return
    summary = run_check(args.pilot, args.seeds, args.folds, warm)
    print(summary.to_string(index=False))
    print(f"\n{int(summary['ok'].sum())} of {len(summary)} rebuilt from their sidecars with a gap of exactly 0")
    sys.exit(0 if summary["ok"].all() else 1)


if __name__ == "__main__":
    main()
