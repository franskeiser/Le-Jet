"""The "CNN-only, same inputs" baseline of the E2 study: one network that reads the SMS text AND the 37 URL / page features, with no Link RF.

This is the baseline the panel asked for. The text-only CNN (cnn_baseline.py) reads no link at all; this one is given what E2 is given
(the text and the link evidence) but has to learn the link evidence from the raw 37 features itself, in the same network (a Dense branch
beside the Conv, then one head; src/features/model/raw_cnn.py). Everything else matches the study: the same folds, rows and order
(film_data.part_inputs), the same shared loop and recipe (fit_model.fit, with the same modality dropout: page hidden 0.25, text
hidden 0.10), trained from scratch like E2 (from scratch) and the control (from scratch). A "page hidden" message has its 13 page columns
set to -1 and page_missing = 1 (link_lane.blank_page), the same version the forest-based models score.

Per seed and fold, writes models/film/<dataset>/cnn_raw_seed<S>_fold<F>.pt with its .json sidecar (hashes of the vocabulary, the
starting matrix, the weights and the dataset file; the recipe; the validation metrics) and experiments/film/<dataset>/cnn_raw/
{train_summary,train_log}.csv. Step check rebuilds every model from its sidecar alone and repeats its validation metrics, which must
equal the recorded ones exactly. The test part is never read here.

Usage:
    python -m src.unified.raw_cnn_control train --pilot
    python -m src.unified.raw_cnn_control check --pilot
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.config import CROWD_N_FOLDS, DENSE_WEIGHT_DECAY, LABEL_SMOOTHING, SEEDS
from src.data.split_crowd import load_fold
from src.features.model.raw_cnn import N_FEATURES, RAW_WIDTH, RawFeatureCNN
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc
from src.unified.film_data import PartData, part_inputs
from src.unified.prepare_web_37 import FEATURES_37

KIND = "cnn_raw"


def expected_trainable(embedding_dim: int = 300, num_filters: int = 64, kernel_size: int = 3, branch_units: int = 64,
                       dense_units: int = 64) -> int:
    """Trainable parameters without the embedding: conv + link branch (39 -> 64) + dense (64 + 64 -> 64) + output."""
    return (embedding_dim * kernel_size * num_filters + num_filters + RAW_WIDTH * branch_units + branch_units
            + (num_filters + branch_units) * dense_units + dense_units + dense_units * 2 + 2)


EXPECTED_TRAINABLE = expected_trainable()        # 68,610


# ------------------------------------------------------------------ inputs
def raw_link_matrix(rows: pd.DataFrame) -> np.ndarray:
    """(n, 39) float32 [37 raw features, page_missing, has_url] of `rows` as they are. A message with no link gets zeros for the features
    (the model ignores them anyway), so a NaN in a no-link row never travels."""
    has_url = rows["has_url"].astype(int).to_numpy() if "has_url" in rows.columns else np.ones(len(rows), dtype=int)
    features = rows[FEATURES_37].to_numpy(dtype=np.float64, copy=True)
    features[has_url == 0] = 0.0
    page_missing = 1 - rows["page_fetched"].astype(int).to_numpy()
    out = np.column_stack([features, page_missing, has_url]).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("a feature of a message with a link is missing or not finite")
    return out


def feature_stats(train_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the 37 features over the TRAIN messages that have a link (page as inspected); a constant column gets std 1."""
    linked = train_rows[ll.has_link(train_rows).to_numpy()]
    x = linked[FEATURES_37].to_numpy(dtype=np.float64)
    std = x.std(axis=0)
    return x.mean(axis=0).astype(np.float32), np.where(std > 0, std, 1.0).astype(np.float32)


def raw_part_inputs(seed: int, fold: int, part: str, word_to_idx: Dict[str, int], seq_len: int, pilot: bool = False) -> PartData:
    """film_data.part_inputs with link_full / link_blank replaced by the (n, 39) raw matrices: the page as inspected and the page hidden.
    Same rows, same order, same labels and ids as every other model of the study."""
    base = part_inputs(seed, fold, part, word_to_idx, seq_len, pilot)
    rows = load_fold(seed, fold, pilot=pilot)[part]
    assert np.array_equal(rows["message_id"].to_numpy(), base.message_id), "the raw rows and the text rows are not in the same order"
    return base._replace(link_full=torch.tensor(raw_link_matrix(rows)), link_blank=torch.tensor(raw_link_matrix(ll.blank_page(rows))))


# ------------------------------------------------------------------ the model
def build_raw(seed: int, fold: int, pilot: bool = False, **options) -> tuple[RawFeatureCNN, Dict[str, int], int]:
    """(model, word_to_idx, SEQ_LEN_L) of one seed and fold, untrained, with a fresh embedding layer of its own and the train part's
    feature mean / std."""
    dataset = tc.dataset_path(pilot)
    embedding, word_to_idx, seq_len = tc.fold_text_setup(dataset, seed, fold)
    mean, std = feature_stats(load_fold(seed, fold, pilot=pilot)["train"])
    tc.seed_everything(seed)
    return RawFeatureCNN(embedding, torch.tensor(mean), torch.tensor(std), **options), word_to_idx, seq_len


def trainable_counts(model: RawFeatureCNN) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    own = model.embedding.weight.numel() if model.embedding.weight.requires_grad else 0
    return total - own, total


def sidecar_extra(pilot: bool) -> dict:
    return {"dataset.csv": tc.dataset_path(pilot)}


def train_cell(seed: int, fold: int, pilot: bool, epochs: int = fm.EPOCHS) -> tuple[dict, pd.DataFrame]:
    dataset = tc.dataset_path(pilot)
    model, word_to_idx, seq_len = build_raw(seed, fold, pilot)
    train, val = (raw_part_inputs(seed, fold, p, word_to_idx, seq_len, pilot) for p in ("train", "val"))
    checkpoint = fm.checkpoint_path(dataset, KIND, seed, fold)
    result = fm.fit(model, (train.token_ids, train.link_full, train.link_blank, train.labels),
                    (val.token_ids, val.link_full, val.labels), seed, checkpoint, epochs=epochs)
    without, with_embedding = trainable_counts(model)
    assert without == EXPECTED_TRAINABLE
    emb_dir = tc.embeddings_dir(dataset, seed, fold)
    fm.write_sidecar(checkpoint, {
        "kind": KIND, "seed": seed, "fold": fold, "num_filters": model.num_filters, "kernel_size": model.kernel_size,
        "branch_units": model.branch_units, "dense_units": model.dense.out_features, "freeze_embedding": model.freeze_embedding,
        "seq_len_l": seq_len, "vocab_size": len(word_to_idx), "embeddings_dir": str(emb_dir.relative_to(tc.ROOT)),
        "sha256": fm.hashes(fm.sidecar_files(dataset, seed, fold, checkpoint, sidecar_extra(pilot))),
        "inputs": "text + 37 raw URL/page features + page_missing + has_url; no Link RF", "warm_start": None,
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
    out = fm.out_dir(dataset, KIND)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "train_summary.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(out / "train_log.csv", index=False)
    return summary


def load_raw(seed: int, fold: int, pilot: bool = False) -> tuple[RawFeatureCNN, Dict[str, int], int, dict]:
    """(model in eval mode, word_to_idx, SEQ_LEN_L, sidecar), rebuilt from the sidecar alone; refuses a changed file."""
    dataset = tc.dataset_path(pilot)
    checkpoint = fm.checkpoint_path(dataset, KIND, seed, fold)
    sidecar = fm.read_sidecar(checkpoint)
    fm.check_hashes(sidecar["sha256"], fm.sidecar_files(dataset, seed, fold, checkpoint, sidecar_extra(pilot)))
    embedding, word_to_idx = tc.frozen_embedding(tc.ROOT / sidecar["embeddings_dir"])
    model = RawFeatureCNN(embedding, num_filters=sidecar["num_filters"], kernel_size=sidecar["kernel_size"],
                          branch_units=sidecar["branch_units"], dense_units=sidecar["dense_units"],
                          freeze_embedding=sidecar["freeze_embedding"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))        # the feature mean / std come back with the weights
    return model.eval(), word_to_idx, sidecar["seq_len_l"], sidecar


def run_check(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for seed in seeds:
        for fold in folds:
            model, word_to_idx, seq_len, sidecar = load_raw(seed, fold, pilot)
            val = raw_part_inputs(seed, fold, "val", word_to_idx, seq_len, pilot)
            again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
            gap = max(abs(again[k] - sidecar["val"][k]) for k in fm.VAL_METRICS)
            rows.append({"seed": seed, "fold": fold, "val_f1": again["val_f1"], "max_gap": gap, "ok": gap == 0.0})
    summary = pd.DataFrame(rows)
    out = fm.out_dir(dataset, KIND)
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
        print(f"\n{len(summary)} CNN-only (text + 37 features) models trained; mean val F1 {summary['val_f1'].mean():.3f} "
              f"(std {summary['val_f1'].std():.3f})")
        return
    summary = run_check(args.pilot, args.seeds, args.folds)
    print(summary.to_string(index=False))
    print(f"\n{int(summary['ok'].sum())} of {len(summary)} rebuilt from their sidecars with a gap of exactly 0")
    sys.exit(0 if summary["ok"].all() else 1)


if __name__ == "__main__":
    main()
