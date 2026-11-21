"""The text lane, Phase A: the full 64-filter CNN on its own (the text-only baseline, and the start of every fused model).

Per seed and fold:
  * text: the fold's own vocabulary + frozen fastText matrix (src/features/build_crowd_embeddings.py) and
    its SEQ_LEN_L (split_report.json). The link becomes <URL>; words outside the train vocabulary are
    dropped (fusion_data.encode_batch_tolerant), as for a new message in the live app;
  * model: TextEncoder, 64 filters of width 3, with its temporary 2-way head. The embedding is FROZEN
    (requires_grad = False), so only Conv + head train;
  * training: train_phase_a with the shared recipe (Adam 1e-3, 30 epochs, patience 3 on val balanced
    accuracy, label smoothing 0.1 from config, class weights). Early stopping reads the fold's val part
    only; test is never read here.
Modality dropout is not used: it hides one lane of a FUSED model, and Phase A has one lane.

Writes, per seed and fold, to models/c_prune/<dataset>/:
    phase_a_seed<S>_fold<F>.pt     the best epoch's weights (saved by train_phase_a)
    phase_a_seed<S>_fold<F>.json   the sidecar: what is needed to rebuild and trust it -- filters, kernel,
                                   SEQ_LEN_L, the embedding folder, sha256 of vocab.json, embedding_matrix.npy
                                   and the .pt, and the best epoch's val metrics
and experiments/c_prune/<dataset>/text_cnn/phase_a_summary.csv.

Step check: every checkpoint is rebuilt from its sidecar alone (load_phase_a), the hashes are compared, and
its val metrics are recomputed; they must equal the ones recorded at training time.

Usage:
    python -m src.unified.text_cnn --pilot train
    python -m src.unified.text_cnn --pilot check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.config import (
    ARTIFACTS_ROOT, CROWD_DATASET_PATH, CROWD_EMBEDDINGS_DIR, CROWD_N_FOLDS, CROWD_PILOT_PATH, CROWD_SPLITS_DIR,
    KERNEL_SIZE, NUM_FILTERS, SEEDS,
)
from src.data.split_crowd import load_fold
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.embedding import build_torch_embedding_layer
from src.features.model.fusion_data import class_weights, encode_batch_tolerant
from src.features.model.train import LABEL_TO_IDX, SmishDataset, train_phase_a

ROOT = Path(__file__).resolve().parents[2]
EPOCHS, LR, PATIENCE, BATCH_SIZE = 30, 1e-3, 3, 64
VAL_METRICS = ["val_accuracy", "val_balanced_accuracy", "val_precision", "val_recall", "val_f1"]


def dataset_path(pilot: bool) -> Path:
    return ROOT / (CROWD_PILOT_PATH if pilot else CROWD_DATASET_PATH)


def model_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "models" / "c_prune" / dataset.stem


def out_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "experiments" / "c_prune" / dataset.stem / "text_cnn"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def embeddings_dir(dataset: Path, seed: int, fold: int) -> Path:
    return ROOT / CROWD_EMBEDDINGS_DIR / dataset.stem / f"seed{seed}_fold{fold}"


def frozen_embedding(emb_dir: Path) -> tuple:
    """(frozen embedding layer, word_to_idx) from a folder holding vocab.json + embedding_matrix.npy."""
    vocab = json.loads((emb_dir / "vocab.json").read_text(encoding="utf-8"))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    embedding = build_torch_embedding_layer(np.load(emb_dir / "embedding_matrix.npy"), word_to_idx)
    embedding.weight.requires_grad = False
    return embedding, word_to_idx


def fold_text_setup(dataset: Path, seed: int, fold: int) -> tuple:
    """(frozen embedding layer, word_to_idx, SEQ_LEN_L) of one fold."""
    embedding, word_to_idx = frozen_embedding(embeddings_dir(dataset, seed, fold))
    report = json.loads((ROOT / CROWD_SPLITS_DIR / dataset.stem / "split_report.json").read_text(encoding="utf-8"))
    seq_len = next(f["seq_len_l"] for f in report["folds"] if f["seed"] == seed and f["fold"] == fold)
    return embedding, word_to_idx, seq_len


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sidecar(checkpoint: Path, emb_dir: Path, seed: int, fold: int, metrics: dict) -> dict:
    sidecar = {
        "seed": seed, "fold": fold, "num_filters": NUM_FILTERS, "kernel_size": KERNEL_SIZE,
        "seq_len_l": metrics["seq_len_l"], "vocab_size": metrics["vocab_size"],
        "embeddings_dir": emb_dir.relative_to(ROOT).as_posix(),
        "sha256": {"vocab.json": sha256(emb_dir / "vocab.json"),
                   "embedding_matrix.npy": sha256(emb_dir / "embedding_matrix.npy"),
                   checkpoint.name: sha256(checkpoint)},
        "trainable_parameters": metrics["trainable_parameters"],
        "val": {k: metrics[k] for k in VAL_METRICS},
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def load_phase_a(checkpoint: Path) -> tuple:
    """(TextEncoder in eval mode, word_to_idx, sidecar), rebuilt from the sidecar alone; refuses a changed file."""
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    emb_dir = ROOT / sidecar["embeddings_dir"]
    for name, digest in sidecar["sha256"].items():
        path = checkpoint if name == checkpoint.name else emb_dir / name
        if sha256(path) != digest:
            raise ValueError(f"{path} changed since the CNN was trained")
    embedding, word_to_idx = frozen_embedding(emb_dir)
    cnn = TextEncoder(embedding, sidecar["num_filters"], sidecar["kernel_size"])
    cnn.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    cnn.eval()
    return cnn, word_to_idx, sidecar


def evaluate(cnn: TextEncoder, ids: torch.Tensor, labels: torch.Tensor) -> dict:
    """The same metrics train_phase_a records for the best epoch."""
    with torch.no_grad():
        preds = torch.argmax(cnn(ids), dim=1).tolist()
    y = labels.tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    return {"val_accuracy": accuracy_score(y, preds), "val_balanced_accuracy": balanced_accuracy_score(y, preds),
            "val_precision": precision, "val_recall": recall, "val_f1": f1}


def tensors(part: pd.DataFrame, word_to_idx: Dict[str, int], seq_len: int) -> tuple:
    ids = encode_batch_tolerant(part["sms_text"].astype(str).tolist(), word_to_idx, seq_len)
    labels = torch.tensor([LABEL_TO_IDX[l] for l in part["label"]], dtype=torch.long)
    return ids, labels


def trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_fold(parts: Dict[str, pd.DataFrame], embedding, word_to_idx: Dict[str, int], seq_len: int, seed: int,
               checkpoint: Path, epochs: int = EPOCHS) -> dict:
    """Trains one Phase A CNN; the best epoch is saved to `checkpoint`. Returns its val metrics."""
    seed_everything(seed)
    train_ids, train_y = tensors(parts["train"], word_to_idx, seq_len)
    val_ids, val_y = tensors(parts["val"], word_to_idx, seq_len)
    cnn = TextEncoder(embedding, NUM_FILTERS)
    assert not cnn.embedding.weight.requires_grad, "the embedding must stay frozen"
    loader = DataLoader(SmishDataset(train_ids, train_y), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    metrics = train_phase_a(cnn, loader, DataLoader(SmishDataset(val_ids, val_y), batch_size=512),
                            epochs=epochs, lr=LR, patience=PATIENCE, checkpoint_path=str(checkpoint),
                            class_weights=class_weights(train_y))
    return {**metrics, "trainable_parameters": trainable_parameters(cnn), "seq_len_l": seq_len,
            "vocab_size": len(word_to_idx)}


def run_train(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    models, out = model_dir(dataset), out_dir(dataset)
    models.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            print(f"--- seed {seed} fold {fold}")
            embedding, word_to_idx, seq_len = fold_text_setup(dataset, seed, fold)
            checkpoint = models / f"phase_a_seed{seed}_fold{fold}.pt"
            metrics = train_fold(load_fold(seed, fold, pilot=pilot), embedding, word_to_idx, seq_len, seed, checkpoint)
            write_sidecar(checkpoint, embeddings_dir(dataset, seed, fold), seed, fold, metrics)
            rows.append({"seed": seed, "fold": fold, **{k: round(v, 4) if isinstance(v, float) else v
                                                         for k, v in metrics.items()}})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "phase_a_summary.csv", index=False)
    return summary


def run_check(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            cnn, word_to_idx, sidecar = load_phase_a(model_dir(dataset) / f"phase_a_seed{seed}_fold{fold}.pt")
            ids, labels = tensors(load_fold(seed, fold, pilot=pilot)["val"], word_to_idx, sidecar["seq_len_l"])
            again = evaluate(cnn, ids, labels)
            gap = max(abs(again[k] - sidecar["val"][k]) for k in VAL_METRICS)
            assert gap < 1e-9, f"seed {seed} fold {fold}: reloaded CNN differs from training (gap {gap})"
            rows.append({"seed": seed, "fold": fold, "val_f1": round(again["val_f1"], 4), "max_gap": gap})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir(dataset) / "phase_a_check.csv", index=False)
    print(f"all {len(rows)} checkpoints reload from their sidecars and repeat their val metrics exactly")
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("step", choices=["train", "check"])
    a = ap.parse_args(argv)
    summary = run_train(a.pilot) if a.step == "train" else run_check(a.pilot)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
