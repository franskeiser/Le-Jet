"""The C-prune text lane, Phase A: the full 64-filter CNN on its own (docs/c_prune_architecture.md Phase 4).

Per seed and fold:
  * text: the fold's own vocabulary + frozen fastText matrix (src/features/build_crowd_embeddings.py) and
    its SEQ_LEN_L (split_report.json). The link becomes <URL>; words outside the train vocabulary are
    dropped (fusion_data.encode_batch_tolerant), as for a new message in the live app;
  * model: TextEncoder, 64 filters of width 3, with its temporary 2-way head. The embedding is FROZEN
    (requires_grad = False), so only Conv + head train;
  * training: train_phase_a with the shared recipe (Adam 1e-3, 30 epochs, patience 3 on val balanced
    accuracy, label smoothing 0.1 from config, class weights). Early stopping reads the fold's val part
    only; test is never read here.
Modality dropout is not used: it hides one lane of the FUSED model (Phase 6), and Phase A has one lane.

Writes models/c_prune/<dataset>/phase_a_seed<S>_fold<F>.pt (the best epoch, by train_phase_a) and
experiments/c_prune/<dataset>/text_cnn/phase_a_summary.csv.

Usage:
    python -m src.unified.text_cnn --pilot train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import (
    ARTIFACTS_ROOT, CROWD_DATASET_PATH, CROWD_EMBEDDINGS_DIR, CROWD_N_FOLDS, CROWD_PILOT_PATH, CROWD_SPLITS_DIR,
    NUM_FILTERS, SEEDS,
)
from src.data.split_crowd import load_fold
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.embedding import build_torch_embedding_layer
from src.features.model.fusion_data import class_weights, encode_batch_tolerant
from src.features.model.train import LABEL_TO_IDX, SmishDataset, train_phase_a

ROOT = Path(__file__).resolve().parents[2]
EPOCHS, LR, PATIENCE, BATCH_SIZE = 30, 1e-3, 3, 64


def dataset_path(pilot: bool) -> Path:
    return ROOT / (CROWD_PILOT_PATH if pilot else CROWD_DATASET_PATH)


def model_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "models" / "c_prune" / dataset.stem


def out_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "experiments" / "c_prune" / dataset.stem / "text_cnn"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def fold_text_setup(dataset: Path, seed: int, fold: int) -> tuple:
    """(frozen embedding layer, word_to_idx, SEQ_LEN_L) of one fold."""
    emb_dir = ROOT / CROWD_EMBEDDINGS_DIR / dataset.stem / f"seed{seed}_fold{fold}"
    vocab = json.loads((emb_dir / "vocab.json").read_text(encoding="utf-8"))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    embedding = build_torch_embedding_layer(np.load(emb_dir / "embedding_matrix.npy"), word_to_idx)
    embedding.weight.requires_grad = False
    report = json.loads((ROOT / CROWD_SPLITS_DIR / dataset.stem / "split_report.json").read_text(encoding="utf-8"))
    seq_len = next(f["seq_len_l"] for f in report["folds"] if f["seed"] == seed and f["fold"] == fold)
    return embedding, word_to_idx, seq_len


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
            metrics = train_fold(load_fold(seed, fold, pilot=pilot), embedding, word_to_idx, seq_len, seed,
                                 models / f"phase_a_seed{seed}_fold{fold}.pt")
            rows.append({"seed": seed, "fold": fold, **{k: round(v, 4) if isinstance(v, float) else v
                                                         for k, v in metrics.items()}})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "phase_a_summary.csv", index=False)
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("step", choices=["train"])
    a = ap.parse_args(argv)
    summary = run_train(a.pilot)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
