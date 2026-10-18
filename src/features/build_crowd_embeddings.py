"""Vocabulary + frozen fastText matrix for every C-prune fold (docs/c_prune_architecture.md Phase 2).

Unlike src/features/build_embeddings.py (one vocabulary over all splits), the study builds the vocabulary
from each fold's TRAIN part only, so no val or test word shapes it. Words outside it are dropped at encoding
time (fusion_data.encode_batch_tolerant), the same as for a fresh message in the live app. Vectors: the same
English -> Tagalog -> English-subword order as build_embeddings; the matrix stays frozen in every model.

Reads the split files of src/data/split_crowd.py. Writes, per fold:
    <embeddings_dir>/<dataset name>/seed<S>_fold<F>/vocab.json, embedding_matrix.npy

Usage:
    python -m src.features.build_crowd_embeddings --pilot
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.config import CROWD_EMBEDDINGS_DIR, CROWD_N_FOLDS, CROWD_SPLITS_DIR, SEEDS
from src.data.split_crowd import load_fold
from src.features.build_embeddings import DEFAULT_PRIMARY, DEFAULT_SECONDARY, build_vocab
from src.features.model.embedding import MmapFastTextVectors, build_embedding_matrix

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    a = ap.parse_args()
    for path in (DEFAULT_PRIMARY, DEFAULT_SECONDARY):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Unpack the .bin.gz first (see README).")
    primary = MmapFastTextVectors(str(DEFAULT_PRIMARY))
    secondary = MmapFastTextVectors(str(DEFAULT_SECONDARY))

    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            parts = load_fold(seed, fold, pilot=a.pilot, splits_dir=CROWD_SPLITS_DIR)
            vocab, _ = build_vocab(parts["train"]["sms_text"].astype(str).tolist())
            matrix, _, sources = build_embedding_matrix(vocab, primary, secondary)
            dataset = "crowd_pilot_synthetic" if a.pilot else "crowd_smishing_v1"
            out = ROOT / CROWD_EMBEDDINGS_DIR / dataset / f"seed{seed}_fold{fold}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "embedding_matrix.npy", matrix)
            (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
            print(f"seed {seed} fold {fold}: {len(vocab)} tokens {sources} -> {out}")


if __name__ == "__main__":
    main()
