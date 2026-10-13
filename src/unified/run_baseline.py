"""Baseline for the unified model: the same CNN plus the RAW link and page features, no forest, no gates.

Why: a CNN that only sees `<URL>` where the link was is a weak baseline, because the proposed model
also gets link evidence. This baseline gets the same evidence directly: has_url, a page_missing flag
and all 31 link/page features, concatenated to the 64 CNN features before the dense layers. Every gate
is 1.0. If the proposed model (CNN + forest outputs + gates) beats this, the forest and the gating
are what helped, not the extra information.

Held equal to the proposed run (src/unified/run_unified.py), per seed:
    the splits, the validation slice, the Phase A CNN (loaded from models/unified/<tag>), the
    class weights and the Phase C recipe. Only the head's non-text inputs differ.

Raw features are standardized with the mean and std of the Training link rows. Rows with no link
get zeros (and has_url = 0). A missing page feature is set to 0 (the mean) and page_missing = 1.

Usage (run run_unified first, so each seed's Phase A checkpoint exists):
    python -m src.unified.run_baseline
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import DENSE_UNITS, NUM_FILTERS, SEEDS, SEQ_LEN_L
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier, load_phase_a_encoder_weights
from src.features.model.fusion_data import class_weights, load_embedding_layer, load_vocab
from src.features.model.rf_extractor import URL_TEXT_FEATURES
from src.features.model.run_experiment import evaluate_groups, predict_labels, seed_everything
from src.features.model.train import LABEL_TO_IDX, FusionDataset, encode_batch, train_phase_c
from src.unified import unified_rf as urf
from src.unified.build_unified_dataset import OUT_PATH as DEFAULT_DATA
from src.unified.make_splits import SPLITS_DIR
from src.unified.run_unified import DEFAULT_MODEL_DIR, DEFAULT_OUT, VAL_FRACTION_OF_TRAIN, load_part

FEATURES = urf.INPUTS["full"]
PAGE_FEATURES = [f for f in FEATURES if f not in URL_TEXT_FEATURES]
AUX_COLUMNS = ["has_url", "page_missing"] + FEATURES


def fit_scaler(train_link_rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = train_link_rows[FEATURES].mean()                # NaN-aware
    std = train_link_rows[FEATURES].std().replace(0, 1.0).fillna(1.0)
    return mean, std


def raw_aux(df: pd.DataFrame, scaler: tuple[pd.Series, pd.Series]) -> pd.DataFrame:
    mean, std = scaler
    aux = pd.DataFrame(0.0, index=df.index, columns=AUX_COLUMNS)
    aux["has_url"] = df["has_url"].astype(float)
    link = urf.link_mask(df)
    feats = df.loc[link, FEATURES]
    aux.loc[link, "page_missing"] = feats[PAGE_FEATURES].isna().any(axis=1).astype(float)
    aux.loc[link, FEATURES] = ((feats - mean) / std).fillna(0.0)
    return aux


def tensors(df: pd.DataFrame, scaler, word_to_idx: dict):
    token_ids = encode_batch(df["sms_text"].astype(str).tolist(), word_to_idx, SEQ_LEN_L)
    aux = torch.tensor(raw_aux(df, scaler).values, dtype=torch.float32)
    y = torch.tensor([LABEL_TO_IDX[label] for label in df["label"]], dtype=torch.long)
    return token_ids, aux, y


def run_seed(df: pd.DataFrame, seed: int, args) -> tuple[list[dict], dict]:
    seed_everything(seed)
    model_dir = Path(args.model_dir) / "baseline_raw31"
    model_dir.mkdir(parents=True, exist_ok=True)
    phase_a_path = Path(args.model_dir) / args.tag / f"phase_a_seed{seed}.pt"
    if not phase_a_path.exists():
        raise FileNotFoundError(f"{phase_a_path} missing: run src.unified.run_unified first")
    word_to_idx = load_vocab()

    train_full = load_part(df, seed, "train", args.splits_dir)
    test_df = load_part(df, seed, "test", args.splits_dir)
    scaler = fit_scaler(train_full[urf.link_mask(train_full)])

    pos = np.arange(len(train_full))
    train_pos, val_pos = train_test_split(     # the same slice run_unified carves
        pos, test_size=VAL_FRACTION_OF_TRAIN, stratify=train_full["label"], random_state=seed,
    )
    train_ids, train_aux, train_y = tensors(train_full.iloc[train_pos], scaler, word_to_idx)
    val_ids, val_aux, val_y = tensors(train_full.iloc[val_pos], scaler, word_to_idx)
    weights = class_weights(train_y)

    embedding, _ = load_embedding_layer()
    encoder = TextEncoder(embedding, NUM_FILTERS)
    load_phase_a_encoder_weights(encoder, str(phase_a_path))
    model = FusionClassifier(encoder, torch.ones(NUM_FILTERS), torch.ones(len(AUX_COLUMNS)),
                             url_dim=len(AUX_COLUMNS), dense_units=DENSE_UNITS)
    phase_c_path = model_dir / f"phase_c_seed{seed}.pt"
    phase_c_metrics = train_phase_c(
        model,
        DataLoader(FusionDataset(train_ids, train_aux, train_y), batch_size=args.batch_size, shuffle=True),
        DataLoader(FusionDataset(val_ids, val_aux, val_y), batch_size=512),
        epochs=args.epochs, lr=args.lr, checkpoint_path=str(phase_c_path), patience=args.patience,
        class_weights=weights,
    )
    model.load_state_dict(torch.load(phase_c_path, map_location="cpu"))

    test_ids, test_aux, test_y = tensors(test_df, scaler, word_to_idx)
    has_url = test_aux[:, AUX_COLUMNS.index("has_url")].numpy()
    preds = predict_labels(model, test_ids, test_aux)
    rows = [{"seed": seed, "model": "cnn_raw31", **g} for g in evaluate_groups(test_df, test_y.numpy(), preds, has_url)]
    return rows, {"seed": seed, "phase_c_val": phase_c_metrics, "n_train": len(train_y), "n_test": len(test_y)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--tag", default="full", help="run_unified subfolder holding the Phase A checkpoints")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.data, low_memory=False, keep_default_na=False, na_values=[""])
    all_rows, infos = [], []
    for seed in args.seeds:
        rows, info = run_seed(df, seed, args)
        all_rows += rows
        infos.append(info)

    out = args.out / "baseline_raw31"
    out.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(out / "metrics_per_seed.csv", index=False)
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    summary = per_seed.groupby(["model", "group"])[metrics + ["n"]].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "metrics_summary.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "runs": infos}, indent=2))

    pd.set_option("display.width", 220)
    print(summary[["model", "group", "n_mean", "balanced_accuracy_mean", "balanced_accuracy_std", "f1_mean", "f1_std"]]
          .round(4).to_string(index=False))
    print("NOTE: page features are the hybrid_temp placeholder. Not study results.")


if __name__ == "__main__":
    main()
