"""Ablation of the proposed model: the same CNN + forest outputs, but every gate fixed at 1.0.

Held equal to the proposed run (src/unified/run_unified.py), per seed: the splits, the validation
slice, the Phase A CNN (loaded from models/unified/<tag>), the final forest and the out-of-fold
P, u, c_ for the Training rows (recomputed with the forest settings stored in the saved forest, which
gives the same folds and the same values), the class weights and the Phase C recipe. Only the gates
differ. Comparing it with the gated model shows what the gating itself contributes.

Usage (run run_unified first):
    python -m src.unified.run_nogates
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import DENSE_UNITS, NUM_FILTERS, SEEDS
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier, load_phase_a_encoder_weights
from src.features.model.fusion_data import class_weights, load_embedding_layer, load_vocab
from src.features.model.rf_extractor import PARAM_GRID
from src.features.model.run_experiment import evaluate_groups, predict_labels, seed_everything
from src.features.model.train import FusionDataset, train_phase_c
from src.unified import unified_rf as urf
from src.unified.build_unified_dataset import OUT_PATH as DEFAULT_DATA
from src.unified.make_splits import SPLITS_DIR
from src.unified.run_unified import AUX_COLUMNS, DEFAULT_MODEL_DIR, DEFAULT_OUT, VAL_FRACTION_OF_TRAIN, load_part, tensors


def run_seed(df: pd.DataFrame, seed: int, args) -> tuple[list[dict], dict]:
    seed_everything(seed)
    full_dir = Path(args.model_dir) / args.tag
    model_dir = Path(args.model_dir) / "nogates"
    model_dir.mkdir(parents=True, exist_ok=True)
    word_to_idx = load_vocab()

    train_full = load_part(df, seed, "train", args.splits_dir)
    test_df = load_part(df, seed, "test", args.splits_dir)
    bundle = joblib.load(full_dir / f"unified_rf_seed{seed}.joblib")
    params = {k: v for k, v in bundle["model"].get_params().items() if k in PARAM_GRID}
    oof = urf.oof_outputs(train_full, bundle["features"], seed, params)

    pos = np.arange(len(train_full))
    train_pos, val_pos = train_test_split(
        pos, test_size=VAL_FRACTION_OF_TRAIN, stratify=train_full["label"], random_state=seed,
    )
    train_ids, train_aux, train_y = tensors(train_full.iloc[train_pos], oof.iloc[train_pos], word_to_idx)
    val_ids, val_aux, val_y = tensors(train_full.iloc[val_pos], oof.iloc[val_pos], word_to_idx)
    weights = class_weights(train_y)

    embedding, _ = load_embedding_layer()
    encoder = TextEncoder(embedding, NUM_FILTERS)
    load_phase_a_encoder_weights(encoder, str(full_dir / f"phase_a_seed{seed}.pt"))
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

    test_ids, test_aux, test_y = tensors(test_df, urf.outputs(bundle, test_df), word_to_idx)
    has_url = test_aux[:, AUX_COLUMNS.index("has_url")].numpy()
    preds = predict_labels(model, test_ids, test_aux)
    rows = [{"seed": seed, "model": "unified_nogates", **g} for g in evaluate_groups(test_df, test_y.numpy(), preds, has_url)]
    return rows, {"seed": seed, "phase_c_val": phase_c_metrics, "rf_params": {k: str(v) for k, v in params.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--tag", default="full", help="run_unified subfolder holding the forest and Phase A checkpoints")
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

    out = args.out / "nogates"
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
