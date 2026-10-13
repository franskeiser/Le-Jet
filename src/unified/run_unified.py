"""Train the RF extractor and the CNN fusion model together, on ONE dataset and ONE set of splits.

Per seed (docs/unified_architecture.md):
    RF       tune on the training link rows (grouped CV), then
             - out-of-fold P, u, c_ for every training row (a row is never scored by a forest that saw it),
             - one final forest on all training link rows, which scores rf_fit, test and live links.
    Phase A  CNN alone on Training minus a validation slice, early stopping on the slice.
    Gates    CNN gates: a forest on the frozen CNN's features of the rf_fit rows (held out from the RF).
             RF-output gates: the final forest's permutation importance on those same rf_fit rows.
             has_url, P and u keep a fixed gate of 1.0.
    Phase C  fusion head on the out-of-fold RF outputs. Validation rows use out-of-fold outputs too.
    Test     everything above is fixed before the Test split is read, once.

The rf_fit split of the old design is kept as the held-out slice that fits the gates, so the split
files (data/splits) are unchanged.

--rf-inputs full      the forest reads link text + page/domain features (31 inputs)
--rf-inputs url_only  link text only (14 inputs); the page-inspection ablation

Usage (from the project root):
    python -m src.unified.run_unified
    python -m src.unified.run_unified --seeds 42 --epochs 3 --no-tune
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import ARTIFACTS_ROOT, DENSE_UNITS, NUM_FILTERS, SEEDS, SEQ_LEN_L
from src.unified.build_unified_dataset import OUT_PATH as DEFAULT_DATA
from src.data.split_dataset import load_split_ids
from src.unified import unified_rf as urf
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier, load_phase_a_encoder_weights
from src.features.model.fusion_data import RECIPE_COLUMNS, class_weights, load_embedding_layer, load_vocab
from src.features.model.rf_feature_extractor import extractor_gates
from src.features.model.rf_gating_module import run_one_seed
from src.features.model.run_experiment import evaluate_groups, predict_labels, seed_everything
from src.features.model.train import (
    LABEL_TO_IDX, FusionDataset, SmishDataset, encode_batch, train_phase_a, train_phase_c,
)

AUX_COLUMNS = RECIPE_COLUMNS["rf_vec"]  # has_url, P, u, then the 31 contributions
DEFAULT_OUT = ARTIFACTS_ROOT / "experiments" / "unified"
DEFAULT_MODEL_DIR = ARTIFACTS_ROOT / "models" / "unified"
VAL_FRACTION_OF_TRAIN = 0.15


def load_part(df: pd.DataFrame, seed: int, name: str) -> pd.DataFrame:
    """Rows of one split. Redacted-link rows are dropped (docs/training_rules.md rule 1)."""
    ids = load_split_ids(seed, name)
    part = df[df["message_id"].isin(ids) & (df["link_status"] != "redacted")]
    return part.reset_index(drop=True)


def tensors(df: pd.DataFrame, rf_out: pd.DataFrame, word_to_idx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = encode_batch(df["sms_text"].astype(str).tolist(), word_to_idx, SEQ_LEN_L)
    aux = pd.concat([df[["has_url"]], rf_out], axis=1)[AUX_COLUMNS]
    y = torch.tensor([LABEL_TO_IDX[label] for label in df["label"]], dtype=torch.long)
    return token_ids, torch.tensor(aux.values, dtype=torch.float32), y


def rf_test_metrics(bundle: dict, test_df: pd.DataFrame, rf_out: pd.DataFrame) -> dict:
    mask = urf.link_mask(test_df)
    if not mask.any():
        return {}
    y, p = urf.labels(test_df[mask]), rf_out.loc[mask, "P"].to_numpy()
    return {"rf_test_links": int(mask.sum()), "rf_test_f1": float(f1_score(y, p >= 0.5)),
            "rf_test_auc": float(roc_auc_score(y, p))}


def run_seed(df: pd.DataFrame, seed: int, args) -> tuple[list[dict], dict]:
    seed_everything(seed)
    inputs = urf.INPUTS[args.rf_inputs]
    model_dir = Path(args.model_dir) / args.rf_inputs
    model_dir.mkdir(parents=True, exist_ok=True)
    word_to_idx = load_vocab()

    train_full = load_part(df, seed, "train")
    rf_fit_df = load_part(df, seed, "rf_fit")
    test_df = load_part(df, seed, "test")
    label_col = "label"

    # ---- RF extractor: settings, out-of-fold outputs for training rows, final forest ----
    if args.no_tune:
        params, grid = dict(urf.DEFAULT_PARAMS), None
    else:
        params, grid = urf.tune(train_full, inputs, seed)
        grid.assign(seed=seed).to_csv(model_dir / f"rf_tuning_seed{seed}.csv", index=False)
    print(f"[seed {seed}] RF settings {params}")
    oof = urf.oof_outputs(train_full, inputs, seed, params)
    bundle = urf.fit(train_full, inputs, seed, params)
    joblib.dump(bundle, model_dir / f"unified_rf_seed{seed}.joblib")

    # Validation slice for early stopping, carved from Training (its RF outputs stay out-of-fold).
    pos = np.arange(len(train_full))
    train_pos, val_pos = train_test_split(
        pos, test_size=VAL_FRACTION_OF_TRAIN, stratify=train_full[label_col], random_state=seed,
    )
    train_ids, train_aux, train_y = tensors(train_full.iloc[train_pos], oof.iloc[train_pos], word_to_idx)
    val_ids, val_aux, val_y = tensors(train_full.iloc[val_pos], oof.iloc[val_pos], word_to_idx)
    weights = class_weights(train_y)
    print(f"[seed {seed}] train {len(train_y)} / val {len(val_y)}, class weights {weights.tolist()}")

    # ---- Phase A: CNN alone -------------------------------------------------------------
    embedding, _ = load_embedding_layer()
    cnn = TextEncoder(embedding, NUM_FILTERS)
    phase_a_path = model_dir / f"phase_a_seed{seed}.pt"
    phase_a_metrics = train_phase_a(
        cnn,
        DataLoader(SmishDataset(train_ids, train_y), batch_size=args.batch_size, shuffle=True),
        DataLoader(SmishDataset(val_ids, val_y), batch_size=512),
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        checkpoint_path=str(phase_a_path), class_weights=weights,
    )
    cnn.load_state_dict(torch.load(phase_a_path, map_location="cpu"))

    # ---- Gates: both fit on the rf_fit rows, which neither the RF nor the CNN trained on ----
    rf_fit_out = urf.outputs(bundle, rf_fit_df)
    importance = urf.gate_importance(bundle, rf_fit_df, seed)
    url_gates = np.concatenate([np.ones(3, dtype=np.float32), extractor_gates(importance)])
    _, rf_fit_aux, _ = tensors(rf_fit_df, rf_fit_out, word_to_idx)
    gate_result = run_one_seed(
        cnn, rf_fit_df["sms_text"].astype(str).tolist(),
        pd.DataFrame(rf_fit_aux.numpy(), columns=AUX_COLUMNS).to_dict("records"),
        rf_fit_df[label_col].tolist(), word_to_idx, SEQ_LEN_L, seed, n_repeats=args.gate_repeats,
        out_dir=str(model_dir), url_gates_override=url_gates,
    )
    gates = torch.from_numpy(np.load(gate_result["gate_path"]).astype(np.float32))
    pd.DataFrame({"feature": gate_result["feature_names"], "gate": gates.numpy()}).to_csv(
        model_dir / f"gates_seed{seed}.csv", index=False)
    cnn_gates, head_gates = gates[:NUM_FILTERS], gates[NUM_FILTERS:]

    # ---- Phase C: fusion head ------------------------------------------------------------
    embedding, _ = load_embedding_layer()
    encoder = TextEncoder(embedding, NUM_FILTERS)
    load_phase_a_encoder_weights(encoder, str(phase_a_path))
    model = FusionClassifier(encoder, cnn_gates, head_gates, url_dim=len(AUX_COLUMNS), dense_units=DENSE_UNITS)
    phase_c_path = model_dir / f"phase_c_seed{seed}.pt"
    phase_c_metrics = train_phase_c(
        model,
        DataLoader(FusionDataset(train_ids, train_aux, train_y), batch_size=args.batch_size, shuffle=True),
        DataLoader(FusionDataset(val_ids, val_aux, val_y), batch_size=512),
        epochs=args.epochs, lr=args.lr, checkpoint_path=str(phase_c_path), patience=args.patience,
        class_weights=weights,
    )
    model.load_state_dict(torch.load(phase_c_path, map_location="cpu"))

    # ---- Test split (first and only time it is read) -------------------------------------
    test_out = urf.outputs(bundle, test_df)
    test_ids, test_aux, test_y = tensors(test_df, test_out, word_to_idx)
    has_url = test_aux[:, AUX_COLUMNS.index("has_url")].numpy()
    y_true = test_y.numpy()

    rows = []
    for model_name, preds in [
        ("cnn_only", predict_labels(cnn, test_ids, None)),
        (f"unified_{args.rf_inputs}", predict_labels(model, test_ids, test_aux)),
    ]:
        for group in evaluate_groups(test_df, y_true, preds, has_url):
            rows.append({"seed": seed, "model": model_name, **group})

    info = {
        "seed": seed, "rf_inputs": args.rf_inputs, "rf_params": {k: str(v) for k, v in params.items()},
        "phase_a_val": phase_a_metrics, "phase_c_val": phase_c_metrics,
        "n_train": len(train_y), "n_val": len(val_y), "n_rf_fit": len(rf_fit_df), "n_test": len(y_true),
        **rf_test_metrics(bundle, test_df, test_out),
    }
    return rows, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--rf-inputs", choices=list(urf.INPUTS), default="full")
    parser.add_argument("--no-tune", action="store_true", help="skip the RF grid search, use DEFAULT_PARAMS")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gate-repeats", type=int, default=30)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.data, low_memory=False, keep_default_na=False, na_values=[""])
    all_rows, infos = [], []
    for seed in args.seeds:
        rows, info = run_seed(df, seed, args)
        all_rows += rows
        infos.append(info)

    out = args.out / args.rf_inputs
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
