"""McNemar's test between the trained models, per seed, from the saved checkpoints.

For two models scored on the same Test messages, only the messages where exactly one is right
carry information: b = A right / B wrong, c = A wrong / B right. The exact two-sided binomial
test asks whether b and c differ more than chance (docs/dataset_and_training_plan.md 7.4).

Run run_experiment first (it writes the checkpoints). No training happens here.

Usage (from the project root):
    python -m src.features.model.mcnemar
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

from src.config import DENSE_UNITS, NUM_FILTERS, SEEDS
from src.data.split_dataset import load_split_df
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier
from src.features.model.fusion_data import ROOT, load_embedding_layer, load_seed_inputs
from src.features.model.run_experiment import (
    DEFAULT_MODEL_DIR, DEFAULT_OUT, REAL_SOURCE, VARIANT_COLUMNS, VARIANT_RECIPE, columns_index, predict_labels,
)

# (model A, model B): is A's accuracy different from B's?
COMPARISONS = [
    ("with_pu", "cnn_only", "gated_with_pu"),
    ("no_pu", "cnn_only", "gated_no_pu"),
    ("with_pu", "gated_no_pu", "gated_with_pu"),
    ("pu_only", "cnn_only", "gated_pu_only"),
    ("rf_vec", "cnn_only", "gated_rf_vec"),
    ("rf_vec", "gated_with_pu", "gated_rf_vec"),   # is the RF-as-feature-extractor better than the first design?
    ("rf_vec", "gated_pu_only", "gated_rf_vec"),
]


def load_cnn(model_dir: Path, variant: str, seed: int) -> TextEncoder:
    embedding, _ = load_embedding_layer()
    cnn = TextEncoder(embedding, NUM_FILTERS)
    cnn.load_state_dict(torch.load(model_dir / variant / f"phase_a_seed{seed}.pt", map_location="cpu"))
    return cnn.eval()


def load_fusion(model_dir: Path, variant: str, seed: int) -> FusionClassifier:
    embedding, _ = load_embedding_layer()
    gates = torch.from_numpy(np.load(model_dir / variant / f"gates_seed{seed}.npy").astype(np.float32))
    columns = VARIANT_COLUMNS[variant]
    model = FusionClassifier(TextEncoder(embedding, NUM_FILTERS), gates[:NUM_FILTERS], gates[NUM_FILTERS:],
                             url_dim=len(columns), dense_units=DENSE_UNITS)
    model.load_state_dict(torch.load(model_dir / variant / f"phase_c_seed{seed}.pt", map_location="cpu"))
    return model.eval()


def test_predictions(model_dir: Path, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-message correctness of every trained model for one seed, plus the has_url flag.
    Variants whose checkpoints are missing are skipped."""
    test_df = load_split_df(seed, "test")
    ids, base_aux, y = load_seed_inputs(seed, "test", test_df)
    y = y.numpy()
    out = pd.DataFrame({"real": (test_df["source"] == REAL_SOURCE).to_numpy(),
                        "has_url": base_aux[:, columns_index(["has_url"])[0]].numpy() == 1})
    out["cnn_only"] = predict_labels(load_cnn(model_dir, "with_pu", seed), ids, None) == y
    for variant in VARIANT_COLUMNS:
        if not (model_dir / variant / f"phase_c_seed{seed}.pt").exists():
            continue
        recipe = VARIANT_RECIPE[variant]
        _, aux, _ = load_seed_inputs(seed, "test", test_df, recipe)
        sub = aux[:, columns_index(VARIANT_COLUMNS[variant], recipe)]
        out[f"gated_{variant}"] = predict_labels(load_fusion(model_dir, variant, seed), ids, sub) == y
    return out, y


def mcnemar_row(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    b = int((correct_a & ~correct_b).sum())  # only A right
    c = int((~correct_a & correct_b).sum())  # only B right
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"n": len(correct_a), "acc_a": correct_a.mean(), "acc_b": correct_b.mean(),
            "only_a_right": b, "only_b_right": c, "p_value": p}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "mcnemar.csv")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        preds, _ = test_predictions(args.model_dir, seed)
        groups = {"real_all": preds["real"], "real_with_link": preds["real"] & preds["has_url"],
                  "real_no_link": preds["real"] & ~preds["has_url"]}
        for _, a, b in COMPARISONS:
            if a not in preds or b not in preds:
                continue
            for group, mask in groups.items():
                rows.append({"seed": seed, "a": a, "b": b, "group": group,
                             **mcnemar_row(preds.loc[mask, a].to_numpy(), preds.loc[mask, b].to_numpy())})
    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
