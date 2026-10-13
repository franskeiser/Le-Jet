"""Compare the four unified-design models on the same test messages, with paired significance tests.

Models (all use the same Phase A CNN weights, splits and test rows):
    cnn_text       the CNN alone, link masked as <URL>             (models/unified/<tag>/phase_a)
    cnn_raw31      CNN + has_url, page_missing and the 31 raw features, no forest, no gates (run_baseline)
    unified_nogates  CNN + forest P, u, c_, every gate 1.0                                   (run_nogates)
    unified_gated  CNN + forest P, u, c_ with static gates                                 (run_unified)

Per seed it predicts every test message with each model, then reports balanced accuracy (mean and
std over seeds) and an exact McNemar test for each pair, on the real test messages (all of them, and
only those with a link). McNemar compares only the messages the two models disagree on: b = first model
right and second wrong, c = the reverse; the two-sided p-value is an exact binomial test of b vs c.

Run run_unified, run_baseline and run_nogates first.

Usage:
    python -m src.unified.compare_models
"""
import argparse
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

from src.config import DENSE_UNITS, NUM_FILTERS, SEEDS
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier
from src.features.model.fusion_data import load_embedding_layer, load_vocab
from src.features.model.run_experiment import REAL_SOURCE, predict_labels, score_group
from src.unified import run_baseline as rb
from src.unified import run_unified as ru
from src.unified import unified_rf as urf
from src.unified.build_unified_dataset import OUT_PATH as DEFAULT_DATA
from src.unified.make_splits import SPLITS_DIR

MODELS = ["cnn_text", "cnn_raw31", "unified_nogates", "unified_gated"]
GROUPS = {"real_all": lambda real, link: real, "real_with_link": lambda real, link: real & link,
          "real_no_link": lambda real, link: real & ~link}


def fusion(path: Path, cnn_gates: torch.Tensor, head_gates: torch.Tensor) -> FusionClassifier:
    embedding, _ = load_embedding_layer()
    model = FusionClassifier(TextEncoder(embedding, NUM_FILTERS), cnn_gates, head_gates,
                             url_dim=len(head_gates), dense_units=DENSE_UNITS)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def predict_all(df: pd.DataFrame, seed: int, args) -> tuple[pd.DataFrame, dict]:
    model_dir = Path(args.model_dir)
    full = model_dir / args.tag
    word_to_idx = load_vocab()
    train_full = ru.load_part(df, seed, "train", args.splits_dir)
    test_df = ru.load_part(df, seed, "test", args.splits_dir)

    embedding, _ = load_embedding_layer()
    cnn = TextEncoder(embedding, NUM_FILTERS)
    cnn.load_state_dict(torch.load(full / f"phase_a_seed{seed}.pt", map_location="cpu"))
    bundle = joblib.load(full / f"unified_rf_seed{seed}.joblib")
    ids, aux, y = ru.tensors(test_df, urf.outputs(bundle, test_df), word_to_idx)
    preds = {"cnn_text": predict_labels(cnn, ids, None)}

    gates = torch.from_numpy(np.load(full / f"gates_seed{seed}.npy").astype(np.float32))
    preds["unified_gated"] = predict_labels(
        fusion(full / f"phase_c_seed{seed}.pt", gates[:NUM_FILTERS], gates[NUM_FILTERS:]), ids, aux)
    ones = torch.ones(NUM_FILTERS)
    preds["unified_nogates"] = predict_labels(
        fusion(model_dir / "nogates" / f"phase_c_seed{seed}.pt", ones, torch.ones(len(ru.AUX_COLUMNS))), ids, aux)

    scaler = rb.fit_scaler(train_full[urf.link_mask(train_full)])
    rids, raux, _ = rb.tensors(test_df, scaler, word_to_idx)
    preds["cnn_raw31"] = predict_labels(
        fusion(model_dir / "baseline_raw31" / f"phase_c_seed{seed}.pt", ones, torch.ones(len(rb.AUX_COLUMNS))), rids, raux)

    table = pd.DataFrame({"seed": seed, "message_id": test_df["message_id"], "y": y.numpy(),
                          "real": (test_df["source"] == REAL_SOURCE).to_numpy(),
                          "has_url": test_df["has_url"].to_numpy() == 1, **preds})
    return table, preds


def mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    a_right, b_right = a == y, b == y
    only_a, only_b = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    p = 1.0 if only_a + only_b == 0 else binomtest(only_a, only_a + only_b, 0.5).pvalue
    return {"only_first_right": only_a, "only_second_right": only_b, "p_value": p}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--model-dir", type=Path, default=ru.DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=ru.DEFAULT_OUT / "comparison")
    args = parser.parse_args()

    df = pd.read_csv(args.data, low_memory=False, keep_default_na=False, na_values=[""])
    tables = [predict_all(df, seed, args)[0] for seed in args.seeds]
    all_preds = pd.concat(tables, ignore_index=True)

    score_rows, test_rows = [], []
    for seed, t in zip(args.seeds, tables):
        for group, select in GROUPS.items():
            sub = t[select(t["real"].to_numpy(), t["has_url"].to_numpy())]
            y = sub["y"].to_numpy()
            for m in MODELS:
                score_rows.append({"seed": seed, "model": m, "group": group, **score_group(y, sub[m].to_numpy())})
            for first, second in combinations(MODELS, 2):
                test_rows.append({"seed": seed, "group": group, "first": first, "second": second,
                                  **mcnemar(y, sub[first].to_numpy(), sub[second].to_numpy())})

    scores = pd.DataFrame(score_rows)
    metrics = ["balanced_accuracy", "f1", "accuracy"]
    summary = scores.groupby(["group", "model"])[metrics + ["n"]].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    tests = pd.DataFrame(test_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    all_preds.to_csv(args.out / "predictions.csv", index=False)
    scores.to_csv(args.out / "metrics_per_seed.csv", index=False)
    summary.to_csv(args.out / "metrics_summary.csv", index=False)
    tests.to_csv(args.out / "mcnemar_per_seed.csv", index=False)

    pd.set_option("display.width", 220)
    order = {m: i for i, m in enumerate(MODELS)}
    show = summary.sort_values(["group", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
    print(show[["group", "model", "n_mean", "balanced_accuracy_mean", "balanced_accuracy_std", "f1_mean"]].round(4).to_string(index=False))
    print("\nMcNemar (exact, per seed; p < 0.05 would mean the two models differ on the messages they disagree on)")
    wide = tests.pivot_table(index=["group", "first", "second"], columns="seed", values="p_value")
    print(wide.round(3).to_string())
    print("\nNOTE: page features are the hybrid_temp placeholder. Not study results.")


if __name__ == "__main__":
    main()
