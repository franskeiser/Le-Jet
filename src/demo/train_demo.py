"""Train the demo model on the synthetic data (src/demo/synthetic.py).

PROTOTYPE ONLY: scores printed here come from synthetic data and are not study results.

Same recipe as the real pipeline (src/features/model/run_experiment.py), one seed:
    Phase A  CNN alone      ->  gates from forests on the rf_fit part  ->  Phase C  fusion model
Here P and u come from the demo extractors, routed per row: `full` where the page was
reachable, `url_only` where it was not.

Usage (from the project root):
    python -m src.demo.synthetic        # once: builds the data and the demo extractors
    python -m src.demo.train_demo
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import DENSE_UNITS, NUM_FILTERS, SEQ_LEN_L
from src.demo.synthetic import DEMO_DATA, DEMO_MODEL_DIR, PAGE_COLUMNS, SEED
from src.features.link_features import LINK_TEXT_FEATURES
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier, load_phase_a_encoder_weights
from sklearn.inspection import permutation_importance

from src.features.model.fusion_data import (
    AUX_COLUMNS, RECIPE_COLUMNS, aux_from_inputs, class_weights, encode_batch_tolerant, load_embedding_layer, load_vocab,
)
from src.features.model.rf_extractor import VARIANTS, extract_p_u, route_p_u
from src.features.model.rf_feature_extractor import extract_vector, extractor_gates, route_vector
from src.features.model.rf_gating_module import run_one_seed
from src.features.model.run_experiment import predict_labels, seed_everything
from src.features.model.train import (
    LABEL_TO_IDX, FusionDataset, MixedAuxDataset, SmishDataset, train_phase_a, train_phase_c,
)


def load_extractors(model_dir: Path) -> dict[str, dict]:
    return {v: joblib.load(model_dir / "extractors" / f"{v}.joblib") for v in ("url_only", "full")}


def aux_matrix(df: pd.DataFrame, extractors: dict, routed: bool = True, recipe: str = "link_pu") -> pd.DataFrame:
    """The head's non-text inputs (RECIPE_COLUMNS[recipe] order) for the rows of `df`. routed=True:
    the RF output comes from `full` where the page was reached, else `url_only` (what live serving
    does). routed=False: from `url_only` for every row (what the old SMS data can offer)."""
    inputs = df[LINK_TEXT_FEATURES + PAGE_COLUMNS + ["has_url", "page_fetched"]]
    if recipe == "link_pu":
        out = route_p_u(extractors["url_only"], extractors["full"], inputs) if routed             else extract_p_u(extractors["url_only"], inputs)
        return aux_from_inputs(inputs, out[["P", "u"]], recipe)
    out = route_vector(extractors["url_only"], extractors["full"], inputs) if routed         else extract_vector(extractors["url_only"], inputs)
    return aux_from_inputs(inputs, out, recipe)


def demo_url_gates(recipe: str, extractors: dict, rf_df: pd.DataFrame) -> np.ndarray | None:
    """Gates for inputs that come from the RF extractor (None = fit them on the rf_fit rows as usual).
    rf_vec: the full forest's permutation importance on the rf_fit rows whose page was reached (the demo
    extractors never saw these URLs)."""
    if recipe == "pu_only":
        return np.ones(3, dtype=np.float32)
    if recipe == "rf_vec":
        rows = rf_df[(rf_df["has_url"] == 1) & (rf_df["page_fetched"] == 1)]
        features = VARIANTS["full"]
        result = permutation_importance(
            extractors["full"]["model"], rows[features].to_numpy(np.float32),
            (rows["label"] == "smishing").astype(int).to_numpy(), scoring="balanced_accuracy",
            n_repeats=10, random_state=SEED, n_jobs=-1)
        return np.concatenate([np.ones(3, dtype=np.float32),
                               extractor_gates(pd.Series(result.importances_mean, index=features))])
    return None


def evaluate(test: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray]) -> dict:
    fetched = (test["page_fetched"] == 1).to_numpy()
    ambiguous = (test["text_kind"] == "ambiguous").to_numpy()
    groups = {"all": np.ones(len(test), bool),
              "no_link": (test["has_url"] == 0).to_numpy(),
              "ambiguous_page_reached": ambiguous & fetched,
              "ambiguous_page_missing": ambiguous & ~fetched,
              "link_class_specific_text": (test["text_kind"] == "class_specific").to_numpy(),
              "link_ambiguous_text": (test["text_kind"] == "ambiguous").to_numpy()}
    return {model: {group: {"n": int(mask.sum()), "accuracy": round(float(accuracy_score(y[mask], p[mask])), 4),
                            "balanced_accuracy": round(float(balanced_accuracy_score(y[mask], p[mask])), 4)
                            if len(set(y[mask])) == 2 else None}
                    for group, mask in groups.items()}
            for model, p in preds.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEMO_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEMO_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gate-repeats", type=int, default=10)
    parser.add_argument("--pu-training", choices=["url_only", "routed", "mixed"], default="mixed",
                        help="P and u the head trains on: url_only everywhere (all the old SMS data can offer), "
                             "routed (full where a page was reached), or mixed (random per read)")
    parser.add_argument("--hide-prob", type=float, default=0.5,
                        help="mixed only: chance a row's page evidence is hidden (url_only P and u used instead)")
    parser.add_argument("--recipe", choices=list(RECIPE_COLUMNS), default="rf_vec",
                        help="what the head receives besides the text (see fusion_data.RECIPE_COLUMNS)")
    parser.add_argument("--train-seed", type=int, default=SEED, help="weight init and batch order only")
    parser.add_argument("--out-dir", type=Path, default=None, help="where checkpoints go (default: --model-dir)")
    args = parser.parse_args()
    out_dir = args.out_dir or args.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.train_seed)
    df = pd.read_csv(args.data, keep_default_na=False)
    extractors = load_extractors(args.model_dir)
    aux = aux_matrix(df, extractors, recipe=args.recipe)  # routed: what serving sees; gates, validation, test
    aux_url_only = aux_matrix(df, extractors, routed=False, recipe=args.recipe)
    aux_columns = RECIPE_COLUMNS[args.recipe]
    word_to_idx = load_vocab()
    token_ids = encode_batch_tolerant(df["sms_text"].tolist(), word_to_idx, SEQ_LEN_L)
    y_all = df["label"].map(LABEL_TO_IDX).to_numpy()
    aux_all = torch.tensor(aux.values, dtype=torch.float32)
    aux_url_only_all = torch.tensor(aux_url_only.values, dtype=torch.float32)

    train_full = np.flatnonzero(df["split"] == "train")
    train_idx, val_idx = train_test_split(train_full, test_size=0.15, stratify=y_all[train_full], random_state=SEED)
    rf_idx, test_idx = np.flatnonzero(df["split"] == "rf_fit"), np.flatnonzero(df["split"] == "test")
    y_t = torch.tensor(y_all, dtype=torch.long)
    weights = class_weights(y_t[train_idx])

    # Phase A: the CNN alone
    embedding, _ = load_embedding_layer()
    cnn = TextEncoder(embedding, NUM_FILTERS)
    phase_a_path = out_dir / "phase_a.pt"
    train_phase_a(
        cnn, DataLoader(SmishDataset(token_ids[train_idx], y_t[train_idx]), batch_size=64, shuffle=True),
        DataLoader(SmishDataset(token_ids[val_idx], y_t[val_idx]), batch_size=512),
        epochs=args.epochs, lr=1e-3, patience=3, checkpoint_path=str(phase_a_path), class_weights=weights)
    cnn.load_state_dict(torch.load(phase_a_path, map_location="cpu"))

    # Gates from forests on the rf_fit part (the frozen CNN's features + the 17 inputs)
    rf_df = df.iloc[rf_idx]
    result = run_one_seed(
        cnn, rf_df["sms_text"].tolist(), aux.iloc[rf_idx].to_dict("records"), rf_df["label"].tolist(),
        word_to_idx, SEQ_LEN_L, SEED, n_repeats=args.gate_repeats, out_dir=str(out_dir),
        encode_fn=encode_batch_tolerant,
        url_gates_override=demo_url_gates(args.recipe, extractors, rf_df))
    gates = torch.from_numpy(np.load(result["gate_path"]).astype(np.float32))
    pd.DataFrame({"feature": result["feature_names"], "gate": gates.numpy()}).to_csv(
        out_dir / "gates.csv", index=False)

    # Phase C: the fusion model
    embedding, _ = load_embedding_layer()
    encoder = TextEncoder(embedding, NUM_FILTERS)
    load_phase_a_encoder_weights(encoder, str(phase_a_path))
    model = FusionClassifier(encoder, gates[:NUM_FILTERS], gates[NUM_FILTERS:],
                             url_dim=len(aux_columns), dense_units=DENSE_UNITS)
    phase_c_path = out_dir / "phase_c.pt"
    if args.pu_training == "url_only":
        train_set = FusionDataset(token_ids[train_idx], aux_url_only_all[train_idx], y_t[train_idx])
    elif args.pu_training == "routed":
        train_set = FusionDataset(token_ids[train_idx], aux_all[train_idx], y_t[train_idx])
    else:
        train_set = MixedAuxDataset(token_ids[train_idx], aux_url_only_all[train_idx], aux_all[train_idx],
                                    y_t[train_idx], p_alt=1.0 - args.hide_prob)
    train_phase_c(
        model, DataLoader(train_set, batch_size=64, shuffle=True),
        DataLoader(FusionDataset(token_ids[val_idx], aux_all[val_idx], y_t[val_idx]), batch_size=512),
        epochs=args.epochs, lr=1e-3, checkpoint_path=str(phase_c_path), patience=3, class_weights=weights)
    model.load_state_dict(torch.load(phase_c_path, map_location="cpu"))

    # Test part: scored once, at the end
    test_df = df.iloc[test_idx]
    preds = {"cnn_only": predict_labels(cnn, token_ids[test_idx], None),
             "fusion": predict_labels(model, token_ids[test_idx], aux_all[test_idx])}
    metrics = evaluate(test_df, y_all[test_idx], preds)
    meta = {"WARNING": "Synthetic prototype data. These scores are NOT study results.",
            "seed": SEED, "train_seed": args.train_seed, "recipe": args.recipe, "pu_training": args.pu_training,
            "hide_prob": args.hide_prob if args.pu_training == "mixed" else None, "aux_columns": aux_columns, "n_train": len(train_idx), "n_test": len(test_idx),
            "test_metrics": metrics}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
