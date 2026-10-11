"""Experiment: raw features straight into the head (each with its own RF gate) instead of P and u.

PROTOTYPE ONLY, synthetic data (src/demo/synthetic.py): what this shows is how the architecture
behaves, not how it performs on real messages.

Two models are compared, both on the demo data and both with the Phase A CNN already trained by
src.demo.train_demo:
    pu   the proposed model: 14 link features + has_url + P + u (models/demo)
    raw  this experiment: 14 link features + has_url + the 17 page/domain/external features,
         every one gated by its own forest importance; no P and u

Page status is not an input to either, but in `raw` the page block is -1 when the page is missing,
which the head can see. During training the page block is hidden at random (modality dropout, same
idea as the `mixed` P/u training) so it learns both situations.

Attribution: gradient x input of the winning class's score (the same method as
src/serving/explain_text.py), summed in absolute value per input group and averaged over test
messages. It says which inputs the model's score is sensitive to, not what is causally "true".

Usage (from the project root, after src.demo.synthetic and src.demo.train_demo):
    python -m src.demo.raw_experiment
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import DENSE_UNITS, NUM_FILTERS, SEQ_LEN_L
from src.demo.synthetic import DEMO_DATA, DEMO_MODEL_DIR, PAGE_COLUMNS, ROOT, SEED
from src.demo.train_demo import aux_matrix, evaluate, load_extractors
from src.features.link_features import LINK_TEXT_FEATURES
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier, load_phase_a_encoder_weights
from src.features.model.fusion_data import AUX_COLUMNS, class_weights, encode_batch_tolerant, load_embedding_layer
from src.features.model.rf_gating_module import extract_cnn_features, fit_and_score_group, fit_and_score_url_group
from src.features.model.run_experiment import predict_labels, seed_everything
from src.features.model.train import LABEL_TO_IDX, FusionDataset, MixedAuxDataset, train_phase_c

RAW_COLUMNS = LINK_TEXT_FEATURES + ["has_url"] + PAGE_COLUMNS
PATH_AND_LENGTH = ["url_length", "path_length", "number_of_slashes", "length_hostname"]
OUT_DIR = ROOT / "models" / "demo_experiments" / "raw"
REPORT_DIR = ROOT / "experiments" / "demo"


def raw_gates(cnn, texts_ids, aux_rf: pd.DataFrame, y: np.ndarray, repeats: int) -> np.ndarray:
    """CNN group, link-spelling group and page group are each fitted as their own forest (one joint
    forest starves the small groups, see rf_gating_module). The page group is fitted only on rows
    whose page was reached; has_url keeps a fixed gate of 1.0."""
    idx_fit, idx_score = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)
    cnn_res = fit_and_score_group(extract_cnn_features(cnn, texts_ids), y, idx_fit, idx_score, SEED,
                                  "balanced_accuracy", repeats, class_weight="balanced")
    link_cols = LINK_TEXT_FEATURES + ["has_url"]
    link_res = fit_and_score_url_group(aux_rf[link_cols].to_numpy(np.float32), link_cols, y, idx_fit, idx_score, SEED, repeats)
    reached = (aux_rf["page_rank"] != -1).to_numpy() | (aux_rf[PAGE_COLUMNS] != -1).any(axis=1).to_numpy()
    page_res = fit_and_score_group(aux_rf[PAGE_COLUMNS].to_numpy(np.float32), y, idx_fit[reached[idx_fit]],
                                   idx_score[reached[idx_score]], SEED, "balanced_accuracy", repeats,
                                   class_weight="balanced")
    return np.concatenate([cnn_res["gates"], link_res["gates"], page_res["gates"]]).astype(np.float32)


def group_of(name: str) -> str:
    if name.startswith("cnn_"):
        return "text (CNN)"
    if name in PATH_AND_LENGTH:
        return "link: URL path and length"
    if name in LINK_TEXT_FEATURES or name == "has_url":
        return "link: other spelling features"
    if name in ("P", "u"):
        return "RF verdict (P, u)"
    return "page / domain / external"


def attribution(model: FusionClassifier, token_ids: torch.Tensor, aux: torch.Tensor, aux_names: list[str]) -> pd.DataFrame:
    """Per message: |gradient x input| of the winning class's score, for every input."""
    model.eval()
    names = [f"cnn_{i}" for i in range(NUM_FILTERS)] + aux_names
    rows = []
    for start in range(0, len(token_ids), 256):
        ids, a = token_ids[start:start + 256], aux[start:start + 256]
        cnn_f = model.encoder.encode(ids).detach().requires_grad_(True)
        a_in = a.clone().requires_grad_(True)
        fused = torch.cat([cnn_f * model.cnn_gates, a_in * model.url_gates], dim=1)
        logits = model.dense2(torch.relu(model.dense1(fused)))
        logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
        contrib = torch.cat([cnn_f.grad * cnn_f.detach(), a_in.grad * a_in.detach()], dim=1).abs()
        rows.append(contrib.numpy())
    return pd.DataFrame(np.concatenate(rows), columns=names)


def group_shares(contrib: pd.DataFrame) -> pd.Series:
    by_group = contrib.T.groupby([group_of(c) for c in contrib.columns]).sum().T
    return (by_group.div(by_group.sum(axis=1), axis=0)).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hide-prob", type=float, default=0.5)
    parser.add_argument("--gate-repeats", type=int, default=10)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED)
    df = pd.read_csv(DEMO_DATA, keep_default_na=False)
    word_to_idx = {w: i for i, w in enumerate(json.load(open(ROOT / "models" / "embeddings" / "vocab.json", encoding="utf-8")))}
    token_ids = encode_batch_tolerant(df["sms_text"].tolist(), word_to_idx, SEQ_LEN_L)
    y_all = df["label"].map(LABEL_TO_IDX).to_numpy()
    y_t = torch.tensor(y_all, dtype=torch.long)

    train_full = np.flatnonzero(df["split"] == "train")
    train_idx, val_idx = train_test_split(train_full, test_size=0.15, stratify=y_all[train_full], random_state=SEED)
    rf_idx, test_idx = np.flatnonzero(df["split"] == "rf_fit"), np.flatnonzero(df["split"] == "test")

    # The CNN from the proposed model's Phase A (same data, same split): comparable by construction.
    embedding, _ = load_embedding_layer()
    cnn = TextEncoder(embedding, NUM_FILTERS)
    cnn.load_state_dict(torch.load(DEMO_MODEL_DIR / "phase_a.pt", map_location="cpu"))

    raw = df[RAW_COLUMNS]
    raw_hidden = raw.copy()
    raw_hidden[PAGE_COLUMNS] = -1
    t_full = torch.tensor(raw.to_numpy(np.float32))
    t_hidden = torch.tensor(raw_hidden.to_numpy(np.float32))

    gates = torch.from_numpy(raw_gates(cnn, token_ids[rf_idx], raw.iloc[rf_idx], y_all[rf_idx], args.gate_repeats))
    gate_table = pd.DataFrame({"input": [f"cnn_{i}" for i in range(NUM_FILTERS)] + RAW_COLUMNS, "gate": gates.numpy()})
    gate_table["group"] = gate_table["input"].map(group_of)
    gate_table.to_csv(REPORT_DIR / "raw_gates.csv", index=False)

    embedding, _ = load_embedding_layer()
    encoder = TextEncoder(embedding, NUM_FILTERS)
    load_phase_a_encoder_weights(encoder, str(DEMO_MODEL_DIR / "phase_a.pt"))
    model = FusionClassifier(encoder, gates[:NUM_FILTERS], gates[NUM_FILTERS:], url_dim=len(RAW_COLUMNS), dense_units=DENSE_UNITS)
    path = OUT_DIR / "phase_c.pt"
    train_phase_c(
        model,
        DataLoader(MixedAuxDataset(token_ids[train_idx], t_hidden[train_idx], t_full[train_idx], y_t[train_idx],
                                   p_alt=1.0 - args.hide_prob), batch_size=64, shuffle=True),
        DataLoader(FusionDataset(token_ids[val_idx], t_full[val_idx], y_t[val_idx]), batch_size=512),
        epochs=args.epochs, lr=1e-3, checkpoint_path=str(path), patience=3, class_weights=class_weights(y_t[train_idx]))
    model.load_state_dict(torch.load(path, map_location="cpu"))

    # ---- test scores, same groups as train_demo ------------------------------------------
    test_df = df.iloc[test_idx]
    raw_metrics = evaluate(test_df, y_all[test_idx], {"raw": predict_labels(model, token_ids[test_idx], t_full[test_idx])})["raw"]
    pu_meta = json.loads((DEMO_MODEL_DIR / "meta.json").read_text())["test_metrics"]["fusion"]
    scores = pd.DataFrame({"pu (proposed)": {g: m["accuracy"] for g, m in pu_meta.items()},
                           "raw (experiment)": {g: m["accuracy"] for g, m in raw_metrics.items()}})

    # ---- attribution on linked test messages ---------------------------------------------
    from src.demo.model import DemoModel
    pu_model = DemoModel().model
    pu_aux = torch.tensor(aux_matrix(df, load_extractors(DEMO_MODEL_DIR)).to_numpy(np.float32))
    linked = test_df["has_url"].to_numpy() == 1
    ambiguous = (test_df["text_kind"] == "ambiguous").to_numpy()
    subsets = {"linked test messages": linked, "linked, neutral wording (text cannot decide)": linked & ambiguous}
    report = {}
    for label, mask in subsets.items():
        sel = test_idx[mask]
        report[label] = {
            "pu (proposed)": group_shares(attribution(pu_model, token_ids[sel], pu_aux[sel], AUX_COLUMNS)),
            "raw (experiment)": group_shares(attribution(model, token_ids[sel], t_full[sel], RAW_COLUMNS)),
        }

    pd.set_option("display.width", 200)
    print("\nTest accuracy (synthetic data):\n", scores.round(4).to_string())
    for label, tables in report.items():
        print(f"\nShare of the model's sensitivity by input group, {label}:")
        print((pd.DataFrame(tables).fillna(0) * 100).round(1).to_string())
    top = attribution(model, token_ids[test_idx[linked]], t_full[test_idx[linked]], RAW_COLUMNS).iloc[:, NUM_FILTERS:].mean()
    print("\nTop raw inputs by mean |gradient x input| (linked test messages):")
    print(pd.DataFrame({"mean_abs_contribution": top, "gate": gates[NUM_FILTERS:].numpy()}, index=RAW_COLUMNS)
          .sort_values("mean_abs_contribution", ascending=False).head(12).round(4).to_string())
    print("\nP and u gates in the proposed model:",
          dict(zip(AUX_COLUMNS[-3:], pu_model.url_gates[-3:].tolist())))
    scores.to_csv(REPORT_DIR / "raw_vs_pu_accuracy.csv")
    pd.concat({k: pd.DataFrame(v) for k, v in report.items()}).to_csv(REPORT_DIR / "raw_vs_pu_attribution.csv")


if __name__ == "__main__":
    main()
