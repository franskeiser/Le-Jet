"""Score the trained models on a fresh set (docs/fresh_set_spec.md) of live links.

This is the test of the web-inspection claim. Nothing here trains anything; it loads the
frozen RF extractors and the fusion checkpoints written by run_experiment.

Two questions, in this order:
  1. Extractor alone: on links whose page was fetched, does the `full` extractor (page, domain
     and WHOIS features) classify better than `url_only` (link text only)? No fusion head involved.
  2. Fusion model: does feeding it the RF output from `full` (routed, route_vector) beat feeding it
     the `url_only` output? The head is trained on a mix of both (MixedAuxDataset) only if the
     training data has page evidence; trained on old SMS data alone it has never seen `full` values,
     so a weak result here can come from that mismatch and does not undo result 1.

Groups: no_link, link_dead (link, page not fetched), link_live (page fetched). Claims about web
inspection belong to link_live only (plan section 9). Also reports how many links were alive.

Usage (from the project root):
    python -m src.features.model.evaluate_fresh data/fresh/fresh_2026-11-01.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import SEEDS
from src.features.fresh_set import load_fresh
from src.features.model.fusion_data import ROOT, aux_from_inputs, encode_batch_tolerant, load_vocab
from src.features.model.mcnemar import load_cnn, load_fusion, mcnemar_row
from src.features.model.rf_extractor import load_extractor
from src.features.model.rf_feature_extractor import extract_vector, route_vector
from src.features.model.run_experiment import DEFAULT_MODEL_DIR, predict_labels, score_group

DEFAULT_OUT = ROOT / "experiments" / "fresh"
VARIANT = "rf_vec"  # the final architecture (docs/final_model.md)
RECIPE = "rf_vec"
# (model A, model B, group): same rows, is A's accuracy different from B's?
COMPARISONS = [
    ("extractor_url_only", "extractor_full", "link_live"),
    ("fusion_url_only_pu", "fusion_routed_pu", "link_live"),
    ("cnn_only", "fusion_routed_pu", "link_live"),
    ("cnn_only", "fusion_url_only_pu", "link_live"),
]


def aux_tensor(df: pd.DataFrame, rf_output: pd.DataFrame) -> torch.Tensor:
    frame = aux_from_inputs(df.reset_index(drop=True), rf_output.reset_index(drop=True), RECIPE)
    return torch.tensor(frame.values, dtype=torch.float32)


def seed_predictions(df: pd.DataFrame, seed: int, model_dir: Path, token_ids: torch.Tensor) -> pd.DataFrame:
    """0/1 prediction (1 = smishing) of every model for every row. -1 where a model does not
    apply (the extractors on messages without a link, `full` on rows with no fetched page)."""
    url_only, full = load_extractor("url_only", seed), load_extractor("full", seed)
    pu_url_only = extract_vector(url_only, df)
    pu_full = extract_vector(full, df[df["page_fetched"] == 1])
    pu_routed = route_vector(url_only, full, df)

    preds = pd.DataFrame(index=df.index)
    linked = (df["has_url"] == 1).to_numpy()
    preds["extractor_url_only"] = np.where(linked, (pu_url_only["P"] >= 0.5).astype(int), -1)
    full_p = pu_full["P"].reindex(df.index)
    preds["extractor_full"] = np.where(full_p.notna(), (full_p >= 0.5).astype(int), -1)

    preds["cnn_only"] = predict_labels(load_cnn(model_dir, VARIANT, seed), token_ids, None)
    fusion = load_fusion(model_dir, VARIANT, seed)
    preds["fusion_url_only_pu"] = predict_labels(fusion, token_ids, aux_tensor(df, pu_url_only))
    preds["fusion_routed_pu"] = predict_labels(fusion, token_ids, aux_tensor(df, pu_routed))
    return preds


def group_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    linked = (df["has_url"] == 1).to_numpy()
    live = (df["page_fetched"] == 1).to_numpy()
    return {"all": np.ones(len(df), bool), "no_link": ~linked, "link_dead": linked & ~live, "link_live": live}


def liveness(df: pd.DataFrame) -> pd.DataFrame:
    linked = df[df["has_url"] == 1]
    out = linked.groupby("label").agg(
        links=("message_id", "size"), fetched=("page_fetched", "sum"),
        median_hours_to_visit=("hours_to_visit", "median"),
    )
    out["share_alive"] = out["fetched"] / out["links"]
    statuses = linked.groupby(["label", "fetch_status"]).size().unstack(fill_value=0).reset_index()
    return out.reset_index().merge(statuses, on="label")


def evaluate(df: pd.DataFrame, seeds: list[int], model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = (df["label"] == "smishing").astype(int).to_numpy()
    token_ids = encode_batch_tolerant(df["sms_text"].astype(str).tolist(), load_vocab())
    masks = group_masks(df)

    metric_rows, test_rows = [], []
    for seed in seeds:
        preds = seed_predictions(df, seed, model_dir, token_ids)
        for model in preds.columns:
            for group, mask in masks.items():
                applies = mask & (preds[model].to_numpy() != -1)
                metric_rows.append({"seed": seed, "model": model, "group": group,
                                    **score_group(y[applies], preds[model].to_numpy()[applies])})
        for a, b, group in COMPARISONS:
            applies = masks[group] & (preds[a].to_numpy() != -1) & (preds[b].to_numpy() != -1)
            test_rows.append({"seed": seed, "a": a, "b": b, "group": group,
                              **mcnemar_row(preds[a].to_numpy()[applies] == y[applies],
                                            preds[b].to_numpy()[applies] == y[applies])})
    return pd.DataFrame(metric_rows), pd.DataFrame(test_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = load_fresh(str(args.csv))  # raises, listing every problem, if the file breaks the spec
    metrics, tests = evaluate(df, args.seeds, args.model_dir)
    alive = liveness(df)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out / "metrics_per_seed.csv", index=False)
    tests.to_csv(args.out / "mcnemar.csv", index=False)
    alive.to_csv(args.out / "liveness.csv", index=False)

    pd.set_option("display.width", 220)
    print("Links alive:\n", alive.round(3).to_string(index=False))
    summary = metrics.groupby(["model", "group"])[["n", "balanced_accuracy", "f1"]].mean().round(4)
    print("\nMean over seeds:\n", summary.to_string())
    print("\nMcNemar:\n", tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
