"""The C-prune fused model: pruned CNN + link lane (docs/c_prune_architecture.md Phase 6).

Step 1, build: per seed, fold and k (48, 32, 16), the fused model of section 2.1, before any fine-tuning:
    pruned CNN (k filters, gate_rf.pruned_phase_a: the Phase A weights, keep_idx from the gate mask sidecar)
    + the 40 link numbers (link_lane.load_lane: P, u, 37 contributions, page_missing)
    -> Concatenate (k + 40) -> Dense 64 (weight decay on its weights) -> ReLU -> dropout 0.5 -> 2 (+ softmax)
The model is fusion_classifier.PrunedFusionClassifier. Dense 64 and the output layer are new, initialised
from the seed: no trained layer reads the 64 Phase A filters except the temporary head, which is dropped.
Each message is lined up with its own link numbers by message_id (fusion_inputs); a missing or extra row stops.
Checked for every seed, fold and k, on the fold's val messages:
  * trainable parameters = section 5.3: Conv 300 x 3 x k + k, Dense (k + 40) x 64 + 64, output 64 x 2 + 2;
    the embedding stays frozen and the temporary head is gone;
  * the CNN inside is the pruned Phase A CNN: same filter scores, bit for bit;
  * the forward pass works on the real inputs: (messages, 2) logits, probabilities summing to 1;
  * the weight decay group holds Dense 64's weights only.
Nothing is trained or saved here (step 2 fine-tunes).

Writes experiments/c_prune/<dataset>/fusion/build_summary.csv.

Usage:
    python -m src.unified.fusion --pilot build
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from src.config import ARTIFACTS_ROOT, CROWD_N_FOLDS, DENSE_UNITS, DENSE_WEIGHT_DECAY, PRUNE_KEEP, SEEDS
from src.data.split_crowd import load_fold
from src.features.model.fusion_classifier import PrunedFusionClassifier
from src.unified.gate_rf import pruned_phase_a
from src.unified.link_lane import LANE, load_lane
from src.unified.text_cnn import dataset_path, tensors, trainable_parameters


def out_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "experiments" / "c_prune" / dataset.stem / "fusion"


def fusion_inputs(part: pd.DataFrame, lane: pd.DataFrame, word_to_idx: Dict[str, int], seq_len: int) -> tuple:
    """(token ids, 40 link numbers, labels) of the rows of `part`, each message with its own link numbers."""
    link = part[["message_id"]].merge(lane, on="message_id", how="left", validate="one_to_one")
    if len(link) != len(part) or link[LANE].isna().any().any() or len(lane) != len(part):
        raise ValueError("the messages and their link numbers do not match one to one")
    ids, labels = tensors(part, word_to_idx, seq_len)
    return ids, torch.tensor(link[LANE].to_numpy(), dtype=torch.float32), labels


def expected_trainable(k: int, embedding_dim: int, kernel_size: int, link: int = len(LANE),
                       dense: int = DENSE_UNITS) -> dict:
    """Section 5.3's count per layer."""
    return {"conv": embedding_dim * kernel_size * k + k, "dense": (k + link) * dense + dense, "output": dense * 2 + 2}


def build_pruned_fusion(seed: int, fold: int, k: int, pilot: bool = False) -> tuple:
    """(fused model, keep_idx, word_to_idx, Phase A sidecar). Dense 64 and the output start from `seed`."""
    pruned, keep, word_to_idx, sidecar = pruned_phase_a(seed, fold, k, pilot)
    torch.manual_seed(seed)
    model = PrunedFusionClassifier(pruned, keep, url_dim=len(LANE), dense_units=DENSE_UNITS)
    return model, keep, word_to_idx, sidecar


def run_build(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    dataset = dataset_path(pilot)
    out = out_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            val = load_fold(seed, fold, pilot=pilot)["val"]
            lane_full, _ = load_lane(seed, fold, pilot)["val"]
            for k in PRUNE_KEEP:
                model, keep, word_to_idx, sidecar = build_pruned_fusion(seed, fold, k, pilot)
                model.eval()
                ids, link, _ = fusion_inputs(val, lane_full, word_to_idx, sidecar["seq_len_l"])
                pruned, _, _, _ = pruned_phase_a(seed, fold, k, pilot)
                expected = expected_trainable(k, model.encoder.embedding.embedding_dim, sidecar["kernel_size"])
                assert model.encoder.temp_head is None and not model.encoder.embedding.weight.requires_grad
                assert trainable_parameters(model) == sum(expected.values()), "trainable parameters differ from 5.3"
                groups = model.parameter_groups(DENSE_WEIGHT_DECAY)
                assert len(groups[0]["params"]) == 1 and groups[0]["params"][0] is model.dense1.weight
                with torch.no_grad():
                    assert torch.equal(model.encoder.encode(ids), pruned.encode(ids)), "not the pruned Phase A CNN"
                    proba = model.predict_proba(ids, link)
                assert proba.shape == (len(val), 2) and torch.allclose(proba.sum(dim=1), torch.ones(len(val)))
                rows.append({"seed": seed, "fold": fold, "k": k, "val_rows": len(val),
                             **{f"{name}_parameters": n for name, n in expected.items()},
                             "trainable_parameters": trainable_parameters(model),
                             "keep_idx": " ".join(map(str, keep))})
            print(f"seed {seed} fold {fold}: built k = {', '.join(map(str, PRUNE_KEEP))}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "build_summary.csv", index=False)
    return summary


STEPS = {"build": run_build}


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("step", choices=list(STEPS))
    a = ap.parse_args(argv)
    summary = STEPS[a.step](a.pilot)
    pd.set_option("display.width", 200)
    print(summary.drop(columns="keep_idx").to_string(index=False))


if __name__ == "__main__":
    main()
