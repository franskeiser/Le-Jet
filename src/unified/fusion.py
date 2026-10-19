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

Step 2, train: fine-tune every model of step 1 with the shared recipe (Phase 0), everything except the Link RF
(its 40 numbers are fixed inputs) and the embedding (frozen for every model):
  * Adam, lr 1e-3, weight decay DENSE_WEIGHT_DECAY on Dense 64's weights only (parameter_groups);
  * up to 30 epochs, batches of 64, early stopping with patience 3 on val balanced accuracy, the best epoch kept
    (as train_phase_c); label smoothing 0.1 (config), class weights from the train labels;
  * modality dropout, train only, redrawn every epoch with a generator seeded by the seed (as
    run_fused_lean.fit_head): a message's page is hidden with p = 0.25, meaning its 40 numbers are swapped for
    its blank-page version (link_lane: page columns -1, page_missing = 1, scored by the same forest), and its
    text is hidden with p = 0.10 (its k filter scores set to 0).
Train rows read their out-of-fold link numbers (no row is scored by a forest that saw it); val reads the final
forest's numbers, page as inspected, no dropout. Test is never read here.
The best epoch's weights go to models/c_prune/<dataset>/c_prune_seed<S>_fold<F>_k<K>.pt (the sidecar is a later
step). Per model the summary records the best epoch's val metrics (accuracy, balanced accuracy, precision,
recall, F1, log-loss), the epochs run, and the wall-clock training time (RQ2, "fine-tuning").

Step 3, choose: the final k, on validation only (decided 2026-10-19): the one-standard-error rule (Breiman et
al. 1984; Hastie et al., The Elements of Statistical Learning, 7.10). Per k, the mean val F1 over every seed and
fold of the train step and its standard error (std / sqrt(seed-folds)); the best k has the highest mean (a tie
goes to the smaller k); the final k is the SMALLEST k whose mean is at least the best mean minus the best k's
standard error. So a smaller, faster model wins when the accuracy gap is within noise (RQ2). One k for every
seed and fold. Re-run after the train step changes (e.g. a new ranking rule).

Step 4, save: per seed, fold and k, a sidecar c_prune_seed<S>_fold<F>_k<K>.json next to the checkpoint, so the
model can be rebuilt and trusted from it alone: keep_idx, the ranking rule, whether this k is the chosen final
k, the architecture (incl. the 40 link inputs in order), the recipe, the trainable parameters, the val metrics,
and the file names + sha256 of the checkpoint, gate mask, Link RF (link_rf_seed<S>_fold<F>.joblib, saved in
Phase 3), Phase A checkpoint and embedding. load_c_prune rebuilds the model and the Link RF and refuses the
bundle when any of those files changed; predict_messages runs raw rows (text + 37 link features) through both.
Checked for every model: the saved weights repeat the train step's val metrics, and the rebuilt bundle, fed the
RAW val rows (the saved Link RF making the 40 numbers again), repeats them exactly.

Writes experiments/c_prune/<dataset>/fusion/build_summary.csv (build), train_summary.csv (train),
final_k.json (choose: the rule, the per-k numbers, the chosen k) and save_summary.csv (save).

Usage:
    python -m src.unified.fusion --pilot build
    python -m src.unified.fusion --pilot train
    python -m src.unified.fusion --pilot choose
    python -m src.unified.fusion --pilot save
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss, precision_recall_fscore_support,
)
from torch import nn

from src.config import (
    ARTIFACTS_ROOT, CROWD_N_FOLDS, DENSE_UNITS, DENSE_WEIGHT_DECAY, LABEL_SMOOTHING, PRUNE_KEEP, SEEDS,
)
from src.data.split_crowd import load_fold
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import PrunedFusionClassifier
from src.features.model.fusion_data import class_weights
from src.unified.gate_rf import gate_mask_path, load_gate_mask, pruned_phase_a
from src.unified.link_lane import LANE, lane_numbers, load_lane
from src.unified.prepare_web_37 import FEATURES_37
from src.unified.text_cnn import (
    BATCH_SIZE, EPOCHS, LR, PATIENCE, ROOT, dataset_path, frozen_embedding, model_dir, sha256, tensors,
    trainable_parameters,
)

P_PAGE_HIDDEN, P_TEXT_HIDDEN = 0.25, 0.10


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


def evaluate(model: PrunedFusionClassifier, ids: torch.Tensor, link: torch.Tensor, labels: torch.Tensor) -> dict:
    """Val metrics, eval mode, no modality dropout. Smishing (1) is the positive class."""
    p = model.predict_proba(ids, link)[:, 1].numpy()
    y, preds = labels.numpy(), (p >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    return {"val_accuracy": accuracy_score(y, preds), "val_balanced_accuracy": balanced_accuracy_score(y, preds),
            "val_precision": precision, "val_recall": recall, "val_f1": f1,
            "val_log_loss": log_loss(y, p, labels=[0, 1])}


def fine_tune(model: PrunedFusionClassifier, train: tuple, val: tuple, seed: int, checkpoint: Path,
              epochs: int = EPOCHS, lr: float = LR, patience: int = PATIENCE,
              weight_decay: float = DENSE_WEIGHT_DECAY, p_page_hidden: float | None = None,
              p_text_hidden: float | None = None) -> dict:
    """Fine-tunes `model` in place; the best epoch is saved to `checkpoint` and loaded back. Returns its metrics.
    train = (token ids, full link numbers, blank-page link numbers, labels); val = (token ids, link numbers, labels).
    The modality dropout rates default to P_PAGE_HIDDEN / P_TEXT_HIDDEN (a text-only model passes 0 for the text).
    """
    p_page = P_PAGE_HIDDEN if p_page_hidden is None else p_page_hidden
    p_text = P_TEXT_HIDDEN if p_text_hidden is None else p_text_hidden
    ids, link_full, link_blank, y = train
    assert not model.encoder.embedding.weight.requires_grad, "the embedding must stay frozen"
    torch.manual_seed(seed)                                   # the dropout layers' draws
    gen = torch.Generator().manual_seed(seed)                 # modality dropout and batch order
    optimizer = torch.optim.Adam(model.parameter_groups(weight_decay), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y), label_smoothing=LABEL_SMOOTHING)
    best, best_state, best_epoch, bad, epoch = -1.0, None, -1, 0, -1
    for epoch in range(epochs):
        model.train()
        page_hidden = torch.rand(len(y), generator=gen) < p_page
        link = torch.where(page_hidden[:, None], link_blank, link_full)
        text_keep = (torch.rand(len(y), generator=gen) >= p_text).float()
        order = torch.randperm(len(y), generator=gen)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            criterion(model(ids[batch], link[batch], text_keep[batch]), y[batch]).backward()
            optimizer.step()
        model.eval()
        metrics = evaluate(model, *val)
        if metrics["val_balanced_accuracy"] > best:
            best, best_state, best_epoch, bad = metrics["val_balanced_accuracy"], copy.deepcopy(model.state_dict()), epoch, 0
            best_metrics = metrics
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, checkpoint)
    return {**best_metrics, "best_epoch": best_epoch, "epochs_run": epoch + 1}


def train_inputs(seed: int, fold: int, word_to_idx: Dict[str, int], seq_len: int, pilot: bool = False) -> tuple:
    """(train, val) inputs of fine_tune for one seed and fold."""
    parts, lane = load_fold(seed, fold, pilot=pilot), load_lane(seed, fold, pilot)
    ids, full, y = fusion_inputs(parts["train"], lane["train"][0], word_to_idx, seq_len)
    _, blank, _ = fusion_inputs(parts["train"], lane["train"][1], word_to_idx, seq_len)
    assert (blank[:, LANE.index("page_missing")] == 1).all(), "the blank-page numbers must have page_missing = 1"
    return (ids, full, blank, y), fusion_inputs(parts["val"], lane["val"][0], word_to_idx, seq_len)


def checkpoint_path(seed: int, fold: int, k: int, pilot: bool = False) -> Path:
    return model_dir(dataset_path(pilot)) / f"c_prune_seed{seed}_fold{fold}_k{k}.pt"


def run_train(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    out = out_dir(dataset_path(pilot))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            for k in PRUNE_KEEP:
                model, _, word_to_idx, sidecar = build_pruned_fusion(seed, fold, k, pilot)
                train, val = train_inputs(seed, fold, word_to_idx, sidecar["seq_len_l"], pilot)
                before = evaluate(model.eval(), *val)
                start = time.perf_counter()
                metrics = fine_tune(model, train, val, seed, checkpoint_path(seed, fold, k, pilot))
                seconds = time.perf_counter() - start
                rows.append({"seed": seed, "fold": fold, "k": k, "train_rows": len(train[3]),
                             "val_rows": len(val[2]), "val_f1_before": round(before["val_f1"], 4),
                             **{m: round(v, 4) if isinstance(v, float) else v for m, v in metrics.items()},
                             "train_seconds": round(seconds, 2)})
                print(f"seed {seed} fold {fold} k {k}: val F1 {metrics['val_f1']:.4f}, "
                      f"best epoch {metrics['best_epoch']} of {metrics['epochs_run']}, {seconds:.1f}s")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "train_summary.csv", index=False)
    return summary


def choose_k(summary: pd.DataFrame, metric: str = "val_f1") -> dict:
    """The one-standard-error rule: the smallest k whose mean val F1 is within one standard error of the best k's."""
    per_k = summary.groupby("k")[metric].agg(["mean", "std", "count"]).sort_index()
    per_k["se"] = per_k["std"] / np.sqrt(per_k["count"])
    best = max(per_k.index, key=lambda k: (per_k.loc[k, "mean"], -k))     # a tie goes to the smaller k
    threshold = per_k.loc[best, "mean"] - per_k.loc[best, "se"]
    chosen = min(k for k in per_k.index if per_k.loc[k, "mean"] >= threshold)
    return {"k": int(chosen), "best_k": int(best), "threshold": round(float(threshold), 6),
            "per_k": {str(k): {"mean": round(float(r["mean"]), 6), "se": round(float(r["se"]), 6),
                               "seed_folds": int(r["count"])} for k, r in per_k.iterrows()}}


def final_k_path(pilot: bool = False) -> Path:
    return out_dir(dataset_path(pilot)) / "final_k.json"


def final_k(pilot: bool = False) -> int:
    """The k chosen on validation (the choose step)."""
    return json.loads(final_k_path(pilot).read_text(encoding="utf-8"))["k"]


def run_choose(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    summary = pd.read_csv(out_dir(dataset_path(pilot)) / "train_summary.csv")
    expected = {(s, f, k) for s in seeds for f in range(n_folds) for k in PRUNE_KEEP}
    assert set(zip(summary["seed"], summary["fold"], summary["k"])) == expected, "train did not cover every model"
    decision = choose_k(summary)
    final_k_path(pilot).write_text(json.dumps({
        "chosen_on": date.today().isoformat(), "dataset": dataset_path(pilot).name,
        "rule": "one standard error: the smallest k whose mean val F1 is at least the best k's mean minus its "
                "standard error (std / sqrt(seed-folds)); val only, test never read", **decision}, indent=2),
        encoding="utf-8")
    print(f"final k = {decision['k']} (best mean val F1 at k = {decision['best_k']}, "
          f"threshold {decision['threshold']:.4f})")
    return pd.DataFrame([{"k": int(k), **v, "within_1se": v["mean"] >= decision["threshold"],
                          "chosen": int(k) == decision["k"]} for k, v in decision["per_k"].items()])


def sidecar_path(seed: int, fold: int, k: int, pilot: bool = False) -> Path:
    return checkpoint_path(seed, fold, k, pilot).with_suffix(".json")


def write_final_sidecar(seed: int, fold: int, k: int, pilot: bool = False) -> dict:
    """Step 4: what is needed to rebuild and trust one fine-tuned model, next to its checkpoint."""
    models = model_dir(dataset_path(pilot))
    checkpoint = checkpoint_path(seed, fold, k, pilot)
    phase_a_pt = models / f"phase_a_seed{seed}_fold{fold}.pt"
    phase_a = json.loads(phase_a_pt.with_suffix(".json").read_text(encoding="utf-8"))
    mask_file = gate_mask_path(seed, fold, pilot)
    link_rf = models / f"link_rf_seed{seed}_fold{fold}.joblib"
    keep = load_gate_mask(seed, fold, k, pilot)
    model, _, word_to_idx, _ = load_c_prune_weights(checkpoint, phase_a, keep)
    _, val = train_inputs(seed, fold, word_to_idx, phase_a["seq_len_l"], pilot)
    emb_dir = ROOT / phase_a["embeddings_dir"]
    sidecar = {
        "seed": seed, "fold": fold, "k": k, "final": k == final_k(pilot), "dataset": dataset_path(pilot).name,
        "keep_idx": keep.tolist(),
        "ranking_rule": json.loads(mask_file.read_text(encoding="utf-8"))["ranking_rule"],
        "architecture": {"num_filters_full": phase_a["num_filters"], "k": k, "kernel_size": phase_a["kernel_size"],
                         "embedding_dim": model.encoder.embedding.embedding_dim, "dense_units": DENSE_UNITS,
                         "seq_len_l": phase_a["seq_len_l"], "vocab_size": phase_a["vocab_size"],
                         "link_inputs": list(LANE)},
        "files": {"checkpoint": checkpoint.name, "gate_mask": mask_file.name, "link_rf": link_rf.name,
                  "phase_a": phase_a_pt.name, "embeddings_dir": phase_a["embeddings_dir"]},
        "sha256": {checkpoint.name: sha256(checkpoint), mask_file.name: sha256(mask_file),
                   link_rf.name: sha256(link_rf), phase_a_pt.name: sha256(phase_a_pt),
                   "vocab.json": sha256(emb_dir / "vocab.json"),
                   "embedding_matrix.npy": sha256(emb_dir / "embedding_matrix.npy")},
        "recipe": {"optimizer": "Adam", "lr": LR, "dense_weight_decay": DENSE_WEIGHT_DECAY, "max_epochs": EPOCHS,
                   "patience": PATIENCE, "monitor": "val balanced accuracy", "batch_size": BATCH_SIZE,
                   "label_smoothing": LABEL_SMOOTHING, "class_weights": True, "p_page_hidden": P_PAGE_HIDDEN,
                   "p_text_hidden": P_TEXT_HIDDEN, "embedding": "frozen"},
        "trainable_parameters": trainable_parameters(model),
        "val": evaluate(model, *val),
    }
    sidecar_path(seed, fold, k, pilot).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def load_c_prune_weights(checkpoint: Path, phase_a: dict, keep) -> tuple:
    """(model in eval mode, keep_idx, word_to_idx, phase_a) from a checkpoint and the Phase A settings."""
    embedding, word_to_idx = frozen_embedding(ROOT / phase_a["embeddings_dir"])
    encoder = TextEncoder(embedding, len(keep), phase_a["kernel_size"])
    model = PrunedFusionClassifier(encoder, keep, url_dim=len(LANE), dense_units=DENSE_UNITS)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))   # strict: every key must match
    assert model.keep_idx.tolist() == list(keep), f"{checkpoint.name} holds another keep_idx"
    return model.eval(), keep, word_to_idx, phase_a


def load_c_prune(seed: int, fold: int, k: int | None = None, pilot: bool = False) -> tuple:
    """(model, Link RF, word_to_idx, sidecar), rebuilt from the sidecar alone; k defaults to the chosen final k.
    Refuses the bundle when any file it names changed since it was saved."""
    k = final_k(pilot) if k is None else k
    sidecar = json.loads(sidecar_path(seed, fold, k, pilot).read_text(encoding="utf-8"))
    models, emb_dir = model_dir(dataset_path(pilot)), ROOT / sidecar["files"]["embeddings_dir"]
    for name, digest in sidecar["sha256"].items():
        path = emb_dir / name if name in ("vocab.json", "embedding_matrix.npy") else models / name
        if sha256(path) != digest:
            raise ValueError(f"{path} changed since the model was saved")
    assert sidecar["architecture"]["link_inputs"] == list(LANE), "the link lane's 40 inputs changed order"
    keep = load_gate_mask(seed, fold, k, pilot)
    assert keep.tolist() == sidecar["keep_idx"], "the gate mask and the sidecar disagree"
    phase_a = {key: sidecar["architecture"][key] for key in ("kernel_size", "seq_len_l")}
    phase_a["embeddings_dir"] = sidecar["files"]["embeddings_dir"]
    model, _, word_to_idx, _ = load_c_prune_weights(models / sidecar["files"]["checkpoint"], phase_a, keep)
    link_rf = joblib.load(models / sidecar["files"]["link_rf"])
    return model, link_rf, word_to_idx, sidecar


def predict_messages(model: PrunedFusionClassifier, link_rf: dict, word_to_idx: Dict[str, int], seq_len: int,
                     rows: pd.DataFrame) -> np.ndarray:
    """P(smishing) of raw rows (sms_text + the 37 link features + page_fetched): the Link RF makes the 40 numbers,
    then the model reads text and numbers. The whole path of a new message, as in the app."""
    assert link_rf["features"] == list(FEATURES_37), "the Link RF reads other features"
    link = torch.tensor(lane_numbers(link_rf["model"], rows)[LANE].to_numpy(), dtype=torch.float32)
    ids, _ = tensors(rows.assign(label="smishing"), word_to_idx, seq_len)   # the label is not used here
    return model.predict_proba(ids, link)[:, 1].numpy()


def run_save(pilot: bool, seeds: List[int] = SEEDS, n_folds: int = CROWD_N_FOLDS) -> pd.DataFrame:
    train_summary = pd.read_csv(out_dir(dataset_path(pilot)) / "train_summary.csv").set_index(["seed", "fold", "k"])
    rows = []
    for seed in seeds:
        for fold in range(n_folds):
            val_rows = load_fold(seed, fold, pilot=pilot)["val"]
            y = (val_rows["label"] == "smishing").to_numpy().astype(int)
            for k in PRUNE_KEEP:
                saved = write_final_sidecar(seed, fold, k, pilot)
                # The metrics of the saved weights equal the train step's (rounded there to 4 decimals).
                recorded = train_summary.loc[(seed, fold, k)]
                assert all(abs(saved["val"][m] - recorded[m]) < 6e-5 for m in saved["val"]), "not the trained model"
                # Rebuilt from the sidecar alone, from the RAW val rows through the saved Link RF.
                model, link_rf, word_to_idx, sidecar = load_c_prune(seed, fold, k, pilot)
                p = predict_messages(model, link_rf, word_to_idx, sidecar["architecture"]["seq_len_l"], val_rows)
                again = {"val_f1": f1_score(y, p >= 0.5, zero_division=0), "val_log_loss": log_loss(y, p, labels=[0, 1])}
                gap = max(abs(again[m] - saved["val"][m]) for m in again)
                assert gap < 1e-9, f"seed {seed} fold {fold} k {k}: the rebuilt model differs (gap {gap})"
                rows.append({"seed": seed, "fold": fold, "k": k, "final": saved["final"],
                             "val_f1": round(saved["val"]["val_f1"], 4), "rebuilt_gap": gap,
                             "sidecar": sidecar_path(seed, fold, k, pilot).name})
            print(f"seed {seed} fold {fold}: saved and rebuilt k = {', '.join(map(str, PRUNE_KEEP))}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir(dataset_path(pilot)) / "save_summary.csv", index=False)
    return summary


STEPS = {"build": run_build, "train": run_train, "choose": run_choose, "save": run_save}


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (pipeline test only)")
    ap.add_argument("step", choices=list(STEPS))
    a = ap.parse_args(argv)
    summary = STEPS[a.step](a.pilot)
    pd.set_option("display.width", 200)
    print(summary.drop(columns="keep_idx", errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
