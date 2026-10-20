"""E2 (FiLM text CNN): build, warm-start, train, save, check and predict (plan docs/film_architecture_plan.md, Phase 3).

Step build (3.1): per seed and fold, the E2 model of section A, before any training:
    text      the fold's own vocabulary and fastText matrix (text_cnn.fold_text_setup), the embedding fine-tuned (D6);
    model     src/features/model/film_cnn.FiLMTextCNN, 64 filters, generator at zero, bounded gamma (D2);
    link      P, u, page_missing per message, from the link lane (film_inputs.link_inputs), full page and page hidden.
Checked for every seed and fold, on the fold's validation messages:
  * trainable parameters excluding the embedding = section A5: 70,658 (conv 57,664 + generator 8,704 + dense 4,160 + output
    130); including the embedding, vocabulary size x 300 more;
  * the same seed builds the same weights twice;
  * the generator starts at zero: gamma = 1 and beta = 0 for every validation message;
  * the forward pass works on the real inputs: (messages, 2) logits, probabilities summing to 1;
  * the optimizer groups cover every trainable parameter once;
  * predict_messages() (raw rows -> Link RF -> model, the path the app will take) gives the same P and u as the cached link
    lane and the same logits as the model fed the lane's numbers.

Step train (3.2, 3.3, 3.4): per seed and fold
  * warm start (D7b): the embedding and the Conv are copied from the fold's retrained text-only baseline (cnn_baseline.py, step 3.6);
    the generator starts at zero, Dense 64 and Output are new and seeded. At epoch 0 E2 must give the baseline's 64 pooled text
    numbers exactly (checked on the validation messages; the run stops if not);
  * training: fit_model.fit, the recipe shared with the baseline (Adam 1e-3, up to 30 epochs, patience 3 on validation balanced
    accuracy, label smoothing 0.1, class weights, weight decay 1e-4 on Dense 64 and the generator, page hidden 0.25 and text
    hidden 0.10 while training); train rows carry out-of-fold link numbers, validation rows the final forest's;
  * how far gamma and beta moved is recorded every epoch (train_log.csv) and per filter at the best epoch (filter_stats.csv).
  Writes models/film/<dataset>/film_seed<S>_fold<F>.pt and its .json sidecar (the best epoch's weights; what it was built from,
  with sha256 of the vocabulary, the starting embedding matrix, the weights and the Link RF; the warm-start source; the recipe;
  the validation metrics), and experiments/film/<dataset>/train/{train_summary,train_log,filter_stats}.csv.
  --warm-start none trains E2 from scratch instead (D7a, the check run), as film_scratch_*.

Step check (3.5): every model is rebuilt from its sidecar alone (hash-checked) and scored on the validation messages; the metrics
must equal the ones recorded at training time exactly, and the app path (predict_messages) must agree with the model.

The test part is never read here.

Usage:
    python -m src.unified.film build --pilot
    python -m src.unified.film train --pilot                     # after: python -m src.unified.cnn_baseline train --pilot
    python -m src.unified.film check --pilot
    python -m src.unified.film train --pilot --seeds 42 --folds 0 --warm-start none
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import CROWD_N_FOLDS, DENSE_UNITS, DENSE_WEIGHT_DECAY, KERNEL_SIZE, LABEL_SMOOTHING, NUM_FILTERS, SEEDS
from src.data.split_crowd import load_fold
from src.features.model.film_cnn import LINK_DIM, FiLMTextCNN
from src.features.model.fusion_data import encode_batch_tolerant
from src.features.model.rf_extractor import p_and_u
from src.unified import cnn_baseline as cb
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc
from src.unified.film_data import PartData, part_inputs  # noqa: F401  (re-exported)
from src.unified.prepare_web_37 import FEATURES_37

PARTS = ("train", "val", "test")
EXPECTED_TRAINABLE = 70658            # section A5, embedding not included


def expected_trainable(embedding_dim: int = 300, num_filters: int = NUM_FILTERS, kernel_size: int = KERNEL_SIZE,
                       link_dim: int = LINK_DIM, dense_units: int = DENSE_UNITS, use_beta: bool = True) -> int:
    """Trainable parameters of E2 without the embedding: conv + generator + dense + output (section A5)."""
    outputs = (2 if use_beta else 1) * num_filters
    return (embedding_dim * kernel_size * num_filters + num_filters          # conv
            + (link_dim + num_filters) * outputs + outputs                   # generator
            + num_filters * dense_units + dense_units                        # dense 64
            + dense_units * 2 + 2)                                           # output


def trainable_counts(model: FiLMTextCNN) -> tuple[int, int]:
    """(trainable parameters without the embedding, trainable parameters with it)."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    own = model.embedding.weight.numel() if model.embedding.weight.requires_grad else 0
    return total - own, total


def build_film(seed: int, fold: int, pilot: bool = False, **options) -> tuple[FiLMTextCNN, Dict[str, int], int]:
    """(model, word_to_idx, SEQ_LEN_L) of one seed and fold, untrained. `options` go to FiLMTextCNN (defaults: the plan's)."""
    dataset = tc.dataset_path(pilot)
    embedding, word_to_idx, seq_len = tc.fold_text_setup(dataset, seed, fold)      # a fresh layer for this model only
    tc.seed_everything(seed)
    return FiLMTextCNN(embedding, **options), word_to_idx, seq_len


def link_rf_path(dataset: Path, seed: int, fold: int) -> Path:
    return ll.model_dir(dataset) / f"link_rf_seed{seed}_fold{fold}.joblib"


def load_link_rf(dataset: Path, seed: int, fold: int):
    bundle = joblib.load(link_rf_path(dataset, seed, fold))
    assert bundle["features"] == list(FEATURES_37)
    return bundle["model"]


def predict_messages(model: FiLMTextCNN, link_rf, word_to_idx: Dict[str, int], seq_len: int, rows: pd.DataFrame,
                     hide_page: bool = False) -> pd.DataFrame:
    """Score raw messages like the app: `rows` has sms_text, the 37 link/page features and page_fetched.

    Returns a frame with the index of `rows` and P, u, page_missing (what the generator read), p_smishing and pred (1 = smishing).
    """
    scored = ll.blank_page(rows) if hide_page else rows
    p, u = p_and_u(link_rf, scored[FEATURES_37])
    page_missing = 1 - scored["page_fetched"].astype(int).to_numpy()              # blank_page sets page_fetched to 0
    link = torch.tensor(np.column_stack([p, u, page_missing]), dtype=torch.float32)
    token_ids = encode_batch_tolerant(rows["sms_text"].astype(str).tolist(), word_to_idx, seq_len)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        p_smishing = torch.softmax(model(token_ids, link), dim=1)[:, 1].numpy()
    model.train(was_training)
    return pd.DataFrame({"P": p, "u": u, "page_missing": page_missing, "p_smishing": p_smishing,
                         "pred": (p_smishing >= 0.5).astype(int)}, index=rows.index)


# ------------------------------------------------------------------ step build
def check_cell(seed: int, fold: int, pilot: bool) -> dict:
    """Build the model of one seed and fold and run every check of step build. Returns the summary row; `ok` is False on any miss."""
    dataset = tc.dataset_path(pilot)
    model, word_to_idx, seq_len = build_film(seed, fold, pilot)
    again, _, _ = build_film(seed, fold, pilot)
    without, with_embedding = trainable_counts(model)
    vocab = model.embedding.num_embeddings
    checks = {
        "params": without == EXPECTED_TRAINABLE == expected_trainable(model.embedding.embedding_dim),
        "params_with_embedding": with_embedding == EXPECTED_TRAINABLE + vocab * model.embedding.embedding_dim,
        "same_seed_same_weights": all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), again.state_dict().values())),
        "groups": ({id(p) for g in model.parameter_groups(1e-4) for p in g["params"]}
                   == {id(p) for p in model.parameters() if p.requires_grad}),
    }

    val = part_inputs(seed, fold, "val", word_to_idx, seq_len, pilot)
    model.eval()
    with torch.no_grad():
        gamma, beta = model.film_parameters(val.link_full, model.conv_features(val.token_ids)[1])
        logits = model(val.token_ids, val.link_full)
    probabilities = torch.softmax(logits, dim=1)
    checks["generator_at_zero"] = bool((gamma == 1).all() and (beta == 0).all())
    checks["forward"] = (tuple(logits.shape) == (len(val.labels), 2) and bool(torch.isfinite(logits).all())
                         and bool(torch.allclose(probabilities.sum(dim=1), torch.ones(len(val.labels)), atol=1e-6)))

    rows = load_fold(seed, fold, pilot=pilot)["val"]
    scored = predict_messages(model, load_link_rf(dataset, seed, fold), word_to_idx, seq_len, rows)
    checks["app_path_link_numbers"] = (list(rows["message_id"]) == list(val.message_id)
                                       and bool(np.allclose(scored["P"], val.link_full[:, 0].numpy(), atol=1e-6))
                                       and bool(np.allclose(scored["u"], val.link_full[:, 1].numpy(), atol=1e-6)))
    checks["app_path_logits"] = bool(np.allclose(scored["p_smishing"], probabilities[:, 1].numpy(), atol=1e-6))
    return {"seed": seed, "fold": fold, "n_val": len(val.labels), "vocab": vocab, "seq_len": seq_len,
            "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
            **checks, "ok": all(checks.values())}


def run_build(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for seed in seeds:
        for fold in folds:
            rows.append(check_cell(seed, fold, pilot))
            print(f"seed {seed} fold {fold}: {'ok' if rows[-1]['ok'] else 'FAILED'} | {rows[-1]['trainable_without_embedding']:,} "
                  f"trainable without the embedding, {rows[-1]['trainable_with_embedding']:,} with it (vocabulary {rows[-1]['vocab']})",
                  flush=True)
    summary = pd.DataFrame(rows)
    out = fm.out_dir(dataset, "build")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "build_summary.csv", index=False)
    return summary


# ------------------------------------------------------------------ step train
def kind_of(warm_start: bool) -> str:
    return "film" if warm_start else "film_scratch"


def warm_start_from_baseline(model: torch.nn.Module, dataset: Path, seed: int, fold: int,
                             new_layers: set | None = None) -> dict:
    """Step 3.2: copy the embedding and the Conv from the fold's trained baseline into `model`. Returns the source (name, sha256).

    The generator keeps its zero start; Dense 64 and Output keep their seeded start. The baseline's own sidecar is hash-checked
    first, so a changed baseline file is refused. `new_layers` names the layers the model has that the baseline does not (default:
    E2's generator, dense and output; the concatenation control passes dense and output).
    """
    new_layers = {"generator", "dense", "output"} if new_layers is None else new_layers
    checkpoint = fm.checkpoint_path(dataset, cb.KIND, seed, fold)
    if not checkpoint.with_suffix(".json").exists():
        raise FileNotFoundError(f"{checkpoint.name} not found: train the baseline first (python -m src.unified.cnn_baseline train)")
    sidecar = fm.read_sidecar(checkpoint)
    fm.check_hashes(sidecar["sha256"], fm.sidecar_files(dataset, seed, fold, checkpoint))
    state = torch.load(checkpoint, map_location="cpu")
    result = model.load_state_dict({k: v for k, v in state.items() if k.startswith(("embedding.", "conv."))}, strict=False)
    if result.unexpected_keys or {k.split(".")[0] for k in result.missing_keys} != new_layers:
        raise ValueError(f"the baseline does not fit this model: unexpected {result.unexpected_keys}, missing {result.missing_keys}")
    return {"checkpoint": checkpoint.name, "sha256": sidecar["sha256"][checkpoint.name]}


def start_matches_baseline(model: FiLMTextCNN, seed: int, fold: int, pilot: bool, word_to_idx: Dict[str, int], seq_len: int) -> bool:
    """Step 3.2's check: with the generator at zero, E2's 64 pooled text numbers equal the baseline's, exactly, on the validation rows."""
    baseline, _, _, _ = cb.load_baseline(seed, fold, pilot)
    val = part_inputs(seed, fold, "val", word_to_idx, seq_len, pilot)
    model.eval()
    with torch.no_grad():
        return bool(torch.equal(model.encode(val.token_ids, val.link_full), baseline.encode(val.token_ids)))


def model_options(model: FiLMTextCNN) -> dict:
    return {"gamma_mode": model.gamma_mode, "mask_padding": model.mask_padding, "average_after_relu": model.average_after_relu,
            "use_beta": model.use_beta, "freeze_embedding": model.freeze_embedding}


def train_cell(seed: int, fold: int, pilot: bool, warm_start: bool = True, epochs: int = fm.EPOCHS, **options) -> tuple:
    """Train, save and describe one seed and fold. Returns (summary row, per-epoch log, per-filter table)."""
    dataset = tc.dataset_path(pilot)
    kind = kind_of(warm_start)
    model, word_to_idx, seq_len = build_film(seed, fold, pilot, **options)
    source = None
    if warm_start:
        source = warm_start_from_baseline(model, dataset, seed, fold)
        if not start_matches_baseline(model, seed, fold, pilot, word_to_idx, seq_len):
            raise AssertionError(f"seed {seed} fold {fold}: at epoch 0 E2's text numbers differ from the baseline's")
    train, val = (part_inputs(seed, fold, p, word_to_idx, seq_len, pilot) for p in ("train", "val"))
    checkpoint = fm.checkpoint_path(dataset, kind, seed, fold)
    result = fm.fit(model, (train.token_ids, train.link_full, train.link_blank, train.labels),
                    (val.token_ids, val.link_full, val.labels), seed, checkpoint, epochs=epochs)
    without, with_embedding = trainable_counts(model)
    moved = result.log.iloc[result.best_epoch]
    film = {k: float(moved[k]) for k in ("gamma_dev_mean", "gamma_dev_max", "beta_abs_mean", "beta_abs_max", "lift_abs_mean",
                                         "no_phrase_share")} | {"filters_moved": int(moved["filters_moved"])}
    emb_dir = tc.embeddings_dir(dataset, seed, fold)
    fm.write_sidecar(checkpoint, {
        "kind": kind, "seed": seed, "fold": fold, "options": model_options(model), "num_filters": model.num_filters,
        "kernel_size": model.kernel_size, "link_dim": model.link_dim, "dense_units": model.dense.out_features,
        "seq_len_l": seq_len, "vocab_size": len(word_to_idx), "embeddings_dir": str(emb_dir.relative_to(tc.ROOT)),
        "sha256": fm.hashes(fm.sidecar_files(dataset, seed, fold, checkpoint, {"link_rf": link_rf_path(dataset, seed, fold)})),
        "warm_start": source,
        "recipe": {"optimizer": "adam", "lr": fm.LR, "epochs": epochs, "patience": fm.PATIENCE, "batch": fm.BATCH_SIZE,
                   "label_smoothing": LABEL_SMOOTHING, "class_weights": True, "dense_weight_decay": DENSE_WEIGHT_DECAY,
                   "generator_weight_decay": DENSE_WEIGHT_DECAY, "p_page_hidden": fm.P_PAGE_HIDDEN, "p_text_hidden": fm.P_TEXT_HIDDEN},
        "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
        "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "train_seconds": result.seconds,
        "val": {k: result.metrics[k] for k in fm.VAL_METRICS}, "film_at_best_epoch": film})
    row = {"seed": seed, "fold": fold, "kind": kind, "n_train": len(train.labels), "n_val": len(val.labels),
           "vocab": len(word_to_idx), "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
           "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "train_seconds": round(result.seconds, 3),
           **{k: result.metrics[k] for k in fm.VAL_METRICS}, **film}
    return row, result.log.assign(seed=seed, fold=fold), result.filters.assign(seed=seed, fold=fold)


def run_train(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None, warm_start: bool = True,
              epochs: int = fm.EPOCHS) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows, logs, filters = [], [], []
    for seed in seeds:
        for fold in folds:
            row, log, table = train_cell(seed, fold, pilot, warm_start, epochs)
            rows.append(row)
            logs.append(log)
            filters.append(table)
            print(f"seed {seed} fold {fold}: val F1 {row['val_f1']:.3f}, balanced accuracy {row['val_balanced_accuracy']:.3f} | "
                  f"best epoch {row['best_epoch']} of {row['epochs_run']} | {row['train_seconds']:.1f} s | "
                  f"mean |gamma-1| {row['gamma_dev_mean']:.3f}, mean |beta| {row['beta_abs_mean']:.3f}, "
                  f"{row['filters_moved']} of {fm_filters(table)} filters moved", flush=True)
    out = fm.out_dir(dataset, "train" if warm_start else "train_scratch")
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "train_summary.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(out / "train_log.csv", index=False)
    pd.concat(filters, ignore_index=True).to_csv(out / "filter_stats.csv", index=False)
    return summary


def fm_filters(table: pd.DataFrame) -> int:
    return len(table)


# ------------------------------------------------------------------ step check
def load_film(seed: int, fold: int, pilot: bool = False, warm_start: bool = True) -> tuple[FiLMTextCNN, Dict[str, int], int, dict]:
    """(model in eval mode, word_to_idx, SEQ_LEN_L, sidecar), rebuilt from the sidecar alone; refuses a changed file.

    The files checked are the ones the model is made of and served from: the vocabulary, the starting embedding matrix, the
    weights and the Link RF. The warm-start source is recorded in the sidecar as provenance only (retraining the baseline later
    does not invalidate E2).
    """
    dataset = tc.dataset_path(pilot)
    checkpoint = fm.checkpoint_path(dataset, kind_of(warm_start), seed, fold)
    sidecar = fm.read_sidecar(checkpoint)
    fm.check_hashes(sidecar["sha256"], fm.sidecar_files(dataset, seed, fold, checkpoint, {"link_rf": link_rf_path(dataset, seed, fold)}))
    embedding, word_to_idx = tc.frozen_embedding(tc.ROOT / sidecar["embeddings_dir"])
    model = FiLMTextCNN(embedding, sidecar["num_filters"], sidecar["kernel_size"], sidecar["link_dim"], sidecar["dense_units"],
                        **sidecar["options"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval(), word_to_idx, sidecar["seq_len_l"], sidecar


def run_check(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None, warm_start: bool = True) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for seed in seeds:
        for fold in folds:
            model, word_to_idx, seq_len, sidecar = load_film(seed, fold, pilot, warm_start)
            val = part_inputs(seed, fold, "val", word_to_idx, seq_len, pilot)
            again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
            gap = max(abs(again[k] - sidecar["val"][k]) for k in fm.VAL_METRICS)
            frame = load_fold(seed, fold, pilot=pilot)["val"]
            scored = predict_messages(model, load_link_rf(dataset, seed, fold), word_to_idx, seq_len, frame)
            with torch.no_grad():
                direct = torch.softmax(model(val.token_ids, val.link_full), dim=1)[:, 1].numpy()
            app_ok = bool(np.allclose(scored["p_smishing"], direct, atol=1e-6))
            rows.append({"seed": seed, "fold": fold, "val_f1": again["val_f1"], "max_gap": gap, "app_path_ok": app_ok,
                         "ok": gap == 0.0 and app_ok})
    summary = pd.DataFrame(rows)
    out = fm.out_dir(dataset, "check" if warm_start else "check_scratch")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "check_summary.csv", index=False)
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["build", "train", "check"])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a pipeline test only)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=None, help=f"folds to run (default 0-{CROWD_N_FOLDS - 1})")
    ap.add_argument("--warm-start", choices=["baseline", "none"], default="baseline",
                    help="train: start from the retrained baseline (default) or from scratch (the D7a check run); check: which set")
    args = ap.parse_args(argv)
    warm = args.warm_start == "baseline"
    if args.step == "build":
        summary = run_build(args.pilot, args.seeds, args.folds)
        failed = summary[~summary["ok"]]
        print(f"\n{len(summary) - len(failed)} of {len(summary)} seed x fold models built and checked")
        if len(failed):
            checks = [c for c in summary.columns if summary[c].dtype == bool and c != "ok"]
            for _, r in failed.iterrows():
                print(f"  seed {r['seed']} fold {r['fold']}: failed {[c for c in checks if not r[c]]}")
            sys.exit(1)
    elif args.step == "train":
        summary = run_train(args.pilot, args.seeds, args.folds, warm)
        print(f"\n{len(summary)} E2 models trained; mean val F1 {summary['val_f1'].mean():.3f} (std {summary['val_f1'].std():.3f})")
    else:
        summary = run_check(args.pilot, args.seeds, args.folds, warm)
        print(summary.to_string(index=False))
        print(f"\n{int(summary['ok'].sum())} of {len(summary)} rebuilt from their sidecars with a gap of exactly 0")
        sys.exit(0 if summary["ok"].all() else 1)


if __name__ == "__main__":
    main()
