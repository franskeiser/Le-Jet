"""The concatenation control of the E2 study (plan docs/film_architecture_plan.md, D12): the same CNN and the same Link RF numbers as
E2, but P, u and page_missing are attached to the 64 text numbers before Dense 64. No generator, no FiLM.

Everything else is identical to E2, so only how the link reaches the verdict differs: the same folds and inputs (film_data.part_inputs),
the same shared loop and recipe (fit_model.fit, with the same modality dropout: page hidden 0.25, text hidden 0.10), and the same warm
start (the embedding and the Conv copied from the retrained text-only baseline, cnn_baseline.py). Dense 64 and Output are new and seeded.

Per seed and fold, writes models/film/<dataset>/concat_seed<S>_fold<F>.pt with its .json sidecar (hashes of the vocabulary, the starting
matrix, the weights and the Link RF; the recipe; the validation metrics) and experiments/film/<dataset>/concat/{train_summary,train_log}.csv.
Step check rebuilds every model from its sidecar alone and repeats its validation metrics, which must equal the recorded ones exactly.
The test part is never read here.

Usage:
    python -m src.unified.concat_control train --pilot      # after: python -m src.unified.cnn_baseline train --pilot
    python -m src.unified.concat_control check --pilot
    python -m src.unified.concat_control train --warm-start none --preset tuned --resume --minutes 9

--preset tuned (adopted 2026-11-25, docs/validation_only_improvement_plan.md Phases E-F, `CF1_w345x64`): Conv widths 3, 4, 5 x 64,
patience 10, text hidden 0.20 (Adam 1e-3 and page hidden 0.25 as before). --resume / --minutes / --threads as in film.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from src.config import CROWD_N_FOLDS, DENSE_WEIGHT_DECAY, LABEL_SMOOTHING, SEEDS
from src.features.model import conv_blocks
from src.features.model.concat_cnn import ConcatTextCNN
from src.unified import cell_runs
from src.unified import film
from src.unified import fit_model as fm
from src.unified import text_cnn as tc
from src.unified.film_data import part_inputs
from src.unified.film_inputs import LINK_WIDTH

KIND = "concat"


def kind_of(warm_start: bool) -> str:
    return KIND if warm_start else KIND + "_scratch"


def expected_trainable(link_dim: int, embedding_dim: int = 300, num_filters: int = 64, kernel_size: int = 3, dense_units: int = 64,
                       widths=None, per_width: int | None = None) -> int:
    """Trainable parameters without the embedding: conv + dense reading (text numbers + link_dim link numbers) + output. With
    `widths`, the Conv is conv_blocks.MultiWidthConv and there are len(widths) x per_width text numbers."""
    if widths:
        num_filters = len(widths) * per_width
        conv = conv_blocks.conv_parameters(embedding_dim, widths, per_width)
    else:
        conv = embedding_dim * kernel_size * num_filters + num_filters
    return conv + (num_filters + link_dim) * dense_units + dense_units + dense_units * 2 + 2


# The pipeline gives the control P, u, page_missing AND has_url (film_inputs.LINK_WIDTH = 4), so it can tell "no link" as well as E2 can:
# 62,210 trainable parameters without the embedding (62,146 with the three numbers alone).
EXPECTED_TRAINABLE = expected_trainable(LINK_WIDTH)
KIND_NAME, OUT_NAME = "concat", KIND + "_scratch"
# ADOPTED 2026-11-25 (Franz): the tuned Concatenation of the validation-only tuning (docs/validation_only_improvement_plan.md, Phases
# E-F, `CF1_w345x64`: mean validation balanced accuracy .9563; the earlier recipe .9464), trained with --preset tuned.
TUNED = {"architecture": {"widths": [3, 4, 5], "per_width": 64}, "recipe": {"patience": 10, "p_text_hidden": 0.20}}


def build_control(seed: int, fold: int, pilot: bool = False, architecture: dict | None = None,
                  **options) -> tuple[ConcatTextCNN, Dict[str, int], int]:
    """(model, word_to_idx, SEQ_LEN_L) of one seed and fold, untrained, with a fresh embedding layer of its own. architecture:
    {"widths", "per_width"} for several Conv widths (TUNED), None for the earlier one-width model."""
    arch = dict(architecture or {})
    if arch.get("widths"):
        options["num_filters"] = len(arch["widths"]) * arch["per_width"]
    embedding, word_to_idx, seq_len = tc.fold_text_setup(tc.dataset_path(pilot), seed, fold)
    tc.seed_everything(seed)
    options.setdefault("link_dim", LINK_WIDTH)
    model = ConcatTextCNN(embedding, **options)
    if arch.get("widths"):
        model.use_widths(arch["widths"], arch["per_width"])
    return model, word_to_idx, seq_len


def trainable_counts(model: ConcatTextCNN) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    own = model.embedding.weight.numel() if model.embedding.weight.requires_grad else 0
    return total - own, total


def train_cell(seed: int, fold: int, pilot: bool, warm_start: bool = True, epochs: int = fm.EPOCHS, architecture: dict | None = None,
               recipe: dict | None = None) -> tuple[dict, pd.DataFrame]:
    dataset = tc.dataset_path(pilot)
    kind = kind_of(warm_start)
    recipe = {**film.DEFAULT_RECIPE, **(recipe or {})}
    model, word_to_idx, seq_len = build_control(seed, fold, pilot, architecture)
    if warm_start and model.widths:
        raise ValueError("the warm start copies a one-width text CNN; it cannot be combined with Conv widths")
    source = film.warm_start_from_baseline(model, dataset, seed, fold, new_layers={"dense", "output"}) if warm_start else None
    train, val = (part_inputs(seed, fold, p, word_to_idx, seq_len, pilot) for p in ("train", "val"))
    checkpoint = fm.checkpoint_path(dataset, kind, seed, fold)
    result = fm.fit(model, (train.token_ids, train.link_full, train.link_blank, train.labels),
                    (val.token_ids, val.link_full, val.labels), seed, checkpoint, epochs=epochs, **recipe)
    without, with_embedding = trainable_counts(model)
    assert without == expected_trainable(model.link_dim, widths=model.widths, per_width=model.per_width)
    emb_dir = tc.embeddings_dir(dataset, seed, fold)
    fm.write_sidecar(checkpoint, {
        "kind": kind, "seed": seed, "fold": fold, "num_filters": model.num_filters, "kernel_size": model.kernel_size,
        "link_dim": model.link_dim, "dense_units": model.dense.out_features, "freeze_embedding": model.freeze_embedding,
        "architecture": {"widths": list(model.widths) if model.widths else None, "per_width": model.per_width},
        "seq_len_l": seq_len, "vocab_size": len(word_to_idx), "embeddings_dir": str(emb_dir.relative_to(tc.ROOT)),
        "sha256": fm.hashes(fm.sidecar_files(dataset, seed, fold, checkpoint, {"link_rf": film.link_rf_path(dataset, seed, fold)})),
        "warm_start": source,
        "recipe": {"optimizer": "adam", "lr": recipe["lr"], "epochs": epochs, "patience": recipe["patience"], "batch": fm.BATCH_SIZE,
                   "label_smoothing": LABEL_SMOOTHING, "class_weights": True, "dense_weight_decay": DENSE_WEIGHT_DECAY,
                   "p_page_hidden": recipe["p_page_hidden"], "p_text_hidden": recipe["p_text_hidden"]},
        "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
        "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "train_seconds": result.seconds,
        "val": {k: result.metrics[k] for k in fm.VAL_METRICS}})
    row = {"seed": seed, "fold": fold, "n_train": len(train.labels), "n_val": len(val.labels), "vocab": len(word_to_idx),
           "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
           "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "train_seconds": round(result.seconds, 3),
           **{k: result.metrics[k] for k in fm.VAL_METRICS}}
    return row, result.log.assign(seed=seed, fold=fold)


def run_train(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None, warm_start: bool = True,
              epochs: int = fm.EPOCHS, architecture: dict | None = None, recipe: dict | None = None, resume: bool = False,
              minutes: float | None = None) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    if resume:
        if warm_start:
            raise ValueError("--resume is for the from-scratch run (--warm-start none)")
        return run_train_cells(pilot, seeds, folds, warm_start, epochs, architecture, recipe, minutes)
    rows, logs = [], []
    for seed in seeds:
        for fold in folds:
            row, log = train_cell(seed, fold, pilot, warm_start, epochs, architecture, recipe)
            rows.append(row)
            logs.append(log)
            print(f"seed {seed} fold {fold}: val F1 {row['val_f1']:.3f}, balanced accuracy {row['val_balanced_accuracy']:.3f} | "
                  f"best epoch {row['best_epoch']} of {row['epochs_run']} | {row['train_seconds']:.1f} s", flush=True)
    out = fm.out_dir(dataset, kind_of(warm_start))
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "train_summary.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(out / "train_log.csv", index=False)
    return summary


def run_train_cells(pilot: bool, seeds: List[int], folds: List[int], warm_start: bool, epochs: int, architecture: dict | None,
                    recipe: dict | None, minutes: float | None) -> pd.DataFrame:
    """run_train --resume: cell by cell (cell_runs.py), skipping cells already trained with this configuration and stopping before a cell
    that would end after `minutes`; writes train_summary / train_log from every finished cell."""
    dataset = tc.dataset_path(pilot)
    out = fm.out_dir(dataset, OUT_NAME)
    signature = {"model": KIND_NAME, "warm_start": warm_start, "epochs": epochs, "architecture": architecture or {},
                 "recipe": {**film.DEFAULT_RECIPE, **(recipe or {})}, "dataset": dataset.stem}
    budget = cell_runs.Budget(minutes)
    stop = False
    for seed in seeds:
        for fold in folds:
            if cell_runs.cell_done(out, seed, fold, signature):
                continue
            if not budget.fits():
                print(f"TIME BUDGET: stopped before seed {seed} fold {fold}", flush=True)
                stop = True
                break
            row, log = train_cell(seed, fold, pilot, warm_start, epochs, architecture, recipe)
            budget.add(row["train_seconds"])
            cell_runs.save_cell(out, seed, fold, {"train_summary": pd.DataFrame([row]), "train_log": log}, signature)
            print(f"seed {seed} fold {fold}: val balanced accuracy {row['val_balanced_accuracy']:.4f} | best epoch {row['best_epoch']} "
                  f"of {row['epochs_run']} | {row['train_seconds']:.0f} s", flush=True)
        if stop:
            break
    tables, finished = cell_runs.assemble(out, seeds, folds, ["train_summary", "train_log"], signature)
    expected = len(seeds) * len(folds)
    print(f"{finished} of {expected} cells finished with this configuration" + ("" if finished == expected else " (run again)"),
          flush=True)
    return tables["train_summary"]


def load_control(seed: int, fold: int, pilot: bool = False, warm_start: bool = True) -> tuple[ConcatTextCNN, Dict[str, int], int, dict]:
    """(model in eval mode, word_to_idx, SEQ_LEN_L, sidecar), rebuilt from the sidecar alone; refuses a changed file."""
    dataset = tc.dataset_path(pilot)
    checkpoint = fm.checkpoint_path(dataset, kind_of(warm_start), seed, fold)
    sidecar = fm.read_sidecar(checkpoint)
    fm.check_hashes(sidecar["sha256"], fm.sidecar_files(dataset, seed, fold, checkpoint,
                                                        {"link_rf": film.link_rf_path(dataset, seed, fold)}))
    embedding, word_to_idx = tc.frozen_embedding(tc.ROOT / sidecar["embeddings_dir"])
    model = ConcatTextCNN(embedding, sidecar["num_filters"], sidecar["kernel_size"], sidecar["link_dim"], sidecar["dense_units"],
                          freeze_embedding=sidecar["freeze_embedding"])
    arch = sidecar.get("architecture") or {}
    if arch.get("widths"):
        model.use_widths(arch["widths"], arch["per_width"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval(), word_to_idx, sidecar["seq_len_l"], sidecar


def run_check(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None, warm_start: bool = True) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for seed in seeds:
        for fold in folds:
            model, word_to_idx, seq_len, sidecar = load_control(seed, fold, pilot, warm_start)
            val = part_inputs(seed, fold, "val", word_to_idx, seq_len, pilot)
            again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
            gap = max(abs(again[k] - sidecar["val"][k]) for k in fm.VAL_METRICS)
            rows.append({"seed": seed, "fold": fold, "val_f1": again["val_f1"], "max_gap": gap, "ok": gap == 0.0})
    summary = pd.DataFrame(rows)
    out = fm.out_dir(dataset, kind_of(warm_start))
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "check_summary.csv", index=False)
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["train", "check"])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a pipeline test only)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--warm-start", choices=["baseline", "none"], default="none",
                    help="train: from scratch (default, like E2, decision D7) or from the retrained baseline (ablation); check: which set")
    ap.add_argument("--preset", choices=["none", "tuned"], default="none",
                    help="train: 'tuned' = the configuration adopted 2026-11-25 (TUNED); 'none' = the earlier model and recipe")
    ap.add_argument("--resume", action="store_true", help="train cell by cell, skipping cells already trained with this configuration")
    ap.add_argument("--minutes", type=float, default=None, help="with --resume: stop before a cell that would end after this budget")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (the validated tuning runs used 5)")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    preset = TUNED if args.preset == "tuned" else {}
    warm = args.warm_start == "baseline"
    if args.step == "train":
        summary = run_train(args.pilot, args.seeds, args.folds, warm, architecture=preset.get("architecture"),
                            recipe=preset.get("recipe"), resume=args.resume, minutes=args.minutes)
        if args.resume:
            return
        print(f"\n{len(summary)} concatenation controls trained; mean val F1 {summary['val_f1'].mean():.3f} (std {summary['val_f1'].std():.3f})")
        return
    summary = run_check(args.pilot, args.seeds, args.folds, warm)
    print(summary.to_string(index=False))
    print(f"\n{int(summary['ok'].sum())} of {len(summary)} rebuilt from their sidecars with a gap of exactly 0")
    sys.exit(0 if summary["ok"].all() else 1)


if __name__ == "__main__":
    main()
