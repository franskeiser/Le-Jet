"""E2 ablations that change what the generator reads (plan docs/film_architecture_plan.md, steps 5.3 and 5.6).

Each variant is E2 trained FROM SCRATCH (D7a) with the shared recipe (fit_model.fit), on the same folds, seeds and messages as the main E2
(`film_scratch`), changing only the generator's input:

    no_average     V3, step 5.3   the generator reads the 3 link numbers only, not the Conv average of the text. The same link
                                  then gives every message the same gamma and beta. 62,466 trainable parameters (-8,192).
    contributions  V2, step 5.6   the generator reads the Link RF's 37 per-feature contributions (c_ columns of the link lane)
                                  instead of P, u and page_missing, plus the Conv average. 75,010 trainable parameters (+4,352).
    no_u           V5, step 5.6   P and page_missing only: the trees' disagreement u is dropped. 70,530 trainable parameters (-128).

Each model still gets has_url as a last column and bypasses its generator for a message with no link (D11), so the variants differ
from the main E2 in the generator's input and nothing else. Modality dropout is the same: a hidden page swaps in the blank-page version
of the same numbers (for `contributions` the blank-page contributions of the lane).

Per seed and fold, writes models/film/<dataset>/film_scratch_<variant>_seed<S>_fold<F>.pt with a JSON sidecar (hashes of the
vocabulary, starting matrix, weights and Link RF; the recipe; the validation metrics), and experiments/film/<dataset>/ablations/
{train_summary,train_log}.csv. `check` rebuilds each model from its sidecar alone and repeats its validation metrics (gap 0).
`compare` scores the variants, the main E2, the concatenation control and the text-only baseline on the test part, McNemar per seed
(E2 against each other model, Holm-corrected over the list). The test folds of the real data are read once, with --read-test; the pilot
may be read as a rehearsal, and each read is logged.

Usage:
    python -m src.unified.film_ablations train --pilot                    # all three variants, 15 cells each
    python -m src.unified.film_ablations train --pilot --variants no_average --seeds 42 --folds 0
    python -m src.unified.film_ablations check --pilot
    python -m src.unified.film_ablations compare --pilot
"""
from __future__ import annotations

import argparse
import datetime
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.config import CROWD_N_FOLDS, DENSE_WEIGHT_DECAY, LABEL_SMOOTHING, SEEDS
from src.features.model.film_cnn import FiLMTextCNN
from src.unified import film
from src.unified import film_compare as fc
from src.unified import fit_model as fm
from src.unified import link_lane as ll
from src.unified import text_cnn as tc
from src.unified.film_data import PartData, part_inputs

N_CONTRIBUTIONS = len(ll.CONTRIBUTIONS)       # 37

# name -> what it is, the FiLMTextCNN options that differ from the main E2, and which link numbers its tensors hold
VARIANTS: Dict[str, dict] = {
    "no_average": {"label": "E2 without the Conv average (V3)", "options": {"use_average": False}, "link": "pu"},
    "contributions": {"label": "E2 reading the 37 contributions (V2)", "options": {"link_dim": N_CONTRIBUTIONS}, "link": "contributions"},
    "no_u": {"label": "E2 without u (V5)", "options": {"link_dim": 2}, "link": "no_u"},
}


def label_of(variant: str) -> str:
    return VARIANTS[variant]["label"]


def kind_of(variant: str) -> str:
    return f"film_scratch_{variant}"


# ------------------------------------------------------------------ the link tensors of a variant
def contribution_links(seed: int, fold: int, part: str, pilot: bool, base: PartData) -> tuple[torch.Tensor, torch.Tensor]:
    """(full, blank) tensors (n, 38) = the 37 contributions and has_url, in the order of `base` (the text rows of the part)."""
    full, blank = ll.load_lane(seed, fold, pilot=pilot)[part]
    has_url = base.link_full[:, 3:4]
    out = []
    for frame in (full, blank):
        aligned = pd.DataFrame({"message_id": base.message_id}).merge(
            frame[["message_id"] + ll.CONTRIBUTIONS], on="message_id", how="left", validate="one_to_one")
        values = aligned[ll.CONTRIBUTIONS].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{part}: a contribution is missing or not finite")
        out.append(torch.cat([torch.tensor(values), has_url], dim=1))
    return out[0], out[1]


def variant_part(variant: str, seed: int, fold: int, part: str, word_to_idx: Dict[str, int], seq_len: int, pilot: bool = False) -> PartData:
    """The text ids, labels and link tensors of one part for `variant`. The rows and their order are those of film_data.part_inputs."""
    base = part_inputs(seed, fold, part, word_to_idx, seq_len, pilot)
    kind = VARIANTS[variant]["link"]
    if kind == "pu":
        return base
    if kind == "no_u":                                                   # [P, page_missing, has_url]
        keep = [0, 2, 3]
        return base._replace(link_full=base.link_full[:, keep], link_blank=base.link_blank[:, keep])
    full, blank = contribution_links(seed, fold, part, pilot, base)
    return base._replace(link_full=full, link_blank=blank)


def expected_trainable(variant: str) -> int:
    options = VARIANTS[variant]["options"]
    return film.expected_trainable(link_dim=options.get("link_dim", film.LINK_DIM), use_average=options.get("use_average", True))


# ------------------------------------------------------------------ train
def train_cell(variant: str, seed: int, fold: int, pilot: bool, epochs: int = fm.EPOCHS) -> tuple[dict, pd.DataFrame]:
    dataset = tc.dataset_path(pilot)
    kind = kind_of(variant)
    model, word_to_idx, seq_len = film.build_film(seed, fold, pilot, **VARIANTS[variant]["options"])
    train, val = (variant_part(variant, seed, fold, p, word_to_idx, seq_len, pilot) for p in ("train", "val"))
    checkpoint = fm.checkpoint_path(dataset, kind, seed, fold)
    result = fm.fit(model, (train.token_ids, train.link_full, train.link_blank, train.labels),
                    (val.token_ids, val.link_full, val.labels), seed, checkpoint, epochs=epochs)
    without, with_embedding = film.trainable_counts(model)
    assert without == expected_trainable(variant), f"{variant}: {without} trainable parameters, expected {expected_trainable(variant)}"
    moved = result.log.iloc[result.best_epoch]
    emb_dir = tc.embeddings_dir(dataset, seed, fold)
    fm.write_sidecar(checkpoint, {
        "kind": kind, "variant": variant, "seed": seed, "fold": fold, "options": film.model_options(model),
        "num_filters": model.num_filters, "kernel_size": model.kernel_size, "link_dim": model.link_dim,
        "dense_units": model.dense.out_features, "link_numbers": VARIANTS[variant]["link"], "seq_len_l": seq_len,
        "vocab_size": len(word_to_idx), "embeddings_dir": str(emb_dir.relative_to(tc.ROOT)),
        "sha256": fm.hashes(fm.sidecar_files(dataset, seed, fold, checkpoint, {"link_rf": film.link_rf_path(dataset, seed, fold)})),
        "warm_start": None,
        "recipe": {"optimizer": "adam", "lr": fm.LR, "epochs": epochs, "patience": fm.PATIENCE, "batch": fm.BATCH_SIZE,
                   "label_smoothing": LABEL_SMOOTHING, "class_weights": True, "dense_weight_decay": DENSE_WEIGHT_DECAY,
                   "generator_weight_decay": DENSE_WEIGHT_DECAY, "p_page_hidden": fm.P_PAGE_HIDDEN, "p_text_hidden": fm.P_TEXT_HIDDEN},
        "trainable_without_embedding": without, "trainable_with_embedding": with_embedding,
        "best_epoch": result.best_epoch, "epochs_run": result.epochs_run, "train_seconds": result.seconds,
        "val": {k: result.metrics[k] for k in fm.VAL_METRICS}})
    row = {"variant": variant, "seed": seed, "fold": fold, "n_train": len(train.labels), "n_val": len(val.labels),
           "trainable_without_embedding": without, "best_epoch": result.best_epoch, "epochs_run": result.epochs_run,
           "train_seconds": round(result.seconds, 3), "gamma_dev_mean": float(moved["gamma_dev_mean"]),
           "beta_abs_mean": float(moved["beta_abs_mean"]), "filters_moved": int(moved["filters_moved"]),
           **{k: result.metrics[k] for k in fm.VAL_METRICS}}
    return row, result.log.assign(variant=variant, seed=seed, fold=fold)


def run_train(pilot: bool, variants: List[str] | None = None, seeds: List[int] = SEEDS, folds: List[int] | None = None,
              epochs: int = fm.EPOCHS) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    variants = list(VARIANTS) if variants is None else variants
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows, logs = [], []
    for variant in variants:
        for seed in seeds:
            for fold in folds:
                row, log = train_cell(variant, seed, fold, pilot, epochs)
                rows.append(row)
                logs.append(log)
                print(f"{variant} seed {seed} fold {fold}: val F1 {row['val_f1']:.3f}, balanced accuracy {row['val_balanced_accuracy']:.3f} "
                      f"| best epoch {row['best_epoch']} of {row['epochs_run']} | {row['train_seconds']:.1f} s | "
                      f"{row['filters_moved']} filters moved", flush=True)
    out = fm.out_dir(dataset, "ablations")
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "train_summary.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(out / "train_log.csv", index=False)
    return summary


# ------------------------------------------------------------------ load and check
def load_variant(variant: str, seed: int, fold: int, pilot: bool = False) -> tuple[FiLMTextCNN, Dict[str, int], int, dict]:
    """(model in eval mode, word_to_idx, SEQ_LEN_L, sidecar), rebuilt from the sidecar alone; refuses a changed file."""
    dataset = tc.dataset_path(pilot)
    checkpoint = fm.checkpoint_path(dataset, kind_of(variant), seed, fold)
    sidecar = fm.read_sidecar(checkpoint)
    fm.check_hashes(sidecar["sha256"], fm.sidecar_files(dataset, seed, fold, checkpoint, {"link_rf": film.link_rf_path(dataset, seed, fold)}))
    embedding, word_to_idx = tc.frozen_embedding(tc.ROOT / sidecar["embeddings_dir"])
    model = FiLMTextCNN(embedding, sidecar["num_filters"], sidecar["kernel_size"], sidecar["link_dim"], sidecar["dense_units"],
                        **sidecar["options"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval(), word_to_idx, sidecar["seq_len_l"], sidecar


def run_check(pilot: bool, variants: List[str] | None = None, seeds: List[int] = SEEDS, folds: List[int] | None = None) -> pd.DataFrame:
    dataset = tc.dataset_path(pilot)
    variants = list(VARIANTS) if variants is None else variants
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for variant in variants:
        for seed in seeds:
            for fold in folds:
                model, word_to_idx, seq_len, sidecar = load_variant(variant, seed, fold, pilot)
                val = variant_part(variant, seed, fold, "val", word_to_idx, seq_len, pilot)
                again = fm.evaluate(model, val.token_ids, val.link_full, val.labels)
                gap = max(abs(again[k] - sidecar["val"][k]) for k in fm.VAL_METRICS)
                rows.append({"variant": variant, "seed": seed, "fold": fold, "val_f1": again["val_f1"], "max_gap": gap, "ok": gap == 0.0})
    summary = pd.DataFrame(rows)
    out = fm.out_dir(dataset, "ablations")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "check_summary.csv", index=False)
    return summary


# ------------------------------------------------------------------ compare
REFERENCES = [fc.BASELINE, fc.CONTROL_SCRATCH, fc.E2_SCRATCH]


def log_read(dataset, names: List[str], seeds: List[int], folds: List[int], read_test: bool) -> None:
    out = fm.out_dir(dataset, "ablations")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_reads.log", "a", encoding="utf-8") as log:
        log.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} dataset={dataset.name} seeds={seeds} folds={folds} "
                  f"read_test_flag={read_test} models={names}\n")


def test_predictions(pilot: bool, variants: List[str], seeds: List[int], folds: List[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for fold in folds:
            loaded = {name: fc.LOADERS[name][1](seed, fold, pilot) for name in REFERENCES}
            for variant in variants:
                loaded[label_of(variant)] = load_variant(variant, seed, fold, pilot)[:3]
            vocab, length = loaded[fc.BASELINE][1], loaded[fc.BASELINE][2]
            assert all(v == vocab and n == length for _, v, n in loaded.values()), "the models disagree on the fold's vocabulary"
            base = part_inputs(seed, fold, "test", vocab, length, pilot)
            by_label = {label_of(v): v for v in variants}
            for name, (model, _, _) in loaded.items():
                test = variant_part(by_label[name], seed, fold, "test", vocab, length, pilot) if name in by_label else base
                assert np.array_equal(test.message_id, base.message_id)
                for condition in fc.CONDITIONS:
                    p = fc.predict(model, name, test, condition)
                    rows.append(pd.DataFrame({"seed": seed, "fold": fold, "message_id": base.message_id, "label": base.labels.numpy(),
                                              "model": name, "condition": condition, "p_smishing": p, "pred": (p >= 0.5).astype(int),
                                              "has_url": base.link_full[:, 3].numpy().astype(int),
                                              "page_live": (1 - base.link_full[:, 2]).numpy().astype(int)}))
    out = pd.concat(rows, ignore_index=True)
    for (seed, model, condition), group in out.groupby(["seed", "model", "condition"]):
        assert group["message_id"].is_unique, f"seed {seed}: a message was tested twice by {model}"
    return out


def mcnemar_table(predictions: pd.DataFrame, variants: List[str]) -> pd.DataFrame:
    """Per seed, exact McNemar of the main E2 against every other model listed, Holm-corrected over that list (complete pages)."""
    complete = predictions[predictions["condition"] == "complete"]
    others = [fc.CONTROL_SCRATCH, fc.BASELINE] + [label_of(v) for v in variants]
    rows = []
    for seed, g in complete.groupby("seed"):
        wide = {m: gm.sort_values("message_id") for m, gm in g.groupby("model")}
        results = [fc.mcnemar(wide[fc.E2_SCRATCH]["label"].to_numpy(), wide[fc.E2_SCRATCH]["pred"].to_numpy(), wide[o]["pred"].to_numpy())
                   for o in others]
        adjusted = fc.holm([r["p_exact"] for r in results])
        for other, r, p in zip(others, results, adjusted):
            rows.append({"seed": seed, "model_a": fc.E2_SCRATCH, "model_b": other, **r, "p_holm": p, "significant_0.05": p < 0.05})
    return pd.DataFrame(rows)


def validation_table(pilot: bool, variants: List[str], seeds: List[int], folds: List[int]) -> pd.DataFrame:
    """Validation metrics and parameter counts from the sidecars (mean over the cells): this is what choices are made on."""
    dataset = tc.dataset_path(pilot)
    rows = []
    for name, kind in ((fc.E2_SCRATCH, "film_scratch"), (fc.CONTROL_SCRATCH, "concat_scratch")) + tuple(
            (label_of(v), kind_of(v)) for v in variants):
        sidecars = [fm.read_sidecar(fm.checkpoint_path(dataset, kind, s, f)) for s in seeds for f in folds]
        rows.append({"model": name, "cells": len(sidecars), "trainable_without_embedding": sidecars[0]["trainable_without_embedding"],
                     **{k: float(np.mean([c["val"][k] for c in sidecars])) for k in ("val_accuracy", "val_f1", "val_log_loss")},
                     "best_epoch_mean": float(np.mean([c["best_epoch"] for c in sidecars]))})
    return pd.DataFrame(rows)


def run_compare(pilot: bool, read_test: bool = False, variants: List[str] | None = None, seeds: List[int] = SEEDS,
                folds: List[int] | None = None) -> dict:
    dataset = tc.dataset_path(pilot)
    variants = list(VARIANTS) if variants is None else variants
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    if not pilot and not read_test:
        raise PermissionError("the test folds of the real data are read once, after everything is fixed: pass --read-test")
    log_read(dataset, REFERENCES + [label_of(v) for v in variants], seeds, folds, read_test)
    predictions = test_predictions(pilot, variants, seeds, folds)
    per_seed, summary = fc.metrics_tables(predictions)
    mc = mcnemar_table(predictions, variants)
    val = validation_table(pilot, variants, seeds, folds)
    out = fm.out_dir(dataset, "ablations")
    predictions.to_csv(out / "ablation_predictions.csv", index=False)
    per_seed.to_csv(out / "ablation_metrics.csv", index=False)
    summary.to_csv(out / "ablation_summary.csv", index=False)
    mc.to_csv(out / "ablation_mcnemar.csv", index=False)
    val.to_csv(out / "ablation_val.csv", index=False)
    return {"predictions": predictions, "per_seed": per_seed, "summary": summary, "mcnemar": mc, "val": val}


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["train", "check", "compare"])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a rehearsal; its numbers are not study results)")
    ap.add_argument("--read-test", action="store_true", help="compare, real data: reads the test folds (once)")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    if args.step == "train":
        summary = run_train(args.pilot, args.variants, args.seeds, args.folds)
        print("\n" + summary.groupby("variant")[["val_accuracy", "val_f1", "filters_moved"]].mean().round(4).to_string())
    elif args.step == "check":
        summary = run_check(args.pilot, args.variants, args.seeds, args.folds)
        print(f"{int(summary['ok'].sum())} of {len(summary)} rebuilt from their sidecars with a gap of exactly 0")
        sys.exit(0 if summary["ok"].all() else 1)
    else:
        try:
            result = run_compare(args.pilot, args.read_test, args.variants, args.seeds, args.folds)
        except PermissionError as error:
            print(error)
            sys.exit(2)
        print(result["val"].round(4).to_string(index=False))
        print("\n" + fc.format_summary(result["summary"]))
        print("\nMcNemar (exact, complete pages), E2 against each model, Holm-corrected per seed over this list:")
        print(result["mcnemar"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
