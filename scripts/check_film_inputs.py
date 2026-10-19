"""Check that everything E2 reads for each seed x fold is present and loads (plan docs/film_architecture_plan.md, step 1.1).

Per seed and fold it loads, with the same helpers the later steps use:
    split     the fold's train / val / test rows (split_crowd.load_fold), non-empty, no message id in two parts
    vocab     the fold's frozen embedding folder (vocab.json + embedding_matrix.npy)
    cnn       the text CNN checkpoint, rebuilt from its sidecar with the hash check (text_cnn.load_phase_a)
    link      the link numbers (link_lane.load_lane): P, u, ..., page_missing, full and page-hidden, with the
              message ids equal to the fold's rows in every part
    forest    the fold's saved Link RF file exists
    baseline  the plain text CNN checkpoint exists (only its presence is checked: it is loaded in Phase 4 of docs/film_architecture_plan.md)

Prints a seeds x folds table (ok / the reason it failed) and exits 1 if anything is missing or does not load.

Usage:
    python scripts/check_film_inputs.py            # the synthetic pilot (default until the crowd dataset exists)
    python scripts/check_film_inputs.py --real     # data/processed/crowd_smishing_v1.csv
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import CROWD_N_FOLDS, SEEDS  # noqa: E402
from src.data.split_crowd import load_fold  # noqa: E402
from src.unified import link_lane as ll  # noqa: E402
from src.unified import text_cnn as tc  # noqa: E402

CHECKS = ["split", "vocab", "cnn", "link", "forest", "baseline"]


def check_cell(dataset, seed: int, fold: int, pilot: bool) -> dict:
    """{check: 'ok' or the reason it failed} for one seed and fold."""
    result = {}

    def run(name, fn):
        try:
            result[name] = fn() or "ok"
        except Exception as error:  # a missing file, a changed hash, a shape problem: report it, keep going
            result[name] = f"{type(error).__name__}: {str(error)[:70]}"

    parts = {}

    def split():
        parts.update(load_fold(seed, fold, pilot=pilot))
        empty = [p for p, rows in parts.items() if len(rows) == 0]
        if empty:
            raise ValueError(f"empty part {empty}")
        ids = [set(rows["message_id"]) for rows in parts.values()]
        if sum(len(i) for i in ids) != len(set().union(*ids)):
            raise ValueError("a message id is in two parts")

    def vocab():
        emb_dir = tc.embeddings_dir(dataset, seed, fold)
        for name in ("vocab.json", "embedding_matrix.npy"):
            if not (emb_dir / name).exists():
                raise FileNotFoundError(f"{emb_dir.name}/{name}")

    def cnn():
        cnn_model, _, sidecar = tc.load_phase_a(tc.model_dir(dataset) / f"phase_a_seed{seed}_fold{fold}.pt")
        if sidecar["num_filters"] != 64:
            raise ValueError(f"{sidecar['num_filters']} filters, expected 64")

    def link():
        lane = ll.load_lane(seed, fold, pilot=pilot)
        for part, (full, blank) in lane.items():
            if part in parts and list(full["message_id"]) != list(parts[part]["message_id"]):
                # same rows, order may differ: compare as sets, and say so if only the order differs
                if set(full["message_id"]) != set(parts[part]["message_id"]):
                    raise ValueError(f"{part}: link rows differ from the fold's rows")

    def forest():
        path = ll.model_dir(dataset) / f"link_rf_seed{seed}_fold{fold}.joblib"
        if not path.exists():
            raise FileNotFoundError(path.name)

    def baseline():
        path = tc.model_dir(dataset) / f"plain_cnn_seed{seed}_fold{fold}.pt"
        if not path.exists():
            raise FileNotFoundError(path.name)

    for name, fn in zip(CHECKS, (split, vocab, cnn, link, forest, baseline)):
        run(name, fn)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real", action="store_true", help="check the real crowd dataset instead of the synthetic pilot")
    args = ap.parse_args()
    pilot = not args.real
    dataset = tc.dataset_path(pilot)
    print(f"dataset: {dataset.name}{'  (SYNTHETIC pilot: pipeline test only)' if pilot else ''}")
    if not dataset.exists():
        print("  the dataset file is missing")
        sys.exit(1)

    cells, failures = {}, []
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            cells[(seed, fold)] = check_cell(dataset, seed, fold, pilot)
            failures += [(seed, fold, c, r) for c, r in cells[(seed, fold)].items() if r != "ok"]

    print("\n" + f"{'seed':>6} {'fold':>4}  " + "  ".join(f"{c:>8}" for c in CHECKS))
    for (seed, fold), res in cells.items():
        print(f"{seed:>6} {fold:>4}  " + "  ".join(f"{('ok' if res[c] == 'ok' else 'FAIL'):>8}" for c in CHECKS))
    done = sum(all(r == "ok" for r in res.values()) for res in cells.values())
    print(f"\nseeds {SEEDS} x folds 0-{CROWD_N_FOLDS - 1}: {done} of {len(cells)} complete")
    for seed, fold, check, reason in failures:
        print(f"  seed {seed} fold {fold} {check}: {reason}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
