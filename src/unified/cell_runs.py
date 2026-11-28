"""Resumable training runs, one seed x fold cell at a time (2026-11-25).

A training command (film.py, raw_cnn_control.py, concat_control.py with --resume) writes every finished cell's tables to <out>/cells/
together with the cell's `signature` (the architecture, recipe and options it was trained with). A later run with the same signature
skips that cell, so a run stopped by its time budget (--minutes), or several processes given different seeds, add up to one complete
set; assemble() then writes the usual <out>/<table>.csv files (train_summary, train_log, ...) in seed x fold order. A cell whose saved
signature differs is trained again, so an older configuration is never mistaken for the current one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def cells_dir(out: Path) -> Path:
    return out / "cells"


def _stem(seed: int, fold: int) -> str:
    return f"seed{seed}_fold{fold}"


def cell_done(out: Path, seed: int, fold: int, signature: dict) -> bool:
    marker = cells_dir(out) / f"{_stem(seed, fold)}.json"
    if not marker.exists():
        return False
    return json.loads(marker.read_text(encoding="utf-8")).get("signature") == json.loads(json.dumps(signature, default=str))


def save_cell(out: Path, seed: int, fold: int, tables: Dict[str, pd.DataFrame], signature: dict) -> None:
    """Writes each table of one finished cell, then its marker (written last, so a half-written cell is never counted as done)."""
    folder = cells_dir(out)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}_{_stem(seed, fold)}.csv", index=False)
    (folder / f"{_stem(seed, fold)}.json").write_text(json.dumps({"seed": seed, "fold": fold, "tables": sorted(tables),
                                                                  "signature": signature}, indent=1, default=str), encoding="utf-8")


def assemble(out: Path, seeds: Iterable[int], folds: Iterable[int], tables: List[str], signature: dict) -> tuple[Dict[str, pd.DataFrame], int]:
    """(table name -> all finished cells' rows in seed x fold order, number of finished cells). Writes <out>/<name>.csv for each table
    from the cells done with `signature` (a partial set is written too; the caller reports how many cells are missing)."""
    frames: Dict[str, list] = {name: [] for name in tables}
    finished = 0
    for seed in seeds:
        for fold in folds:
            if not cell_done(out, seed, fold, signature):
                continue
            finished += 1
            for name in tables:
                frames[name].append(pd.read_csv(cells_dir(out) / f"{name}_{_stem(seed, fold)}.csv"))
    result = {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for name, parts in frames.items()}
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        if len(frame):
            frame.to_csv(out / f"{name}.csv", index=False)
    return result, finished


class Budget:
    """A wall-clock budget: fits(guess) is False when a cell expected to take `guess` seconds would end after the deadline."""

    def __init__(self, minutes: float | None):
        self.deadline = None if minutes is None else time.perf_counter() + 60 * minutes
        self.times: List[float] = []

    def fits(self, first_guess: float = 240.0) -> bool:
        if self.deadline is None:
            return True
        guess = 1.3 * (sum(self.times) / len(self.times) if self.times else first_guess)
        return time.perf_counter() + guess <= self.deadline

    def add(self, seconds: float) -> None:
        self.times.append(seconds)
