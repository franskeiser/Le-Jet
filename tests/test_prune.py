"""
Tests for the C-prune pruning steps (docs/c_prune_architecture.md Phase 5, steps 5-9).

Run with:  python -m pytest tests/test_prune.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.model.rf_gating_module import select_top_k
from src.unified import gate_rf as gr


def test_select_top_k_takes_the_first_k_in_ascending_id_order():
    ranking = [7, 2, 9, 0, 5, 1, 3, 4, 6, 8]
    assert select_top_k(ranking, 3).tolist() == [2, 7, 9]
    assert select_top_k(ranking, 10).tolist() == list(range(10))   # k = all: every filter, unchanged order
    assert select_top_k(np.array(ranking), 1).tolist() == [7]


@pytest.mark.parametrize("ranking", [[0, 1, 1, 3], [0, 1, 2, 5], [1, 2, 3, 4]])
def test_select_top_k_refuses_a_ranking_that_is_not_every_filter_once(ranking):
    with pytest.raises(ValueError, match="exactly once"):
        select_top_k(ranking, 2)


@pytest.mark.parametrize("k", [0, -1, 5])
def test_select_top_k_refuses_an_impossible_k(k):
    with pytest.raises(ValueError, match="k must be between 1 and 4"):
        select_top_k([3, 1, 0, 2], k)


def test_load_keep_idx_reads_the_saved_ranking(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "out_dir", lambda dataset: tmp_path)
    # Saved shuffled on purpose: the order comes from the rank column, not the row order.
    pd.DataFrame({"rank": [2, 0, 3, 1], "filter": [0, 3, 1, 2]}).to_csv(
        tmp_path / "filter_importance_seed42_fold0.csv", index=False)
    assert gr.load_keep_idx(42, 0, 2, pilot=True).tolist() == [2, 3]
