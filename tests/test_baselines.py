"""
Tests for src/unified/baselines.py (docs/c_prune_architecture.md Phase 7).

Run with:  python -m pytest tests/test_baselines.py
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest

from src.unified import baselines as bl


def test_mcnemar_counts_only_the_messages_where_one_model_is_right():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    a = np.array([0.9, 0.8, 0.2, 0.1, 0.9, 0.7, 0.2, 0.3])   # wrong on messages 5 and 6
    b = np.array([0.2, 0.9, 0.8, 0.1, 0.4, 0.2, 0.3, 0.3])   # wrong on messages 0, 2, 4 and 6
    result = bl.mcnemar(y, a, b)
    # A right / B wrong: 0, 2, 4 -> b = 3; A wrong / B right: 5 -> c = 1; both wrong on 6: not counted.
    assert (result["b"], result["c"]) == (3, 1)
    assert result["p_exact"] == pytest.approx(binomtest(3, 4, 0.5).pvalue)
    assert bl.mcnemar(y, a, a) == {"b": 0, "c": 0, "p_exact": 1.0}


def test_scores_treat_smishing_as_positive():
    s = bl.scores(np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.6, 0.1]))
    assert s == {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_real_test_folds_are_not_read_without_asking():
    with pytest.raises(SystemExit, match="--read-test"):
        bl.run_compare(pilot=False)


def test_time_per_message_skips_the_warm_up(monkeypatch):
    monkeypatch.setattr(bl, "WARM_UP", 3)
    calls = []
    ms = bl.time_per_message(calls.append, 5)
    assert calls == [0, 1, 2, 0, 1, 2, 3, 4] and ms.shape == (5,) and (ms >= 0).all()


def test_holm_correction():
    # Sorted 0.01, 0.02, 0.04 with m = 3: 0.03, max(0.03, 0.04) = 0.04, max(0.04, 0.04) = 0.04; original order back.
    assert bl.holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert bl.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])   # capped at 1
    assert bl.holm([0.003]) == pytest.approx([0.003])         # one test: unchanged


def test_ungated_is_c_prune_with_every_filter_kept_and_the_link_lane():
    assert bl.BASELINES["ungated"]["link"] and not bl.BASELINES["plain_cnn"]["link"]
