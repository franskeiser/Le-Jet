"""Tests for src/unified/time_link_rf.py (plan docs/film_architecture_plan.md, step 1.3).

Run with:  python -m pytest tests/test_time_link_rf.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.unified import time_link_rf as tl
from src.unified.prepare_web_37 import FEATURES_37


@pytest.fixture(scope="module")
def forest_and_rows():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((120, len(FEATURES_37))), columns=FEATURES_37)
    y = (X.iloc[:, 0] + X.iloc[:, 1] > 1).astype(int)
    rf = RandomForestClassifier(n_estimators=8, random_state=0, n_jobs=1).fit(X, y)
    rows = X.copy()
    rows.insert(0, "message_id", [f"m{i}" for i in range(len(rows))])
    return rf, rows


def test_time_variants_counts_only_the_timed_messages(forest_and_rows):
    rf, rows = forest_and_rows
    out = tl.time_variants(rf, rows, n=20, warmup=5, seed=1)
    assert len(out) == 20 and out["message_id"].is_unique
    assert list(out.columns) == ["message_id", "ms_with_contributions", "ms_p_u_only"]
    assert (out[["ms_with_contributions", "ms_p_u_only"]] > 0).all().all()


def test_time_variants_refuses_when_there_are_too_few_messages(forest_and_rows):
    rf, rows = forest_and_rows
    with pytest.raises(ValueError, match="need"):
        tl.time_variants(rf, rows.iloc[:10], n=20, warmup=5)


def test_time_variants_runs_the_forest_on_one_thread(forest_and_rows):
    rf, rows = forest_and_rows
    rf.n_jobs = 4
    tl.time_variants(rf, rows, n=3, warmup=1)
    assert rf.n_jobs == 1


def test_contributions_cost_more_than_p_and_u(forest_and_rows):
    """The walk through every tree for the 37 contributions is the extra work; the P and u path must be cheaper."""
    rf, rows = forest_and_rows
    out = tl.time_variants(rf, rows, n=30, warmup=5)
    summary = tl.summarize(out)
    assert summary["n"] == 30
    assert summary["median_ratio"] > 1
    assert summary["with_contributions_p95"] >= summary["with_contributions_median"]


def test_machine_info_records_the_threads_and_versions():
    info = tl.machine_info()
    assert info["threads"]["OMP_NUM_THREADS"] == "1" and info["cpu_count"] and info["scikit_learn"]
