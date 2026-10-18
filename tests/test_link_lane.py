"""
Tests for src/unified/link_lane.py (docs/c_prune_architecture.md Phase 3).

Run with:  python -m pytest tests/test_link_lane.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.model.rf_extractor import PARAM_GRID
from src.unified import link_lane as ll
from src.unified.prepare_web_37 import FEATURES_37


def _train(n_domains=30, per_domain=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_domains):
        smishing = d % 2 == 1
        for _ in range(per_domain):
            # length_url separates the classes; the other features are noise.
            feats = {f: float(rng.normal()) for f in FEATURES_37}
            feats["length_url"] = 60.0 + rng.normal() if smishing else 20.0 + rng.normal()
            rows.append({"label": "smishing" if smishing else "legitimate", "link_domain": f"d{d}.xyz",
                         "template_id": f"t{len(rows)}", **feats})
    return pd.DataFrame(rows)


def test_tune_returns_a_grid_candidate_and_every_score():
    params, grid = ll.tune(_train(), "domain", seed=42)
    assert set(params) == set(PARAM_GRID)
    assert len(grid) == np.prod([len(v) for v in PARAM_GRID.values()])
    assert grid["f1"].max() == 1.0          # length_url alone separates the classes
    best = grid.sort_values(["f1", "log_loss"], ascending=[False, True]).iloc[0]
    assert {k: str(v) for k, v in params.items()} == {k: best[k] for k in PARAM_GRID}


def test_tune_is_repeatable():
    a, grid_a = ll.tune(_train(), "domain", seed=7)
    b, grid_b = ll.tune(_train(), "domain", seed=7)
    assert a == b
    pd.testing.assert_frame_equal(grid_a, grid_b)
