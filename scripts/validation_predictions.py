r"""Phase A1 of docs/validation_only_improvement_plan.md: score the VALIDATION part of every fold with the saved models.

    .venv\Scripts\python.exe scripts/validation_predictions.py

Reads: the saved Step 4 models (models/film/<dataset>/*.pt, *.joblib) and the `val` rows of each seed/fold split.
Never writes or uses a `test` row: the loaders in src/ build all three parts of a fold in memory (film_inputs.link_inputs), but this
script keeps only `val` rows and asserts that every written message id belongs to that fold's val part and not to its test part.

Models: Proposed Model (film_scratch), CNN-only (cnn_raw), Concatenation Fusion (concat_scratch), KNN-RF-ETC (saved .joblib).
Per row: seed, fold, message_id, label (1 = smishing), model, p_smishing, and for the neural models the Link RF numbers the model
was given (rf_P, rf_u, page_missing, has_url; the same numbers for every neural model).
For KNN-RF-ETC p_smishing is its vote fraction (text route) or URL-classifier probability (link route), not a calibrated probability.

Writes experiments/film/<dataset>/validation_study/validation_predictions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_N_FOLDS, SEEDS                              # noqa: E402
from src.data.split_crowd import load_fold                               # noqa: E402
from src.unified import film_compare as fc                               # noqa: E402
from src.unified import fit_model as fm                                  # noqa: E402
from src.unified import raw_cnn_control as rc                            # noqa: E402
from src.unified import text_cnn as tc                                   # noqa: E402
from src.unified import voting_baseline as vb                            # noqa: E402
from src.unified.film_data import part_inputs                            # noqa: E402

NEURAL = {fc.E2_SCRATCH: "Proposed Model", fc.RAW_CNN: "CNN-only", fc.CONTROL_SCRATCH: "Concatenation Fusion"}
KNN = "KNN-RF-ETC"


def main() -> int:
    dataset = tc.dataset_path(False)
    rows = []
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            parts = load_fold(seed, fold)
            val_ids = set(parts["val"]["message_id"])
            test_ids = set(parts["test"]["message_id"])
            assert not (val_ids & test_ids)
            loaded = {n: fc.LOADERS[n][1](seed, fold, False) for n in NEURAL}
            vocab, seq_len = loaded[fc.E2_SCRATCH][1], loaded[fc.E2_SCRATCH][2]
            assert all(v == vocab and l == seq_len for _, v, l in loaded.values())
            val = part_inputs(seed, fold, "val", vocab, seq_len)
            raw_val = rc.raw_part_inputs(seed, fold, "val", vocab, seq_len)
            assert set(val.message_id) == val_ids and set(raw_val.message_id) == val_ids
            link = val.link_full.numpy()
            for name, (model, _, _) in loaded.items():
                data = raw_val if name == fc.RAW_CNN else (fc.inputs_of(name, model, seed, fold, "val", vocab, seq_len)
                                                           if getattr(model, "url_branch_units", 0) else val)
                p = fc.predict(model, name, data, "complete")
                rows.append(pd.DataFrame({"seed": seed, "fold": fold, "message_id": data.message_id, "label": data.labels.numpy(),
                                          "model": NEURAL[name], "p_smishing": p, "rf_P": link[:, 0], "rf_u": link[:, 1],
                                          "page_missing": link[:, 2], "has_url": link[:, 3]}))
            p, _, _ = vb.score(vb.load_or_fit(parts, seed, fold, False), parts["val"])
            rows.append(pd.DataFrame({"seed": seed, "fold": fold, "message_id": parts["val"]["message_id"].to_numpy(),
                                      "label": (parts["val"]["label"] == "smishing").astype(int).to_numpy(), "model": KNN,
                                      "p_smishing": p, "rf_P": np.nan, "rf_u": np.nan, "page_missing": np.nan, "has_url": np.nan}))
            print(f"seed {seed} fold {fold}: scored {len(val_ids)} validation messages", flush=True)
    out = pd.concat(rows, ignore_index=True)
    # the checks: only val ids, each message once per model/seed/fold, same ids for every model, probabilities valid
    for (seed, fold), g in out.groupby(["seed", "fold"]):
        parts = load_fold(seed, fold)
        assert set(g["message_id"]) == set(parts["val"]["message_id"]), "written ids are not exactly the val part"
        assert not (set(g["message_id"]) & set(parts["test"]["message_id"])), "a test row was written"
        for m, gm in g.groupby("model"):
            assert gm["message_id"].is_unique, f"{m}: a message twice"
    assert out["p_smishing"].between(0, 1).all()
    path = fm.out_dir(dataset, "validation_study")
    path.mkdir(parents=True, exist_ok=True)
    out.to_csv(path / "validation_predictions.csv", index=False)
    print(f"wrote {len(out)} rows to {path / 'validation_predictions.csv'}; checks passed (val rows only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
