# Experiment papers: one copy per retraining

Written by `scripts/build_experiment_papers.py` (called at the end of `scripts/build_step10_files.py`). A new copy appears when the Step 5 predictions change; older copies are never overwritten.

F1 order: CNN-only, text + 37 URL features / Concatenation fusion / KNN-RF-ETC voting / Proposed (RF-conditioned FiLM). Seeds significant (McNemar, Proposed better) order: CNN-only, text + 37 URL features / Concatenation fusion / KNN-RF-ETC voting.

| File | Generated | Predictions fingerprint | Commit | Mean F1 | Seeds significant (of 3) |
|---|---|---|---|---|---|
| [Experiment_Papers_20261122-1113_bf398f3095.md](Experiment_Papers_20261122-1113_bf398f3095.md) | 2026-11-22T11:13 | bf398f3095 | 1157a84 | 0.9450 / 0.9461 / 0.8192 / 0.9487 | 1 / 1 / 3 |
