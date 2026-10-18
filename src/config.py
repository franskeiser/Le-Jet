# src/config.py
import os
from pathlib import Path

# Where trained models, experiment reports and the demo data are written and read. Defaults to the
# repository (models/, experiments/, data/demo/). Set LEJET_ARTIFACTS to another folder to keep them
# elsewhere (e.g. D:\LeJetData\artifacts when Windows Controlled Folder Access blocks writes into the repo).
# Inputs that are committed (the embeddings, the cleaned data, the splits) are always read from the repo.
ARTIFACTS_ROOT = Path(os.environ.get("LEJET_ARTIFACTS", Path(__file__).resolve().parents[1]))

# Shared random seeds — used for ALL stochastic operations:
# train/val/test splits, Random Forest fits, NN weight initialization, etc.
SEEDS = [42, 123, 2024]

# Task 1.4 checkpoint 3 — fixed sequence length for padding/truncation.
# Retested 2026-09-30 against the current production dataset
# (data/labeled/v2_2026-09-15.csv, 18,885 rows) using the same
# post-URL-masking/placeholder-fix, pre-padding methodology:
#   median=19  p90=36  p95=50  p99=79  max=229
# The original L=40 was fit to the 2026-08-29 dataset (p95=38) and had drifted
# stale — it was truncating 8.0% of current messages instead of the intended
# ~5%. L=50 (95th percentile) restores the intended ~5% truncation rate.
# See experiments/token_count_histogram_retest_2026-09-30.png.
SEQ_LEN_L = 50

# Phase 0 decisions #2/#3, locked 2026-09-08 (DEV_TEAM_BUILD_PLAN.md:1579-1580). Single
# source of truth -- notebooks/07_run_phase_c.py, notebooks/08_run_ungated_baseline.py,
# and load_model() all import these rather than redefining them locally.
NUM_FILTERS = 64
KERNEL_SIZE = 3  # one conv branch (Le-Jet decision), so the pooled CNN output is NUM_FILTERS-dim
DENSE_UNITS = 64

# Label smoothing for every training loss (Phase A, Phase C, ungated baseline). With 2
# classes, 0.1 trains toward 0.95/0.05 targets instead of hard 1/0, so the model can't
# drive its logits to the extremes that made the UI show a 100% confidence score.
# Temperature fitting (notebooks/09) deliberately keeps plain NLL -- calibration is
# measured against the true labels, not the smoothed ones. Set to 0.0 to reproduce the
# pre-smoothing runs (backed up under models/backup_pre_label_smoothing/).
LABEL_SMOOTHING = 0.1

# Active labeled dataset, its link features and its split files. Every loader
# (split_dataset.load_split_df and friends) reads these.
# sms_v2_balanced = v2_2026-09-15 + 3,025 synthetic legitimate-with-link messages
# (src/data/add_synthetic_legit_links.py, docs/sms_synthetic_legit_links.md).
DATASET_PATH = "data/processed/sms_v2_balanced.csv"
LINK_FEATURES_PATH = "data/processed/sms_link_features.csv"  # src/data/build_sms_link_features.py
SPLITS_DIR = "data/splits"  # src/data/split_dataset.py (grouped by domain / template)

# C-prune study (docs/c_prune_architecture.md). Kept apart from the paths above so the current
# model keeps working. The pilot is SYNTHETIC text: a pipeline test only, never study results.
CROWD_DATASET_PATH = "data/processed/crowd_smishing_v1.csv"   # src/data/build_crowd_dataset.py
CROWD_PILOT_PATH = "data/processed/crowd_pilot_synthetic.csv"  # ... --pilot
CROWD_SPLITS_DIR = "data/splits_crowd"                         # src/data/split_crowd.py
CROWD_EMBEDDINGS_DIR = "models/embeddings_crowd"               # src/features/build_crowd_embeddings.py
CROWD_N_FOLDS = 5
CROWD_VAL_FRACTION = 0.15  # of each training part, grouped like the folds, for early stopping

# Optional but recommended alongside this:
DATA_PATH = "data/processed/smishing_dataset.csv"
LABEL_COLUMN = "subcategory"
TRAIN_VAL_TEST_SPLIT = (0.7, 0.15, 0.15)