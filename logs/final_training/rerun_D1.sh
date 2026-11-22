#!/bin/bash
# Adopt D1 (2026-11-22, Franz): retrain the Proposed model (film_scratch) with text dropout before the Conv, then rebuild Steps 4-10.
# The baselines (cnn_raw, concat_scratch, voting, cnn) are unchanged and not retrained. Step 5 and Step 6 read the test folds (4th read).
cd /d/Le-Jet
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step(){ echo "=== $(date +%T) $*"; }
set -e
if [ "$1" != "steps" ]; then
  step film_train; $PY -m src.unified.film train --warm-start none --text-dropout input
  step film_check; $PY -m src.unified.film check --warm-start none
  [ "$1" = "train" ] && { step TRAIN_DONE; exit 0; }
fi
step step4;  $PY scripts/build_step4_files.py
step step5;  $PY scripts/build_step5_files.py --read-test
step step6;  $PY scripts/build_step6_files.py --read-test
step step7;  $PY scripts/build_step7_files.py
step step8;  $PY scripts/build_step8_files.py
step step9;  $PY scripts/build_step9_files.py
step step10; $PY scripts/build_step10_files.py
step ALL_DONE
