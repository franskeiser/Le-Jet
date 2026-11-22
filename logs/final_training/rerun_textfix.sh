#!/bin/bash
# Rerun of the group mate's Steps 3-9 plus Step 10 on the dataset with the 16 rewritten bare-link messages (2026-11-22). His scripts, unchanged, in his order:
# chain.sh (link lane + E2), step4.sh (the other models), then his build_step3..9 files. Stops at the first failure.
cd /d/Le-Jet
PY=.venv/Scripts/python.exe
step(){ echo "=== $(date +%T) $*"; }
set -e
step chain.sh;  bash logs/final_training/chain.sh
step step4.sh;  bash logs/final_training/step4.sh
step step3;     $PY scripts/build_step3_files.py
step step4;     $PY scripts/build_step4_files.py
step step5;     $PY scripts/build_step5_files.py --read-test
step step6;     $PY scripts/build_step6_files.py --read-test
step step7;     $PY scripts/build_step7_files.py
step step8;     $PY scripts/build_step8_files.py
step step9;     $PY scripts/build_step9_files.py
step step10;    $PY scripts/build_step10_files.py
step ALL_DONE
