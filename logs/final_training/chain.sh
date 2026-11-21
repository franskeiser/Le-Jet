set -e
PY=.venv/Scripts/python.exe
step(){ echo "=== $1 $(date +%T)"; }
step rows;  $PY -m src.unified.link_rf_study rows --trees 200
step tune;  $PY -m src.unified.link_lane tune
step oof;   $PY -m src.unified.link_lane oof
step final; $PY -m src.unified.link_lane final
step check; $PY -m src.unified.link_lane check
step film_build; $PY -m src.unified.film build
step film_foldlen; $PY -m src.unified.film train --warm-start none
step film_len50;   $PY -m src.unified.film train --warm-start none --seq-len 50
echo ALL_DONE
