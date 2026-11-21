set -e
PY=.venv/Scripts/python.exe
step(){ echo "=== $1 $(date +%T)"; }
step cnn_only_train;  $PY -m src.unified.raw_cnn_control train
step cnn_only_check;  $PY -m src.unified.raw_cnn_control check
step concat_train;    $PY -m src.unified.concat_control train
step concat_check;    $PY -m src.unified.concat_control check
step knn_train;       $PY -m src.unified.voting_baseline train
step textcnn_train;   $PY -m src.unified.cnn_baseline train
step textcnn_check;   $PY -m src.unified.cnn_baseline check
step e2_check;        $PY -m src.unified.film check
echo ALL_DONE
