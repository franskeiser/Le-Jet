r"""Step 4 of the experimental testing guide: Training_Records.xlsx, from the files each model's training wrote.

    .venv\Scripts\python.exe scripts/build_step4_files.py [--dataset lejet_final_dataset]

Nothing is trained here. It gathers what training wrote, scores every saved model on its own TRAINING part (training performance:
evaluation mode, no dropout; training messages carry their out-of-fold Link RF numbers, as during training) and checks that every
required run completed. The test folds are not read.

The guide's "Results to record" and where each one is:
  Training and validation performance     Runs: train_* and val_* columns (accuracy, balanced accuracy, precision, recall, F1)
  Number of training and validation msgs  Runs: n_train, n_val
  Best epoch and early stopping           Runs: best_epoch, epochs_run, stopped_early, patience
  Model configuration, seed and fold      Configuration (one row per model) + Runs: seed, fold
  Training completion or failure          Runs: status; Summary: runs_completed of 15
  Saved model checkpoints                 Runs: checkpoint, checkpoint_exists, rebuild_check (rebuilt from its saved files, gap 0)
Files to save: Training Logs = Training_Log sheet (every epoch of every neural run) and experiments/film/<dataset>/<model>/train_log.csv;
Model Checkpoints = models/film/<dataset>/ (.pt + .json sidecar, or .joblib); Model Configuration Records = Configuration sheet and
each checkpoint's .json sidecar.
Validation chose the best epoch, so validation scores are a training record, not results (those are Steps 5 and 7).
Output: experiments/film/<dataset>/step4/Training_Records.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_N_FOLDS, SEEDS                              # noqa: E402
from src.unified import fit_model as fm                                 # noqa: E402
from src.unified.link_lane import LINK_TREES                            # noqa: E402

PERF = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
# guide name, role, experiments sub-folder (train / check), checkpoint prefix, film_compare loader name (None: not a neural network)
MODELS = [
    ("Proposed model (E2: CNN + RF + FiLM)", "Step 4 main model", "train_scratch", "check_scratch", "film_scratch", "E2_SCRATCH"),
    ("CNN-only (text + 37 URL features)", "Step 4 main model", "cnn_raw", "cnn_raw", "cnn_raw", "RAW_CNN"),
    ("Concatenation fusion", "Step 4 main model", "concat_scratch", "concat_scratch", "concat_scratch", "CONTROL_SCRATCH"),
    ("KNN-RF-ETC (Jain et al. 2022)", "Step 4 main model", "voting", None, None, None),
    ("Text-only CNN", "Step 6 component check", "cnn_baseline", "cnn_baseline", "cnn", "BASELINE"),
]
CONFIG = {
    "Proposed model (E2: CNN + RF + FiLM)": ("text + Link RF's P, u, page_missing (+ has_url bypass)",
                                              "fastText 300 (fine-tuned) -> Conv1D 64x3 -> FiLM (generator 67->128) -> ReLU, max-pool -> Dense 64 -> 2"),
    "CNN-only (text + 37 URL features)": ("text + 37 URL features (prepared: log, standardised, missing -> 0 + 6 flags) + has_url",
                                           "text branch Conv1D 64x3 + URL branch Dense 44->64 -> concatenate -> Dense 64 -> 2"),
    "Concatenation fusion": ("text + Link RF's P, u, page_missing, has_url", "Conv1D 64x3 -> max-pool -> [64 + 4] -> Dense 64 -> 2 (no FiLM)"),
    "KNN-RF-ETC (Jain et al. 2022)": ("no URL: text; URL: the link string only", "TF-IDF 3,000 -> SMOTE -> vote KNN + RF + Extra Trees | "
                                      "link tokens -> CountVectorizer -> Logistic Regression"),
    "Text-only CNN": ("text only", "fastText 300 (fine-tuned) -> Conv1D 64x3 -> max-pool -> Dense 64 -> 2"),
}
RECIPE = ("Adam lr 0.001, batch 64, up to 30 epochs, early stop after 3 epochs without better validation balanced accuracy (best epoch "
          "kept), class weights, label smoothing 0.1, weight decay 1e-4 on Dense 64 (and the generator), modality dropout page 0.25 / "
          "text 0.10 (models that read the link), from scratch")
RUN_COLUMNS = (["model", "seed", "fold", "status", "n_train", "n_val", "best_epoch", "epochs_run", "stopped_early", "patience",
                "train_seconds"] + [f"train_{m}" for m in PERF] + [f"val_{m}" for m in PERF]
               + ["val_log_loss", "checkpoint", "checkpoint_exists", "rebuild_check"])


def train_performance_neural(loader_name: str, seed: int, fold: int) -> dict:
    """The saved model scored on its own training part, as validation is scored (fit_model.evaluate)."""
    from src.unified import film_compare as fc
    from src.unified import raw_cnn_control as rc
    from src.unified.film_data import part_inputs
    name = getattr(fc, loader_name)
    model, w2i, seq_len = fc.LOADERS[name][1](seed, fold, False)
    data = (rc.raw_part_inputs if loader_name == "RAW_CNN" else part_inputs)(seed, fold, "train", w2i, seq_len)
    m = fm.evaluate(model, data.token_ids, data.link_full, data.labels)
    return {f"train_{k}": m[f"val_{k}"] for k in PERF}


def train_performance_knn(seed: int, fold: int) -> dict:
    import joblib
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
    from src.data.split_crowd import load_fold
    from src.unified import voting_baseline as vb
    train = load_fold(seed, fold)["train"]
    detector = joblib.load(vb.model_path(False, seed, fold))["detector"]
    _, pred, _ = vb.score(detector, train)
    y = (train["label"] == "smishing").astype(int).to_numpy()
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"train_accuracy": accuracy_score(y, pred), "train_balanced_accuracy": balanced_accuracy_score(y, pred),
            "train_precision": p, "train_recall": r, "train_f1": f}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lejet_final_dataset")
    a = ap.parse_args()
    dataset = Path(a.dataset)
    exp, models_dir = fm.out_dir(dataset, ""), fm.model_dir(dataset)
    runs, details, logs, summary, config = [], [], [], [], []
    for name, role, train_dir, check_dir, prefix, loader in MODELS:
        expected = len(SEEDS) * CROWD_N_FOLDS
        path = exp / train_dir / "train_summary.csv"
        if not path.exists():
            summary.append({"model": name, "role": role, "runs_completed": 0, "of": expected})
            continue
        t = pd.read_csv(path)
        if check_dir and (exp / check_dir / "check_summary.csv").exists():
            c = pd.read_csv(exp / check_dir / "check_summary.csv")[["seed", "fold", "ok"]].rename(columns={"ok": "rebuild_check"})
            t = t.merge(c, on=["seed", "fold"], how="left")
            t["rebuild_check"] = t["rebuild_check"].map({True: "passed (gap 0)", False: "FAILED"}).fillna("not run")
        else:
            t["rebuild_check"] = "n/a (scikit-learn model, saved with joblib)"
        t.insert(0, "model", name)
        ext = ".pt" if prefix else ".joblib"
        stem = prefix or "voting"
        t["checkpoint"] = [f"models/film/{dataset}/{stem}_seed{s}_fold{f}{ext}" for s, f in zip(t.seed, t.fold)]
        t["checkpoint_exists"] = [(models_dir / f"{stem}_seed{s}_fold{f}{ext}").exists() for s, f in zip(t.seed, t.fold)]
        if prefix:
            t["stopped_early"] = np.where(t["epochs_run"] < fm.EPOCHS, "yes", "no (ran all epochs)")
            t["patience"] = fm.PATIENCE
        else:
            t["best_epoch"] = t["epochs_run"] = t["stopped_early"] = t["patience"] = "n/a (not trained in epochs)"
            t["train_seconds"] = t["fit_seconds"]
            t["val_log_loss"] = np.nan
        ok = t["checkpoint_exists"] & (~t["rebuild_check"].eq("FAILED"))
        t["status"] = np.where(ok, "completed", "FAILED")
        perf = [train_performance_neural(loader, s, f) if loader else train_performance_knn(s, f) for s, f in zip(t.seed, t.fold)]
        t = pd.concat([t, pd.DataFrame(perf)], axis=1)
        print(f"{name}: training performance measured on {len(perf)} runs", flush=True)
        runs.append(t[RUN_COLUMNS])
        extra = [c for c in t.columns if c not in RUN_COLUMNS and c not in ("kind", "fit_seconds")]
        details.append(t[["model", "seed", "fold"] + [c for c in extra if c not in ("seed", "fold")]])
        if (exp / train_dir / "train_log.csv").exists():
            lg = pd.read_csv(exp / train_dir / "train_log.csv")
            lg.insert(0, "model", name)
            logs.append(lg)
        row = {"model": name, "role": role, "runs_completed": int((t["status"] == "completed").sum()), "of": expected,
               "runs_failed": int((t["status"] == "FAILED").sum()) + expected - len(t)}
        for m in PERF:
            row[f"train_{m}_mean"] = t[f"train_{m}"].mean()
            row[f"val_{m}_mean"], row[f"val_{m}_sd"] = t[f"val_{m}"].mean(), t[f"val_{m}"].std()
        row["best_epoch_mean"] = pd.to_numeric(t["best_epoch"], errors="coerce").mean()
        row["runs_stopped_early"] = int((t["stopped_early"] == "yes").sum()) if prefix else "n/a"
        summary.append(row)
        reads, layers = CONFIG[name]
        if prefix:
            sidecar = json.loads((models_dir / f"{prefix}_seed{SEEDS[0]}_fold0.json").read_text(encoding="utf-8"))
            params, seq = sidecar.get("trainable_without_embedding"), sidecar.get("seq_len_l")
        else:
            params, seq = "not a neural network", "n/a (TF-IDF)"
        config.append({"model": name, "role": role, "reads": reads, "layers": layers, "trainable_parameters_without_embedding": params,
                       "message_length": seq, "link_rf_trees": LINK_TREES if "RF" in reads else "none",
                       "recipe": RECIPE if prefix else "scikit-learn defaults (as the paper), fitted on the training part, not tuned",
                       "folds": "Stratified Group 5-fold x seeds 42, 123, 2024 (Step 3)",
                       "config_record_per_run": f"models/film/{dataset}/{prefix}_seed<S>_fold<F>.json" if prefix else "inside the .joblib"})
    out = exp / "step4" / "Training_Records.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    guide = pd.DataFrame({"guide item": [
        "Training and validation performance", "Number of training and validation messages", "Best epoch and early stopping information",
        "Model configuration, random seed, and fold", "Training completion or failure", "Saved model checkpoints",
        "File: Training Logs", "File: Model Checkpoints", "File: Model Configuration Records"], "where": [
        "Runs: train_* and val_* columns; Summary: means", "Runs: n_train, n_val", "Runs: best_epoch, epochs_run, stopped_early, patience",
        "Configuration sheet; Runs: seed, fold", "Runs: status; Summary: runs_completed / runs_failed",
        "Runs: checkpoint, checkpoint_exists, rebuild_check", f"Training_Log sheet; experiments/film/{dataset}/<model>/train_log.csv",
        f"models/film/{dataset}/ (.pt + .json for neural models, .joblib for KNN-RF-ETC)",
        f"Configuration sheet; the .json sidecar next to every .pt"]})
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        guide.to_excel(w, sheet_name="Guide_Checklist", index=False)
        pd.DataFrame(summary).to_excel(w, sheet_name="Summary", index=False)
        pd.concat(runs, ignore_index=True).to_excel(w, sheet_name="Runs", index=False)
        pd.DataFrame(config).to_excel(w, sheet_name="Configuration", index=False)
        pd.concat(logs, ignore_index=True).to_excel(w, sheet_name="Training_Log", index=False)
        pd.concat(details, ignore_index=True).to_excel(w, sheet_name="Model_Details", index=False)
    s = pd.DataFrame(summary)
    pd.set_option("display.width", 220)
    print(s[["model", "runs_completed", "of", "train_f1_mean", "val_f1_mean", "val_f1_sd", "best_epoch_mean", "runs_stopped_early"]]
          .round(4).to_string(index=False))
    print(f"wrote {out}")
    main_models = s[s.role == "Step 4 main model"]
    return 0 if (main_models.runs_completed == main_models["of"]).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
