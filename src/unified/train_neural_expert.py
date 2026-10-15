"""Neural link expert: the 1D feature CNN trained on the SAME web set, splits and recipe as the forest.

Why (docs/progress.md, section 2h): a fair neural baseline for the forest must have the same data, the same
tuning effort and the same protocol. The forest is trained on the web train rows (every page complete and with the
17 page/outside columns blank) with a grid chosen by grouped CV. This script gives the 1D CNN (FeatureCNN from
run_features_only.py) exactly that: web train rows, complete + blanked, domain-grouped validation, a grid of 8
settings (filters 16/32, dropout 0.3/0.5, signed log1p off/on), best by validation F1 (ties: lower log-loss).
Blanked pages: NaN -> 0 (the mean after z-score) plus a page_missing flag. Scored on web_test complete and blanked,
the same two conditions as the forest (forest 37 features: F1 0.911 complete, 0.825 blanked).

Per seed it saves models/static_gating/nn_link_seed{seed}.pt (weights + settings + scaler) and writes
experiments/static_gating/nn_expert_{per_seed,summary,tuning}.csv. The fused pipeline (run_fused_lean.py) loads it.

Usage:
    python -m src.unified.train_neural_expert [--seeds 42 123 2024]
"""
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from itertools import product
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SEEDS
from src.data.split_dataset import split_groups
from src.features.model.run_experiment import score_group
from src.unified import unified_rf as urf
from src.unified.run_features_only import FeatureCNN, fit_cnn
from src.unified.train_static_gating import FEATURE_SETS, MODEL_DIR, OUT_DIR, WEB_SPLITS, blank, load_web, part

FS = FEATURE_SETS["37"]
GRID = [{"filters": f, "dropout": d, "log1p": l} for f, d, l in product([16, 32], [0.3, 0.5], [False, True])]
TRAIN_ARGS = SimpleNamespace(lr=1e-3, batch_size=64, epochs=100, patience=8)
METRICS = ["accuracy", "precision", "recall", "f1"]


def transform(x: np.ndarray, log1p: bool) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x)) if log1p else x


class Scaler:
    """Signed log1p (optional), z-score with training statistics, NaN -> 0, plus a page_missing flag."""

    def __init__(self, df_train: pd.DataFrame, log1p: bool):
        self.log1p = log1p
        x = transform(df_train[FS.inputs].to_numpy(dtype=np.float64), log1p)
        self.mean, self.std = np.nanmean(x, axis=0), np.nanstd(x, axis=0) + 1e-8
        self.page_idx = [FS.inputs.index(c) for c in FS.page]

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        x = transform(df[FS.inputs].to_numpy(dtype=np.float64), self.log1p)
        missing = np.isnan(x[:, self.page_idx]).any(axis=1).astype(np.float32)
        z = np.nan_to_num((x - self.mean) / self.std, nan=0.0)
        return np.column_stack([z, missing]).astype(np.float32)

    def state(self) -> dict:
        return {"log1p": self.log1p, "mean": self.mean, "std": self.std}


def predict_proba(model: FeatureCNN, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def load_expert(seed: int) -> tuple[FeatureCNN, Scaler]:
    """Rebuild a saved expert: (model, scaler). The scaler is rebuilt from the saved statistics."""
    ck = torch.load(MODEL_DIR / f"nn_link_seed{seed}.pt", map_location="cpu", weights_only=False)
    model = FeatureCNN(ck["n_inputs"], filters=ck["params"]["filters"], dropout=ck["params"]["dropout"])
    model.load_state_dict(ck["state"])
    scaler = Scaler.__new__(Scaler)
    scaler.log1p, scaler.mean, scaler.std = ck["scaler"]["log1p"], ck["scaler"]["mean"], ck["scaler"]["std"]
    scaler.page_idx = [FS.inputs.index(c) for c in FS.page]
    return model.eval(), scaler


def run_seed(seed: int) -> tuple[list, list]:
    web = load_web(FS)
    train, test = (part(web, WEB_SPLITS, seed, n) for n in ("train", "test"))
    train = train[urf.link_mask(train)].reset_index(drop=True)
    test = test[urf.link_mask(test)].reset_index(drop=True)
    y, groups = urf.labels(train), split_groups(train).to_numpy()
    fit_idx, val_idx = next(iter(StratifiedGroupKFold(6, shuffle=True, random_state=seed).split(train, y, groups)))
    fit_rows, val_rows = train.iloc[fit_idx], train.iloc[val_idx]
    fit_aug = pd.concat([fit_rows, blank(fit_rows, FS.page)], ignore_index=True)      # every page twice, as for the forest
    val_aug = pd.concat([val_rows, blank(val_rows, FS.page)], ignore_index=True)
    y_fit, y_val = urf.labels(fit_aug), urf.labels(val_aug)

    tuning, best = [], None
    for params in GRID:
        scaler = Scaler(fit_aug, params["log1p"])
        model = fit_cnn(scaler(fit_aug), y_fit, scaler(val_aug), y_val, seed, TRAIN_ARGS,
                        filters=params["filters"], dropout=params["dropout"])
        p = predict_proba(model, scaler(val_aug))
        row = {"seed": seed, **{k: str(v) for k, v in params.items()},
               "val_f1": float(f1_score(y_val, p >= 0.5)), "val_log_loss": float(log_loss(y_val, p, labels=[0, 1]))}
        tuning.append(row)
        key = (-row["val_f1"], row["val_log_loss"])
        if best is None or key < best[0]:
            best = (key, params, model, scaler)
        print(f"[seed {seed}] {params} val F1 {row['val_f1']:.4f}", flush=True)

    _, params, model, scaler = best
    torch.save({"state": model.state_dict(), "params": params, "scaler": scaler.state(),
                "n_inputs": len(FS.inputs) + 1}, MODEL_DIR / f"nn_link_seed{seed}.pt")
    y_test = urf.labels(test)
    scores = []
    for cond, frame in (("complete", test), ("blanked", blank(test, FS.page))):
        pred = (predict_proba(model, scaler(frame)) >= 0.5).astype(int)
        scores.append({"seed": seed, "group": f"web_test_{cond}", "model": "1D CNN (37, trained like the forest)",
                       **score_group(y_test, pred)})
    print(f"[seed {seed}] chosen {params}: " + ", ".join(f"{r['group']} F1 {r['f1']:.3f}" for r in scores), flush=True)
    return scores, tuning


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores, tuning = [], []
    for seed in args.seeds:
        s, t = run_seed(seed)
        scores += s
        tuning += t
    per_seed = pd.DataFrame(scores)
    per_seed.to_csv(OUT_DIR / "nn_expert_per_seed.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT_DIR / "nn_expert_tuning.csv", index=False)
    summary = per_seed.groupby("group")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.reset_index().to_csv(OUT_DIR / "nn_expert_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
