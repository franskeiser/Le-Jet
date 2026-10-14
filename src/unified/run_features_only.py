"""Features-only comparison: Random Forest vs a 1D CNN on the per-link features, no SMS text at all.

Why: the CNN on text is confident because of the words, so it cannot show which learner reads the
link evidence better. Here both models see only the link's features and nothing else:
    url_only  the 14 URL text features (valid on any link, no page needed)
    full      the 31 features (14 URL + 17 page/domain; a page that was not reached is NaN)

    rf   forest tuned by grouped CV on the Training rows (urf.tune), then fit on all Training rows
         (NaN is read as is, scikit-learn forests accept it);
    cnn  1D CNN over the standardized feature vector (length = number of features, one channel).
         NaN -> 0 (the mean) with a page_missing flag in the full variant, as in run_baseline.py.
         Early stopping on the same validation slice every other run carves from Training.

Same splits and seeds as the other runs; only rows with a link are used. Writes
<out>/features_only/{metrics_per_seed,metrics_summary}.csv and predictions.csv. Nothing else is read or changed.

Usage:
    python -m src.unified.run_features_only [--data ...] [--splits-dir ...] [--out ...] [--variants url_only full]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import SEEDS
from src.features.model.rf_extractor import fit_rf
from src.features.model.run_experiment import score_group, seed_everything
from src.unified import unified_rf as urf
from src.unified.run_unified import VAL_FRACTION_OF_TRAIN, load_part

DEFAULT_DATA = Path("data/processed/unified_every_sms_linked.csv")
DEFAULT_SPLITS = Path("data/splits_every_sms")
DEFAULT_OUT = Path("experiments/every_sms")


class FeatureCNN(nn.Module):
    """Two conv layers over the feature vector, global max pool, one dense layer, one logit."""

    def __init__(self, n_inputs: int, filters: int = 32, dense_units: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(filters, filters * 2, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(filters * 2 * 2, dense_units), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(dense_units, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))                                   # (batch, channels, n_inputs)
        h = torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)      # global max + mean pool
        return self.head(h).squeeze(1)


def standardize(train: pd.DataFrame, inputs: list[str]) -> tuple[pd.Series, pd.Series]:
    mean = train[inputs].mean()
    std = train[inputs].std().replace(0, 1.0).fillna(1.0)
    return mean, std


def cnn_matrix(df: pd.DataFrame, inputs: list[str], scaler, page_cols: list[str]) -> np.ndarray:
    mean, std = scaler
    x = ((df[inputs] - mean) / std).fillna(0.0)
    if page_cols:
        x["page_missing"] = df[page_cols].isna().any(axis=1).astype(float)
    return x.to_numpy(dtype=np.float32)


def fit_cnn(x_tr, y_tr, x_val, y_val, seed: int, args) -> FeatureCNN:
    seed_everything(seed)
    model = FeatureCNN(x_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr, dtype=torch.float32)),
                        batch_size=args.batch_size, shuffle=True)
    xv, yv = torch.tensor(x_val), torch.tensor(y_val, dtype=torch.float32)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        if val_loss < best - 1e-4:
            best, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def evaluate(test_df: pd.DataFrame, y: np.ndarray, pred: np.ndarray, model: str, variant: str, seed: int) -> list[dict]:
    groups = {"all": np.ones(len(y), dtype=bool)}
    if "pair_type" in test_df:
        for kind in test_df["pair_type"].dropna().unique():
            groups[str(kind)] = (test_df["pair_type"] == kind).to_numpy()
    return [{"seed": seed, "variant": variant, "model": model, "group": g, **score_group(y[m], pred[m])}
            for g, m in groups.items()]


def run_seed(df: pd.DataFrame, seed: int, variant: str, args) -> tuple[list[dict], pd.DataFrame, dict]:
    inputs = urf.INPUTS[variant]
    page_cols = [c for c in inputs if c not in urf.INPUTS["url_only"]]
    train_full = load_part(df, seed, "train", args.splits_dir)
    test_df = load_part(df, seed, "test", args.splits_dir)
    train_full = train_full[urf.link_mask(train_full)].reset_index(drop=True)
    test_df = test_df[urf.link_mask(test_df)].reset_index(drop=True)
    y_test = urf.labels(test_df)

    params, _ = urf.tune(train_full, inputs, seed)
    rf = fit_rf(train_full[inputs], urf.labels(train_full), seed, **params)
    rf_pred = rf.predict(test_df[inputs])

    tr_pos, val_pos = train_test_split(np.arange(len(train_full)), test_size=VAL_FRACTION_OF_TRAIN,
                                       stratify=train_full["label"], random_state=seed)
    tr, val = train_full.iloc[tr_pos], train_full.iloc[val_pos]
    scaler = standardize(tr, inputs)
    model = fit_cnn(cnn_matrix(tr, inputs, scaler, page_cols), urf.labels(tr),
                    cnn_matrix(val, inputs, scaler, page_cols), urf.labels(val), seed, args)
    with torch.no_grad():
        cnn_pred = (torch.sigmoid(model(torch.tensor(cnn_matrix(test_df, inputs, scaler, page_cols)))) >= 0.5).numpy().astype(int)

    rows = (evaluate(test_df, y_test, rf_pred, "rf", variant, seed)
            + evaluate(test_df, y_test, cnn_pred, "cnn", variant, seed))
    preds = pd.DataFrame({"seed": seed, "variant": variant, "message_id": test_df["message_id"], "y": y_test,
                          "rf": rf_pred, "cnn": cnn_pred})
    return rows, preds, {"seed": seed, "variant": variant, "rf_params": {k: str(v) for k, v in params.items()},
                         "n_train": len(train_full), "n_test": len(test_df)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--variants", nargs="+", default=["url_only", "full"], choices=list(urf.INPUTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(args.data, low_memory=False, keep_default_na=False, na_values=[""])
    rows, preds, infos = [], [], []
    for variant in args.variants:
        for seed in args.seeds:
            r, p, info = run_seed(df, seed, variant, args)
            rows += r
            preds.append(p)
            infos.append(info)
            print(f"done {variant} seed {seed}")

    out = args.out / "features_only"
    out.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "metrics_per_seed.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions.csv", index=False)
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    summary = per_seed.groupby(["variant", "group", "model"])[metrics + ["n"]].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "metrics_summary.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "runs": infos}, indent=2))

    pd.set_option("display.width", 220)
    print(summary[["variant", "group", "model", "n_mean", "balanced_accuracy_mean", "balanced_accuracy_std",
                   "f1_mean", "f1_std"]].round(4).to_string(index=False))
    print("NOTE: page features are the hybrid_temp placeholder. Only url_only is meaningful now.")


if __name__ == "__main__":
    main()
