"""Does static gating help at all? A check on each side separately, with no fresh data (docs/progress.md).

A small head (Dense 64 -> ReLU -> Dropout 0.5 -> 2, the architecture's head without the other branch) is trained on
one side's features, with the features multiplied by a gate vector before the head:

  text side   64 frozen-CNN features, v2: train on v2_train, early stop on v2_val, score v2_test.
              ungated (gates 1.0)  |  gated (the CNN gates)  |  shuffled (the same gate values on the wrong features)
  link side   37 forest contributions, web: out-of-fold contributions for web_train (a row is never scored by a forest
              that fit it), the final forest for web_test. Complete pages only.
              ungated | gated_first (min-max, F1, one slice: the first fit) | gated_fix (5-fold log-loss, rank, noise
              floor) | shuffled (the gated_fix values on the wrong features)
              Two input modes: `contrib` (the 37 contributions alone) and `contrib_pu` (plus P and u, ungated, as in
              the architecture). If P alone carries nearly everything, contrib_pu hides any gating effect; contrib
              isolates it.

Features are z-scored with the training mean and std BEFORE the gates, so the gates change the relative scale
(rescaling after the gates would cancel them). Shuffled controls are repeated 5 times per seed and averaged: if the
real gates beat that average, the importance information is what helps, not just the rescaling.

Same recipe everywhere: class weights, label smoothing from src.config, Adam 1e-3, up to 30 epochs, patience 3 on
validation balanced accuracy. Writes experiments/static_gating/gating_check_{per_seed,summary}.csv.

Usage:
    python -m src.unified.gating_check [--sides text link] [--seeds 42 123 2024]
"""
import argparse
import copy

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch import nn

from src.config import DENSE_UNITS, LABEL_SMOOTHING, NUM_FILTERS, SEEDS
from src.data.split_dataset import split_groups
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_data import class_weights, load_embedding_layer
from src.features.model.rf_extractor import fit_rf, tree_probabilities
from src.features.model.rf_feature_extractor import path_contributions
from src.features.model.run_experiment import score_group
from src.unified import unified_rf as urf
from src.unified.train_static_gating import (
    EMB_V2, FEATURE_SETS, MODEL_DIR, OUT_DIR, V2, V2_SPLITS, WEB_SPLITS, blank, load_web, part, tokens,
)

FS = FEATURE_SETS["37"]
N_SHUFFLES = 5
TUNED = ("n_estimators", "max_depth", "max_features", "min_samples_leaf")
METRICS = ["accuracy", "precision", "recall", "f1"]


# ------------------------------------------------------------------ head
class Head(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, DENSE_UNITS), nn.ReLU(), nn.Dropout(0.5), nn.Linear(DENSE_UNITS, 2))

    def forward(self, x):
        return self.net(x)


def fit_head(x_tr, y_tr, x_val, y_val, seed: int, epochs: int = 30, patience: int = 3, lr: float = 1e-3) -> Head:
    torch.manual_seed(seed)
    model = Head(x_tr.shape[1])
    weights = class_weights(torch.tensor(y_tr))
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt, yt = torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr)
    xv = torch.tensor(x_val, dtype=torch.float32)
    best, best_state, bad = -1.0, None, 0
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt), generator=gen)
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            score = balanced_accuracy_score(y_val, model(xv).argmax(dim=1).numpy())
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def predict(model: Head, x) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).argmax(dim=1).numpy()


def run_variants(Xtr, ytr, Xval, yval, Xte, yte, gate_sets: dict, seed: int, side: str, mode: str, n_gated: int) -> list[dict]:
    """Train one head per gate vector. The first n_gated columns are gated; any further columns pass through at 1.0."""
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    z = lambda X: (X - mean) / std
    rows = []
    for name, gates in gate_sets.items():
        g = np.ones(Xtr.shape[1], dtype=np.float32)
        g[:n_gated] = gates
        model = fit_head(z(Xtr) * g, ytr, z(Xval) * g, yval, seed)
        rows.append({"side": side, "mode": mode, "variant": name, "seed": seed, **score_group(yte, predict(model, z(Xte) * g))})
    return rows


def shuffled(gates: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed * 1000)
    return {f"shuffled_{j}": rng.permutation(gates) for j in range(N_SHUFFLES)}


# ------------------------------------------------------------------ text side
def text_side(seed: int) -> list[dict]:
    v2 = pd.read_csv(V2, keep_default_na=False)
    train, val, test = (part(v2, V2_SPLITS, seed, n) for n in ("train", "val", "test"))
    embedding, vocab = load_embedding_layer(EMB_V2)
    cnn = TextEncoder(embedding, NUM_FILTERS)
    cnn.load_state_dict(torch.load(MODEL_DIR / f"phase_a_seed{seed}.pt", map_location="cpu"))
    cnn.eval()

    def feats(df):
        ids, y = tokens(df, vocab)
        with torch.no_grad():
            return cnn.encode(ids).numpy(), y.numpy()

    (Xtr, ytr), (Xval, yval), (Xte, yte) = feats(train), feats(val), feats(test)
    gates = np.load(MODEL_DIR / f"cnn_gates_seed{seed}.npy")
    sets = {"ungated": np.ones_like(gates), "gated": gates, **shuffled(gates, seed)}
    print(f"[seed {seed}] text side: {len(Xtr)} train / {len(Xval)} val / {len(Xte)} test, 64 features", flush=True)
    return run_variants(Xtr, ytr, Xval, yval, Xte, yte, sets, seed, "text", "cnn64", n_gated=64)


# ------------------------------------------------------------------ link side
def outputs(rf, rows: pd.DataFrame) -> np.ndarray:
    """37 contributions, then P and u, for complete pages."""
    X = rows[FS.inputs].to_numpy(dtype=np.float32)
    contributions, _ = path_contributions(rf, X)
    per_tree = tree_probabilities(rf, rows[FS.inputs])
    return np.column_stack([contributions, per_tree.mean(axis=0), per_tree.std(axis=0)])


def link_side(seed: int) -> list[dict]:
    web = load_web(FS)
    train, test = (part(web, WEB_SPLITS, seed, n) for n in ("train", "test"))
    train = train[urf.link_mask(train)].reset_index(drop=True)
    test = test[urf.link_mask(test)].reset_index(drop=True)
    bundle = joblib.load(MODEL_DIR / f"forest_seed{seed}{FS.suffix}.joblib")
    params = {k: v for k, v in bundle["model"].get_params().items() if k in TUNED}

    # Out-of-fold outputs for the training rows (grouped by domain), the final forest for the test rows.
    ytr_all, groups = urf.labels(train), split_groups(train).to_numpy()
    oof = np.zeros((len(train), len(FS.inputs) + 2))
    for fit_idx, val_idx in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(train, ytr_all, groups):
        fit = train.iloc[fit_idx]
        aug = pd.concat([fit, blank(fit, FS.page)], ignore_index=True)
        rf = fit_rf(aug[FS.inputs], urf.labels(aug), seed, **params)
        oof[val_idx] = outputs(rf, train.iloc[val_idx])
    Xte, yte = outputs(bundle["model"], test), urf.labels(test)

    tr_idx, val_idx = train_test_split(np.arange(len(train)), test_size=0.15, stratify=ytr_all, random_state=seed)
    Xtr, ytr, Xval, yval = oof[tr_idx], ytr_all[tr_idx], oof[val_idx], ytr_all[val_idx]
    first = np.load(MODEL_DIR / f"rf_gates_seed{seed}{FS.suffix}.npy")
    fix = np.load(MODEL_DIR / f"rf_gates_seed{seed}{FS.suffix}_fix.npy")
    sets = {"ungated": np.ones_like(fix), "gated_first": first, "gated_fix": fix, **shuffled(fix, seed)}
    print(f"[seed {seed}] link side: {len(Xtr)} train / {len(Xval)} val / {len(Xte)} test, 37 contributions + P, u", flush=True)

    rows = []
    n = len(FS.inputs)
    rows += run_variants(Xtr, ytr, Xval, yval, Xte, yte, sets, seed, "link", "contrib_pu", n_gated=n)          # P, u ungated
    rows += run_variants(Xtr[:, :n], ytr, Xval[:, :n], yval, Xte[:, :n], yte, sets, seed, "link", "contrib", n_gated=n)
    return rows


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    df = per_seed.copy()
    df["group"] = np.where(df["variant"].str.startswith("shuffled"), "shuffled (mean of 5)", df["variant"])
    inner = df.groupby(["side", "mode", "group", "seed"])[METRICS].mean().reset_index()      # average the 5 shuffles inside a seed
    summary = inner.groupby(["side", "mode", "group"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    return summary.reset_index(), inner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sides", nargs="+", default=["text", "link"], choices=["text", "link"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        if "text" in args.sides:
            rows += text_side(seed)
        if "link" in args.sides:
            rows += link_side(seed)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT_DIR / "gating_check_per_seed.csv", index=False)
    summary, inner = summarise(per_seed)
    summary.to_csv(OUT_DIR / "gating_check_summary.csv", index=False)
    inner.to_csv(OUT_DIR / "gating_check_by_seed.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    # paired differences per seed against ungated, so the spread is visible
    base = inner[inner["group"] == "ungated"].set_index(["side", "mode", "seed"])[METRICS]
    diff = inner[inner["group"] != "ungated"].set_index(["side", "mode", "seed"])[METRICS] - base.reindex(
        inner[inner["group"] != "ungated"].set_index(["side", "mode", "seed"]).index).to_numpy()
    diff = diff.assign(group=inner[inner["group"] != "ungated"]["group"].to_numpy()).reset_index()
    print("\nF1 difference from ungated, per seed:")
    print(diff.pivot_table(index=["side", "mode", "group"], columns="seed", values="f1").round(4).to_string())


if __name__ == "__main__":
    main()
