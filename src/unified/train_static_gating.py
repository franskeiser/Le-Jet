"""Train the parts of the static-gating architecture that need no fresh data (docs/progress.md, section 4).

Per seed, four stages:
    cnn       Phase A text CNN on v2_train (link text removed), early stopping on v2_val, scored on v2_test.
    forest    ONE forest on web_train, every page seen twice: complete and with the page/domain columns blank.
              Tuned by grouped CV. Scored on web_test complete and blanked. Calibrated (isotonic and Platt,
              fit on out-of-fold probabilities); ECE reported before and after.
    rfgates   Forest gates: permutation importance (F1) of the forest on the web gate slice, rescaled to 0.1-3.0.
    cnngates  CNN gates: a forest on the frozen CNN's 64 features of the v2 gate slice, permutation importance
              (balanced accuracy) on its held-out 30%, rescaled to 0.1-3.0.
Both gate slices are held out from the model they score. The gates are saved as fixed numbers.

--features 31   the first feature set (14 URL + 17 page/domain), data/processed/web_phishing_linked.csv
--features 37   the final feature set (20 URL + 17 page/domain), data/processed/web_phishing_37_linked.csv
                (python -m src.unified.prepare_web_37). Same message ids and splits as the 31 run.
Files of the 37 run carry the suffix _f37, so both runs sit side by side.

Nothing here reads fresh data, so nothing here is the study result.

Writes models/static_gating/ and experiments/static_gating/.

Usage:
    python -m src.unified.train_static_gating --features 37 --stages forest rfgates
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader

from src.config import ARTIFACTS_ROOT, NUM_FILTERS, SEEDS, SEQ_LEN_L
from src.data.split_dataset import load_split_ids, split_groups
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_data import class_weights, load_embedding_layer
from src.features.model.rf_extractor import expected_calibration_error, fit_rf
from src.features.model.rf_gating_module import (
    PERM_IMPORTANCE_HOLDOUT_FRACTION, extract_cnn_features, fit_and_score_group, gate_transform,
)
from src.features.model.run_experiment import predict_labels, score_group, seed_everything
from src.features.model.train import LABEL_TO_IDX, SmishDataset, encode_batch, train_phase_a
from src.unified import prepare_web_37 as w37
from src.unified import unified_rf as urf

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data" / "processed" / "v2_text.csv"
V2_SPLITS = ROOT / "data" / "splits_v2"
EMB_V2 = ROOT / "models" / "embeddings_v2"
WEB_SPLITS = ROOT / "data" / "splits_web_phishing"
MODEL_DIR = ARTIFACTS_ROOT / "models" / "static_gating"
OUT_DIR = ARTIFACTS_ROOT / "experiments" / "static_gating"
METRICS = ["accuracy", "precision", "recall", "f1"]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    web_path: Path
    inputs: list
    page: list          # blanked when the page is not reached
    suffix: str


_FULL31 = urf.INPUTS["full"]
FEATURE_SETS = {
    "31": FeatureSet("31", ROOT / "data" / "processed" / "web_phishing_linked.csv", _FULL31,
                     [c for c in _FULL31 if c not in urf.INPUTS["url_only"]], ""),
    "37": FeatureSet("37", ROOT / "data" / "processed" / "web_phishing_37_linked.csv", w37.FEATURES_37,
                     w37.PAGE_FEATURES, "_f37"),
}


def part(df: pd.DataFrame, splits_dir: Path, seed: int, name: str) -> pd.DataFrame:
    ids = load_split_ids(seed, name, str(splits_dir))
    return df[df["message_id"].isin(ids)].reset_index(drop=True)


def blank(df: pd.DataFrame, page: list) -> pd.DataFrame:
    out = df.copy()
    out[page] = np.nan
    return out


def load_web(fs: FeatureSet) -> pd.DataFrame:
    return pd.read_csv(fs.web_path, low_memory=False, keep_default_na=False, na_values=[""])


# ------------------------------------------------------------------ text CNN
def tokens(df: pd.DataFrame, vocab: dict) -> tuple[torch.Tensor, torch.Tensor]:
    ids = encode_batch(df["sms_text"].astype(str).tolist(), vocab, SEQ_LEN_L)
    y = torch.tensor([LABEL_TO_IDX[label] for label in df["label"]], dtype=torch.long)
    return ids, y


def stage_cnn(seed: int, args) -> list[dict]:
    seed_everything(seed)
    df = pd.read_csv(V2, keep_default_na=False)
    train, val, test = (part(df, V2_SPLITS, seed, n) for n in ("train", "val", "test"))
    embedding, vocab = load_embedding_layer(EMB_V2)
    train_ids, train_y = tokens(train, vocab)
    val_ids, val_y = tokens(val, vocab)
    test_ids, test_y = tokens(test, vocab)
    weights = class_weights(train_y)
    cnn = TextEncoder(embedding, NUM_FILTERS)
    path = MODEL_DIR / f"phase_a_seed{seed}.pt"
    train_phase_a(
        cnn, DataLoader(SmishDataset(train_ids, train_y), batch_size=args.batch_size, shuffle=True),
        DataLoader(SmishDataset(val_ids, val_y), batch_size=512), epochs=args.epochs, lr=args.lr,
        patience=args.patience, checkpoint_path=str(path), class_weights=weights,
    )
    cnn.load_state_dict(torch.load(path, map_location="cpu"))
    pred = predict_labels(cnn, test_ids, None)
    link = (test["link_status"] == "link").to_numpy()
    rows = [{"seed": seed, "stage": "cnn", "group": "v2_test", **score_group(test_y.numpy(), pred)},
            {"seed": seed, "stage": "cnn", "group": "v2_test_with_link", **score_group(test_y.numpy()[link], pred[link])},
            {"seed": seed, "stage": "cnn", "group": "v2_test_no_link", **score_group(test_y.numpy()[~link], pred[~link])}]
    print(f"[seed {seed}] cnn: train {len(train)}, val {len(val)}, test {len(test)}")
    return rows


# ------------------------------------------------------------------ forest
def oof_probability(rows: pd.DataFrame, inputs: list, seed: int, params: dict, n_splits: int = 5) -> np.ndarray:
    y, groups = urf.labels(rows), split_groups(rows).to_numpy()
    p = np.zeros(len(rows))
    for fit_idx, val_idx in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(rows, y, groups):
        rf = fit_rf(rows.iloc[fit_idx][inputs], y[fit_idx], seed, **params)
        p[val_idx] = rf.predict_proba(rows.iloc[val_idx][inputs])[:, 1]
    return p


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stage_forest(seed: int, args) -> list[dict]:
    fs = FEATURE_SETS[args.features]
    web = load_web(fs)
    train, test = (part(web, WEB_SPLITS, seed, n) for n in ("train", "test"))
    train = train[urf.link_mask(train)].reset_index(drop=True)
    test = test[urf.link_mask(test)].reset_index(drop=True)
    aug = pd.concat([train, blank(train, fs.page)], ignore_index=True)     # duplicates keep their domain: CV stays grouped

    params, grid = urf.tune(aug, fs.inputs, seed)
    grid.assign(seed=seed).to_csv(OUT_DIR / f"forest_tuning_seed{seed}{fs.suffix}.csv", index=False)
    bundle = urf.fit(aug, fs.inputs, seed, params)
    joblib.dump(bundle, MODEL_DIR / f"forest_seed{seed}{fs.suffix}.joblib")

    # Calibrators: fit on out-of-fold probabilities, so the calibrator never sees a forest that fit the row.
    oof = oof_probability(aug, fs.inputs, seed, params)
    y_aug = urf.labels(aug)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y_aug)
    platt = LogisticRegression(C=1e6).fit(logit(oof).reshape(-1, 1), y_aug)
    joblib.dump({"isotonic": iso, "platt": platt}, MODEL_DIR / f"calibrators_web_seed{seed}{fs.suffix}.joblib")

    rows = []
    y_test = urf.labels(test)
    for cond, frame in (("complete", test), ("blanked", blank(test, fs.page))):
        p = bundle["model"].predict_proba(frame[fs.inputs])[:, 1]
        variants = {"raw": p, "isotonic": iso.predict(p), "platt": platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]}
        for name, q in variants.items():
            rows.append({"seed": seed, "stage": "forest", "features": fs.name, "group": f"web_test_{cond}", "calibration": name,
                         **score_group(y_test, (q >= 0.5).astype(int)),
                         "ece": expected_calibration_error(y_test, q), "brier": float(brier_score_loss(y_test, q))})
    print(f"[seed {seed}] forest ({fs.name} features): params {params}, train rows {len(aug)} (complete + blanked), test {len(test)}")
    return rows


# ------------------------------------------------------------------ static gates
def stage_rfgates(seed: int, args) -> list[dict]:
    fs = FEATURE_SETS[args.features]
    gate_web = part(load_web(fs), WEB_SPLITS, seed, "rf_fit")
    gate_web = gate_web[urf.link_mask(gate_web)]
    bundle = joblib.load(MODEL_DIR / f"forest_seed{seed}{fs.suffix}.joblib")
    result = permutation_importance(bundle["model"], gate_web[bundle["features"]], urf.labels(gate_web),
                                    scoring="f1", n_repeats=args.gate_repeats, random_state=seed, n_jobs=-1)
    importance = pd.Series(result.importances_mean, index=bundle["features"])
    gates = gate_transform(importance.to_numpy()).astype(np.float32)
    np.save(MODEL_DIR / f"rf_gates_seed{seed}{fs.suffix}.npy", gates)
    pd.DataFrame({"feature": importance.index, "importance": importance.to_numpy(),
                  "importance_std": result.importances_std, "gate": gates}).to_csv(
        OUT_DIR / f"rf_gates_seed{seed}{fs.suffix}.csv", index=False)
    print(f"[seed {seed}] forest gates ({fs.name}): {len(gates)} gates, range {gates.min():.2f}-{gates.max():.2f}, "
          f"{int((importance < 0).sum())} negative importances, {len(gate_web)} gate rows")
    return [{"seed": seed, "stage": "rfgates", "features": fs.name, "group": "rf_gate_slice", "n_gate_rows": len(gate_web),
             "n_negative_importance": int((importance < 0).sum())}]


def stage_cnngates(seed: int, args) -> list[dict]:
    v2 = pd.read_csv(V2, keep_default_na=False)
    gate_v2 = part(v2, V2_SPLITS, seed, "gate")
    embedding, vocab = load_embedding_layer(EMB_V2)
    cnn = TextEncoder(embedding, NUM_FILTERS)
    cnn.load_state_dict(torch.load(MODEL_DIR / f"phase_a_seed{seed}.pt", map_location="cpu"))
    ids, y_t = tokens(gate_v2, vocab)
    X, y = extract_cnn_features(cnn, ids), y_t.numpy()
    idx_fit, idx_score = train_test_split(np.arange(len(y)), test_size=PERM_IMPORTANCE_HOLDOUT_FRACTION,
                                          stratify=y, random_state=seed)
    result = fit_and_score_group(X, y, idx_fit, idx_score, seed, "balanced_accuracy", args.gate_repeats, class_weight="balanced")
    np.save(MODEL_DIR / f"cnn_gates_seed{seed}.npy", result["gates"].astype(np.float32))
    pd.DataFrame({"feature": [f"cnn_{i}" for i in range(X.shape[1])], "importance": result["perm_mean"],
                  "gini": result["gini"], "gate": result["gates"]}).to_csv(OUT_DIR / f"cnn_gates_seed{seed}.csv", index=False)
    print(f"[seed {seed}] CNN gates: {X.shape[1]} (range {result['gates'].min():.2f}-{result['gates'].max():.2f}, "
          f"held-out {result['held_out_score']:.3f} vs baseline {result['baseline_score']:.3f}, gini fallback {result['used_gini_fallback']})")
    return [{"seed": seed, "stage": "cnngates", "group": "cnn_gate_forest", "held_out_balanced_accuracy": result["held_out_score"],
             "baseline_balanced_accuracy": result["baseline_score"], "used_gini_fallback": bool(result["used_gini_fallback"]),
             "n_fit": len(idx_fit), "n_score": len(idx_score)}]


STAGES = {"cnn": stage_cnn, "forest": stage_forest, "rfgates": stage_rfgates, "cnngates": stage_cnngates}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="31")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gate-repeats", type=int, default=30)
    args = ap.parse_args()

    suffix = FEATURE_SETS[args.features].suffix
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stage in args.stages:
        rows = []
        for seed in args.seeds:
            rows += STAGES[stage](seed, args)
        per_seed = pd.DataFrame(rows)
        tag = suffix if stage in ("forest", "rfgates") else ""
        per_seed.to_csv(OUT_DIR / f"{stage}_per_seed{tag}.csv", index=False)
        keys = [c for c in ("features", "group", "calibration") if c in per_seed]
        value_cols = [c for c in per_seed.columns if c in METRICS + ["n", "ece", "brier", "held_out_balanced_accuracy",
                                                                     "baseline_balanced_accuracy", "n_negative_importance"]]
        summary = per_seed.groupby(keys)[value_cols].agg(["mean", "std"])
        summary.columns = [f"{c}_{s}" for c, s in summary.columns]
        summary.reset_index().to_csv(OUT_DIR / f"{stage}_summary{tag}.csv", index=False)
        pd.set_option("display.width", 220)
        print(f"\n== {stage}")
        print(summary.round(4).to_string())
    (OUT_DIR / f"run_config{suffix}.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}}, indent=2))


if __name__ == "__main__":
    main()
