"""Train and evaluate the gated fusion model for every seed.

Per seed (docs/dataset_and_training_plan.md 7.2-7.3, docs/training_rules.md):
    Phase A  train the CNN alone (embedding + conv + temp head) on the Training split, with
             a validation slice carved from Training for early stopping.
    Gates    freeze that CNN, fit the two gating forests on the RF-Fitting split, scale the
             importances to 0.1-3.0. has_url, P and u keep a fixed gate of 1.0.
    Phase C  train the fusion model (CNN + 14 link features + has_url + P + u) with those gates.
    Test     score on the Test split. Nothing above ever reads it.

Headline scores use real messages only; synthetic rows are reported separately. Every group is
also split into with-link and without-link messages (training_rules.md rules 4 and 5).

Variants:
    with_pu  14 raw link features + has_url + P + u          (the first design)
    no_pu    same without P and u (experiment row 4, the "gating, no extractor" baseline)
    pu_only  has_url + P + u: link features only inside the RF
    rf_vec   has_url + P + u + 31 per-feature RF contributions (the RF as a feature extractor)

Optional (--live-train data/live/live_links.csv --tag _live): Phase C's head also trains on rows built from the
lab-inspected links of the `live_train` part (src/features/model/live_training.py), each shown to the head with or without
its page evidence (MixedAuxDataset). Phase A, the gates and every validation/test score stay on the real data only, and
`fresh_test` is never read here. The default run is unchanged.

Usage (from the project root):
    python -m src.features.model.run_experiment
    python -m src.features.model.run_experiment --seeds 42 --variant no_pu --epochs 3
    python -m src.features.model.run_experiment --variant rf_vec --live-train data/live/live_links.csv --tag _live
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.config import ARTIFACTS_ROOT, DENSE_UNITS, NUM_FILTERS, SEEDS, SEQ_LEN_L
from src.data.split_dataset import load_phase_a_train_val, load_split_df, resolve_label_column
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_classifier import FusionClassifier, load_phase_a_encoder_weights
from src.features.model.fusion_data import (
    AUX_COLUMNS, RECIPE_COLUMNS, ROOT, build_aux_frame, class_weights, load_embedding_layer, load_vocab, load_seed_inputs,
)
from src.features.model.rf_extractor import load_extractor
from src.features.model.rf_feature_extractor import extractor_gates
from src.features.model.rf_gating_module import run_one_seed
from src.features.model.train import (
    LABEL_TO_IDX, FusionDataset, MixedAuxDataset, SmishDataset, train_phase_a, train_phase_c,
)

REAL_SOURCE = "v2_2026-09-15"
VARIANT_COLUMNS = {"with_pu": AUX_COLUMNS, "no_pu": AUX_COLUMNS[:-2],  # no_pu drops P and u
                   "pu_only": RECIPE_COLUMNS["pu_only"], "rf_vec": RECIPE_COLUMNS["rf_vec"]}
VARIANT_RECIPE = {"with_pu": "link_pu", "no_pu": "link_pu", "pu_only": "pu_only", "rf_vec": "rf_vec"}
IMPORTANCE_CSV = ARTIFACTS_ROOT / "experiments" / "rf_extractor" / "importance_per_seed.csv"
DEFAULT_OUT = ARTIFACTS_ROOT / "experiments" / "fusion"
DEFAULT_MODEL_DIR = ARTIFACTS_ROOT / "models" / "fusion"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def columns_index(columns: list[str], recipe: str = "link_pu") -> list[int]:
    return [RECIPE_COLUMNS[recipe].index(c) for c in columns]


def recipe_url_gates(recipe: str, seed: int) -> np.ndarray | None:
    """Gates for the non-text inputs when they come from the RF extractor, else None (fit them on the
    RF-Fitting split as before). has_url, P and u are fixed at 1.0. For rf_vec the 31 per-feature slots
    are gated by the full forest's own permutation importances (validation part of the public web set,
    experiments/rf_extractor/importance_per_seed.csv), because the old SMS data has no page features to fit
    the page slots on."""
    if recipe == "pu_only":
        return np.ones(3, dtype=np.float32)
    if recipe == "rf_vec":
        table = pd.read_csv(IMPORTANCE_CSV)
        importance = table[(table["variant"] == "full") & (table["seed"] == seed)].set_index("feature")["perm_mean"]
        return np.concatenate([np.ones(3, dtype=np.float32), extractor_gates(importance)])
    return None


@torch.no_grad()
def predict_labels(model, token_ids: torch.Tensor, aux: torch.Tensor | None, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(token_ids), batch_size):
        ids = token_ids[start:start + batch_size]
        logits = model(ids) if aux is None else model(ids, aux[start:start + batch_size])
        preds.append(torch.argmax(logits, dim=1))
    return torch.cat(preds).numpy()


def score_group(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"n": 0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0,
    )
    return {
        "n": len(y_true), "smishing_share": round(float(y_true.mean()), 4),
        "accuracy": accuracy_score(y_true, y_pred),
        # balanced accuracy is undefined (and warns) when a group has only one class
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if len(set(y_true)) == 2 else np.nan,
        "precision": precision, "recall": recall, "f1": f1,
    }


def evaluate_groups(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, has_url: np.ndarray) -> list[dict]:
    real = (test_df["source"] == REAL_SOURCE).to_numpy()
    with_link = has_url == 1
    groups = {
        "real_all": real, "real_with_link": real & with_link, "real_no_link": real & ~with_link,
        "synthetic_all": ~real, "synthetic_with_link": ~real & with_link,
    }
    return [{"group": name, **score_group(y_true[mask], y_pred[mask])} for name, mask in groups.items()]


def live_head_set(live_path, args, seed, recipe, col_idx, train_ids, train_aux, train_y):
    """The real SMS rows (their only P and u come from url_only, as before) plus the live_train rows (each read with or
    without page evidence at random). Returns (dataset, all token ids, all labels)."""
    from src.features.model import live_training as lt
    rows = lt.build_rows(Path(live_path), "live_train", repeats=args.live_repeats, seed=seed)
    ids, base, alt, y = lt.build_tensors(rows, load_extractor("url_only", seed), load_extractor("full", seed),
                                         load_vocab(), recipe)
    base, alt = base[:, col_idx], alt[:, col_idx]
    print(f"[seed {seed}] live rows added to the head: {len(rows)} "
          f"({int(y.sum())} smishing, {int((1 - y).sum())} legitimate, {int((rows['page_fetched'] == 1).sum())} with a page)")
    all_ids, all_y = torch.cat([train_ids, ids]), torch.cat([train_y, y])
    all_base, all_alt = torch.cat([train_aux, base]), torch.cat([train_aux, alt])
    return MixedAuxDataset(all_ids, all_base, all_alt, all_y, p_alt=args.live_p_alt), all_ids, all_y


def run_seed(seed: int, variant: str, args) -> tuple[list[dict], dict]:
    seed_everything(seed)
    columns = VARIANT_COLUMNS[variant]
    recipe = VARIANT_RECIPE[variant]
    col_idx = columns_index(columns, recipe)
    tag = getattr(args, "tag", "") or ""
    model_dir = Path(args.model_dir) / (variant + tag)
    model_dir.mkdir(parents=True, exist_ok=True)
    word_to_idx = load_vocab()

    # ---- Phase A: CNN alone on the Training split ----------------------------------------
    train_df, val_df, label_col = load_phase_a_train_val(seed)
    train_ids, _, train_y = load_seed_inputs(seed, "train", train_df, recipe)
    val_ids, _, val_y = load_seed_inputs(seed, "train", val_df, recipe)
    weights = class_weights(train_y)
    print(f"[seed {seed}] train {len(train_y)} / val {len(val_y)}, class weights {weights.tolist()}")

    embedding, _ = load_embedding_layer()
    cnn = TextEncoder(embedding, NUM_FILTERS)
    phase_a_path = model_dir / f"phase_a_seed{seed}.pt"
    reuse = getattr(args, "reuse_from", None)
    if reuse:
        # Phase A and the gates of an earlier finished run: only Phase C is trained again, so any change in the
        # scores comes from the head alone. The CNN and the gates are copied untouched.
        for name in (f"phase_a_seed{seed}.pt", f"gates_seed{seed}.csv", f"gates_seed{seed}.npy"):
            shutil.copy2(Path(reuse) / name, model_dir / name)
        cnn.load_state_dict(torch.load(phase_a_path, map_location="cpu"))
        phase_a_metrics = {"reused_from": str(reuse)}
        gates = torch.from_numpy(np.load(model_dir / f"gates_seed{seed}.npy").astype(np.float32))
    else:
        phase_a_metrics = train_phase_a(
            cnn,
            DataLoader(SmishDataset(train_ids, train_y), batch_size=args.batch_size, shuffle=True),
            DataLoader(SmishDataset(val_ids, val_y), batch_size=512),
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            checkpoint_path=str(phase_a_path), class_weights=weights,
        )
        cnn.load_state_dict(torch.load(phase_a_path, map_location="cpu"))  # best epoch, not the last

        # ---- Gates: forests on the RF-Fitting split, frozen CNN --------------------------
        rf_df = load_split_df(seed, "rf_fit")
        _, rf_aux, _ = load_seed_inputs(seed, "rf_fit", rf_df, recipe)
        gate_result = run_one_seed(
            cnn, rf_df["sms_text"].astype(str).tolist(),
            pd.DataFrame(rf_aux[:, col_idx].numpy(), columns=columns).to_dict("records"),
            rf_df[label_col].tolist(), word_to_idx, SEQ_LEN_L, seed, n_repeats=args.gate_repeats,
            out_dir=str(model_dir), url_gates_override=recipe_url_gates(recipe, seed),
        )
        gates = torch.from_numpy(np.load(gate_result["gate_path"]).astype(np.float32))
        pd.DataFrame({"feature": gate_result["feature_names"], "gate": gates.numpy()}).to_csv(
            model_dir / f"gates_seed{seed}.csv", index=False)
    cnn_gates, url_gates = gates[:NUM_FILTERS], gates[NUM_FILTERS:]
    assert len(url_gates) == len(columns), "gate array does not match the aux columns"

    # ---- Phase C: fusion model, its own copy of the Phase A encoder ----------------------
    embedding, _ = load_embedding_layer()
    encoder = TextEncoder(embedding, NUM_FILTERS)
    load_phase_a_encoder_weights(encoder, str(phase_a_path))
    model = FusionClassifier(encoder, cnn_gates, url_gates, url_dim=len(columns), dense_units=DENSE_UNITS)

    _, train_aux, _ = load_seed_inputs(seed, "train", train_df, recipe)
    _, val_aux, _ = load_seed_inputs(seed, "train", val_df, recipe)
    train_aux, val_aux = train_aux[:, col_idx], val_aux[:, col_idx]
    phase_c_path = model_dir / f"phase_c_seed{seed}.pt"
    live_path = getattr(args, "live_train", None)
    head_ids, head_y = train_ids, train_y
    if live_path:
        head_set, head_ids, head_y = live_head_set(live_path, args, seed, recipe, col_idx, train_ids, train_aux, train_y)
        weights = class_weights(head_y)                    # recomputed on everything the head trains on
    else:
        head_set = FusionDataset(train_ids, train_aux, train_y)
    phase_c_metrics = train_phase_c(
        model,
        DataLoader(head_set, batch_size=args.batch_size, shuffle=True),
        DataLoader(FusionDataset(val_ids, val_aux, val_y), batch_size=512),
        epochs=args.epochs, lr=args.lr, checkpoint_path=str(phase_c_path), patience=args.patience,
        class_weights=weights,
    )
    model.load_state_dict(torch.load(phase_c_path, map_location="cpu"))

    # ---- Test split (first and only time it is read) -------------------------------------
    test_df = load_split_df(seed, "test")
    test_ids, test_aux, test_y = load_seed_inputs(seed, "test", test_df, recipe)
    test_aux = test_aux[:, col_idx]
    has_url = test_aux[:, columns.index("has_url")].numpy()
    y_true = test_y.numpy()

    rows = []
    for model_name, preds in [
        ("cnn_only", predict_labels(cnn, test_ids, None)),
        (f"gated_{variant}", predict_labels(model, test_ids, test_aux)),
    ]:
        for group in evaluate_groups(test_df, y_true, preds, has_url):
            rows.append({"seed": seed, "model": model_name, **group})

    info = {
        "seed": seed, "variant": variant, "phase_a_val": phase_a_metrics, "phase_c_val": phase_c_metrics,
        "class_weights": weights.tolist(), "n_train": len(train_y), "n_val": len(val_y), "n_test": len(y_true),
    }
    return rows, info


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    agg = per_seed.groupby(["model", "group"])[metrics + ["n"]].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--variant", choices=list(VARIANT_COLUMNS), default="with_pu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gate-repeats", type=int, default=30, help="permutation-importance repeats")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--live-train", type=Path, default=None,
                        help="live_links.csv: also train the head on its live_train rows (see the module docstring)")
    parser.add_argument("--live-repeats", type=int, default=8, help="synthetic texts per live link")
    parser.add_argument("--live-p-alt", type=float, default=0.5, help="chance a live row is read WITH its page evidence")
    parser.add_argument("--reuse-from", type=Path, default=None,
                        help="folder of a finished run of the same variant: copy its Phase A and gates, train only Phase C")
    parser.add_argument("--tag", default="", help="suffix for the output folders, e.g. _live (keeps the default run intact)")
    args = parser.parse_args()

    all_rows, infos = [], []
    for seed in args.seeds:
        rows, info = run_seed(seed, args.variant, args)
        all_rows += rows
        infos.append(info)

    out = args.out / (args.variant + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(out / "metrics_per_seed.csv", index=False)
    summary = summarize(per_seed)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    with open(out / "run_config.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "runs": infos}, f, indent=2)

    pd.set_option("display.width", 220)
    show = summary[["model", "group", "n_mean", "balanced_accuracy_mean", "balanced_accuracy_std", "f1_mean", "f1_std"]]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
