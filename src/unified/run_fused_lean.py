"""Fused pipeline end to end on the lean fresh set: CNN vs UNGATED vs the static-gated model, same test messages.

Input: data/processed/fresh_lean_synthetic.csv (python -m src.unified.prepare_lean_synthetic). Its SMS text is
SYNTHETIC, so every score here is a pipeline smoke test, NOT a study result (docs/progress.md, section 2g).

Per seed (42, 123, 2024):
  1. Split. At most 20 rows per link domain per class, then 70% fresh_dev / 30% fresh_test grouped by domain
     (a domain never spans both). fresh_dev is cut again into head-train / head-val (about 85/15, grouped).
  2. Text path. Links are removed from the SMS (as for v2); the frozen v2 text CNN gives 64 features. Words the v2
     vocabulary never saw are dropped (encode_batch_tolerant); the share is reported.
  3. Link path. The frozen web-trained forest (forest_seed*_f37.joblib) gives 37 contributions, P and u for each
     page, complete and with the 17 page/outside columns blanked. P is calibrated (Platt) on fresh_dev. The forest never
     saw fresh rows, so no out-of-fold step is needed.
  4. Static gates, fixed numbers: cnn_gates_seed*.npy (64) and rf_gates_seed*_f37_fix.npy (37), made on v2 and the web
     set, never on the fresh rows. They are applied after z-scoring.
  5. Head (all three models share it and the recipe): Dense 64, ReLU, Dropout 0.5, 2 outputs; class weights, label
     smoothing, Adam 1e-3, 30 epochs, patience 3 on head-val balanced accuracy.
       CNN       the 64 text features only (no gates)
       CNN_CONCAT the 64 text features + the 37 RAW link/page features (signed log1p, z-score) + page_missing, has_url.
                 No forest, no gates: the plain "concatenate the features" baseline. A blanked page has its 17 page
                 columns set to the training mean (0 after z-score) and page_missing=1.
       UNGATED   64 text + 37 contributions + [P_cal, u, page_missing, has_url]; every gate 1.0
       GATED     the same inputs with the static gates on the 64 text features and 37 contributions
                 (P_cal, u, page_missing, has_url are never gated)
     Modality dropout while training the fused heads: a page is replaced by its blanked version with p=0.25
     (page_missing=1), the text features are zeroed with p=0.10. Same random draws for UNGATED and GATED.
  6. fresh_test is scored once: complete pages (headline) and pages blanked. Predictions are saved with message_id
     and the three models are asserted to hold identical ids. Paired McNemar (exact) per seed on complete pages.

CNN_CONCAT trains with the same modality dropout and the same random draws as UNGATED and GATED.
Also reported for reference: the frozen v2 CNN's own classifier, and GATED/UNGATED with the link or the text zeroed.

Writes experiments/fused_lean/ (per_seed.csv, summary.csv, mcnemar.csv, predictions.csv, split_sizes.csv,
run_config.json) and data/splits_fresh_lean/.

Usage:
    python -m src.unified.run_fused_lean [--seeds 42 123 2024]
"""
import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from src.config import ARTIFACTS_ROOT, LABEL_SMOOTHING, NUM_FILTERS, SEEDS
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.fusion_data import class_weights, encode_batch_tolerant, load_embedding_layer
from src.features.model.rf_extractor import tree_probabilities
from src.features.model.rf_feature_extractor import path_contributions
from src.features.model.run_experiment import predict_labels, score_group
from src.features.text_preprocessing import text_tokens
from src.unified import prepare_web_37 as w37
from src.unified.gating_check import Head
from src.unified.prepare_v2 import strip_links
from src.unified.train_static_gating import EMB_V2, MODEL_DIR

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "fresh_lean_synthetic.csv"
WEB = ROOT / "data" / "processed" / "web_phishing_37_linked.csv"
SPLITS = ROOT / "data" / "splits_fresh_lean"
OUT_DIR = ARTIFACTS_ROOT / "experiments" / "fused_lean"
FEATURES, PAGE = w37.FEATURES_37, w37.PAGE_FEATURES
DOMAIN_CAP = 20
P_BLANK, P_TEXT_DROP = 0.25, 0.10
N_TEXT, N_RF = 64, len(FEATURES)
METRICS = ["accuracy", "precision", "recall", "f1"]


# ------------------------------------------------------------------ split
def make_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Cap rows per domain and class, then 70/30 dev/test grouped by domain. Returns df with a `part` column."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    shuffled = df.iloc[order]
    capped = shuffled[shuffled.groupby(["label", "link_domain"]).cumcount() < DOMAIN_CAP].copy()
    y, groups = (capped["label"] == "smishing").to_numpy().astype(int), capped["link_domain"].to_numpy()
    folds = list(StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(capped, y, groups))
    test_idx = np.concatenate([folds[i][1] for i in range(3)])
    capped["part"] = "dev"
    capped.iloc[test_idx, capped.columns.get_loc("part")] = "test"
    assert set(capped.loc[capped.part == "dev", "link_domain"]).isdisjoint(capped.loc[capped.part == "test", "link_domain"])
    out = SPLITS / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    capped[["message_id", "part"]].to_csv(out / "parts.csv", index=False)
    return capped.reset_index(drop=True)


def head_split(dev: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y, groups = (dev["label"] == "smishing").to_numpy().astype(int), dev["link_domain"].to_numpy()
    tr, va = next(iter(StratifiedGroupKFold(7, shuffle=True, random_state=seed).split(dev, y, groups)))
    return tr, va


def labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] == "smishing").to_numpy().astype(int)


# ------------------------------------------------------------------ features
def text_features(cnn: TextEncoder, vocab: dict, df: pd.DataFrame) -> np.ndarray:
    ids = encode_batch_tolerant([strip_links(t) for t in df["sms_text"]], vocab)
    with torch.no_grad():
        return cnn.encode(ids).numpy(), ids


def link_outputs(rf, df: pd.DataFrame, blank: bool):
    X = df[FEATURES].copy()
    if blank:
        X[PAGE] = np.nan
    contrib, _ = path_contributions(rf, X.to_numpy(dtype=np.float32))
    per_tree = tree_probabilities(rf, X)
    return contrib, per_tree.mean(axis=0), per_tree.std(axis=0)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Fused:
    """Turns text features + link outputs into head inputs: z-score, then the gates, then the ungated tail."""

    def __init__(self, text_tr, contrib_tr, text_gates, rf_gates):
        self.mt, self.st = text_tr.mean(0), text_tr.std(0) + 1e-8
        self.mc, self.sc = contrib_tr.mean(0), contrib_tr.std(0) + 1e-8
        self.gt, self.gr = text_gates.astype(np.float32), rf_gates.astype(np.float32)

    def __call__(self, text, contrib, p_cal, u, missing: float) -> np.ndarray:
        n = len(text)
        tail = np.column_stack([p_cal, u, np.full(n, missing), np.ones(n)])          # P_cal, u, page_missing, has_url
        return np.column_stack([(text - self.mt) / self.st * self.gt, (contrib - self.mc) / self.sc * self.gr,
                                tail]).astype(np.float32)


class RawBlock:
    """The 37 raw link/page features for the concat baseline: signed log1p, z-score, blanked page columns -> 0."""

    def __init__(self, df_train: pd.DataFrame):
        x = self._log(df_train)
        self.m, self.s = x.mean(0), x.std(0) + 1e-8
        self.page_idx = [FEATURES.index(c) for c in PAGE]

    @staticmethod
    def _log(df: pd.DataFrame) -> np.ndarray:
        x = df[FEATURES].to_numpy(dtype=np.float64)
        return np.sign(x) * np.log1p(np.abs(x))

    def __call__(self, df: pd.DataFrame, text: np.ndarray, blank: bool) -> np.ndarray:
        z = (self._log(df) - self.m) / self.s
        if blank:
            z[:, self.page_idx] = 0.0
        n = len(df)
        return np.column_stack([text, z, np.full(n, float(blank)), np.ones(n)]).astype(np.float32)


# ------------------------------------------------------------------ head
def fit_head(Xc, Xb, y, Xv, yv, seed: int, text_cols: int, epochs: int = 30, patience: int = 3, lr: float = 1e-3) -> Head:
    """Xc: complete-page inputs, Xb: the same rows with blanked pages (None for the text-only CNN model)."""
    torch.manual_seed(seed)
    model = Head(Xc.shape[1])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(torch.tensor(y)), label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xc_t, yt, Xv_t = torch.tensor(Xc), torch.tensor(y), torch.tensor(Xv)
    Xb_t = None if Xb is None else torch.tensor(Xb)
    gen = torch.Generator().manual_seed(seed)
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        X = Xc_t
        if Xb_t is not None:
            blank_rows = torch.rand(len(X), generator=gen) < P_BLANK
            X = torch.where(blank_rows[:, None], Xb_t, Xc_t)
            drop_text = torch.rand(len(X), generator=gen) < P_TEXT_DROP
            X = X.clone()
            X[drop_text, :text_cols] = 0.0
        order = torch.randperm(len(X), generator=gen)
        for s in range(0, len(order), 64):
            idx = order[s:s + 64]
            opt.zero_grad()
            loss_fn(model(X[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            score = balanced_accuracy_score(yv, model(Xv_t).argmax(dim=1).numpy())
        if score > best:
            best, state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model.eval()


def predict(model: Head, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(X)).argmax(dim=1).numpy()


def mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    only_a, only_b = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    n = only_a + only_b
    return {"a_only_right": only_a, "b_only_right": only_b, "p_value": 1.0 if n == 0 else binomtest(only_a, n, 0.5).pvalue}


# ------------------------------------------------------------------ one seed
def run_seed(seed: int, data: pd.DataFrame, web_urls: set) -> tuple[list, list, list, dict]:
    df = make_split(data, seed)
    dev, test = df[df.part == "dev"].reset_index(drop=True), df[df.part == "test"].reset_index(drop=True)
    tr_i, va_i = head_split(dev, seed)
    tr, va = dev.iloc[tr_i].reset_index(drop=True), dev.iloc[va_i].reset_index(drop=True)
    sizes = {"seed": seed, "capped_rows": len(df), "dev": len(dev), "head_train": len(tr), "head_val": len(va),
             "test": len(test), "test_smishing_share": round(float(labels(test).mean()), 3),
             "test_domains": test.link_domain.nunique(), "dev_domains": dev.link_domain.nunique()}
    print(f"[seed {seed}] {sizes}", flush=True)

    # frozen parts
    embedding, vocab = load_embedding_layer(EMB_V2)
    cnn = TextEncoder(embedding, NUM_FILTERS)
    cnn.load_state_dict(torch.load(MODEL_DIR / f"phase_a_seed{seed}.pt", map_location="cpu"))
    cnn.eval()
    rf = joblib.load(MODEL_DIR / f"forest_seed{seed}_f37.joblib")
    assert rf["features"] == FEATURES
    rf_gates = np.load(MODEL_DIR / f"rf_gates_seed{seed}_f37_fix.npy")
    cnn_gates = np.load(MODEL_DIR / f"cnn_gates_seed{seed}.npy")
    assert rf_gates.shape == (N_RF,) and cnn_gates.shape == (N_TEXT,) and np.isfinite(rf_gates).all() and np.isfinite(cnn_gates).all()

    text, ids = {}, {}
    for name, part in (("tr", tr), ("va", va), ("dev", dev), ("te", test)):
        text[name], ids[name] = text_features(cnn, vocab, part)
    link = {}
    for name, part in (("tr", tr), ("va", va), ("dev", dev), ("te", test)):
        link[name] = {b: link_outputs(rf["model"], part, b) for b in (False, True)}

    # Platt calibration of P on fresh_dev (complete + blanked pages pooled); the forest never saw these rows
    P_dev = np.concatenate([link["dev"][False][1], link["dev"][True][1]])
    y_dev = np.concatenate([labels(dev), labels(dev)])
    platt = LogisticRegression(C=1e6).fit(logit(P_dev).reshape(-1, 1), y_dev)
    cal = lambda p: platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]

    def inputs(fused: Fused, name: str, blank: bool) -> np.ndarray:
        contrib, p, u = link[name][blank]
        return fused(text[name], contrib, cal(p), u, float(blank))

    ones_t, ones_r = np.ones(N_TEXT, dtype=np.float32), np.ones(N_RF, dtype=np.float32)
    variants = {"UNGATED": Fused(text["tr"], link["tr"][False][0], ones_t, ones_r),
                "GATED": Fused(text["tr"], link["tr"][False][0], cnn_gates, rf_gates)}
    assert (variants["UNGATED"].gt == 1).all() and (variants["UNGATED"].gr == 1).all()
    assert np.allclose(variants["GATED"].gt, cnn_gates) and np.allclose(variants["GATED"].gr, rf_gates)

    y_tr, y_va, y_te = labels(tr), labels(va), labels(test)
    models, preds = {}, {}      # preds[(model, condition)] -> labels on `test`

    # CNN baseline: the 64 text features only, no gates
    mt, st = text["tr"].mean(0), text["tr"].std(0) + 1e-8
    z = lambda t: ((t - mt) / st).astype(np.float32)
    cnn_head = fit_head(z(text["tr"]), None, y_tr, z(text["va"]), y_va, seed, text_cols=N_TEXT)
    for cond in ("complete", "blanked"):
        preds[("CNN", cond)] = predict(cnn_head, z(text["te"]))      # the text path does not read the page

    # CNN_CONCAT baseline: z-scored text features + raw link features, no forest and no gates
    raw = RawBlock(tr)
    cc = lambda name, part, blank: raw(part, z(text[name]), blank)
    concat_head = fit_head(cc("tr", tr, False), cc("tr", tr, True), y_tr, cc("va", va, False), y_va, seed, text_cols=N_TEXT)
    preds[("CNN_CONCAT", "complete")] = predict(concat_head, cc("te", test, False))
    preds[("CNN_CONCAT", "blanked")] = predict(concat_head, cc("te", test, True))

    for name, fused in variants.items():
        head = fit_head(inputs(fused, "tr", False), inputs(fused, "tr", True), y_tr,
                        inputs(fused, "va", False), y_va, seed, text_cols=N_TEXT)
        models[name] = head
        preds[(name, "complete")] = predict(head, inputs(fused, "te", False))
        preds[(name, "blanked")] = predict(head, inputs(fused, "te", True))
        Xz = inputs(fused, "te", False).copy()                      # zeroing checks: one branch removed at test time
        Xz[:, N_TEXT:] = 0.0
        preds[(name + "_link_zeroed", "complete")] = predict(head, Xz)
        Xz = inputs(fused, "te", False).copy()
        Xz[:, :N_TEXT] = 0.0
        preds[(name + "_text_zeroed", "complete")] = predict(head, Xz)

    # reference: the frozen v2 CNN's own classifier
    direct = predict_labels(cnn, ids["te"], None)
    for cond in ("complete", "blanked"):
        preds[("CNN_frozen_v2_direct", cond)] = direct

    # same test messages for every model
    id_sets = {k: tuple(test["message_id"]) for k in preds}
    assert len(set(id_sets.values())) == 1 and all(len(p) == len(test) for p in preds.values())

    rows, pred_rows, tests = [], [], []
    for (model, cond), p in preds.items():
        rows.append({"seed": seed, "model": model, "condition": cond, **score_group(y_te, p)})
        pred_rows.append(pd.DataFrame({"seed": seed, "message_id": test["message_id"], "condition": cond, "model": model,
                                       "label": y_te, "pred": p}))
    for a, b in (("GATED", "UNGATED"), ("GATED", "CNN"), ("UNGATED", "CNN"), ("CNN_CONCAT", "CNN"),
                 ("GATED", "CNN_CONCAT"), ("UNGATED", "CNN_CONCAT")):
        tests.append({"seed": seed, "model_a": a, "model_b": b, "condition": "complete",
                      **mcnemar(y_te, preds[(a, "complete")], preds[(b, "complete")])})

    oov = np.mean([t not in vocab for s in test["sms_text"] for t in text_tokens(strip_links(s))])
    sizes["test_oov_token_share"] = round(float(oov), 4)
    return rows, pred_rows, tests, sizes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA, low_memory=False)
    web_urls = set(pd.read_csv(WEB, usecols=["url"])["url"])
    shared = data["link"].isin(web_urls)
    print(f"{int(shared.sum())} rows share an exact URL with the web set (the forest and gates saw it); dropped")
    data = data[~shared].reset_index(drop=True)

    rows, preds, tests, sizes = [], [], [], []
    for seed in args.seeds:
        r, p, t, s = run_seed(seed, data, web_urls)
        rows += r
        preds += p
        tests += t
        sizes.append(s)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT_DIR / "per_seed.csv", index=False)
    pd.concat(preds).to_csv(OUT_DIR / "predictions.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT_DIR / "mcnemar.csv", index=False)
    pd.DataFrame(sizes).to_csv(OUT_DIR / "split_sizes.csv", index=False)
    summary = per_seed.groupby(["condition", "model"], sort=False)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.reset_index().to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "run_config.json").write_text(json.dumps({"seeds": args.seeds, "domain_cap": DOMAIN_CAP, "p_blank": P_BLANK,
                                                         "p_text_drop": P_TEXT_DROP, "data": str(DATA)}, indent=2))
    pd.set_option("display.width", 220)
    print("\nmean over seeds (std in the csv):")
    print(summary[[f"{m}_mean" for m in METRICS]].round(3).to_string())
    print("\nMcNemar (exact), complete pages:")
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    print("\nsplit sizes:")
    print(pd.DataFrame(sizes).to_string(index=False))


if __name__ == "__main__":
    main()
