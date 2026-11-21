"""KNN-RF-ETC baseline: the smishing detector of Jain et al. (2022), with its URL routing, scored like film_compare's models.

The paper's method (Fig. 4; src/features/model/jain_baseline.py, unchanged recipes), decided for the study on 2026-11-21
(docs/ARCHITECTURE.md, guide Step 4: "text ensemble + separate URL classifier"):
    message with NO link  -> TEXT classifier: lowercase, tokenize, drop special characters and stopwords, Porter stem, TF-IDF
                             (3,000 features), SMOTE on the training matrix, hard majority vote of KNN + Random Forest + Extra Trees
    message WITH a link   -> URL classifier: the link's letter tokens, Snowball stem, CountVectorizer, Logistic Regression.
                             The message text is not used for these messages (as in the paper).
p_smishing is the routed classifier's own score: Logistic Regression's probability on the URL route, the share of the three voters
saying smishing (0, 1/3, 2/3, 1) on the text route. The verdict is the routed classifier's (LR argmax / majority vote).

Adapted to the study (each one recorded in the predictions file's `route` column or below):
  * Training data: each run's TRAIN part only (the paper used separate SMS and URL corpora), so it is tested on the same messages as
    every other model, by a model that never saw them or their group (domain / template / near-duplicate). Not tuned: the validation
    part is not used.
  * Routing uses the dataset's own has_url and link columns (the link finder every other model uses), not UrlExtract, so the three
    message conditions (no URL / page not loaded / page loaded) are the same messages for every model.
  * The text classifier is fitted on ALL training messages with their links removed (fitting it on the no-link messages alone would
    leave about 50 smishing examples per run). The URL classifier is fitted on the links of the training messages that have one,
    each labelled with its message's label; duplicate (link, label) pairs removed, as in the paper.
  * The page is never read, so the predictions are the same in both conditions (`complete` and `page hidden`).

Writes experiments/film/<dataset>/compare/voting_{predictions,metrics,summary}.csv and, when the other models' predictions exist,
voting_mcnemar.csv (exact, complete pages; this model against every other model, Holm-corrected per seed over those pairs).

Step train (guide Step 4): per seed and fold, fit on the TRAIN part, score the VALIDATION part (a record only: nothing is tuned), save
the fitted model to models/film/<dataset>/voting_seed<S>_fold<F>.joblib (the file efficiency.py times) and write
experiments/film/<dataset>/voting/train_summary.csv. Step predict (guide Step 5, the default) scores the TEST part with the saved
model when it exists (fitting it otherwise).

Usage:
    python -m src.unified.voting_baseline train
    python -m src.unified.voting_baseline [predict]
"""
from __future__ import annotations

import argparse
import time
from typing import List

import joblib

import numpy as np
import pandas as pd

from src.config import CROWD_N_FOLDS, SEEDS
from src.data.split_crowd import load_fold
from src.features.model.jain_baseline import JainSmishingDetector, transform_text, transform_url
from src.features.text_preprocessing import mask_urls
from src.unified import fit_model as fm
from src.unified import text_cnn as tc
from src.unified.film_compare import CONDITIONS, holm, mcnemar, metrics_tables

VOTING = "KNN-RF-ETC (Jain et al. 2022)"


def strip_links(text: str) -> str:
    """The SMS with its link(s) removed (the same link finder that sets has_url), so the text classifier reads words only."""
    return " ".join(mask_urls(str(text)).replace("<URL>", " ").split())


def _has_url(rows: pd.DataFrame) -> np.ndarray:
    return rows["has_url"].astype(int).to_numpy() == 1


def fit_detector(train: pd.DataFrame, seed: int) -> JainSmishingDetector:
    """Both classifiers fitted on `train` only: TF-IDF + the KNN / RF / Extra Trees vote (`tfidf`, `text_clf`) on every message with its
    links removed, and CountVectorizer + Logistic Regression (`url_vec`, `url_clf`) on the links of the messages that have one."""
    detector = JainSmishingDetector(seed)
    y = (train["label"] == "smishing").astype(int).to_numpy()
    detector.text_info = detector._fit_text_classifier([strip_links(t) for t in train["sms_text"]], y)
    linked = _has_url(train)
    detector.url_info = detector.fit_url_classifier_on(train.loc[linked, "link"].astype(str).tolist(), y[linked])
    return detector


def score(detector: JainSmishingDetector, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p_smishing, verdict, route) for every row: route "url" (has_url = 1, the URL classifier on the row's link) or "text"."""
    linked = _has_url(rows)
    p, pred = np.zeros(len(rows)), np.zeros(len(rows), dtype=int)
    if (~linked).any():
        X = detector.tfidf.transform([_prep(t) for t in rows.loc[~linked, "sms_text"]])
        votes = np.column_stack([est.predict(X) for est in detector.text_clf.estimators_])
        p[~linked], pred[~linked] = votes.mean(axis=1), detector.text_clf.predict(X).astype(int)
    if linked.any():
        U = detector.url_vec.transform([transform_url(u) for u in rows.loc[linked, "link"].astype(str)])
        p[linked], pred[linked] = detector.url_clf.predict_proba(U)[:, 1], detector.url_clf.predict(U).astype(int)
    return p, pred, np.where(linked, "url", "text")


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(p_smishing, verdict) for every test row, from a detector fitted on `train` only."""
    p, pred, _ = score(fit_detector(train, seed), test)
    return p, pred


def _prep(text: str) -> str:
    return transform_text(strip_links(text))


def model_path(pilot: bool, seed: int, fold: int):
    return fm.model_dir(tc.dataset_path(pilot)) / f"voting_seed{seed}_fold{fold}.joblib"


def load_or_fit(parts: dict, seed: int, fold: int, pilot: bool) -> JainSmishingDetector:
    """The saved model of this seed and fold (step train), or a fresh fit on the TRAIN part if none was saved."""
    path = model_path(pilot, seed, fold)
    if path.exists():
        bundle = joblib.load(path)
        assert bundle["seed"] == seed and bundle["fold"] == fold
        return bundle["detector"]
    return fit_detector(parts["train"], seed)


def run_train(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None) -> pd.DataFrame:
    """Guide Step 4 for KNN-RF-ETC: fit on TRAIN, score VALIDATION (record only), save the model."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for seed in seeds:
        for fold in folds:
            parts = load_fold(seed, fold, pilot=pilot)
            start = time.perf_counter()
            detector = fit_detector(parts["train"], seed)
            seconds = time.perf_counter() - start
            path = model_path(pilot, seed, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            # tfidf / text_clf / url_vec / url_clf are also stored on their own: the keys efficiency.py loads
            joblib.dump({"detector": detector, "tfidf": detector.tfidf, "text_clf": detector.text_clf, "url_vec": detector.url_vec,
                         "url_clf": detector.url_clf, "seed": seed, "fold": fold}, path)
            _, pred, route = score(detector, parts["val"])
            y = (parts["val"]["label"] == "smishing").astype(int).to_numpy()
            prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            rows.append({"seed": seed, "fold": fold, "n_train": len(parts["train"]), "n_val": len(parts["val"]),
                         "text_clf_messages": detector.text_info["n_rows_after_cleaning"],
                         "url_clf_unique_links": detector.url_info["n_unique_urls"], "fit_seconds": round(seconds, 2),
                         "val_accuracy": accuracy_score(y, pred), "val_balanced_accuracy": balanced_accuracy_score(y, pred),
                         "val_precision": prec, "val_recall": rec, "val_f1": f1,
                         "val_url_route": int((route == "url").sum()), "val_text_route": int((route == "text").sum())})
            print(f"seed {seed} fold {fold}: val F1 {f1:.3f}, accuracy {rows[-1]['val_accuracy']:.3f} | {seconds:.1f} s", flush=True)
    out = fm.out_dir(tc.dataset_path(pilot), "voting")
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "train_summary.csv", index=False)
    return summary


def voting_predictions(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None) -> pd.DataFrame:
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for seed in seeds:
        for fold in folds:
            parts = load_fold(seed, fold, pilot=pilot)
            p, pred, route = score(load_or_fit(parts, seed, fold, pilot), parts["test"])
            label = (parts["test"]["label"] == "smishing").astype(int).to_numpy()
            for condition in CONDITIONS:
                rows.append(pd.DataFrame({"seed": seed, "fold": fold, "message_id": parts["test"]["message_id"].to_numpy(), "label": label,
                                          "model": VOTING, "condition": condition, "p_smishing": p, "pred": pred, "route": route}))
            print(f"seed {seed} fold {fold}: test F1 {metrics_tables(rows[-1].drop(columns='route'))[0]['f1'].iloc[0]:.4f}", flush=True)
    return pd.concat(rows, ignore_index=True)


def mcnemar_vs_voting(reference: pd.DataFrame, voting: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([reference[reference["condition"] == "complete"], voting[voting["condition"] == "complete"]])
    rows = []
    for seed, g in both.groupby("seed"):
        wide = {m: gm.sort_values(["fold", "message_id"]) for m, gm in g.groupby("model")}
        others = [m for m in wide if m != VOTING]
        for m in others:
            assert (wide[m]["message_id"].to_numpy() == wide[VOTING]["message_id"].to_numpy()).all(), f"seed {seed}: {m} rows differ"
        results = [mcnemar(wide[VOTING]["label"].to_numpy(), wide[m]["pred"].to_numpy(), wide[VOTING]["pred"].to_numpy()) for m in others]
        adjusted = holm([r["p_exact"] for r in results])
        for m, r, p in zip(others, results, adjusted):      # b = the other model right and voting wrong
            rows.append({"seed": seed, "model_a": m, "model_b": VOTING, **r, "p_holm": p, "significant_0.05": p < 0.05})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", nargs="?", choices=["train", "predict"], default="predict")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args(argv)
    if args.step == "train":
        summary = run_train(args.pilot, args.seeds)
        print(f"\n{len(summary)} KNN-RF-ETC models fitted and saved; mean val F1 {summary['val_f1'].mean():.3f} "
              f"(std {summary['val_f1'].std():.3f})")
        return
    out = fm.out_dir(tc.dataset_path(args.pilot), "compare")
    out.mkdir(parents=True, exist_ok=True)
    voting = voting_predictions(args.pilot, args.seeds)
    per_seed, summary = metrics_tables(voting.drop(columns="route"))
    voting.to_csv(out / "voting_predictions.csv", index=False)
    per_seed.to_csv(out / "voting_metrics.csv", index=False)
    summary.to_csv(out / "voting_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print(per_seed.round(4).to_string(index=False))
    others = [out / "test_predictions.csv", out / "rf_only_predictions.csv"]
    if all(f.exists() for f in others):          # McNemar needs the other models' test predictions (guide Step 10)
        reference = pd.concat([pd.read_csv(f) for f in others], ignore_index=True)
        mc = mcnemar_vs_voting(reference[reference["seed"].isin(args.seeds)], voting.drop(columns="route"))
        mc.to_csv(out / "voting_mcnemar.csv", index=False)
        print(mc.round(4).to_string(index=False))
    else:
        print("McNemar skipped: the other models' test predictions are not written yet")


if __name__ == "__main__":
    main()
