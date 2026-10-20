"""KNN + Random Forest + Extra Trees voting baseline (the text classifier of Jain et al. 2022), scored like film_compare's models.

Text only, no link evidence. Per seed and fold it is fitted on the fold's TRAIN part and scored on the fold's TEST part, so every
message is tested once per seed by a model that never saw it, on the same messages as the CNN / concatenation / E2 / RF-only rows.
The recipe is src/features/model/jain_baseline.py unchanged: lowercase, tokenize, drop special characters and stopwords, Porter
stem, TF-IDF (3000 features), SMOTE on the training matrix, hard majority vote of three scikit-learn default classifiers.
The validation part is not used (nothing is tuned).

The one change for this comparison: links are cut out of the SMS before the text is read. The CNN sees the link only as <URL>;
left in, TF-IDF would read the domain's spelling, which is link evidence the text-only models are not meant to have. The paper's
own routing (messages with a link go to a separate URL classifier) is not used here.

p_smishing is the share of the three voters that say smishing (0, 1/3, 2/3 or 1); the verdict is the majority, as the paper's.
The same predictions stand for both conditions (`complete` and `page hidden`): the model never reads the page.

Writes experiments/film/<dataset>/compare/voting_{predictions,metrics,summary,mcnemar}.csv. McNemar (exact, complete pages) is
the voting model against every other model in test_predictions.csv and rf_only_predictions.csv, Holm-corrected per seed over those
pairs only, so the other tables are left untouched.

Usage:
    python -m src.unified.voting_baseline --pilot
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd

from src.config import CROWD_N_FOLDS, SEEDS
from src.data.split_crowd import load_fold
from src.features.model.jain_baseline import JainSmishingDetector, find_urls, transform_text
from src.unified import fit_model as fm
from src.unified import text_cnn as tc
from src.unified.film_compare import CONDITIONS, holm, mcnemar, metrics_tables

VOTING = "KNN-RF-ETC voting (text only)"


def strip_links(text: str) -> str:
    """The SMS with its link(s) removed, so the text model reads words only."""
    for url in find_urls(text):
        text = text.replace(url, " ")
    return " ".join(text.split())


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(share of voters saying smishing, majority verdict) for every test row, from a detector fitted on `train` only."""
    detector = JainSmishingDetector(seed)
    y = (train["label"] == "smishing").astype(int).to_numpy()
    detector._fit_text_classifier([strip_links(t) for t in train["sms_text"]], y)   # the text classifier only, no URL classifier
    X = detector.tfidf.transform([_prep(t) for t in test["sms_text"]])
    votes = np.column_stack([est.predict(X) for est in detector.text_clf.estimators_])
    return votes.mean(axis=1), detector.text_clf.predict(X).astype(int)


def _prep(text: str) -> str:
    return transform_text(strip_links(text))


def voting_predictions(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None) -> pd.DataFrame:
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    rows = []
    for seed in seeds:
        for fold in folds:
            parts = load_fold(seed, fold, pilot=pilot)
            p, pred = fit_and_score(parts["train"], parts["test"], seed)
            label = (parts["test"]["label"] == "smishing").astype(int).to_numpy()
            for condition in CONDITIONS:
                rows.append(pd.DataFrame({"seed": seed, "fold": fold, "message_id": parts["test"]["message_id"].to_numpy(), "label": label,
                                          "model": VOTING, "condition": condition, "p_smishing": p, "pred": pred}))
            print(f"seed {seed} fold {fold}: test F1 {metrics_tables(rows[-1])[0]['f1'].iloc[0]:.4f}", flush=True)
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
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args(argv)
    out = fm.out_dir(tc.dataset_path(args.pilot), "compare")
    reference = pd.concat([pd.read_csv(out / "test_predictions.csv"), pd.read_csv(out / "rf_only_predictions.csv")], ignore_index=True)
    voting = voting_predictions(args.pilot, args.seeds)
    per_seed, summary = metrics_tables(voting)
    mc = mcnemar_vs_voting(reference[reference["seed"].isin(args.seeds)], voting)
    voting.to_csv(out / "voting_predictions.csv", index=False)
    per_seed.to_csv(out / "voting_metrics.csv", index=False)
    summary.to_csv(out / "voting_summary.csv", index=False)
    mc.to_csv(out / "voting_mcnemar.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print(per_seed.round(4).to_string(index=False))
    print(mc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
