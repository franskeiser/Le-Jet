r"""The thesis "experiment papers" (Appendix 1, 2, 4 and 5) as one Markdown file, filled from the saved results.

    .venv\Scripts\python.exe scripts/build_experiment_papers.py            (after Steps 5-10)

Every number comes from the Step 5 out-of-fold prediction files (experiments/film/<dataset>/step5/<Model>_Predictions.csv); nothing is
trained, scored or read from a test fold here. McNemar uses scripts/build_step10_files.py's own test and is cross-checked against its
workbook when it exists.

ONE NEW COPY PER RETRAINING: the file is written to experiments/film/<dataset>/experiment_papers/ as
    Experiment_Papers_<YYYYMMDD-HHMM>_<fingerprint>.md
where <fingerprint> is the first 10 characters of the sha256 of the four prediction files. A retraining changes the predictions, so it gets
a new file next to the old ones; running this again without retraining finds the same fingerprint and writes nothing (--force writes anyway).
experiment_papers/index.md lists every copy with its headline numbers. scripts/build_step10_files.py calls this at its end.

The fixed text of the template (Table 1.4, the "how the link evidence is used" column, Appendix 4, Appendix 5) is reproduced with the
corrections found when it was checked against the code on 2026-11-22; each correction is listed at the end of the file.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.config import SEEDS                                    # noqa: E402
from src.unified import fit_model as fm                         # noqa: E402
from src.unified import text_cnn as tc                          # noqa: E402
from build_step10_files import ALPHA, holm, mcnemar             # noqa: E402

# (name in the papers, name in the prediction files, file stem); the papers' order
MODELS = [("CNN-only, text + 37 URL features", "CNN-only", "CNN_only"),
          ("Concatenation fusion", "Concatenation Fusion", "Concatenation"),
          ("KNN-RF-ETC voting", "KNN-RF-ETC", "KNN_RF_ETC"),
          ("Proposed (RF-conditioned FiLM)", "Proposed Model", "Proposed_Model")]
PROPOSED = MODELS[3][0]
CONDITIONS = [("No link", "no URL"), ("Link, page not reached", "inaccessible webpage"), ("Link, page reached", "accessible webpage")]
METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]

HOW_LINK_IS_USED = {
    "CNN-only, text + 37 URL features": (
        "SMS text (fastText embedding, Conv1D with 64 filters) and the 37 URL, webpage and domain features (long-tailed counts log-scaled, "
        "standardized on the training part, -1 replaced by 0 with missing-value flags, through a Dense branch), with page_missing and has_url, "
        "combined in one network without a Random Forest"),
    "Concatenation fusion": (
        "64 CNN text features concatenated with the Random Forest outputs P and u, plus page_missing and has_url, before the Dense layer"),
    "KNN-RF-ETC voting": (
        "Routed by link: a message with no link goes to the text classifier (TF-IDF, 3,000 features, link removed, SMOTE; majority vote of "
        "KNN, Random Forest and Extra Trees); a message with a link goes to the URL classifier (the link's word tokens, CountVectorizer, "
        "Logistic Regression), and its text is not used"),
    "Proposed (RF-conditioned FiLM)": (
        "64 CNN text features modulated by FiLM (γ, β), generated from the Random Forest outputs P and u, page_missing, and the average of "
        "the 64 convolution features; has_url switches the modulation off when there is no link (γ = 1, β = 0); the classifier sees only the "
        "64 modulated text features"),
}

APPENDIX_4 = [
    # (group, feature, operational definition, encoding); corrected rows are marked in CORRECTIONS
    ("URL Presence", "has_url", "Whether at least one URL is detected in the SMS message", "1 = URL present; 0 = no URL"),
    ("URL Presence", "page_missing", "Whether the webpage was not available to the model: the link's page could not be reached at inspection time, or the message has no link", "1 = not reached or no URL; 0 = reached"),
    ("URL Shape", "length_url", "Number of characters in the full URL", "Integer"),
    ("URL Shape", "length_hostname", "Number of characters in the hostname", "Integer"),
    ("URL Shape", "nb_dots", "Number of dot characters in the URL", "Integer"),
    ("URL Shape", "nb_slash", "Number of slash characters in the URL", "Integer"),
    ("URL Shape", "nb_www", 'Number of times "www" appears in the URL', "Integer"),
    ("URL Shape", "https_token", "Whether the URL uses the HTTPS scheme", "1 = HTTPS; 0 = not HTTPS"),
    ("URL Shape", "ratio_digits_url", "Number of digits divided by the number of characters in the URL", "Ratio, 0–1"),
    ("URL Shape", "ratio_digits_host", "Number of digits divided by the number of characters in the hostname", "Ratio, 0–1"),
    ("URL Shape", "prefix_suffix", "Whether the registered domain contains a hyphen", "1 = hyphen present; 0 = none"),
    ("URL Shape", "shortening_service", "Whether the domain belongs to a known URL-shortening service", "1 = shortener; 0 = not a shortener"),
    ("URL Shape", "nb_redirection", "Number of redirections followed from the original URL to the final page (the lab follows at most eight hops; an app-store hand-off is counted as one more)", "Integer, 0–9; −1 = not measured"),
    ("URL Words", "length_words_raw", "Number of words in the URL after splitting on delimiters such as dots, slashes, and hyphens", "Integer"),
    ("URL Words", "char_repeat", "Number of URL words that contain a character repeated consecutively", "Integer"),
    ("URL Words", "shortest_word_host", "Length of the shortest word in the hostname", "Integer"),
    ("URL Words", "shortest_word_path", "Length of the shortest word in the URL path", "Integer"),
    ("URL Words", "longest_words_raw", "Length of the longest word in the URL", "Integer"),
    ("URL Words", "longest_word_host", "Length of the longest word in the hostname", "Integer"),
    ("URL Words", "avg_word_host", "Average length of the words in the hostname", "Decimal"),
    ("URL Words", "phish_hints", "Number of predefined phishing hint words, such as login, secure, account, update, or verify, in the URL", "Integer"),
    ("URL Words", "domain_in_brand", "Whether the registered domain name matches an entry in a list of brand names", "1 = match; 0 = none"),
    ("Webpage", "nb_hyperlinks", "Number of hyperlinks in the page HTML", "Integer; −1 = page not reached"),
    ("Webpage", "ratio_intHyperlinks", "Share of hyperlinks that point to the same domain", "Ratio, 0–1; −1 = page not reached"),
    ("Webpage", "nb_extCSS", "Number of stylesheets loaded from another domain", "Integer; −1 = page not reached"),
    ("Webpage", "login_form", "Whether the page contains a form whose action is empty, blank, or points to another domain", "1 = present; 0 = none; −1 = page not reached"),
    ("Webpage", "external_favicon", "Whether the page icon is loaded from another domain", "1 = external; 0 = not external; −1 = page not reached"),
    ("Webpage", "links_in_tags", "Percentage of links in link, script, and meta tags that point to the same domain", "Percentage, 0–100; −1 = page not reached"),
    ("Webpage", "ratio_intMedia", "Percentage of images, audio, and video loaded from the same domain", "Percentage, 0–100; −1 = page not reached"),
    ("Webpage", "ratio_extMedia", "Percentage of images, audio, and video loaded from another domain", "Percentage, 0–100; −1 = page not reached"),
    ("Webpage", "iframe", "Whether the page contains an invisible or borderless frame", "1 = present; 0 = none; −1 = page not reached"),
    ("Webpage", "safe_anchor", 'Percentage of anchor tags whose link contains "#", "javascript", or "mailto" (unsafe anchors)', "Percentage, 0–100 (0 if the page has no anchors); −1 = page not reached"),
    ("Webpage", "empty_title", "Whether the page title is empty or missing", "1 = empty; 0 = not; −1 = page not reached"),
    ("Webpage", "domain_in_title", "Whether the registered domain name is ABSENT from the page title", "1 = absent; 0 = present; −1 = page not reached"),
    ("Webpage", "domain_with_copyright", "Whether the page has a copyright or trademark mark WITHOUT the registered domain name within 50 characters of it", "1 = mark without the domain; 0 = no mark, or the domain is next to it; −1 = page not reached"),
    ("Domain", "whois_registered_domain", "Whether no registration record is found for the domain through RDAP", "1 = not found; 0 = found; −1 = lookup failed"),
    ("Domain", "domain_registration_length", "Number of days from the inspection date to the domain's registration expiry", "Integer days; −1 = unknown"),
    ("Domain", "domain_age", "Number of days from the domain's registration date to the inspection date", "Integer days; −1 = unknown"),
    ("Domain", "page_rank", "Open PageRank score of the domain", "0–10; 0 = unknown domain; −1 = failed lookup"),
    ("Random Forest Output", "P", "Random Forest probability assigned to the smishing class from the 37 features (mean over its 200 trees; training rows are scored out of fold)", "Decimal, 0–1"),
    ("Random Forest Output", "u", "Uncertainty of the Random Forest output: the standard deviation of its 200 trees' smishing probabilities", "Decimal, 0–0.5"),
]

APPENDIX_5 = [
    ("SMS Text", "Processes SMS messages using trainable fastText embeddings and a one-dimensional CNN with 64 convolutional filters",
     "Not applicable", "Not applicable", "Provides textual feature representations for conditioning and classification"),
    ("URL Structural Features", "Not directly processed by the CNN text encoder",
     "Extracts 37 URL structural features covering URL shape, URL words, webpage, and domain characteristics",
     "Processes the 37 URL structural features to generate link-related predictions",
     "Not read directly; reaches the generator only through the Random Forest's P and u"),
    ("CNN Feature Representations", "Generates 64 convolutional feature maps", "Not applicable", "Not applicable",
     "The generator reads the average of each of the 64 convolution maps over all positions (before activation); the classifier reads "
     "the 64 max-pooled activations after FiLM"),
    ("Random Forest Probability (P)", "Not applicable", "Derived from URL structural features",
     "Generates the probability assigned to the smishing class", "Used as a conditioning input"),
    ("Random Forest Uncertainty (u)", "Not applicable", "Derived from URL structural features",
     "Generates uncertainty as the standard deviation of its decision trees' probabilities", "Used as a conditioning input"),
    ("Page Availability (page_missing)", "Not applicable", "Obtained through webpage inspection", "Not generated by the Random Forest",
     "Used as a conditioning input"),
    ("URL Presence (has_url)", "Processes the available SMS text", "Identifies whether an embedded URL is present", "Not an RF output",
     "Determines whether FiLM modulation is applied; an identity transformation (γ = 1, β = 0) is used when no URL is present"),
    ("FiLM Parameters (γ, β)", "Provides the convolutional feature maps to be modulated", "Not applicable",
     "Supplies URL-related conditioning outputs", "Generates 64 scale (γ = 1 + tanh(δ), between 0 and 2) and 64 shift (β) parameters"),
    ("Final Classification", "Provides the textual representations",
     "Provides the 37 extracted URL structural features through the Random Forest",
     "Supplies conditioning outputs rather than directly determining the final SMS classification",
     "Uses the modulated CNN representations to classify the SMS message as smishing or legitimate"),
]

CORRECTIONS = [
    ("Table 1.4, False Negative", 'The template says "Negative cases incorrectly predicted as negative". A false negative is a POSITIVE case predicted as negative.'),
    ("Table 1.2, KNN-RF-ETC voting", "The model routes by link (Jain et al., 2022; src/unified/voting_baseline.py): only messages WITHOUT a link go to the TF-IDF + SMOTE + KNN/RF/Extra Trees vote; messages WITH a link are classified from the link string alone by a Logistic Regression on CountVectorizer tokens. The template mentions only the text route."),
    ("Table 1.2, Proposed", "has_url is not an input of the generator: the generator reads P, u, page_missing and the 64 convolution averages; has_url only switches FiLM off (γ = 1, β = 0) for messages without a link."),
    ("Table 1.2, Concatenation and CNN-only", "page_missing and has_url are not Random Forest outputs (only P and u are). The CNN-only branch also log-scales long-tailed counts and adds missing-value flags (src/features/model/raw_cnn.py)."),
    ("Appendix 4, page_missing", 'The template says "0 = reached or no URL". In the data and the code a message with no URL has page_missing = 1 (all 888 no-link messages): 1 = not reached or no URL; 0 = reached.'),
    ("Appendix 4, u", "u is the standard deviation of the 200 trees' probabilities (src/features/model/rf_extractor.p_and_u), so its range is 0–0.5, not 0–1."),
    ("Appendix 4, domain_in_title", "Inverted in the template. The lab follows the public dataset's rule: 1 when the domain is NOT in the title, 0 when it is (src/lejet/features/html_features.py)."),
    ("Appendix 4, domain_with_copyright", "Inverted in the template. 1 = a copyright/trademark mark exists and the domain is NOT within 50 characters of it; 0 = no mark, or the domain is next to it."),
    ("Appendix 4, safe_anchor", 'Unsafe anchors are links containing "#", "javascript" or "mailto" (not links to another domain), and the value is a percentage 0–100 (docs/hannousse_87_features.md).'),
    ("Appendix 4, links_in_tags / ratio_intMedia / ratio_extMedia", "Percentages 0–100 (the public dataset's scale), not ratios 0–1. ratio_intHyperlinks is the only 0–1 ratio among the page features."),
    ("Appendix 4, missing values", "−1 means not measured: every webpage feature is −1 when the page was not reached, the domain features are −1 when the RDAP lookup failed, and nb_redirection is −1 when the redirect chain was not recorded. nb_redirection reaches 9 in the data (eight followed hops plus an app-store hand-off)."),
    ("Appendix 5, CNN Feature Representations", "The generator does not receive the 64 pooled activations: it reads the average of each convolution map over all positions, taken before activation (src/features/model/film_cnn.py, conv_average). The classifier reads the 64 max-pooled activations after FiLM."),
]
CHECKED = ("Checked against the code on 2026-11-22: has_url, page_missing, P, u, nb_redirection, ratio_intHyperlinks, links_in_tags, "
           "ratio_intMedia, ratio_extMedia, safe_anchor, domain_in_title, domain_with_copyright, whois_registered_domain, "
           "domain_registration_length, domain_age, page_rank, and the model descriptions. The other Appendix 4 rows are as the template "
           "gives them (not re-checked here).")


# ------------------------------------------------------------------ numbers
def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    tp, tn = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 0)).sum())
    fp, fn = int(((y == 0) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "N": tp + tn + fp + fn, "Accuracy": (tp + tn) / (tp + tn + fp + fn),
            "Precision": precision, "Recall": recall, "F1-score": f1}


def load_predictions(step5: Path) -> dict:
    out = {}
    for paper, inner, stem in MODELS:
        d = pd.read_csv(step5 / f"{stem}_Predictions.csv")
        assert set(d["Model"]) == {inner}, f"{stem}: unexpected model name {set(d['Model'])}"
        d["y"] = (d["Actual Label"] == "smishing").astype(int)
        d["pred"] = (d["Predicted Label"] == "smishing").astype(int)
        out[paper] = d
    ids = {p: d.groupby("Seed")["Message ID"].apply(frozenset).to_dict() for p, d in out.items()}
    first = next(iter(ids.values()))
    assert all(v == first for v in ids.values()), "the four models were not tested on the same messages"
    return out


def fmt(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def seeds_text(seeds: list) -> str:
    """[42] -> "seed 42"; [123, 2024] -> "seeds 123 and 2024"."""
    seeds = [str(x) for x in seeds]
    return f"seed {seeds[0]}" if len(seeds) == 1 else "seeds " + ", ".join(seeds[:-1]) + f" and {seeds[-1]}"


def p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def provenance(dataset: Path, step5: Path, preds: dict, fingerprint: str) -> list:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(git("status", "--porcelain"))
    side_path = fm.checkpoint_path(dataset, "film_scratch", SEEDS[0], 0).with_suffix(".json")
    options = json.loads(side_path.read_text(encoding="utf-8")).get("options", {}) if side_path.exists() else {}
    reads = fm.out_dir(dataset, "compare") / "test_reads.log"
    read_lines = reads.read_text(encoding="utf-8").splitlines() if reads.exists() else []
    n_msgs = preds[PROPOSED].query("Seed == @SEEDS[0]")["Message ID"].nunique()
    return [
        f"- Generated: {datetime.datetime.now().isoformat(timespec='minutes')} by `scripts/build_experiment_papers.py`",
        f"- Code: git commit `{commit}`" + (" (with uncommitted changes in the working tree)" if dirty else ""),
        f"- Predictions: `{step5.relative_to(ROOT).as_posix()}/*_Predictions.csv`, fingerprint `{fingerprint}`",
        f"- Dataset: `{dataset.relative_to(ROOT).as_posix()}`, {n_msgs:,} messages; 5-fold grouped cross-validation x 3 seeds "
        f"({', '.join(map(str, SEEDS))}); every message is a test message once per seed",
        f"- Proposed model settings (sidecar of seed {SEEDS[0]} fold 0): " + (", ".join(f"{k} = {v}" for k, v in options.items()) or "not found"),
        f"- Test-fold reads logged so far: {len(read_lines)} lines in `compare/test_reads.log`"
        + (f" (last: {read_lines[-1][:19]})" if read_lines else ""),
    ]


# ------------------------------------------------------------------ the document
def build(preds: dict, prov: list, step10_xlsx: Path) -> tuple[str, dict]:
    per_seed = {(m, s): scores(g["y"].to_numpy(), g["pred"].to_numpy()) for m, d in preds.items() for s, g in d.groupby("Seed")}
    overall = {m: {k: (np.mean([per_seed[(m, s)][k] for s in SEEDS]), np.std([per_seed[(m, s)][k] for s in SEEDS], ddof=1))
                   for k in METRICS} for m, _, _ in MODELS}
    rank = {m: i + 1 for i, m in enumerate(sorted(overall, key=lambda m: -overall[m]["F1-score"][0]))}
    out = ["# Experiment Papers (Appendix 1, 2, 4, 5)", "", *prov, "",
           "Smishing is the positive class. Per seed, every metric is computed on all test messages of that seed; mean ± SD is over the "
           "three seeds (sample SD, n − 1).", ""]

    # ---- Appendix 1
    out += ["## Appendix 1. MODEL PERFORMANCE LOG", "", "**Table 1.1** Experiment Paper for Classification Performance per Seed", ""]
    out.append(table(["Model", "Seed"] + METRICS,
                     [[m, s] + [f"{per_seed[(m, s)][k]:.4f}" for k in METRICS] for m, _, _ in MODELS for s in SEEDS]))
    out += ["", "**Table 1.2** Experiment Paper for Overall Classification Performance", ""]
    out.append(table(["Model", "How the link evidence is used"] + METRICS + ["Rank (F1)"],
                     [[m, HOW_LINK_IS_USED[m]] + [fmt(*overall[m][k]) for k in METRICS] + [rank[m]] for m, _, _ in MODELS]))

    out += ["", "**Table 1.3** Experiment Paper for F1-score by Message Condition (mean ± SD over the three seeds)", ""]
    cond_rows = []
    any_model = preds[PROPOSED]
    for label, key in CONDITIONS + [("All messages", None)]:
        n = (any_model.query("Seed == @SEEDS[0]") if key is None
             else any_model[(any_model.Seed == SEEDS[0]) & (any_model["Message Condition"] == key)]).shape[0]
        cells = []
        for m, _, _ in MODELS:
            d = preds[m] if key is None else preds[m][preds[m]["Message Condition"] == key]
            f1s = [scores(g["y"].to_numpy(), g["pred"].to_numpy())["F1-score"] for _, g in d.groupby("Seed")]
            cells.append(fmt(np.mean(f1s), np.std(f1s, ddof=1)))
        cond_rows.append([label, f"{n:,}"] + cells)
    out.append(table(["Message condition", "No. of test messages (per seed)"] + [m for m, _, _ in MODELS], cond_rows))
    n_nolink_smish = int(any_model[(any_model.Seed == SEEDS[0]) & (any_model["Message Condition"] == "no URL")]["y"].sum())
    out += ["", f"Note: only {n_nolink_smish} of the no-link messages are smishing, so the no-link F1 rests on few positives."]

    out += ["", "**Table 1.4** Definition and Identification of Confusion Matrix Components", ""]
    out.append(table(["Component", "Definition", "Application in the Study"], [
        ["True Positive (TP)", "Correctly predicted positive cases.", "Smishing messages correctly classified as smishing."],
        ["True Negative (TN)", "Correctly predicted negative cases.", "Legitimate messages correctly classified as legitimate."],
        ["False Positive (FP)", "Negative cases incorrectly predicted as positive.", "Legitimate messages incorrectly classified as smishing."],
        ["False Negative (FN)", "Positive cases incorrectly predicted as negative.", "Smishing messages incorrectly classified as legitimate."]]))

    out += ["", "**Table 1.5** Experiment Paper for Confusion Matrix Results (the three seeds added; "
            f"{sum(per_seed[(PROPOSED, s)]['N'] for s in SEEDS):,} predictions per model)", ""]
    out.append(table(["Model", "True positive", "True negative", "False positive", "False negative"],
                     [[m] + [f"{sum(per_seed[(m, s)][k] for s in SEEDS):,}" for k in ("TP", "TN", "FP", "FN")] for m, _, _ in MODELS]))
    out += ["", "Per seed:", ""]
    out.append(table(["Model", "Seed", "TP", "TN", "FP", "FN"],
                     [[m, s] + [f"{per_seed[(m, s)][k]:,}" for k in ("TP", "TN", "FP", "FN")] for m, _, _ in MODELS for s in SEEDS]))

    # ---- Appendix 2
    tests = []
    for m, _, _ in MODELS[:3]:
        for s in SEEDS:
            a = preds[PROPOSED][preds[PROPOSED].Seed == s].set_index("Message ID")
            o = preds[m][preds[m].Seed == s].set_index("Message ID").loc[a.index]
            assert (a["y"] == o["y"]).all(), f"{m} seed {s}: the true labels differ"
            ok_a, ok_o = (a["pred"] == a["y"]).to_numpy(), (o["pred"] == o["y"]).to_numpy()
            b, c = int((ok_a & ~ok_o).sum()), int((~ok_a & ok_o).sum())
            stat, p = mcnemar(b, c)
            tests.append({"baseline": m, "seed": s, "b": b, "c": c, "chi2": stat, "p": p})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p"].to_numpy())
    if step10_xlsx.exists():                       # same numbers as the Step 10 workbook
        ref = pd.read_excel(step10_xlsx, sheet_name="Summary")
        bcol = [c for c in ref.columns if c.startswith("b (")][0]
        ccol = [c for c in ref.columns if c.startswith("c (")][0]
        assert sorted(zip(ref[bcol], ref[ccol])) == sorted(zip(tests.b, tests.c)), "b and c differ from step10/McNemar_Test_Results.xlsx"

    out += ["", "## Appendix 2. STATISTICAL TEST LOG", "",
            "**Table 2.1** Experiment Paper for Statistical Significance Using McNemar's Test", "",
            "b = messages the Proposed model classified correctly and the baseline incorrectly; c = the reverse; same messages, same seed. "
            f"χ² = (|b − c| − 1)² / (b + c), 1 degree of freedom; α = {ALPHA}.", ""]
    rows = []
    for t in tests.itertuples():
        decision = "Reject H₀" if t.p < ALPHA else "Fail to reject H₀"
        if t.p < ALPHA:
            decision += " (Proposed better)" if t.b > t.c else " (baseline better)"
        rows.append([f"Proposed vs {t.baseline}", t.seed, f"{t.b:,}", f"{t.c:,}", f"{t.chi2:.3f}", p_text(t.p), decision])
    out.append(table(["Model", "Seed", "b", "c", "McNemar statistic (χ²)", "p-value", "Decision"], rows))

    out += ["", "**Table 2.2** Experiment Paper for Summary of Hypothesis Testing", "",
            "H₀: the Proposed model and the baseline have the same error rate on the same messages. Decision rule used here: H₀ is "
            "rejected for a comparison when it is rejected in all three seeds; otherwise the seeds where it is rejected are named.", ""]
    rows, summary = [], {}
    for m, _, _ in MODELS[:3]:
        t = tests[tests.baseline == m]
        better = t[(t.p < ALPHA) & (t.b > t.c)].seed.tolist()
        worse = t[(t.p < ALPHA) & (t.b < t.c)].seed.tolist()
        holm_better = int(((t.p_holm < ALPHA) & (t.b > t.c)).sum())
        if len(better) == len(SEEDS):
            decision = "Reject H₀ (all 3 seeds)"
            meaning = f"The Proposed model makes significantly fewer errors than {m} in every seed."
        elif better:
            decision = f"Reject H₀ in {seeds_text(better)} only ({len(better)} of 3)"
            rest = [s for s in SEEDS if s not in better]
            lead = ["Proposed" if r.b > r.c else ("baseline" if r.b < r.c else "tie") for r in t[t.seed.isin(rest)].itertuples()]
            meaning = (f"Significantly better than {m} in {seeds_text(better)}; in {seeds_text(rest)} the difference is not "
                       f"significant (more correct among the disagreements: {', '.join(lead)}).")
        elif worse:
            decision = "Fail to reject H₀ in the Proposed model's favour"
            meaning = f"{m} is significantly better in {seeds_text(worse)}."
        else:
            decision = "Fail to reject H₀ (all 3 seeds)"
            meaning = f"No significant difference from {m} in any seed."
        meaning += f" With a Holm correction over the 9 tests: {holm_better} of 3 seeds significant in the Proposed model's favour."
        rows.append([f"Proposed vs {m}", f"{len(better)}", decision, meaning])
        summary[m] = len(better)
    out.append(table(["Model comparison", "Seeds significant (of 3)", "Decision on H₀", "Interpretation"], rows))
    out += ["", "Seeds significant = seeds where p < α and the Proposed model is the more accurate one. The multiple-comparison "
            "correction (none, Holm or Bonferroni) is still the adviser's decision (guide item 39)."]

    # ---- Appendix 4 and 5
    out += ["", "## Appendix 4. FEATURE OPERATIONALIZATION", ""]
    out.append(table(["Feature group", "Feature", "Operational definition", "Encoding"], [list(r) for r in APPENDIX_4]))
    out += ["", "## Appendix 5. FEATURE-TO-MODEL MAPPING", ""]
    out.append(table(["Feature Group", "CNN Text Component", "URL Feature Component", "Random Forest Component",
                      "FiLM Generator and Classification"], [list(r) for r in APPENDIX_5]))

    out += ["", "## Corrections to the template text", "",
            "The fixed text of the template was checked against the code and the data; these rows differ from the template:", ""]
    out += [f"{i}. **{where}.** {what}" for i, (where, what) in enumerate(CORRECTIONS, 1)]
    out += ["", CHECKED, ""]
    headline = {m: round(float(overall[m]["F1-score"][0]), 4) for m, _, _ in MODELS}
    return "\n".join(out), {"f1": headline, "seeds_significant": summary}


def write_index(folder: Path) -> None:
    rows = []
    for path in sorted(folder.glob("Experiment_Papers_*.md")):
        meta = re.search(r"<!-- papers (\{.*\}) -->", path.read_text(encoding="utf-8"))
        info = json.loads(meta.group(1)) if meta else {}
        f1 = info.get("f1", {})
        sig = info.get("seeds_significant", {})
        rows.append([f"[{path.name}]({path.name})", info.get("generated", ""), info.get("fingerprint", ""), info.get("commit", ""),
                     " / ".join(f"{v:.4f}" for v in f1.values()), " / ".join(str(v) for v in sig.values())])
    text = ["# Experiment papers: one copy per retraining", "",
            "Written by `scripts/build_experiment_papers.py` (called at the end of `scripts/build_step10_files.py`). A new copy appears when the "
            "Step 5 predictions change; older copies are never overwritten.", "",
            "F1 order: " + " / ".join(m for m, _, _ in MODELS) + ". Seeds significant (McNemar, Proposed better) order: "
            + " / ".join(m for m, _, _ in MODELS[:3]) + ".", "",
            table(["File", "Generated", "Predictions fingerprint", "Commit", "Mean F1", "Seeds significant (of 3)"], rows), ""]
    (folder / "index.md").write_text("\n".join(text), encoding="utf-8")


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="write a new copy even if one exists for the same predictions")
    a = ap.parse_args(argv)
    dataset = tc.dataset_path(False)
    step5 = fm.out_dir(dataset, "step5")
    files = [step5 / f"{stem}_Predictions.csv" for _, _, stem in MODELS]
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.read_bytes())
    fingerprint = digest.hexdigest()[:10]
    folder = fm.out_dir(dataset, "experiment_papers")
    folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(folder.glob(f"Experiment_Papers_*_{fingerprint}.md"))
    if existing and not a.force:
        print(f"these predictions already have a copy: {existing[-1]} (nothing written; --force writes another)")
        write_index(folder)
        return 0
    preds = load_predictions(step5)
    prov = provenance(dataset, step5, preds, fingerprint)
    text, headline = build(preds, prov, fm.out_dir(dataset, "step10") / "McNemar_Test_Results.xlsx")
    now = datetime.datetime.now()
    commit = re.search(r"git commit `([^`]+)`", "\n".join(prov))
    meta = {"generated": now.isoformat(timespec="minutes"), "fingerprint": fingerprint, "commit": commit.group(1) if commit else "",
            **headline}
    path = folder / f"Experiment_Papers_{now:%Y%m%d-%H%M}_{fingerprint}.md"
    path.write_text(f"<!-- papers {json.dumps(meta, ensure_ascii=False)} -->\n" + text, encoding="utf-8")
    write_index(folder)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
