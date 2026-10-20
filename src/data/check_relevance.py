r"""Audit data/processed/crowd_relevant_synthetic.csv: is the text about its link, and can a model cheat?

Prints a report and writes <dataset>.audit.json. Nothing here changes the dataset.

1. RELEVANCE   per pair type and class: how many rows share a brand with their link, share a theme, are generic (compatible, nothing
               shared), or keep their own real link; and the number of CONTRADICTIONS (text names one brand, link another). Those must be 0
               among written rows. Re-derived here from the text and link alone, not read from the build's own bookkeeping.
2. SHORTCUTS   what each part of the data can do ON ITS OWN, by 5-fold cross-validation grouped by link domain (a domain never sits in
               train and test). AUC 0.5 = no information.
                 source labels only   (feature source, text source, has_url, page_fetched): should be near 0.5 if both classes come
                                      from every source in similar numbers
                 text only            TF-IDF + logistic regression on the words with the link cut out
                 text only, neutral   the same on the rows whose wording is the shared neutral pool: must be near 0.5, or the "neutral"
                                      wording is not neutral
                 link only            random forest on the 37 features + page_missing
                 link only, per source  the same inside one feature source (the genuine link signal, free of source differences)
                 text + link          the two probabilities averaged (shows the two carry different evidence)
3. VARIETY     distinct templates, repeated links, the largest share of one wording, domain counts.
4. SAMPLES     random rows of every pair type, to read.

Usage:
    python -m src.data.check_relevance [--data PATH] [--samples 5]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.data import build_crowd_dataset as bcd
from src.data import relevance_lexicon as lex
from src.features.link_features import extract_links
from src.unified.prepare_web_37 import FEATURES_37

DEFAULT = Path(__file__).resolve().parents[2] / "data" / "processed" / "crowd_relevant_synthetic.csv"
NEUTRAL_FRAGMENTS = ["check this", "link you asked for", "details here", "document is ready", "follow-up on your request", "kindly review this",
                     "more information is available", "fyi, details", "confirm receipt", "request has been updated", "about your inquiry",
                     "check this out", "sending you the information", "review before end of day", "pakitingnan po ito", "eto na yung link",
                     "paki-check po ng details", "para sa inquiry", "tingnan mo 'to", "ito po yung hinihingi", "paki-review po"]


def cut_links(text: str) -> str:
    for link in reversed(extract_links(text)):
        text = text[: link["start"]] + " " + text[link["end"]:]
    return " ".join(text.split())


def relevance_table(df: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    m = df.merge(meta[["message_id", "pair_type", "text_source", "feature_source", "link_source"]], on="message_id")
    linked = m[m["has_url"] == 1].copy()
    rel = []
    for r in linked.itertuples():
        words = lex.strip_links(r.sms_text)                                  # tags are read from the words, never from the link inside them
        tt, lt = lex.tag_text(words), lex.tag_url(r.link)
        genuine = r.label == "legitimate" or r.pair_type == "S3"                   # a clean link: the brand must be the link's own
        rel.append("own_link" if r.link_source == "isda_own" else lex.relation(tt, lt, words, genuine))
    linked["relation"] = rel
    table = linked.groupby(["pair_type", "label", "relation"]).size().unstack(fill_value=0)
    contradictions = int((linked["relation"] == "conflict").sum())
    return table, contradictions


def auc_cv(X, y, groups, make_model, n_splits=5) -> float:
    p = np.zeros(len(y))
    for train, test in GroupKFold(n_splits).split(X, y, groups):
        model = make_model().fit(X[train] if not hasattr(X, "iloc") else X.iloc[train], y[train])
        p[test] = model.predict_proba(X[test] if not hasattr(X, "iloc") else X.iloc[test])[:, 1]
    return float(roc_auc_score(y, p)), p


def shortcut_report(df: pd.DataFrame, meta: pd.DataFrame) -> dict:
    m = df.merge(meta[["message_id", "pair_type", "text_source", "feature_source"]], on="message_id")
    y = (m["label"] == "smishing").astype(int).to_numpy()
    groups = m["link_domain"].where(m["has_url"] == 1, m["message_id"]).to_numpy()
    out: dict = {}
    m["crawler"] = m["feature_source"].replace({"lean_borrowed": "live_lean"})          # a borrowed page IS a lean-crawled page
    onehot = pd.get_dummies(m[["crawler", "text_source"]]).assign(has_url=m["has_url"], page_fetched=m["page_fetched"]).astype(float)
    out["source_labels_only"], _ = auc_cv(onehot.to_numpy(), y, groups, lambda: LogisticRegression(max_iter=1000))

    m["text_nolink"] = m["sms_text"].map(cut_links)
    vec = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    from sklearn.pipeline import make_pipeline
    text_model = lambda: make_pipeline(vec(), LogisticRegression(max_iter=2000, C=3.0))
    texts = m["text_nolink"].to_numpy()
    out["text_only"], p_text = auc_cv(texts, y, groups, text_model)
    by_source = {}
    for source in sorted(m["text_source"].unique()):
        idx = (m["text_source"] == source).to_numpy()
        by_source[source] = {"rows": int(idx.sum()), "smishing_share": round(float(y[idx].mean()), 3), "auc": round(float(roc_auc_score(y[idx], p_text[idx])), 4)}
    out["text_only_by_text_source"] = by_source
    by_pair = {}
    for pair in sorted(m["pair_type"].unique()):
        idx = (m["pair_type"] == pair).to_numpy()
        by_pair[pair] = {"rows": int(idx.sum()), "mean_p_smishing": round(float(p_text[idx].mean()), 3)}
    out["text_only_mean_p_by_pair_type"] = by_pair
    neutral = m["text_nolink"].str.lower().map(lambda t: any(f in t for f in NEUTRAL_FRAGMENTS)).to_numpy() & (m["has_url"] == 1).to_numpy()
    out["neutral_rows"] = int(neutral.sum())
    out["neutral_rows_smishing_share"] = round(float(y[neutral].mean()), 3) if neutral.any() else None
    if neutral.sum() > 100 and 0 < y[neutral].mean() < 1:
        out["text_only_on_neutral_rows"], _ = auc_cv(texts[neutral], y[neutral], groups[neutral], text_model)

    link = m["has_url"] == 1
    feats = m.loc[link, FEATURES_37].astype(float).assign(page_missing=1 - m.loc[link, "page_fetched"]).reset_index(drop=True)
    yl, gl = y[link.to_numpy()], groups[link.to_numpy()]
    forest = lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=-1)
    out["link_only"], p_link = auc_cv(feats, yl, gl, forest)
    per_source = {}
    fs = m.loc[link, "crawler"].reset_index(drop=True)
    for source in sorted(fs.unique()):
        idx = (fs == source).to_numpy()
        if idx.sum() > 150 and 0.1 < yl[idx].mean() < 0.9:
            f = feats[idx].reset_index(drop=True)
            per_source[source] = {"rows": int(idx.sum()), "smishing_share": round(float(yl[idx].mean()), 3),
                                  "auc": round(auc_cv(f, yl[idx], gl[idx], forest)[0], 4)}
        else:
            per_source[source] = {"rows": int(idx.sum()), "smishing_share": round(float(yl[idx].mean()), 3), "auc": None}
    out["link_only_per_source"] = per_source
    p_text_linked = p_text[link.to_numpy()]
    out["text_plus_link"] = float(roc_auc_score(yl, (p_text_linked + p_link) / 2))
    out["text_only_on_linked_rows"] = float(roc_auc_score(yl, p_text_linked))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def brand_coverage(df: pd.DataFrame, meta: pd.DataFrame, brands=("smart", "maya", "gcash", "globe", "dito", "bdo", "bpi")) -> dict:
    """For texts that name a brand: how many carry a link that names it too (a look-alike counts), by class."""
    m = df.merge(meta[["message_id", "text_brands", "link_brands", "pair_type"]], on="message_id")
    out = {}
    for b in brands:
        named = m[m["text_brands"].str.split("|").map(lambda x: b in x)]
        row = {}
        for label, g in named.groupby("label"):
            row[label] = {"texts_naming_it": len(g), "with_link": int((g["has_url"] == 1).sum()),
                          "link_names_it": int(g["link_brands"].str.split("|").map(lambda x: b in x).sum())}
        out[b] = row
    return out


def variety_report(df: pd.DataFrame, meta: pd.DataFrame) -> dict:
    linked = df[df["has_url"] == 1]
    t = df["template_id"].value_counts()
    return {"rows": len(df), "templates": int(t.size), "largest_template_share": round(float(t.iloc[0] / len(df)), 4),
            "rows_in_templates_used_5plus_times": int(t[t >= 5].sum()),
            "repeated_links": int(linked["link"].duplicated().sum()), "domains": int(linked["link_domain"].nunique()),
            "max_rows_per_domain_per_class": int(linked.groupby(["link_domain", "label"]).size().max()),
            "exact_duplicate_texts": int(df["sms_text"].duplicated().sum()),
            "language": df["language"].value_counts().to_dict(), "category": df["category"].value_counts().to_dict()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT)
    ap.add_argument("--samples", type=int, default=5)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.data, keep_default_na=False)
    meta = pd.read_csv(a.data.with_suffix(".meta.csv"), keep_default_na=False)
    assert list(df["message_id"]) == list(meta["message_id"])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 150)

    print(f"{len(df)} rows: {df['label'].value_counts().to_dict()}, with a link {int(df['has_url'].sum())}")
    table, contradictions = relevance_table(df, meta)
    print("\n1. RELEVANCE (rows with a link)\n", table.to_string())
    print(f"   contradictions (text names one brand, link another): {contradictions} (real ISDA rows keep their own link and are counted separately)")
    print("\n2. SHORTCUTS (AUC by 5-fold CV grouped by link domain; 0.5 = nothing)")
    shortcuts = shortcut_report(df, meta)
    for k, v in shortcuts.items():
        print(f"   {k}: {v}")
    coverage = brand_coverage(df, meta)
    print("\n   brand coverage (texts naming the brand -> how many have a link naming it too):")
    for b, row in coverage.items():
        print(f"     {b}: {row}")
    print("\n3. VARIETY")
    variety = variety_report(df, meta)
    for k, v in variety.items():
        print(f"   {k}: {v}")
    print("\n4. SAMPLES")
    m = df.merge(meta[["message_id", "pair_type", "relation"]], on="message_id")
    rng = np.random.default_rng(0)
    for pair_type, g in m.groupby("pair_type"):
        print(f"\n  [{pair_type}] ({len(g)} rows)")
        for r in g.iloc[rng.permutation(len(g))[: a.samples]].itertuples():
            print(f"   {r.label[:5]:5s} {r.relation:8s} | {r.sms_text[:170]}")
    a.data.with_suffix(".audit.json").write_text(json.dumps({"contradictions": contradictions, "shortcuts": shortcuts, "variety": variety,
                                                              "brand_coverage": coverage, "relevance": table.reset_index().to_dict("records")}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
