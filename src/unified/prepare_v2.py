"""Prepare the raw v2 SMS corpus for the text path: strip link text, split 70/10/10/10, build embeddings.

Why (docs/progress.md, section 3): only 483 legitimate v2 messages carry a link or the <URL> placeholder against
7,419 smishing ones, so a link token alone predicts smishing about 94% of the time. The text CNN must not see
the link, so every link and every <URL> placeholder is removed from the text, in both classes.

Writes
    data/processed/v2_text.csv          message_id, sms_text (links removed), label, link_status, link_domain
    data/splits_v2/seed_<s>/{train,val,gate,test}_ids.txt   70/10/10/10 per seed
    models/embeddings_v2/               vocab + fastText matrix over every v2 message (no labels used)

Splits: grouped by link domain (a domain never spans two parts); messages without a link are grouped by their
stripped text, so exact duplicates after stripping stay together. Groups of 100 messages or more always go to
train (three scam domains hold 2,095 link messages). The val part is early stopping for the CNN; the gate part is
the held-out slice that fits the CNN-side static gates; test is the text-only check.

Usage:
    python -m src.unified.prepare_v2 [--build-embeddings]
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SEEDS
from src.data.build_sms_link_features import message_features
from src.data.split_dataset import save_ids
from src.features.link_features import extract_links

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "sms" / "v2_2026-09-15.csv"
OUT = ROOT / "data" / "processed" / "v2_text.csv"
SPLITS = ROOT / "data" / "splits_v2"
EMB_OUT = ROOT / "models" / "embeddings_v2"
EMB_REPORT = ROOT / "experiments" / "embeddings_v2"
BIG_GROUP = 100
N_FOLDS = 10   # 1 fold test, 1 gate, 1 val, 7 train

PLACEHOLDER = re.compile(r"<URL>", re.IGNORECASE)
SCHEME_RESIDUE = re.compile(r"\bhttps?\s*:\s*/+\s*", re.IGNORECASE)
WWW_RESIDUE = re.compile(r"\bwww\.", re.IGNORECASE)


def strip_links(text: str) -> str:
    for link in reversed(extract_links(text)):
        text = text[:link["start"]] + " " + text[link["end"]:]
    text = PLACEHOLDER.sub(" ", text)
    text = SCHEME_RESIDUE.sub(" ", text)
    text = WWW_RESIDUE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_table() -> pd.DataFrame:
    raw = pd.read_csv(RAW).rename(columns={"sms": "sms_text"})
    raw["sms_text"] = raw["sms_text"].astype(str)
    feats = pd.DataFrame([message_features(t) for t in raw["sms_text"]])
    df = pd.concat([raw, feats[["link_status", "link_domain"]]], axis=1)
    df["message_id"] = ["v2_" + hashlib.md5(t.encode("utf-8")).hexdigest()[:12] for t in df["sms_text"]]
    assert df["message_id"].is_unique, "hash collision or duplicate raw text"
    df["stripped"] = df["sms_text"].map(strip_links)
    empty = df["stripped"].str.len() == 0
    print(f"dropped {int(empty.sum())} messages that were only a link")
    df = df[~empty].copy()
    df["sms_text"] = df["stripped"]
    return df.drop(columns="stripped").reset_index(drop=True)


def group_keys(df: pd.DataFrame) -> pd.Series:
    text_key = "text:" + df["sms_text"].str.lower().map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()[:12])
    return pd.Series(np.where(df["link_status"] == "link", "domain:" + df["link_domain"].astype(str), text_key), index=df.index)


def split_seed(df: pd.DataFrame, groups: pd.Series, seed: int) -> dict[str, pd.DataFrame]:
    big = groups.map(groups.value_counts()) >= BIG_GROUP
    rest = df[~big]
    fold = np.empty(len(rest), dtype=int)
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for i, (_, idx) in enumerate(folds.split(rest, rest["label"], groups[rest.index])):
        fold[idx] = i
    parts = {"test": rest[fold == 0], "gate": rest[fold == 1], "val": rest[fold == 2],
             "train": pd.concat([rest[fold >= 3], df[big]])}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(groups[parts[a].index]) & set(groups[parts[b].index]), f"a group spans {a} and {b}"
    return parts


def build_embeddings(df: pd.DataFrame) -> None:
    from src.features.build_embeddings import DEFAULT_PRIMARY, DEFAULT_SECONDARY, build_vocab
    from src.features.model.embedding import MmapFastTextVectors, build_embedding_matrix

    vocab, counts = build_vocab(df["sms_text"].tolist())
    print(f"{len(df)} messages, vocabulary of {len(vocab)} tokens")
    primary, secondary = MmapFastTextVectors(str(DEFAULT_PRIMARY)), MmapFastTextVectors(str(DEFAULT_SECONDARY))
    matrix, _, _ = build_embedding_matrix(vocab, primary, secondary)
    EMB_OUT.mkdir(parents=True, exist_ok=True)
    EMB_REPORT.mkdir(parents=True, exist_ok=True)
    np.save(EMB_OUT / "embedding_matrix.npy", matrix)
    (EMB_OUT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (EMB_REPORT / "coverage.json").write_text(json.dumps({"vocab_size": len(vocab), "messages": len(df), "dim": int(matrix.shape[1])}, indent=2))
    print(f"saved {EMB_OUT / 'embedding_matrix.npy'} {matrix.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-embeddings", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    df = build_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{OUT}: {len(df)} messages, {df['label'].value_counts().to_dict()}")
    print(f"messages still containing 'http' after stripping: {int(df['sms_text'].str.contains('http', case=False).sum())}")

    groups = group_keys(df)
    rows = []
    for seed in args.seeds:
        for name, part in split_seed(df, groups, seed).items():
            save_ids(part, str(SPLITS / f"seed_{seed}" / f"{name}_ids.txt"))
            rows.append({"seed": seed, "part": name, "rows": len(part), "share": round(len(part) / len(df), 3),
                         "smishing_share": round((part["label"] == "smishing").mean(), 3),
                         "link_rows": int((part["link_status"] == "link").sum())})
    report = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(report.to_string(index=False))
    report.to_csv(SPLITS / "split_report.csv", index=False)

    if args.build_embeddings:
        build_embeddings(df)


if __name__ == "__main__":
    main()
