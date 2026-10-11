"""Build the vocabulary and the fastText embedding matrix for the CNN.

Every word gets a 300-d vector, in this order of preference:
    1. English fastText knows the word       (cc.en.300.bin, "primary")
    2. Tagalog fastText knows the word       (cc.tl.300.bin, "secondary")
    3. English fastText's guess from the word's spelling (subword pieces)
<PAD> is the zero vector. The vectors are a starting point; the CNN fine-tunes them.

The vocabulary covers every message the model can see (all splits, <URL>-placeholder
messages excluded -- docs/training_rules.md rule 1), so training and evaluation never meet
an unknown word. This uses no labels and the vectors are pretrained, so it does not leak
test information. It is built once and shared by all seeds.

Needs pretrained/cc.en.300.bin and pretrained/cc.tl.300.bin (unpacked). The files are
read straight from disk (MmapFastTextVectors), so this needs well under 1 GB of RAM.

Usage (from the project root):
    python -m src.features.build_embeddings
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASET_PATH, LINK_FEATURES_PATH
from src.data.split_dataset import ID_COLUMN, ensure_message_id
from src.features.model.embedding import PAD_TOKEN, MmapFastTextVectors, build_embedding_matrix, resolve_vector
from src.features.text_preprocessing import text_tokens

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY = ROOT / "pretrained" / "cc.en.300.bin"
DEFAULT_SECONDARY = ROOT / "pretrained" / "cc.tl.300.bin"
DEFAULT_OUT = ROOT / "models" / "embeddings"
DEFAULT_REPORT = ROOT / "experiments" / "embeddings"
URL_TOKEN = "<URL>"


def load_messages() -> list[str]:
    df = ensure_message_id(pd.read_csv(ROOT / DATASET_PATH, keep_default_na=False))
    status = pd.read_csv(ROOT / LINK_FEATURES_PATH, usecols=[ID_COLUMN, "link_status"])
    redacted = set(status.loc[status["link_status"] == "redacted", ID_COLUMN])
    return df.loc[~df[ID_COLUMN].isin(redacted), "sms_text"].astype(str).tolist()


def build_vocab(messages: list[str]) -> tuple[list[str], Counter]:
    # <PAD> = index 0 (the CNN's padding_idx), <URL> = index 1, then words by frequency
    # (ties alphabetical) so the order is the same on every machine.
    counts = Counter(w for text in messages for w in text_tokens(text))
    words = sorted((w for w in counts if w not in (PAD_TOKEN, URL_TOKEN)), key=lambda w: (-counts[w], w))
    return [PAD_TOKEN, URL_TOKEN] + words, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--secondary", type=Path, default=DEFAULT_SECONDARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    for path in (args.primary, args.secondary):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Unpack the .bin.gz first (see README).")

    messages = load_messages()
    vocab, counts = build_vocab(messages)
    print(f"{len(messages)} messages, vocabulary of {len(vocab)} tokens")

    primary = MmapFastTextVectors(str(args.primary))
    secondary = MmapFastTextVectors(str(args.secondary))
    print(f"English knows {len(primary.key_to_index):,} words, Tagalog {len(secondary.key_to_index):,}")
    matrix, word_to_idx, _ = build_embedding_matrix(vocab, primary, secondary)
    sources = ["pad" if w == PAD_TOKEN else resolve_vector(w, primary, secondary)[1] for w in vocab]

    args.out.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embedding_matrix.npy", matrix)
    (args.out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    words = pd.DataFrame({"word": vocab, "count": [counts.get(w, 0) for w in vocab], "source": sources})
    words.to_csv(args.report / "word_sources.csv", index=False, encoding="utf-8")
    real = words[words["source"] != "pad"]
    by_type = real["source"].value_counts()
    by_token = real.groupby("source")["count"].sum()
    coverage = {
        "vocab_size": len(vocab), "messages": len(messages), "dim": int(matrix.shape[1]),
        "word_types": {k: int(v) for k, v in by_type.items()},
        "word_types_share": {k: round(v / len(real), 4) for k, v in by_type.items()},
        "token_share": {k: round(v / by_token.sum(), 4) for k, v in by_token.items()},
    }
    (args.report / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    print(json.dumps(coverage, indent=2))
    guesses = real[real["source"] == "primary_subword_guess"].nlargest(20, "count")
    print("\nMost frequent words neither language knows (vector guessed from spelling):")
    print(", ".join(f"{w} ({c})" for w, c in zip(guesses["word"], guesses["count"])))
    print(f"\nSaved {args.out / 'embedding_matrix.npy'} {matrix.shape} and vocab.json")


if __name__ == "__main__":
    main()
