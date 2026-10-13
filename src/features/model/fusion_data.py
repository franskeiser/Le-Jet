# src/features/model/fusion_data.py
"""Turns one split of the SMS data into the three tensors the fusion model trains on.

LEGACY (two web-set extractors, superseded 2026-10-14 by src/unified/, see docs/unified_architecture.md).
Still used by src/serving and the demo until they move to the unified forest.


Per message:
    token_ids  fastText vocabulary indices for the CNN (padded to SEQ_LEN_L)
    aux        17 numbers, in AUX_COLUMNS order: the 14 link-text features, has_url, P, u
    label      0 = legitimate, 1 = smishing

P and u come from the frozen url_only RF extractor of the same seed (docs/training_rules.md
rule 2): old SMS data never gets page, WHOIS, DNS or page_rank values. Messages without a link
get -1 for the 14 features, P and u, and has_url = 0.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import LINK_FEATURES_PATH, SEQ_LEN_L
from src.data.split_dataset import ID_COLUMN, load_split_df, resolve_label_column
from src.features.link_features import LINK_TEXT_FEATURES
from src.features.model.embedding import build_torch_embedding_layer
from src.features.model.rf_extractor import extract_p_u, load_extractor
from src.features.model.rf_feature_extractor import CONTRIBUTION_COLUMNS, extract_vector
from src.features.model.train import LABEL_TO_IDX, encode_batch

ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_DIR = ROOT / "models" / "embeddings"

# Column order of the aux tensor, and therefore of url_gates in the fusion model.
AUX_COLUMNS = LINK_TEXT_FEATURES + ["has_url", "P", "u"]

# What the head receives besides the text. In every recipe the RF is the only place the link
# features are read from, except link_pu which also passes the 14 link features through raw.
#   link_pu  14 raw link features + has_url + P + u            (the design built first)
#   pu_only  has_url + P + u                                     (link features only inside the RF)
#   rf_vec   has_url + P + u + 31 per-feature RF contributions   (the RF as a feature extractor)
RECIPE_COLUMNS = {
    "link_pu": AUX_COLUMNS,
    "pu_only": ["has_url", "P", "u"],
    "rf_vec": ["has_url", "P", "u"] + CONTRIBUTION_COLUMNS,
}


def load_vocab(embeddings_dir: Path = EMBEDDINGS_DIR) -> dict[str, int]:
    with open(embeddings_dir / "vocab.json", encoding="utf-8") as f:
        return {word: i for i, word in enumerate(json.load(f))}


def load_embedding_layer(embeddings_dir: Path = EMBEDDINGS_DIR):
    word_to_idx = load_vocab(embeddings_dir)
    matrix = np.load(embeddings_dir / "embedding_matrix.npy")
    return build_torch_embedding_layer(matrix, word_to_idx), word_to_idx


def aux_from_inputs(inputs: pd.DataFrame, extractor_output: pd.DataFrame, recipe: str) -> pd.DataFrame:
    """Assemble a recipe's columns from the link features (`inputs`) and an extractor's output."""
    frame = pd.concat([inputs[LINK_TEXT_FEATURES + ["has_url"]], extractor_output], axis=1)
    return frame[RECIPE_COLUMNS[recipe]]


def build_aux_frame(df: pd.DataFrame, link_features: pd.DataFrame, extractor: dict, recipe: str = "link_pu") -> pd.DataFrame:
    """The recipe's table for the rows of `df` (same order). `df` needs message_id. Old SMS data
    has no page evidence, so `extractor` is the url_only one."""
    link_cols = [ID_COLUMN, "has_url"] + LINK_TEXT_FEATURES
    merged = df[[ID_COLUMN]].merge(link_features[link_cols], on=ID_COLUMN, how="left", validate="one_to_one")
    assert merged[LINK_TEXT_FEATURES + ["has_url"]].notna().all().all(), "a message has no link-feature row"
    output = extract_vector(extractor, merged) if recipe == "rf_vec" else extract_p_u(extractor, merged)
    return aux_from_inputs(merged, output, recipe)


def build_split_inputs(
    df: pd.DataFrame, link_features: pd.DataFrame, extractor: dict,
    word_to_idx: dict[str, int], seq_len: int = SEQ_LEN_L, recipe: str = "link_pu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_col = resolve_label_column(df)
    token_ids = encode_batch(df["sms_text"].astype(str).tolist(), word_to_idx, seq_len)
    aux = build_aux_frame(df, link_features, extractor, recipe)
    aux_tensor = torch.tensor(aux.values, dtype=torch.float32)
    labels = torch.tensor([LABEL_TO_IDX[label] for label in df[label_col]], dtype=torch.long)
    return token_ids, aux_tensor, labels


def load_seed_inputs(seed: int, split_name: str, df: pd.DataFrame | None = None, recipe: str = "link_pu"):
    """Tensors for one split of one seed. Pass `df` to use a sub-slice of the split (for
    example the validation part of Training); redacted rows are dropped by load_split_df."""
    if df is None:
        df = load_split_df(seed, split_name)
    link_features = pd.read_csv(ROOT / LINK_FEATURES_PATH, keep_default_na=False)
    extractor = load_extractor("url_only", seed)
    word_to_idx = load_vocab()
    return build_split_inputs(df, link_features, extractor, word_to_idx, recipe=recipe)


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights, n / (2 * count_c), so both classes carry equal total loss.
    docs/training_rules.md rule 1, option 1 (keeps every row)."""
    counts = torch.bincount(labels, minlength=2).float()
    assert (counts > 0).all(), "a class is missing from the training labels"
    return counts.sum() / (2.0 * counts)


def encode_batch_tolerant(texts: list[str], word_to_idx: dict[str, int], seq_len: int = SEQ_LEN_L) -> torch.Tensor:
    """Like train.encode_batch, but words outside the vocabulary are dropped instead of
    raising. Needed for fresh messages, which contain words the vocabulary never saw."""
    from src.features.text_preprocessing import pad_or_truncate, text_tokens

    pad = "<PAD>"
    rows = []
    for text in texts:
        tokens = [t for t in text_tokens(text) if t in word_to_idx and t != pad]
        rows.append([word_to_idx[t] for t in pad_or_truncate(tokens, seq_len)])
    return torch.tensor(rows, dtype=torch.long)
