# src/features/model/embedding.py
import numpy as np
from gensim.models.fasttext import load_facebook_vectors
import torch
import torch.nn as nn

PAD_TOKEN = '<PAD>'

def load_fasttext_vectors(bin_path: str):
    # Facebook .bin format specifically — see Section 1 above for why .vec doesn't work here.
    return load_facebook_vectors(bin_path)

def build_vocab(tokenized_messages: list[list[str]]) -> list[str]:
    seen = {}
    for tokens in tokenized_messages:
        for t in tokens:
            seen.setdefault(t, None)
    return list(seen.keys())

def resolve_vector(word: str, kv_primary, kv_secondary) -> tuple[np.ndarray, str]:
    # Section 2.3's priority order: a word genuinely known to the primary
    # language (real dictionary membership, not a subword guess) wins first;
    # then the secondary language's real membership; only if neither truly
    # knows it do we fall back to the primary language's spelling-based guess.
    if word in kv_primary.key_to_index:
        return kv_primary[word], 'primary'
    if word in kv_secondary.key_to_index:
        return kv_secondary[word], 'secondary'
    return kv_primary[word], 'primary_subword_guess'

def build_embedding_matrix(
    vocab: list[str], kv_primary, kv_secondary,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(vocab), kv_primary.vector_size), dtype=np.float32)
    source_counts = {'primary': 0, 'secondary': 0, 'primary_subword_guess': 0}
    for word, idx in word_to_idx.items():
        if word == PAD_TOKEN:
            continue  # left as the zero vector — torch's padding_idx expects this, see Section 6
        vec, source = resolve_vector(word, kv_primary, kv_secondary)
        matrix[idx] = vec
        source_counts[source] += 1
    return matrix, word_to_idx, source_counts

def oov_coverage_report(vocab: list[str], kv_primary, kv_secondary) -> dict:
    # source_counts from build_embedding_matrix already answers this, but this
    # standalone version is handy for a quick check before building the full matrix.
    counts = {'primary': 0, 'secondary': 0, 'primary_subword_guess': 0}
    for w in vocab:
        if w in kv_primary.key_to_index:
            counts['primary'] += 1
        elif w in kv_secondary.key_to_index:
            counts['secondary'] += 1
        else:
            counts['primary_subword_guess'] += 1
    total = len(vocab)
    return {
        'vocab_size': total,
        **counts,
        'primary_rate': round(counts['primary'] / total, 4) if total else 0.0,
        'secondary_rate': round(counts['secondary'] / total, 4) if total else 0.0,
        'subword_guess_rate': round(counts['primary_subword_guess'] / total, 4) if total else 0.0,
    }

def build_torch_embedding_layer(
       matrix: np.ndarray, word_to_idx: dict[str, int], pad_token: str = PAD_TOKEN,
   ) -> nn.Embedding:
       vocab_size, embedding_dim = matrix.shape
       pad_idx = word_to_idx[pad_token]

       layer = nn.Embedding(
           num_embeddings=vocab_size,
           embedding_dim=embedding_dim,
           padding_idx=pad_idx,
       )
       layer.weight.data.copy_(torch.from_numpy(matrix))
       return layer