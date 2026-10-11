"""
Tests for src/features/model/embedding.py: the memory-mapped fastText reader returns the
same vectors as gensim's full loader, for known words and for subword guesses.

Run with:  pytest tests/test_embedding.py
"""

import numpy as np
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_vectors, save_facebook_model

from src.features.model.embedding import MmapFastTextVectors, build_embedding_matrix

SENTENCES = [
    ["your", "gcash", "account", "is", "locked", "verify", "now"],
    ["pa", "share", "naman", "po", "salamat"],
    ["meeting", "link", "for", "tomorrow", "at", "nine"],
] * 20


def _tiny_bin(tmp_path) -> str:
    model = FastText(vector_size=16, min_count=1, min_n=3, max_n=5, bucket=1000, epochs=2, seed=1, workers=1)
    model.build_vocab(SENTENCES)
    model.train(SENTENCES, total_examples=len(SENTENCES), epochs=2)
    path = str(tmp_path / "tiny.bin")
    save_facebook_model(model, path)
    return path


def test_mmap_vectors_match_gensim(tmp_path):
    path = _tiny_bin(tmp_path)
    reference = load_facebook_vectors(path)
    mmap = MmapFastTextVectors(path)

    assert set(mmap.key_to_index) == set(reference.key_to_index)
    assert mmap.vector_size == reference.vector_size
    for word in ["gcash", "account", "salamat", "verifyy", "gcashh", "unseenword", "ab"]:
        np.testing.assert_allclose(mmap[word], reference[word], rtol=1e-5, atol=1e-6, err_msg=word)


def test_build_embedding_matrix_with_mmap(tmp_path):
    path = _tiny_bin(tmp_path)
    mmap = MmapFastTextVectors(path)
    vocab = ["<PAD>", "gcash", "naman", "unseenword"]
    matrix, word_to_idx, counts = build_embedding_matrix(vocab, mmap, mmap)
    assert matrix.shape == (4, 16)
    assert not matrix[word_to_idx["<PAD>"]].any()  # padding stays the zero vector
    assert counts["primary"] == 2 and counts["primary_subword_guess"] == 1
