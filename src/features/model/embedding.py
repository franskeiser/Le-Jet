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

class MmapFastTextVectors:
    """Read word vectors from a Facebook fastText .bin WITHOUT loading it into RAM.

    load_fasttext_vectors() (gensim) reads the whole input matrix into memory -- about
    7-10 GB at peak for cc.en.300.bin, more than a 16 GB laptop usually has free. This
    memory-maps the matrix and reads only the rows a word needs, so building the
    embedding matrix for ~22k words takes well under 1 GB.

    The vectors are the same as gensim's (tests/test_embedding.py compares them): the
    header, word list and subword hashing are gensim's own code, and a vector is
    gensim's average of the word's own row (if known) and its character n-gram rows.
    Offers key_to_index, vector_size and [word], like gensim's FastTextKeyedVectors.
    """

    def __init__(self, bin_path: str, encoding: str = 'utf-8'):
        from gensim.models import _fasttext_bin as fb

        with open(bin_path, 'rb') as fin:
            magic, version = fb._struct_unpack(fin, '@2i')
            new_format = magic == fb._FASTTEXT_FILEFORMAT_MAGIC
            spec = fb._NEW_HEADER_FORMAT if new_format else fb._OLD_HEADER_FORMAT
            header = {name: fb._struct_unpack(fin, fmt)[0] for name, fmt in spec}
            if not new_format:
                header['dim'] = magic
            raw_vocab, _, nwords, _ = fb._load_vocab(fin, new_format, encoding=encoding)
            if new_format:
                fb._struct_unpack(fin, '@?')  # quant_input flag
            rows, dim = fb._struct_unpack(fin, '@2q')
            offset = fin.tell()

        self.key_to_index = {word: i for i, word in enumerate(raw_vocab)}
        self.vector_size = dim
        self.min_n, self.max_n, self.bucket = header['minn'], header['maxn'], header['bucket']
        self._nwords = len(self.key_to_index)
        assert rows == self._nwords + self.bucket, 'unexpected input-matrix shape'
        self._matrix = np.memmap(bin_path, dtype=np.float32, mode='r', offset=offset, shape=(rows, dim))

    def __getitem__(self, word: str) -> np.ndarray:
        from gensim.models.fasttext import ft_ngram_hashes

        rows = [self._nwords + h for h in ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket)]
        if word in self.key_to_index:
            rows.insert(0, self.key_to_index[word])
        if not rows:
            return np.zeros(self.vector_size, dtype=np.float32)  # gensim does the same
        return np.asarray(self._matrix[rows], dtype=np.float32).mean(axis=0)

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