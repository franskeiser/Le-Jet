# src/features/model/cnn_encoder.py
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import KERNEL_SIZE


class TextEncoder(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding, num_filters: int, kernel_size: int = KERNEL_SIZE):
        super().__init__()
        self.embedding = embedding_layer
        embedding_dim = embedding_layer.embedding_dim

        # One conv branch: num_filters filters of width kernel_size (default 64 x 3).
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size)
        self.dropout = nn.Dropout(0.2)

        # Temporary head -- Phase A only, discarded/replaced in Phase C (Task 2.4).
        self.temp_head = nn.Linear(num_filters, 2)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len) of vocab indices. Returns the pooled feature
        # vector -- this is what Task 2.3 freezes and extracts.
        embedded = self.embedding(token_ids)          # (batch, seq_len, embedding_dim)
        x = embedded.transpose(1, 2)                   # (batch, embedding_dim, seq_len) -- Section 1.1
        out = F.relu(self.conv(x))                     # (batch, num_filters, reduced_seq_len)
        pooled = F.adaptive_max_pool1d(out, output_size=1)  # (batch, num_filters, 1) -- Section 1.2
        return self.dropout(pooled.squeeze(-1))        # (batch, num_filters)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Raw logits -- no Softmax here, see Section 1.3. Used only for Phase A training;
        # Task 2.4 discards temp_head and reuses encode() directly.
        features = self.encode(token_ids)
        return self.temp_head(features)

    def encode_with_positions(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Explainability-only sibling of encode() -- returns the SAME (batch, num_filters)
        # pooled feature vector, plus each filter's winning window start position
        # (batch, num_filters), so src/serving's explain_text.py can point back at the
        # actual n-gram that "won" each filter's max-pool. Never called from forward()/
        # encode(), training (train.py), or predict_proba()'s normal inference path, so it
        # cannot change what the model predicts.
        #
        # F.adaptive_max_pool1d(out, 1) and out.max(dim=2) compute the identical value;
        # .max() just also hands back the index adaptive_max_pool1d doesn't expose.
        embedded = self.embedding(token_ids)          # (batch, seq_len, embedding_dim)
        x = embedded.transpose(1, 2)                   # (batch, embedding_dim, seq_len)
        out = F.relu(self.conv(x))                     # (batch, num_filters, reduced_seq_len)
        max_val, max_idx = out.max(dim=2)              # each (batch, num_filters)
        return self.dropout(max_val), max_idx


def _checked_keep(keep_idx, num_filters: int) -> torch.Tensor:
    keep = np.asarray(keep_idx, dtype=int)
    # Ascending and unique (what select_top_k returns): kept position i is original filter keep_idx[i],
    # which the explanations (Phase 8) rely on.
    if keep.ndim != 1 or len(keep) == 0 or (np.diff(keep) <= 0).any() or keep[0] < 0 or keep[-1] >= num_filters:
        raise ValueError(f"keep_idx must be ascending, unique ids in 0..{num_filters - 1}")
    return torch.from_numpy(keep)


def prune_input_columns(linear: nn.Linear, keep_idx, cnn_dim: int) -> nn.Linear:
    # C-prune (docs/c_prune_architecture.md 2.3): a layer reading [cnn_dim filter scores | the rest] loses the
    # columns of the removed filters; the rest (e.g. the 40 link numbers) and the bias are copied unchanged.
    keep = _checked_keep(keep_idx, cnn_dim)
    columns = torch.cat([keep, torch.arange(cnn_dim, linear.in_features)])
    pruned = nn.Linear(len(columns), linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        pruned.weight.copy_(linear.weight[:, columns])
        if linear.bias is not None:
            pruned.bias.copy_(linear.bias)
    return pruned


def prune_encoder(encoder: TextEncoder, keep_idx) -> TextEncoder:
    # C-prune (docs/c_prune_architecture.md 2.3): a new k-filter TextEncoder holding only the kept filters'
    # Conv rows and biases, and the temporary head's matching columns. The embedding is copied (frozen or not,
    # as in `encoder`), so the source model is left untouched. Dropout has no weights.
    num_filters = encoder.conv.out_channels
    keep = _checked_keep(keep_idx, num_filters)
    pruned = TextEncoder(copy.deepcopy(encoder.embedding), len(keep), encoder.conv.kernel_size[0])
    with torch.no_grad():
        pruned.conv.weight.copy_(encoder.conv.weight[keep])
        pruned.conv.bias.copy_(encoder.conv.bias[keep])
    pruned.temp_head = prune_input_columns(encoder.temp_head, keep, num_filters)
    pruned.train(encoder.training)
    return pruned
