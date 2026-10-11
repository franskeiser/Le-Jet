# src/features/model/cnn_encoder.py
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
