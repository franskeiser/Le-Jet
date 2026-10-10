# src/features/model/cnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding, num_filters: int):
        super().__init__()
        self.embedding = embedding_layer
        embedding_dim = embedding_layer.embedding_dim

         # Three parallel branches, kernel sizes 3/4/5 — Kim (2014)'s multi-kernel TextCNN.
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3)
        self.conv4 = nn.Conv1d(embedding_dim, num_filters, kernel_size=4)
        self.conv5 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5)
        self.dropout = nn.Dropout(0.2)

        # Temporary head — Phase A only, discarded/replaced in Phase C (Task 2.4).
        self.temp_head = nn.Linear(num_filters * 3, 2)

    def _branch(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embedding_dim, seq_len) — already transposed, see forward()
        out = F.relu(conv(x))                              # (batch, num_filters, reduced_seq_len)
        pooled = F.adaptive_max_pool1d(out, output_size=1)  # (batch, num_filters, 1) -- Section 1.2
        return self.dropout(pooled.squeeze(-1))             # (batch, num_filters)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len) of vocab indices. Returns the pooled 3xF feature
        # vector — this is what Task 2.3 freezes and extracts, per Section 3 above.
        embedded = self.embedding(token_ids)          # (batch, seq_len, embedding_dim)
        x = embedded.transpose(1, 2)                   # (batch, embedding_dim, seq_len) -- Section 1.1
        b3 = self._branch(self.conv3, x)
        b4 = self._branch(self.conv4, x)
        b5 = self._branch(self.conv5, x)
        return torch.cat([b3, b4, b5], dim=1)          # (batch, 3*num_filters)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Raw logits -- no Softmax here, see Section 1.3. Used only for Phase A training;
        # Task 2.4 discards temp_head and reuses encode() directly.
        features = self.encode(token_ids)
        return self.temp_head(features)

    def encode_with_positions(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Explainability-only sibling of encode() -- returns the SAME (batch, 3*F) pooled
        # feature vector, but also each filter's winning window position (src/serving's
        # explain_text.py needs this to point back at the actual n-gram that "won" each
        # filter's adaptive-max-pool). Never called from forward()/encode(), training
        # (train.py), or predict_proba()'s normal inference path, so it cannot change
        # anything about what the model predicts -- purely an extra read of the same
        # conv outputs encode() already computes.
        #
        # F.adaptive_max_pool1d(out, 1) (encode()'s _branch) and out.max(dim=2) (below)
        # compute the identical value -- both are "max over the whole reduced-seq-len
        # axis" when the pooled output size is 1 -- .max() just also hands back the
        # index adaptive_max_pool1d doesn't expose.
        embedded = self.embedding(token_ids)          # (batch, seq_len, embedding_dim)
        x = embedded.transpose(1, 2)                   # (batch, embedding_dim, seq_len)
        pooled_parts = []
        positions = []
        for conv in (self.conv3, self.conv4, self.conv5):
            out = F.relu(conv(x))                       # (batch, num_filters, reduced_seq_len)
            max_val, max_idx = out.max(dim=2)            # each (batch, num_filters)
            pooled_parts.append(self.dropout(max_val))
            positions.append(max_idx)
        return torch.cat(pooled_parts, dim=1), positions