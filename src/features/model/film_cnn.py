# src/features/model/film_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DENSE_UNITS, KERNEL_SIZE, NUM_FILTERS

# The three numbers the Link RF gives E2 for each message, in this order (src/unified/film_inputs.py E2_LINK).
LINK_DIM = 3

# How gamma is made from the generator's output (decision D2). "free" is the default: gamma = 1 + delta, unbounded, as in the
# FiLM paper. "bounded" is the artifact's version, gamma = 1 + tanh(delta), between 0 and 2; it is kept as an ablation.
GAMMA_MODES = ("free", "bounded")


class FiLMTextCNN(nn.Module):
    """E2: a text CNN whose filters are scaled and shifted per message by the link evidence (docs/film_architecture_plan.md).

    text -> frozen embedding -> Conv1d (64 filters, width 3) -> x            (batch, filters, positions)
    generator: [P, u, page_missing] + the Conv average of x -> (delta_gamma, beta), one pair per filter
    FiLM:      x' = gamma * x + beta, applied before ReLU at every position
    x' -> ReLU -> max-pool -> dropout 0.2 -> Dense 64 -> ReLU -> dropout 0.5 -> Output 2

    The head sees only the 64 text numbers; the link reaches the verdict only through gamma and beta. The generator's weights
    and bias start at zero, so at the start gamma = 1 and beta = 0 and the model is the plain text CNN it was built from.
    `embedding` and `conv` are named like TextEncoder's, so a trained text CNN's weights load into them directly.

    Options (all default to the plan's choice):
      gamma_mode          "free": gamma = 1 + delta (D2 default); "bounded": gamma = 1 + tanh(delta), between 0 and 2.
      mask_padding        False: the Conv average is over all positions (D5 default). True: only over positions whose
                          3-word window holds at least one real word, so the average does not depend on message length.
      average_after_relu  False: the Conv average is taken before ReLU (D5 default). True: over ReLU(x), so negative
                          scores cannot cancel positive ones.
    """

    def __init__(self, embedding_layer: nn.Embedding, num_filters: int = NUM_FILTERS, kernel_size: int = KERNEL_SIZE,
                 link_dim: int = LINK_DIM, dense_units: int = DENSE_UNITS, gamma_mode: str = "free",
                 mask_padding: bool = False, average_after_relu: bool = False):
        super().__init__()
        if gamma_mode not in GAMMA_MODES:
            raise ValueError(f"gamma_mode must be one of {GAMMA_MODES}, got {gamma_mode!r}")
        if mask_padding and embedding_layer.padding_idx is None:
            raise ValueError("mask_padding needs an embedding layer with a padding_idx")
        self.num_filters, self.kernel_size, self.link_dim = num_filters, kernel_size, link_dim
        self.gamma_mode, self.mask_padding, self.average_after_relu = gamma_mode, mask_padding, average_after_relu

        # Frozen for every model of the study (decided 2026-10-18): it is counted separately from the trainable parameters.
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = False
        self.conv = nn.Conv1d(embedding_layer.embedding_dim, num_filters, kernel_size=kernel_size)

        # One Dense layer: (link numbers + one average per filter) -> a delta_gamma and a beta per filter.
        self.generator = nn.Linear(link_dim + num_filters, 2 * num_filters)
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

        self.dropout_pool = nn.Dropout(0.2)
        self.dense = nn.Linear(num_filters, dense_units)
        self.dropout_head = nn.Dropout(0.5)
        self.output = nn.Linear(dense_units, 2)

    def conv_average(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        # The mean of each filter's scores over the positions (batch, filters): what the generator reads about the text.
        scores = F.relu(x) if self.average_after_relu else x
        if not self.mask_padding:
            return scores.mean(dim=2)
        # A position (window of kernel_size tokens) counts if any of its tokens is a real word, not padding.
        real = (token_ids != self.embedding.padding_idx).to(scores.dtype)                     # (batch, seq_len)
        counts = F.max_pool1d(real.unsqueeze(1), kernel_size=self.kernel_size, stride=1)       # (batch, 1, positions)
        total = counts.sum(dim=2).clamp(min=1.0)                                               # a message with no words -> 0
        return (scores * counts).sum(dim=2) / total

    def conv_features(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # token_ids: (batch, seq_len) of vocab indices. Returns x (batch, filters, positions), the Conv scores before the
        # FiLM step, and their average over the positions (batch, filters), which is what the generator reads.
        embedded = self.embedding(token_ids)           # (batch, seq_len, embedding_dim)
        x = self.conv(embedded.transpose(1, 2))        # (batch, filters, positions)
        return x, self.conv_average(x, token_ids)

    def film_parameters(self, link: torch.Tensor, average: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # link: (batch, 3) = [P, u, page_missing]. Returns gamma and beta, each (batch, filters).
        if link.dim() != 2 or link.shape[1] != self.link_dim or link.shape[0] != average.shape[0]:
            raise ValueError(f"link must be (batch, {self.link_dim}) with the same batch as the text, got {tuple(link.shape)}")
        delta_gamma, beta = self.generator(torch.cat([link, average], dim=1)).chunk(2, dim=1)
        gamma = 1.0 + (torch.tanh(delta_gamma) if self.gamma_mode == "bounded" else delta_gamma)
        return gamma, beta

    def encode(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # The pooled text numbers (batch, filters), after the link has adjusted the filters.
        # text_keep: optional (batch,) of 1 / 0. A 0 hides that message's text numbers (modality dropout, training only).
        x, average = self.conv_features(token_ids)
        gamma, beta = self.film_parameters(link, average)
        x = gamma.unsqueeze(2) * x + beta.unsqueeze(2)                 # FiLM, before ReLU, at every position
        pooled = F.adaptive_max_pool1d(F.relu(x), output_size=1).squeeze(-1)
        if text_keep is not None:
            pooled = pooled * text_keep.to(pooled.dtype).unsqueeze(1)
        return self.dropout_pool(pooled)

    def forward(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # Raw logits (batch, 2); no Softmax here, as in TextEncoder (the loss applies it).
        features = self.encode(token_ids, link, text_keep)
        return self.output(self.dropout_head(F.relu(self.dense(features))))

    def parameter_groups(self, weight_decay: float, generator_weight_decay: float | None = None) -> list[dict]:
        # Optimizer groups: L2 decay on Dense 64's weights, as for every model of the study, and on the generator's weights
        # (decision D13: the generator is new and the data is small; same value unless generator_weight_decay is given).
        # Biases, the Conv and the output layer get none. The frozen embedding is in no group.
        generator_decay = weight_decay if generator_weight_decay is None else generator_weight_decay
        decayed = [(self.dense.weight, weight_decay), (self.generator.weight, generator_decay)]
        decayed_ids = {id(p) for p, _ in decayed}
        plain = [p for p in self.parameters() if p.requires_grad and id(p) not in decayed_ids]
        return [{"params": [self.dense.weight], "weight_decay": weight_decay},
                {"params": [self.generator.weight], "weight_decay": generator_decay},
                {"params": plain, "weight_decay": 0.0}]
