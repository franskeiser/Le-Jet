# src/features/model/film_cnn.py
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DENSE_UNITS, KERNEL_SIZE, NUM_FILTERS

# The three numbers the Link RF gives E2 for each message, in this order (src/unified/film_inputs.py E2_LINK).
LINK_DIM = 3

class FilterExplanation(NamedTuple):
    """What the link did to each filter of each message (FiLMTextCNN.explain). Every field is (batch, filters) unless noted.

    pooled            the value the model uses for the filter, after the link adjusted it
    position          where the winning 3-word window starts
    window_ids        (batch, filters, kernel) the token ids of that window
    window_has_words  False if the winning window is all padding: then the filter's strength came from the link (beta)
                      and the filter's bias, not from a phrase in the message
    gamma, beta       the link's volume and nudge for this filter
    plain_pooled      the same filter's value with no link (gamma = 1, beta = 0): what the plain text CNN would use
    plain_position    where its winning window would have been
    lift              pooled - plain_pooled: above 0 the link raised the filter, below 0 it lowered it
    """

    pooled: torch.Tensor
    position: torch.Tensor
    window_ids: torch.Tensor
    window_has_words: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    plain_pooled: torch.Tensor
    plain_position: torch.Tensor
    lift: torch.Tensor


# How gamma is made from the generator's output (decision D2, changed 2026-10-20). "bounded" is the default: gamma = 1 +
# tanh(delta), between 0 and 2. "free" is gamma = 1 + delta, unbounded, as in the FiLM paper; it is kept as an ablation.
GAMMA_MODES = ("free", "bounded")


class FiLMTextCNN(nn.Module):
    """E2: a text CNN whose filters are scaled and shifted per message by the link evidence (docs/film_architecture_plan.md).

    text -> embedding (fine-tuned) -> Conv1d (64 filters, width 3) -> x      (batch, filters, positions)
    generator: [P, u, page_missing] + the Conv average of x -> (delta_gamma, beta), one pair per filter
    FiLM:      x' = gamma * x + beta, applied before ReLU at every position
    x' -> ReLU -> max-pool -> dropout 0.2 -> Dense 64 -> ReLU -> dropout 0.5 -> Output 2

    The head sees only the 64 text numbers; the link reaches the verdict only through gamma and beta. The generator's weights
    and bias start at zero, so at the start gamma = 1 and beta = 0 and the model is the plain text CNN it was built from.
    `embedding` and `conv` are named like TextEncoder's, so a trained text CNN's weights load into them directly.

    Options (all default to the plan's choice):
      gamma_mode          "bounded": gamma = 1 + tanh(delta), between 0 and 2 (D2 default); "free": gamma = 1 + delta.
      mask_padding        False: the Conv average is over all positions (D5 default). True: only over positions whose
                          3-word window holds at least one real word, so the average does not depend on message length.
      average_after_relu  False: the Conv average is taken before ReLU (D5 default). True: over ReLU(x), so negative
                          scores cannot cancel positive ones.
      use_beta            True: the generator gives gamma and beta (D3 default). False: gamma only (variant V1), beta = 0,
                          and the generator has half the outputs (4,352 parameters instead of 8,704).
      freeze_embedding    False: the embedding is fine-tuned during training, as in the diagram (D6, changed 2026-10-20).
                          True: it stays at its fastText values (the frozen-embedding check run). The layer you pass in is
                          used as given and its requires_grad is set here, so do not share one layer between two models.
    """

    def __init__(self, embedding_layer: nn.Embedding, num_filters: int = NUM_FILTERS, kernel_size: int = KERNEL_SIZE,
                 link_dim: int = LINK_DIM, dense_units: int = DENSE_UNITS, gamma_mode: str = "bounded",
                 mask_padding: bool = False, average_after_relu: bool = False, use_beta: bool = True,
                 freeze_embedding: bool = False):
        super().__init__()
        if gamma_mode not in GAMMA_MODES:
            raise ValueError(f"gamma_mode must be one of {GAMMA_MODES}, got {gamma_mode!r}")
        if mask_padding and embedding_layer.padding_idx is None:
            raise ValueError("mask_padding needs an embedding layer with a padding_idx")
        self.num_filters, self.kernel_size, self.link_dim = num_filters, kernel_size, link_dim
        self.gamma_mode, self.mask_padding, self.average_after_relu = gamma_mode, mask_padding, average_after_relu
        self.use_beta, self.freeze_embedding = use_beta, freeze_embedding

        # Fine-tuned for every model of the study (D6, changed 2026-10-20 from frozen). Its parameters are counted apart from
        # the 70,658 of the layers below (vocabulary size x 300 more), and it has its own optimizer group.
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = not freeze_embedding
        self.conv = nn.Conv1d(embedding_layer.embedding_dim, num_filters, kernel_size=kernel_size)

        # One Dense layer: (link numbers + one average per filter) -> a delta_gamma and a beta per filter (delta_gamma only
        # when use_beta is False).
        self.generator = nn.Linear(link_dim + num_filters, (2 if use_beta else 1) * num_filters)
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
        generated = self.generator(torch.cat([link, average], dim=1))
        delta_gamma, beta = generated.chunk(2, dim=1) if self.use_beta else (generated, torch.zeros_like(generated))
        gamma = 1.0 + (torch.tanh(delta_gamma) if self.gamma_mode == "bounded" else delta_gamma)
        return gamma, beta

    def _pool(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None) -> tuple:
        # The one place the FiLM step and the pooling are done, so encode(), encode_with_positions() and explain() cannot
        # disagree. Returns (pooled, position, x, gamma, beta): the pooled numbers (batch, filters), the start of each
        # filter's winning window, the Conv scores before FiLM, and gamma and beta.
        x, average = self.conv_features(token_ids)
        gamma, beta = self.film_parameters(link, average)
        adjusted = gamma.unsqueeze(2) * x + beta.unsqueeze(2)          # FiLM, before ReLU, at every position
        pooled, position = F.relu(adjusted).max(dim=2)                 # the same value adaptive_max_pool1d gives
        if text_keep is not None:
            pooled = pooled * text_keep.to(pooled.dtype).unsqueeze(1)
        return pooled, position, x, gamma, beta

    def encode(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # The pooled text numbers (batch, filters), after the link has adjusted the filters.
        # text_keep: optional (batch,) of 1 / 0. A 0 hides that message's text numbers (modality dropout, training only).
        return self.dropout_pool(self._pool(token_ids, link, text_keep)[0])

    def forward(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # Raw logits (batch, 2); no Softmax here, as in TextEncoder (the loss applies it).
        features = self.encode(token_ids, link, text_keep)
        return self.output(self.dropout_head(F.relu(self.dense(features))))

    def encode_with_positions(self, token_ids: torch.Tensor, link: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Explainability-only sibling of encode(), with the same contract as TextEncoder.encode_with_positions() plus the
        # link: returns the SAME (batch, filters) pooled numbers encode() gives, and each filter's winning window start
        # (batch, filters), so serving's explain_text.py can point back at the words. It keeps the autograd graph (the
        # explanation there is gradient x input). Never called from forward() or training.
        pooled, position = self._pool(token_ids, link)[:2]
        return self.dropout_pool(pooled), position

    @torch.no_grad()
    def explain(self, token_ids: torch.Tensor, link: torch.Tensor) -> "FilterExplanation":
        # What the link did to each filter of each message, for the sentence "the link raised these filters".
        # Read-only: no gradients, and it does not change what the model predicts.
        pooled, position, x, gamma, beta = self._pool(token_ids, link)
        plain_pooled, plain_position = F.relu(x).max(dim=2)            # the same filters with gamma = 1 and beta = 0
        steps = torch.arange(self.kernel_size, device=position.device)
        index = (position.unsqueeze(-1) + steps).clamp(max=token_ids.shape[1] - 1)             # (batch, filters, k)
        window_ids = token_ids.unsqueeze(1).expand(-1, self.num_filters, -1).gather(2, index)
        pad = self.embedding.padding_idx
        has_words = torch.ones_like(position, dtype=torch.bool) if pad is None else (window_ids != pad).any(dim=2)
        return FilterExplanation(pooled=pooled, position=position, window_ids=window_ids, window_has_words=has_words,
                                 gamma=gamma, beta=beta, plain_pooled=plain_pooled, plain_position=plain_position,
                                 lift=pooled - plain_pooled)

    def parameter_groups(self, weight_decay: float, generator_weight_decay: float | None = None) -> list[dict]:
        # Optimizer groups: L2 decay on Dense 64's weights, as for every model of the study, and on the generator's weights
        # (decision D13: the generator is new and the data is small; same value unless generator_weight_decay is given).
        # Biases, the Conv and the output layer get none. A trainable embedding has a group of its own with no decay (so its
        # learning rate can be set apart later); a frozen one is in no group.
        generator_decay = weight_decay if generator_weight_decay is None else generator_weight_decay
        decayed = [(self.dense.weight, weight_decay), (self.generator.weight, generator_decay)]
        apart = {id(p) for p, _ in decayed} | {id(self.embedding.weight)}
        plain = [p for p in self.parameters() if p.requires_grad and id(p) not in apart]
        groups = [{"params": [self.dense.weight], "weight_decay": weight_decay},
                  {"params": [self.generator.weight], "weight_decay": generator_decay},
                  {"params": plain, "weight_decay": 0.0}]
        if self.embedding.weight.requires_grad:
            groups.append({"params": [self.embedding.weight], "weight_decay": 0.0})
        return groups
