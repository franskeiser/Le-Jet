# src/features/model/film_cnn.py
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DENSE_UNITS, KERNEL_SIZE, NUM_FILTERS
from src.features.model import conv_blocks
from src.features.model.raw_cnn import BRANCH_WIDTH, RAW_WIDTH, scale_raw_link

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
# How modality dropout hides a message's text (fit_model.fit passes text_keep while training). "pooled" (the original): the 64
# pooled text numbers are set to 0 AFTER FiLM, so the link's gamma and beta are hidden with them and the head sees zeros. "input"
# (validation study 2026-11-22): the words are hidden BEFORE the Conv (its input is zeroed), so the filters see no phrase but the
# link's gamma and beta still act; the head then sees what the link alone does, as the baselines' text dropout keeps their link numbers.
TEXT_DROPOUT_MODES = ("pooled", "input")


class FiLMTextCNN(nn.Module):
    """E2: a text CNN whose filters are scaled and shifted per message by the link evidence (docs/film_architecture_plan.md).

    text -> embedding (fine-tuned) -> Conv1d (64 filters, width 3) -> x      (batch, filters, positions)
    generator: [P, u, page_missing] + the Conv average of x -> (delta_gamma, beta), one pair per filter
    FiLM:      x' = gamma * x + beta, applied before ReLU at every position
    x' -> ReLU -> max-pool -> dropout 0.2 -> Dense 64 -> ReLU -> dropout 0.5 -> Output 2

    No link (diagram, 2026-10-20): has_url, a (batch,) of 1 / 0, either passed on its own or as the LAST column of `link`
    ([P, u, page_missing, has_url], width link_dim + 1; the data pipeline does this). A message with 0 skips the generator:
    gamma = 1 and beta = 0 for it whatever its link numbers hold (they may be NaN), so it gets exactly the plain text CNN's numbers
    and the generator receives no gradient from it. Leaving has_url out (and passing a width-3 link) means every message has a
    link, as before. bypass_no_link=False ignores has_url altogether: the generator reads the link numbers of every message, a
    no-link message included (the ablation that shows what the bypass adds).

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
      bypass_no_link      True: a message with has_url = 0 gets gamma = 1, beta = 0 (default). False: has_url is ignored.
      use_average         True: the generator reads the link numbers AND the Conv average of the text (default, D4). False: the
                          link numbers alone (variant V3, step 5.3): the same link then gives the same gamma and beta to every
                          message, whatever its words; the generator has 64 fewer inputs (-8,192 parameters).
      freeze_embedding    False: the embedding is fine-tuned during training, as in the diagram (D6, changed 2026-10-20).
                          True: it stays at its fastText values (the frozen-embedding check run). The layer you pass in is
                          used as given and its requires_grad is set here, so do not share one layer between two models.
      text_dropout        "pooled" (default): text_keep = 0 zeroes the pooled numbers after FiLM. "input": it zeroes the Conv's input
                          instead, so the link's gamma and beta still reach the head (TEXT_DROPOUT_MODES).
      generator_hidden    0 (default): the generator is one Dense layer. > 0: Dense (that many units) -> ReLU -> Dense; only the last
                          layer starts at zero, so the model still starts as the plain text CNN.
      head_link           False (default): the head sees only the 64 text numbers. True: [P, u, page_missing, has_url] are also
                          attached before Dense 64, as in the concatenation control (has_url = 1 when it is not given).
      link_dim            3 (default) = [P, u, page_missing]. A larger value lets the generator read more link numbers placed after
                          those three (for example the Link RF's 37 contributions); has_url, when given, is still the last column.

    Changes made after construction (adopted 2026-11-25 for the Proposed model after the validation-only tuning, configuration
    `PG2_ln` of docs/validation_only_improvement_plan.md; all off by default, and then the model is exactly as above). Call them in
    this order, as src/unified/film.py apply_architecture does:
      use_widths(widths, per_width)  several Conv widths side by side (3, 4, 5 x 64 = 192 filters; build with num_filters = 192);
                          FiLM then scales and shifts every filter of every width.
      use_layer_norm()    layer norm across the filters at each position, before FiLM (no affine: FiLM supplies the scale and shift).
      add_url_branch(mean, std, units)  CNN-only's URL branch beside FiLM: the 37 raw URL / page features prepared as CNN-only
                          prepares them (raw_cnn.scale_raw_link, with the train part's mean / std as buffers) -> Dense (units) -> ReLU
                          -> dropout 0.2, joined to the text numbers before Dense 64. The generator still reads only P, u,
                          page_missing and the filter averages. `link` is then (batch, link_dim + 1 + 39):
                          [P, u, page_missing, has_url, 37 raw features, page_missing, has_url] (film.inputs_for builds it).
    """

    def __init__(self, embedding_layer: nn.Embedding, num_filters: int = NUM_FILTERS, kernel_size: int = KERNEL_SIZE,
                 link_dim: int = LINK_DIM, dense_units: int = DENSE_UNITS, gamma_mode: str = "bounded",
                 mask_padding: bool = False, average_after_relu: bool = False, use_beta: bool = True,
                 freeze_embedding: bool = False, bypass_no_link: bool = True, use_average: bool = True,
                 text_dropout: str = "pooled", generator_hidden: int = 0, head_link: bool = False):
        super().__init__()
        if gamma_mode not in GAMMA_MODES:
            raise ValueError(f"gamma_mode must be one of {GAMMA_MODES}, got {gamma_mode!r}")
        if text_dropout not in TEXT_DROPOUT_MODES:
            raise ValueError(f"text_dropout must be one of {TEXT_DROPOUT_MODES}, got {text_dropout!r}")
        if link_dim < LINK_DIM:
            raise ValueError(f"link_dim must be at least {LINK_DIM} (P, u, page_missing come first), got {link_dim}")
        if generator_hidden < 0:
            raise ValueError(f"generator_hidden must be 0 or more, got {generator_hidden}")
        if mask_padding and embedding_layer.padding_idx is None:
            raise ValueError("mask_padding needs an embedding layer with a padding_idx")
        self.num_filters, self.kernel_size, self.link_dim = num_filters, kernel_size, link_dim
        self.gamma_mode, self.mask_padding, self.average_after_relu = gamma_mode, mask_padding, average_after_relu
        self.use_beta, self.freeze_embedding, self.bypass_no_link = use_beta, freeze_embedding, bypass_no_link
        self.use_average = use_average
        self.text_dropout, self.generator_hidden, self.head_link = text_dropout, generator_hidden, head_link
        self.widths, self.per_width = None, None          # use_widths
        self.layer_norm = False                           # use_layer_norm
        self.url_branch_units = 0                         # add_url_branch

        # Fine-tuned for every model of the study (D6, changed 2026-10-20 from frozen). Its parameters are counted apart from
        # the 70,658 of the layers below (vocabulary size x 300 more), and it has its own optimizer group.
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = not freeze_embedding
        self.conv = nn.Conv1d(embedding_layer.embedding_dim, num_filters, kernel_size=kernel_size)

        # One Dense layer: (link numbers + one average per filter) -> a delta_gamma and a beta per filter (delta_gamma only
        # when use_beta is False).
        generator_in, generator_out = link_dim + (num_filters if use_average else 0), (2 if use_beta else 1) * num_filters
        if generator_hidden:
            self.generator = nn.Sequential(nn.Linear(generator_in, generator_hidden), nn.ReLU(),
                                           nn.Linear(generator_hidden, generator_out))
        else:
            self.generator = nn.Linear(generator_in, generator_out)
        last = self.generator_layers()[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        self.dropout_pool = nn.Dropout(0.2)
        self.dense = nn.Linear(num_filters + (LINK_DIM + 1 if head_link else 0), dense_units)
        self.dropout_head = nn.Dropout(0.5)
        self.output = nn.Linear(dense_units, 2)

    # ------------------------------------------------------------------ changes after construction (adopted 2026-11-25)
    def use_widths(self, widths, per_width: int) -> None:
        """Several Conv widths instead of one (conv_blocks.MultiWidthConv). Build with num_filters = len(widths) x per_width first."""
        if self.mask_padding:
            raise ValueError("mask_padding assumes one Conv width; it cannot be combined with use_widths")
        if self.layer_norm or self.url_branch_units:
            raise ValueError("call use_widths before use_layer_norm and add_url_branch")
        conv_blocks.use_widths(self, widths, per_width)

    def use_layer_norm(self) -> None:
        """Layer norm across the filters at each position of the Conv output, before FiLM (no affine)."""
        if self.layer_norm or self.url_branch_units:
            raise ValueError("use_layer_norm is called once, before add_url_branch")
        self.conv = nn.Sequential(self.conv, conv_blocks.ChannelNorm(self.num_filters, affine=False))
        self.layer_norm = True

    def add_url_branch(self, feature_mean: torch.Tensor, feature_std: torch.Tensor, units: int = NUM_FILTERS) -> None:
        """CNN-only's URL branch beside FiLM (see the class notes). Dense 64 is rebuilt to read the text numbers and the branch."""
        if self.head_link:
            raise ValueError("add_url_branch cannot be combined with head_link")
        if self.url_branch_units:
            raise ValueError("the URL branch is already there")
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32).clone())
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32).clone())
        self.link_branch = nn.Linear(BRANCH_WIDTH, units)
        self.dropout_branch = nn.Dropout(0.2)
        self.dense = nn.Linear(self.num_filters + units, self.dense.out_features)
        self.url_branch_units = int(units)

    def link_width(self) -> int:
        """How many link columns forward() takes: link_dim (+ has_url) (+ the 39 raw columns with a URL branch)."""
        return self.link_dim + 1 + (RAW_WIDTH if self.url_branch_units else 0)

    def generator_layers(self) -> list:
        """The generator's Dense layers, first to last (one, or two when generator_hidden > 0)."""
        return [m for m in self.generator.modules() if isinstance(m, nn.Linear)]

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

    def conv_features(self, token_ids: torch.Tensor, words_keep: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # token_ids: (batch, seq_len) of vocab indices. Returns x (batch, filters, positions), the Conv scores before the
        # FiLM step, and their average over the positions (batch, filters), which is what the generator reads.
        # words_keep: optional (batch,) of 1 / 0; a 0 zeroes that message's Conv input (text_dropout="input").
        embedded = self.embedding(token_ids)           # (batch, seq_len, embedding_dim)
        if words_keep is not None:
            embedded = embedded * words_keep.to(embedded.dtype).view(-1, 1, 1)
        x = self.conv(embedded.transpose(1, 2))        # (batch, filters, positions)
        return x, self.conv_average(x, token_ids)

    def film_parameters(self, link: torch.Tensor, average: torch.Tensor,
                        has_url: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # link: (batch, 3) = [P, u, page_missing], or (batch, 4) with has_url as the last column. Returns gamma and beta, each
        # (batch, filters). has_url: optional (batch,) of 1 / 0. A 0 gives gamma = 1, beta = 0 (no link: the filters are unchanged).
        if self.url_branch_units and link.dim() == 2 and link.shape[1] == self.link_dim + 1 + RAW_WIDTH:
            link = link[:, : self.link_dim + 1]                        # the generator reads [P, u, page_missing, has_url] only
        width_ok = link.dim() == 2 and link.shape[1] in (self.link_dim, self.link_dim + 1)
        if not width_ok or link.shape[0] != average.shape[0]:
            raise ValueError(f"link must be (batch, {self.link_dim}) or (batch, {self.link_dim + 1}) with has_url last, with the same "
                             f"batch as the text, got {tuple(link.shape)}")
        if link.shape[1] == self.link_dim + 1:
            if has_url is None and self.bypass_no_link:
                has_url = link[:, self.link_dim]
            link = link[:, : self.link_dim]
        if not self.bypass_no_link:
            has_url = None
        linked = None
        if has_url is not None:
            if has_url.shape != (average.shape[0],) or not ((has_url == 0) | (has_url == 1)).all():
                raise ValueError(f"has_url must be (batch,) of 0 / 1 with the same batch as the text, got {tuple(has_url.shape)}")
            linked = has_url.bool().unsqueeze(1)
            link = torch.where(linked, link, torch.zeros_like(link))      # a no-link row's numbers (even NaN) never reach the generator
        generated = self.generator(torch.cat([link, average], dim=1) if self.use_average else link)
        delta_gamma, beta = generated.chunk(2, dim=1) if self.use_beta else (generated, torch.zeros_like(generated))
        gamma = 1.0 + (torch.tanh(delta_gamma) if self.gamma_mode == "bounded" else delta_gamma)
        if linked is not None:
            gamma = torch.where(linked, gamma, torch.ones_like(gamma))
            beta = torch.where(linked, beta, torch.zeros_like(beta))
        return gamma, beta

    def _pool(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None,
              has_url: torch.Tensor | None = None) -> tuple:
        # The one place the FiLM step and the pooling are done, so encode(), encode_with_positions() and explain() cannot
        # disagree. Returns (pooled, position, x, gamma, beta): the pooled numbers (batch, filters), the start of each
        # filter's winning window, the Conv scores before FiLM, and gamma and beta.
        hide_input = text_keep is not None and self.text_dropout == "input"
        x, average = self.conv_features(token_ids, text_keep if hide_input else None)
        gamma, beta = self.film_parameters(link, average, has_url)
        adjusted = gamma.unsqueeze(2) * x + beta.unsqueeze(2)          # FiLM, before ReLU, at every position
        pooled, position = F.relu(adjusted).max(dim=2)                 # the same value adaptive_max_pool1d gives
        if text_keep is not None and not hide_input:
            pooled = pooled * text_keep.to(pooled.dtype).unsqueeze(1)
        return pooled, position, x, gamma, beta

    def encode(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None,
               has_url: torch.Tensor | None = None) -> torch.Tensor:
        # The pooled text numbers (batch, filters), after the link has adjusted the filters.
        # text_keep: optional (batch,) of 1 / 0. A 0 hides that message's text numbers (modality dropout, training only).
        # has_url: optional (batch,) of 1 / 0. A 0 means no link: gamma = 1, beta = 0 for that message.
        return self.dropout_pool(self._pool(token_ids, link, text_keep, has_url)[0])

    def forward(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None,
                has_url: torch.Tensor | None = None) -> torch.Tensor:
        # Raw logits (batch, 2); no Softmax here, as in TextEncoder (the loss applies it).
        if self.url_branch_units:
            if link.dim() != 2 or link.shape[1] != self.link_width() or link.shape[0] != token_ids.shape[0]:
                raise ValueError(f"with the URL branch, link must be (batch, {self.link_width()}) = [P, u, page_missing, has_url, 37 raw "
                                 f"features, page_missing, has_url], got {tuple(link.shape)}")
            # the branch runs before the text part, the order of the validated runs (the two dropouts draw in this order)
            digested = self.dropout_branch(F.relu(self.link_branch(scale_raw_link(link[:, -RAW_WIDTH:], self.feature_mean,
                                                                                  self.feature_std))))
            return self.output(self.dropout_head(F.relu(self.dense(torch.cat([self.encode(token_ids, link, text_keep, has_url),
                                                                               digested], dim=1)))))
        features = self.encode(token_ids, link, text_keep, has_url)
        if self.head_link:
            features = torch.cat([features, self.head_numbers(link, has_url)], dim=1)
        return self.output(self.dropout_head(F.relu(self.dense(features))))

    def head_numbers(self, link: torch.Tensor, has_url: torch.Tensor | None = None) -> torch.Tensor:
        # head_link only: (batch, 4) [P, u, page_missing, has_url], what the concatenation control's head reads. has_url comes from the
        # argument, else the link's last column when it has one, else 1. A no-link row's numbers (even NaN) are replaced by
        # P = 0, u = 0, page_missing = 1, as the link lane writes them.
        if has_url is None:
            has_url = link[:, self.link_dim] if link.shape[1] == self.link_dim + 1 else torch.ones(link.shape[0], dtype=link.dtype)
        has_url = has_url.to(link.dtype)
        numbers = link[:, :LINK_DIM]
        no_link = torch.tensor([0.0, 0.0, 1.0], dtype=link.dtype).expand_as(numbers)
        numbers = torch.where(has_url.bool().unsqueeze(1), numbers, no_link)
        return torch.cat([numbers, has_url.unsqueeze(1)], dim=1)

    def encode_with_positions(self, token_ids: torch.Tensor, link: torch.Tensor,
                              has_url: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Explainability-only sibling of encode(), with the same contract as TextEncoder.encode_with_positions() plus the
        # link: returns the SAME (batch, filters) pooled numbers encode() gives, and each filter's winning window start
        # (batch, filters), so serving's explain_text.py can point back at the words. It keeps the autograd graph (the
        # explanation there is gradient x input). Never called from forward() or training.
        pooled, position = self._pool(token_ids, link, None, has_url)[:2]
        return self.dropout_pool(pooled), position

    @torch.no_grad()
    def explain(self, token_ids: torch.Tensor, link: torch.Tensor, has_url: torch.Tensor | None = None) -> "FilterExplanation":
        # What the link did to each filter of each message, for the sentence "the link raised these filters".
        # Read-only: no gradients, and it does not change what the model predicts.
        pooled, position, x, gamma, beta = self._pool(token_ids, link, None, has_url)
        plain_pooled, plain_position = F.relu(x).max(dim=2)            # the same filters with gamma = 1 and beta = 0
        widths = conv_blocks.filter_widths(self).to(position.device)                           # (filters,) each filter's window
        steps = torch.arange(int(widths.max()), device=position.device)
        index = (position.unsqueeze(-1) + steps).clamp(max=token_ids.shape[1] - 1)             # (batch, filters, widest window)
        window_ids = token_ids.unsqueeze(1).expand(-1, self.num_filters, -1).gather(2, index)
        inside = steps.unsqueeze(0) < widths.unsqueeze(1)                                      # (filters, widest): within its width
        pad = self.embedding.padding_idx
        if pad is not None and not bool(inside.all()):
            window_ids = torch.where(inside.unsqueeze(0), window_ids, torch.full_like(window_ids, pad))   # beyond a filter's width
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
        generator_weights = [layer.weight for layer in self.generator_layers()]
        apart = {id(self.dense.weight), id(self.embedding.weight)} | {id(w) for w in generator_weights}
        plain = [p for p in self.parameters() if p.requires_grad and id(p) not in apart]
        groups = [{"params": [self.dense.weight], "weight_decay": weight_decay},
                  {"params": generator_weights, "weight_decay": generator_decay},
                  {"params": plain, "weight_decay": 0.0}]
        if self.embedding.weight.requires_grad:
            groups.append({"params": [self.embedding.weight], "weight_decay": 0.0})
        return groups
