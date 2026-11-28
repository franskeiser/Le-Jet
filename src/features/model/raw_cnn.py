# src/features/model/raw_cnn.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DENSE_UNITS, KERNEL_SIZE, NUM_FILTERS
from src.features.model import conv_blocks
from src.unified.prepare_web_37 import FEATURES_37, PAGE

N_FEATURES = 37                  # the URL / page features the Link RF reads (prepare_web_37.FEATURES_37)
RAW_WIDTH = N_FEATURES + 2       # what the model is GIVEN: [37 raw features, page_missing, has_url] (unchanged input contract)
CLIP = 5.0                       # prepared features are clipped to [-CLIP, CLIP]
MISSING = -1.0                   # the dataset's "not available" value (no negative value is real: checked on the final dataset)

# Counts and lengths with long right tails (length_url up to 1,134, nb_hyperlinks up to 1,666): log(1 + x) before standardising, so a
# few huge values do not squash the rest. Ratios (0-1 or 0-100), 0/1 flags and page_rank (0-10) are left as they are.
LOG_FEATURES = ["length_url", "length_hostname", "nb_dots", "nb_slash", "nb_www", "nb_redirection", "length_words_raw", "char_repeat",
                "shortest_word_host", "shortest_word_path", "longest_words_raw", "longest_word_host", "avg_word_host", "phish_hints",
                "nb_hyperlinks", "nb_extCSS", "domain_registration_length", "domain_age"]
# Features outside the page that can be missing on their own (WHOIS is missing for ~1,480 messages whose page DID load), each with its
# own missing flag. The 13 page features are missing exactly when the page did not load, which page_missing already says.
FLAGGED = ["nb_redirection", "whois_registered_domain", "domain_registration_length", "domain_age", "page_rank"]
LOG_INDEX = [FEATURES_37.index(f) for f in LOG_FEATURES]
FLAG_INDEX = [FEATURES_37.index(f) for f in FLAGGED]
BRANCH_WIDTH = N_FEATURES + 1 + len(FLAGGED) + 1     # what the URL branch reads: 37 prepared, page_missing, 5 missing flags, has_url = 44
assert set(LOG_FEATURES) <= set(FEATURES_37) and not set(FLAGGED) & set(PAGE)


def log_columns(x: np.ndarray | torch.Tensor):
    """log(1 + x) on the LOG_FEATURES columns of a (n, 37) array, leaving MISSING cells as they are."""
    out = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    cols = out[:, LOG_INDEX]
    real = cols != MISSING
    logged = torch.log1p(cols.clamp(min=0)) if isinstance(x, torch.Tensor) else np.log1p(np.clip(cols, 0, None))
    out[:, LOG_INDEX] = torch.where(real, logged, cols) if isinstance(x, torch.Tensor) else np.where(real, logged, cols)
    return out


def scale_raw_link(link: torch.Tensor, feature_mean: torch.Tensor, feature_std: torch.Tensor) -> torch.Tensor:
    """(batch, 44) the URL branch's input from the (batch, 39) raw link [37 raw features, page_missing, has_url]: the 37 prepared
    features (log, standardised with the train part's mean / std, -1 -> 0, clipped), page_missing, the 5 missing flags and has_url. All
    zeros for a message with no link. Shared by CNN-only and the Proposed model's URL branch, so both read exactly the same numbers."""
    has_url = link[:, N_FEATURES + 1:N_FEATURES + 2]
    raw = torch.where(has_url == 1, link[:, :N_FEATURES], torch.zeros_like(link[:, :N_FEATURES]))   # a no-link NaN never travels
    missing = raw == MISSING
    scaled = ((log_columns(raw) - feature_mean) / feature_std).clamp(-CLIP, CLIP)
    scaled = torch.where(missing, torch.zeros_like(scaled), scaled)
    flags = missing[:, FLAG_INDEX].to(scaled.dtype)
    out = torch.cat([scaled, link[:, N_FEATURES:N_FEATURES + 1], flags, has_url], dim=1)
    return torch.where(has_url == 1, out, torch.zeros_like(out))


class RawFeatureCNN(nn.Module):
    """The "CNN-only" baseline of the study (guide Step 4: SMS text + the 37 URL features, no Random Forest). docs/ARCHITECTURE.md.

    A two-branch network joined before the classifier (late fusion):
      text branch   embedding (fine-tuned) -> Conv1d (64 filters, width 3) -> ReLU -> max-pool -> dropout 0.2 -> 64 numbers
                    (identical to E2's text path)
      URL branch    37 features PREPARED (below) + page_missing + 5 missing flags + has_url = 44 -> Dense 64 -> ReLU -> dropout 0.2
      head          [64 text, 64 URL] -> Dense 64 -> ReLU -> dropout 0.5 -> Output 2   (E2's head)

    Preparing the 37 features (fitted on the TRAINING part only, stored as buffers so the checkpoint is self-contained):
      1. log(1 + x) on the long-tailed counts and lengths (LOG_FEATURES);
      2. standardise with the mean and std of the REAL values only (the -1 "missing" cells are left out of the statistics);
      3. a missing cell (-1) becomes 0, i.e. "average", and the network is told it is missing by page_missing (the 13 page features)
         and one flag per FLAGGED feature; then clip to +-5.
    A message with no link (has_url = 0) gets an all-zero URL input whatever its cells hold (a NaN or -1 never reaches a layer).

    Compared with E2 it reads the same text and the same 37 features with the same head, recipe and modality dropout; the only
    differences are who digests the 37 features (a Dense layer here, the Link RF in E2) and how the link evidence is combined
    (side by side here, FiLM in E2). 68,930 trainable parameters without the embedding: conv 57,664 + URL branch 44 x 64 + 64 = 2,880
    + dense 128 x 64 + 64 = 8,256 + output 130 (E2: 70,658).
    """

    def __init__(self, embedding_layer: nn.Embedding, feature_mean: torch.Tensor | None = None, feature_std: torch.Tensor | None = None,
                 num_filters: int = NUM_FILTERS, kernel_size: int = KERNEL_SIZE, branch_units: int = NUM_FILTERS,
                 dense_units: int = DENSE_UNITS, freeze_embedding: bool = False):
        super().__init__()
        self.num_filters, self.kernel_size, self.branch_units = num_filters, kernel_size, branch_units
        self.freeze_embedding = freeze_embedding
        self.widths, self.per_width = None, None              # set by use_widths (several Conv widths)
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = not freeze_embedding
        self.conv = nn.Conv1d(embedding_layer.embedding_dim, num_filters, kernel_size=kernel_size)
        self.dropout_pool = nn.Dropout(0.2)
        # mean / std of the PREPARED (logged) real values; defaults leave the numbers unscaled
        mean = torch.zeros(N_FEATURES) if feature_mean is None else torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.ones(N_FEATURES) if feature_std is None else torch.as_tensor(feature_std, dtype=torch.float32)
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        self.link_branch = nn.Linear(BRANCH_WIDTH, branch_units)
        self.dropout_branch = nn.Dropout(0.2)
        self.dense = nn.Linear(num_filters + branch_units, dense_units)
        self.dropout_head = nn.Dropout(0.5)
        self.output = nn.Linear(dense_units, 2)

    def encode(self, token_ids: torch.Tensor, text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # The pooled text numbers (batch, filters); text_keep hides a message's text numbers (modality dropout, training only).
        x = self.conv(self.embedding(token_ids).transpose(1, 2))
        pooled = F.relu(x).max(dim=2).values
        if text_keep is not None:
            pooled = pooled * text_keep.to(pooled.dtype).unsqueeze(1)
        return self.dropout_pool(pooled)

    def scaled_link(self, link: torch.Tensor) -> torch.Tensor:
        # (batch, 44): the 37 prepared features, page_missing, the 5 missing flags, has_url. All zeros for a message with no link.
        return scale_raw_link(link, self.feature_mean, self.feature_std)

    def use_widths(self, widths, per_width: int) -> None:
        """Several Conv widths instead of one (conv_blocks.MultiWidthConv; adopted 2026-11-25: widths 3, 4, 5 x 64). Build the model
        with num_filters = len(widths) x per_width first, as raw_cnn_control.build_raw does."""
        conv_blocks.use_widths(self, widths, per_width)

    def forward(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # Raw logits (batch, 2); no Softmax here (the loss applies it). link: (batch, 39) = [37 raw features, page_missing, has_url].
        if link.dim() != 2 or link.shape[1] != RAW_WIDTH or link.shape[0] != token_ids.shape[0]:
            raise ValueError(f"link must be (batch, {RAW_WIDTH}) = 37 features, page_missing, has_url; got {tuple(link.shape)}")
        digested = self.dropout_branch(F.relu(self.link_branch(self.scaled_link(link))))
        features = torch.cat([self.encode(token_ids, text_keep), digested], dim=1)
        return self.output(self.dropout_head(F.relu(self.dense(features))))

    def parameter_groups(self, weight_decay: float, generator_weight_decay: float | None = None) -> list[dict]:
        # L2 decay on Dense 64's weights, as for every model of the study; none on the rest. (generator_weight_decay is accepted and
        # ignored: no generator.) The scaling buffers are not parameters.
        apart = {id(self.dense.weight), id(self.embedding.weight)}
        plain = [p for p in self.parameters() if p.requires_grad and id(p) not in apart]
        groups = [{"params": [self.dense.weight], "weight_decay": weight_decay},
                  {"params": plain, "weight_decay": 0.0}]
        if self.embedding.weight.requires_grad:
            groups.append({"params": [self.embedding.weight], "weight_decay": 0.0})
        return groups
