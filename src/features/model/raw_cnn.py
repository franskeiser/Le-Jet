# src/features/model/raw_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DENSE_UNITS, KERNEL_SIZE, NUM_FILTERS

N_FEATURES = 37                  # the URL / page features the Link RF reads (prepare_web_37.FEATURES_37)
RAW_WIDTH = N_FEATURES + 2       # [37 raw features, page_missing, has_url]
CLIP = 5.0                       # scaled features are clipped to [-CLIP, CLIP]: one huge URL length must not drown the rest


class RawFeatureCNN(nn.Module):
    """The "CNN-only" baseline of the E2 study with the SAME inputs as E2: the SMS text AND the 37 URL / page features, in one network,
    with no Link RF (docs/film_architecture_plan.md, the panel's baseline).

    text -> embedding (fine-tuned) -> Conv1d (64 filters, width 3) -> ReLU -> max-pool -> dropout 0.2 -> 64 text numbers
    [37 features scaled, page_missing, has_url] -> Dense 64 -> ReLU -> 64 link numbers
    [64 text numbers, 64 link numbers] -> Dense 64 -> ReLU -> dropout 0.5 -> Output 2

    It answers "does the frozen Link RF add anything over a network that learns from the raw features by itself?" The inputs are the
    ones the forest reads (plus page_missing and has_url, which E2 and the control read too); only who digests the 37 features differs.
    The features are standardised inside the model with the train part's mean and std (buffers, so the checkpoint is self-contained),
    clipped to +-5, and a message with no link (has_url = 0) gets zeros for them whatever its feature cells hold (so a NaN or a -1 never
    reaches a layer). Layer names follow the other models where they exist (`embedding`, `conv`, `dense`, `output`).
    68,610 trainable parameters without the embedding: conv 57,664 + link branch 39 x 64 + 64 = 2,560 + dense 128 x 64 + 64 = 8,256
    + output 130.
    """

    def __init__(self, embedding_layer: nn.Embedding, feature_mean: torch.Tensor | None = None, feature_std: torch.Tensor | None = None,
                 num_filters: int = NUM_FILTERS, kernel_size: int = KERNEL_SIZE, branch_units: int = NUM_FILTERS,
                 dense_units: int = DENSE_UNITS, freeze_embedding: bool = False):
        super().__init__()
        self.num_filters, self.kernel_size, self.branch_units = num_filters, kernel_size, branch_units
        self.freeze_embedding = freeze_embedding
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = not freeze_embedding
        self.conv = nn.Conv1d(embedding_layer.embedding_dim, num_filters, kernel_size=kernel_size)
        self.dropout_pool = nn.Dropout(0.2)
        mean = torch.zeros(N_FEATURES) if feature_mean is None else torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.ones(N_FEATURES) if feature_std is None else torch.as_tensor(feature_std, dtype=torch.float32)
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        self.link_branch = nn.Linear(RAW_WIDTH, branch_units)
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
        # (batch, 39): the 37 features standardised and clipped, then page_missing and has_url unchanged; zeros for the features of a
        # message with no link.
        has_url = link[:, N_FEATURES + 1:N_FEATURES + 2]
        scaled = ((link[:, :N_FEATURES] - self.feature_mean) / self.feature_std).clamp(-CLIP, CLIP)
        scaled = torch.where(has_url == 1, scaled, torch.zeros_like(scaled))
        return torch.cat([scaled, link[:, N_FEATURES:]], dim=1)

    def forward(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # Raw logits (batch, 2); no Softmax here (the loss applies it). link: (batch, 39) = [37 raw features, page_missing, has_url].
        if link.dim() != 2 or link.shape[1] != RAW_WIDTH or link.shape[0] != token_ids.shape[0]:
            raise ValueError(f"link must be (batch, {RAW_WIDTH}) = 37 features, page_missing, has_url; got {tuple(link.shape)}")
        digested = F.relu(self.link_branch(self.scaled_link(link)))
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
