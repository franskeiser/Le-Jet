# src/features/model/concat_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DENSE_UNITS, KERNEL_SIZE, NUM_FILTERS
from src.features.model.film_cnn import LINK_DIM


class ConcatTextCNN(nn.Module):
    """The concatenation control of the E2 study (docs/film_architecture_plan.md, D12): the text CNN with the same link numbers
    attached to its 64 text numbers before Dense 64. No generator, no FiLM.

    text -> embedding (fine-tuned) -> Conv1d (64 filters, width 3) -> ReLU -> max-pool -> dropout 0.2 -> 64 text numbers
    [64 text numbers, P, u, page_missing] -> Dense 64 -> ReLU -> dropout 0.5 -> Output 2

    It answers "is FiLM better than simply giving the head the same link numbers?": the inputs are identical to E2's (the same
    P, u and page_missing from the same Link RF); only how they reach the verdict differs. Layer names follow E2's
    (`embedding`, `conv`, `dense`, `output`), so the embedding and Conv can be warm-started from the text-only baseline the same way.
    62,146 trainable parameters without the embedding: conv 57,664 + dense (64 + 3) x 64 + 64 = 4,352 + output 130.
    """

    def __init__(self, embedding_layer: nn.Embedding, num_filters: int = NUM_FILTERS, kernel_size: int = KERNEL_SIZE,
                 link_dim: int = LINK_DIM, dense_units: int = DENSE_UNITS, freeze_embedding: bool = False):
        super().__init__()
        self.num_filters, self.kernel_size, self.link_dim = num_filters, kernel_size, link_dim
        self.freeze_embedding = freeze_embedding
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = not freeze_embedding
        self.conv = nn.Conv1d(embedding_layer.embedding_dim, num_filters, kernel_size=kernel_size)
        self.dropout_pool = nn.Dropout(0.2)
        self.dense = nn.Linear(num_filters + link_dim, dense_units)
        self.dropout_head = nn.Dropout(0.5)
        self.output = nn.Linear(dense_units, 2)

    def encode(self, token_ids: torch.Tensor, text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # The pooled text numbers (batch, filters). text_keep: optional (batch,) of 1 / 0; a 0 hides that message's text numbers
        # (modality dropout, training only), as in E2: the link numbers are not hidden by it.
        x = self.conv(self.embedding(token_ids).transpose(1, 2))
        pooled = F.relu(x).max(dim=2).values
        if text_keep is not None:
            pooled = pooled * text_keep.to(pooled.dtype).unsqueeze(1)
        return self.dropout_pool(pooled)

    def forward(self, token_ids: torch.Tensor, link: torch.Tensor, text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # Raw logits (batch, 2); no Softmax here (the loss applies it). link: (batch, 3) = [P, u, page_missing].
        if link.dim() != 2 or link.shape[1] != self.link_dim or link.shape[0] != token_ids.shape[0]:
            raise ValueError(f"link must be (batch, {self.link_dim}) with the same batch as the text, got {tuple(link.shape)}")
        features = torch.cat([self.encode(token_ids, text_keep), link], dim=1)
        return self.output(self.dropout_head(F.relu(self.dense(features))))

    def parameter_groups(self, weight_decay: float, generator_weight_decay: float | None = None) -> list[dict]:
        # L2 decay on Dense 64's weights, as for every model of the study; none on the rest. A trainable embedding has a group of
        # its own with no decay. (generator_weight_decay is accepted and ignored: no generator.)
        apart = {id(self.dense.weight), id(self.embedding.weight)}
        plain = [p for p in self.parameters() if p.requires_grad and id(p) not in apart]
        groups = [{"params": [self.dense.weight], "weight_decay": weight_decay},
                  {"params": plain, "weight_decay": 0.0}]
        if self.embedding.weight.requires_grad:
            groups.append({"params": [self.embedding.weight], "weight_decay": 0.0})
        return groups
