# src/features/model/plain_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DENSE_UNITS, KERNEL_SIZE, NUM_FILTERS


class PlainTextCNN(nn.Module):
    """The text-only baseline of the E2 study (docs/film_architecture_plan.md): the same CNN as E2 with no link and no generator.

    text -> embedding (fine-tuned) -> Conv1d (64 filters, width 3) -> ReLU -> max-pool -> dropout 0.2
         -> Dense 64 -> ReLU -> dropout 0.5 -> Output 2

    Its layers have the same names as FiLMTextCNN's (`embedding`, `conv`, `dense`, `output`), and with the generator at zero
    FiLMTextCNN computes exactly this model, so E2 can start from a trained baseline. forward() takes the link and text_keep
    arguments only so that one training loop serves both models; they are ignored (the baseline has no link and no
    modality dropout). 61,954 trainable parameters without the embedding: conv 57,664 + dense 4,160 + output 130.
    """

    def __init__(self, embedding_layer: nn.Embedding, num_filters: int = NUM_FILTERS, kernel_size: int = KERNEL_SIZE,
                 dense_units: int = DENSE_UNITS, freeze_embedding: bool = False):
        super().__init__()
        self.num_filters, self.kernel_size, self.freeze_embedding = num_filters, kernel_size, freeze_embedding
        # Fine-tuned for every model of the study (D6). The layer you pass in is used as given and the model sets its
        # requires_grad, so do not share one layer between two models.
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = not freeze_embedding
        self.conv = nn.Conv1d(embedding_layer.embedding_dim, num_filters, kernel_size=kernel_size)
        self.dropout_pool = nn.Dropout(0.2)
        self.dense = nn.Linear(num_filters, dense_units)
        self.dropout_head = nn.Dropout(0.5)
        self.output = nn.Linear(dense_units, 2)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        # The pooled text numbers (batch, filters): the same 64 numbers FiLMTextCNN gives when its generator is at zero.
        x = self.conv(self.embedding(token_ids).transpose(1, 2))       # (batch, filters, positions)
        pooled = F.relu(x).max(dim=2).values
        return self.dropout_pool(pooled)

    def forward(self, token_ids: torch.Tensor, link: torch.Tensor | None = None,
                text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # Raw logits (batch, 2); no Softmax here (the loss applies it). `link` and `text_keep` are ignored.
        return self.output(self.dropout_head(F.relu(self.dense(self.encode(token_ids)))))

    def parameter_groups(self, weight_decay: float, generator_weight_decay: float | None = None) -> list[dict]:
        # Optimizer groups: L2 decay on Dense 64's weights, as for every model of the study; none on the rest. A trainable
        # embedding has a group of its own with no decay. (generator_weight_decay is accepted and ignored: no generator.)
        apart = {id(self.dense.weight), id(self.embedding.weight)}
        plain = [p for p in self.parameters() if p.requires_grad and id(p) not in apart]
        groups = [{"params": [self.dense.weight], "weight_decay": weight_decay},
                  {"params": plain, "weight_decay": 0.0}]
        if self.embedding.weight.requires_grad:
            groups.append({"params": [self.embedding.weight], "weight_decay": 0.0})
        return groups
