# src/features/model/conv_blocks.py
"""Text-CNN parts shared by the three neural models of the study, adopted 2026-11-25 after the validation-only tuning
(docs/validation_only_improvement_plan.md, Phases F and G; selected there as `PG2_ln`, `KF1_w345x64` and `CF1_w345x64`).

MultiWidthConv  several Conv1d of different widths side by side (as in Kim 2014), used as a model's `conv` layer.
ChannelNorm     layer norm across the filters at each position, used before FiLM in the Proposed model.

Every model is first built with num_filters = len(widths) x per_width (so its Dense layers and FiLM generator are sized for all the
filters) and its single Conv is then replaced (use_widths). That order is the one the validated runs used
(scripts/tune_validation.py), so the same seed gives the same starting weights and training repeats those runs exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiWidthConv(nn.Module):
    """Several Conv1d side by side: per_width filters of each width. Each width's input is right-padded with (width - smallest) zero
    vectors, which is what a padding token gives (its embedding row is 0 and messages are padded at the end), so every width yields the
    same number of positions (seq_len - smallest + 1) and the outputs stack along the filter axis: (batch, len(widths) x per_width,
    positions). Filters are ordered by width: the first per_width are the smallest width's."""

    def __init__(self, in_dim: int, widths: tuple, per_width: int):
        super().__init__()
        widths = tuple(int(w) for w in widths)
        if not widths or min(widths) < 1 or per_width < 1:
            raise ValueError(f"widths must be positive and per_width at least 1, got {widths} x {per_width}")
        self.widths, self.per_width, self.smallest = widths, int(per_width), min(widths)
        self.convs = nn.ModuleList(nn.Conv1d(in_dim, self.per_width, kernel_size=k) for k in widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([conv(F.pad(x, (0, k - self.smallest))) for k, conv in zip(self.widths, self.convs)], dim=1)


class ChannelNorm(nn.Module):
    """Layer norm across the filters at each position of a (batch, filters, positions) map. Each position is normalized on its own,
    so the many padding windows of a short SMS do not distort the real words' statistics (batch norm, which pools over positions,
    hurt: validation .9508 against .9633, Phase G). affine=False when FiLM supplies the scale and shift."""

    def __init__(self, channels: int, affine: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def use_widths(model: nn.Module, widths, per_width: int) -> None:
    """Replace model.conv (a single Conv1d, the model built with num_filters = len(widths) x per_width) by a MultiWidthConv and record
    model.widths / model.per_width. Draws the new Conv weights from torch's generator, as the validated runs did."""
    widths = tuple(int(w) for w in widths)
    if model.num_filters != len(widths) * per_width:
        raise ValueError(f"build the model with num_filters = {len(widths)} x {per_width} = {len(widths) * per_width} before "
                         f"use_widths, got {model.num_filters}")
    if not isinstance(model.conv, nn.Conv1d):
        raise ValueError("use_widths replaces the model's single Conv1d; call it once, before any other change to `conv`")
    model.conv = MultiWidthConv(model.embedding.embedding_dim, widths, per_width)
    model.widths, model.per_width = widths, int(per_width)


def filter_widths(model: nn.Module) -> torch.Tensor:
    """(num_filters,) the window width of every filter, in the order of the Conv output."""
    widths = getattr(model, "widths", None)
    if not widths:
        return torch.full((model.num_filters,), int(model.kernel_size), dtype=torch.long)
    return torch.tensor(widths, dtype=torch.long).repeat_interleave(model.per_width)


def conv_parameters(embedding_dim: int, widths, per_width: int) -> int:
    """Trainable parameters of a MultiWidthConv (weights and biases)."""
    return sum(embedding_dim * int(k) * per_width + per_width for k in widths)
