"""A fold's inputs for the E2 study: the text as token ids, the labels and the link numbers, in one place.

PartData and part_inputs() are shared by film.py (E2) and cnn_baseline.py (the text-only baseline), so both read exactly the same
rows in the same order. Train rows carry out-of-fold link numbers; validation and test rows carry the final forest's
(film_inputs.link_inputs does the lining up and the checks).
"""
from __future__ import annotations

from typing import Dict, NamedTuple

import numpy as np
import torch

from src.data.split_crowd import load_fold
from src.features.model.train import LABEL_TO_IDX
from src.unified import film_inputs as fi
from src.unified import text_cnn as tc


class PartData(NamedTuple):
    """One part (train, val or test) of a fold, ready for the model. Rows are in the text rows' order."""

    message_id: np.ndarray
    token_ids: torch.Tensor        # (n, seq_len)
    labels: torch.Tensor           # (n,) 1 = smishing
    link_full: torch.Tensor        # (n, 4) P, u, page_missing, has_url: the page as inspected (train rows out-of-fold)
    link_blank: torch.Tensor       # (n, 4) the same with the page hidden (page_missing = 1); has_url is the same in both


def part_inputs(seed: int, fold: int, part: str, word_to_idx: Dict[str, int], seq_len: int, pilot: bool = False) -> PartData:
    rows = load_fold(seed, fold, pilot=pilot)[part]
    token_ids, labels = tc.tensors(rows, word_to_idx, seq_len)
    link = fi.link_inputs(seed, fold, pilot=pilot)[part]
    assert LABEL_TO_IDX["smishing"] == 1 and torch.equal(labels, torch.tensor(link.label)), "labels disagree with the link rows"
    return PartData(message_id=link.message_id, token_ids=token_ids, labels=labels,
                    link_full=torch.tensor(link.stack4("full")), link_blank=torch.tensor(link.stack4("blank")))
