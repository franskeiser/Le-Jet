"""The Hybrid CNN–Random Forest Model with Feature-wise Linear Modulation (FiLM) as the backend serves it: SMS text + the Link RF's numbers in, a verdict and its reasons out.

What a deployment of the FiLM model needs (the same files the SOP 2 measurement counts, src/unified/efficiency.py):
    the weights (.pt) and their sidecar (.json), the fold's vocabulary (vocab.json), and the Link RF (.joblib).
The starting fastText matrix is not loaded: the trained embedding is inside the weights.

The Link RF reads the 37 features of the study (prepare_web_37.FEATURES_37):
    20 URL features   computed here from the link's spelling (lejet.features.hannousse_url, a pure function: nothing is opened or fetched),
                      exactly as the pilot dataset was built (data/build_relevant_dataset.blank_features);
    13 page features  only from what the lab already saw (the LinkStore row of that link), all together or not at all: if the lab did
                      not reach the page, or its result lacks a column the FiLM model needs (the lab results of before 2026-10-14 have no
                      `safe_anchor`), they are -1 and page_missing = 1. That is the evaluation's "page hidden" (ll.blank_page);
     4 outside features (whois, registration length, domain age, page rank) are domain lookups, not the page: kept when the lab has
                      them, as blank_page keeps them, and -1 for a link the lab never saw.
A message with no link never reaches the forest: P = 0, u = 0, page_missing = 1, has_url = 0, and the generator is bypassed (D11).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import ARTIFACTS_ROOT
from src.features.model.embedding import PAD_TOKEN
from src.features.model.film_cnn import FiLMTextCNN
from src.features.model.fusion_data import encode_batch_tolerant
from src.features.model.rf_extractor import p_and_u
from src.features.text_preprocessing import pad_or_truncate, text_tokens
from src.serving.explain_text import explain_message
from src.unified import film
from src.unified import fit_model as fm
from src.unified import text_cnn as tc
from src.unified.prepare_web_37 import FEATURES_37, OUTSIDE, PAGE, URL_FEATURES

SMISHING = 1
DEFAULT_KIND = "film_scratch"            # "proposed" of the SOP 2 measurement (D7a)
LAB_SRC = Path(__file__).resolve().parents[1]
PAGE_FLOOR = -1.0


def _url_features(normalised: str) -> dict:
    """The 20 URL features of the study, read off the link. Imported late: the lab's code lives in src/lejet and imports as `lejet`."""
    if str(LAB_SRC) not in sys.path:
        sys.path.insert(0, str(LAB_SRC))
    from lejet.features.hannousse_url import url_features
    computed = url_features(normalised)
    return {name: float(computed[name]) for name in URL_FEATURES}


def _has(stored: dict | None, name: str) -> bool:
    """The lab's row has a real number under `name`. A column added later is empty ("" or NaN) in the older rows of the same file."""
    if not stored or name not in stored:
        return False
    try:
        return bool(np.isfinite(float(stored[name])))
    except (TypeError, ValueError):
        return False


class _BoundEncoder:
    """encode_with_positions(token_ids) for ONE message whose link numbers are fixed, the shape explain_message() asks of an encoder."""

    def __init__(self, net: FiLMTextCNN, link: torch.Tensor):
        self.net, self.link = net, link

    def encode_with_positions(self, token_ids: torch.Tensor):
        return self.net.encode_with_positions(token_ids, self.link)


class _Explainable:
    """Presents the FiLM model to explain_message() (gradient x input over the 64 pooled filter numbers) without changing the FiLM model."""

    def __init__(self, net: FiLMTextCNN, link: torch.Tensor):
        self.net = net
        self.encoder = _BoundEncoder(net, link)

    def eval(self):
        self.net.eval()

    def zero_grad(self, set_to_none: bool = True):
        self.net.zero_grad(set_to_none=set_to_none)

    def logits_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        # everything after the pooling, as FiLMTextCNN.forward does it (eval mode: both dropouts are off)
        return self.net.output(torch.relu(self.net.dense(pooled)))


class FilmModel:
    def __init__(self, dataset: Path | str, kind: str = DEFAULT_KIND, seed: int = 42, fold: int = 0):
        self.dataset = Path(dataset)
        self.kind, self.seed, self.fold = kind, seed, fold
        checkpoint = fm.checkpoint_path(self.dataset, kind, seed, fold)
        sidecar = fm.read_sidecar(checkpoint)
        vocab = json.loads((tc.embeddings_dir(self.dataset, seed, fold) / "vocab.json").read_text(encoding="utf-8"))
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        embedding = nn.Embedding(len(vocab), 300, padding_idx=self.word_to_idx[PAD_TOKEN])      # trained weights come with the checkpoint
        self.net = FiLMTextCNN(embedding, sidecar["num_filters"], sidecar["kernel_size"], sidecar["link_dim"], sidecar["dense_units"],
                               **sidecar["options"])
        self.net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.net.eval()
        self.seq_len = sidecar["seq_len_l"]
        self.sidecar = sidecar
        self.rf = film.load_link_rf(self.dataset, seed, fold)
        self.rf.n_jobs = 1
        _url_features("http://example.com/")              # loads the public-suffix list now, not during the first request

    # ------------------------------------------------------------------ what is served
    @property
    def info(self) -> dict:
        name = self.dataset.stem
        return {"model": "Hybrid CNN–Random Forest Model with Feature-wise Linear Modulation (FiLM)", "dataset": name, "kind": self.kind, "seed": self.seed, "fold": self.fold,
                "forest_trees": int(getattr(self.rf, "n_estimators", 0)), "synthetic": "synthetic" in name,
                "val_f1": self.sidecar.get("val", {}).get("val_f1")}

    @staticmethod
    def files(dataset: Path | str, kind: str = DEFAULT_KIND, seed: int = 42, fold: int = 0) -> list[Path]:
        """Everything the FiLM model needs from disk; the server starts without the FiLM model when one is missing."""
        dataset = Path(dataset)
        checkpoint = fm.checkpoint_path(dataset, kind, seed, fold)
        return [checkpoint, checkpoint.with_suffix(".json"), tc.embeddings_dir(dataset, seed, fold) / "vocab.json",
                film.link_rf_path(dataset, seed, fold)]

    @classmethod
    def available(cls, dataset: Path | str, kind: str = DEFAULT_KIND, seed: int = 42, fold: int = 0) -> bool:
        return all(p.exists() for p in cls.files(dataset, kind, seed, fold))

    # ------------------------------------------------------------------ the 37 features of one link
    @staticmethod
    def link_features(normalised: str, stored: dict | None) -> tuple[pd.DataFrame, bool]:
        """(one-row frame of FEATURES_37, whether the page evidence is in it). `stored` is the lab's row for this link, or None."""
        row = _url_features(normalised)
        fetched = bool(stored and int(stored.get("page_fetched", 0)) == 1)
        have_page = fetched and all(_has(stored, name) for name in PAGE)
        have_outside = all(_has(stored, name) for name in OUTSIDE)
        row["nb_redirection"] = float(max(int(float(stored.get("n_redirects", 0))), 0)) if have_page else 0.0
        for name in PAGE:
            row[name] = float(stored[name]) if have_page else PAGE_FLOOR
        for name in OUTSIDE:
            row[name] = float(stored[name]) if have_outside else PAGE_FLOOR
        return pd.DataFrame([row])[FEATURES_37], have_page

    @staticmethod
    def link_inputs(features: pd.DataFrame, page_used: bool, stored: dict | None) -> dict:
        """The 37 numbers the Link RF read for this link, grouped for the research page: 20 from the link's spelling, 13 from the page
        (only when the lab saw it), 4 domain lookups. -1 means the number was not available."""
        row = features.iloc[0]

        def clean(value: float):
            value = float(value)
            return int(value) if value.is_integer() else round(value, 4)

        groups = (("url", URL_FEATURES), ("page", PAGE), ("lookup", OUTSIDE))
        return {"page_used": bool(page_used), "lookups_used": all(_has(stored, name) for name in OUTSIDE),
                "features": [{"name": name, "value": clean(row[name]), "group": group} for group, names in groups for name in names]}

    def link_numbers(self, features: pd.DataFrame) -> tuple[float, float]:
        p, u = p_and_u(self.rf, features)
        return float(p[0]), float(u[0])

    # ------------------------------------------------------------------ the text
    def tokens(self, text: str) -> tuple[list[str], list[str]]:
        """(every token, the known tokens padded to the model's length)."""
        every = text_tokens(text)
        known = [t for t in every if t in self.word_to_idx and t != PAD_TOKEN]
        return every, pad_or_truncate(known, self.seq_len)

    def token_ids(self, text: str) -> torch.Tensor:
        return encode_batch_tolerant([text], self.word_to_idx, self.seq_len)

    # ------------------------------------------------------------------ the verdict
    @staticmethod
    def link_tensor(p: float, u: float, page_missing: int, has_url: int) -> torch.Tensor:
        return torch.tensor([[p, u, float(page_missing), float(has_url)]], dtype=torch.float32)

    def p_smishing(self, ids: torch.Tensor, link: torch.Tensor) -> float:
        with torch.no_grad():
            return torch.softmax(self.net(ids, link), dim=1)[0, SMISHING].item()

    def explain(self, ids: torch.Tensor, link: torch.Tensor, tokens: list[str], predicted: int, raw_text: str) -> dict:
        """The phrases behind the verdict (same method and output shape as the earlier model) and what the link did to the filters."""
        out = explain_message(_Explainable(self.net, link), ids, torch.zeros(1, 0), tokens, predicted, max_reasons=3, raw_text=raw_text)
        out["link_effect"] = self.link_effect(ids, link)
        return out

    def link_effect(self, ids: torch.Tensor, link: torch.Tensor) -> dict:
        """How far the link moved the 64 text filters of this message (gamma scales, beta shifts); all zero when there is no link."""
        e = self.net.explain(ids, link)
        gamma, beta, lift = e.gamma[0], e.beta[0], e.lift[0]
        return {"p": round(float(link[0, 0]), 4), "u": round(float(link[0, 1]), 4), "page_missing": int(link[0, 2]),
                "has_url": int(link[0, 3]), "n_filters": int(lift.numel()),
                "filters_raised": int((lift > 0.01).sum()), "filters_lowered": int((lift < -0.01).sum()),
                "mean_abs_gamma_shift": round(float((gamma - 1).abs().mean()), 4), "mean_abs_beta": round(float(beta.abs().mean()), 4),
                "net_lift": round(float(lift.sum()), 4)}
