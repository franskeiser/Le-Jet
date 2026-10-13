"""Training rows for the fusion head built from the lab-inspected links (data/live/live_links.csv, split `live_train`).

Why: the real SMS data has no page evidence, so the head only ever trained on P and u from the url_only extractor.
Links the lab really opened give pages whose evidence (full extractor) can be shown to the head next to a text.
Only `live_train` rows are used here; `fresh_test` never reaches training, gating or tuning (checked below).

The text around each link is SYNTHETIC (templates in src/demo/synthetic.py: class-specific wording half of the
time, ambiguous wording the other half), so these rows are marked non-real and the headline scores stay on real
messages (docs/training_rules.md). The CNN (Phase A) and the gates never see these rows: only Phase C's head does.

Each row has two versions of the head's non-text inputs: `base` (link spelling only, url_only extractor) and `alt`
(full extractor where the page was reached, else url_only), exactly what live routing produces. MixedAuxDataset
draws one of them at random per read.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch

from src.config import SEQ_LEN_L
from src.demo.synthetic import AMBIGUOUS_LINK, LEGIT_LINK, SMISH_LINK, fill, url_matches_text
from src.features.link_features import LINK_TEXT_FEATURES, link_text_features, normalize_link
from src.features.model.fusion_data import aux_from_inputs, encode_batch_tolerant
from src.features.model.rf_extractor import PAGE_FETCHED, VARIANTS
from src.features.model.rf_feature_extractor import extract_vector, route_vector

PAGE_COLUMNS = VARIANTS["full"][len(LINK_TEXT_FEATURES):]
LABEL_TO_IDX = {"legitimate": 0, "smishing": 1}


def refang(text: str) -> str:
    return str(text).replace("[.]", ".").replace("hxxp", "http")


def build_rows(live_csv: Path, split: str = "live_train", repeats: int = 8, seed: int = 42) -> pd.DataFrame:
    """One row per (link, template). Refuses to build from fresh_test."""
    if split != "live_train":
        raise ValueError("only live_train may be used for training")
    live = pd.read_csv(live_csv, keep_default_na=False)
    live = live[live["split"] == split]
    rng = random.Random(seed)
    rows = []
    for _, r in live.iterrows():
        url = refang(r["link"])
        smishing = r["label"] == "smishing"
        fetched = int(r["page_fetched"]) == 1
        for _ in range(repeats):
            if rng.random() < 0.5:
                pool = SMISH_LINK if smishing else LEGIT_LINK
            else:
                pool = AMBIGUOUS_LINK
            text = fill(rng.choice(pool), rng, url)
            if not url_matches_text(text, url):
                continue                                    # the serving code would read a different link: skip
            row = {"sms_text": text, "label": r["label"], "link_id": r["link_id"], "group": r["group"],
                   "has_url": 1, PAGE_FETCHED: int(fetched), "source": "live_synthetic_text",
                   **link_text_features(normalize_link(url))}
            for name in PAGE_COLUMNS:
                row[name] = r[name] if fetched else -1
            rows.append(row)
    return pd.DataFrame(rows)


def build_tensors(df: pd.DataFrame, url_only: dict, full: dict, word_to_idx: dict[str, int], recipe: str = "rf_vec"):
    """(token_ids, aux_base, aux_alt, labels) for the rows of `build_rows`."""
    inputs = df[LINK_TEXT_FEATURES + PAGE_COLUMNS + ["has_url", PAGE_FETCHED]]
    base = aux_from_inputs(inputs, extract_vector(url_only, inputs), recipe)
    alt = aux_from_inputs(inputs, route_vector(url_only, full, inputs), recipe)
    ids = encode_batch_tolerant(df["sms_text"].tolist(), word_to_idx, SEQ_LEN_L)
    y = torch.tensor(df["label"].map(LABEL_TO_IDX).to_numpy(), dtype=torch.long)
    return (ids, torch.tensor(base.to_numpy(dtype="float32")), torch.tensor(alt.to_numpy(dtype="float32")), y)
