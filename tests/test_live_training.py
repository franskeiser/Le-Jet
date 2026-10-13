"""Tests for src/features/model/live_training.py: rows built from lab-inspected links for the fusion head.

The two rules that matter: only `live_train` rows may be used (fresh_test never reaches training), and the head must
see the same link features at training time as the serving code computes. Tiny random forests stand in for the
extractors, so no trained model is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.link_features import LINK_TEXT_FEATURES, extract_links, link_text_features, normalize_link
from src.features.model import live_training as lt
from src.features.model.fusion_data import RECIPE_COLUMNS
from src.features.model.rf_extractor import VARIANTS, fit_rf


def live_frame():
    rows = []
    for i in range(12):
        fetched = i % 3 != 0
        smishing = i % 2 == 0
        host = f"site{i}.example{i}.{'xyz' if smishing else 'com'}"
        row = {"link_id": f"{i:016x}", "label": "smishing" if smishing else "legitimate",
               "split": "live_train" if i < 9 else "fresh_test", "group": f"g{i}",
               "link": f"hxxps://{host.replace('.', '[.]')}/p{i}", "page_fetched": int(fetched)}
        for name in lt.PAGE_COLUMNS:
            row[name] = (i % 5) + 1 if fetched else -1
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def live_csv(tmp_path):
    path = tmp_path / "live_links.csv"
    live_frame().to_csv(path, index=False)
    return path


def tiny_extractors():
    rng = np.random.default_rng(0)
    out = {}
    for variant, features in VARIANTS.items():
        X = pd.DataFrame(rng.normal(size=(60, len(features))), columns=features)
        y = np.array(["phishing", "legitimate"] * 30)
        out[variant] = {"model": fit_rf(X, y, 42, n_estimators=10), "features": features, "variant": variant, "seed": 42}
    return out


def test_only_live_train_rows_are_used_and_fresh_test_is_refused(live_csv):
    rows = lt.build_rows(live_csv, "live_train", repeats=3)
    assert set(rows["link_id"]) <= {f"{i:016x}" for i in range(9)}                 # the 3 fresh_test links never appear
    with pytest.raises(ValueError, match="only live_train"):
        lt.build_rows(live_csv, "fresh_test")


def test_rows_are_marked_synthetic_text_and_reproducible(live_csv):
    a, b = lt.build_rows(live_csv, repeats=4, seed=1), lt.build_rows(live_csv, repeats=4, seed=1)
    assert (a["source"] == "live_synthetic_text").all()
    assert a["sms_text"].tolist() == b["sms_text"].tolist()
    assert a["sms_text"].tolist() != lt.build_rows(live_csv, repeats=4, seed=2)["sms_text"].tolist()


def test_the_link_in_each_text_is_the_link_the_features_describe(live_csv):
    rows = lt.build_rows(live_csv, repeats=3)
    assert len(rows) > 0
    for _, r in rows.iterrows():
        link = normalize_link(extract_links(r["sms_text"])[0]["raw"])
        assert {k: r[k] for k in LINK_TEXT_FEATURES} == link_text_features(link)    # same code the API runs


def test_pages_that_were_not_reached_carry_minus_one(live_csv):
    rows = lt.build_rows(live_csv, repeats=2)
    unreached = rows[rows["page_fetched"] == 0]
    assert len(unreached) > 0 and (unreached[lt.PAGE_COLUMNS] == -1).all().all()
    assert (rows[rows["page_fetched"] == 1][lt.PAGE_COLUMNS] != -1).all().all()


def test_tensor_shapes_and_alt_equals_base_when_no_page_was_reached(live_csv):
    rows = lt.build_rows(live_csv, repeats=2)
    vocab = {"<PAD>": 0, "your": 1, "account": 2}
    ids, base, alt, y = lt.build_tensors(rows, tiny_extractors()["url_only"], tiny_extractors()["full"], vocab, "rf_vec")
    n, width = len(rows), len(RECIPE_COLUMNS["rf_vec"])
    assert ids.shape[0] == base.shape[0] == alt.shape[0] == y.shape[0] == n and base.shape[1] == alt.shape[1] == width
    assert set(y.tolist()) == {0, 1} and y.dtype == torch.long
    no_page = torch.tensor((rows["page_fetched"] == 0).to_numpy())
    assert torch.equal(base[no_page], alt[no_page])                                  # nothing new to show for these rows
    assert not torch.equal(base[~no_page], alt[~no_page])                            # page evidence changes the inputs


def test_labels_follow_the_lab_ground_truth(live_csv):
    rows = lt.build_rows(live_csv, repeats=1)
    _, _, _, y = lt.build_tensors(rows, tiny_extractors()["url_only"], tiny_extractors()["full"], {"<PAD>": 0}, "rf_vec")
    assert y.tolist() == [1 if l == "smishing" else 0 for l in rows["label"]]
