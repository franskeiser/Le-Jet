"""Plumbing tests for the fresh-set validator and evaluator.

The fake fresh set is built from the PUBLIC web set, which the extractors were trained on, so
the scores printed by evaluate() on it mean nothing; these tests only check the code paths.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.fresh_set import EXTRACTOR_INPUTS, load_fresh, validate
from src.features.model.evaluate_fresh import evaluate, liveness
from src.features.model.run_experiment import DEFAULT_MODEL_DIR


@pytest.fixture(scope="module")
def fake_fresh() -> pd.DataFrame:
    w = pd.read_csv("data/processed/web_phishing_clean.csv").sample(300, random_state=0).reset_index(drop=True)
    df = pd.DataFrame({
        "message_id": [f"fresh_{i}" for i in range(len(w))],
        "sms_text": np.where(w.status == "phishing", "Your account is on hold, verify now at the link",
                             "Hi, your parcel is on its way"),
        "label": np.where(w.status == "phishing", "smishing", "legitimate"),
        "source": "fake", "link": w.url, "link_status": "link", "has_url": 1,
    })
    for c in EXTRACTOR_INPUTS:
        df[c] = w[c]
    rng = np.random.default_rng(0)
    no_link = rng.random(len(df)) < 0.3
    dead = ~no_link & (rng.random(len(df)) < 0.5)
    df.loc[no_link, EXTRACTOR_INPUTS] = -1
    df.loc[no_link, "has_url"] = 0
    df.loc[no_link, "link_status"] = "none"
    df.loc[no_link, "link"] = ""
    df.loc[dead, EXTRACTOR_INPUTS[14:]] = -1  # page, domain and external columns
    fetched = ~no_link & ~dead
    df["page_fetched"] = fetched.astype(int)
    df["fetch_status"] = np.where(fetched, "live", np.where(no_link, "not_attempted", "dead"))
    df["final_url"] = np.where(fetched, w.url, "")
    df["final_domain"] = ""
    df["n_redirects"] = np.where(fetched, 0, -1)
    df["received_at"] = "2026-11-01T00:00:00Z"
    df["visited_at"] = np.where(fetched, "2026-11-01T03:00:00Z", "")
    df["hours_to_visit"] = np.where(fetched, 3.0, np.nan)
    return df


def test_valid_file_passes(fake_fresh):
    assert validate(fake_fresh) == []


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d.drop(columns=["page_rank"]), "missing columns"),
    (lambda d: d.assign(label=d["label"].replace("smishing", "spam")), "label must be"),
    (lambda d: d.assign(url_length=d["url_length"].astype(float).where(d.index != 0)), "blank or non-numeric"),
    (lambda d: d.assign(message_id="same"), "not unique"),
    (lambda d: d.assign(fetch_status=d["fetch_status"].where(d["page_fetched"] == 0, "dead")), "requires fetch_status"),
    (lambda d: d.assign(link_status=d["link_status"].replace("none", "redacted")), "link_status"),
    (lambda d: d.assign(hours_to_visit=-1.0), "negative"),
])
def test_broken_files_are_rejected(fake_fresh, mutate, expected):
    errors = validate(mutate(fake_fresh.copy()))
    assert any(expected in e for e in errors), errors


def test_no_link_row_with_real_values_is_rejected(fake_fresh):
    d = fake_fresh.copy()
    d.loc[d["has_url"] == 0, "url_length"] = 30
    assert any("no link has extractor inputs" in e for e in validate(d))


def test_only_smishing_with_links_is_flagged(fake_fresh):
    d = fake_fresh.copy()
    legit = d["label"] == "legitimate"
    d.loc[legit, EXTRACTOR_INPUTS] = -1
    d.loc[legit, ["has_url", "page_fetched"]] = 0
    d.loc[legit, "link_status"] = "none"
    d.loc[legit, "fetch_status"] = "not_attempted"
    assert any("shortcut" in e for e in validate(d))


def test_load_fresh_roundtrip_and_liveness(fake_fresh, tmp_path):
    path = tmp_path / "fresh.csv"
    fake_fresh.to_csv(path, index=False)
    loaded = load_fresh(str(path))
    assert len(loaded) == len(fake_fresh)
    alive = liveness(loaded)
    assert set(alive["label"]) == {"smishing", "legitimate"} and (alive["share_alive"].between(0, 1)).all()


def test_evaluate_runs_end_to_end(fake_fresh):
    metrics, tests = evaluate(fake_fresh, [42], DEFAULT_MODEL_DIR)
    live = metrics[(metrics.group == "link_live") & (metrics.model == "extractor_full")].iloc[0]
    assert live["n"] == (fake_fresh["page_fetched"] == 1).sum()
    # url_only extractor must not be scored on rows without a link
    url_only_all = metrics[(metrics.group == "all") & (metrics.model == "extractor_url_only")].iloc[0]
    assert url_only_all["n"] == (fake_fresh["has_url"] == 1).sum()
    assert set(tests["group"]) == {"link_live"} and len(tests) == 4
