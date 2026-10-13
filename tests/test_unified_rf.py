import numpy as np
import pandas as pd

from src.unified.build_unified_dataset import COLUMNS, PAGE_COLUMNS, build
from src.unified import unified_rf as urf
from src.features.model.rf_feature_extractor import CONTRIBUTION_COLUMNS, FULL_FEATURES, path_contributions


def make_frame(n=400, seed=0, nan_pages=0.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FULL_FEATURES})
    df["url_length"] += 2 * y            # one informative feature
    df["page_rank"] += 2 * y
    if nan_pages:
        df.loc[rng.random(n) < nan_pages, [c for c in FULL_FEATURES if c not in urf.INPUTS["url_only"]]] = np.nan
    df["message_id"] = [f"m{i}" for i in range(n)]
    df["label"] = np.where(y == 1, "smishing", "legitimate")
    df["source"] = "v2_2026-09-15"
    df["template_id"] = np.nan
    df["link_domain"] = [f"d{i // 2}.com" for i in range(n)]  # two messages per domain
    df["link_status"] = "link"
    df["has_url"] = 1
    df.loc[:19, ["link_status", "has_url"]] = ["none", 0]
    return df


PARAMS = {"n_estimators": 30, "min_samples_leaf": 2, "max_features": "sqrt"}


def test_rows_without_a_link_get_minus_one_and_zero_contributions():
    df = make_frame()
    bundle = urf.fit(df, urf.INPUTS["full"], 0, PARAMS)
    out = urf.outputs(bundle, df)
    no_link = (df["link_status"] != "link").to_numpy()
    assert (out.loc[no_link, ["P", "u"]] == -1).all().all()
    assert (out.loc[no_link, CONTRIBUTION_COLUMNS] == 0).all().all()
    linked = out.loc[~no_link]
    assert linked["P"].between(0, 1).all() and (linked["u"] >= 0).all()


def test_contributions_add_up_to_p_even_with_missing_page_features():
    df = make_frame(nan_pages=0.3)
    bundle = urf.fit(df.assign(**{c: df[c].fillna(0.0) for c in FULL_FEATURES}), urf.INPUTS["full"], 0, PARAMS)
    linked = df[urf.link_mask(df)]
    out = urf.outputs(bundle, df).loc[linked.index]
    _, bias = path_contributions(bundle["model"], linked[FULL_FEATURES].to_numpy(dtype=np.float32))
    assert np.allclose(out["P"], bias + out[CONTRIBUTION_COLUMNS].sum(axis=1), atol=1e-4)


def test_url_only_forest_leaves_page_contributions_at_zero():
    df = make_frame()
    bundle = urf.fit(df, urf.INPUTS["url_only"], 0, PARAMS)
    out = urf.outputs(bundle, df)
    page_cols = [f"c_{f}" for f in PAGE_COLUMNS]
    assert (out[page_cols] == 0).all().all()


def test_out_of_fold_rows_are_never_scored_by_a_forest_that_saw_their_domain():
    df = make_frame(n=300)
    oof = urf.oof_outputs(df, urf.INPUTS["full"], 0, PARAMS, n_splits=5)
    # A row scored by a forest fit on itself is nearly perfect; out-of-fold P on pure-noise labels is not.
    rng = np.random.default_rng(1)
    noisy = df.assign(label=np.where(rng.random(len(df)) < 0.5, "smishing", "legitimate"))
    oof_noise = urf.oof_outputs(noisy, urf.INPUTS["full"], 0, PARAMS, n_splits=5)
    linked = urf.link_mask(noisy)
    in_sample = urf.outputs(urf.fit(noisy, urf.INPUTS["full"], 0, PARAMS), noisy)
    y = urf.labels(noisy)[linked]
    acc_oof = ((oof_noise.loc[linked, "P"] >= 0.5).to_numpy() == y).mean()
    acc_in = ((in_sample.loc[linked, "P"] >= 0.5).to_numpy() == y).mean()
    assert acc_in > 0.85 and acc_oof < 0.65
    assert (oof.loc[~urf.link_mask(df), "P"] == -1).all()


def test_unified_dataset_marks_only_borrowed_rows_as_loaded():
    sms = pd.DataFrame({"sms": ["a b", "c d", "e f"], "label": ["smishing", "legitimate", "smishing"],
                        "source": ["v2_2026-09-15"] * 3, "template_id": [np.nan] * 3})
    from src.data.split_dataset import ensure_message_id
    ids = ensure_message_id(sms)["message_id"]
    links = pd.DataFrame({"message_id": ids, "link_status": ["link", "none", "redacted"],
                          "link": ["http://x.com", np.nan, np.nan], "link_domain": ["x.com", np.nan, np.nan],
                          "n_links": [1, 0, 0], "has_url": [1, 0, 1], "hybrid_page": [1, 0, 0],
                          "page_source_url": ["http://p.com", np.nan, np.nan]})
    for c in urf.INPUTS["url_only"] + PAGE_COLUMNS:
        links[c] = [1.0, -1.0, -1.0] if c in urf.INPUTS["url_only"] else [2.0, np.nan, np.nan]
    out = build(sms, links)
    assert list(out.columns) == COLUMNS
    assert out["inspect_status"].tolist() == ["loaded", "not_attempted", "not_attempted"]
    assert out["page_origin"].tolist() == ["hybrid_temp", "none", "none"]
    assert out.loc[1:, PAGE_COLUMNS].isna().all().all()
