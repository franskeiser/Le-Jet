"""Tests for src/data/build_live_set.py: the live training set and the fresh test set must never leak into each other.
A tiny fake lab output (invented domains, no network) is built in a temp folder; each leak rule is then broken on purpose.

Run with:  python -m pytest tests/test_live_set.py
"""
from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from src.data import build_live_set as bls
from src.features.model.rf_extractor import EXTERNAL_FEATURES, URL_TEXT_FEATURES, WEBPAGE_FEATURES

FEATS = {n: 1 for n in URL_TEXT_FEATURES + WEBPAGE_FEATURES + EXTERNAL_FEATURES}


def lab_row(i, host, *, fetched=True, fail="none", hops=0, final=None, shortener=False):
    url = f"hxxps://{host.replace('.', '[.]')}/p{i}"
    feats = dict(FEATS, shortened_url_indicator=int(shortener))
    if not fetched:
        feats.update({n: -1 for n in WEBPAGE_FEATURES})
    final = final or host
    return {"url_id": hashlib.sha256(url.encode()).hexdigest()[:16], "url": url, "features": feats,
            "page_fetched": int(fetched), "fail_reason": fail, "hop_count": hops,
            "final_url": f"hxxps://{final.replace('.', '[.]')}/" if fetched else "",
            "final_reg_domain": final.replace(".", "[.]") if fetched else "",
            "fetch_ts": "2026-10-13T01:00:00Z", "free_hosting_subdomain": 0, "hosting_platform": ""}


def write_batch(tmp_path, name, rows, ranks=True):
    with open(tmp_path / f"{name}.clean.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if ranks:
        pd.DataFrame({"url_id": [r["url_id"] for r in rows], "page_rank": [4] * len(rows),
                      "page_rank_ok": [1] * len(rows)}).to_csv(tmp_path / f"{name}.model_input.csv", index=False)


def make(tmp_path, n_each=30):
    smish = [lab_row(i, f"scam{i}.example{i % 7}.xyz") for i in range(n_each)]
    smish += [lab_row(100 + i, "bit.ly", shortener=True, hops=1, final=f"landing{i}.example.top") for i in range(8)]
    smish += [lab_row(200 + i, f"dead{i}.example.cc", fetched=False, fail="dns_nxdomain") for i in range(5)]
    benign = [lab_row(i, f"shop{i}.legit{i}.com") for i in range(n_each + 10)]
    write_batch(tmp_path, "smishX", smish)
    write_batch(tmp_path, "benignX", benign)
    return [("smishX", "smishing", "team"), ("benignX", "legitimate", "generated")]


def build(tmp_path, blocked=None, **kw):
    return bls.build(make(tmp_path), clean_dir=tmp_path, blocked=blocked if blocked is not None else set(), **kw)


# ---------------------------------------------------------------------------------------
def test_a_clean_build_has_no_problems_and_both_classes_in_both_sets(tmp_path):
    frame, report = build(tmp_path)
    assert report["problems"] == []
    counts = report["by_split_label"]
    assert all(counts[f"{s}/{l}"] > 0 for s in ("live_train", "fresh_test") for l in bls.LABELS)


def test_no_domain_group_is_in_both_sets(tmp_path):
    frame, _ = build(tmp_path)
    shared = frame.groupby("group")["split"].nunique()
    assert (shared == 1).all()


def test_the_test_share_is_close_to_the_requested_one_per_class(tmp_path):
    frame, _ = build(tmp_path, test_share=0.4)
    for label in bls.LABELS:
        sub = frame[frame.label == label]
        assert abs((sub.split == "fresh_test").mean() - 0.4) < 0.2


def test_the_split_is_reproducible_and_depends_on_the_seed(tmp_path):
    a, _ = build(tmp_path, seed=7)
    b, _ = build(tmp_path, seed=7)
    c, _ = build(tmp_path, seed=8)
    assert a["split"].tolist() == b["split"].tolist()
    assert a["split"].tolist() != c["split"].tolist()


def test_shorteners_are_grouped_by_where_they_lead_not_by_their_own_domain(tmp_path):
    frame, _ = build(tmp_path)
    short = frame[frame["link"].str.contains("bit[.]ly", regex=False)]
    assert len(short) == 8 and short["group"].nunique() == 8                  # one group each (their final domains)
    assert not (short["group"] == "bit.ly").any()
    assert short["split"].nunique() == 2                                      # not all in one set


def test_a_dead_shortener_is_its_own_group(tmp_path):
    rows = [lab_row(1, "bit.ly", fetched=False, fail="connect_timeout", shortener=True)]
    write_batch(tmp_path, "s1", rows)
    rec = bls.load_batch(tmp_path, "s1", "smishing", "team")[0]
    assert rec["group"].startswith("link:")


def test_domains_the_models_have_already_seen_never_reach_the_test_set(tmp_path):
    frame, report = build(tmp_path, blocked={f"legit{i}.com" for i in range(0, 40)} | {"example3.xyz"})
    seen = frame[frame["link"].str.contains("legit", regex=False)]
    assert (seen["split"] == "live_train").all()                              # every blocked benign domain: train only
    assert seen["overlap_moved"].all()
    assert not any("already known" in p or "both sets" in p for p in report["problems"])   # moved, so nothing leaks


def test_blocked_domains_are_flagged_in_the_leak_check(tmp_path):
    frame, _ = build(tmp_path)
    records = frame.to_dict("records")
    victim = next(r for r in records if r["split"] == "fresh_test" and r["label"] == "legitimate")
    domain = bls.registrable_domain(bls.refang(victim["link"])).lower()
    problems = bls.check_leaks(records, blocked_domains={domain})
    assert any("already known" in p for p in problems)


def test_a_group_in_both_sets_is_detected(tmp_path):
    frame, _ = build(tmp_path)
    records = copy.deepcopy(frame.to_dict("records"))
    first = next(r for r in records if r["split"] == "live_train" and r["label"] == "legitimate")
    second = next(r for r in records if r["split"] == "fresh_test" and r["label"] == "legitimate")
    second["group"] = first["group"]
    assert any("both sets" in p for p in bls.check_leaks(records))


def test_a_group_holding_both_classes_goes_to_training(tmp_path):
    a = lab_row(1, "mixed.example.com")
    b = lab_row(2, "mixed.example.com")
    write_batch(tmp_path, "sA", [a])
    write_batch(tmp_path, "bA", [b])
    records = bls.load_batch(tmp_path, "sA", "smishing", "t") + bls.load_batch(tmp_path, "bA", "legitimate", "t")
    out = bls.assign_split(records, test_share=0.9)
    assert {r["split"] for r in out} == {"live_train"}


@pytest.mark.parametrize("mutation, expected", [
    (lambda rs: rs[0].update(link_id=rs[1]["link_id"]), "duplicate link_id"),
    (lambda rs: rs[0].update(key=rs[1]["key"]), "same link appears twice"),
    (lambda rs: rs[0].update(label="phishing"), "bad label"),
    (lambda rs: rs[0].update(page_fetched=1, final_url="", fetch_status="live"), "page_fetched=1 without"),
    (lambda rs: rs[0].update(page_fetched=0, nb_hyperlinks=5), "not -1"),
    (lambda rs: rs[0].update(domain_age=float("nan")), "blank or NaN"),
    (lambda rs: rs[0].update(domain_age=""), "blank or NaN"),
])
def test_each_integrity_rule_is_enforced(tmp_path, mutation, expected):
    frame, _ = build(tmp_path)
    records = copy.deepcopy(frame.to_dict("records"))
    mutation(records)
    assert any(expected in p for p in bls.check_leaks(records)), bls.check_leaks(records)


def test_a_class_missing_from_a_set_is_detected(tmp_path):
    frame, _ = build(tmp_path)
    records = [dict(r, split="live_train") for r in frame.to_dict("records")]
    assert any("fresh_test has no" in p for p in bls.check_leaks(records))


def test_not_fetched_rows_have_minus_one_webpage_features_and_no_final_url(tmp_path):
    frame, _ = build(tmp_path)
    dead = frame[frame.page_fetched == 0]
    assert len(dead) == 5
    assert (dead[WEBPAGE_FEATURES] == -1).all().all() and (dead["final_url"] == "").all()
    assert (dead["n_redirects"] == -1).all()


def test_labels_are_never_changed_by_liveness(tmp_path):
    frame, _ = build(tmp_path)
    dead = frame[frame.page_fetched == 0]
    assert (dead["label"] == "smishing").all()                                 # dead links stay smishing
    assert set(frame["label"]) == set(bls.LABELS)


def test_fetch_status_follows_the_spec():
    assert bls.fetch_status("none", 0) == "live" and bls.fetch_status("none", 2) == "redirected"
    assert bls.fetch_status("http_4xx", 0) == "blocked" and bls.fetch_status("connect_timeout", 0) == "timeout"
    assert bls.fetch_status("dns_nxdomain", 0) == "dead" and bls.fetch_status("rejected_url", 0) == "not_attempted"


def test_page_rank_comes_from_the_lookup_and_is_minus_one_when_missing(tmp_path):
    rows = [lab_row(1, "a.example.com"), lab_row(2, "b.example.com")]
    write_batch(tmp_path, "r1", rows, ranks=False)
    pd.DataFrame({"url_id": [rows[0]["url_id"]], "page_rank": [7], "page_rank_ok": [1]}).to_csv(
        tmp_path / "r1.model_input.csv", index=False)
    recs = bls.load_batch(tmp_path, "r1", "legitimate", "t")
    assert [r["page_rank"] for r in recs] == [7, -1]


def test_no_raw_url_is_written_and_the_key_is_a_hash(tmp_path):
    frame, _ = build(tmp_path)
    assert not frame["link"].str.contains(r"https?://", regex=True).any()
    assert frame["key"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert bls.link_key("hxxps://a[.]example[.]com/x") == bls.link_key("https://a.example.com/x")


def test_the_same_link_in_two_batches_counts_once(tmp_path):
    rows = [lab_row(1, "dup.example.com")]
    write_batch(tmp_path, "d1", rows)
    write_batch(tmp_path, "d2", rows)
    write_batch(tmp_path, "b9", [lab_row(i, f"x{i}.ok{i}.com") for i in range(20)])
    frame, _ = bls.build([("d1", "smishing", "a"), ("d2", "smishing", "b"), ("b9", "legitimate", "c")],
                         clean_dir=tmp_path, blocked=set())
    assert (frame["link_id"] == rows[0]["url_id"]).sum() == 1


def test_a_bad_label_name_is_rejected(tmp_path):
    write_batch(tmp_path, "z", [lab_row(1, "z.example.com")])
    with pytest.raises(ValueError):
        bls.load_batch(tmp_path, "z", "phishing", "t")


@pytest.mark.parametrize("column", ["sms", "sms_text"])
def test_known_domains_reads_the_sms_file_whatever_its_text_column_is_called(tmp_path, column):
    web = tmp_path / "web.csv"
    sms = tmp_path / "sms.csv"
    pd.DataFrame({"url": ["http://a.seen-by-extractor.com/a", "http://[broken/x"]}).to_csv(web, index=False)
    pd.DataFrame({column: ["claim now at http://www.seen-by-cnn.org/x", "no link here"], "label": ["smishing", "legitimate"]}).to_csv(sms, index=False)
    seen = bls.known_domains(web, sms)
    assert seen == {"seen-by-extractor.com", "seen-by-cnn.org"}              # registrable domains; a malformed address is skipped
