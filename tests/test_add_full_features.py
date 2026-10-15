"""The host-side completion of the exported feature table (scripts/add_full_features.py). No network."""
from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import add_full_features as af                              # noqa: E402
from lejet import schema, urlnorm                           # noqa: E402


def record(raw, final_domain="example.com", with_full=True):
    row = schema.new_row(urlnorm.compute_url_id(raw), raw, "b001")
    row.update(final_reg_domain=final_domain, fail_reason="none", fetch_ok=1)
    rec = schema.export_record(row)
    if not with_full:
        del rec["features_full"]
    return rec


def write_clean(path, recs):
    with open(path, "w", encoding="ascii") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def test_page_rank_is_filled_from_the_host_file(tmp_path):
    recs = [record("https://a.example.com/"), record("https://b.test.org/", "test.org"),
            record("https://c.nowhere.xyz/", "nowhere.xyz")]
    clean = tmp_path / "b.clean.jsonl"
    write_clean(clean, recs)
    pr = tmp_path / "pr.csv"
    pr.write_text(f"url_id,page_rank\n{recs[0]['url_id']},6\n{recs[1]['url_id']},-1\n", encoding="utf-8")
    out = tmp_path / "full.csv"
    assert af.main([str(clean), "--out", str(out), "--page-rank-csv", str(pr)]) == 0
    rows = {r["url_id"]: r for r in csv.DictReader(open(out, encoding="utf-8"))}
    a, b, c = (rows[r["url_id"]] for r in recs)
    assert list(a)[:5] == af.HEAD and list(a)[5:] == schema.FULL_FEATURES
    assert (a["page_rank"], b["page_rank"], c["page_rank"]) == ("6", "-1", "-1")           # a failed lookup stays -1
    for dropped in schema.DROPPED_FROM_FULL:
        assert dropped not in a


def test_without_the_host_file_page_rank_stays_unmeasured(tmp_path):
    clean = tmp_path / "b.clean.jsonl"
    write_clean(clean, [record("https://a.example.com/")])
    out = tmp_path / "full.csv"
    assert af.main([str(clean), "--out", str(out)]) == 0
    row = next(csv.DictReader(open(out, encoding="utf-8")))
    assert row["page_rank"] == "-1"


def test_batches_from_before_the_extra_export_still_convert(tmp_path):
    clean = tmp_path / "b.clean.jsonl"
    write_clean(clean, [record("https://a.example.com/", with_full=False)])
    out = tmp_path / "full.csv"
    assert af.main([str(clean), "--out", str(out)]) == 0
    row = next(csv.DictReader(open(out, encoding="utf-8")))
    assert all(row[n] == "-1" for n in schema.FULL_FEATURES)
