"""
Tests for the host-side validator scripts/validate_output.py (WEB_INSPECTION_PLAN.md sections
12.5, 16 Level 0): "tested with malicious fixtures (formula injection, ANSI/bidi strings,
oversized lines, zip bomb, extra/missing fields)". No network.

Run with:  python -m pytest tests/test_validate_output.py
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import validate_output as vo                                    # noqa: E402
from lejet import schema, urlnorm                              # noqa: E402

RAWS = ["https://evil.example.com/login", "http://bit.ly/abc", "gcash-verify.xyz/otp", "https://shop.test/"]


def good_record(raw, **over):
    row = schema.new_row(urlnorm.compute_url_id(raw), raw, "b001")
    row.update(fetch_ok=1, html_available=1, page_fetched=1, rdap_ok=1, html_bytes=1500, script_count=2,
               near_empty=0, reg_domain="example.com", final_url=raw, final_reg_domain="example.com",
               stage_ms_dns=5, stage_ms_rdap=6, stage_ms_http=7, stage_ms_html=8, duration_ms=30,
               url_length=len(raw), nb_hyperlinks=7, ratio_intHyperlinks=0.5, links_in_tags=40.0,
               ratio_intMedia=10.0, ratio_extMedia=90.0, login_form=1, domain_in_title=1,
               domain_age=3000, domain_registration_length=200, dns_record=0, whois_registered_domain=0,
               empty_title=0, nb_extCSS=1, external_favicon=0, iframe=0, domain_with_copyright=0)
    row.update(over)
    return schema.export_record(row)


def write_gz(path, lines, mtime=0):
    data = b"".join(l if isinstance(l, bytes) else l.encode("ascii") for l in lines)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=mtime) as gz:
        gz.write(data)


def jl(rec):
    return json.dumps(rec, separators=(",", ":"), ensure_ascii=True) + "\n"


@pytest.fixture
def good(tmp_path):
    recs = [good_record(r) for r in RAWS]
    path = str(tmp_path / "b001.jsonl.gz")
    write_gz(path, [jl(r) for r in recs])
    ids = [r["url_id"] for r in recs]
    return path, recs, ids, tmp_path


def problems_of(path, **kw):
    return vo.validate(path, **kw)[1]


def test_a_good_file_passes_everything(good):
    path, recs, ids, tmp = good
    sidecar = tmp / "b001.jsonl.gz.sha256"
    sidecar.write_text(hashlib.sha256(open(path, "rb").read()).hexdigest() + "  b001.jsonl.gz\n")
    records, problems = vo.validate(path, input_ids=ids, expected_rows=4, sha256_file=str(sidecar))
    assert problems == [] and len(records) == 4


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: r.update(extra_field=1), "unexpected keys"),
    (lambda r: r.pop("duration_ms"), "missing keys"),
    (lambda r: r["features"].update(url_length="30"), "not a finite number"),
    (lambda r: r["features"].update(url_length=None), "not a finite number"),
    (lambda r: r["features"].update(url_length=True), "not a finite number"),
    (lambda r: r["features"].pop("nb_hyperlinks"), "wrong keys or order"),
    (lambda r: r["features"].update(zzz=1), "wrong keys or order"),
    (lambda r: r["features"].update(ratio_intHyperlinks=5), "above 1"),
    (lambda r: r["features"].update(links_in_tags=250), "above 100"),
    (lambda r: r["features"].update(login_form=2), "not 0, 1 or -1"),
    (lambda r: r["features"].update(nb_hyperlinks=-5), "below -1"),
    (lambda r: r["features"].update(nb_hyperlinks=10**12), "implausibly large"),
    (lambda r: r.update(url_id="ZZZZ"), "url_id"),
    (lambda r: r.update(url_id="0123456789ABCDEF"), "url_id"),
    (lambda r: r.update(fail_reason="totally_invented"), "unknown fail_reason"),
    (lambda r: r.update(fetch_ok=False), "fetch_ok disagrees"),
    (lambda r: r.update(page_fetched=0), "page_fetched disagrees"),
    (lambda r: r.update(fetch_ok=1), "not a boolean"),
    (lambda r: r.update(has_url=0), "has_url must be 1"),
    (lambda r: r.update(fetch_ts="yesterday"), "fetch_ts"),
    (lambda r: r.update(url="https://evil.example.com/login"), "clickable URL"),
    (lambda r: r.update(final_url="see http://evil.test/x"), "clickable URL"),
    (lambda r: r.update(url="evil.example.com/login"), "not defanged"),
    (lambda r: r.update(hosting_platform="=cmd|' /C calc'!A0"), "formula"),
    (lambda r: r.update(hosting_platform="@SUM(1+1)"), "formula"),
    (lambda r: r.update(reg_domain="example[.]com\x1b[31m"), "control/ANSI/bidi"),
    (lambda r: r.update(reg_domain="exam‮ple"), "control/ANSI/bidi"),
    (lambda r: r.update(hosting_platform="x" * 500), "longer than"),
    (lambda r: r.update(stage_ms={"dns": 1}), "stage_ms"),
    (lambda r: r["stage_ms"].update(dns=-5), "stage_ms"),
    (lambda r: r.update(hop_count="3"), "hop_count"),
    (lambda r: r.update(html_available=False), "webpage features present"),
    (lambda r: r.update(rdap_ok=False), "RDAP features present"),
    (lambda r: r.update(duration_ms=1.5), "duration_ms"),
])
def test_malicious_or_malformed_records_are_rejected(good, mutate, fragment):
    path, recs, ids, tmp = good
    bad = copy.deepcopy(recs)
    mutate(bad[0])
    out = str(tmp / "bad.jsonl.gz")
    write_gz(out, [jl(r) for r in bad])
    probs = problems_of(out)
    assert probs, "should have been rejected"
    assert any(fragment in p for p in probs), probs


def test_nan_and_infinity_are_rejected_at_the_parser(good):
    path, recs, ids, tmp = good
    base = jl(recs[0])
    for token in ("NaN", "Infinity", "-Infinity"):
        text = base.replace('"url_length":%d' % recs[0]["features"]["url_length"], '"url_length":' + token)
        assert text != base
        out = str(tmp / "n.jsonl.gz")
        write_gz(out, [text])
        assert any("invalid JSON" in p for p in problems_of(out))


def test_invalid_json_non_ascii_and_empty_lines(good):
    path, recs, ids, tmp = good
    for line in ('{"a": 1', "not json at all", '{"url_id": "café"}', "[1,2,3]", '"string"', "null"):
        out = str(tmp / "x.jsonl.gz")
        write_gz(out, [line.encode("utf-8") + b"\n"])
        assert problems_of(out)


def test_duplicate_url_ids_are_rejected(good):
    path, recs, ids, tmp = good
    out = str(tmp / "d.jsonl.gz")
    write_gz(out, [jl(recs[0]), jl(recs[1]), jl(recs[0])])
    assert any("duplicate url_id" in p for p in problems_of(out))


def test_an_oversized_line_is_rejected(good):
    path, recs, ids, tmp = good
    out = str(tmp / "big.jsonl.gz")
    write_gz(out, [jl(recs[0]), b"x" * (vo.MAX_LINE + 100) + b"\n"])
    assert any("longer than" in p for p in problems_of(out))


def test_a_zip_bomb_is_stopped_by_the_decompression_cap(good, monkeypatch):
    path, recs, ids, tmp = good
    monkeypatch.setattr(vo, "MAX_DECOMPRESSED", 1_000_000)
    out = str(tmp / "bomb.jsonl.gz")
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for _ in range(100):
            gz.write(b"\x00" * 100_000)                      # 10 MB of zeros, ~10 KB compressed
    assert os.path.getsize(out) < 100_000
    probs = problems_of(out)
    assert any("zip bomb" in p or "longer than" in p for p in probs)


def test_oversized_compressed_file_is_refused(good, monkeypatch):
    path, recs, ids, tmp = good
    monkeypatch.setattr(vo, "MAX_COMPRESSED", 10)
    assert any("compressed file larger" in p for p in problems_of(path))


def test_not_a_gzip_file(tmp_path):
    p = tmp_path / "x.jsonl.gz"
    p.write_bytes(b"this is plain text, not gzip\n")
    assert any("not a readable gzip" in x for x in problems_of(str(p)))
    q = tmp_path / "trunc.jsonl.gz"
    q.write_bytes(gzip.compress(b"x" * 1000)[:20])
    assert problems_of(str(q))


def test_sha256_sidecar_mismatch_is_detected(good):
    path, recs, ids, tmp = good
    sidecar = tmp / "wrong.sha256"
    sidecar.write_text("0" * 64 + "  b001.jsonl.gz\n")
    assert any("sha256" in p for p in problems_of(path, sha256_file=str(sidecar)))
    assert any("sidecar" in p for p in problems_of(path, sha256_file=str(tmp / "missing.sha256")))


def test_row_count_and_input_ids_are_enforced(good):
    path, recs, ids, tmp = good
    assert any("row count" in p for p in problems_of(path, expected_rows=5))
    assert any("missing from the output" in p for p in problems_of(path, input_ids=ids + ["a" * 16]))
    assert any("never pushed" in p for p in problems_of(path, input_ids=ids[:3]))
    assert problems_of(path, input_ids=list(reversed(ids))) == []


def test_main_writes_clean_files_only_on_success(good):
    path, recs, ids, tmp = good
    push = tmp / "push.txt"
    push.write_text("\n".join(f"{i}\thttps://x/{n}" for n, i in enumerate(ids)) + "\n")
    clean = tmp / "clean"
    rc = vo.main([path, "--input", str(push), "--clean-dir", str(clean), "--csv", str(tmp / "o.csv")])
    assert rc == 0
    out = clean / "b001.clean.jsonl"
    assert len(out.read_text().splitlines()) == 4
    assert (tmp / "o.csv").exists()

    bad = copy.deepcopy(recs)
    bad[1]["features"]["url_length"] = "boom"
    badpath = str(tmp / "bad.jsonl.gz")
    write_gz(badpath, [jl(r) for r in bad])
    clean2 = tmp / "clean2"
    assert vo.main([badpath, "--clean-dir", str(clean2)]) == 1
    assert not clean2.exists()                               # nothing was written


def test_csv_neutralizes_formula_characters(tmp_path):
    rec = good_record("https://x.test/")
    rec["url"] = "=cmd|calc"                                 # only the url field may start with '='
    path = str(tmp_path / "f.jsonl.gz")
    write_gz(path, [jl(rec)])
    records, problems = vo.validate(path)
    assert problems == [], problems
    out = vo.write_clean(records, str(tmp_path / "c"), "b001", str(tmp_path / "o.csv"))
    rows = list(csv.DictReader(open(tmp_path / "o.csv", encoding="utf-8")))
    assert rows[0]["url"] == "'=cmd|calc"


def test_it_runs_as_a_script_with_isolated_python(good):
    path, recs, ids, tmp = good
    script = os.path.join(ROOT, "scripts", "validate_output.py")
    ok = subprocess.run([sys.executable, "-I", script, path], capture_output=True, text=True, cwd=str(tmp))
    assert ok.returncode == 0 and "VALIDATION OK: 4 rows" in ok.stdout
    broken = str(tmp / "broken.jsonl.gz")
    write_gz(broken, ['{"not": "valid"}\n'])
    bad = subprocess.run([sys.executable, "-I", script, broken], capture_output=True, text=True, cwd=str(tmp))
    assert bad.returncode == 1 and "VALIDATION FAILED" in bad.stdout


def test_real_exports_from_the_pipeline_storage_validate(tmp_path):
    """The exporter and the validator must agree: whatever export_jsonl writes passes."""
    from lejet.storage import Storage, export_jsonl
    db = str(tmp_path / "x.sqlite")
    s = Storage(db)
    s.start_run("b001", "f" * 64, 3, {"python": "3"})
    for raw in RAWS[:3]:
        row = schema.new_row(urlnorm.compute_url_id(raw), raw, "b001")
        row["fail_reason"] = "dns_nxdomain"                  # a failed row: everything else unknown (-1)
        s.save(row)
    ok_row = schema.new_row(urlnorm.compute_url_id("http://ok.test/"), "http://ok.test/", "b001")
    ok_row.update(fetch_ok=1, html_available=1, page_fetched=1, nb_hyperlinks=3, ratio_intHyperlinks=0.5)
    for n in schema.WEBPAGE_FEATURES:
        ok_row.setdefault(n, 0)
    ok_row.update(nb_hyperlinks=3, ratio_intHyperlinks=0.5, links_in_tags=0.0, ratio_intMedia=0.0,
                  ratio_extMedia=0.0, nb_extCSS=0, login_form=0, external_favicon=0, iframe=0,
                  empty_title=0, domain_in_title=1, domain_with_copyright=0)
    s.save(ok_row)
    s.close()
    info = export_jsonl(db, str(tmp_path / "out"), "b001")
    records, problems = vo.validate(info["jsonl"], sha256_file=info["jsonl"] + ".sha256")
    assert problems == [], problems
    assert len(records) == 4


# ---------------------------------------------------------------------------------------
# features_full: the 73 exported public-set columns
# ---------------------------------------------------------------------------------------
def test_features_full_is_checked_when_present_and_optional_when_absent(tmp_path):
    rec = good_record(RAWS[0])
    assert list(rec["features_full"]) == schema.FULL_FEATURES
    assert vo.check_record(rec) == []
    old_format = {k: v for k, v in rec.items() if k != "features_full"}
    assert vo.check_record(old_format) == []                       # batches exported before it existed still validate


@pytest.mark.parametrize("name, value, fragment", [
    ("ip", 2, "not 0, 1 or -1"),
    ("ratio_digits_url", 1.5, "above 1"),
    ("safe_anchor", 120, "above 100"),
    ("statistical_report", 3, "not 0, 1, 2"),
    ("length_url", -2, "out of range"),
    ("length_url", 10**10, "out of range"),
    ("length_url", float("nan"), "not a finite number"),
    ("length_url", "12", "not a finite number"),
])
def test_features_full_values_are_range_checked(name, value, fragment):
    rec = good_record(RAWS[0])
    rec["features_full"][name] = value
    assert any(fragment in e for e in vo.check_record(rec)), vo.check_record(rec)


def test_features_full_must_have_exactly_the_73_names_in_order():
    rec = good_record(RAWS[0])
    rec["features_full"].pop("page_rank")
    assert any("wrong keys or order" in e for e in vo.check_record(rec))
    rec = good_record(RAWS[0])
    rec["features_full"] = dict(reversed(list(rec["features_full"].items())))
    assert any("wrong keys or order" in e for e in vo.check_record(rec))
    rec = good_record(RAWS[0])
    rec["features_full"] = "nope"
    assert any("wrong keys or order" in e for e in vo.check_record(rec))


def test_page_columns_must_be_unknown_when_no_page_was_read():
    rec = good_record(RAWS[0], html_available=0, page_fetched=0, fetch_ok=0, fail_reason="http_4xx")
    rec["features_full"]["safe_anchor"] = 50.0
    assert any("page columns present" in e for e in vo.check_record(rec))


def test_the_clean_step_writes_the_full_feature_table(tmp_path):
    recs = [good_record(r) for r in RAWS]
    path = str(tmp_path / "b001.jsonl.gz")
    write_gz(path, [jl(r) for r in recs])
    records, problems = vo.validate(path)
    assert problems == []
    csv_path = str(tmp_path / "b001.csv")
    vo.write_clean(records, str(tmp_path / "clean"), "b001", csv_path)
    with open(str(tmp_path / "b001.full_features.csv"), encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["url_id", "batch_id", "url", "page_fetched", "fail_reason"] + schema.FULL_FEATURES
    assert len(rows) == 1 + len(RAWS) and all(len(r) == 5 + 73 for r in rows)
