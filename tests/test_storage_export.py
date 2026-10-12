"""
Tests for Stage 6, persistence, and the JSONL export (WEB_INSPECTION_PLAN.md sections 13.8, 15).
No network.

Run with:  python -m pytest tests/test_storage_export.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3

import pytest

from lejet import schema, urlnorm
from lejet.stages.http_stage import Hop
from lejet.storage import Storage, export_jsonl, sha256_file


def make_row(raw, batch="b001", fail="none", **over):
    row = schema.new_row(urlnorm.compute_url_id(raw), raw, batch)
    row["fail_reason"] = fail
    row["fetch_ok"] = 1 if fail == "none" else 0
    row["nb_hyperlinks"] = 5
    row["reg_domain"] = "example.com"
    row.update(over)
    return row


@pytest.fixture
def store(tmp_path):
    s = Storage(str(tmp_path / "t.sqlite"))
    yield s
    s.close()


def test_wal_mode_and_tables(store):
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    names = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "urls", "hops", "errors"} <= names


def test_save_and_read_back_with_terminal_status(store):
    assert store.save(make_row("https://a.test/")) == "done"
    assert store.save(make_row("https://b.test/", fail="dns_nxdomain")) == "failed:dns_nxdomain"
    assert store.status_counts() == {"done": 1, "failed:dns_nxdomain": 1}
    rows = list(store.rows())
    assert [r["url"] for r in rows] == sorted([r["url"] for r in rows], key=lambda u: urlnorm.compute_url_id(u))
    assert all(set(schema.new_row("x" * 16, "x", "b")) <= set(r) for r in rows)


def test_one_row_per_url_resaving_replaces(store):
    row = make_row("https://a.test/")
    store.save(row, hops=[Hop(0, 302, "a.test", "b.test", 5)], errors=[("http", "note", "first")])
    row["nb_hyperlinks"] = 99
    store.save(row, hops=[Hop(0, 200, "a.test", "", 7)], errors=[])
    assert store.conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0] == 1
    assert next(store.rows())["nb_hyperlinks"] == 99
    assert [h[1] for h in store.hops_of(row["url_id"])] == [200]
    assert store.errors_of(row["url_id"]) == []


def test_hops_and_errors_are_stored_and_sanitized(store):
    row = make_row("https://a.test/")
    store.save(row, hops=[Hop(0, 302, "a.test\x1b[0m", "‮b.test", 5), Hop(1, 200, "b.test", "", 9)],
               errors=[("http", "RuntimeError\x00", "bad \x1b[31mthing" + "x" * 1000)])
    hops = store.hops_of(row["url_id"])
    assert [h[0] for h in hops] == [0, 1]
    assert "\x1b" not in hops[0][2] and "‮" not in hops[0][3]
    stage, exc, msg = store.errors_of(row["url_id"])[0]
    assert "\x00" not in exc and "\x1b" not in msg and len(msg) <= 200


def test_hostile_strings_cannot_inject_sql(store):
    evil = "https://x.test/'; DROP TABLE urls; --"
    store.save(make_row(evil, hosting_platform="'); DROP TABLE urls; --"))
    assert store.conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0] == 1
    assert next(store.rows())["url"] == evil


def test_done_ids_excludes_internal_errors_so_they_are_retried(store):
    store.save(make_row("https://ok.test/"))
    store.save(make_row("https://dead.test/", fail="dns_nxdomain"))
    store.save(make_row("https://flaky.test/", fail="internal_error"))
    done = store.done_ids()
    assert urlnorm.compute_url_id("https://ok.test/") in done
    assert urlnorm.compute_url_id("https://dead.test/") in done
    assert urlnorm.compute_url_id("https://flaky.test/") not in done


def test_a_transaction_is_all_or_nothing(store):
    class Boom:
        idx = 0
        status = 0
        host = ""
        location_host = ""

        @property
        def ms(self):
            raise RuntimeError("boom in the middle of a save")
    row = make_row("https://a.test/")
    with pytest.raises(RuntimeError):
        store.save(row, hops=[Boom()])
    assert store.conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0] == 0


def test_run_bookkeeping(store):
    store.start_run("b001", "ab" * 32, 10, {"python": "3.12"}, "an-configured-clean")
    run = store.get_run("b001")
    assert run["n_input"] == 10 and run["snapshot_name"] == "an-configured-clean"
    assert run["finished_at"] is None
    store.finish_run("b001")
    assert store.get_run("b001")["finished_at"]
    store.start_run("b001", "cd" * 32, 12, {"python": "3.12"})          # a resumed run updates, not duplicates
    assert store.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert store.get_run("b001")["n_input"] == 12


def test_two_connections_can_read_while_one_writes(tmp_path):
    path = str(tmp_path / "w.sqlite")
    a, b = Storage(path), Storage(path)
    try:
        a.save(make_row("https://a.test/"))
        assert len(list(b.rows())) == 1
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------
def populate(path):
    s = Storage(path)
    s.start_run("b001", "f" * 64, 3, {"python": "3.12", "httpx": "0.28.1"}, "an-configured-clean")
    s.save(make_row("https://evil.example.com/login", html_available=1, page_fetched=1, rdap_ok=1,
                    final_url="https://www.other.com/x", final_reg_domain="other.com"))
    s.save(make_row("http://dead.test/a", fail="dns_nxdomain"))
    s.save(make_row("http://blocked.test/", fail="http_4xx"))
    s.save(make_row("https://other-batch.test/", batch="b002"))
    s.finish_run("b001")
    s.close()


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="ascii") as fh:
        return [json.loads(line) for line in fh]


def test_export_files_and_contents(tmp_path):
    db = str(tmp_path / "x.sqlite")
    populate(db)
    info = export_jsonl(db, str(tmp_path / "out"), "b001", {"vpn_provider": "Mullvad", "preflight_pass": True})
    records = read_jsonl(info["jsonl"])
    assert len(records) == info["n_rows"] == 3                       # only batch b001
    assert [r["url_id"] for r in records] == sorted(r["url_id"] for r in records)
    for rec in records:
        assert list(rec["features"]) == schema.VM_FEATURES
        assert "page_rank" not in rec["features"]
        assert rec["url"].startswith("hxx")
    text = open(info["jsonl"], "rb").read()
    assert b"http://" not in gzip.decompress(text) and b"https://" not in gzip.decompress(text)

    digest = sha256_file(info["jsonl"])
    assert digest == info["sha256"] == hashlib.sha256(text).hexdigest()
    sidecar = open(info["jsonl"] + ".sha256", encoding="ascii").read().split()
    assert sidecar[0] == digest and sidecar[1] == "b001.jsonl.gz"

    manifest = json.load(open(info["manifest"], encoding="utf-8"))
    assert manifest["batch_id"] == "b001" and manifest["n_rows"] == 3 and manifest["n_done"] == 1
    assert manifest["n_failed_by_reason"] == {"dns_nxdomain": 1, "http_4xx": 1}
    assert manifest["input_sha256"] == "f" * 64 and manifest["snapshot_name"] == "an-configured-clean"
    assert manifest["jsonl_gz_sha256"] == digest
    assert manifest["vpn_provider"] == "Mullvad" and manifest["preflight_pass"] is True
    assert manifest["tool_versions"]["httpx"] == "0.28.1"
    assert manifest["started"] and manifest["finished"]


def test_export_is_deterministic(tmp_path):
    db = str(tmp_path / "x.sqlite")
    populate(db)
    a = export_jsonl(db, str(tmp_path / "o1"), "b001")
    b = export_jsonl(db, str(tmp_path / "o2"), "b001")
    assert a["sha256"] == b["sha256"]


def test_export_of_an_empty_batch_is_valid(tmp_path):
    db = str(tmp_path / "e.sqlite")
    Storage(db).close()
    info = export_jsonl(db, str(tmp_path / "o"), "nothing")
    assert info["n_rows"] == 0 and read_jsonl(info["jsonl"]) == []


def test_exported_json_is_strict(tmp_path):
    db = str(tmp_path / "s.sqlite")
    s = Storage(db)
    s.save(make_row("https://x.test/", domain_age=float("nan"), nb_hyperlinks=float("inf")))
    s.close()
    info = export_jsonl(db, str(tmp_path / "o"), "b001")
    line = gzip.decompress(open(info["jsonl"], "rb").read()).decode("ascii")
    assert "NaN" not in line and "Infinity" not in line
    rec = json.loads(line)
    assert rec["features"]["domain_age"] == -1 and rec["features"]["nb_hyperlinks"] == -1
