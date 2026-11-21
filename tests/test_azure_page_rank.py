"""
Tests for the Azure lab's page_rank look-up: src/lejet/azure_page_rank.py (the unit on the VM) and scripts/azure_page_rank.py
(the PC side). No internet, no VM, no real key: the service and the ssh/scp step are replaced by fakes.

Run with:  python -m pytest tests/test_azure_page_rank.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import azure_page_rank as pc                                      # noqa: E402  (the PC side)
from lejet import azure_page_rank as vm                           # noqa: E402  (the VM side)
from lejet import schema, urlnorm                                 # noqa: E402

SECRET = "TESTKEY-abcdef123456"


def make_clean(tmp_path, specs, name="b001"):
    """specs: [(domain, page_fetched)] -> a .clean.jsonl in the lab's export format."""
    recs = []
    for i, (dom, fetched) in enumerate(specs):
        raw = f"https://{dom}/p{i}"
        row = schema.new_row(urlnorm.compute_url_id(raw), raw, name)
        row.update(fetch_ok=1 if fetched else 0, html_available=1 if fetched else 0, page_fetched=1 if fetched else 0,
                   final_reg_domain=dom, rdap_ok=0)
        recs.append(schema.export_record(row))
    path = str(tmp_path / f"{name}.clean.jsonl")
    with open(path, "w", encoding="ascii") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------- the VM side


def test_vm_reads_key_and_domains_from_the_credentials_directory(tmp_path, monkeypatch):
    (tmp_path / "key").write_text(SECRET + "\n")
    (tmp_path / "domains").write_text("Example.com\nbad domain!\n\nexample.com\nfoo[.]org\n")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    assert vm.load_key() == SECRET
    assert vm.load_domains() == ["example.com", "foo.org"]        # refanged, invalid dropped, no repeats


def test_vm_accepts_the_websites_key_file_format_and_rejects_a_bad_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    (tmp_path / "key").write_text(f"OPENPAGERANK_API_KEY={SECRET}")
    assert vm.load_key() == SECRET
    (tmp_path / "key").write_text("not a key !!")
    with pytest.raises(SystemExit):
        vm.load_key()


def test_vm_without_credentials_stops(monkeypatch):
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    with pytest.raises(SystemExit):
        vm.load_key()


def test_vm_lookup_splits_into_batches_of_100_and_counts_failures():
    calls = []

    def fake(chunk, key, base):
        calls.append(len(chunk))
        if len(calls) == 2:
            raise RuntimeError("HTTP 500")
        return {d: 3 for d in chunk}

    domains = [f"d{i}.example.com" for i in range(250)]
    ranks, failed = vm.lookup(domains, SECRET, pause=0, retries=0, fetch=fake)
    assert calls == [100, 100, 50]
    assert failed == 1 and len(ranks) == 150                       # the failed batch is simply absent -> -1 later


def test_vm_main_prints_one_json_line_and_never_the_key(tmp_path, monkeypatch, capsys):
    (tmp_path / "key").write_text(SECRET)
    (tmp_path / "domains").write_text("example.com\nunrated.org\n")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(vm, "fetch_batch", lambda chunk, key, base_url=None, timeout=15.0: {"example.com": 7, "unrated.org": 0})
    monkeypatch.setattr(vm, "PAUSE", 0)
    monkeypatch.setattr(vm.time, "sleep", lambda s: None)
    rc = vm.main()
    out = capsys.readouterr()
    assert rc == 0 and SECRET not in out.out and SECRET not in out.err
    data = json.loads(out.out)
    assert data == {"ranks": {"example.com": 7, "unrated.org": 0}, "asked": 2, "failed_batches": 0}


def test_vm_error_text_never_contains_the_key(monkeypatch):
    import urllib.error

    def boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 403, "no", {"API-OPR": SECRET}, io.BytesIO(b""))

    monkeypatch.setattr(vm.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as exc:
        vm.fetch_batch(["example.com"], SECRET)
    assert SECRET not in str(exc.value) and "403" in str(exc.value)


# ---------------------------------------------------------------- the PC side


def test_only_pages_that_loaded_are_sent(tmp_path):
    path = make_clean(tmp_path, [("loaded.com", True), ("dead.net", False), ("loaded.com", True), ("other.org", True)])
    recs = pc.lpr.read_clean(path)
    assert pc.fetched_domains(recs) == ["loaded.com", "other.org"]


def test_parse_vm_answer_keeps_only_asked_domains_and_whole_numbers():
    text = json.dumps({"ranks": {"a.com": 4, "b.com": 11, "c.com": True, "evil.com": 5, "d.com": "3", "e.com": 0}})
    assert pc.parse_vm_answer(text, ["a.com", "b.com", "c.com", "d.com", "e.com"]) == {"a.com": 4, "e.com": 0}
    with pytest.raises(RuntimeError):
        pc.parse_vm_answer("<html>", ["a.com"])


def test_end_to_end_writes_model_input_with_minus_one_for_unanswered(tmp_path, monkeypatch):
    path = make_clean(tmp_path, [("answered.com", True), ("unanswered.net", True), ("dead.org", False)])
    cache = str(tmp_path / "cache.json")
    asked = []

    def fake_ask(domains, batch, host):
        asked.append((list(domains), batch, host))
        return {"answered.com": 6}                                # the VM did not answer unanswered.net

    monkeypatch.setattr(pc, "ask_vm", fake_ask)
    rc = pc.main([path, "--cache", cache, "--batch", "t_pr", "--host", "myhost"])
    assert rc == 3                                                # incomplete -> the caller is told to rerun later
    assert asked == [(["answered.com", "unanswered.net"], "t_pr", "myhost")]    # the dead page's domain never leaves the PC
    assert json.load(open(cache)) == {"answered.com": 6}          # a cache with domains and ranks only
    rows = list(csv.DictReader(open(str(tmp_path / "b001.model_input.csv"), encoding="utf-8")))
    by = {r["final_reg_domain"].replace("[.]", "."): r for r in rows}
    assert by["answered.com"]["page_rank"] == "6" and by["answered.com"]["page_rank_ok"] == "1"
    assert by["unanswered.net"]["page_rank"] == "-1" and by["unanswered.net"]["page_rank_ok"] == "0"   # never 0
    assert by["dead.org"]["page_rank"] == "-1"


def test_cached_domains_are_not_asked_again_and_no_key_is_not_an_error(tmp_path, monkeypatch):
    path = make_clean(tmp_path, [("cached.com", True), ("new.com", True)])
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"cached.com": 2}))
    seen = []
    monkeypatch.setattr(pc, "ask_vm", lambda d, b, h: seen.append(list(d)) or None)      # None = "NO KEY" on the VM
    rc = pc.main([path, "--cache", str(cache), "--batch", "t_pr"])
    assert rc == 0 and seen == [["new.com"]]
    rows = {r["final_reg_domain"].replace("[.]", "."): r
            for r in csv.DictReader(open(str(tmp_path / "b001.model_input.csv"), encoding="utf-8"))}
    assert rows["cached.com"]["page_rank"] == "2" and rows["new.com"]["page_rank"] == "-1"


def test_a_vm_failure_never_raises_and_leaves_minus_one(tmp_path, monkeypatch, capsys):
    path = make_clean(tmp_path, [("x.com", True)])

    def boom(d, b, h):
        raise RuntimeError("scp of the domain list failed (is the VM running?)")

    monkeypatch.setattr(pc, "ask_vm", boom)
    rc = pc.main([path, "--cache", str(tmp_path / "c.json"), "--batch", "t_pr"])
    assert rc == 3
    rows = list(csv.DictReader(open(str(tmp_path / "b001.model_input.csv"), encoding="utf-8")))
    assert rows[0]["page_rank"] == "-1" and rows[0]["page_rank_ok"] == "0"


def test_bad_batch_or_host_is_refused(tmp_path):
    path = make_clean(tmp_path, [("x.com", True)])
    with pytest.raises(SystemExit):
        pc.main([path, "--batch", "bad batch;rm", "--cache", str(tmp_path / "c.json")])
    with pytest.raises(SystemExit):
        pc.main([path, "--host", "a b", "--cache", str(tmp_path / "c.json")])
