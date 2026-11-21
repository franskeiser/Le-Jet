"""Tests for the Azure FEATURES-ONLY mode (src/lejet/azure_features.py, scripts/run_parts_azure.py, infra/azure/lejet-azure-features.sh).

The real Pipeline, Storage and export run against the local fixture server (loopback test mode); there is no browser in this mode at
all, which is the point: no screenshot code is imported or started. No VM, no internet.

Run with:  python -m pytest tests/test_azure_features.py
"""
from __future__ import annotations

import asyncio
import gzip
import importlib.util
import json
import os

import pytest

from lejet import azure_features as af
from lejet import urlnorm
from lejet.settings import Settings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def push(path, urls):
    path.write_text("".join(f"{urlnorm.compute_url_id(u)}\t{u}\n" for u in urls), encoding="ascii")


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    out = tmp_path / "out"
    inp = tmp_path / "urls.txt"
    monkeypatch.setattr(af, "preflight_ok", lambda token, max_age: (True, "ok"))
    monkeypatch.setattr(af, "settings_for_features", lambda: Settings(
        allow_loopback=True, parse_workers=0, rate=1000.0, jitter=0.0, concurrency=3, rdap_min_interval=0.0,
        hop_deadline=8, url_deadline=15, connect_timeout=3, ct_enabled=False, ct_probe=False))
    return dict(out=str(out), inp=str(inp), path=inp)


def run(lab, batch="feat01"):
    return asyncio.run(af.amain(batch, lab["inp"], lab["out"]))


def rows_of(lab, batch="feat01"):
    with gzip.open(os.path.join(lab["out"], "export", f"{batch}.jsonl.gz"), "rt") as fh:
        return [json.loads(x) for x in fh]


def test_many_links_one_call_the_three_result_files_and_no_picture_files(server, lab):
    urls = [server.url(p) for p in ("/page/html_all_internal", "/page/plain", "/404")]
    push(lab["path"], urls)
    assert run(lab) == 0
    export = os.path.join(lab["out"], "export")
    names = sorted(os.listdir(export))
    assert names == ["feat01.jsonl.gz", "feat01.jsonl.gz.sha256", "feat01.manifest.json"]       # no .shots.tar, no shots- folder
    rows = rows_of(lab)
    assert sorted(r["url_id"] for r in rows) == sorted(urlnorm.compute_url_id(u) for u in urls)
    by_id = {r["url_id"]: r for r in rows}
    ok = by_id[urlnorm.compute_url_id(urls[0])]
    assert ok["page_fetched"] == 1 and ok["fail_reason"] == "none" and ok["features"]["nb_hyperlinks"] == 10
    assert len(ok["features_full"]) == 73                                                        # the full export the dataset builder needs
    assert by_id[urlnorm.compute_url_id(urls[2])]["page_fetched"] == 0                            # a dead link is a row, not a failure


def test_the_manifest_says_azure_and_that_the_self_test_passed(server, lab):
    push(lab["path"], [server.url("/page/plain")])
    assert run(lab) == 0
    flat = json.dumps(json.load(open(os.path.join(lab["out"], "export", "feat01.manifest.json"))))
    assert "azure" in flat and '"preflight_pass": true' in flat


def test_no_fresh_self_test_means_nothing_is_fetched(server, lab, monkeypatch):
    monkeypatch.setattr(af, "preflight_ok", lambda token, max_age: (False, "stale"))
    push(lab["path"], [server.url("/page/plain")])
    before = len(server.hits)
    assert run(lab) == 3
    assert len(server.hits) == before and not os.path.exists(lab["out"])


def test_bad_input_is_refused_before_any_fetch(server, lab):
    before = len(server.hits)
    lab["path"].write_text("0123456789abcdef\t" + server.url("/page/plain") + "\n", encoding="ascii")      # id is not sha256(url)
    assert run(lab) == 2
    lab["path"].write_text(server.url("/page/plain") + "\n", encoding="ascii")                           # no id at all
    assert run(lab) == 2
    lab["path"].write_text("", encoding="ascii")
    assert run(lab) == 2
    assert len(server.hits) == before


def test_more_than_the_cap_is_refused(server, lab, monkeypatch):
    monkeypatch.setattr(af, "MAX_LINKS", 2)
    push(lab["path"], [server.url(f"/page/plain?n={i}") for i in range(3)])
    assert run(lab) == 2


def test_a_batch_id_is_never_reused(server, lab):
    push(lab["path"], [server.url("/page/plain")])
    assert run(lab) == 0
    os.makedirs(lab["out"], exist_ok=True)
    open(os.path.join(lab["out"], "feat02.sqlite"), "wb").close()
    assert run(lab, "feat02") == 2


def test_the_program_does_not_import_the_browser_code():
    src = open(os.path.join(ROOT, "src", "lejet", "azure_features.py"), encoding="utf-8").read()
    assert "import shot_stage" not in src and "ShotRunner" not in src and "import playwright" not in src.lower()


def test_the_root_helper_only_takes_checked_input_and_a_cap():
    sh = open(os.path.join(ROOT, "infra", "azure", "lejet-azure-features.sh"), encoding="utf-8").read()
    assert "MAX_LINKS=400" in sh and "lejet.azure_features" in sh and "azure_run" not in sh
    assert "[A-Za-z0-9_-]{1,40}" in sh and "-P '^[0-9a-f]{16}\\t\\S+$'" in sh.replace("grep -cP", "-P") or "[0-9a-f]{16}" in sh
    assert "LOCK=/run/lejet-azure-live.lock" in sh                      # one inspection at a time, shared with the live mode
    assert "InaccessiblePaths=/var/lib/lejetweb" in sh and "MemoryMax=1500M" in sh


def test_the_sudoers_rule_allows_the_helper_with_a_batch_id_only():
    s = open(os.path.join(ROOT, "infra", "azure", "92-lejet-azure.sudoers"), encoding="utf-8").read()
    assert "/usr/local/sbin/lejet-azure-features [A-Za-z0-9_-]*" in s and "lejet-azure-live [A-Za-z0-9_-]*" in s


# ---------------------------------------------------------------------------------- the PC-side runner
def test_run_parts_azure_names_batches_like_run_parts_and_skips_finished_ones(tmp_path):
    rp = load_script("run_parts_azure")
    d = tmp_path
    for n in (1, 2, 3):
        (d / f"feature_input_20261112_031122_part{n}of3.txt").write_text(f"hxxps://a{n}[.]example/\nhxxps://b{n}[.]example/\n")
    (d / "feature_input_20261001_000000_part1of1.txt").write_text("hxxps://old[.]example/\n")
    assert rp.newest_stamp(d) == "20261112_031122"
    jobs = list(rp.jobs(d, "20261112_031122"))
    assert [(b, k) for b, _f, k in jobs] == [("feat20261112031122p1", 2), ("feat20261112031122p2", 2), ("feat20261112031122p3", 2)]
    assert [b for b, _f, _k in rp.jobs(d, "20261112_031122", first=2, last=2)] == ["feat20261112031122p2"]


def test_run_parts_azure_refuses_a_part_the_vm_would_refuse(tmp_path):
    rp = load_script("run_parts_azure")
    (tmp_path / "feature_input_20261112_031122_part1of1.txt").write_text("\n".join(f"hxxps://a{i}[.]example/" for i in range(401)))
    with pytest.raises(SystemExit):
        list(rp.jobs(tmp_path, "20261112_031122"))


def test_run_parts_azure_dry_run_starts_nothing(tmp_path, monkeypatch, capsys):
    rp = load_script("run_parts_azure")
    (tmp_path / "feature_input_20261112_031122_part1of1.txt").write_text("hxxps://a[.]example/\n")
    monkeypatch.setattr(rp, "CLEAN", tmp_path / "clean")
    called = []
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: called.append(a))
    assert rp.main(["--datasets", str(tmp_path), "--dry-run"]) == 0
    assert called == [] and "would run feat20261112031122p1: 1 links" in capsys.readouterr().out
