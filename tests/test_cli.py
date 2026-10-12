"""
Tests for the command line (WEB_INSPECTION_PLAN.md sections 8.1, 12.3, 16): input parsing, the
preflight-token gate, and a complete `run` + `export` in loopback test mode against the local
web server. No internet.

Run with:  python -m pytest tests/test_cli.py
"""

from __future__ import annotations

import gzip
import json
import os
import time

import pytest

from lejet import cli, schema, urlnorm


def line(raw):
    return f"{urlnorm.compute_url_id(raw)}\t{raw}"


# ---------------------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------------------
def write(tmp_path, lines, name="urls.txt", newline="\n"):
    p = tmp_path / name
    p.write_bytes(newline.join(lines).encode("utf-8"))
    return str(p)


def test_read_input_good_lines(tmp_path):
    path = write(tmp_path, ["# comment", "", line("https://a.test/"), line("http://b.test/x?y=1"), "   "])
    items, sha, problems = cli.read_input(path)
    assert problems == [] and len(items) == 2 and len(sha) == 64
    assert items[0] == (urlnorm.compute_url_id("https://a.test/"), "https://a.test/")


def test_read_input_handles_windows_line_endings(tmp_path):
    path = write(tmp_path, [line("https://a.test/"), line("https://b.test/")], newline="\r\n")
    items, _, problems = cli.read_input(path)
    assert problems == [] and [r for _, r in items] == ["https://a.test/", "https://b.test/"]


def test_read_input_drops_duplicates_keeping_the_first(tmp_path):
    path = write(tmp_path, [line("https://a.test/"), line("https://b.test/"), line("https://a.test/")])
    items, _, problems = cli.read_input(path)
    assert problems == [] and len(items) == 2


def test_read_input_reports_bad_lines(tmp_path):
    good = line("https://a.test/")
    path = write(tmp_path, [good, "no-tab-here", "XYZ\thttps://x.test/", "0123456789abcdef\thttps://x.test/",
                            "0123456789ABCDEF\thttps://x.test/"])
    items, _, problems = cli.read_input(path)
    assert len(items) == 1 and len(problems) == 4
    assert any("expected url_id<TAB>url" in p for p in problems)
    assert any("does not match" in p for p in problems)


def test_read_input_can_skip_id_verification(tmp_path):
    path = write(tmp_path, ["0123456789abcdef\thttps://x.test/"])
    items, _, problems = cli.read_input(path, verify_ids=False)
    assert problems == [] and items == [("0123456789abcdef", "https://x.test/")]


def test_read_input_sha_is_stable_and_content_sensitive(tmp_path):
    a = write(tmp_path, [line("https://a.test/")], name="a.txt")
    b = write(tmp_path, [line("https://b.test/")], name="b.txt")
    assert cli.read_input(a)[1] == cli.read_input(a)[1] != cli.read_input(b)[1]


def test_urls_keep_tabs_inside_the_url_out_of_the_id_field(tmp_path):
    raw = "https://a.test/path"
    path = write(tmp_path, [f"{urlnorm.compute_url_id(raw)}\t{raw}"])
    assert cli.read_input(path)[0][0][1] == raw


# ---------------------------------------------------------------------------------------
# preflight token
# ---------------------------------------------------------------------------------------
def token(tmp_path, content):
    p = tmp_path / "tok"
    p.write_text(content)
    return str(p)


def test_preflight_token_states(tmp_path):
    now = time.time()
    assert cli.preflight_ok(token(tmp_path, str(int(now))), 3600)[0] is True
    ok, why = cli.preflight_ok(token(tmp_path, str(int(now - 7200))), 3600)
    assert not ok and "old" in why
    ok, why = cli.preflight_ok(token(tmp_path, str(int(now + 3600))), 3600)
    assert not ok and "future" in why
    assert not cli.preflight_ok(token(tmp_path, "garbage"), 3600)[0]
    assert not cli.preflight_ok(token(tmp_path, ""), 3600)[0]
    ok, why = cli.preflight_ok(str(tmp_path / "missing"), 3600)
    assert not ok and "not found" in why


def test_run_refuses_to_start_without_a_fresh_token(tmp_path, capsys):
    inp = write(tmp_path, [line("https://a.test/")])
    rc = cli.main(["run", "--input", inp, "--db", str(tmp_path / "d.sqlite"), "--batch-id", "b1",
                   "--preflight-token", str(tmp_path / "nope")])
    assert rc == 3
    assert "REFUSING TO START" in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "d.sqlite")                 # nothing was even created


def test_check_token_command(tmp_path, capsys):
    assert cli.main(["check-token", "--preflight-token", token(tmp_path, str(int(time.time())))]) == 0
    assert cli.main(["check-token", "--preflight-token", str(tmp_path / "missing")]) == 1


def test_run_with_bad_input_exits_2_and_does_not_fetch(tmp_path, capsys, server):
    inp = write(tmp_path, [line(server.url("/page/plain")), "garbage line"])
    rc = cli.main(["run", "--input", inp, "--db", str(tmp_path / "d.sqlite"), "--batch-id", "b1",
                   "--allow-loopback", "--workers", "0"])
    assert rc == 2 and server.hits == []


def test_run_with_an_empty_input_exits_2(tmp_path):
    inp = write(tmp_path, ["# nothing here"])
    assert cli.main(["run", "--input", inp, "--db", str(tmp_path / "d.sqlite"), "--batch-id", "b1",
                     "--allow-loopback", "--workers", "0"]) == 2


# ---------------------------------------------------------------------------------------
# a complete run + export (loopback test mode, real worker process)
# ---------------------------------------------------------------------------------------
def test_run_then_export_end_to_end(tmp_path, server, capsys):
    paths = ["/page/plain", "/page/html_login_empty_action", "/404", "/redirect/2", "/page/parked", "/loop"]
    raws = [server.url(p) for p in paths] + ["ftp://x.test/"]
    inp = write(tmp_path, [line(r) for r in raws])
    db = str(tmp_path / "out.sqlite")
    common = ["--db", db, "--batch-id", "b0001"]

    rc = cli.main(["run", "--input", inp, *common, "--allow-loopback", "--workers", "1",
                   "--rate", "1000", "--jitter", "0", "--seed", "3"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["n_input"] == 7
    assert summary["status_counts"] == {
        "done": 3, "failed:http_4xx": 1, "failed:parked_or_default_page": 1,
        "failed:too_many_redirects": 1, "failed:rejected_url": 1}

    # a second run has nothing to do
    hits = len(server.hits)
    assert cli.main(["run", "--input", inp, *common, "--allow-loopback", "--workers", "0",
                     "--rate", "1000", "--jitter", "0"]) == 0
    assert len(server.hits) == hits

    out_dir = str(tmp_path / "export")
    assert cli.main(["export", *common[:2], "--out-dir", out_dir, "--batch-id", "b0001",
                     "--vpn-provider", "Mullvad", "--vpn-exit-country", "PH",
                     "--preflight-token", str(tmp_path / "missing")]) == 0
    info = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    records = [json.loads(l) for l in gzip.decompress(open(info["jsonl"], "rb").read()).decode().splitlines()]
    assert len(records) == 7
    by_url = {r["url"]: r for r in records}
    plain = by_url[schema.safety.defang_url(server.url("/page/plain"))]
    assert plain["fetch_ok"] is True and plain["page_fetched"] == 1 and plain["fail_reason"] == "none"
    login = by_url[schema.safety.defang_url(server.url("/page/html_login_empty_action"))]
    assert login["features"]["login_form"] == 1
    manifest = json.load(open(info["manifest"], encoding="utf-8"))
    assert manifest["n_done"] == 3 and manifest["vpn_exit_country"] == "PH"
    assert manifest["preflight_pass"] is False                      # no token in this test
    assert manifest["tool_versions"]["config_hash"]
    assert manifest["input_sha256"] == cli.read_input(inp)[1]
