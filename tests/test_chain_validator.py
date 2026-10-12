"""
The whole chain in one test, using only local fixtures (no internet):

    urls.txt  ->  make_push_file  ->  lejet.cli run  ->  lejet.cli export  ->  validate_output

Whatever the real pipeline and exporter write must pass the strict host-side validator, and every
pushed url_id must come back exactly once. If the producer and the validator ever disagree about
the format, this is the test that fails.

Run with:  python -m pytest tests/test_chain_validator.py
"""

from __future__ import annotations

import gzip
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_push_file as mp                                  # noqa: E402
import validate_output as vo                                 # noqa: E402
from lejet import cli                                       # noqa: E402


def test_pipeline_output_passes_the_strict_validator(tmp_path, server, capsys):
    paths = ["/page/plain", "/page/html_login_empty_action", "/page/html_media_mix", "/404", "/500",
             "/redirect/3", "/loop", "/to-private", "/page/challenge", "/page/parked", "/json", "/empty",
             "/cookie-loop", "/page/evil_strings", "/page/js_redirect"]
    urls = [server.url(p) for p in paths] + ["ftp://x.test/f", "http://10.0.0.1/admin", "http://example.com:8080/x",
                                              "hxxp://127[.]0[.]0[.]1:1/ignored-by-gate"]
    src = tmp_path / "urls.txt"
    src.write_text("\n".join(urls) + "\n", encoding="utf-8")
    push = str(tmp_path / "urls_push.txt")
    assert mp.main([str(src), "--out", push]) == 0
    capsys.readouterr()

    db = str(tmp_path / "b0042.sqlite")
    assert cli.main(["run", "--input", push, "--db", db, "--batch-id", "b0042", "--allow-loopback",
                     "--workers", "1", "--rate", "1000", "--jitter", "0"]) == 0
    capsys.readouterr()
    out_dir = str(tmp_path / "export")
    assert cli.main(["export", "--db", db, "--out-dir", out_dir, "--batch-id", "b0042"]) == 0
    capsys.readouterr()

    jsonl = os.path.join(out_dir, "b0042.jsonl.gz")
    clean_dir = str(tmp_path / "clean")
    rc = vo.main([jsonl, "--input", push, "--clean-dir", clean_dir, "--csv", str(tmp_path / "b0042.csv")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"VALIDATION OK: {len(urls)} rows, sha256 verified" in out

    clean = [json.loads(l) for l in open(os.path.join(clean_dir, "b0042.clean.jsonl"), encoding="ascii")]
    assert len(clean) == len(urls)
    reasons = {r["fail_reason"] for r in clean}
    assert {"none", "http_4xx", "http_5xx", "too_many_redirects", "redirect_to_blocked", "challenge_page",
            "parked_or_default_page", "non_html", "blocked_or_empty", "rejected_url", "port_blocked"} <= reasons
    # a challenge page must route to url_only; a fetched page to full
    by_reason = {}
    for r in clean:
        by_reason.setdefault(r["fail_reason"], []).append(r)
    assert all(r["page_fetched"] == 0 for r in by_reason["challenge_page"])
    assert all(r["page_fetched"] == 1 for r in by_reason["none"] if r["html_available"])
    # nothing clickable anywhere in what left the VM
    raw = gzip.decompress(open(jsonl, "rb").read())
    assert b"http://" not in raw and b"https://" not in raw


def test_the_validator_catches_a_pipeline_row_that_was_tampered_with_after_export(tmp_path, server, capsys):
    src = tmp_path / "urls.txt"
    src.write_text(server.url("/page/plain") + "\n", encoding="utf-8")
    push = str(tmp_path / "p.txt")
    mp.main([str(src), "--out", push])
    db = str(tmp_path / "b.sqlite")
    cli.main(["run", "--input", push, "--db", db, "--batch-id", "b1", "--allow-loopback", "--workers", "0",
              "--rate", "1000", "--jitter", "0"])
    cli.main(["export", "--db", db, "--out-dir", str(tmp_path / "e"), "--batch-id", "b1"])
    capsys.readouterr()
    jsonl = str(tmp_path / "e" / "b1.jsonl.gz")
    rec = json.loads(gzip.decompress(open(jsonl, "rb").read()).decode())
    rec["features"]["nb_hyperlinks"] = "=cmd|' /C calc'!A0"
    tampered = str(tmp_path / "t.jsonl.gz")
    with gzip.open(tampered, "wb") as fh:
        fh.write((json.dumps(rec) + "\n").encode())
    assert vo.main([tampered]) == 1                            # tampered content is refused
    assert vo.main([jsonl, "--sha256-file", jsonl + ".sha256"]) == 0
    flipped = open(jsonl, "rb").read()
    open(str(tmp_path / "f.jsonl.gz"), "wb").write(flipped[:-5] + bytes([flipped[-5] ^ 1]) + flipped[-4:])
    assert vo.main([str(tmp_path / "f.jsonl.gz"), "--sha256-file", jsonl + ".sha256"]) == 1
