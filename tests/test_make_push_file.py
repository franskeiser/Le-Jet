"""
Tests for scripts/make_push_file.py (WEB_INSPECTION_PLAN.md sections 12.3, 13.2).

Run with:  python -m pytest tests/test_make_push_file.py
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_push_file as mp                                  # noqa: E402
from lejet import cli, urlnorm                              # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("hxxps://evil[.]example[.]com/login", "https://evil.example.com/login"),
    ("HXXP://a(.)b(.)c/x", "http://a.b.c/x"),
    ("hxxp[:]//a[dot]b/x", "http://a.b/x"),
    ("  gcash-verify[.]xyz/otp  ", "gcash-verify.xyz/otp"),
    ("https://plain.test/a", "https://plain.test/a"),
    ("bit.ly/abc", "bit.ly/abc"),
])
def test_refang(raw, expected):
    assert mp.refang(raw) == expected


def test_build_ids_dedup_and_rejections():
    lines = ["# header", "", "hxxps://a[.]test/x", "https://a.test/x", "bit.ly/abc", "has a space.com",
             "x" * 3000, "tab\tinside", "gcash-verify[.]xyz/otp", "   "]
    items, report = mp.build(lines)
    urls = [u for _, u in items]
    assert sorted(urls) == sorted(["https://a.test/x", "bit.ly/abc", "gcash-verify.xyz/otp"])
    assert report["kept"] == 3 and report["duplicates"] == 1 and report["blank_or_comment"] == 3
    assert report["rejected"] == 3
    assert report["rejected_reasons"] == {"longer_than_2048": 1, "whitespace_or_control_characters": 2}
    for url_id, url in items:
        assert url_id == urlnorm.compute_url_id(url)
    assert [i for i, _ in items] == sorted(i for i, _ in items)


def test_outputs_are_readable_by_the_vm_cli_and_the_view_copy_is_defanged(tmp_path):
    src = tmp_path / "urls.txt"
    src.write_text("hxxps://evil[.]example[.]com/login\nbit.ly/abc\n", encoding="utf-8")
    out = tmp_path / "push" / "urls_push.txt"
    assert mp.main([str(src), "--out", str(out)]) == 0
    items, _, problems = cli.read_input(str(out))
    assert problems == [] and len(items) == 2                # the VM-side parser accepts it
    view = (tmp_path / "push" / "urls_push.txt.defanged.txt").read_text()
    assert "http://" not in view and "https://" not in view
    assert "hxxps://evil[.]example[.]com/login" in view
    report = json.loads((tmp_path / "push" / "urls_push.txt.report.json").read_text())
    assert report["kept"] == 2
    assert b"\r" not in out.read_bytes()                     # LF only


def test_empty_input_exits_nonzero(tmp_path):
    src = tmp_path / "u.txt"
    src.write_text("# nothing\n")
    assert mp.main([str(src), "--out", str(tmp_path / "o.txt")]) == 1
