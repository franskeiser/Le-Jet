"""Tests for webapp/deploy/import_rows.py (adds rows that already have features to the website's CSV). No VM, no network."""
from __future__ import annotations

import csv
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from webapp import dataset                                    # noqa: E402

TOOL = os.path.join(ROOT, "webapp", "deploy", "import_rows.py")


def row(text, label="smishing", language="en", link="https://a.example/x"):
    r = {c: "-1" for c in dataset.COLUMNS}
    r.update(message_id="", sms_text=text, label=label, language=language, date_submitted="2026-11-20 10:00",
             link=link, link_domain="a.example", template_id="tabc", link_shortened="0", page_fetched="1", fetch_status="live")
    return r


def write(path, rows, header=None):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header or dataset.COLUMNS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(tmp_path, rows, header=None):
    src = tmp_path / "rows.csv"
    write(src, rows, header)
    env = dict(os.environ, LEJET_WEB_ROOT=ROOT, LEJET_DATASET=str(tmp_path / "web.csv"))
    return subprocess.run([sys.executable, "-I", TOOL, str(src)], capture_output=True, text=True, env=env)


def test_adds_rows_with_new_ids_skips_duplicates_and_bad_values(tmp_path):
    first = run(tmp_path, [row("hello one"), row("hello two", "legitimate", "taglish"), row("hello one")])
    assert first.returncode == 0, first.stderr
    again = run(tmp_path, [row("hello two", "legitimate", "taglish"), row("three"), row("", "smishing"),
                           row("bad", "spam"), row("worse", "smishing", "fr")])
    assert again.returncode == 0, again.stderr
    rows = list(csv.DictReader(open(tmp_path / "web.csv", encoding="utf-8")))
    assert [r["message_id"] for r in rows] == ["web_000001", "web_000002", "web_000003"]
    assert [r["sms_text"] for r in rows] == ["hello one", "hello two", "three"]
    assert "duplicate': 1" in again.stdout and "bad_value': 3" in again.stdout


def test_a_file_with_the_wrong_columns_changes_nothing(tmp_path):
    out = run(tmp_path, [row("x")], header=dataset.COLUMNS[:-1])
    assert out.returncode == 3
    assert list(csv.DictReader(open(tmp_path / "web.csv", encoding="utf-8"))) == [] if (tmp_path / "web.csv").exists() else True


def test_spreadsheet_formulas_are_made_inert_like_a_normal_submission(tmp_path):
    out = run(tmp_path, [row("=1+1 click")])
    assert out.returncode == 0, out.stderr
    stored = list(csv.DictReader(open(tmp_path / "web.csv", encoding="utf-8")))[0]["sms_text"]
    assert not stored.startswith("=")


def run_update(tmp_path, rows):
    src = tmp_path / "upd.csv"
    write(src, rows)
    env = dict(os.environ, LEJET_WEB_ROOT=ROOT, LEJET_DATASET=str(tmp_path / "web.csv"))
    return subprocess.run([sys.executable, "-I", TOOL, "--update", str(src)], capture_output=True, text=True, env=env)


def test_update_mode_replaces_only_the_inspection_result_of_matching_rows(tmp_path):
    first = row("kept as is", "legitimate", "taglish", link="https://a.example/x")
    first.update(page_fetched="0", fetch_status="dead", nb_hyperlinks="-1")
    other = row("untouched", "smishing", "en", link="https://b.example/y")
    assert run(tmp_path, [first, other]).returncode == 0
    better = dict(first, page_fetched="1", fetch_status="live", nb_hyperlinks="42", link_domain="a.example",
                  label="smishing", language="en", date_submitted="2099-01-01 00:00")      # these must NOT change
    nomatch = row("not in the file", link="https://c.example/z")
    out = run_update(tmp_path, [better, nomatch])
    assert out.returncode == 0, out.stderr
    assert "changed 1" in out.stdout and "matched nothing 1" in out.stdout
    rows = list(csv.DictReader(open(tmp_path / "web.csv", encoding="utf-8")))
    assert [r["message_id"] for r in rows] == ["web_000001", "web_000002"]
    a = rows[0]
    assert (a["page_fetched"], a["fetch_status"], a["nb_hyperlinks"]) == ("1", "live", "42")
    assert (a["label"], a["language"], a["date_submitted"]) == ("legitimate", "taglish", "2026-11-20 10:00")
    assert rows[1]["sms_text"] == "untouched" and rows[1]["nb_hyperlinks"] == "-1"
    again = run_update(tmp_path, [better])
    assert "changed 0" in again.stdout and "already identical 1" in again.stdout
