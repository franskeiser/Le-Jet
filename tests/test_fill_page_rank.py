"""
Tests for scripts/fill_page_rank.py (adds page_rank to the website's CSV on the host) against a FAKE local Open PageRank
API. No internet, no real key.

Run with:  python -m pytest tests/test_fill_page_rank.py
"""

from __future__ import annotations

import csv
import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.parse

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import fill_page_rank as fpr                                 # noqa: E402

SECRET = "TESTKEY-abcdef123456"
RANKS = {"example.com": 7, "bit.ly": 9, "unrated.example.org": None}


class _API(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass

    def do_GET(self):
        srv = self.server
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        srv.asked.extend(query.get("domains[]", []))
        if srv.mode == "500" or self.headers.get("API-OPR") != SECRET:
            self.send_response(500)
            self.end_headers()
            return
        items = [{"domain": d, "status_code": 200, "page_rank_integer": RANKS.get(d)} for d in query.get("domains[]", [])]
        body = json.dumps({"response": items}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def api():
    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    srv = Server(("127.0.0.1", 0), _API)
    srv.asked, srv.mode = [], "ok"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}/getPageRank"
    yield srv
    srv.shutdown()
    srv.server_close()


COLUMNS = ["message_id", "sms_text", "label", "language", "link", "link_domain", "page_fetched", "fetch_status", "nb_hyperlinks", "page_rank"]


def _csv(tmp_path, rows, name="web.csv"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for i, (dom, fetched) in enumerate(rows, 1):
            w.writerow({"message_id": f"web_{i:06d}", "sms_text": f"line one\nline two, with a comma https://{dom}/x", "label": "smishing",
                        "language": "en", "link": f"https://{dom}/x", "link_domain": dom, "page_fetched": fetched,
                        "fetch_status": "live" if fetched else "dead", "nb_hyperlinks": 3 if fetched else -1, "page_rank": -1})
    return path


def _env(tmp_path):
    p = tmp_path / ".env"
    p.write_text(f"OPENPAGERANK_API_KEY={SECRET}\n", encoding="ascii")
    return str(p)


def _run(tmp_path, api, csv_path, *extra):
    return fpr.main([str(csv_path), "--env-file", _env(tmp_path), "--cache", str(tmp_path / "cache.json"),
                     "--base-url", api.url, "--pause", "0", *extra])


def _read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_only_pages_that_loaded_are_looked_up(tmp_path, api, capsys):
    src = _csv(tmp_path, [("example.com", 1), ("bit.ly", 0), ("unrated.example.org", 1), ("example.com", 1)])
    before = src.read_text(encoding="utf-8")
    assert _run(tmp_path, api, src) == 0
    out = _read(str(src) + ".with_page_rank.csv")
    assert [r["page_rank"] for r in out] == ["7", "-1", "0", "7"]            # dead row untouched, unrated = 0, repeats share one answer
    assert sorted(api.asked) == ["example.com", "unrated.example.org"]        # the dead page's domain was NOT sent
    assert src.read_text(encoding="utf-8") == before                          # the input is never changed
    text = capsys.readouterr().out + (tmp_path / "cache.json").read_text()
    assert SECRET not in text and SECRET not in open(str(src) + ".with_page_rank.csv", encoding="utf-8").read()


def test_every_other_cell_is_copied_exactly(tmp_path, api):
    src = _csv(tmp_path, [("example.com", 1)])
    _run(tmp_path, api, src)
    a, b = _read(src)[0], _read(str(src) + ".with_page_rank.csv")[0]
    assert {k: v for k, v in b.items() if k != "page_rank"} == {k: v for k, v in a.items() if k != "page_rank"}
    assert b["sms_text"] == "line one\nline two, with a comma https://example.com/x"
    assert list(b) == COLUMNS


def test_all_looks_up_rows_whose_page_was_not_fetched_too(tmp_path, api):
    src = _csv(tmp_path, [("example.com", 1), ("bit.ly", 0)])
    assert _run(tmp_path, api, src, "--all") == 0
    assert [r["page_rank"] for r in _read(str(src) + ".with_page_rank.csv")] == ["7", "9"]


def test_a_failed_lookup_stays_minus_one_never_zero(tmp_path, api):
    api.mode = "500"
    src = _csv(tmp_path, [("example.com", 1), ("bit.ly", 0)])
    assert _run(tmp_path, api, src) == 3                                      # reported as incomplete
    out = _read(str(src) + ".with_page_rank.csv")                             # the file is still written
    assert [r["page_rank"] for r in out] == ["-1", "-1"]
    assert not (tmp_path / "cache.json").exists() or "example.com" not in json.loads((tmp_path / "cache.json").read_text())


def test_answers_are_cached_so_a_domain_is_asked_once(tmp_path, api):
    src = _csv(tmp_path, [("example.com", 1)])
    _run(tmp_path, api, src)
    asked = list(api.asked)
    _run(tmp_path, api, src, "--out", str(tmp_path / "again.csv"))
    assert api.asked == asked                                                # nothing new was sent
    assert _read(str(tmp_path / "again.csv"))[0]["page_rank"] == "7"


def test_wrong_file_and_unsafe_choices_are_refused(tmp_path, api):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(tmp_path, api, bad)                                             # not the website's CSV
    src = _csv(tmp_path, [("example.com", 1)])
    with pytest.raises(SystemExit):
        fpr.main([str(src), "--out", str(src), "--env-file", _env(tmp_path)])    # would overwrite the input
    with pytest.raises(SystemExit):
        fpr.main([str(src), "--env-file", _env(tmp_path), "--base-url", "http://example.com/x"])   # http to a real host


def test_nothing_to_look_up_needs_no_key(tmp_path, api):
    src = _csv(tmp_path, [("bit.ly", 0)])                                    # only an unreached page: no request, no key needed
    assert fpr.main([str(src), "--env-file", str(tmp_path / "missing.env"), "--cache", str(tmp_path / "c.json"),
                     "--base-url", api.url]) == 0
    assert api.asked == [] and _read(str(src) + ".with_page_rank.csv")[0]["page_rank"] == "-1"
