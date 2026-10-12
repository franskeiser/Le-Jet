"""
Tests for scripts/lookup_page_rank.py (WEB_INSPECTION_PLAN.md section 14.0.2) against a FAKE local
API. No internet, no real key.

Run with:  python -m pytest tests/test_page_rank_lookup.py
"""

from __future__ import annotations

import csv
import gzip
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

import lookup_page_rank as lp                                # noqa: E402
from lejet import schema, urlnorm                           # noqa: E402

SECRET = "TESTKEY-abcdef123456"
RANKS = {"example.com": 7, "bit.ly": 9, "unrated.example.org": None, "badscale.test": 55}


class _API(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass

    def do_GET(self):
        srv = self.server
        srv.requests.append((self.path, dict(self.headers)))
        if srv.mode == "403" or self.headers.get("API-OPR") != SECRET:
            self.send_response(403)
            self.end_headers()
            return
        if srv.mode == "500":
            self.send_response(500)
            self.end_headers()
            return
        if srv.mode == "garbage":
            body = b"<html>not json</html>"
        else:
            doms = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("domains[]", [])
            items = []
            for d in doms:
                if d in RANKS and RANKS[d] is not None:
                    items.append({"status_code": 200, "page_rank_integer": RANKS[d], "domain": d})
                elif d in RANKS:
                    items.append({"status_code": 404, "page_rank_integer": None, "domain": d})
            body = json.dumps({"status_code": 200, "response": items}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _API)
        self.requests = []
        self.mode = "ok"


@pytest.fixture
def api():
    s = _Srv()
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    s.url = f"http://127.0.0.1:{s.server_port}/api/v1.0/getPageRank"
    yield s
    s.shutdown()
    s.server_close()


def clean_file(tmp_path, domains):
    recs = []
    for i, dom in enumerate(domains):
        raw = f"https://{dom}/p{i}"
        row = schema.new_row(urlnorm.compute_url_id(raw), raw, "b001")
        row.update(fetch_ok=1, html_available=1, page_fetched=1, final_reg_domain=dom, nb_hyperlinks=3,
                   ratio_intHyperlinks=0.5, links_in_tags=0.0, ratio_intMedia=0.0, ratio_extMedia=0.0,
                   nb_extCSS=0, login_form=0, external_favicon=0, iframe=0, empty_title=0,
                   domain_in_title=1, domain_with_copyright=0, rdap_ok=0)
        recs.append(schema.export_record(row))
    path = str(tmp_path / "b001.clean.jsonl")
    with open(path, "w", encoding="ascii") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return path


def run(tmp_path, api, domains, extra=()):
    key_file = tmp_path / "key.txt"
    key_file.write_text(SECRET)
    out = str(tmp_path / "out.csv")
    rc = lp.main([clean_file(tmp_path, domains), "--out", out, "--key-file", str(key_file),
                  "--base-url", api.url, "--pause", "0", *extra])
    rows = list(csv.DictReader(open(out, encoding="utf-8"))) if os.path.exists(out) else []
    return rc, rows


def test_ranks_unrated_and_columns(tmp_path, api):
    rc, rows = run(tmp_path, api, ["example.com", "unrated.example.org", "bit.ly"])
    assert rc == 0
    by = {r["final_reg_domain"].replace("[.]", "."): r for r in rows}
    assert by["example.com"]["page_rank"] == "7" and by["example.com"]["page_rank_ok"] == "1"
    assert by["bit.ly"]["page_rank"] == "9"
    assert by["unrated.example.org"]["page_rank"] == "0"           # answered but unrated -> 0, ok
    assert by["unrated.example.org"]["page_rank_ok"] == "1"
    expected_cols = (["url_id", "has_url", "page_fetched"] + schema.URL_TEXT_FEATURES + ["page_rank"]
                     + schema.WEBPAGE_FEATURES + schema.EXTERNAL_FEATURES)
    assert list(rows[0])[:len(expected_cols)] == expected_cols    # what route_p_u() reads, in order


def test_a_domain_the_service_did_not_answer_is_minus_one_not_zero(tmp_path, api):
    rc, rows = run(tmp_path, api, ["example.com", "never-answered.test"])
    by = {r["final_reg_domain"].replace("[.]", "."): r for r in rows}
    assert by["never-answered.test"]["page_rank"] == "-1" and by["never-answered.test"]["page_rank_ok"] == "0"


def test_a_different_scale_is_a_failure_not_a_value(tmp_path, api):
    rc, rows = run(tmp_path, api, ["badscale.test"])
    assert rows[0]["page_rank"] == "-1" and rows[0]["page_rank_ok"] == "0"


@pytest.mark.parametrize("mode", ["403", "500", "garbage"])
def test_service_failures_are_minus_one_and_a_nonzero_exit(tmp_path, api, mode):
    api.mode = mode
    rc, rows = run(tmp_path, api, ["example.com"])
    assert rc == 3
    assert rows[0]["page_rank"] == "-1" and rows[0]["page_rank_ok"] == "0"


def test_the_key_is_sent_as_a_header_and_never_in_the_url_or_the_output(tmp_path, api, capsys):
    rc, rows = run(tmp_path, api, ["example.com"])
    assert all(SECRET not in path for path, _ in api.requests)
    sent = {{k.lower(): v for k, v in h.items()}.get("api-opr") for _, h in api.requests}   # names are case-insensitive
    assert sent == {SECRET}
    printed = capsys.readouterr()
    assert SECRET not in printed.out and SECRET not in printed.err
    assert SECRET not in open(tmp_path / "out.csv", encoding="utf-8").read()


def test_wrong_key_never_leaks_into_the_error(tmp_path, api, capsys):
    key_file = tmp_path / "k.txt"
    key_file.write_text("WRONGKEY-123456789")
    out = str(tmp_path / "o.csv")
    lp.main([clean_file(tmp_path, ["example.com"]), "--out", out, "--key-file", str(key_file),
             "--base-url", api.url, "--pause", "0"])
    text = "".join(capsys.readouterr())
    assert "WRONGKEY-123456789" not in text


def test_cache_avoids_repeat_requests_and_never_stores_failures(tmp_path, api):
    cache = str(tmp_path / "cache.json")
    run(tmp_path, api, ["example.com", "never-answered.test"], extra=["--cache", cache])
    first = len(api.requests)
    saved = json.load(open(cache))
    assert saved == {"example.com": 7}
    run(tmp_path, api, ["example.com"], extra=["--cache", cache])
    assert len(api.requests) == first                            # served from the cache


def test_up_to_100_domains_per_request(tmp_path, api):
    doms = [f"d{i}.example.net" for i in range(230)]
    run(tmp_path, api, doms)
    assert len(api.requests) == 3                                # 100 + 100 + 30


def test_identical_domains_are_asked_for_once(tmp_path, api):
    run(tmp_path, api, ["example.com"] * 5 + ["bit.ly"] * 3)
    queries = [urllib.parse.parse_qs(urllib.parse.urlsplit(p).query)["domains[]"] for p, _ in api.requests]
    assert sorted(d for q in queries for d in q) == ["bit.ly", "example.com"]


def test_non_https_remote_api_is_refused(tmp_path):
    key_file = tmp_path / "k.txt"
    key_file.write_text(SECRET)
    with pytest.raises(SystemExit):
        lp.main([clean_file(tmp_path, ["example.com"]), "--out", str(tmp_path / "o.csv"),
                 "--key-file", str(key_file), "--base-url", "http://openpagerank.example/api"])


def test_key_loading_rules(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENPAGERANK_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        lp.load_key(None)
    bad = tmp_path / "bad.txt"
    bad.write_text("has spaces and ;&| junk")
    with pytest.raises(SystemExit):
        lp.load_key(str(bad))
    monkeypatch.setenv("OPENPAGERANK_API_KEY", "ENVKEY-123456789")
    assert lp.load_key(None) == "ENVKEY-123456789"


ENV_KEY = "ENVFILEKEY-9876543210"


@pytest.mark.parametrize("text", [
    f"OPENPAGERANK_API_KEY={ENV_KEY}\n",
    f"OPENPAGERANK_API_KEY={ENV_KEY}",                                   # no final newline
    f"OPENPAGERANK_API_KEY={ENV_KEY}\r\n",                               # Windows line ending
    f"# Open PageRank\n\nOTHER_THING=ignored\nOPENPAGERANK_API_KEY={ENV_KEY}\n",
    f"  OPENPAGERANK_API_KEY={ENV_KEY}   \n",                            # stray spaces around the whole line
])
def test_an_env_file_gives_the_key(tmp_path, monkeypatch, text):
    monkeypatch.delenv("OPENPAGERANK_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_bytes(text.encode("ascii"))
    assert lp.load_key(None, str(env)) == ENV_KEY


@pytest.mark.parametrize("text, why", [
    ("", "not found"),
    ("OTHER=1\n", "not found"),
    ("# OPENPAGERANK_API_KEY=" + ENV_KEY + "\n", "not found"),            # commented out
    (f"OPENPAGERANK_API_KEY = {ENV_KEY}\n", "not found"),                # spaces around '=' are not accepted
    (f'OPENPAGERANK_API_KEY="{ENV_KEY}"\n', "quotes"),
    (f"OPENPAGERANK_API_KEY='{ENV_KEY}'\n", "quotes"),
    ("OPENPAGERANK_API_KEY=\n", "no API key"),
    ("OPENPAGERANK_API_KEY=short\n", "unexpected format"),
    ("OPENPAGERANK_API_KEY=has spaces inside it\n", "unexpected format"),
    ("OPENPAGERANK_API_KEY=" + "x" * 300 + "\n", "unexpected format"),
])
def test_a_bad_env_file_is_refused_and_never_echoes_the_value(tmp_path, monkeypatch, capsys, text, why):
    monkeypatch.delenv("OPENPAGERANK_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_bytes(text.encode("ascii"))
    with pytest.raises(SystemExit) as exc:
        lp.load_key(None, str(env))
    message = str(exc.value)
    assert why in message
    assert ENV_KEY not in message and "has spaces inside" not in message and "x" * 50 not in message


def test_a_missing_binary_or_huge_env_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENPAGERANK_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        lp.load_key(None, str(tmp_path / "nope.env"))
    binary = tmp_path / "bin.env"
    binary.write_bytes(b"\xff\xfe\x00junk")
    with pytest.raises(SystemExit):
        lp.load_key(None, str(binary))
    huge = tmp_path / "huge.env"
    huge.write_bytes(b"A=" + b"x" * 10000)
    with pytest.raises(SystemExit):
        lp.load_key(None, str(huge))


def test_the_environment_variable_wins_over_files(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"OPENPAGERANK_API_KEY={ENV_KEY}\n")
    monkeypatch.setenv("OPENPAGERANK_API_KEY", "FROMENV-1234567890")
    assert lp.load_key(None, str(env)) == "FROMENV-1234567890"


def test_the_env_file_works_end_to_end_and_the_key_stays_out_of_output(tmp_path, api, capsys, monkeypatch):
    monkeypatch.delenv("OPENPAGERANK_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"# comment\nOPENPAGERANK_API_KEY={SECRET}\n")
    out = str(tmp_path / "out.csv")
    rc = lp.main([clean_file(tmp_path, ["example.com"]), "--out", out, "--env-file", str(env),
                  "--base-url", api.url, "--pause", "0"])
    assert rc == 0
    shown = capsys.readouterr()
    assert SECRET not in shown.out and SECRET not in shown.err and SECRET not in open(out).read()


def test_free_hosting_flags_are_carried_for_the_analysis(tmp_path, api):
    path = clean_file(tmp_path, ["example.com"])
    recs = [json.loads(l) for l in open(path)]
    recs[0]["free_hosting_subdomain"] = 1
    with open(path, "w") as fh:
        fh.write(json.dumps(recs[0]) + "\n")
    key_file = tmp_path / "k.txt"
    key_file.write_text(SECRET)
    out = str(tmp_path / "o.csv")
    lp.main([path, "--out", out, "--key-file", str(key_file), "--base-url", api.url, "--pause", "0"])
    assert list(csv.DictReader(open(out)))[0]["free_hosting_subdomain"] == "1"
