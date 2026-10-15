"""scripts/refresh_page_rank.py: page rank only for pages that loaded, never the key in the output. No network."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import refresh_page_rank as rpr                              # noqa: E402
from lejet import schema, urlnorm                            # noqa: E402


def rec(raw, fetched, ts="2026-10-15T01:00:00Z", domain="example.com"):
    row = schema.new_row(urlnorm.compute_url_id(raw), raw, "b")
    row.update(page_fetched=int(fetched), html_available=int(fetched), fetch_ok=int(fetched), fetch_ts=ts,
               fail_reason="none" if fetched else "http_4xx", final_reg_domain=domain)
    return schema.export_record(row)


def write(path, recs):
    with open(path, "w", encoding="ascii") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in recs))


def test_only_loaded_pages_are_looked_up_the_newest_result_wins_and_combined_files_are_skipped(tmp_path, monkeypatch):
    live_old, live_new = rec("https://a.example.com/", False, ts="2026-10-14T01:00:00Z"), rec("https://a.example.com/", True)
    write(tmp_path / "feat1.clean.jsonl", [live_old, rec("https://dead.example.org/", False), rec("https://b.example.net/", True)])
    write(tmp_path / "feat2.clean.jsonl", [live_new])
    write(tmp_path / "feat_all.clean.jsonl", [rec("https://zzz.example.io/", True)])         # combined file: ignored
    env = tmp_path / ".env"
    env.write_text("OPENPAGERANK_API_KEY=not-a-real-key\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(rpr.lookup_page_rank, "main", lambda argv: calls.append(argv) or 0)
    assert rpr.main(["--clean-dir", str(tmp_path), "--env-file", str(env)]) == 0
    sent = [json.loads(l) for l in open(tmp_path / "lookup_input_fetched.jsonl", encoding="ascii")]
    assert sorted(r["url"] for r in sent) == sorted(["hxxps://a[.]example[.]com/", "hxxps://b[.]example[.]net/"])
    assert all(r["page_fetched"] == 1 for r in sent)                                       # the dead link is not sent
    assert calls and calls[0][calls[0].index("--env-file") + 1] == str(env)
    assert not (tmp_path / "lookup_input_fetched.jsonl").name.startswith("feat")           # it can never count as a lab result


def test_without_the_key_file_nothing_is_looked_up_and_the_run_is_not_stopped(tmp_path, monkeypatch, capsys):
    write(tmp_path / "feat1.clean.jsonl", [rec("https://a.example.com/", True)])
    monkeypatch.setattr(rpr.lookup_page_rank, "main", lambda argv: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert rpr.main(["--clean-dir", str(tmp_path), "--env-file", str(tmp_path / "missing.env")]) == 0
    assert "stays -1" in capsys.readouterr().out


def test_failed_lookups_do_not_stop_the_run_and_the_key_is_never_printed(tmp_path, monkeypatch, capsys):
    write(tmp_path / "feat1.clean.jsonl", [rec("https://a.example.com/", True)])
    env = tmp_path / ".env"
    env.write_text("OPENPAGERANK_API_KEY=SECRET-VALUE-123\n", encoding="utf-8")
    monkeypatch.setattr(rpr.lookup_page_rank, "main", lambda argv: 3)
    assert rpr.main(["--clean-dir", str(tmp_path), "--env-file", str(env)]) == 0
    out = capsys.readouterr().out
    assert "SECRET-VALUE-123" not in out and "keep -1" in out
