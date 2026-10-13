"""Tests for src/serving/live_inspect.py: the demo hands a link to the isolated lab; nothing here opens a link.

The lab itself (VMs, ssh) is never started: the pipeline call is replaced by a fake, so these tests check the job
bookkeeping, the store hand-over, and that a failed or unavailable lab can never add a row.
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from src.serving import live_inspect as li
from src.serving.lejet_service import LinkStore


def record(key: str, link_id: str = "0123456789abcdef") -> dict:
    return {"link_id": link_id, "label": "unlabelled", "source": "on_demand", "key": key, "page_fetched": 1,
            "fetch_status": "live", "final_url": "hxxps://example[.]test/", "final_domain": "example.test",
            "page_rank": 3}


def store(tmp_path) -> LinkStore:
    return LinkStore(tmp_path / "missing.csv", shot_dirs=(tmp_path / "shots",))


def wait_for(job, states=("done", "failed"), seconds=5):
    end = time.time() + seconds
    while job.state not in states and time.time() < end:
        time.sleep(0.02)
    return job.state


def test_store_add_keeps_the_row_and_survives_a_restart(tmp_path):
    s = store(tmp_path)
    csv = tmp_path / "on_demand.csv"
    shots = tmp_path / "shots" / "live1"
    shots.mkdir(parents=True)
    (shots / "0123456789abcdef.png").write_bytes(b"png")
    s.add(record("k1"), csv, shots)
    assert s.rows["k1"]["page_fetched"] == 1 and s.rows["k1"]["page_rank"] == 3
    assert s.has_shot("0123456789abcdef")
    s.add(record("k2", "fedcba9876543210"), csv)
    assert len(pd.read_csv(csv)) == 2
    reloaded = LinkStore(tmp_path / "missing.csv", shot_dirs=(tmp_path / "shots",), on_demand_csv=csv)
    assert set(reloaded.rows) == {"k1", "k2"} and reloaded.has_shot("0123456789abcdef")


def test_a_link_already_in_the_store_is_answered_without_the_lab_only_when_asked(tmp_path, monkeypatch):
    s = store(tmp_path)
    inspector = li.LiveInspector(s, reuse_stored=True)
    from src.data.build_live_set import link_key
    from src.features.link_features import normalize_link
    key = link_key(normalize_link("https://known.example.test/a"))
    s.rows[key] = record(key)
    monkeypatch.setattr(inspector, "_lab_ready", lambda: (False, "no lab"))
    job = inspector.submit("https://known.example.test/a")
    assert job.state == "done" and job.cached and all(x["state"] == "done" for x in job.steps)
    assert job.view()["link"].startswith("hxxps://")                 # only the defanged link is ever shown


def test_by_default_every_link_goes_to_the_lab_even_a_stored_one(tmp_path, monkeypatch):
    s = store(tmp_path)
    inspector = li.LiveInspector(s)
    from src.data.build_live_set import link_key
    from src.features.link_features import normalize_link
    key = link_key(normalize_link("https://known.example.test/a"))
    s.rows[key] = record(key)
    monkeypatch.setattr(inspector, "_lab_ready", lambda: (False, "The lab virtual machines are not running."))
    with pytest.raises(RuntimeError, match="not running"):
        inspector.submit("https://known.example.test/a")                 # no shortcut through the stored row
    monkeypatch.setattr(inspector, "_lab_ready", lambda: (True, "ok"))
    monkeypatch.setattr(inspector, "_work", lambda: None)
    job = inspector.submit("https://known.example.test/a")
    assert not job.cached and job.state == "queued"


def test_a_new_link_needs_a_ready_lab(tmp_path, monkeypatch):
    inspector = li.LiveInspector(store(tmp_path))
    monkeypatch.setattr(inspector, "_lab_ready", lambda: (False, "The lab virtual machines are not running."))
    with pytest.raises(RuntimeError, match="not running"):
        inspector.submit("https://new.example.test/")
    assert inspector.queue.qsize() == 0


def test_a_finished_job_reports_every_step_with_its_time(tmp_path, monkeypatch):
    inspector = li.LiveInspector(store(tmp_path))
    monkeypatch.setattr(inspector, "_lab_ready", lambda: (True, "ok"))

    def fake_run(job):
        for sid, _, _ in li.STEPS:
            job.advance_to(sid)
        job.end("rank")
    monkeypatch.setattr(inspector, "_run", fake_run)
    job = inspector.submit("https://new.example.test/")
    assert wait_for(job) == "done"
    assert [x["state"] for x in job.steps] == ["done"] * len(li.STEPS)
    assert all(x["seconds"] is not None for x in job.steps)


def test_a_failed_job_adds_nothing_and_says_so_plainly(tmp_path, monkeypatch):
    s = store(tmp_path)
    inspector = li.LiveInspector(s)
    monkeypatch.setattr(inspector, "_lab_ready", lambda: (True, "ok"))

    def broken(job):
        job.advance_to("open")
        raise RuntimeError("The lab run did not finish.\x1b[31m <script>")   # lab text is hostile data
    monkeypatch.setattr(inspector, "_run", broken)
    job = inspector.submit("https://new.example.test/")
    assert wait_for(job) == "failed"
    assert "\x1b" not in job.error and job.error.startswith("The lab run did not finish")
    assert len(s) == 0
    assert any(x["state"] == "failed" for x in job.steps)


def test_the_queue_has_a_limit(tmp_path, monkeypatch):
    inspector = li.LiveInspector(store(tmp_path))
    monkeypatch.setattr(inspector, "_lab_ready", lambda: (True, "ok"))
    monkeypatch.setattr(inspector, "_work", lambda: time.sleep(0.01))     # nobody drains the queue
    for i in range(li.MAX_QUEUED):
        inspector.submit(f"https://n{i}.example.test/")
    with pytest.raises(RuntimeError, match="busy"):
        inspector.submit("https://one-too-many.example.test/")


def test_lab_text_is_reduced_to_printable_ascii():
    assert li._plain("ok\x1b[31m ‮ evil\n") == "ok[31m  evil"
    assert len(li._plain("x" * 1000)) == 160


def test_step_headers_of_run_batch_map_to_steps():
    for header in ("0", "1", "2", "3", "4", "4b", "5", "5b", "6"):
        assert header in li.STEP_OF_HEADER
    assert li.HEADER.match("== 4b  real screenshots of the fetched pages").group(1) == "4b"
