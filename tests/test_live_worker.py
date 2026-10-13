"""Tests for src/lejet/live_worker.py: the resident worker that does fetch -> screenshot -> export for one link.

The real Pipeline runs against the local fixture server (loopback test mode), and the real Storage, export and tar packing
are used; only the browser is a fake that writes a valid PNG. Runs on the host and inside the VM (no internet, no browser).
"""
from __future__ import annotations

import asyncio
import gzip
import json
import os
import tarfile
import time

import pytest

from lejet import pngcheck, urlnorm
from lejet.live_worker import LiveWorker, settings_for_vm
from lejet.pipeline import Pipeline
from lejet.settings import Settings
from lejet.stages.dns_stage import RawAnswer
from lejet.stages.rdap_stage import Bootstrap
from lejet.stages.shot_stage import ShotResult
from test_pngcheck import png


class NoDns:
    async def query(self, name, rdtype):
        return RawAnswer("nxdomain")


async def no_rdap(url):
    return 404, b""


class FakeShots:
    """Stands in for the shared Chromium: writes a valid PNG into out_dir and reports success."""
    def __init__(self):
        self.out_dir, self.calls = "", []

    async def run(self, items, progress=None):
        results = []
        for item in items:
            uid, url = item[0], item[1]
            self.calls.append((uid, url, self.out_dir))
            data = png(8, 6)
            with open(os.path.join(self.out_dir, f"{uid}.png"), "wb") as fh:
                fh.write(data)
            results.append(ShotResult(url_id=uid, ok=True, error="none", bytes=len(data), width=8, height=6))
        return results


def fresh_token(path, age=0):
    path.write_text(f"{int(time.time()) - age}\n", encoding="ascii")


def push(path, raw):
    path.write_text(f"{urlnorm.compute_url_id(raw)}\t{raw}\n", encoding="ascii")


@pytest.fixture()
def env(tmp_path):
    spool, out = tmp_path / "live", tmp_path / "out"
    inp, token = tmp_path / "urls.txt", tmp_path / "token"
    fresh_token(token)
    return dict(spool=str(spool), out=str(out), inp=inp, token=token)


def run_worker(env, raw_or_none, batch="live01", shots=None, token_age=0, pipeline_settings=None):
    shots = shots or FakeShots()
    if raw_or_none is not None:
        push(env["inp"], raw_or_none)
    fresh_token(env["token"], token_age)

    async def go():
        s = pipeline_settings or Settings(allow_loopback=True, parse_workers=0, rate=1000.0, jitter=0.0,
                                          rdap_min_interval=0.0, hop_deadline=8, url_deadline=15, connect_timeout=3)
        async with Pipeline(s, resolver=NoDns(), rdap_fetch=no_rdap, bootstrap=Bootstrap([])) as pipeline:
            w = LiveWorker(pipeline, shots, settings=s, spool=env["spool"], input_path=str(env["inp"]), out_dir=env["out"],
                           token=str(env["token"]), allow_loopback=True, extras={"vpn_provider": "test"})
            return w, await w.handle(batch)
    return (*asyncio.run(go()), shots)


def test_one_link_gives_the_same_three_files_a_batch_gives_and_a_screenshot(server, env):
    raw = server.url("/page/html_all_internal")
    worker, payload, shots = run_worker(env, raw)
    assert payload == {"ok": True, "n_rows": 1, "shot_ok": 1}
    export = os.path.join(env["out"], "export")
    for ext in (".jsonl.gz", ".jsonl.gz.sha256", ".manifest.json", ".shots.tar", ".shots.tar.sha256"):
        assert os.path.isfile(os.path.join(export, f"live01{ext}")), ext
    row = json.loads(gzip.open(os.path.join(export, "live01.jsonl.gz"), "rt").readline())
    assert row["url_id"] == urlnorm.compute_url_id(raw) and row["batch_id"] == "live01"
    assert row["fail_reason"] == "none" and row["page_fetched"] == 1 and row["features"]["nb_hyperlinks"] == 10
    with tarfile.open(os.path.join(export, "live01.shots.tar")) as tar:
        assert sorted(tar.getnames()) == sorted([f"{row['url_id']}.png", "shots.jsonl"])
        assert pngcheck.check_png(tar.extractfile(f"{row['url_id']}.png").read())[0]
    assert shots.calls and shots.calls[0][0] == row["url_id"]


def test_the_manifest_records_the_provenance_and_that_the_self_test_was_fresh(server, env):
    run_worker(env, server.url("/page/plain"))
    manifest = json.load(open(os.path.join(env["out"], "export", "live01.manifest.json")))
    flat = json.dumps(manifest)
    assert '"preflight_pass": true' in flat and "test" in flat


def test_without_a_fresh_self_test_nothing_is_fetched(server, env):
    before = len(server.hits)
    _, payload, shots = run_worker(env, server.url("/page/plain"), token_age=601)
    assert payload == {"ok": False, "error": "no_fresh_self_test"}
    assert len(server.hits) == before and shots.calls == []                    # no request, no browser page
    assert not os.path.exists(os.path.join(env["out"], "live01.sqlite"))


def test_a_missing_token_also_blocks(server, env):
    env["token"].unlink(missing_ok=True)

    async def go():
        async with Pipeline(Settings(allow_loopback=True, parse_workers=0), resolver=NoDns(), rdap_fetch=no_rdap,
                            bootstrap=Bootstrap([])) as p:
            push(env["inp"], server.url("/page/plain"))
            w = LiveWorker(p, FakeShots(), spool=env["spool"], input_path=str(env["inp"]), out_dir=env["out"],
                           token=str(env["token"]), allow_loopback=True)
            return await w.handle("live02")
    assert asyncio.run(go()) == {"ok": False, "error": "no_fresh_self_test"}


def test_exactly_one_link_and_checked_ids(server, env):
    two = "\n".join(f"{urlnorm.compute_url_id(server.url(p))}\t{server.url(p)}" for p in ("/page/plain", "/404")) + "\n"
    env["inp"].write_text(two, encoding="ascii")
    _, payload, _ = run_worker(env, None)
    assert payload == {"ok": False, "error": "bad_input"}
    env["inp"].write_text("0123456789abcdef\thttp://127.0.0.1/x\n", encoding="ascii")       # id does not match the link
    _, payload, _ = run_worker(env, None, batch="live03")
    assert payload == {"ok": False, "error": "bad_input"}


def test_a_batch_id_is_never_reused(server, env):
    run_worker(env, server.url("/page/plain"), batch="same")
    _, payload, _ = run_worker(env, server.url("/page/plain"), batch="same")
    assert payload == {"ok": False, "error": "batch_exists"}


def test_a_link_with_nothing_to_photograph_still_exports_and_has_no_screenshot_archive(env):
    _, payload, shots = run_worker(env, "http://10.0.0.1/admin", batch="live04")        # refused by the safety gate
    assert payload["ok"] is True and payload["shot_ok"] == 0 and shots.calls == []
    export = os.path.join(env["out"], "export")
    assert os.path.isfile(os.path.join(export, "live04.jsonl.gz"))
    assert not os.path.exists(os.path.join(export, "live04.shots.tar"))                  # the host handles a missing archive


def test_the_serve_loop_takes_requests_writes_done_files_and_survives_a_bad_one(server, env):
    shots = FakeShots()
    push(env["inp"], server.url("/page/plain"))
    fresh_token(env["token"])

    async def go():
        s = Settings(allow_loopback=True, parse_workers=0, rate=1000.0, jitter=0.0, rdap_min_interval=0.0)
        async with Pipeline(s, resolver=NoDns(), rdap_fetch=no_rdap, bootstrap=Bootstrap([])) as pipeline:
            w = LiveWorker(pipeline, shots, settings=s, spool=env["spool"], input_path=str(env["inp"]), out_dir=env["out"],
                           token=str(env["token"]), allow_loopback=True)
            req = os.path.join(env["spool"], "req")
            with open(os.path.join(req, "stale.json"), "w") as fh:                        # left over from before a restart
                fh.write('{"batch_id": "stale"}')
            task = asyncio.create_task(w.serve())
            await asyncio.sleep(0.5)
            assert not os.path.exists(os.path.join(env["spool"], "done", "stale.json"))   # dropped, not run
            with open(os.path.join(req, "evil.json"), "w") as fh:
                fh.write('{"batch_id": "other-id"}')                                      # name and content disagree
            with open(os.path.join(req, "live05.json"), "w") as fh:
                fh.write('{"batch_id": "live05"}')
            for _ in range(100):
                if os.path.exists(os.path.join(env["spool"], "done", "live05.json")):
                    break
                await asyncio.sleep(0.1)
            assert os.path.exists(os.path.join(env["spool"], "ready"))                    # heartbeat
            w.stop.set()
            await task
            assert not os.path.exists(os.path.join(env["spool"], "ready"))                # gone on shutdown
            return w
    w = asyncio.run(go())
    done = json.load(open(os.path.join(env["spool"], "done", "live05.json")))
    assert done["ok"] is True
    assert not os.path.exists(os.path.join(env["spool"], "done", "evil.json"))
    assert not os.path.exists(os.path.join(env["spool"], "req", "evil.json"))             # consumed
    assert w.handled == 1


def test_the_vm_settings_match_what_the_batch_helper_uses():
    s = settings_for_vm("/x")
    assert s.dns_server == "10.77.0.1" and s.allow_loopback is False
    assert s.ct_enabled and s.ct_probe and s.parse_workers == 3
    assert s.rdap_cache_path.replace("\\", "/") == "/x/rdap_cache.json"
