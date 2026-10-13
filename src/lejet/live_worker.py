"""live_worker.py -- the RESIDENT live worker (analysis VM only).

The batch tools start a new Python process, a new browser and a new database for every run; for one demo link that start-up
costs far more than the inspection itself. This worker stays running as the unprivileged user `fetcher` and keeps the
pipeline (parse workers, DNS/RDAP/CT clients and caches) and ONE shared Chromium (a fresh context per page, as in the batch
screenshots) loaded between links. It does the same work as `run` + `screenshots` + `export`, with the same code:

    1. the root-owned helper `lejet-sudo-live` validates and installs the link (one line, url_id<TAB>url), checks the isolation
       gate, and drops a request file  <spool>/req/<batch>.json ;
    2. this worker picks it up: fetch (Pipeline.inspect) -> database -> screenshot (ShotRunner.run) -> export (export_jsonl);
    3. it writes <spool>/done/<batch>.json ({"ok": true|false, "error": a short code}); the helper then copies the result
       files to `ops`, and the host pulls and validates them exactly as for a batch.

What does NOT change: the unit's limits (see lejet-live.service), the safety gate for every URL, the guard proxy for every
browser page, the output format and the host-side strict validation. What the worker adds to protect that:
    * it checks the isolation token itself (at most 10 minutes old) before every link, like `run` does;
    * it takes exactly ONE link per request and never overwrites an earlier batch;
    * request files are tiny, id-checked JSON; nothing in them chooses a path or an option;
    * a failed request writes only a short error code (never page-influenced text);
    * the service is recycled every hour (RuntimeMaxSec) and the browser restarts itself if it dies.

Nothing here is used by the host; it only runs inside the VM under systemd (infra/analysis/systemd/lejet-live.service).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from lejet.cli import DEFAULT_TOKEN, preflight_ok, read_input, tool_versions
from lejet.settings import Settings

SPOOL = "/var/lib/lejet/live"
INPUT = "/var/lib/lejet/in/urls.txt"
OUT = "/var/lib/lejet/out"
TOKEN_MAX_AGE = 600                    # 10 minutes: the same limit as lejet-sudo-live
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_REQUEST_BYTES = 2048
HEARTBEAT_S = 5.0
POLL_S = 0.2
FETCH_DEADLINE_S = 90.0
SHOT_DEADLINE_S = 150.0
STAGES = ("fetch", "shot", "export")   # written to <spool>/state/<batch>; the helper maps them to progress lines
ERRORS = ("no_fresh_self_test", "bad_input", "batch_exists", "fetch_failed", "internal_error")


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="ascii") as fh:
        fh.write(text)
    os.replace(tmp, path)


def settings_for_vm(out_dir: str = OUT) -> Settings:
    """The settings `lejet.cli run` gets from lejet-sudo-batch (same defaults), for a single link."""
    s = Settings()
    s.allow_loopback = False
    s.dns_server = "10.77.0.1"
    bootstrap = "/opt/lejet/rdap-dns.json"
    s.rdap_bootstrap_path = bootstrap if os.path.exists(bootstrap) else None
    s.rdap_cache_path = os.path.join(out_dir, "rdap_cache.json")
    s.ct_enabled = True
    s.ct_probe = True
    s.ct_cache_path = os.path.join(out_dir, "ct_cache.json")
    s.popular_domains_path = None
    s.parse_workers = 3
    s.concurrency = 1
    s.rate = 2.0
    s.jitter = 0.0                     # one link: nothing to spread out
    s.seed = 1
    return s


class LiveWorker:
    """One request at a time. `pipeline` and `shots` are injected (the real ones on the VM, fakes in the tests)."""

    def __init__(self, pipeline: Any, shots: Any, *, settings: Optional[Settings] = None, spool: str = SPOOL,
                 input_path: str = INPUT, out_dir: str = OUT, token: str = DEFAULT_TOKEN,
                 token_max_age: int = TOKEN_MAX_AGE, extras: Optional[Dict[str, Any]] = None,
                 allow_loopback: bool = False, snapshot_name: str = ""):
        self.pipeline, self.shots = pipeline, shots
        self.settings = settings or Settings()
        self.spool, self.input_path, self.out_dir = spool, input_path, out_dir
        self.token, self.token_max_age = token, token_max_age
        self.extras = dict(extras or {})
        self.allow_loopback = allow_loopback
        self.snapshot_name = snapshot_name
        self.export_dir = os.path.join(out_dir, "export")
        self.stop = asyncio.Event()
        self.handled = 0
        for sub in ("req", "state", "done"):
            os.makedirs(os.path.join(spool, sub), exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

    # ------------------------------------------------------------------ small helpers
    def _stage(self, batch: str, stage: str) -> None:
        assert stage in STAGES
        _atomic_write(os.path.join(self.spool, "state", batch), stage + "\n")

    def _finish(self, batch: str, payload: Dict[str, Any]) -> None:
        _atomic_write(os.path.join(self.spool, "done", f"{batch}.json"), json.dumps(payload, sort_keys=True) + "\n")

    def _heartbeat(self) -> None:
        _atomic_write(os.path.join(self.spool, "ready"), f"{int(time.time())}\n")

    def _save_caches(self) -> None:
        for client, path in ((getattr(self.pipeline, "rdap", None), self.settings.rdap_cache_path),
                             (getattr(self.pipeline, "ct", None), self.settings.ct_cache_path)):
            if client is not None and path:
                try:
                    client.save_cache(path)
                except OSError:
                    pass

    @staticmethod
    def _fail(code: str) -> Dict[str, Any]:
        assert code in ERRORS
        return {"ok": False, "error": code}

    # ------------------------------------------------------------------ one request
    async def handle(self, batch: str) -> Dict[str, Any]:
        """fetch -> screenshot -> export for the one link in the input file. Never raises: returns the done payload."""
        from lejet.stages import shot_stage
        from lejet.storage import Storage, export_jsonl, sha256_file

        ok, _why = preflight_ok(self.token, self.token_max_age)
        if not ok:
            return self._fail("no_fresh_self_test")
        try:
            items, sha, problems = read_input(self.input_path, verify_ids=True)
        except OSError:
            return self._fail("bad_input")
        if problems or len(items) != 1:
            return self._fail("bad_input")
        url_id, raw = items[0]
        db = os.path.join(self.out_dir, f"{batch}.sqlite")
        if os.path.exists(db):
            return self._fail("batch_exists")                  # never mix into or overwrite an earlier batch

        self._stage(batch, "fetch")
        store = Storage(db)
        try:
            store.start_run(batch, sha, 1, tool_versions(self.settings), self.snapshot_name)
            insp = await asyncio.wait_for(self.pipeline.inspect(url_id, raw, batch), FETCH_DEADLINE_S)
            store.save(insp.row, insp.hops, insp.errors, insp.normalized_url)
            store.finish_run(batch)
            rows: List[Dict[str, Any]] = list(store.rows(batch))
        except asyncio.TimeoutError:
            return self._fail("fetch_failed")
        finally:
            store.close()
        self._save_caches()

        self._stage(batch, "shot")
        shot_ok = 0
        target = shot_stage.capture_target(rows[0], allow_loopback=self.allow_loopback) if rows else None
        if target is not None:
            png_dir = os.path.join(self.export_dir, f"shots-{batch}")
            os.makedirs(png_dir, exist_ok=True)
            self.shots.out_dir = png_dir                        # the shared browser stays; only the folder changes
            try:
                results = await asyncio.wait_for(self.shots.run([(url_id, target[0], target[1])], None), SHOT_DEADLINE_S)
            except asyncio.TimeoutError:
                results = None                                  # no picture; the fetched data is still worth exporting
            if results is not None:
                tar_path = os.path.join(self.export_dir, f"{batch}.shots.tar")
                shot_ok = shot_stage.pack_tar(png_dir, results, tar_path)
                digest = sha256_file(tar_path)
                with open(tar_path + ".sha256", "w", encoding="ascii") as fh:
                    fh.write(f"{digest}  {os.path.basename(tar_path)}\n")

        self._stage(batch, "export")
        extras = dict(self.extras, kernel=platform.release(), preflight_pass=True)
        info = export_jsonl(db, self.export_dir, batch, extras)
        return {"ok": True, "n_rows": int(info.get("n_rows", 0)), "shot_ok": int(shot_ok)}

    async def _handle_safely(self, batch: str) -> Dict[str, Any]:
        try:
            return await self.handle(batch)
        except Exception as exc:                                # noqa: BLE001 -- one bad request must not stop the worker
            print(f"live worker: {type(exc).__name__} on a request", file=sys.stderr, flush=True)
            return self._fail("internal_error")

    # ------------------------------------------------------------------ the loop
    def _pending(self) -> List[str]:
        req_dir = os.path.join(self.spool, "req")
        found = []
        for entry in os.scandir(req_dir):
            m = re.fullmatch(r"([A-Za-z0-9_-]{1,40})\.json", entry.name)
            if m and entry.is_file(follow_symlinks=False):
                found.append((entry.stat(follow_symlinks=False).st_mtime, m.group(1)))
        return [batch for _, batch in sorted(found)]

    def _take(self, batch: str) -> bool:
        """Read and remove one request file. True if it is a valid request for exactly this batch id."""
        path = os.path.join(self.spool, "req", f"{batch}.json")
        try:
            with open(path, "rb") as fh:
                blob = fh.read(MAX_REQUEST_BYTES + 1)
            os.unlink(path)
            data = json.loads(blob.decode("ascii"))
        except (OSError, ValueError, UnicodeDecodeError):
            return False
        return len(blob) <= MAX_REQUEST_BYTES and isinstance(data, dict) and data.get("batch_id") == batch \
            and bool(ID_RE.match(batch))

    async def _beat(self) -> None:
        while not self.stop.is_set():
            self._heartbeat()
            try:
                await asyncio.wait_for(self.stop.wait(), HEARTBEAT_S)
            except asyncio.TimeoutError:
                pass

    async def serve(self) -> None:
        for stale in self._pending():                           # requests left over from before a restart are dropped
            self._take(stale)
        beat = asyncio.create_task(self._beat())
        try:
            while not self.stop.is_set():
                for batch in self._pending():
                    if not self._take(batch):
                        continue
                    payload = await self._handle_safely(batch)
                    self.handled += 1
                    self._finish(batch, payload)
                await asyncio.sleep(POLL_S)
        finally:
            self.stop.set()
            await beat
            try:
                os.unlink(os.path.join(self.spool, "ready"))     # tells the helper to use the per-process path
            except OSError:
                pass


async def amain() -> int:
    from lejet.pipeline import Pipeline
    from lejet.stages.shot_stage import ShotRunner

    settings = settings_for_vm()
    extra_wait = int(os.environ.get("LEJET_SHOT_EXTRA_WAIT_MS", "8000"))      # same default as lejet-sudo-shots
    extras = {"git_commit": os.environ.get("LEJET_GIT_COMMIT", ""), "vbox_version": os.environ.get("LEJET_VBOX_VERSION", ""),
              "vpn_provider": os.environ.get("LEJET_VPN_PROVIDER", ""),
              "vpn_exit_country": os.environ.get("LEJET_VPN_EXIT_COUNTRY", "")}
    async with Pipeline(settings) as pipeline:
        async with ShotRunner(os.path.join(OUT, "export", "shots-live"), mobile=True, realistic=True,
                              extra_wait_ms=extra_wait) as shots:
            worker = LiveWorker(pipeline, shots, settings=settings, extras=extras,
                                snapshot_name=os.environ.get("LEJET_SNAPSHOT_NAME", "an-configured-clean"))
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, worker.stop.set)
            print("live worker ready", file=sys.stderr, flush=True)
            await worker.serve()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args(argv)
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
