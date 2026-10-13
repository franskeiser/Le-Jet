"""On-demand inspection: the demo opens a link INSIDE the isolated lab VMs, never on this computer.

The API process itself never opens, fetches or resolves a link. A request only hands the link to the same
unattended pipeline the batches use (scripts/run_batch.ps1: push file -> isolation self-test -> fetch in the
analysis VM -> screenshot -> pull -> strict validator). What comes back is treated as hostile data, exactly as
in a batch: it is accepted only after scripts/validate_output.py passed. The new row goes to a SEPARATE store
(data/live/on_demand_links.csv), which the training/evaluation code never reads.

One job at a time (there is one analysis VM). Jobs report real steps and real timings for the frontend.
"""
from __future__ import annotations

import queue
import re
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.config import ARTIFACTS_ROOT
from src.data.build_live_set import LABELS, load_batch
from src.features.link_features import normalize_link

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
LAB = Path(r"D:\LeJetData")
INPUT_DIR, CLEAN_DIR, QUARANTINE, REPORTS = LAB / "input", LAB / "clean", LAB / "quarantine", LAB / "reports"
SECRETS_ENV = LAB / "secrets" / ".env"
ON_DEMAND_CSV = ARTIFACTS_ROOT / "data" / "live" / "on_demand_links.csv"
VBOXMANAGE = Path(r"C:\Program Files\Oracle\VirtualBox\VBoxManage.exe")
POWERSHELL = "powershell.exe"
JOB_TIMEOUT_S = 15 * 60
MAX_QUEUED = 3
HEADER = re.compile(r"^== (\S+)\s+(.*)$")

# step id -> (what a person reads, run_batch.ps1 headers that belong to it)
STEPS = [
    ("check", "Checking the link", ("0", "1")),
    ("send", "Sending it into the isolated lab", ("2",)),
    ("isolation", "Isolation gate", ("3",)),
    ("open", "Opening the page inside the lab VM", ("4",)),
    ("shot", "Taking a screenshot of the page", ("4b",)),
    ("bring", "Bringing the evidence back and validating it", ("5", "5b", "6")),
    ("rank", "Looking up the domain's page rank", ()),
]
STEP_OF_HEADER = {h: sid for sid, _, headers in STEPS for h in headers}


def _plain(text: str, limit: int = 160) -> str:
    """Anything that came out of the lab is data, not text to trust: keep printable ASCII only."""
    return re.sub(r"[^\x20-\x7e]", "", text)[:limit]


class Job:
    def __init__(self, job_id: str, link: str, batch: str, key: str):
        self.id, self.batch, self.key = job_id, batch, key
        self.link = link                                     # normalised; only the defanged form is ever shown
        self.state = "queued"                                # queued | running | done | failed
        self.cached = False
        self.error = ""
        self.started = time.time()
        self.finished: float | None = None
        self.steps = [{"id": sid, "label": label, "state": "pending", "seconds": None} for sid, label, _ in STEPS]
        self._t0: dict[str, float] = {}

    def begin(self, sid: str) -> None:
        for s in self.steps:
            if s["id"] == sid and s["state"] == "pending":
                s["state"] = "running"
                self._t0[sid] = time.time()

    def end(self, sid: str, ok: bool = True) -> None:
        for s in self.steps:
            if s["id"] == sid and s["state"] in ("running", "pending"):
                s["state"] = "done" if ok else "failed"
                s["seconds"] = round(time.time() - self._t0.get(sid, time.time()), 1)

    def advance_to(self, sid: str) -> None:
        """Everything before `sid` is finished, `sid` has started."""
        for s in self.steps:
            if s["id"] == sid:
                break
            if s["state"] == "running":
                self.end(s["id"])
            elif s["state"] == "pending":
                s["state"] = "done"
                s["seconds"] = 0.0
        self.begin(sid)

    def view(self) -> dict:
        from src.lejet.safety import defang_url
        end = self.finished or time.time()
        return {"job_id": self.id, "state": self.state, "cached": self.cached, "error": self.error,
                "link": defang_url(self.link), "elapsed_seconds": round(end - self.started, 1),
                "steps": [dict(s) for s in self.steps]}


class LiveInspector:
    def __init__(self, store, enabled: bool = True, reuse_stored: bool = False):
        self.store = store
        self.reuse_stored = reuse_stored                       # False: EVERY link is opened in the lab, even a stored one
        self.enabled = enabled and (SCRIPTS / "run_batch.ps1").exists()
        self.jobs: dict[str, Job] = {}
        self.queue: "queue.Queue[Job]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ready_cache = (0.0, False, "")
        self._counter = 0

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        ready, why = self._lab_ready()
        return {"enabled": self.enabled, "ready": ready, "note": why}

    def _lab_ready(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "Live inspection is switched off."
        at, ok, why = self._ready_cache
        if time.time() - at < 15:
            return ok, why
        try:
            vms = subprocess.run([str(VBOXMANAGE), "list", "runningvms"], capture_output=True, text=True,
                                 timeout=10).stdout
            agent = subprocess.run(["ssh-add", "-l"], capture_output=True, text=True, timeout=10)
            if '"lejet-gw"' not in vms or '"lejet-an"' not in vms:
                ok, why = False, "The lab virtual machines are not running."
            elif agent.returncode != 0:
                ok, why = False, "The lab's SSH key is not loaded."
            else:
                ok, why = True, "The isolated lab is up."
        except (OSError, subprocess.SubprocessError):
            ok, why = False, "The lab could not be checked."
        self._ready_cache = (time.time(), ok, why)
        return ok, why

    # ------------------------------------------------------------------ requests
    def submit(self, link: str, refresh: bool = False) -> Job:
        from src.data.build_live_set import link_key
        normalised = normalize_link(link)
        key = link_key(normalised)
        with self._lock:
            self._counter += 1
            batch = f"live{time.strftime('%m%d%H%M%S')}x{self._counter}"
            job = Job(secrets.token_hex(6), normalised, batch, key)
            self.jobs[job.id] = job
            for old in sorted(self.jobs.values(), key=lambda j: j.started)[:-50]:
                self.jobs.pop(old.id, None)
            if key in self.store.rows and self.reuse_stored and not refresh:
                job.cached, job.state = True, "done"
                for s in job.steps:
                    s["state"], s["seconds"] = "done", 0.0
                job.finished = time.time()
                return job
            if self.queue.qsize() >= MAX_QUEUED:
                raise RuntimeError("busy")
            ready, why = self._lab_ready()
            if not ready:
                raise RuntimeError(why)
            self.queue.put(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, daemon=True)
                self._worker.start()
        return job

    def job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    # ------------------------------------------------------------------ worker
    def _work(self) -> None:
        while True:
            try:
                job = self.queue.get(timeout=60)
            except queue.Empty:
                return
            job.state = "running"
            try:
                self._run(job)
                job.state = "done"
            except Exception as err:                         # a failed job must never take the server down
                job.state = "failed"
                job.error = _plain(str(err)) or "The inspection failed."
                for s in job.steps:
                    if s["state"] == "running":
                        job.end(s["id"], ok=False)
            finally:
                job.finished = time.time()

    def _run(self, job: Job) -> None:
        from src.lejet.safety import defang_url
        for d in (INPUT_DIR, REPORTS):
            d.mkdir(parents=True, exist_ok=True)
        url_file = INPUT_DIR / f"{job.batch}.txt"
        url_file.write_text(defang_url(job.link) + "\n", encoding="ascii")
        push = INPUT_DIR / f"{job.batch}_push.txt"
        log = REPORTS / f"{job.batch}.log"
        job.advance_to("check")

        # the fast runner (one ssh call, background isolation gate) when the VM has the live helper; otherwise the batch runner
        code, lines = (self._stream("run_live.ps1", url_file, job, log, extra=[])
                       if (SCRIPTS / "run_live.ps1").exists() else (4, []))
        if code == 4:
            code, lines = self._stream("run_batch.ps1", url_file, job, log, extra=["-Screenshots"])
        if code != 0:
            blob = "\n".join(lines)
            if "VALIDATION FAILED" in blob:
                self._recover_validation(job, push)
            elif "No usable" in blob or "rejected" in blob and '"kept": 0' in blob:
                raise RuntimeError("The lab refused this link (not a plain web address).")
            else:
                raise RuntimeError("The lab run did not finish. Nothing was added.")
        job.advance_to("rank")
        self._rank_and_store(job)
        job.end("rank")

    def _stream(self, script: str, url_file: Path, job: Job, log: Path, extra: list) -> tuple[int, list]:
        """Run one lab runner script, feed its "== n ..." progress lines into the job, return (exit code, output lines)."""
        cmd = [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPTS / script),
               "-UrlFile", str(url_file), "-Batch", job.batch, *extra]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", creationflags=0x08000000)   # no console window
        timer = threading.Timer(JOB_TIMEOUT_S, proc.kill)
        timer.start()
        lines: list[str] = []
        try:
            with open(log, "a", encoding="utf-8") as fh:
                for line in proc.stdout:
                    fh.write(line)
                    lines.append(line.rstrip("\n"))
                    m = HEADER.match(line.strip())
                    if m and m.group(1) in STEP_OF_HEADER:
                        job.advance_to(STEP_OF_HEADER[m.group(1)])
            return proc.wait(), lines
        finally:
            timer.cancel()

    def _py(self, *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", *args], capture_output=True, text=True, timeout=timeout,
                              creationflags=0x08000000)

    def _recover_validation(self, job: Job, push: Path) -> None:
        """The deployed VM build can leave a clickable link inside a field; the host-side repair defangs it and the
        SAME strict validator then decides. Anything else that fails validation stays failed."""
        job.advance_to("bring")
        folders = sorted(QUARANTINE.glob(f"{job.batch}-2*"))
        if not folders:
            raise RuntimeError("The lab's result did not pass validation. Nothing was added.")
        src = folders[-1] / f"{job.batch}.jsonl.gz"
        fix = self._py(str(SCRIPTS / "repair_embedded_links.py"), str(src))
        if fix.returncode != 0:
            raise RuntimeError("The lab's result did not pass validation. Nothing was added.")
        repaired = folders[-1] / f"{job.batch}.repaired.jsonl.gz"
        ok = self._py(str(SCRIPTS / "validate_output.py"), str(repaired), "--sha256-file", str(repaired) + ".sha256",
                      "--input", str(push), "--clean-dir", str(CLEAN_DIR), "--csv", str(CLEAN_DIR / f"{job.batch}.csv"))
        if ok.returncode != 0:
            raise RuntimeError("The lab's result did not pass validation. Nothing was added.")
        pulled = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                 str(SCRIPTS / "pull_shots.ps1"), "-Batch", job.batch],
                                capture_output=True, text=True, timeout=300, creationflags=0x08000000)
        if pulled.returncode != 0:
            # the screenshot is optional evidence: the verdict does not need it
            pass

    def _rank_and_store(self, job: Job) -> None:
        clean = CLEAN_DIR / f"{job.batch}.clean.jsonl"
        if not clean.exists():
            raise RuntimeError("No validated result came back. Nothing was added.")
        if SECRETS_ENV.exists():
            self._py(str(SCRIPTS / "lookup_page_rank.py"), str(clean), "--out",
                     str(CLEAN_DIR / f"{job.batch}.model_input.csv"), "--env-file", str(SECRETS_ENV), timeout=120)
        records = load_batch(CLEAN_DIR, job.batch, LABELS[0], "on_demand")
        if len(records) != 1:
            raise RuntimeError("The lab returned an unexpected number of rows. Nothing was added.")
        rec = records[0]
        rec.update({"label": "unlabelled", "split": "on_demand", "key": job.key})
        self.store.add(rec, ON_DEMAND_CSV, CLEAN_DIR / "shots" / job.batch)
