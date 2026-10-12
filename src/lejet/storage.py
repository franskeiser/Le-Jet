"""
storage.py -- Stage 6: persist (plan sections 13.8, 15.1, 15.2, 15.3).

SQLite with WAL, parametrized statements ONLY, one transaction per URL, resume support.
Every URL ends in exactly one terminal state: 'done' or 'failed:<reason>'.

Tables: runs, urls (one row per URL, the full cleaned row as JSON plus key columns),
hops, errors. Everything stored has gone through schema.clean_row(), so attacker-influenced
strings are already inert when they are written.

Export: JSONL (gzip, deterministic: mtime 0) + manifest + sha256 sidecar.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from lejet import safety, schema

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  batch_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT,
  tool_versions_json TEXT, input_sha256 TEXT, snapshot_name TEXT, n_input INTEGER);
CREATE TABLE IF NOT EXISTS urls(
  url_id TEXT PRIMARY KEY, batch_id TEXT, normalized_url TEXT, reg_domain TEXT,
  status TEXT NOT NULL, fail_reason TEXT NOT NULL, stage_reached INTEGER,
  duration_ms INTEGER, row_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hops(
  url_id TEXT NOT NULL, idx INTEGER NOT NULL, status INTEGER, host TEXT,
  location_host TEXT, ms INTEGER, PRIMARY KEY(url_id, idx));
CREATE TABLE IF NOT EXISTS errors(
  url_id TEXT NOT NULL, stage TEXT, exc_type TEXT, msg_trunc TEXT);
CREATE INDEX IF NOT EXISTS urls_status ON urls(status);
"""


class Storage:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- runs ---------------------------------------------------------------
    def start_run(self, batch_id: str, input_sha256: str, n_input: int,
                  tool_versions: Dict[str, str], snapshot_name: str = "") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs(batch_id, started_at, tool_versions_json, input_sha256, "
                "snapshot_name, n_input) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(batch_id) DO UPDATE SET tool_versions_json=excluded.tool_versions_json, "
                "input_sha256=excluded.input_sha256, snapshot_name=excluded.snapshot_name, "
                "n_input=excluded.n_input, finished_at=NULL",
                (batch_id, schema.utc_now_iso(), json.dumps(tool_versions, sort_keys=True),
                 input_sha256, snapshot_name, n_input),
            )

    def finish_run(self, batch_id: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE runs SET finished_at=? WHERE batch_id=?",
                              (schema.utc_now_iso(), batch_id))

    def get_run(self, batch_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT batch_id, started_at, finished_at, tool_versions_json, input_sha256, "
            "snapshot_name, n_input FROM runs WHERE batch_id=?", (batch_id,))
        r = cur.fetchone()
        if not r:
            return None
        return {"batch_id": r[0], "started_at": r[1], "finished_at": r[2],
                "tool_versions": json.loads(r[3] or "{}"), "input_sha256": r[4],
                "snapshot_name": r[5], "n_input": r[6]}

    # ---- urls ---------------------------------------------------------------
    def done_ids(self) -> Set[str]:
        """Ids that already have a terminal state. 'failed:internal_error' rows are retried."""
        cur = self.conn.execute("SELECT url_id FROM urls WHERE status != 'failed:internal_error'")
        return {r[0] for r in cur.fetchall()}

    def save(self, row: Dict[str, Any], hops: Sequence[Any] = (),
             errors: Sequence[Tuple[str, str, str]] = (), normalized_url: str = "") -> str:
        """Persist one URL in ONE transaction. Returns its terminal status."""
        clean = schema.clean_row(row)
        status = schema.status_for(clean["fail_reason"])
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO urls(url_id, batch_id, normalized_url, reg_domain, status, "
                "fail_reason, stage_reached, duration_ms, row_json) VALUES(?,?,?,?,?,?,?,?,?)",
                (clean["url_id"], clean["batch_id"], safety.clean_field(normalized_url, 1024),
                 clean["reg_domain"], status, clean["fail_reason"], int(clean["stage_reached"]),
                 int(clean["duration_ms"]), json.dumps(clean, sort_keys=True, separators=(",", ":"))),
            )
            self.conn.execute("DELETE FROM hops WHERE url_id=?", (clean["url_id"],))
            self.conn.execute("DELETE FROM errors WHERE url_id=?", (clean["url_id"],))
            for h in hops:
                self.conn.execute(
                    "INSERT OR REPLACE INTO hops(url_id, idx, status, host, location_host, ms) "
                    "VALUES(?,?,?,?,?,?)",
                    (clean["url_id"], int(h.idx), int(h.status), safety.clean_field(h.host, 253),
                     safety.clean_field(h.location_host, 253), int(h.ms)),
                )
            for stage, exc_type, msg in errors:
                self.conn.execute(
                    "INSERT INTO errors(url_id, stage, exc_type, msg_trunc) VALUES(?,?,?,?)",
                    (clean["url_id"], safety.clean_field(stage, 32),
                     safety.clean_field(exc_type, 64), safety.clean_field(msg, 200)),
                )
        return status

    def rows(self, batch_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        sql = "SELECT row_json FROM urls"
        params: Tuple[Any, ...] = ()
        if batch_id is not None:
            sql += " WHERE batch_id=?"
            params = (batch_id,)
        sql += " ORDER BY url_id"
        for (blob,) in self.conn.execute(sql, params):
            yield json.loads(blob)

    def status_counts(self, batch_id: Optional[str] = None) -> Dict[str, int]:
        sql = "SELECT status, COUNT(*) FROM urls"
        params: Tuple[Any, ...] = ()
        if batch_id is not None:
            sql += " WHERE batch_id=?"
            params = (batch_id,)
        sql += " GROUP BY status"
        return {s: n for s, n in self.conn.execute(sql, params)}

    def hops_of(self, url_id: str) -> List[Tuple[int, int, str, str, int]]:
        cur = self.conn.execute(
            "SELECT idx, status, host, location_host, ms FROM hops WHERE url_id=? ORDER BY idx",
            (url_id,))
        return list(cur.fetchall())

    def errors_of(self, url_id: str) -> List[Tuple[str, str, str]]:
        cur = self.conn.execute(
            "SELECT stage, exc_type, msg_trunc FROM errors WHERE url_id=?", (url_id,))
        return list(cur.fetchall())


# ---------------------------------------------------------------------------
# Export (plan 15.2, 15.3)
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_jsonl(db_path: str, out_dir: str, batch_id: str,
                 manifest_extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write <batch>.jsonl.gz, <batch>.manifest.json and <batch>.jsonl.gz.sha256.
    The gzip is deterministic (mtime 0), so the same data always has the same hash."""
    os.makedirs(out_dir, exist_ok=True)
    store = Storage(db_path)
    try:
        jsonl_path = os.path.join(out_dir, f"{batch_id}.jsonl.gz")
        n_rows = 0
        with open(jsonl_path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for row in store.rows(batch_id):
                line = json.dumps(schema.export_record(row), separators=(",", ":"),
                                  ensure_ascii=True, allow_nan=False)
                gz.write(line.encode("ascii") + b"\n")
                n_rows += 1
        digest = sha256_file(jsonl_path)
        with open(jsonl_path + ".sha256", "w", encoding="ascii") as fh:
            fh.write(f"{digest}  {os.path.basename(jsonl_path)}\n")

        counts = store.status_counts(batch_id)
        failed: Dict[str, int] = {}
        for status, n in counts.items():
            if status.startswith("failed:"):
                failed[status.split(":", 1)[1]] = n
        run = store.get_run(batch_id) or {}
        manifest: Dict[str, Any] = {
            "batch_id": batch_id,
            "started": run.get("started_at"),
            "finished": run.get("finished_at"),
            "input_sha256": run.get("input_sha256"),
            "n_input": run.get("n_input"),
            "n_rows": n_rows,
            "n_done": counts.get("done", 0),
            "n_failed_by_reason": dict(sorted(failed.items())),
            "snapshot_name": run.get("snapshot_name", ""),
            "tool_versions": run.get("tool_versions", {}),
            "python": sys.version.split()[0],
            "jsonl_gz_sha256": digest,
        }
        manifest.update(manifest_extras or {})
        manifest_path = os.path.join(out_dir, f"{batch_id}.manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        return {"jsonl": jsonl_path, "manifest": manifest_path, "sha256": digest, "n_rows": n_rows}
    finally:
        store.close()
