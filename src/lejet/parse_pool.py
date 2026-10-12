"""
parse_pool.py -- HTML parsing outside the event loop, with a hard time limit per task
(plan section 13.5: "run HTML parsing and regex work in a process pool with a per-task time
limit and kill-on-timeout").

Why not concurrent.futures.ProcessPoolExecutor? It cannot kill ONE hung worker. Here every
worker is a persistent `spawn`ed process talking over a Pipe; if a task runs longer than the
limit (or the caller is cancelled) that worker is killed and replaced, and the caller gets
{"error": "parse_timeout"}. Requests carry an id and answers are matched by it, so a late
answer from a killed or retired worker can never be mistaken for the next URL's result.

`workers=0` runs the analysis inline in a thread (no kill possible): tests and tiny runs only.
"""

from __future__ import annotations

import asyncio
import itertools
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional

_rid = itertools.count(1)


def _worker_main(conn) -> None:
    """Child process loop. Import problems are reported once and end the worker."""
    try:
        from lejet.features import html_features
    except Exception as exc:                                  # noqa: BLE001
        try:
            conn.send({"rid": 0, "fatal": f"import:{type(exc).__name__}"})
        except Exception:                                      # noqa: BLE001
            pass
        return
    # Tests only: LEJET_TEST_PARSE_DELAY makes every task artificially slow, so the timeout tests
    # do not depend on how fast the machine parses a page.
    try:
        _TEST_DELAY = float(os.environ.get("LEJET_TEST_PARSE_DELAY", "0") or 0)
    except ValueError:
        _TEST_DELAY = 0.0
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        rid = msg.get("rid", 0)
        if _TEST_DELAY:
            time.sleep(_TEST_DELAY)
        try:
            result = html_features.analyze_page(**msg["args"])
        except Exception as exc:                              # noqa: BLE001
            result = {"error": type(exc).__name__}
        result["rid"] = rid
        try:
            conn.send(result)
        except (OSError, ValueError):
            break


class _Worker:
    def __init__(self, ctx) -> None:
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(child,), daemon=True)
        self.proc.start()
        child.close()

    def call(self, args: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
        """Blocking. Returns the answer, or None on timeout / crash."""
        rid = next(_rid)
        end = time.monotonic() + timeout                       # writing the page counts too
        try:
            self.conn.send({"rid": rid, "args": args})
        except (OSError, ValueError):
            return None
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return None
            try:
                if not self.conn.poll(left):
                    return None
                answer = self.conn.recv()
            except (EOFError, OSError):
                return None
            if answer.get("fatal"):
                return None
            if answer.get("rid") == rid:
                return answer
            # a stale answer (from an abandoned call): drop it and keep waiting

    def kill(self) -> None:
        try:
            self.proc.kill()
        except Exception:                                      # noqa: BLE001
            pass
        try:
            self.proc.join(2)
        except Exception:                                      # noqa: BLE001
            pass
        try:
            self.conn.close()
        except Exception:                                      # noqa: BLE001
            pass

    def stop(self) -> None:
        try:
            self.conn.send(None)
            self.proc.join(2)
        except Exception:                                      # noqa: BLE001
            pass
        self.kill()


class ParsePool:
    def __init__(self, workers: int = 2, timeout: float = 5.0) -> None:
        self.workers = workers
        self.timeout = timeout
        self._ctx = mp.get_context("spawn")
        self._free: "Optional[asyncio.Queue[_Worker]]" = None
        self._all: List[_Worker] = []
        self.restarts = 0

    async def start(self) -> "ParsePool":
        if self.workers > 0:
            self._free = asyncio.Queue()
            spawned = await asyncio.gather(
                *[asyncio.to_thread(_Worker, self._ctx) for _ in range(self.workers)]
            )
            for w in spawned:
                self._all.append(w)
                self._free.put_nowait(w)
        return self

    async def analyze(self, **args: Any) -> Dict[str, Any]:
        """Analyse one page (see html_features.analyze_page). Never raises (except on cancel)."""
        if self.workers == 0:
            from lejet.features import html_features
            try:
                return await asyncio.to_thread(html_features.analyze_page, **args)
            except Exception as exc:                          # noqa: BLE001
                return {"error": type(exc).__name__}
        assert self._free is not None, "ParsePool.start() was not called"
        worker = await self._free.get()
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, worker.call, args, self.timeout
            )
        except BaseException:                                  # cancelled: the worker is busy
            self._replace_in_background(worker)
            raise
        if result is None:
            await self._replace(worker)
            return {"error": "parse_timeout"}
        self._free.put_nowait(worker)
        return result

    async def _replace(self, worker: _Worker) -> None:
        await asyncio.to_thread(worker.kill)
        self.restarts += 1
        fresh = await asyncio.to_thread(_Worker, self._ctx)
        self._swap(worker, fresh)

    def _replace_in_background(self, worker: _Worker) -> None:
        async def _go() -> None:
            try:
                await self._replace(worker)
            except Exception:                                  # noqa: BLE001
                pass
        asyncio.get_running_loop().create_task(_go())

    def _swap(self, old: _Worker, new: _Worker) -> None:
        if old in self._all:
            self._all.remove(old)
        self._all.append(new)
        assert self._free is not None
        self._free.put_nowait(new)

    async def close(self) -> None:
        workers, self._all = self._all, []
        await asyncio.gather(*[asyncio.to_thread(w.stop) for w in workers])
