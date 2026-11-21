"""browser_html.py -- the HTML of a page that the plain fetch could not read, as a real browser sees it (LAB ONLY).

Why: many ordinary, legitimate sites (banks, telcos, government, shops) answer an automated HTTP client with 401 / 403 / 406 /
429 / 503 or a "checking your browser" page, although a person opens them without any trouble. The plain fetch then files them
as dead and every page feature stays -1. A person's browser is what the dataset is about, so for exactly these refusals the lab
opens the address in its own Chromium (the same sandboxed browser, guard proxy and per-page context as the screenshot tool, see
stages/shot_stage.py) and the page's HTML goes through the SAME parser and the SAME feature code as a plain fetch.

What this is not: no fingerprint spoofing, no stealth patches, no solving of challenges. If the browser is refused too (a
non-2xx answer, or a challenge page), the row stays exactly as the plain fetch left it.

Limits (a batch must stay inside the lab unit's 30 minutes): one page at a time, at most `max_pages` per run and a total time budget.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from lejet import urlnorm

# what sites answer to a client they do not want; 404/410 (really gone) and 400 are NOT in this list
BOT_BLOCK_STATUS = frozenset({401, 403, 406, 429, 503})
BOT_BLOCK_REASONS = frozenset({"challenge_page", "blocked_or_empty"})


def needs_browser(fail_reason: str, final_status: Optional[int]) -> bool:
    """Did the plain fetch fail in a way a browser may get past (the site refused the client, not 'the page is not there')?"""
    if fail_reason in BOT_BLOCK_REASONS:
        return True
    return fail_reason in ("http_4xx", "http_5xx") and final_status in BOT_BLOCK_STATUS


@dataclass
class BrowserPage:
    html: str
    final_url: str            # where the browser ended up (untrusted: the pipeline re-validates it)
    status: int               # the main page's HTTP status as the browser saw it


class BrowserHtml:
    """One shared Chromium; `fetch(url)` returns a BrowserPage or None. Plug it into Pipeline.html_fallback."""

    def __init__(self, work_dir: str, *, resolver: Any = None, max_pages: int = 150, budget_s: float = 900.0,
                 extra_wait_ms: int = 2500, runner: Any = None, clock: Callable[[], float] = time.monotonic):
        self.work_dir = work_dir
        self.resolver = resolver
        self.max_pages = max_pages
        self.budget_s = budget_s
        self.extra_wait_ms = extra_wait_ms
        self._runner = runner                      # tests inject a fake with an async shoot(url_id, url)
        self._clock = clock
        self._lock = asyncio.Lock()                 # one browser page at a time
        self._t0: Optional[float] = None
        self.used = 0
        self.got = 0

    async def start(self) -> "BrowserHtml":
        if self._runner is None:
            from lejet.stages.shot_stage import ShotRunner          # lazy: needs playwright (the VM)
            self._runner = ShotRunner(os.path.join(self.work_dir, "browser-html"), html_only=True, mobile=False,
                                      extra_wait_ms=self.extra_wait_ms, resolver=self.resolver)
            await self._runner.start()
        return self

    async def close(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None and hasattr(runner, "close"):
            await runner.close()

    async def __aenter__(self) -> "BrowserHtml":
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def fetch(self, url: str) -> Optional[BrowserPage]:
        if self._runner is None:
            return None
        async with self._lock:
            now = self._clock()
            if self._t0 is None:
                self._t0 = now
            if self.used >= self.max_pages or now - self._t0 > self.budget_s:
                return None                         # out of budget: the row keeps what the plain fetch found
            self.used += 1
            res = await self._runner.shoot(urlnorm.compute_url_id(url), url)
        if not (getattr(res, "ok", False) and str(getattr(res, "html", "")).strip()):
            return None
        status = int(getattr(res, "http_status", -1))
        if status != -1 and not 200 <= status < 400:
            return None                             # the browser was refused as well
        self.got += 1
        return BrowserPage(html=res.html, final_url=getattr(res, "page_url", "") or "", status=status if status != -1 else 200)
