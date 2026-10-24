"""
rank.py -- `page_rank` for the collection website: looked up automatically (Open PageRank) when a message is added.

The lab's rule is that the API key never goes onto a machine that visits hostile pages. Doing this lookup on the VM is a
deliberate exception, so the key is handled as carefully as this allows:

  * it is read from a FILE (default /etc/lejet-web/openpagerank.key, owned by root, readable by the service user only),
    never from an environment variable: the page-parsing workers are separate `spawn`ed processes and would inherit the
    environment, but not the main process's memory;
  * it is never printed, logged, cached, put in an error message or written to the CSV, and never sent anywhere except
    openpagerank.com and the address it redirects to, openpagerank.keywordseverywhere.com, over https;
  * without a usable key page_rank is simply -1 everywhere and the site works as before.

The service has MOVED: openpagerank.com answers every request with a 301 redirect to
openpagerank.keywordseverywhere.com (found 2026-10-24). The lab's urllib scripts follow it silently; this client follows it
too, but only to the hosts below, only over https, at most 3 hops, and the key is never sent to any other host.

The answer rules are the lab's (src/lejet/page_rank.py, shared with scripts/lookup_page_rank.py): a rank is 0..10, a domain
with no rank is 0 ("unrated"), and a FAILED look-up is -1 and is never turned into 0 or cached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.parse
from typing import Dict, Optional

import httpx

from lejet import page_rank as shared

log = logging.getLogger("uvicorn.error")
MISSING = -1
DEFAULT_KEY_FILE = "/etc/lejet-web/openpagerank.key"
TRUSTED_HOSTS = frozenset({"openpagerank.com", "www.openpagerank.com", "openpagerank.keywordseverywhere.com"})
REDIRECT_CODES = (301, 302, 303, 307, 308)
MAX_REDIRECTS = 3
_LOCAL = ("127.0.0.1", "localhost")
_ENV_LINE = "OPENPAGERANK_API_KEY="


def read_key(path: str) -> Optional[str]:
    """The key in `path` (just the key; a `OPENPAGERANK_API_KEY=...` line is accepted too), or None if there is no usable one."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(4097)
    except OSError:
        return None
    if len(raw) > 4096:
        return None
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if text.startswith(_ENV_LINE):
        text = text[len(_ENV_LINE):].strip()
    return text if shared.KEY_RE.fullmatch(text) else None


class RankClient:
    def __init__(self, key: Optional[str], cache_path: Optional[str] = None, *, base_url: str = shared.DEFAULT_URL,
                 transport: Optional[httpx.AsyncBaseTransport] = None, min_interval: float = 0.5, timeout: float = 10.0,
                 pause_seconds: float = 300.0, retries: int = 1, retry_delay: float = 1.5):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "https" and parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("the Open PageRank URL must be https (http only for localhost tests)")
        self._key = key or None
        self.enabled = self._key is not None
        self.base_url = base_url
        self.min_interval = min_interval
        self.last_error = ""
        self.pause_seconds = pause_seconds                        # after "bad key" / "too many requests": stop asking for a while
        self.retries, self.retry_delay = retries, retry_delay     # the service can be slow: a timeout or a 5xx gets one more try
        self._transient = False
        self._pause_until = 0.0
        self._logged = ("", 0.0)
        self._cache_path = cache_path
        self._cache: Dict[str, int] = self._load_cache()
        self._lock = asyncio.Lock()
        self._last = 0.0
        self._client = (httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=False, trust_env=False,
                                          headers={"Accept": "application/json", "User-Agent": "Le-Jet-research/1.0"})
                        if self.enabled else None)

    @classmethod
    def from_file(cls, key_path: str, cache_path: Optional[str] = None, **kw) -> "RankClient":
        key = read_key(key_path)
        if key is None:
            log.warning("page_rank is OFF: no usable key in %s (rows get page_rank = -1)", key_path)
        else:
            log.info("page_rank is ON (key file %s)", key_path)
        return cls(key, cache_path, **kw)

    def __repr__(self) -> str:                                    # the key must never end up in a log line by accident
        return f"RankClient(enabled={self.enabled}, cached={len(self._cache)})"

    # ------------------------------------------------------------------ cache (domains and ranks only, never the key)
    def _load_cache(self) -> Dict[str, int]:
        if not self._cache_path or not os.path.exists(self._cache_path):
            return {}
        try:
            with open(self._cache_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            return {k: v for k, v in raw.items()
                    if shared.DOMAIN_RE.fullmatch(k) and isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 10}
        except (OSError, ValueError, AttributeError):
            return {}

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._cache_path)), exist_ok=True)
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(dict(sorted(self._cache.items())), fh)
            os.replace(tmp, self._cache_path)
        except OSError:
            pass

    # ------------------------------------------------------------------ the look-up
    async def lookup(self, domain: str) -> int:
        """The rank (0..10) of a registrable domain, or -1 if it could not be measured."""
        dom = shared.refang_domain(domain)
        if not self.enabled or not shared.DOMAIN_RE.fullmatch(dom):
            return MISSING
        if dom in self._cache:
            return self._cache[dom]
        if time.monotonic() < self._pause_until:                  # the service said no a moment ago: do not waste more requests
            return MISSING
        async with self._lock:                                    # one request at a time, spaced out
            if dom in self._cache:
                return self._cache[dom]
            if time.monotonic() < self._pause_until:
                return MISSING
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                rank = await self._ask(dom)
                tries = 0
                while rank is None and self._transient and tries < self.retries and time.monotonic() >= self._pause_until:
                    tries += 1
                    await asyncio.sleep(self.retry_delay)
                    rank = await self._ask(dom)
            finally:
                self._last = time.monotonic()
        if rank is None:
            return MISSING                                        # a failure is never 0 and never cached
        self._cache[dom] = rank
        self._save_cache()
        return rank

    def _fail(self, reason: str, dom: str) -> None:
        """Remember why a look-up failed, and say so in the log (reason and domain only: never the key). The same reason is
        logged at most once a minute."""
        self.last_error = reason
        now = time.monotonic()
        if reason != self._logged[0] or now - self._logged[1] > 60:
            self._logged = (reason, now)
            log.warning("page_rank look-up failed (%s) for %s: the row gets page_rank = -1", reason, dom)

    def _trusted_redirect(self, current: str, location: str) -> Optional[str]:
        """Where a redirect may lead: a trusted host (or the configured host itself), over https, on the default port, with no
        credentials in the address. Anything else -> None, and the key is not sent there."""
        if not location:
            return None
        target = urllib.parse.urljoin(current, location)
        parts = urllib.parse.urlsplit(target)
        host = (parts.hostname or "").lower()
        base_host = (urllib.parse.urlsplit(self.base_url).hostname or "").lower()
        local = host in _LOCAL and base_host in _LOCAL                        # only for tests against a fake on localhost
        if host != base_host and host not in TRUSTED_HOSTS:
            return None
        if (parts.scheme != "https" or parts.port not in (None, 443)) and not local:
            return None
        if parts.username or parts.password:
            return None
        return target

    async def _ask(self, dom: str) -> Optional[int]:
        assert self._client is not None and self._key is not None
        self._transient = False
        url, params = self.base_url, [("domains[]", dom)]                     # a redirect carries the query string itself
        try:
            body = b""
            for _ in range(MAX_REDIRECTS + 1):
                async with self._client.stream("GET", url, params=params, headers={"API-OPR": self._key}) as resp:
                    if resp.status_code in REDIRECT_CODES:
                        nxt = self._trusted_redirect(url, resp.headers.get("location", ""))
                        if nxt is None:
                            self._fail("redirected to an address that is not trusted", dom)
                            return None
                        url, params = nxt, None
                        continue
                    if resp.status_code != 200:
                        if resp.status_code in (401, 403, 429):           # a bad key or no quota left: asking again only wastes requests
                            self._pause_until = time.monotonic() + self.pause_seconds
                        self._transient = resp.status_code >= 500
                        self._fail(f"HTTP {resp.status_code}", dom)
                        return None
                    body = b""
                    async for chunk in resp.aiter_bytes():
                        body += chunk
                        if len(body) > shared.MAX_RESPONSE:
                            self._fail("response too large", dom)
                            return None
                break
            else:
                self._fail("too many redirects", dom)
                return None
            got = shared.parse_answer(body, [dom])
        except RuntimeError as exc:                               # parse_answer's messages contain nothing of the answer
            self._fail(str(exc), dom)
            return None
        except Exception as exc:                                  # noqa: BLE001  (only the type: never the request, never the key)
            self._transient = True                                # a timeout / connection problem: worth one more try
            self._fail(type(exc).__name__, dom)
            return None
        if dom not in got:
            self._fail("no usable answer for the domain", dom)
            return None
        self.last_error = ""
        return got[dom]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
