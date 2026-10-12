"""
rdap_stage.py -- Stage 1b: RDAP (plan section 13.4b).

A lookup ABOUT a domain at its registry; it is not a visit to the site. It runs in the
analysis VM over HTTPS through the gateway (port 443 is allowed).

Rules implemented here:
  * the IANA bootstrap file (`dns.json`) is read from a local copy (fetch it during setup while
    the network is open); a TLD without an RDAP server gives rdap_ok = 0;
  * `GET {base}domain/{registrable_domain}`: timeout 4 s, response capped at 256 KB, redirects
    followed at most twice and only to https; accept application/rdap+json;
  * at most 1 request per second per RDAP server; one retry with backoff on 429 / 5xx / errors;
  * results are cached per registrable domain (raw timestamps are cached so age and expiry are
    recomputed with the current time); the cache can be loaded from / saved to a JSON file;
  * free-hosting and path-hosted hosts never reach this stage (rdap_skipped is set upstream).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple
from urllib.parse import urljoin, urlsplit

from lejet import safety
from lejet.features.external_features import FAILED, RdapRecord, record_from_response

Fetcher = Callable[[str], Awaitable[Tuple[int, bytes]]]

_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63}$")
RDAP_TIMEOUT = 4.0


class Bootstrap:
    """IANA RDAP bootstrap for the DNS namespace: TLD -> RDAP base URL."""

    def __init__(self, services: Any):
        self._map: Dict[str, str] = {}
        if isinstance(services, dict):
            services = services.get("services", [])
        for entry in services if isinstance(services, list) else []:
            try:
                tlds, urls = entry[0], entry[1]
            except (IndexError, TypeError, KeyError):
                continue
            https = [u for u in urls if isinstance(u, str) and u.lower().startswith("https://")]
            if not https:
                continue
            base = https[0] if https[0].endswith("/") else https[0] + "/"
            for tld in tlds:
                if isinstance(tld, str):
                    self._map.setdefault(tld.lower(), base)

    @classmethod
    def from_file(cls, path: str) -> "Bootstrap":
        try:
            with open(path, "rb") as fh:
                raw = fh.read(5 * 1024 * 1024)
            return cls(json.loads(raw.decode("utf-8", "replace")))
        except (OSError, ValueError):
            return cls([])

    def base_url(self, tld: str) -> Optional[str]:
        return self._map.get((tld or "").lower())

    def __len__(self) -> int:
        return len(self._map)


@dataclass
class RdapOutcome:
    record: RdapRecord
    from_cache: bool = False
    ms: int = 0
    note: str = ""                    # e.g. "no_rdap_server", "bad_domain", "fetch_failed"


class HttpxRdapFetcher:
    """The real fetcher: https only, redirects <= 2 and only to https, body capped."""

    def __init__(self, timeout: float = RDAP_TIMEOUT):
        import httpx

        self._httpx = httpx
        self._client = httpx.AsyncClient(
            follow_redirects=False, http2=False,
            timeout=httpx.Timeout(timeout, connect=timeout),
            headers={"Accept": "application/rdap+json, application/json;q=0.8",
                     "User-Agent": "Le-Jet-research/1.0 (thesis; RDAP lookup)"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __call__(self, url: str) -> Tuple[int, bytes]:
        current = url
        for _ in range(3):
            if urlsplit(current).scheme != "https":
                return 0, b""
            async with self._client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location")
                    if not loc:
                        return 0, b""
                    current = urljoin(current, loc)
                    continue
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > safety.MAX_RDAP_BYTES:
                        return 0, b""                         # too big: treat as a failed lookup
                return resp.status_code, body
        return 0, b""


class RdapClient:
    def __init__(self, bootstrap: Bootstrap, fetch: Fetcher, *, min_interval: float = 1.0,
                 retry_backoff: float = 1.5, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.bootstrap = bootstrap
        self._fetch = fetch
        self._min_interval = min_interval
        self._backoff = retry_backoff
        self._clock = clock
        self._sleep = sleep
        self._last_start: Dict[str, float] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._cache: Dict[str, RdapRecord] = {}
        self._inflight: Dict[str, "asyncio.Future[RdapOutcome]"] = {}
        self.requests_made = 0

    # ---- cache (host-side file pushed in / pulled back, plan 21.2 / 21.3) ----
    def load_cache(self, path: str) -> int:
        try:
            with open(path, "rb") as fh:
                data = json.loads(fh.read(20 * 1024 * 1024).decode("utf-8", "replace"))
        except (OSError, ValueError):
            return 0
        n = 0
        if isinstance(data, dict):
            for domain, e in data.items():
                if not isinstance(e, dict) or not _DOMAIN_RE.fullmatch(str(domain)):
                    continue
                try:
                    rec = RdapRecord(bool(e["ok"]), int(e["registered"]),
                                     None if e.get("reg_ts") is None else int(e["reg_ts"]),
                                     None if e.get("exp_ts") is None else int(e["exp_ts"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if rec.ok:                       # failures are never cached on disk
                    self._cache[str(domain)] = rec
                    n += 1
        return n

    def save_cache(self, path: str) -> int:
        data = {d: {"ok": r.ok, "registered": r.registered, "reg_ts": r.reg_ts, "exp_ts": r.exp_ts}
                for d, r in self._cache.items() if r.ok}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True)
        return len(data)

    # ---- lookup ----
    async def lookup(self, domain: str, tld: str) -> RdapOutcome:
        """RDAP for one registrable domain (ASCII/IDNA form). Never raises."""
        domain = (domain or "").lower().rstrip(".")
        started = self._clock()
        if not _DOMAIN_RE.fullmatch(domain):
            return RdapOutcome(FAILED, note="bad_domain")
        if domain in self._cache:
            return RdapOutcome(self._cache[domain], from_cache=True)
        base = self.bootstrap.base_url(tld)
        if not base:
            return RdapOutcome(FAILED, note="no_rdap_server")
        if domain in self._inflight:                      # same domain asked twice at once
            return await asyncio.shield(self._inflight[domain])
        fut: "asyncio.Future[RdapOutcome]" = asyncio.get_running_loop().create_future()
        self._inflight[domain] = fut
        try:
            outcome = await self._lookup_uncached(domain, base)
            outcome.ms = int((self._clock() - started) * 1000)
            if outcome.record.ok:
                self._cache[domain] = outcome.record
            fut.set_result(outcome)
            return outcome
        except BaseException as exc:                      # noqa: BLE001
            if not fut.done():
                fut.set_result(RdapOutcome(FAILED, note=f"error:{type(exc).__name__}"))
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                raise
            return RdapOutcome(FAILED, note=f"error:{type(exc).__name__}")
        finally:
            self._inflight.pop(domain, None)

    async def _throttle(self, host: str) -> None:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._last_start.get(host, -1e9) + self._min_interval - self._clock()
            if wait > 0:
                await self._sleep(wait)
            self._last_start[host] = self._clock()

    async def _lookup_uncached(self, domain: str, base: str) -> RdapOutcome:
        url = f"{base}domain/{domain}"
        host = urlsplit(base).hostname or base
        for attempt in (0, 1):
            await self._throttle(host)
            self.requests_made += 1
            try:
                status, body = await asyncio.wait_for(self._fetch(url), RDAP_TIMEOUT * 3)
            except Exception:                              # noqa: BLE001 - network, timeout, ...
                status, body = 0, b""
            retryable = status == 429 or status >= 500 or status == 0
            if not retryable or attempt == 1:
                break
            await self._sleep(self._backoff)
        record = record_from_response(status, body)
        return RdapOutcome(record, note="" if record.ok else "fetch_failed")
