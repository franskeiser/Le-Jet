"""
ct_stage.py -- Stage 1c: certificate-transparency age from crt.sh.

Why: some TLDs (.ph, .my, .jp, .vn, .cn, ...) have no RDAP service and no usable WHOIS, so
`domain_age` is unknown for them. Public certificate logs know when a domain FIRST got a TLS
certificate. That is NOT the registration date (a domain can be registered long before it gets a
certificate), so it is stored in its own column, `ct_first_cert_days`, and never replaces
`domain_age`. For phishing it is still a useful "how new is this site" signal, because most
phishing sites get a certificate within days of going live.

It is a lookup ABOUT a domain name at a public log; it is not a visit to the site. It runs in the
analysis VM over HTTPS through the gateway (port 443 is allowed).

Rules implemented here:
  * `GET https://crt.sh/?q=<domain>&output=json`; if that returns an empty list, once more with
    `q=%.<domain>` (certificates that only name subdomains);
  * https only, redirects not followed, timeout 20 s, response capped at 2 MB (larger = failed);
  * at most 1 request per `min_interval` (default 2 s); one retry with backoff on 429 / 5xx / error;
  * a circuit breaker: after 3 failed lookups in a row crt.sh is left alone for 5 minutes (those rows
    get ct_ok = 0 immediately), because the service is often overloaded and must not slow the run;
  * the answer is THIRD-PARTY and UNTRUSTED: tolerant parsing, only `not_before` of each entry is
    read, dates outside 1995..now are ignored;
  * results are cached per registrable domain (the first-certificate timestamp is cached, the age
    is recomputed with the current time); failures are never cached on disk.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, Optional, Tuple
from urllib.parse import quote

from lejet import safety
from lejet.features.external_features import _parse_dt

Fetcher = Callable[[str], Awaitable[Tuple[int, bytes]]]

CT_TIMEOUT = 10.0
BREAKER_FAILS = 3                   # this many failed lookups in a row ...
BREAKER_PAUSE = 300.0               # ... and crt.sh is left alone for this many seconds
_PAUSED = -1                        # internal status: the breaker is open
PROBE_NAME = "probe.lejet.invalid"  # .invalid never has certificates: a healthy crt.sh says []
CT_BASE = "https://crt.sh/"
_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63}$")
_MIN_TS = 788918400                 # 1996-01-01: no certificate is older than this
_MAX_ENTRIES = 50000


@dataclass(frozen=True)
class CtRecord:
    ok: bool                        # the lookup completed (HTTP 200 and parseable)
    first_ts: Optional[int] = None  # UNIX seconds of the oldest certificate, None = none found


FAILED = CtRecord(ok=False)


@dataclass
class CtOutcome:
    record: CtRecord
    from_cache: bool = False
    ms: int = 0
    note: str = ""


def parse_response(status: int, body: bytes, now_ts: Optional[int] = None) -> CtRecord:
    """Turn an HTTP status + (capped) body into a CtRecord. Never raises."""
    if status != 200:
        return FAILED
    now = int(datetime.now(timezone.utc).timestamp()) if now_ts is None else now_ts
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except (ValueError, RecursionError):
        return FAILED
    if not isinstance(data, list):
        return FAILED
    first: Optional[int] = None
    for entry in data[:_MAX_ENTRIES]:
        if not isinstance(entry, dict):
            continue
        ts = _parse_dt(entry.get("not_before"))
        if ts is None or ts < _MIN_TS or ts > now:
            continue
        if first is None or ts < first:
            first = ts
    return CtRecord(ok=True, first_ts=first)


def age_days(rec: CtRecord, now_ts: Optional[int] = None) -> int:
    """Days since the first certificate; -1 when unknown or none was found."""
    if not rec.ok or rec.first_ts is None:
        return -1
    now = int(datetime.now(timezone.utc).timestamp()) if now_ts is None else now_ts
    return (now - rec.first_ts) // 86400 if now >= rec.first_ts else -1


class HttpxCtFetcher:
    """The real fetcher: https only, no redirects, body capped at MAX_CT_BYTES."""

    def __init__(self, timeout: float = CT_TIMEOUT):
        import httpx

        self._client = httpx.AsyncClient(
            follow_redirects=False, http2=False,
            timeout=httpx.Timeout(timeout, connect=timeout),
            headers={"Accept": "application/json",
                     "User-Agent": "Le-Jet-research/1.0 (thesis; certificate-age lookup)"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __call__(self, url: str) -> Tuple[int, bytes]:
        if not url.lower().startswith("https://"):
            return 0, b""
        async with self._client.stream("GET", url) as resp:
            body = b""
            async for chunk in resp.aiter_bytes():
                body += chunk
                if len(body) > safety.MAX_CT_BYTES:
                    return 0, b""                           # too big: treat as a failed lookup
            return resp.status_code, body


class CtClient:
    def __init__(self, fetch: Fetcher, *, min_interval: float = 2.0, retry_backoff: float = 3.0,
                 breaker_fails: int = BREAKER_FAILS, breaker_pause: float = BREAKER_PAUSE,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self._fetch = fetch
        # circuit breaker: crt.sh is often overloaded; a dead service must not slow every row
        self._breaker_fails = breaker_fails
        self._breaker_pause = breaker_pause
        self._fail_streak = 0
        self._paused_until = -1e9
        self.paused_skips = 0
        self._min_interval = min_interval
        self._backoff = retry_backoff
        self._clock = clock
        self._sleep = sleep
        self._last_start = -1e9
        self._lock = asyncio.Lock()
        self._cache: Dict[str, CtRecord] = {}
        self._inflight: Dict[str, "asyncio.Future[CtOutcome]"] = {}
        self.requests_made = 0

    # ---- cache file ----
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
                ts = e.get("first_ts")
                if ts is not None and not (isinstance(ts, int) and not isinstance(ts, bool)
                                           and _MIN_TS <= ts <= 4102444800):
                    continue
                self._cache[str(domain)] = CtRecord(True, ts)
                n += 1
        return n

    def save_cache(self, path: str) -> int:
        data = {d: {"first_ts": r.first_ts} for d, r in self._cache.items() if r.ok}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True)
        return len(data)

    # ---- health probe ----
    async def probe(self) -> bool:
        """One cheap question to crt.sh before the run starts (a name that cannot have certificates,
        so a healthy service answers `[]` at once). If it fails, the breaker opens immediately and
        the whole pause is skipped instead of burning three timeouts in front of the rows."""
        status, body = await self._get(f"{CT_BASE}?q={quote(PROBE_NAME)}&output=json")
        healthy = status == 200 and parse_response(status, body).ok
        if not healthy:
            self._paused_until = self._clock() + self._breaker_pause
        return healthy

    # ---- lookup ----
    async def lookup(self, domain: str) -> CtOutcome:
        """Certificate age for one registrable domain (ASCII/IDNA form). Never raises."""
        domain = (domain or "").lower().rstrip(".")
        started = self._clock()
        if not _DOMAIN_RE.fullmatch(domain):
            return CtOutcome(FAILED, note="bad_domain")
        if domain in self._cache:
            return CtOutcome(self._cache[domain], from_cache=True)
        if domain in self._inflight:
            return await asyncio.shield(self._inflight[domain])
        if self._clock() < self._paused_until:               # breaker open: do not even try
            self.paused_skips += 1
            return CtOutcome(FAILED, note="ct_paused")
        fut: "asyncio.Future[CtOutcome]" = asyncio.get_running_loop().create_future()
        self._inflight[domain] = fut
        try:
            outcome = await self._lookup_uncached(domain)
            outcome.ms = int((self._clock() - started) * 1000)
            if outcome.record.ok:
                self._cache[domain] = outcome.record
                self._fail_streak = 0
            elif outcome.note != "ct_paused":                 # a skipped lookup is not a new failure
                self._fail_streak += 1
                if self._fail_streak >= self._breaker_fails:
                    self._paused_until = self._clock() + self._breaker_pause
                    self._fail_streak = 0
            fut.set_result(outcome)
            return outcome
        except BaseException as exc:                        # noqa: BLE001
            if not fut.done():
                fut.set_result(CtOutcome(FAILED, note=f"error:{type(exc).__name__}"))
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                raise
            return CtOutcome(FAILED, note=f"error:{type(exc).__name__}")
        finally:
            self._inflight.pop(domain, None)

    async def _get(self, url: str) -> Tuple[int, bytes]:
        """One rate-limited request. Retried once, but only on an explicit 429 / 5xx answer: after a
        timeout the service is not going to answer a second try any better. Returns status
        _PAUSED if the breaker opened while this lookup was waiting in line."""
        status, body = 0, b""
        for attempt in (0, 1):
            async with self._lock:                           # ONE request at a time, in order
                if self._clock() < self._paused_until:       # tripped while we queued
                    return _PAUSED, b""
                wait = self._last_start + self._min_interval - self._clock()
                if wait > 0:
                    await self._sleep(wait)
                self._last_start = self._clock()
                self.requests_made += 1
                try:
                    status, body = await asyncio.wait_for(self._fetch(url), CT_TIMEOUT)
                except Exception:                           # noqa: BLE001 - network, timeout, ...
                    status, body = 0, b""
            retryable = status == 429 or status >= 500
            if not retryable or attempt == 1:
                break
            await self._sleep(self._backoff)
        return status, body

    async def _lookup_uncached(self, domain: str) -> CtOutcome:
        status, body = await self._get(f"{CT_BASE}?q={quote(domain)}&output=json")
        if status == _PAUSED:
            self.paused_skips += 1
            return CtOutcome(FAILED, note="ct_paused")
        rec = parse_response(status, body)
        if rec.ok and rec.first_ts is None:                 # nothing for the bare name: try subdomains
            status, body = await self._get(f"{CT_BASE}?q={quote('%.' + domain)}&output=json")
            if status == _PAUSED:
                self.paused_skips += 1
                return CtOutcome(FAILED, note="ct_paused")
            rec = parse_response(status, body)
        return CtOutcome(rec, note="" if rec.ok else "fetch_failed")
