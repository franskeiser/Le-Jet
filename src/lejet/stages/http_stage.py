"""
http_stage.py -- Stage 3: the HTTP probe (plan section 13.5).

What it does, in plain words: fetch the page the way a phone would, follow the redirect chain
BY HAND (so every hop can be re-checked), read at most 2 MB of the decoded body, and hand the
body to the parse pool. It never executes anything and never loads a linked resource.

Safety properties (each is covered by a test in tests/test_http_stage.py):
  * redirects are followed manually, at most `max_hops`; EVERY hop target goes through the
    Stage 0 gate again (urlnorm.normalize) and through the DNS rebinding backstop;
  * the body is read decoded-and-capped (2 MB; a gzip bomb cannot expand past it); a server
    that announces > 10 MB is refused without reading;
  * a wall-clock deadline per hop and per URL, so a server that trickles one byte at a time
    (which never trips the read timeout) is cut off;
  * a fresh in-memory cookie jar per URL (a new client), HTTP/1.1 only, gzip only;
  * 1 request at a time per registrable domain;
  * <meta refresh> and trivial script redirects are treated as hops (found by pattern, not by
    running JavaScript).
"""

from __future__ import annotations

import asyncio
import socket
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

from lejet import domains, safety, urlnorm
from lejet.features.html_features import looks_like_html
from lejet.settings import Settings

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_LOCATION_LEN = 2048


@dataclass
class Hop:
    idx: int
    status: int
    host: str
    location_host: str
    ms: int


@dataclass
class HttpResult:
    fail_reason: str = "none"
    scheme_used: str = ""
    hops: List[Hop] = field(default_factory=list)
    final_url: str = ""
    final_status: int = -1
    content_type: str = ""
    redirects: int = 0                    # number of redirects followed (-> hop_count)
    chain_complete: bool = False
    body_truncated: bool = False
    analysis: Optional[Dict[str, Any]] = None   # parse-pool output for the final page
    html_ms: int = 0
    elapsed_ms: int = 0
    errors: List[str] = field(default_factory=list)
    phase: str = "connect"                # connect | read (for timeout classification)


class DomainLocks:
    """At most one in-flight request per registrable domain (plan 13.9)."""

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}

    def lock_for(self, domain: str) -> asyncio.Lock:
        lock = self._locks.get(domain)
        if lock is None:
            lock = self._locks[domain] = asyncio.Lock()
        return lock


def _is_loopback_test_host(host: str, settings: Settings) -> bool:
    return settings.allow_loopback and host in urlnorm.LOOPBACK_TEST_HOSTS


def classify_exception(exc: BaseException) -> Tuple[str, str]:
    """Map an httpx / socket exception to (fail_reason, note)."""
    import httpx

    note = type(exc).__name__
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout", note
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout)):
        return "read_timeout", note
    if isinstance(exc, httpx.PoolTimeout):
        return "connect_timeout", note
    if isinstance(exc, httpx.ConnectError):
        cause = exc.__cause__ or exc.__context__
        text = str(exc).lower() + " " + str(cause).lower()
        if isinstance(cause, ssl.SSLError) or "ssl" in text or "certificate" in text or "handshake" in text:
            return "tls_error", note
        if isinstance(cause, socket.gaierror) or "name or service not known" in text \
                or "getaddrinfo" in text or "nodename" in text:
            return "dns_no_answer", note
        return "connect_refused", note
    if isinstance(exc, (httpx.UnsupportedProtocol, httpx.InvalidURL)):
        return "rejected_url", note
    if isinstance(exc, (httpx.RemoteProtocolError, httpx.ReadError, httpx.DecodingError,
                        httpx.WriteError, httpx.CloseError, httpx.TooManyRedirects)):
        return "blocked_or_empty", note
    if isinstance(exc, ssl.SSLError):
        return "tls_error", note
    return "internal_error", note


class HttpStage:
    def __init__(self, settings: Settings, pool, dns=None, locks: Optional[DomainLocks] = None):
        self.s = settings
        self.pool = pool
        self.dns = dns
        self.locks = locks or DomainLocks()

    def _client(self):
        import httpx

        s = self.s
        return httpx.AsyncClient(
            follow_redirects=False, http2=False, verify=False, trust_env=False,
            timeout=httpx.Timeout(connect=s.connect_timeout, read=s.read_timeout,
                                  write=s.write_timeout, pool=s.pool_timeout),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=0),
            headers={
                "User-Agent": s.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Encoding": "gzip",
                "Accept-Language": s.accept_language,
            },
        )

    # ------------------------------------------------------------------ public
    async def fetch(self, norm: "urlnorm.NormResult") -> HttpResult:
        """Fetch one normalized URL. https first, then http once for schemeless links
        (plan 13.2); the fallback only happens if the FIRST hop could not connect."""
        candidates = norm.fetch_candidates or ([norm.normalized_url] if norm.normalized_url else [])
        result = HttpResult()
        for i, candidate in enumerate(candidates):
            result = await self._attempt(candidate)
            no_response = not result.hops
            if no_response and result.fail_reason in ("connect_refused", "connect_timeout", "tls_error") \
                    and i + 1 < len(candidates):
                continue
            break
        return result

    # ----------------------------------------------------------------- internals
    async def _attempt(self, first_url: str) -> HttpResult:
        res = HttpResult()
        started = time.monotonic()
        try:
            async with self._client() as client:
                await self._walk(client, first_url, res)
        except TimeoutError:
            res.fail_reason = "connect_timeout" if res.phase == "connect" else "read_timeout"
            res.chain_complete = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:                               # noqa: BLE001
            res.fail_reason, note = classify_exception(exc)
            res.errors.append(f"http:{note}")
            res.chain_complete = False
        res.elapsed_ms = int((time.monotonic() - started) * 1000)
        return res

    async def _walk(self, client, first_url: str, res: HttpResult) -> None:
        """Follow the chain. The per-URL budget (`url_deadline`) counts ACTIVE time only: time spent
        waiting behind another request to the same registrable domain is excluded, otherwise one
        slow domain would turn its queue of URLs into spurious connect_timeouts."""
        current = first_url
        res.final_url = current
        walk_started = time.monotonic()
        waited_total = 0.0
        while True:
            budget = self.s.url_deadline - (time.monotonic() - walk_started - waited_total)
            if budget <= 0:
                res.fail_reason = "read_timeout" if res.hops else "connect_timeout"
                res.chain_complete = False
                return
            # --- Stage 0 gate on EVERY hop target (plan 13.5) ---
            nr = urlnorm.normalize(current, allow_loopback=self.s.allow_loopback)
            if not nr.ok:
                res.fail_reason = "redirect_to_blocked" if res.hops else (nr.fail_reason or "rejected_url")
                res.chain_complete = False
                return
            host = nr.host_ascii or ""
            res.final_url = current

            # --- DNS rebinding backstop for hosts the pipeline has not resolved yet ---
            if self.dns is not None and not _is_loopback_test_host(host, self.s):
                dres = await self.dns.resolve_host(host)
                if not dres.resolves:
                    if dres.status == "private_blocked" and res.hops:
                        res.fail_reason = "redirect_to_blocked"
                    else:
                        res.fail_reason = dres.fail_reason()
                    res.chain_complete = False
                    return

            domain = nr.registrable_domain or host
            res.phase = "connect"
            lock = self.locks.lock_for(domain)
            waited_from = time.monotonic()
            await lock.acquire()                    # one request at a time per registrable domain
            try:
                waited_total += time.monotonic() - waited_from
                t0 = time.monotonic()
                budget = self.s.url_deadline - (t0 - walk_started - waited_total)
                if budget <= 0:
                    res.fail_reason = "read_timeout" if res.hops else "connect_timeout"
                    res.chain_complete = False
                    return
                async with asyncio.timeout(min(self.s.hop_deadline, budget)):
                    async with client.stream("GET", current) as resp:
                        res.phase = "read"
                        status = resp.status_code
                        location = resp.headers.get("location")
                        ctype = resp.headers.get("content-type", "") or ""
                        is_redirect = status in REDIRECT_STATUSES and bool(location)
                        body = b""
                        too_large = False
                        if not is_redirect and 200 <= status < 300:
                            clen = None
                            try:
                                raw_len = resp.headers.get("content-length")
                                clen = int(raw_len) if raw_len is not None else None
                            except ValueError:
                                clen = None
                            if not safety.content_length_ok(clen):
                                too_large = True
                            else:
                                body, truncated = await self._read_capped(resp)
                                res.body_truncated = res.body_truncated or truncated
                ms = int((time.monotonic() - t0) * 1000)
            finally:
                lock.release()

            if not res.scheme_used:
                res.scheme_used = urlsplit(current).scheme
            res.final_status = status
            res.content_type = ctype[:128]
            loc_host = ""
            if is_redirect:
                try:
                    loc_host = (urlsplit(urljoin(current, location)).hostname or "")[:253]
                except ValueError:
                    loc_host = ""
            res.hops.append(Hop(len(res.hops), status, host, loc_host, ms))

            # --- 3xx with a usable Location -------------------------------------------------
            if is_redirect:
                if len(location) > MAX_LOCATION_LEN:
                    res.fail_reason, res.chain_complete = "redirect_to_blocked", False
                    return
                res.redirects += 1
                if res.redirects > self.s.max_hops:
                    res.fail_reason, res.chain_complete = "too_many_redirects", False
                    return
                try:
                    current = urljoin(current, location)
                except ValueError:
                    res.fail_reason, res.chain_complete = "redirect_to_blocked", False
                    return
                continue

            # --- a final response (or a redirect shell): classify it -----------------------
            res.chain_complete = True
            if 400 <= status < 500:
                res.fail_reason = "http_4xx"
                return
            if status >= 500:
                res.fail_reason = "http_5xx"
                return
            if too_large:
                res.fail_reason = "too_large"
                return
            if not (200 <= status < 300):
                res.fail_reason = "non_html"
                return
            if not body.strip():
                res.fail_reason = "blocked_or_empty"
                return
            if not looks_like_html(body, ctype):
                res.fail_reason = "non_html"
                return

            t1 = time.monotonic()
            analysis = await self.pool.analyze(body=body, content_type=ctype, base_url=current,
                                               want_features=True)
            res.html_ms += int((time.monotonic() - t1) * 1000)
            if analysis.get("error"):
                res.errors.append(f"parse:{analysis['error']}")
                res.fail_reason = "blocked_or_empty"
                return
            res.errors.extend(analysis.get("errors", []))

            target = analysis.get("redirect_url") or ""
            if target:                                         # <meta refresh> / script redirect
                res.redirects += 1
                if res.redirects > self.s.max_hops:
                    res.fail_reason, res.chain_complete = "too_many_redirects", False
                    return
                try:
                    current = urljoin(current, target)
                except ValueError:
                    res.fail_reason, res.chain_complete = "redirect_to_blocked", False
                    return
                res.chain_complete = False       # until the next hop answers
                continue

            res.analysis = analysis
            if analysis.get("challenge"):
                res.fail_reason = "challenge_page"
            elif analysis.get("parked"):
                res.fail_reason = "parked_or_default_page"
            elif not analysis.get("is_html"):
                res.fail_reason = "non_html"
            return

    async def _read_capped(self, resp) -> Tuple[bytes, bool]:
        """Read the DECODED body (gzip already undone) up to the 2 MB cap."""
        limit = safety.MAX_DECODED_BYTES
        chunks: List[bytes] = []
        seen = 0
        async for chunk in resp.aiter_bytes():
            seen += len(chunk)
            if seen > limit:
                keep = len(chunk) - (seen - limit)
                if keep > 0:
                    chunks.append(chunk[:keep])
                return b"".join(chunks), True
            chunks.append(chunk)
        return b"".join(chunks), False
