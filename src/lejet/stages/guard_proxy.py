"""
guard_proxy.py -- a tiny local forward proxy that is the SSRF guard for the screenshot browser.

Why a proxy: Chromium follows redirects, WebSockets and sub-resource loads inside its own network stack,
so a request filter inside the browser cannot see where they really go, and re-fetching every request
from outside the browser (Playwright's `route.fetch`) breaks real pages (a different TLS fingerprint:
bot-protected sites reset the connection). With every connection forced through THIS proxy, Chromium keeps
its native network stack and the check happens where it cannot be bypassed: at connect time.

For every connection the proxy:
  * accepts only `CONNECT host:port` (HTTPS, WebSockets) and absolute-URI `http://` requests;
  * allows only ports 80 and 443 and only plain host names or IP literals (no `%00`, no `0x7f.1`);
  * RESOLVES THE NAME ITSELF and refuses unless EVERY address is a public (global) one, so a name that
    also answers with 10.x.x.x / 127.x.x.x / 169.254.x.x is refused (DNS rebinding), and then connects to
    the address it has just checked (no second lookup);
  * limits head size, idle time, bytes per connection and simultaneous connections.
A refused CONNECT gets `403`; a refused plain-HTTP request just gets its connection closed (a 403 page
would otherwise be rendered as an ordinary empty page). `refused` counts refusals for the manifest.

`allow_loopback=True` is TEST MODE (fixtures on 127.0.0.1): it lets exactly `localhost`/`127.0.0.1`
through on any port, and only to loopback addresses. Real runs never set it.
The VM firewall and the gateway remain the outer barrier; this is the inner one.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from typing import Awaitable, Callable, List, Optional, Tuple
from urllib.parse import urlsplit

_HOST_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?\.)*[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?$")
_NUMERIC_LOOKING = re.compile(r"^[0-9a-fx.:]+$")             # 2130706433, 0x7f.0.0.1, 127.1 ... must be real IPs
_DROP_HEADERS = {"proxy-connection", "connection", "keep-alive", "proxy-authorization", "upgrade-insecure-requests"}
LOOPBACK_TEST_HOSTS = ("localhost", "127.0.0.1")

Resolver = Callable[[str, int], Awaitable[List[str]]]


async def system_resolver(host: str, port: int) -> List[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    out: List[str] = []
    for _fam, _typ, _proto, _canon, sockaddr in infos:
        if sockaddr[0] not in out:
            out.append(sockaddr[0])
    return out


def _is_ip(text: str) -> Optional[ipaddress._BaseAddress]:
    try:
        return ipaddress.ip_address(text.strip("[]"))
    except ValueError:
        return None


class GuardProxy:
    def __init__(self, *, allow_loopback: bool = False, resolver: Optional[Resolver] = None,
                 connect_timeout: float = 8.0, idle_timeout: float = 45.0, head_timeout: float = 15.0,
                 max_connections: int = 64, max_bytes: int = 300 * 1024 * 1024):
        self.allow_loopback = allow_loopback
        self._resolve: Resolver = resolver or system_resolver
        self._connect_timeout = connect_timeout
        self._idle_timeout = idle_timeout
        self._head_timeout = head_timeout
        self._max_bytes = max_bytes
        self._sem = asyncio.Semaphore(max_connections)
        self._server: Optional[asyncio.AbstractServer] = None
        self._tasks: set = set()
        self.port = 0
        self.allowed = 0
        self.refused = 0
        self.last_refusal = ""

    # ---- life cycle ----
    async def start(self) -> int:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0, limit=32 * 1024)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 3)
            except Exception:                                      # noqa: BLE001
                pass
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=3)

    async def __aenter__(self) -> "GuardProxy":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ---- the policy ----
    async def check(self, host: str, port: int) -> Optional[List[str]]:
        """The addresses to connect to, or None if this destination is refused. Never raises."""
        try:
            host = host.strip().lower().rstrip(".")
            if not (1 <= port <= 65535) or not host:
                return self._refuse("bad destination")
            loop_ok = self.allow_loopback and host in LOOPBACK_TEST_HOSTS
            literal = _is_ip(host)
            if not loop_ok:
                if port not in (80, 443):
                    return self._refuse("port")
                if literal is None and (_NUMERIC_LOOKING.match(host) or not _HOST_RE.fullmatch(host)):
                    return self._refuse("host name")
            addrs = [str(literal)] if literal is not None else await asyncio.wait_for(self._resolve(host, port), 10)
            if not addrs:
                return self._refuse("no address")
            for a in addrs:
                ip = _is_ip(a)
                if ip is None:
                    return self._refuse("unparseable address")
                if loop_ok:
                    if not ip.is_loopback:
                        return self._refuse("loopback name resolves elsewhere")
                elif not ip.is_global or ip.is_multicast:
                    return self._refuse("non-public address")     # ANY bad answer refuses the whole name
            addrs.sort(key=lambda a: 0 if ":" not in a else 1)    # IPv4 first (the lab has no IPv6)
            return addrs
        except Exception:                                          # noqa: BLE001
            return self._refuse("lookup failed")

    def _refuse(self, why: str) -> None:
        self.refused += 1
        self.last_refusal = why
        return None

    # ---- one client connection ----
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            async with self._sem:
                await self._serve(reader, writer)
        except (asyncio.CancelledError, Exception):                # noqa: BLE001
            pass
        finally:
            if task is not None:
                self._tasks.discard(task)
            await self._close(writer)

    @staticmethod
    async def _close(writer: Optional[asyncio.StreamWriter]) -> None:
        if writer is None:
            return
        try:
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), 2)
        except Exception:                                          # noqa: BLE001
            pass

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), self._head_timeout)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
            return
        lines = head.split(b"\r\n")
        parts = lines[0].decode("latin-1").split(" ")
        if len(parts) != 3 or not parts[2].startswith("HTTP/"):
            return
        method, target = parts[0].upper(), parts[1]

        if method == "CONNECT":
            host, _, port_text = target.rpartition(":")
            if not host or not port_text.isdigit():
                writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                return
            addrs = await self.check(host, int(port_text))
            if addrs is None:
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                await writer.drain()
                return
            upstream = await self._connect(addrs, int(port_text))
            if upstream is None:
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                await writer.drain()
                return
            self.allowed += 1
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            await self._pipe(reader, writer, upstream[0], upstream[1])
            return

        # plain HTTP, absolute-URI form only
        if not target.lower().startswith("http://"):
            return
        try:
            u = urlsplit(target)
            host, port = u.hostname or "", u.port or 80
        except ValueError:
            return
        addrs = await self.check(host, port)
        if addrs is None:
            return                                                 # close without an answer (see the module note)
        upstream = await self._connect(addrs, port)
        if upstream is None:
            return
        self.allowed += 1
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        out = [f"{method} {path} HTTP/1.1".encode("latin-1")]
        has_host = False
        for raw in lines[1:]:
            if not raw:
                continue
            name = raw.split(b":", 1)[0].strip().lower().decode("latin-1")
            if name in _DROP_HEADERS:
                continue
            has_host = has_host or name == "host"
            out.append(raw)
        if not has_host:
            out.append(f"Host: {u.netloc}".encode("latin-1"))
        out.append(b"Connection: close")
        upstream[1].write(b"\r\n".join(out) + b"\r\n\r\n")
        await upstream[1].drain()
        await self._pipe(reader, writer, upstream[0], upstream[1])

    async def _connect(self, addrs: List[str], port: int) -> Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
        for addr in addrs[:3]:
            try:
                return await asyncio.wait_for(asyncio.open_connection(addr, port), self._connect_timeout)
            except Exception:                                      # noqa: BLE001
                continue
        return None

    async def _pipe(self, cr: asyncio.StreamReader, cw: asyncio.StreamWriter,
                    ur: asyncio.StreamReader, uw: asyncio.StreamWriter) -> None:
        total = [0]

        async def copy(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    chunk = await asyncio.wait_for(src.read(65536), self._idle_timeout)
                    if not chunk:
                        break
                    total[0] += len(chunk)
                    if total[0] > self._max_bytes:
                        break
                    dst.write(chunk)
                    await dst.drain()
            except (asyncio.CancelledError, Exception):            # noqa: BLE001
                pass

        a = asyncio.ensure_future(copy(cr, uw))
        b = asyncio.ensure_future(copy(ur, cw))
        try:
            await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (a, b):
                t.cancel()
            await asyncio.wait({a, b}, timeout=2)
            await self._close(uw)
