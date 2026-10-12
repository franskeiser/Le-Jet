"""
Tests for src/lejet/stages/guard_proxy.py: the local proxy that is the SSRF guard of the screenshot browser.
Everything runs on 127.0.0.1 with local servers and a fake name resolver; nothing touches the internet.

Run with:  python -m pytest tests/test_guard_proxy.py
"""

from __future__ import annotations

import asyncio

import pytest

from lejet.stages.guard_proxy import GuardProxy


def run(coro):
    return asyncio.run(coro)


async def talk(port: int, payload: bytes, *, read_timeout: float = 3.0, expect_close: bool = True) -> bytes:
    """Send bytes to the proxy and return everything it answers until it closes (or the timeout)."""
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(payload)
    await w.drain()
    out = b""
    try:
        while True:
            chunk = await asyncio.wait_for(r.read(65536), read_timeout)
            if not chunk:
                break
            out += chunk
    except asyncio.TimeoutError:
        pass
    w.close()
    return out


class Fixtures:
    """A local echo server and a local HTTP server, with a record of what they received."""

    def __init__(self):
        self.echo_port = self.http_port = 0
        self.heads = []
        self._servers = []

    async def start(self):
        async def echo(r, w):
            try:
                while True:
                    d = await r.read(4096)
                    if not d:
                        break
                    w.write(d)
                    await w.drain()
            finally:
                w.close()

        async def http(r, w):
            head = await r.readuntil(b"\r\n\r\n")
            self.heads.append(head)
            body = b"hello from the fixture"
            w.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body)
            await w.drain()
            w.close()

        s1 = await asyncio.start_server(echo, "127.0.0.1", 0)
        s2 = await asyncio.start_server(http, "127.0.0.1", 0)
        self._servers = [s1, s2]
        self.echo_port = s1.sockets[0].getsockname()[1]
        self.http_port = s2.sockets[0].getsockname()[1]
        return self

    async def stop(self):
        for s in self._servers:
            s.close()
            await s.wait_closed()


def fake_resolver(table):
    async def resolve(host, port):
        if host not in table:
            raise OSError("no such name")
        return list(table[host])
    return resolve


def connect(target: str) -> bytes:
    return f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode()


# ---------------------------------------------------------------------------------------
# refusals (production mode: no loopback exception)
# ---------------------------------------------------------------------------------------
REFUSED_TARGETS = [
    "127.0.0.1:80", "localhost:80", "10.0.0.5:443", "10.77.0.1:80", "192.168.1.1:80", "172.16.0.1:443",
    "169.254.169.254:80", "100.64.0.1:80", "0.0.0.0:80", "224.0.0.1:80", "198.18.0.1:80", "240.0.0.1:80",
    "[::1]:80", "[fe80::1]:80", "[fd00::1]:443", "[::ffff:127.0.0.1]:80",
    "2130706433:80", "0x7f.0.0.1:80", "127.1:80", "017700000001:80", "0177.0.0.1:80",
    "example.com:22", "example.com:25", "example.com:445", "example.com:8080", "example.com:8443", "example.com:3389",
    "example.com:0", "example.com:99999", "example.com:abc",
    "a b.com:80", "%00.com:80", "evil.com%00.example.com:80", ("x" * 70) + ".com:80", "-bad.com:80", "bad..com:80",
]


@pytest.mark.parametrize("target", REFUSED_TARGETS)
def test_dangerous_connect_targets_are_refused(target):
    async def go():
        async with GuardProxy(resolver=fake_resolver({"example.com": ["93.184.216.34"]})) as p:
            reply = await talk(p.port, connect(target))
            return reply, p.allowed
    reply, allowed = run(go())
    assert allowed == 0
    if " " in target:                                      # a request line with a space is not even parseable
        assert reply == b""
    else:                                                  # a real, immediate refusal (never silence)
        assert reply.startswith((b"HTTP/1.1 403", b"HTTP/1.1 400")), reply


@pytest.mark.parametrize("answers", [
    ["10.0.0.5"], ["127.0.0.1"], ["169.254.169.254"], ["192.168.0.10"], ["::1"], ["fe80::1"], ["100.64.0.9"],
    ["93.184.216.34", "10.0.0.5"],                  # a mixed answer: ONE bad address refuses the whole name
    ["10.0.0.5", "93.184.216.34"],
    ["93.184.216.34", "127.0.0.1", "93.184.216.35"],
    [],
])
def test_a_name_that_resolves_to_anything_non_public_is_refused(answers):
    async def go():
        async with GuardProxy(resolver=fake_resolver({"evil.example.com": answers})) as p:
            reply = await talk(p.port, connect("evil.example.com:443"))
            return reply, p.allowed, p.refused
    reply, allowed, refused = run(go())
    assert reply.startswith(b"HTTP/1.1 403") and allowed == 0 and refused == 1


def test_a_name_that_does_not_resolve_is_refused_not_a_crash():
    async def go():
        async with GuardProxy(resolver=fake_resolver({})) as p:
            return await talk(p.port, connect("nothing.example.com:443")), p.refused
    reply, refused = run(go())
    assert reply.startswith(b"HTTP/1.1 403") and refused == 1


def test_a_public_name_is_allowed_and_the_proxy_connects_to_the_address_it_checked():
    """No internet: the connect step is replaced by a local echo server; we record WHICH address was used."""
    async def go():
        fx = await Fixtures().start()
        async with GuardProxy(resolver=fake_resolver({"shop.example.com": ["93.184.216.34"]})) as p:
            seen = []

            async def fake_connect(addrs, port):
                seen.append((addrs, port))
                return await asyncio.open_connection("127.0.0.1", fx.echo_port)
            p._connect = fake_connect
            r, w = await asyncio.open_connection("127.0.0.1", p.port)
            w.write(connect("shop.example.com:443"))
            head = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 3)
            w.write(b"ping through the tunnel")
            await w.drain()
            echoed = await asyncio.wait_for(r.read(100), 3)
            w.close()
        await fx.stop()
        return head, echoed, seen, p.allowed
    head, echoed, seen, allowed = run(go())
    assert head.startswith(b"HTTP/1.1 200") and echoed == b"ping through the tunnel"
    assert seen == [(["93.184.216.34"], 443)] and allowed == 1


def test_ipv4_is_tried_before_ipv6():
    async def go():
        async with GuardProxy(resolver=fake_resolver({"dual.example.com": ["2606:4700::1", "93.184.216.34"]})) as p:
            return await p.check("dual.example.com", 443)
    assert run(go()) == ["93.184.216.34", "2606:4700::1"]


# ---------------------------------------------------------------------------------------
# test mode: loopback fixtures
# ---------------------------------------------------------------------------------------
def test_loopback_works_only_in_test_mode_and_only_for_the_two_names():
    async def go():
        fx = await Fixtures().start()
        out = {}
        async with GuardProxy(allow_loopback=True) as p:
            for target in (f"127.0.0.1:{fx.echo_port}", f"localhost:{fx.echo_port}"):
                r, w = await asyncio.open_connection("127.0.0.1", p.port)
                w.write(connect(target))
                out[target.split(":")[0]] = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 3)
                w.close()
            for target in ("127.0.0.2:80", "10.0.0.5:80", "169.254.169.254:80", "[::1]:80"):
                out[target] = await talk(p.port, connect(target))
        await fx.stop()
        return out
    out = run(go())
    assert out["127.0.0.1"].startswith(b"HTTP/1.1 200") and out["localhost"].startswith(b"HTTP/1.1 200")
    for k in ("127.0.0.2:80", "10.0.0.5:80", "169.254.169.254:80", "[::1]:80"):
        assert out[k].startswith(b"HTTP/1.1 403"), k


def test_loopback_in_production_mode_is_refused_even_with_the_right_port():
    async def go():
        fx = await Fixtures().start()
        async with GuardProxy() as p:
            reply = await talk(p.port, connect(f"127.0.0.1:{fx.echo_port}"))
        await fx.stop()
        return reply
    assert run(go()).startswith(b"HTTP/1.1 403")


# ---------------------------------------------------------------------------------------
# plain HTTP through the proxy
# ---------------------------------------------------------------------------------------
def test_plain_http_is_forwarded_in_origin_form_with_connection_close():
    async def go():
        fx = await Fixtures().start()
        async with GuardProxy(allow_loopback=True) as p:
            req = (f"GET http://127.0.0.1:{fx.http_port}/a/b?x=1 HTTP/1.1\r\nHost: 127.0.0.1:{fx.http_port}\r\n"
                   "Proxy-Connection: keep-alive\r\nProxy-Authorization: Basic AAAA\r\nUser-Agent: t\r\n\r\n").encode()
            reply = await talk(p.port, req)
        head = fx.heads[0] if fx.heads else b""
        await fx.stop()
        return reply, head
    reply, head = run(go())
    assert reply.startswith(b"HTTP/1.1 200") and reply.endswith(b"hello from the fixture")
    text = head.decode("latin-1").lower()
    assert text.startswith("get /a/b?x=1 http/1.1\r\n") and "connection: close" in text
    assert "proxy-connection" not in text and "proxy-authorization" not in text and "user-agent: t" in text


@pytest.mark.parametrize("url", ["http://10.0.0.5/secret", "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data/",
                                 "http://127.0.0.1/", "http://localhost/", "http://[::1]/", "http://example.com:8080/",
                                 "http://2130706433/", "http://user:pw@10.0.0.5/"])
def test_plain_http_to_forbidden_places_is_closed_without_any_answer(url):
    async def go():
        async with GuardProxy(resolver=fake_resolver({"example.com": ["93.184.216.34"]})) as p:
            reply = await talk(p.port, f"GET {url} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
            return reply, p.refused, p.allowed
    reply, refused, allowed = run(go())
    assert reply == b"" and refused >= 1 and allowed == 0                    # no 403 page for the browser to render


# ---------------------------------------------------------------------------------------
# junk and abuse
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    b"", b"\r\n\r\n", b"GARBAGE\r\n\r\n", b"GET\r\n\r\n", b"GET / HTTP/1.1\r\n\r\n",       # origin-form: not a proxy request
    b"GET ftp://example.com/ HTTP/1.1\r\n\r\n", b"GET file:///etc/passwd HTTP/1.1\r\n\r\n",
    b"POST gopher://x/ HTTP/1.1\r\n\r\n", b"CONNECT \r\n\r\n", b"CONNECT nohost HTTP/1.1\r\n\r\n",
    b"\x00\x01\x02\x03\x04\r\n\r\n", b"GET http://example.com/ SPDY/3\r\n\r\n",
])
def test_junk_requests_are_dropped_and_the_proxy_keeps_working(payload):
    async def go():
        async with GuardProxy(resolver=fake_resolver({"example.com": ["93.184.216.34"]}), head_timeout=0.5) as p:
            first = await talk(p.port, payload, read_timeout=2)
            second = await talk(p.port, connect("10.0.0.5:80"))                # still alive and still strict
            return first, second, p.allowed
    first, second, allowed = run(go())
    assert not first.startswith(b"HTTP/1.1 200") and second.startswith(b"HTTP/1.1 403") and allowed == 0


def test_an_oversized_head_is_dropped():
    async def go():
        async with GuardProxy(head_timeout=2) as p:
            return await talk(p.port, b"GET http://example.com/ HTTP/1.1\r\n" + b"X: " + b"a" * 100_000 + b"\r\n\r\n", read_timeout=3)
    assert run(go()) == b""


def test_a_client_that_never_finishes_its_request_is_cut_off():
    async def go():
        async with GuardProxy(head_timeout=0.4) as p:
            r, w = await asyncio.open_connection("127.0.0.1", p.port)
            w.write(b"CONNECT example.com:443 HTTP/1.1\r\nHost: x")           # no blank line
            await w.drain()
            data = await asyncio.wait_for(r.read(100), 3)                      # EOF when the proxy gives up
            w.close()
            return data
    assert run(go()) == b""


def test_an_idle_tunnel_is_closed():
    async def go():
        fx = await Fixtures().start()
        async with GuardProxy(allow_loopback=True, idle_timeout=0.4) as p:
            r, w = await asyncio.open_connection("127.0.0.1", p.port)
            w.write(connect(f"127.0.0.1:{fx.echo_port}"))
            await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 3)
            data = await asyncio.wait_for(r.read(100), 3)                      # nothing sent: closed after ~0.4 s
            w.close()
        await fx.stop()
        return data
    assert run(go()) == b""


def test_a_tunnel_stops_at_the_byte_cap():
    async def go():
        fx = await Fixtures().start()
        async with GuardProxy(allow_loopback=True, max_bytes=50_000) as p:
            r, w = await asyncio.open_connection("127.0.0.1", p.port)
            w.write(connect(f"127.0.0.1:{fx.echo_port}"))
            await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 3)
            got = 0
            try:
                for _ in range(40):
                    w.write(b"x" * 10_000)
                    await w.drain()
                    chunk = await asyncio.wait_for(r.read(100_000), 1.5)
                    if not chunk:
                        break
                    got += len(chunk)
            except (asyncio.TimeoutError, ConnectionError):
                pass
            w.close()
        await fx.stop()
        return got
    assert run(go()) < 400_000                                                # it did not echo everything forever


def test_close_cancels_open_tunnels_quickly():
    async def go():
        fx = await Fixtures().start()
        p = GuardProxy(allow_loopback=True, idle_timeout=60)
        await p.start()
        r, w = await asyncio.open_connection("127.0.0.1", p.port)
        w.write(connect(f"127.0.0.1:{fx.echo_port}"))
        await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 3)
        t0 = asyncio.get_running_loop().time()
        await p.close()
        elapsed = asyncio.get_running_loop().time() - t0
        w.close()
        await fx.stop()
        return elapsed
    assert run(go()) < 6


def test_the_refusal_counter_counts_every_refusal_and_remembers_the_reason():
    async def go():
        async with GuardProxy(resolver=fake_resolver({"a.example.com": ["10.0.0.9"]})) as p:
            await talk(p.port, connect("a.example.com:443"))
            await talk(p.port, connect("10.0.0.5:80"))
            await talk(p.port, connect("example.org:22"))
            return p.refused, p.last_refusal
    refused, why = run(go())
    assert refused == 3 and why == "port"
