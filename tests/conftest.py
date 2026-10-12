"""
Shared pytest setup for the web-inspection tests (WEB_INSPECTION_PLAN.md section 16, Level 0).

* puts `src` and the repository root on sys.path, so both `lejet.*` and `src.features.*`
  import the same way they do in production (plan 8.1, "import style");
* provides `server`: a local web server on 127.0.0.1 that serves the Level 0 fixtures
  (redirect chains, loops, huge bodies, a gzip bomb, trickling bodies, ...). The inspector is
  pointed at it in TEST MODE (`allow_loopback`). No test ever contacts the internet.
"""

from __future__ import annotations

import gzip
import http.server
import os
import socketserver
import sys
import threading
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

sys.dont_write_bytecode = True

from fixtures import pages  # noqa: E402


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"          # the connection closes after each response

    def log_message(self, *args, **kwargs):   # silence the console
        pass

    # -- helpers --------------------------------------------------------------------------
    def _send(self, status, body=b"", ctype="text/html; charset=utf-8", headers=(), length=True):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            if length:
                self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _redirect(self, location, status=302, extra=()):
        self._send(status, b"", headers=[("Location", location), *extra])

    # -- routing --------------------------------------------------------------------------
    def do_GET(self):
        self.server.hits.append((self.path, dict(self.headers)))
        path = self.path
        port = self.server.server_port
        try:
            if path.startswith("/page/"):
                name = path[len("/page/"):]
                if name in pages.PAGES:
                    return self._send(200, pages.PAGES[name])
                return self._send(404, b"no such fixture")
            if path.startswith("/redirect/"):
                n = int(path.rsplit("/", 1)[1])
                # /redirect/N answers with exactly N redirects before the final page
                return self._redirect(f"/redirect/{n - 1}" if n > 1 else "/page/plain")
            if path == "/loop":
                return self._redirect("/loop")
            if path == "/to-private":
                return self._redirect("http://10.0.0.5/secret")
            if path == "/to-ftp":
                return self._redirect("ftp://example.com/x")
            if path == "/to-port":
                return self._redirect("http://example.com:8080/")
            if path == "/to-metadata":
                return self._redirect("http://169.254.169.254/latest/meta-data/")
            if path == "/meta-private":
                return self._send(200, b"<html><head><meta http-equiv='refresh' content='0; url=http://192.168.1.1/admin'>"
                                       b"</head><body></body></html>")
            if path == "/cross-host":
                return self._redirect(f"http://localhost:{port}/page/plain")
            if path == "/huge":
                return self._send(200, pages.big_html(3 * 1024 * 1024))
            if path == "/malformed":
                return self._send(200, pages.malformed())
            if path == "/gzipbomb":
                raw = b"<html><head><title>Bomb</title></head><body>" + b"A" * (40 * 1024 * 1024)
                comp = gzip.compress(raw, 9)
                return self._send(200, comp, headers=[("Content-Encoding", "gzip")])
            if path == "/bigheader":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(50_000_000))
                self.end_headers()
                time.sleep(2)
                return
            if path == "/slow":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body>")
                for _ in range(120):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.2)
                return
            if path == "/delay2":
                time.sleep(1.5)
                return self._send(200, pages.PAGES["plain"])
            if path == "/popups":
                # a hostile page for the browser tests: dialogs, popups, an automatic download and
                # requests (fetch + WebSocket) aimed at a private address
                return self._send(200, b"<html><head><title>popups</title></head><body><h1>popups</h1><script>"
                                       b"alert('a'); confirm('b'); prompt('c');"
                                       b"window.open('about:blank'); window.open('/page/plain');"
                                       b"var a=document.createElement('a'); a.href='/download'; a.download='x.bin';"
                                       b"document.body.appendChild(a); a.click();"
                                       b"fetch('http://10.0.0.5/secret').catch(function(){});"
                                       b"try { new WebSocket('ws://10.0.0.5/x'); } catch (e) {}"
                                       b"</script></body></html>")
            if path == "/download":
                return self._send(200, b"A" * 1000, ctype="application/octet-stream",
                                  headers=[("Content-Disposition", "attachment; filename=x.bin")])
            if path == "/json":
                return self._send(200, b'{"a": 1}', ctype="application/json")
            if path == "/404":
                return self._send(404, b"<html><body>not found</body></html>")
            if path == "/403":
                return self._send(403, b"<html><body>forbidden</body></html>")
            if path == "/500":
                return self._send(500, b"<html><body>boom</body></html>")
            if path == "/empty":
                return self._send(200, b"")
            if path == "/204":
                return self._send(204, b"")
            if path == "/cookie-loop":
                if "seen=1" in self.headers.get("Cookie", ""):
                    return self._send(200, pages.PAGES["plain"])
                return self._redirect("/cookie-loop", extra=[("Set-Cookie", "seen=1; Path=/")])
            return self._send(404, b"unknown route")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.hits = []


class LocalSite:
    def __init__(self):
        self._srv = _Server()
        self.port = self._srv.server_port
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    def url(self, path: str = "/") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    @property
    def hits(self):
        return self._srv.hits

    def reset(self):
        self._srv.hits.clear()

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture(scope="session")
def server():
    site = LocalSite()
    yield site
    site.close()


@pytest.fixture(autouse=True)
def _reset_hits(request):
    if "server" in request.fixturenames:
        request.getfixturevalue("server").reset()
    yield
