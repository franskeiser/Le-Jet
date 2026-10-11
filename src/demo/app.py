"""Local web page for the defense demo.

PROTOTYPE ONLY: the model is trained on synthetic data, and page inspection is simulated from a
stored table. This server never opens, fetches or resolves any link, and it listens on
127.0.0.1 only.

Usage (from the project root, after src.demo.synthetic and src.demo.train_demo):
    python -m src.demo.app            # then open http://127.0.0.1:8000
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from src.demo.model import DemoModel

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 20_000


def make_handler(model: DemoModel):
    examples = model.examples()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/examples":
                self._json(200, {"examples": examples, "warning": model.meta["WARNING"]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/analyze":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                return self._json(413, {"error": "message too long"})
            try:
                body = json.loads(self.rfile.read(length))
                text = str(body["text"]).strip()
            except (ValueError, KeyError):
                return self._json(400, {"error": "send JSON like {\"text\": \"...\", \"inspect\": true}"})
            if not text:
                return self._json(400, {"error": "empty message"})
            self._json(200, model.analyze(text, inspect=bool(body.get("inspect", True))))

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    model = DemoModel()
    server = HTTPServer(("127.0.0.1", args.port), make_handler(model))
    print(f"Le-Jet demo (synthetic prototype) at http://127.0.0.1:{args.port}  -  Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
