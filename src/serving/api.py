"""HTTP API for the Le-Jet frontend (frontend/src/data/backendClient.js talks to this).

    POST /predict          {"message": "...", "model_id": "fusion_gated"|"cnn_only_baseline", "research": bool}
    POST /inspect          {"message": "...", "refresh": bool}  -> hands the first link to the isolated lab (a job)
    GET  /inspect/<job>    the job's real steps and timings; when "done" call /predict with the same message
    GET  /evidence/<id>.png   screenshot the lab took of an inspected page (id = 16 hex characters)
    GET  /health           what is loaded (model seed, how many inspected links are in the store, lab status)

Safety: this server never opens, fetches or resolves a link. /inspect only hands the link to the lab's
unattended pipeline (scripts/run_batch.ps1), where the isolated analysis VM opens it. It listens on 127.0.0.1 only and answers
cross-origin requests only for the local frontend (Vite dev server / preview ports).
Screenshots are served only from the lab's validated screenshot folders, by 16-hex id, never by path.

Usage (project root):
    python -m src.serving.api            # http://127.0.0.1:5000  (the frontend's default VITE_API_BASE_URL)
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.features.link_features import extract_links
from src.serving.lejet_service import MAX_TEXT_CHARS, MODEL_IDS, LejetService, LinkStore
from src.serving.live_inspect import ON_DEMAND_CSV, LiveInspector

MAX_BODY = 20_000


class Server(ThreadingHTTPServer):
    """On Windows, SO_REUSEADDR (the standard library's default) lets a SECOND copy bind the same port and then splits the
    requests between the two servers. Never allow that: a second copy must fail with "address already in use"."""
    allow_reuse_address = False
    daemon_threads = True

ALLOWED_ORIGINS = {f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in (5050, 5173, 4173, 3000, 8080)}
EVIDENCE_RE = re.compile(r"^/evidence/([0-9a-f]{16})\.png$")
JOB_RE = re.compile(r"^/inspect/([0-9a-f]{12})$")


def make_handler(service: LejetService, inspector: LiveInspector):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            origin = self.headers.get("Origin", "")
            if origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self._json(200, {"status": "ok", "seed": service.seed, "models": list(MODEL_IDS),
                                        "fusion_dir": getattr(service, "fusion_dir_name", ""),
                                        "inspected_links": len(service.store), "live_lab": inspector.status()})
            job_match = JOB_RE.match(path)
            if job_match:
                job = inspector.job(job_match.group(1))
                return self._json(200, job.view()) if job else self._json(404, {"error": "no such job"})
            match = EVIDENCE_RE.match(path)
            if match:
                png = service.evidence_png(match.group(1))
                if png is None:
                    return self._json(404, {"error": "no screenshot"})
                return self._send(200, png, "image/png")
            self._json(404, {"error": "not found"})

        def do_POST(self):
            route = self.path.split("?")[0]
            if route not in ("/predict", "/inspect"):
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return self._json(400, {"error": "bad length"})
            if length > MAX_BODY:
                return self._json(413, {"error": "message too long"})
            try:
                body = json.loads(self.rfile.read(length))
                message = str(body["message"]).strip()
                model_id = str(body.get("model_id") or MODEL_IDS[0])
            except (ValueError, KeyError, TypeError):
                return self._json(400, {"error": "send JSON like {\"message\": \"...\"}"})
            if not message:
                return self._json(400, {"error": "empty message"})
            if len(message) > MAX_TEXT_CHARS * 5:
                return self._json(413, {"error": "message too long"})
            if route == "/inspect":
                links = extract_links(message[:MAX_TEXT_CHARS])
                if not links:
                    return self._json(400, {"error": "no link in the message"})
                try:
                    job = inspector.submit(links[0]["raw"], refresh=bool(body.get("refresh")))
                except RuntimeError as err:
                    return self._json(503, {"error": str(err)})
                except ValueError:
                    return self._json(400, {"error": "not a plain web address"})
                return self._json(202, job.view())
            try:
                self._json(200, service.predict(message, model_id))
            except ValueError as err:
                self._json(400, {"error": str(err)})

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

    return Handler


def default_fusion_dir(seed: int) -> Path:
    """The live-trained head when it is complete for this seed, else the committed one (models/fusion/rf_vec)."""
    fusion = Path(__file__).resolve().parents[2] / "models" / "fusion"
    live = fusion / "rf_vec_live"
    needed = (f"gates_seed{seed}.csv", f"phase_a_seed{seed}.pt", f"phase_c_seed{seed}.pt")
    return live if all((live / n).exists() for n in needed) else fusion / "rf_vec"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion-dir", type=Path, default=None,
                        help="folder of the trained head. Default: models/fusion/rf_vec_live (the live-trained one) when it "
                             "is there, otherwise models/fusion/rf_vec. The folder in use is printed at start and shown in /health")
    parser.add_argument("--live", type=Path, default=None, help="live_links.csv, the store of inspected links")
    parser.add_argument("--reuse-stored", action="store_true",
                        help="answer a link that is already in the store without opening it again (default: every link is inspected)")
    parser.add_argument("--no-lab", action="store_true", help="never hand links to the lab (answers come from the store only)")
    args = parser.parse_args()
    fusion_dir = args.fusion_dir or default_fusion_dir(args.seed)
    store = LinkStore(*([args.live] if args.live else []), on_demand_csv=ON_DEMAND_CSV)
    service = LejetService(seed=args.seed, fusion_dir=fusion_dir, store=store)
    service.fusion_dir_name = Path(fusion_dir).name
    inspector = LiveInspector(service.store, enabled=not args.no_lab, reuse_stored=args.reuse_stored)
    try:
        server = Server(("127.0.0.1", args.port), make_handler(service, inspector))
    except OSError as err:
        raise SystemExit(f"port {args.port} is already in use ({err.strerror}). Another Le-Jet API is probably running: "
                         "stop it first, or use --port.")
    print(f"Model folder: {fusion_dir}")
    print(f"Le-Jet API at http://127.0.0.1:{args.port}  ({len(service.store)} inspected links in the store; "
          f"live lab: {inspector.status()['note']})  Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
