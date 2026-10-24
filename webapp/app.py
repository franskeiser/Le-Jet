"""
app.py -- the Le-Jet data-collection website: submit an SMS, its link is found and inspected, one row is appended
to the shared CSV (the crowd study's 57-column layout, see dataset.py).

    POST /api/preview     {"sms_text": ...}                          -> the link that would be used (nothing is saved)
    POST /api/submit      {"sms_text", "label", "language"}          -> inspect the link, append the row
    POST /api/screenshots multipart "files" (PNG/JPEG/WebP)          -> the messages found in the pictures (nothing is
                                                                         saved, the pictures are discarded; the page lets a
                                                                         person correct them and then calls /api/submit)
    GET  /api/rows        newest rows + counts for the page
    GET  /api/dataset.csv the whole CSV
    GET  /                the page (webapp/static/index.html)
    GET  /api/health

If LEJET_ACCESS_CODE is set, every /api call except health needs it in the X-Access-Code header; without it (LEJET_OPEN=1)
anyone who has the address can use the site. The lab pipeline runs
exactly as in `lejet.cli run`; whatever it fetches is untrusted, so the page only receives numbers and defanged links.

Settings come from environment variables (see webapp/deploy/lejet-web.service):
    LEJET_ACCESS_CODE     optional shared password; without it LEJET_OPEN=1 (or LEJET_DEV=1) must be set
    LEJET_DATASET         path of the shared CSV (default data/crowd/web_submissions.csv)
    LEJET_DNS             resolver the pipeline asks (default 1.1.1.1; the lab VM used 10.77.0.1)
    LEJET_RDAP_BOOTSTRAP  path of the IANA dns.json (domain age/expiry are -1 without it)
    LEJET_CACHE_DIR       where the RDAP / certificate lookups and the page_rank answers are cached
    LEJET_RANK_KEY_FILE   file holding the Open PageRank key (default /etc/lejet-web/openpagerank.key); no key = page_rank -1
    LEJET_DEV=1           local testing: no access code needed
"""

from __future__ import annotations

import asyncio
import datetime
import hmac
import logging
import os
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):      # `lejet` lives in src/, `src.features.link_features` needs the root
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile   # noqa: E402
from fastapi.responses import FileResponse, Response                  # noqa: E402
from pydantic import BaseModel, Field                                 # noqa: E402

from lejet import safety, schema, urlnorm                             # noqa: E402
from lejet.pipeline import Pipeline                                   # noqa: E402
from lejet.settings import Settings                                   # noqa: E402
from webapp import dataset, rank, screens                             # noqa: E402

MAX_CONCURRENT = 3                    # inspections running at once on the whole server
RATE_WINDOW_S = 60
RATE_MAX_SUBMITS = 30                 # submissions per client IP per window
RATE_MAX_IMAGES = 12                  # screenshots read per client IP per window
log = logging.getLogger("uvicorn.error")
STATIC = Path(__file__).resolve().parent / "static"

DEV = os.environ.get("LEJET_DEV") == "1"
OPEN = os.environ.get("LEJET_OPEN") == "1"          # the site is deliberately usable by anyone who has the address
ACCESS_CODE = os.environ.get("LEJET_ACCESS_CODE", "")   # optional: if set, every /api call (except health) needs it
DATASET_PATH = os.environ.get("LEJET_DATASET", str(ROOT / "data" / "crowd" / "web_submissions.csv"))
RANK_KEY_FILE = os.environ.get("LEJET_RANK_KEY_FILE", rank.DEFAULT_KEY_FILE)     # page_rank is OFF without a usable key there
CACHE_DIR = os.environ.get("LEJET_CACHE_DIR", "")
RANK_URL = os.environ.get("LEJET_RANK_URL", rank.shared.DEFAULT_URL)             # only for tests: https, or http on localhost


def build_settings() -> Settings:
    s = Settings()
    s.allow_loopback = False
    s.dns_server = os.environ.get("LEJET_DNS", "1.1.1.1")
    bootstrap = os.environ.get("LEJET_RDAP_BOOTSTRAP", "")
    s.rdap_bootstrap_path = bootstrap if bootstrap and os.path.exists(bootstrap) else None
    cache_dir = os.environ.get("LEJET_CACHE_DIR", "")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        s.rdap_cache_path = os.path.join(cache_dir, "rdap_cache.json")
        s.ct_cache_path = os.path.join(cache_dir, "ct_cache.json")
    s.ct_enabled = True              # certificate age only where RDAP gives nothing (e.g. .ph)
    s.ct_probe = False
    s.parse_workers = 2
    s.concurrency = MAX_CONCURRENT
    return s


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ACCESS_CODE and not (DEV or OPEN):
        raise RuntimeError("set LEJET_ACCESS_CODE, or LEJET_OPEN=1 to run without a code")
    app.state.store = dataset.Store(DATASET_PATH)
    app.state.write_lock = asyncio.Lock()
    app.state.pipeline = await Pipeline(build_settings()).start()
    app.state.ranker = rank.RankClient.from_file(                 # after the parse workers exist: they never get the key
        RANK_KEY_FILE, os.path.join(CACHE_DIR, "page_rank_cache.json") if CACHE_DIR else None, base_url=RANK_URL)
    app.state.sem = asyncio.Semaphore(MAX_CONCURRENT)
    app.state.hits = defaultdict(deque)           # ip -> timestamps of recent submissions
    app.state.image_hits = defaultdict(deque)     # ip -> timestamps of recent screenshots
    app.state.ocr_pool = ThreadPoolExecutor(max_workers=1)      # the OCR engine is used by one picture at a time
    try:                                          # screenshot reading is optional: without it only pasting works
        app.state.reader = await asyncio.get_running_loop().run_in_executor(app.state.ocr_pool, screens.Reader)
    except Exception as exc:                                              # noqa: BLE001
        app.state.reader = None
        log.warning("screenshot reading is OFF (%s: %s)", type(exc).__name__, exc)
    try:
        yield
    finally:
        app.state.ocr_pool.shutdown(wait=False, cancel_futures=True)
        await app.state.ranker.aclose()
        await app.state.pipeline.close()


app = FastAPI(title="Le-Jet data collection", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class PreviewRequest(BaseModel):
    sms_text: str = Field(max_length=20000)


class SubmitRequest(BaseModel):
    sms_text: str = Field(min_length=1, max_length=20000)
    label: Literal["smishing", "legitimate"]
    language: Literal["en", "tl", "taglish"]          # English, Tagalog or Taglish


def _auth(code: str) -> None:
    if not ACCESS_CODE:                  # open mode (LEJET_OPEN=1 or LEJET_DEV=1): no code to check
        return
    if not hmac.compare_digest(code.encode(), ACCESS_CODE.encode()):
        raise HTTPException(401, "Wrong access code.")


def _client_ip(request: Request) -> str:
    # Caddy (the reverse proxy on the same machine) sets X-Forwarded-For; nothing else can reach the app.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[-1].strip() if fwd else (request.client.host if request.client else "?")


def _check_rate(app_: FastAPI, ip: str, bucket: str = "hits", limit: int = RATE_MAX_SUBMITS, n: int = 1) -> None:
    now = time.monotonic()
    hits: Deque[float] = getattr(app_.state, bucket)[ip]
    while hits and now - hits[0] > RATE_WINDOW_S:
        hits.popleft()
    if len(hits) + n > limit:
        raise HTTPException(429, "Too many requests in the last minute. Wait a bit and try again.")
    hits.extend([now] * n)


def _link_of(text: str):
    """(raw link, normalised link) or an HTTP error explaining why the message cannot be used."""
    try:
        found = dataset.first_link(text)
    except Exception:                                                   # noqa: BLE001
        found = None
    if found is None:
        raise HTTPException(422, "No link found in this message. The dataset only holds messages that contain a link.")
    return found


@app.post("/api/preview")
async def preview(body: PreviewRequest, x_access_code: str = Header(default="")) -> Dict[str, Any]:
    _auth(x_access_code)
    text = dataset.clean_text(body.sms_text)
    try:
        found = dataset.first_link(text) if text else None
    except Exception:                                                   # noqa: BLE001
        found = None
    return {"link": safety.defang_url(found[0]) if found else None,
            "links_in_message": len(dataset.extract_links(text)) if text else 0}


@app.post("/api/submit")
async def submit(body: SubmitRequest, request: Request, x_access_code: str = Header(default="")) -> Dict[str, Any]:
    _auth(x_access_code)
    text = dataset.clean_text(body.sms_text)
    if not text:
        raise HTTPException(400, "The message is empty.")
    raw_link, norm_link = _link_of(text)
    if len(norm_link) > safety.MAX_URL_LEN:
        raise HTTPException(400, f"The link is longer than {safety.MAX_URL_LEN} characters.")
    store: dataset.Store = request.app.state.store
    dup = store.existing(text, raw_link)
    if dup:
        raise HTTPException(409, f"This exact message with this link is already in the dataset ({dup}).")
    _check_rate(request.app, _client_ip(request))

    submitted = datetime.datetime.now(datetime.timezone.utc)
    async with request.app.state.sem:
        insp = await request.app.state.pipeline.inspect(urlnorm.compute_url_id(norm_link), norm_link, "web")
    lab = schema.clean_row(insp.row)
    features = schema.full_features(lab)
    if lab.get("page_fetched") == 1:          # like the lab: only a page that was reached is looked up, the rest stay -1
        try:
            features["page_rank"] = await asyncio.wait_for(
                request.app.state.ranker.lookup(lab.get("final_reg_domain") or lab.get("reg_domain") or ""), 30)
        except asyncio.TimeoutError:
            pass

    async with request.app.state.write_lock:               # one writer at a time; ids and duplicates stay consistent
        dup = store.existing(text, raw_link)
        if dup:
            raise HTTPException(409, f"This exact message with this link is already in the dataset ({dup}).")
        row = dataset.build_row(store.next_id(), text, body.label, body.language, raw_link, lab, features, submitted)
        store.append(row)
    return {
        "message_id": row["message_id"],
        "link": safety.defang_url(raw_link),
        "final_url": safety.defang_url(row["final_url"]) if isinstance(row["final_url"], str) else None,
        "fetch_status": row["fetch_status"],
        "fail_reason": lab["fail_reason"],
        "features": {n: row[n] for n in dataset.FEATURES_37},
        "total": len(store),
    }


@app.post("/api/screenshots")
async def read_screenshots(request: Request, files: List[UploadFile] = File(...),
                           x_access_code: str = Header(default="")) -> Dict[str, Any]:
    _auth(x_access_code)
    reader = request.app.state.reader
    if reader is None:
        raise HTTPException(503, "Reading screenshots is not available on this server. Paste the message text instead.")
    if not files or len(files) > screens.MAX_IMAGES:
        raise HTTPException(400, f"Upload between 1 and {screens.MAX_IMAGES} screenshots at a time.")
    _check_rate(request.app, _client_ip(request), "image_hits", RATE_MAX_IMAGES, len(files))
    loop = asyncio.get_running_loop()
    images: List[Dict[str, Any]] = []
    for up in files:
        name = safety.clean_field(up.filename or "screenshot", max_len=80) or "screenshot"
        data = await up.read(screens.MAX_IMAGE_BYTES + 1)
        entry: Dict[str, Any] = {"name": name, "error": None, "messages": []}
        if len(data) > screens.MAX_IMAGE_BYTES:
            entry["error"] = f"Larger than {screens.MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        else:
            try:
                bubbles = await asyncio.wait_for(
                    loop.run_in_executor(request.app.state.ocr_pool, screens.read_screenshot, reader, data), 120)
                for b in bubbles:
                    text = dataset.clean_text(b.text)
                    try:
                        found = dataset.first_link(text)
                    except Exception:                                  # noqa: BLE001
                        found = None
                    entry["messages"].append({"text": text, "direction": b.direction,
                                              "link": safety.defang_url(found[0]) if found else None})
                if not entry["messages"]:
                    entry["error"] = "No text was found in this picture."
            except screens.ImageRejected as exc:
                entry["error"] = str(exc)
            except asyncio.TimeoutError:
                entry["error"] = "Reading this picture took too long."
            except Exception as exc:                                    # noqa: BLE001
                log.warning("screenshot failed: %s: %s", type(exc).__name__, exc)
                entry["error"] = "This picture could not be read."
        images.append(entry)
    return {"images": images}


@app.get("/api/rows")
async def rows(request: Request, limit: int = 50, x_access_code: str = Header(default="")) -> Dict[str, Any]:
    _auth(x_access_code)
    store: dataset.Store = request.app.state.store
    return {"total": len(store), "counts": store.counts, "languages": store.lang_counts,
            "rows": store.newest(max(1, min(limit, 200)))}


@app.get("/api/dataset.csv")
async def download(request: Request, x_access_code: str = Header(default="")) -> Response:
    _auth(x_access_code)
    data = Path(request.app.state.store.path).read_bytes()
    return Response(data, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="lejet_web_submissions.csv"'})


@app.get("/api/health")
async def health(request: Request) -> Dict[str, Any]:
    rdap = request.app.state.pipeline.rdap
    return {"ok": True, "rows": len(request.app.state.store), "code_required": bool(ACCESS_CODE),
            "screenshots": request.app.state.reader is not None,
            "page_rank": request.app.state.ranker.enabled,
            "page_rank_error": getattr(request.app.state.ranker, "last_error", "") or None,
            "rdap_tlds_known": len(rdap.bootstrap) if rdap is not None else 0}


@app.get("/robots.txt")
async def robots() -> Response:
    return Response("User-agent: *\nDisallow: /\n", media_type="text/plain")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"})
