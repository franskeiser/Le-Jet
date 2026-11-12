"""
shot_stage.py -- Stage 6 (optional): a real screenshot of the page, taken by headless Chromium.

THIS IS THE RISKIEST PART OF THE LAB: the page's JavaScript runs. The safeguards, from outside in:
  * it runs only in the analysis VM, as the unprivileged user `fetcher`, in its own systemd unit with
    memory / task / time limits (infra/analysis/helpers/lejet-sudo-shots.sh);
  * the Chromium SANDBOX IS ON (`chromium_sandbox=True`, never `--no-sandbox`; guarded in code);
  * EVERY connection the browser makes (page, redirects, sub-resources, WebSockets) goes through a local
    guard proxy (`guard_proxy.py`), one per page, which resolves each name itself and refuses anything
    that is not a public address on port 80 or 443, at connect time. Chromium keeps its own network
    stack, so real pages and bot-protected sites behave as in a normal visit. (An earlier design
    re-fetched every request through Playwright's Node client: it could not see redirects natively
    and it broke bot-protected pages because of its different TLS fingerprint.) The start URL is also
    checked with `request_allowed` (the Stage 0 gate) before a browser context is even created;
  * downloads, popups, dialogs, permissions, service workers: all blocked or dismissed;
  * one fresh, empty browser context per page (no cookies or storage ever shared between URLs);
  * hard per-page deadline; a hung or crashed browser is killed and replaced;
  * the PNG the browser writes is checked structurally (`pngcheck`) before it is kept, and again on
    the host before it is accepted. The firewall (VM and gateway) remains the outer barrier.

Playwright is imported lazily: the host (and the VM's unit tests) never need it unless a browser is
actually launched.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import ipaddress
import json
import os
import re
import sys
import tarfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from lejet import pngcheck, safety, urlnorm
from lejet.stages.guard_proxy import GuardProxy, Resolver

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36 Le-Jet-research/1.0")
VIEWPORT = {"width": 1280, "height": 800}
# Smishing links are opened on phones, and many pages hide from desktop browsers (blank or 403): the phone
# profile looks like what the victim's device sends (Android Chrome, a phone-sized screen, touch).
MOBILE_USER_AGENT = ("Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/124.0.0.0 Mobile Safari/537.36")
MOBILE_VIEWPORT = {"width": 412, "height": 915}
GOTO_TIMEOUT_MS = 20_000
SETTLE_MS = 3_000                   # extra time for the page to go quiet after "load"
SHOT_TIMEOUT_MS = 20_000            # heavy pages (video, ads) can need a while to settle for the shot
PAGE_DEADLINE_S = 25.0              # hard limit for one page, browser start-up excluded (was 70: a stuck page made a live check wait over a minute)
RETRY_ERRORS = frozenset({"screenshot_error", "internal_error", "browser_crash"})   # one more try for these
CHROMIUM_ARGS = [
    "--disable-extensions", "--disable-background-networking", "--disable-sync", "--no-first-run",
    "--disable-dev-shm-usage", "--disable-default-apps", "--mute-audio", "--hide-scrollbars",
    "--disable-breakpad", "--disable-component-update", "--disable-domain-reliability",
]
EARLY_SHOTS_AT_S = (0.3, 0.5)       # early pictures: ~0.3 s and ~0.8 s after the load event (pages that wipe
                                    # themselves do it within about a second; polling for text proved unreliable)
BLANK_BYTES = 12_000                # a 1280x800 / phone picture under this is (nearly) blank
REALISTIC_ARGS = ["--disable-blink-features=AutomationControlled"]    # stops announcing `navigator.webdriver`
MAX_SHOTS_TAR_MEMBERS = 20000

# --- "whole site" mode (the live tool on the cloud VM; the dataset lab leaves all of this switched off) ---------
FULL_PAGE_MAX_HEIGHT = 16_000       # Chromium cannot make a picture taller than 16 384 px (pngcheck.MAX_DIMENSION)
FULL_PAGE_HEIGHT_STEPS = (16_000, 8_000, 4_000, 2_000)   # a tall PNG over 4 MB (pngcheck.MAX_BYTES) is retried shorter
MAX_LINKED_PAGES = 4                # same-site pages followed after the landing page
MAX_TEXT_HEAD = 400                 # how much of the page's visible text is kept (whole-site mode): enough to read "this page has been removed"
SLOW_GOTO_TIMEOUT_MS = 45_000       # the "try harder" navigation timeout used by the reach ladder
LADDER_MAX_ATTEMPTS = 4
LADDER_BUDGET_S = 120.0             # no new rung is started after this long (a live check must not run for ten minutes)
REACH = ("live", "live_retry", "dead")                 # `archived_*` is decided later, on the host (it needs web APIs)
REASONS = ("ok", "nxdomain", "refused", "reset", "timeout", "tls", "blocked", "unreachable", "other")
PAGE_FILE_RE = re.compile(r"^([0-9a-f]{16})_p([1-9][0-9]?)\.png$")
_SKIP_LINK_EXT = frozenset({
    "zip", "rar", "7z", "gz", "tar", "apk", "exe", "msi", "dmg", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "mp3", "mp4", "avi", "mov", "wav", "jpg", "jpeg", "png", "gif", "webp", "svg", "ico", "css", "js", "json", "xml"})
_SKIP_LINK_WORDS = re.compile(r"(log-?out|sign-?out|unsubscribe|delete|remove|download)", re.I)


def pick_links(base_url: str, hrefs: Iterable[Any], limit: int = MAX_LINKED_PAGES,
               allow_loopback: bool = False) -> List[str]:
    """Which links of the landing page are worth a second look? Only pages of the SAME site (same host, `www.`
    ignored), at most `limit`, no duplicates, no files, no mail / phone / javascript links, no log-out style
    links. Pure function: the hrefs come from the page, so every one is re-checked with `request_allowed`."""
    from urllib.parse import urldefrag, urljoin, urlsplit

    try:
        base = urlsplit(base_url)
    except ValueError:
        return []
    base_host = (base.hostname or "").lower().removeprefix("www.")
    if not base_host:
        return []
    start = (base.path or "/", base.query)
    seen = {start}
    out: List[str] = []
    for href in hrefs:
        if len(out) >= limit:
            break
        if not isinstance(href, str) or not href.strip() or len(href) > 2000:
            continue
        low = href.strip().lower()
        if low.startswith(("mailto:", "tel:", "sms:", "javascript:", "data:", "blob:", "#")):
            continue
        try:
            full, _ = urldefrag(urljoin(base_url, href.strip()))
            parts = urlsplit(full)
        except ValueError:
            continue
        if parts.scheme not in ("http", "https"):
            continue
        if (parts.hostname or "").lower().removeprefix("www.") != base_host:
            continue
        path = parts.path or "/"
        ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
        if ext in _SKIP_LINK_EXT or _SKIP_LINK_WORDS.search(path):
            continue
        key = (path, parts.query)
        if key in seen:
            continue
        if not request_allowed(full, allow_loopback)[0]:
            continue
        seen.add(key)
        out.append(full)
    return out


def ladder_steps(url: str, mobile: bool) -> List[Tuple[str, bool, bool]]:
    """The 'open every site' ladder: (url, mobile profile, slow). First the normal visit, then a patient one with the
    other device profile, then the other scheme (http <-> https) and the `www.` toggle. At most LADDER_MAX_ATTEMPTS."""
    from urllib.parse import urlsplit, urlunsplit

    steps: List[Tuple[str, bool, bool]] = [(url, mobile, False), (url, not mobile, True)]
    try:
        p = urlsplit(url)
        host = p.hostname or ""
        other = "http" if p.scheme == "https" else "https"
        if host and p.scheme in ("http", "https"):
            steps.append((urlunsplit((other, p.netloc, p.path, p.query, "")), mobile, True))
            alt_host = host[4:] if host.startswith("www.") else "www." + host
            if not _is_ip(host) and host.count(".") <= 2:
                netloc = p.netloc.replace(host, alt_host, 1)
                steps.append((urlunsplit((p.scheme, netloc, p.path, p.query, "")), mobile, True))
    except ValueError:
        pass
    seen, out = set(), []
    for s in steps:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:LADDER_MAX_ATTEMPTS]


def classify_nav_error(text: str, refused: bool = False) -> str:
    """A short, closed reason for why a page could not be opened (the raw text is page/network influenced and is never stored)."""
    low = (text or "").lower()
    if refused:
        return "blocked"
    if "err_name_not_resolved" in low or "name_not_resolved" in low or "err_name_resolution" in low:
        return "nxdomain"
    if "refused" in low:
        return "refused"
    if "reset" in low or "err_empty_response" in low or "err_connection_closed" in low:
        return "reset"
    if "timed out" in low or "timeout" in low or "timed_out" in low:
        return "timeout"
    # the guard proxy could not open the connection the browser asked for (host down, no route, filtered): Chromium reports
    # this as a failed proxy tunnel; our own refusals are reported separately as `blocked`
    if "tunnel_connection_failed" in low or "address_unreachable" in low or "network_unreachable" in low:
        return "unreachable"
    if "ssl" in low or "cert" in low or "tls" in low:
        return "tls"
    return "other"


def launch_args(realistic: bool = False) -> List[str]:
    """The Chromium command-line switches. `realistic` removes the one that tells pages 'an automation tool is
    driving me'; many phishing and gambling pages answer that with a blank page (cloaking). The sandbox is never
    touched."""
    args = list(CHROMIUM_ARGS) + (list(REALISTIC_ARGS) if realistic else [])
    assert chromium_args_ok(args)
    return args

# error codes written to the manifest (a closed list: the host validator checks it)
ERRORS = ("none", "blocked_start", "goto_timeout", "navigation_error", "screenshot_error", "png_invalid",
          "page_deadline", "browser_crash", "internal_error")


# ---------------------------------------------------------------------------------------------
# pure helpers (unit-tested on the host, no browser)
# ---------------------------------------------------------------------------------------------
def selectable(row: Dict[str, Any]) -> bool:
    """Was the page fetched fine by the HTTP stage (so the screenshot matches the features)?"""
    return (row.get("fail_reason") == "none" and bool(row.get("html_available"))
            and isinstance(row.get("final_url"), str) and bool(row.get("final_url")))


# Nothing to photograph: refused by the safety gate, or the name does not exist in DNS at all.
SKIP_REASONS = frozenset({"rejected_url", "port_blocked", "redirect_to_blocked", "dns_nxdomain", "dns_no_answer"})
# The HTTP stage reached a server and got an answer: the browser should start where that answer was.
_ANSWERED = frozenset({"http_4xx", "http_5xx", "challenge_page", "parked_or_default_page", "blocked_or_empty",
                       "non_html", "too_large"})


def capture_target(row: Dict[str, Any], allow_loopback: bool = False) -> Optional[Tuple[str, bool]]:
    """What should the browser open for this row? Returns (url, ignore_tls) or None.

    EVERY URL that is safe to open gets an attempt, whatever the plain HTTP fetch did: a site that
    answered the bot with 403 may still show a page (or its block page) to a real browser, and that is
    exactly what a victim would see. Pages that failed only on TLS are opened with certificate errors
    tolerated (phishing sites often have broken certificates; the lab VM sits behind a VPN and reads
    nothing but pixels)."""
    fr = row.get("fail_reason")
    if fr in SKIP_REASONS:
        return None
    if selectable(row):
        return row["final_url"], False
    final = row.get("final_url")
    if fr in _ANSWERED and isinstance(final, str) and final.lower().startswith(("http://", "https://")):
        return final, False
    raw = row.get("url")
    if not isinstance(raw, str):
        return None
    norm = urlnorm.normalize(raw, allow_loopback=allow_loopback)
    if not norm.ok or not norm.normalized_url:
        return None
    return norm.normalized_url, fr == "tls_error"


_HOST_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?\.)*[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?$")


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def request_allowed(url: str, allow_loopback: bool = False) -> Tuple[bool, str]:
    """May the browser request this URL? Same gate as Stage 0, plus a few browser-only schemes.
    Never raises: it runs inside the browser's request handler, where an exception would leave the
    request hanging. Anything odd is a "no"."""
    try:
        if not isinstance(url, str):
            return False, "not a string"
        low = url.lower()
        if low.startswith(("data:", "blob:", "about:blank")):
            return True, "inline"                           # never leaves the browser
        if not low.startswith(("http://", "https://", "ws://", "wss://")):
            return False, "scheme"
        probe = "http" + url[url.index(":"):] if low.startswith(("ws://", "wss://")) else url
        norm = urlnorm.normalize(probe, allow_loopback=allow_loopback)
        if not norm.ok:
            return False, norm.fail_reason or "rejected_url"
        host = norm.host_ascii or ""
        if not (_HOST_RE.fullmatch(host) or _is_ip(host)):    # e.g. "%00", a 70-letter label
            return False, "bad_host"
        return True, ""
    except Exception:                                       # noqa: BLE001
        return False, "internal"


def chromium_args_ok(args: Sequence[str]) -> bool:
    """The sandbox must never be switched off."""
    bad = ("--no-sandbox", "--disable-setuid-sandbox", "--disable-seccomp-filter-sandbox",
           "--disable-gpu-sandbox", "--single-process")
    return not any(a.split("=", 1)[0] in bad for a in args)


def write_png(path: str, data: bytes) -> Tuple[bool, str, int, int]:
    """Check the PNG and write it atomically. Returns (ok, error, width, height)."""
    ok, reason, w, h = pngcheck.check_png(data)
    if not ok:
        return False, "png_invalid", 0, 0
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return True, "none", w, h


@dataclass
class ShotResult:
    url_id: str
    ok: bool = False
    error: str = "internal_error"
    bytes: int = 0
    sha256: str = ""
    width: int = 0
    height: int = 0
    partial: bool = False           # the page had not finished loading when the shot was taken
    blocked_requests: int = 0
    ms: int = 0
    # where the BROWSER ended up (not the plain fetch): defanged, cleaned and capped; text_chars is the
    # amount of visible text on the page (0 with a near-empty picture means a white screen), -1 = unknown
    final_url: str = ""
    title: str = ""
    text_chars: int = -1
    text_head: str = ""             # first MAX_TEXT_HEAD characters of the visible text (whole-site mode only): cleaned, capped, links defanged
    # True = the page showed content and then wiped itself (a script turned it blank or sent the browser to
    # about:blank, typical cloaking against automated browsers); the picture is the EARLY one, taken before that
    rescued: bool = False
    # --- whole-site mode only (ShotRunner(extended=True)); the dataset lab never sets these --------------------
    extended: bool = False
    reach: str = ""                 # live | live_retry | dead  (see REACH)
    reason: str = "ok"              # why a page could not be opened (see REASONS)
    truncated: bool = False         # the page was taller than FULL_PAGE_MAX_HEIGHT (or the PNG limit) and was cut
    attempts: int = 0
    nav_text: str = field(default="", repr=False)          # raw navigation error, never exported
    pages: List[Dict[str, Any]] = field(default_factory=list)   # linked pages: <url_id>_p<n>.png

    def as_dict(self) -> Dict[str, Any]:
        d = {"url_id": self.url_id, "ok": self.ok, "error": self.error, "bytes": self.bytes,
             "sha256": self.sha256, "width": self.width, "height": self.height,
             "partial": self.partial, "blocked_requests": self.blocked_requests, "ms": self.ms,
             "final_url": self.final_url, "title": self.title, "text_chars": self.text_chars,
             "rescued": self.rescued}
        if self.extended:                                   # extra keys ONLY in whole-site mode (validate_shots allows both)
            d.update({"reach": self.reach, "reason": self.reason, "truncated": self.truncated,
                      "attempts": self.attempts, "pages": self.pages, "text_head": self.text_head})
        return d


def url_id_for_log(res: "ShotResult") -> str:
    return res.url_id if len(res.url_id) == 16 else "?"


def pack_tar(png_dir: str, results: Iterable[ShotResult], out_tar: str) -> int:
    """<out_tar>: the PNGs that succeeded plus shots.jsonl. A plain, deterministic USTAR file: only
    regular files, fixed owner/time/mode, names `<url_id>.png` and `shots.jsonl`."""
    results = list(results)
    n = 0
    with tarfile.open(out_tar, "w", format=tarfile.USTAR_FORMAT) as tar:
        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
        for r in results:
            if r.ok:
                with open(os.path.join(png_dir, f"{r.url_id}.png"), "rb") as fh:
                    add(f"{r.url_id}.png", fh.read())
                n += 1
            for pg in r.pages:                                  # linked pages (whole-site mode)
                name = f"{r.url_id}_p{pg['n']}.png"
                path = os.path.join(png_dir, name)
                if os.path.exists(path):
                    with open(path, "rb") as fh:
                        add(name, fh.read())
        lines = "".join(json.dumps(r.as_dict(), sort_keys=True) + "\n" for r in results)
        add("shots.jsonl", lines.encode("utf-8"))
    return n


# ---------------------------------------------------------------------------------------------
# the browser part (needs playwright + chromium; used inside the VM)
# ---------------------------------------------------------------------------------------------
class ShotRunner:
    """Takes screenshots with one shared Chromium, a fresh context per page."""

    def __init__(self, out_dir: str, *, concurrency: int = 1, allow_loopback: bool = False,
                 page_deadline: float = PAGE_DEADLINE_S, launch_env: Optional[Dict[str, str]] = None,
                 mobile: bool = False, extra_wait_ms: int = 0, realistic: bool = False, rescue: bool = True,
                 extended: bool = False, full_page: bool = False, max_pages: int = 0, ladder: bool = False,
                 resolver: Optional[Resolver] = None, clock: Callable[[], float] = time.monotonic):
        # whole-site mode (the live tool): extended = extra manifest keys; full_page = top-to-bottom picture;
        # max_pages = same-site pages followed after the landing page; ladder = retry harder before giving up.
        # All off by default, so the dataset lab behaves exactly as before.
        self.extended = bool(extended or full_page or max_pages or ladder)
        self.full_page = bool(full_page)
        self.max_pages = max(0, min(int(max_pages), MAX_LINKED_PAGES))
        self.ladder = bool(ladder)
        self.resolver = resolver              # how the guard proxy looks names up (None = the system resolver)
        self.rescue = rescue                  # also take an early picture for pages that wipe themselves
        assert chromium_args_ok(CHROMIUM_ARGS)
        # realistic = the FULL Chromium in its newer headless mode (not the stripped "headless shell" build,
        # whose client hints say HeadlessChrome) and without the automation switch: closest to a victim's browser
        self.realistic = realistic
        self.out_dir = out_dir
        self.concurrency = max(1, concurrency)
        self.mobile = mobile                 # phone identity and screen (what a smishing victim's device sends)
        self.extra_wait_ms = max(0, min(int(extra_wait_ms), 30_000))   # more time for script-heavy pages
        self.allow_loopback = allow_loopback
        self.page_deadline = page_deadline
        self.launch_env = launch_env
        self._clock = clock
        self._pw: Any = None
        self._browser: Any = None
        self._browser_lock = asyncio.Lock()
        self.restarts = 0

    # -- browser life cycle ---------------------------------------------------------------
    async def start(self) -> "ShotRunner":
        from playwright.async_api import async_playwright          # lazy: VM only

        os.makedirs(self.out_dir, exist_ok=True)
        self._pw = await async_playwright().start()
        await self._launch()
        return self

    async def _launch(self) -> None:
        kwargs: Dict[str, Any] = dict(
            headless=True, chromium_sandbox=True, args=launch_args(self.realistic), timeout=60_000,
            env=self.launch_env,
            proxy={"server": "per-context"},          # each page gets its own guard proxy (see _shoot)
        )
        if self.realistic:
            kwargs["channel"] = "chromium"            # the full browser, new headless mode
        self._browser = await self._pw.chromium.launch(**kwargs)

    async def _browser_for_use(self) -> Any:
        async with self._browser_lock:
            if self._browser is None or not self._browser.is_connected():
                try:
                    if self._browser is not None:
                        await self._browser.close()
                except Exception:                                     # noqa: BLE001
                    pass
                self.restarts += 1
                await self._launch()
            return self._browser

    async def close(self) -> None:
        # Playwright's own request-filter tasks may still be waiting on a slow tracker when the last page
        # closes; give them a moment so they end quietly instead of being destroyed mid-wait.
        try:
            others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if others:
                await asyncio.wait(others, timeout=6)
        except Exception:                                             # noqa: BLE001
            pass
        try:
            if self._browser is not None:
                await asyncio.wait_for(self._browser.close(), 10)
        except Exception:                                             # noqa: BLE001
            pass
        try:
            if self._pw is not None:
                await asyncio.wait_for(self._pw.stop(), 10)
        except Exception:                                             # noqa: BLE001
            pass

    async def __aenter__(self) -> "ShotRunner":
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -- one page ---------------------------------------------------------------------------
    async def shoot(self, url_id: str, url: str, ignore_tls: bool = False) -> ShotResult:
        """One page, with ONE more try after a transient failure (a fresh context each time)."""
        started = self._clock()
        if not self.ladder:
            res = await self._attempt(url_id, url, ignore_tls)
            if not res.ok and res.error in RETRY_ERRORS and not res.partial:   # a page that never settled stays that way
                res = await self._attempt(url_id, url, ignore_tls)
            res.extended = self.extended
            if self.extended:
                res.attempts = 1
                res.reach = "live" if res.ok else "dead"
                if res.ok:
                    res.reason = "ok"
                elif res.reason == "ok":
                    res.reason = "other"
        else:
            res = await self._ladder(url_id, url, ignore_tls)
        res.ms = int((self._clock() - started) * 1000)
        return res

    @staticmethod
    def _useful(res: ShotResult) -> bool:
        """Did the visit show something? A blank white picture (almost no text and a tiny PNG) counts as a miss."""
        return res.ok and not (res.text_chars == 0 and res.bytes < BLANK_BYTES)

    async def _ladder(self, url_id: str, url: str, ignore_tls: bool) -> ShotResult:
        """Open the site as patiently as the lab can: normal visit, then the other device profile with longer waits,
        then the other scheme and the www toggle. The first useful result wins; if none is, the best attempt is kept
        and marked `dead` with the reason. A site that is really down stays down: this only removes OUR mistakes
        (a short timeout, a cloaking-prone profile, the wrong scheme)."""
        best: Optional[ShotResult] = None
        n = 0
        t0 = self._clock()
        for step_url, mobile, slow in ladder_steps(url, self.mobile):
            if n and self._clock() - t0 > LADDER_BUDGET_S:
                break                                          # out of time: keep the best result so far
            n += 1
            res = await self._attempt(url_id, step_url, ignore_tls, mobile=mobile, slow=slow)
            if not res.ok and res.error in RETRY_ERRORS and not res.partial:
                res = await self._attempt(url_id, step_url, ignore_tls, mobile=mobile, slow=slow)
            res.attempts = n
            if self._useful(res):
                res.extended, res.reach, res.reason = True, ("live" if n == 1 else "live_retry"), "ok"
                return res
            if best is None or (res.ok and not best.ok) or (res.ok == best.ok and res.bytes > best.bytes):
                best = res
            if res.reason == "nxdomain":
                break                                          # the name does not exist: other schemes cannot help
        assert best is not None
        best.extended, best.attempts = True, n
        best.reach = "dead"
        if best.reason == "ok":
            best.reason = "other"
        if best.ok:                                            # a blank page is still a picture of what a visitor gets
            best.reason = "other"
        return best

    async def _attempt(self, url_id: str, url: str, ignore_tls: bool, mobile: Optional[bool] = None,
                       slow: bool = False) -> ShotResult:
        started = self._clock()
        res = ShotResult(url_id=url_id)
        ok, why = request_allowed(url, self.allow_loopback)
        if not ok or not url.lower().startswith(("http://", "https://")):
            res.error = "blocked_start"
            res.reason = "blocked"
            return res
        # one page deadline per page we will visit (the landing page plus the followed ones), longer when patient
        deadline = self.page_deadline
        if self.extended:
            deadline = deadline * (1 + self.max_pages) * (1.8 if slow else 1.0) + (self.extra_wait_ms / 1000.0) * (1 + self.max_pages)
        try:
            await asyncio.wait_for(self._shoot(res, url, ignore_tls, self.mobile if mobile is None else mobile, slow),
                                   deadline)
        except asyncio.TimeoutError:
            res.ok, res.error = False, "page_deadline"
            res.reason = "timeout"
            self._drop_browser_if_stuck()
        except Exception as exc:                                      # noqa: BLE001
            res.ok = False
            res.error = "browser_crash" if "closed" in str(exc).lower() or "crash" in str(exc).lower() \
                else "internal_error"
            # the reason goes to stderr (page-influenced text: cleaned and capped), not into the data
            print(f"shot {url_id_for_log(res)}: {safety.clean_field(f'{type(exc).__name__}: {exc}', max_len=240)}",
                  file=sys.stderr, flush=True)
        res.ms = int((self._clock() - started) * 1000)
        return res

    def _drop_browser_if_stuck(self) -> None:
        # a page that blew its deadline may have wedged the browser: the next page gets a new one
        self._browser = None

    async def _early_capture(self, context: Any, page: Any) -> Optional[bytes]:
        """For pages that wipe themselves: one or two quick pictures right after the load event, with Chromium's
        own capture (no waiting for the page to be still). Returns the biggest one, or None."""
        best: Optional[bytes] = None
        for delay in EARLY_SHOTS_AT_S:                                # e.g. 0.3 s and 0.8 s after the load event
            await asyncio.sleep(delay)
            data = await self._raw_screenshot(context, page)
            if data is not None and (best is None or len(data) > len(best)):
                best = data
            if best is not None and len(best) >= BLANK_BYTES:
                break                                                 # something real: no need for the second try
        return best

    async def _record_page_facts(self, res: ShotResult, page: Any) -> None:
        """Where the browser really is, its title and how much text it shows. Every value comes from the page,
        so it is cleaned and capped; a restless or frozen page just leaves the defaults. Never raises."""
        try:
            res.final_url = safety.defang_url(safety.clean_field(page.url, max_len=1024))
        except Exception:                                             # noqa: BLE001
            pass
        try:
            title = await asyncio.wait_for(page.title(), 3)
            res.title = safety.clean_field(" ".join(str(title).split()), max_len=safety.MAX_TITLE)
        except Exception:                                             # noqa: BLE001
            pass
        try:
            n = await asyncio.wait_for(
                page.evaluate("() => (document.body ? document.body.innerText.trim().length : 0)"), 3)
            if isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= 10**9:
                res.text_chars = n
        except Exception:                                             # noqa: BLE001
            pass
        if self.extended:        # the start of the text: tells a live page from "removed / suspended / not found"; python re-collapses the blanks
            try:
                head = await asyncio.wait_for(
                    page.evaluate("() => (document.body ? document.body.innerText.trim().slice(0, 900) : '')"), 3)
                if isinstance(head, str):
                    res.text_head = safety.defang_embedded_urls(safety.clean_field(" ".join(head.split()), max_len=MAX_TEXT_HEAD))
            except Exception:                                         # noqa: BLE001
                pass

    async def _raw_screenshot(self, context: Any, page: Any) -> Optional[bytes]:
        """Chromium's own capture of the visible area (Chromium-only DevTools call). None on any failure."""
        try:
            cdp = await context.new_cdp_session(page)
            out = await asyncio.wait_for(
                cdp.send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}),
                SHOT_TIMEOUT_MS / 1000)
            return base64.b64decode(out["data"], validate=True)
        except Exception:                                             # noqa: BLE001
            return None

    # -- whole-site helpers -------------------------------------------------------------------
    async def _scroll_through(self, page: Any) -> None:
        """Scroll the page top to bottom (and back) so lazy-loaded images and sections are really there. Never raises."""
        script = ("""async (cap) => {
            const total = () => Math.max(document.body ? document.body.scrollHeight : 0, document.documentElement.scrollHeight);
            const step = Math.max(400, window.innerHeight || 800);
            for (let y = 0; y < Math.min(total(), cap); y += step) {
                window.scrollTo(0, y);
                await new Promise(r => setTimeout(r, 120));
            }
            window.scrollTo(0, 0);
            return total();
        }""")
        try:
            await asyncio.wait_for(page.evaluate(script, FULL_PAGE_MAX_HEIGHT), 10)
            await asyncio.sleep(0.3)
        except Exception:                                             # noqa: BLE001
            pass

    async def _page_png(self, res: ShotResult, context: Any, page: Any) -> Optional[bytes]:
        """The picture of the page now showing: the visible area, or (full_page) the whole page from top to bottom.
        A page taller than FULL_PAGE_MAX_HEIGHT, or a PNG over the 4 MB limit, is cut at the largest height that fits
        and `res.truncated` is set. Falls back to Chromium's own capture of the visible area on any failure."""
        if not self.full_page:
            try:
                return await page.screenshot(type="png", full_page=False, timeout=SHOT_TIMEOUT_MS,
                                             animations="disabled")
            except Exception:                                         # noqa: BLE001
                res.partial = True
                return await self._raw_screenshot(context, page)
        await self._scroll_through(page)
        try:
            vw = int(await asyncio.wait_for(page.evaluate("() => document.documentElement.clientWidth"), 3))
            total = int(await asyncio.wait_for(page.evaluate(
                "() => Math.max(document.body ? document.body.scrollHeight : 0, document.documentElement.scrollHeight)"), 3))
        except Exception:                                             # noqa: BLE001
            vw, total = (MOBILE_VIEWPORT if self.mobile else VIEWPORT)["width"], 0
        for cap in FULL_PAGE_HEIGHT_STEPS:
            height = min(max(total, 1), cap) if total else cap
            data = None
            for use_clip in (True, False):                            # older Playwright builds may refuse full_page + clip
                if not use_clip and total > height:
                    break                                             # cannot cut a tall page without a clip
                kw: Dict[str, Any] = {"clip": {"x": 0, "y": 0, "width": max(vw, 1), "height": height}} if use_clip else {}
                try:
                    data = await page.screenshot(type="png", full_page=True, timeout=SHOT_TIMEOUT_MS,
                                                 animations="disabled", **kw)
                    break
                except Exception:                                     # noqa: BLE001
                    data = None
            if data is None:
                break
            if len(data) <= pngcheck.MAX_BYTES:
                if total > height:
                    res.truncated = True
                return data
            res.truncated = True                                      # too big for the PNG limit: try a shorter cut
        res.partial = True
        return await self._raw_screenshot(context, page)

    async def _follow_links(self, res: ShotResult, context: Any, page: Any, guard: Any, slow: bool) -> None:
        """After the landing page: visit up to `max_pages` links of the SAME site and take a full picture of each
        (<url_id>_p<n>.png). Same context, same guard proxy; a page that fails is skipped, never fatal."""
        try:
            hrefs = await asyncio.wait_for(page.evaluate(
                "() => Array.from(document.querySelectorAll('a[href]')).slice(0, 300).map(a => a.href)"), 5)
        except Exception:                                             # noqa: BLE001
            return
        base = page.url if isinstance(page.url, str) and page.url.startswith("http") else None
        if not base:
            return
        timeout = SLOW_GOTO_TIMEOUT_MS if slow else GOTO_TIMEOUT_MS
        for n, link in enumerate(pick_links(base, hrefs, self.max_pages, self.allow_loopback), 1):
            sub = ShotResult(url_id=res.url_id)
            try:
                try:
                    await page.goto(link, wait_until="load", timeout=timeout)
                except Exception as exc:                              # noqa: BLE001
                    if "timeout" not in str(exc).lower():
                        continue
                    sub.partial = True
                try:
                    await page.wait_for_load_state("networkidle", timeout=SETTLE_MS)
                except Exception:                                     # noqa: BLE001
                    pass
                await self._record_page_facts(sub, page)
                data = await self._page_png(sub, context, page)
                if data is None:
                    continue
                ok, _err, w, h = write_png(os.path.join(self.out_dir, f"{res.url_id}_p{n}.png"), data)
                if ok:
                    res.pages.append({"n": n, "url": sub.final_url, "title": sub.title, "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest(), "width": w, "height": h,
                                      "text_chars": sub.text_chars, "truncated": sub.truncated})
            except Exception:                                         # noqa: BLE001
                continue

    async def _shoot(self, res: ShotResult, url: str, ignore_tls: bool = False, mobile: Optional[bool] = None,
                     slow: bool = False) -> None:
        mobile = self.mobile if mobile is None else mobile
        browser = await self._browser_for_use()
        guard = GuardProxy(allow_loopback=self.allow_loopback, resolver=self.resolver)   # one per page: its counters are this page's
        port = await guard.start()
        try:
            profile = (dict(viewport=MOBILE_VIEWPORT, user_agent=MOBILE_USER_AGENT, is_mobile=True, has_touch=True,
                            device_scale_factor=1, locale="en-PH", timezone_id="Asia/Manila")
                       if mobile else
                       dict(viewport=VIEWPORT, user_agent=USER_AGENT, locale="en-US", timezone_id="UTC"))
            context = await browser.new_context(
                java_script_enabled=True, accept_downloads=False, service_workers="block", permissions=[],
                ignore_https_errors=bool(ignore_tls), bypass_csp=False,
                proxy={"server": f"http://127.0.0.1:{port}"}, **profile,
            )
        except BaseException:
            await guard.close()
            raise
        try:
            # No request filter inside the browser: Chromium keeps its own network stack (redirects, bot-protected
            # sites and WebSockets all behave like a real visit) and EVERY connection it makes goes through
            # the guard proxy, which refuses non-public addresses and odd ports at connect time.
            page = await context.new_page()
            main_page = page

            def on_new_page(p: Any) -> None:                          # popups: close at once
                if p is not main_page:
                    asyncio.ensure_future(p.close())
            context.on("page", on_new_page)
            page.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))
            page.on("download", lambda d: asyncio.ensure_future(d.cancel()))

            try:
                await page.goto(url, wait_until="load", timeout=SLOW_GOTO_TIMEOUT_MS if slow else GOTO_TIMEOUT_MS)
            except Exception as exc:                                  # noqa: BLE001
                if "timeout" in str(exc).lower():
                    res.partial = True                                # shoot whatever is on screen
                    res.reason = "timeout"
                else:
                    res.error = "navigation_error"
                    # why (page-influenced text: cleaned and capped), to stderr, not into the data
                    why = f"{type(exc).__name__}: {exc}".split("Call log:")[0]
                    res.reason = classify_nav_error(why, bool(guard.refused))
                    refusal = f" [guard proxy refused: {guard.last_refusal}]" if guard.refused else ""
                    print(f"shot {url_id_for_log(res)}: navigation: {safety.clean_field(why + refusal, max_len=200)}",
                          file=sys.stderr, flush=True)
                    return
            try:
                await page.wait_for_load_state("networkidle", timeout=SETTLE_MS)
            except Exception:                                         # noqa: BLE001
                pass
            # picture 1 (early): many casino / phishing pages show content and then wipe themselves a few
            # seconds later when they decide the visitor is a robot. Take a quick picture at the first sign of text.
            early = await self._early_capture(context, page) if self.rescue else None
            if self.extra_wait_ms:                                    # script-heavy pages render late
                try:
                    await page.wait_for_timeout(self.extra_wait_ms)
                except Exception:                                     # noqa: BLE001
                    pass
            await self._record_page_facts(res, page)
            # picture 2 (settled): the normal one
            data: Optional[bytes]
            # (a restless page can make the careful screenshot give up: _page_png then asks Chromium directly for
            # what is on screen, which needs no stillness, and marks the result partial)
            data = await self._page_png(res, context, page)
            # keep the early picture when the settled one is blank (or missing) and the early one has content
            if early is not None and (data is None or (len(data) < BLANK_BYTES and len(early) > max(3 * len(data), BLANK_BYTES))):
                data, res.rescued = early, True
            if data is None:
                res.error = "screenshot_error"
                return
            ok, err, w, h = write_png(os.path.join(self.out_dir, f"{res.url_id}.png"), data)
            res.ok, res.error, res.width, res.height = ok, err, w, h
            if ok:
                res.bytes, res.sha256 = len(data), hashlib.sha256(data).hexdigest()
                if self.max_pages:                                    # whole-site mode: the same site's other pages
                    await self._follow_links(res, context, page, guard, slow)
        finally:
            res.blocked_requests = guard.refused                      # recorded on EVERY exit path
            try:
                await asyncio.wait_for(context.close(), 10)
            except Exception:                                         # noqa: BLE001
                pass
            await guard.close()

    # -- many pages -------------------------------------------------------------------------
    async def run(self, items: Sequence[Tuple[Any, ...]],
                  progress: Optional[Callable[[int, int], None]] = None) -> List[ShotResult]:
        """items: (url_id, url) or (url_id, url, ignore_tls). One result per item, in input order."""
        queue: "asyncio.Queue[Tuple[int, Tuple[Any, ...]]]" = asyncio.Queue()
        for i, item in enumerate(items):
            queue.put_nowait((i, tuple(item)))
        results: List[Optional[ShotResult]] = [None] * len(items)
        done = [0]

        async def worker() -> None:
            while True:
                try:
                    i, item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                uid, url = item[0], item[1]
                results[i] = await self.shoot(uid, url, bool(item[2]) if len(item) > 2 else False)
                done[0] += 1
                if progress:
                    progress(done[0], len(items))

        await asyncio.gather(*[worker() for _ in range(min(self.concurrency, max(1, len(items))))])
        return [r for r in results if r is not None]
