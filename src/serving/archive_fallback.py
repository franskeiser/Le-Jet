"""Archived copies for links the lab could not open (host side, the PC; never the VM).

When a smishing site is already taken down, the cloud lab reports it as `dead` with a reason. This module asks two public
archives whether they hold an older copy, so the tool can still show something, always labelled as ARCHIVED, never as live:

    1. Wayback Machine  (archive.org/wayback/available)  -> a link to the closest snapshot (no key needed);
    2. urlscan.io search (needs the env var URLSCAN_API_KEY; without it this source is skipped) -> the stored screenshot.

It is OFF unless the environment variable LEJET_ARCHIVE_FALLBACK=1 is set: using it sends the link's address to those two
third-party services, which the person running the tool should decide. The urlscan key is read only from the environment,
never written anywhere. Nothing returned is trusted: only https URLs on the two expected hosts are followed, the image is
size-limited and structurally checked (lejet.pngcheck) before use, text is reduced to printable ASCII.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional

from src.lejet import pngcheck

TIMEOUT_S = 12
MAX_JSON_BYTES = 512 * 1024
SOURCES = ("archived_wayback", "archived_urlscan")
_WAYBACK_HOST = "web.archive.org"
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_STAMP = re.compile(r"^[0-9]{8,14}$")

Fetch = Callable[[str, Dict[str, str], int], bytes]


def enabled() -> bool:
    return os.environ.get("LEJET_ARCHIVE_FALLBACK", "").strip() == "1"


def _plain(text: Any, limit: int = 200) -> str:
    return re.sub(r"[^\x20-\x7e]", "", str(text))[:limit]


def _http_get(url: str, headers: Dict[str, str], limit: int) -> bytes:
    """GET with a size limit. Only https; raises on any problem (callers treat that as 'no archive')."""
    if not url.lower().startswith("https://"):
        raise ValueError("https only")
    req = urllib.request.Request(url, headers={"User-Agent": "Le-Jet-research/1.0", **headers})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:       # noqa: S310 -- scheme and hosts are checked by callers
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise ValueError("response too large")
    return data


def wayback(link: str, fetch: Fetch = _http_get) -> Optional[Dict[str, Any]]:
    """The closest Wayback snapshot of the link, or None."""
    api = "https://archive.org/wayback/available?" + urllib.parse.urlencode({"url": link})
    try:
        body = json.loads(fetch(api, {}, MAX_JSON_BYTES))
        snap = body["archived_snapshots"]["closest"]
        url, stamp = str(snap["url"]), str(snap["timestamp"])
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError):
        return None
    parts = urllib.parse.urlsplit(url)
    if parts.hostname != _WAYBACK_HOST or not snap.get("available") or not _STAMP.match(stamp):
        return None
    return {"source": "archived_wayback", "url": "https://" + _WAYBACK_HOST + parts.path, "taken": stamp, "image": None}


def urlscan(link: str, fetch: Fetch = _http_get, key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The newest urlscan.io scan of this link's domain, with its screenshot, or None. Needs URLSCAN_API_KEY."""
    key = key if key is not None else os.environ.get("URLSCAN_API_KEY", "")
    host = urllib.parse.urlsplit(link).hostname
    if not key or not host:
        return None
    headers = {"API-Key": key}
    api = "https://urlscan.io/api/v1/search/?" + urllib.parse.urlencode({"q": f'page.domain:"{host}"', "size": "1"})
    try:
        hit = json.loads(fetch(api, headers, MAX_JSON_BYTES))["results"][0]
        scan_id = str(hit["_id"])
        taken = _plain(hit.get("task", {}).get("time", ""), 40)
        if not _UUID.match(scan_id):
            return None
        image = fetch(f"https://urlscan.io/screenshots/{scan_id}.png", headers, pngcheck.MAX_BYTES)
    except (OSError, ValueError, KeyError, IndexError, TypeError, urllib.error.URLError):
        return None
    ok, _why, _w, _h = pngcheck.check_png(image)
    if not ok:
        return None
    return {"source": "archived_urlscan", "url": f"https://urlscan.io/result/{scan_id}/", "taken": taken, "image": image}


def find(link: str, fetch: Fetch = _http_get, key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The best archived copy: a stored screenshot (urlscan) first, then a Wayback snapshot link. None if there is none
    (or the fallback is switched off)."""
    if not enabled():
        return None
    return urlscan(link, fetch, key) or wayback(link, fetch)
