"""
page_rank.py -- the rules for reading an Open PageRank answer, shared by the host-side look-up
(scripts/lookup_page_rank.py, scripts/fill_page_rank.py) and the collection website (webapp/rank.py), so they cannot drift.

This module never touches the network and never sees the API key: it only turns the service's JSON answer into
{domain: rank}. Standard library only.

Rules (the training column's, plan 14.0.2):
  * a rank is an integer 0..10;
  * a domain the service answers for but has no rank for -> 0 ("unrated" is a real value in the training data);
  * a different scale (e.g. 0-100) would shift the trained feature -> the domain is left out (a failure, -1 later);
  * a domain missing from the answer is simply absent: the caller must treat that as "not measured" (-1), never as 0.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Sequence

DEFAULT_URL = "https://openpagerank.com/api/v1.0/getPageRank"
BATCH = 100
MAX_RESPONSE = 1024 * 1024
DOMAIN_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63}$")
KEY_RE = re.compile(r"[A-Za-z0-9_\-]{8,128}")


def refang_domain(value: str) -> str:
    return (value or "").replace("[.]", ".").strip().lower().rstrip(".")


def parse_answer(body: bytes, domains: Sequence[str]) -> Dict[str, int]:
    """The service's answer -> {domain: rank 0..10} for the domains that were answered. Raises RuntimeError if the answer
    is unusable (too large, not JSON, wrong shape); the message never contains any of the answer."""
    if len(body) > MAX_RESPONSE:
        raise RuntimeError("response too large")
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except (ValueError, RecursionError):
        raise RuntimeError("answer is not JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("response"), list):
        raise RuntimeError("unexpected answer shape")
    out: Dict[str, int] = {}
    for item in data["response"]:
        if not isinstance(item, dict):
            continue
        dom = refang_domain(str(item.get("domain", "")))
        if dom not in domains:
            continue
        status = item.get("status_code")
        rank = item.get("page_rank_integer")
        if status == 200 and isinstance(rank, int) and not isinstance(rank, bool):
            if 0 <= rank <= 10:
                out[dom] = rank
            # a different scale (e.g. 0-100) would shift the trained feature: treat as a failure
        elif status in (200, 404) and rank is None:
            out[dom] = 0                                            # answered, but no rank: unrated = 0
        elif status == 404 and str(item.get("error", "")).strip().lower() == "domain not found":
            # what the real service sends for a domain it does not know (seen 2026-10-15): status_code 404,
            # error "Domain not found", page_rank_integer 0. Unrated = 0, like the training column; NOT a failure.
            # (Before this branch such domains became -1, and since nearly all of them were phishing sites the
            # missing value itself leaked the label.)
            out[dom] = 0
    return out
