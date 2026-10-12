"""
external_features.py -- the three RDAP-derived features (plan sections 13.4b, 14.0.4).

Pure functions over an already-fetched RDAP response. RDAP JSON is THIRD-PARTY, UNTRUSTED
data: it is only ever parsed with `json.loads` (never eval), size-capped by the fetcher, and
reduced to a handful of numbers here; nothing from it is stored as text.

Directions (same as the public dataset, plan 14.0.4):
  whois_registered_domain: 0 = a registration record exists, 1 = NO such registration
                           (the abnormal case), -1 = the lookup failed.
  domain_age:              days since the registration event, -1 if unknown/negative.
  domain_registration_length: days UNTIL EXPIRY (expiration - now). The public set's median
                           is 242 days while its median domain_age is 3,993 days, so it cannot
                           be "expiration - registration" (that would be >= the age). The
                           same number is stored as `days_to_expiry`.
Negative durations are never stored; they become -1 (as the cleaning of the public set did).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RdapRecord:
    """What we keep from one RDAP answer. Timestamps are UNIX seconds (UTC) or None."""
    ok: bool                       # the lookup completed (HTTP 200 or 404)
    registered: int                # 0 / 1 / -1 (see module docstring)
    reg_ts: Optional[int] = None
    exp_ts: Optional[int] = None


FAILED = RdapRecord(ok=False, registered=-1)


def _parse_dt(value: Any) -> Optional[int]:
    if not isinstance(value, str) or not value or len(value) > 64:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return int(dt.timestamp())
    except (OverflowError, OSError, ValueError):
        return None


def record_from_response(status: int, body: bytes) -> RdapRecord:
    """Turn an HTTP status + (capped) body into an RdapRecord."""
    if status == 404:
        return RdapRecord(ok=True, registered=1)
    if status != 200:
        return FAILED
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except (ValueError, RecursionError):
        return FAILED
    if not isinstance(obj, dict):
        return FAILED
    reg_ts: Optional[int] = None
    exp_ts: Optional[int] = None
    events = obj.get("events")
    if isinstance(events, list):
        for ev in events[:200]:
            if not isinstance(ev, dict):
                continue
            action = str(ev.get("eventAction", "")).strip().lower()
            ts = _parse_dt(ev.get("eventDate"))
            if ts is None:
                continue
            if action == "registration" and reg_ts is None:
                reg_ts = ts
            elif action == "expiration" and exp_ts is None:
                exp_ts = ts
    return RdapRecord(ok=True, registered=0, reg_ts=reg_ts, exp_ts=exp_ts)


def features_from_record(rec: RdapRecord, now_ts: Optional[int] = None) -> Dict[str, int]:
    """The three features + days_to_expiry + rdap_ok for one record."""
    now = int(datetime.now(timezone.utc).timestamp()) if now_ts is None else now_ts
    out = {"whois_registered_domain": -1, "domain_registration_length": -1,
           "domain_age": -1, "days_to_expiry": -1, "rdap_ok": 1 if rec.ok else 0}
    if not rec.ok:
        return out
    out["whois_registered_domain"] = rec.registered
    if rec.reg_ts is not None and now >= rec.reg_ts:
        out["domain_age"] = (now - rec.reg_ts) // 86400
    if rec.exp_ts is not None and rec.exp_ts >= now:
        days = (rec.exp_ts - now) // 86400
        out["domain_registration_length"] = days
        out["days_to_expiry"] = days
    return out


def not_looked_up() -> Dict[str, int]:
    """Features for a host whose RDAP lookup was skipped or impossible."""
    return {"whois_registered_domain": -1, "domain_registration_length": -1,
            "domain_age": -1, "days_to_expiry": -1, "rdap_ok": 0}
