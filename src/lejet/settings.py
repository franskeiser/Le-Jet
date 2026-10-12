"""
settings.py -- every tunable of a run in one place (plan sections 13.5, 13.9).

Defaults are the plan's v1 values. Tests override only what they need; the CLI exposes the
few that matter operationally. Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# A Pixel-7-class mobile UA: the phishing kits aimed at SMS victims often cloak against
# desktop browsers (plan section 20, item 3).
MOBILE_UA = (
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Mobile Safari/537.36"
)


@dataclass
class Settings:
    # --- safety / test mode ---
    allow_loopback: bool = False          # TEST MODE ONLY (plan 13.2): fixtures on 127.0.0.1

    # --- HTTP (plan 13.5) ---
    user_agent: str = MOBILE_UA
    accept_language: str = "en-PH,en;q=0.8,fil;q=0.6"
    max_hops: int = 8
    connect_timeout: float = 6.0
    read_timeout: float = 10.0
    write_timeout: float = 6.0
    pool_timeout: float = 6.0
    hop_deadline: float = 20.0            # wall clock per hop (a trickling body never hits read_timeout)
    url_deadline: float = 45.0            # wall clock per URL (all hops)

    # --- DNS / RDAP ---
    dns_server: str = "10.77.0.1"
    rdap_bootstrap_path: Optional[str] = None
    rdap_cache_path: Optional[str] = None
    rdap_min_interval: float = 1.0

    # --- certificate-transparency age (crt.sh): only where RDAP gives nothing, e.g. .ph ---
    ct_enabled: bool = False          # off by default so no test can reach the network by accident
    ct_cache_path: Optional[str] = None
    ct_min_interval: float = 2.0
    ct_backoff: float = 3.0
    ct_probe: bool = False            # ask crt.sh one cheap question at the start of a run (CLI --ct sets it)

    # --- parsing ---
    parse_workers: int = 3
    parse_timeout: float = 5.0

    # --- concurrency and politeness (plan 13.9) ---
    concurrency: int = 20
    rate: float = 2.0                     # max URL starts per second
    jitter: float = 0.5                   # extra random delay 0..jitter seconds per start
    seed: int = 1                         # shuffle + jitter seed (reproducible)

    # --- bookkeeping ---
    popular_domains_path: Optional[str] = None   # frozen list for final_is_popular (flag only)
