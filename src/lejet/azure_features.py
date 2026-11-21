"""azure_features.py -- MANY links, FEATURES ONLY, on the Azure VM (for the dataset; no browser, no screenshots).

    python -I -m lejet.azure_features <batch_id>

Why this exists: `lejet.azure_run` is the live tool's mode (one link per call, a Chromium visit, a whole-page screenshot, up to
4 more pages of the site and a retry ladder). That is 30 to 90 seconds per link, and the dataset needs none of the pictures:
the fetch itself (DNS, registry lookup, page download, HTML parsing) takes 2 to 5 seconds. This module does only that, for every
link of the pushed file at once, with the SAME code as the VirtualBox lab's `lejet.cli run` + `export`:

    read_input (ids re-checked) -> Pipeline.run (a few links in parallel) -> database -> export_jsonl

Where the files are (same layout as azure_run; the program runs as the unprivileged user `fetcher`):
    /var/lib/lejet/in/urls.txt            url_id<TAB>url, one line per link (installed by the root helper lejet-azure-features)
    /var/lib/lejet/out/export/<batch>.*   .jsonl.gz, .jsonl.gz.sha256, .manifest.json   (the PC pulls them)

Safety is the same as for the live mode: the helper checks the isolation token, the unit has a memory cap and no access to the
website's data, the firewall table of user `fetcher` refuses private ranges, the metadata service and the website; this program
checks the token again before it starts. No browser is started, and nothing the page sends is ever executed: the HTML is only parsed.
"""
from __future__ import annotations

import asyncio
import os
import platform
import sys
from typing import List, Optional

from lejet.azure_run import OUT, TOKEN_MAX_AGE, settings_for_azure
from lejet.cli import DEFAULT_TOKEN, preflight_ok, read_input, tool_versions
from lejet.live_worker import ID_RE

INPUT = "/var/lib/lejet/in/urls.txt"
MAX_LINKS = 400                      # the helper enforces the same cap; one call is one batch of the dataset
CONCURRENCY = int(os.environ.get("LEJET_FEATURES_CONCURRENCY", "4"))
RATE = float(os.environ.get("LEJET_FEATURES_RATE", "2.0"))     # new links started per second, at most


def settings_for_features():
    s = settings_for_azure()
    s.concurrency = max(1, min(CONCURRENCY, 8))
    s.rate = RATE
    s.jitter = 0.0
    s.parse_workers = 3
    return s


async def amain(batch: str, input_path: str = INPUT, out_dir: str = OUT) -> int:
    from lejet.pipeline import Pipeline
    from lejet.storage import Storage, export_jsonl

    ok, why = preflight_ok(DEFAULT_TOKEN, TOKEN_MAX_AGE)
    if not ok:
        print(f"REFUSING TO START: {why}", file=sys.stderr)
        return 3
    items, sha, problems = read_input(input_path, verify_ids=True)
    if problems:
        for p in problems[:5]:
            print(f"input problem: {p}", file=sys.stderr)
        return 2
    if not items or len(items) > MAX_LINKS:
        print(f"the input has {len(items)} links; this mode takes 1 to {MAX_LINKS}", file=sys.stderr)
        return 2
    settings = settings_for_features()
    settings.rdap_cache_path = os.path.join(out_dir, "rdap_cache.json")
    settings.ct_cache_path = os.path.join(out_dir, "ct_cache.json")
    os.makedirs(os.path.join(out_dir, "export"), exist_ok=True)
    db = os.path.join(out_dir, f"{batch}.sqlite")
    if os.path.exists(db):
        print("batch already exists on this VM; use a new batch id", file=sys.stderr)
        return 2
    extras = {"git_commit": os.environ.get("LEJET_GIT_COMMIT", ""), "vbox_version": "azure",
              "vpn_provider": "", "vpn_exit_country": "", "kernel": platform.release(), "preflight_pass": True}
    if os.environ.get("LEJET_VPN_ACTIVE") == "1":
        extras.update({"vpn_provider": os.environ.get("LEJET_VPN_PROVIDER", ""),
                       "vpn_exit_country": os.environ.get("LEJET_VPN_EXIT_COUNTRY", "")})

    store = Storage(db)
    try:
        store.start_run(batch, sha, len(items), tool_versions(settings), "azure-features")

        def progress(done: int, total: int) -> None:
            if done == total or done % 10 == 0:
                print(f"progress {done}/{total}", flush=True)

        print(f"== 4 fetching {len(items)} links (features only, no screenshots)", flush=True)
        async with Pipeline(settings) as pipeline:
            counts = await pipeline.run(items, batch, store, progress)
            for client, path in ((getattr(pipeline, "rdap", None), settings.rdap_cache_path),
                                 (getattr(pipeline, "ct", None), settings.ct_cache_path)):
                if client is not None and path:
                    try:
                        client.save_cache(path)
                    except OSError:
                        pass
        store.finish_run(batch)
    finally:
        store.close()
    info = export_jsonl(db, os.path.join(out_dir, "export"), batch, extras)
    print("== result", {"ok": True, "n_rows": int(info.get("n_rows", 0)), "status_counts": counts}, flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or not ID_RE.match(args[0]):
        print("usage: python -m lejet.azure_features <batch_id>   (letters, digits, - and _ only)", file=sys.stderr)
        return 2
    if os.geteuid() == 0:                              # type: ignore[attr-defined]
        print("REFUSING TO START: do not run page-opening code as root", file=sys.stderr)
        return 3
    return asyncio.run(amain(args[0]))


if __name__ == "__main__":
    sys.exit(main())
