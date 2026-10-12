"""
cli.py -- the command line of the inspector (plan section 8.1: run / export / preflight).

    python -I -m lejet.cli run    --input urls.txt --db out.sqlite --batch-id b0001
    python -I -m lejet.cli export --db out.sqlite --out-dir out --batch-id b0001
    python -I -m lejet.cli check-token

`run` refuses to start unless the preflight pass token written by infra/analysis/preflight.sh
is fresh (plan 16, Level 2). `--allow-loopback` is TEST MODE: fixtures on 127.0.0.1, no token
needed, never used for real runs.

Input file: one `url_id<TAB>raw_url` per line (url_id = sha256(raw_url)[:16], computed on the
host, plan 13.2). Blank lines and lines starting with # are ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lejet import schema, urlnorm
from lejet.settings import Settings

DEFAULT_TOKEN = "/run/lejet-preflight-ok"
DEFAULT_TOKEN_MAX_AGE = 6 * 3600


# ---------------------------------------------------------------------------
# preflight token
# ---------------------------------------------------------------------------
def preflight_ok(path: str, max_age: int = DEFAULT_TOKEN_MAX_AGE, now: Optional[float] = None) -> Tuple[bool, str]:
    """(True, '') if the token file holds a UNIX time that is recent (and not in the future)."""
    try:
        with open(path, "r", encoding="ascii") as fh:
            stamp = int(fh.read(32).strip())
    except FileNotFoundError:
        return False, f"preflight token {path} not found (run infra/analysis/preflight.sh)"
    except (OSError, ValueError):
        return False, f"preflight token {path} is unreadable or malformed"
    age = (time.time() if now is None else now) - stamp
    if age < -60:
        return False, "preflight token is dated in the future (clock problem?)"
    if age > max_age:
        return False, f"preflight token is {int(age)} s old (limit {max_age} s): re-run preflight.sh"
    return True, ""


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
def read_input(path: str, verify_ids: bool = True) -> Tuple[List[Tuple[str, str]], str, List[str]]:
    """Returns (items, sha256 of the file, problems). Duplicated ids keep the first line."""
    with open(path, "rb") as fh:
        blob = fh.read()
    sha = hashlib.sha256(blob).hexdigest()
    items: List[Tuple[str, str]] = []
    seen = set()
    problems: List[str] = []
    for n, line in enumerate(blob.decode("utf-8", "replace").splitlines(), 1):
        line = line.strip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" not in line:
            problems.append(f"line {n}: expected url_id<TAB>url")
            continue
        url_id, raw = line.split("\t", 1)
        url_id, raw = url_id.strip(), raw.strip()
        if not schema.is_valid_url_id(url_id):
            problems.append(f"line {n}: bad url_id {url_id[:20]!r}")
            continue
        if verify_ids and urlnorm.compute_url_id(raw) != url_id:
            problems.append(f"line {n}: url_id does not match sha256(url)[:16]")
            continue
        if url_id in seen:
            continue
        seen.add(url_id)
        items.append((url_id, raw))
    return items, sha, problems


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tool_versions(settings: Settings) -> Dict[str, str]:
    from importlib import metadata

    out: Dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for dist in ("httpx", "dnspython", "lxml", "tldextract", "nostril", "pandas", "numpy"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = "not installed"
    from lejet.pipeline import config_hash

    out["config_hash"] = config_hash(settings)
    return out


def _settings_from_args(a: argparse.Namespace) -> Settings:
    s = Settings()
    s.allow_loopback = bool(a.allow_loopback)
    s.dns_server = a.dns_server
    s.rdap_bootstrap_path = a.rdap_bootstrap
    s.rdap_cache_path = a.rdap_cache
    s.ct_enabled = bool(a.ct)
    s.ct_probe = bool(a.ct)
    s.ct_cache_path = a.ct_cache
    s.popular_domains_path = a.popular
    s.parse_workers = a.workers
    s.concurrency = a.concurrency
    s.rate = a.rate
    s.jitter = a.jitter
    s.seed = a.seed
    return s


async def _run_async(a: argparse.Namespace, items: List[Tuple[str, str]], sha: str) -> Dict[str, int]:
    from lejet.pipeline import Pipeline
    from lejet.storage import Storage

    settings = _settings_from_args(a)
    store = Storage(a.db)
    try:
        store.start_run(a.batch_id, sha, len(items), tool_versions(settings), a.snapshot_name)
        last = [0.0]

        def progress(done: int, total: int) -> None:
            now = time.monotonic()
            if done == total or now - last[0] > 5:
                last[0] = now
                print(f"progress {done}/{total}", file=sys.stderr, flush=True)

        async with Pipeline(settings) as pipeline:
            counts = await pipeline.run(items, a.batch_id, store, progress)
        store.finish_run(a.batch_id)
        return counts
    finally:
        store.close()


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_run(a: argparse.Namespace) -> int:
    if not a.allow_loopback and not a.skip_preflight:
        ok, why = preflight_ok(a.preflight_token, a.preflight_max_age)
        if not ok:
            print(f"REFUSING TO START: {why}", file=sys.stderr)
            return 3
    items, sha, problems = read_input(a.input, verify_ids=not a.no_verify_ids)
    if problems:
        for p in problems[:5]:
            print(f"input problem: {p}", file=sys.stderr)
        print(f"{len(problems)} bad input line(s); fix the push file (or use --no-verify-ids)",
              file=sys.stderr)
        return 2
    if not items:
        print("no URLs in the input", file=sys.stderr)
        return 2
    try:
        counts = asyncio.run(_run_async(a, items, sha))
    except KeyboardInterrupt:
        print("interrupted: results so far are saved; run again to resume", file=sys.stderr)
        return 130
    print(json.dumps({"batch_id": a.batch_id, "n_input": len(items), "status_counts": counts},
                     sort_keys=True))
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    from lejet.storage import export_jsonl

    ok, _ = preflight_ok(a.preflight_token, a.preflight_max_age)
    extras = {
        "git_commit": a.git_commit or os.environ.get("LEJET_GIT_COMMIT", ""),
        "vbox_version": a.vbox_version or os.environ.get("LEJET_VBOX_VERSION", ""),
        "kernel": platform.release(),
        "vpn_provider": a.vpn_provider or os.environ.get("LEJET_VPN_PROVIDER", ""),
        "vpn_exit_country": a.vpn_exit_country or os.environ.get("LEJET_VPN_EXIT_COUNTRY", ""),
        "preflight_pass": bool(ok),
    }
    info = export_jsonl(a.db, a.out_dir, a.batch_id, extras)
    print(json.dumps(info, sort_keys=True))
    return 0


async def _shots_async(a: argparse.Namespace, items: List[Tuple[Any, ...]], png_dir: str):
    from lejet.stages.shot_stage import ShotRunner

    def progress(done: int, total: int) -> None:
        if done in (1, total) or done % 5 == 0:
            print(f"screenshot progress {done}/{total}", file=sys.stderr, flush=True)

    async with ShotRunner(png_dir, concurrency=a.concurrency, allow_loopback=a.allow_loopback,
                          mobile=a.mobile, extra_wait_ms=a.extra_wait_ms) as runner:
        results = await runner.run(items, progress)
        return results, runner.restarts


def cmd_screenshots(a: argparse.Namespace) -> int:
    """Real screenshots of the pages of a finished batch (needs Playwright + Chromium; VM only)."""
    from lejet.stages import shot_stage
    from lejet.storage import Storage, sha256_file

    if not a.allow_loopback:
        ok, why = preflight_ok(a.preflight_token, a.preflight_max_age)
        if not ok:
            print(f"REFUSING TO START: {why}", file=sys.stderr)
            return 3
    store = Storage(a.db)
    try:
        all_rows = list(store.rows(a.batch_id))
    finally:
        store.close()
    # EVERY URL that is safe to open gets an attempt (not only the ones the plain fetch loaded): see
    # shot_stage.capture_target. Rows refused by the safety gate or without a DNS name are left out.
    items: List[Tuple[Any, ...]] = []
    skipped = 0
    for r in all_rows:
        target = shot_stage.capture_target(r, allow_loopback=a.allow_loopback)
        if target is None:
            skipped += 1
        else:
            items.append((r["url_id"], target[0], target[1]))
    if a.limit:
        items = items[:a.limit]
    if not items:
        print(json.dumps({"batch_id": a.batch_id, "n_candidates": 0, "n_ok": 0, "skipped": skipped}))
        return 0
    png_dir = os.path.join(a.out_dir, f"shots-{a.batch_id}")
    os.makedirs(png_dir, exist_ok=True)
    try:
        results, restarts = asyncio.run(_shots_async(a, items, png_dir))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    tar_path = os.path.join(a.out_dir, f"{a.batch_id}.shots.tar")
    n_ok = shot_stage.pack_tar(png_dir, results, tar_path)
    digest = sha256_file(tar_path)
    with open(tar_path + ".sha256", "w", encoding="ascii") as fh:
        fh.write(f"{digest}  {os.path.basename(tar_path)}\n")
    errors: Dict[str, int] = {}
    for r in results:
        if not r.ok:
            errors[r.error] = errors.get(r.error, 0) + 1
    print(json.dumps({"batch_id": a.batch_id, "n_candidates": len(items), "n_ok": n_ok, "skipped": skipped,
                      "errors": dict(sorted(errors.items())), "browser_restarts": restarts,
                      "tar": tar_path, "sha256": digest}, sort_keys=True))
    return 0


def cmd_check_token(a: argparse.Namespace) -> int:
    ok, why = preflight_ok(a.preflight_token, a.preflight_max_age)
    print("preflight token OK" if ok else why)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lejet", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def token_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--preflight-token", default=DEFAULT_TOKEN)
        sp.add_argument("--preflight-max-age", type=int, default=DEFAULT_TOKEN_MAX_AGE)

    r = sub.add_parser("run", help="inspect every URL of an input file")
    r.add_argument("--input", required=True)
    r.add_argument("--db", required=True)
    r.add_argument("--batch-id", required=True)
    r.add_argument("--snapshot-name", default="")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--concurrency", type=int, default=20)
    r.add_argument("--workers", type=int, default=3, help="HTML parse workers (0 = inline, tests only)")
    r.add_argument("--rate", type=float, default=2.0, help="max URL starts per second")
    r.add_argument("--jitter", type=float, default=0.5)
    r.add_argument("--dns-server", default="10.77.0.1")
    r.add_argument("--rdap-bootstrap", default=None, help="local copy of IANA dns.json")
    r.add_argument("--rdap-cache", default=None, help="JSON cache file (loaded and saved)")
    r.add_argument("--ct", action="store_true",
                   help="where RDAP gives nothing (e.g. .ph), look up certificate age at crt.sh")
    r.add_argument("--ct-cache", default=None, help="JSON cache file for --ct (loaded and saved)")
    r.add_argument("--popular", default=None, help="frozen list of popular domains (flag only)")
    r.add_argument("--allow-loopback", action="store_true", help="TEST MODE: fixtures on 127.0.0.1")
    r.add_argument("--skip-preflight", action="store_true", help="development only")
    r.add_argument("--no-verify-ids", action="store_true")
    token_args(r)
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("export", help="write <batch>.jsonl.gz + manifest + sha256")
    e.add_argument("--db", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--batch-id", required=True)
    e.add_argument("--git-commit", default="")
    e.add_argument("--vbox-version", default="")
    e.add_argument("--vpn-provider", default="")
    e.add_argument("--vpn-exit-country", default="")
    token_args(e)
    e.set_defaults(fn=cmd_export)

    s = sub.add_parser("screenshots", help="real screenshots (headless Chromium) of a finished batch")
    s.add_argument("--db", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--batch-id", required=True)
    s.add_argument("--limit", type=int, default=0, help="at most this many pages (0 = all)")
    s.add_argument("--concurrency", type=int, default=1, help="pages at the same time (browser contexts)")
    s.add_argument("--mobile", action="store_true",
                   help="look like an Android phone (what a smishing victim's device sends); many pages hide from desktops")
    s.add_argument("--extra-wait-ms", type=int, default=0, help="extra settle time per page for script-heavy pages")
    s.add_argument("--allow-loopback", action="store_true", help="TEST MODE: fixtures on 127.0.0.1")
    token_args(s)
    s.set_defaults(fn=cmd_screenshots)

    c = sub.add_parser("check-token", help="exit 0 if the preflight token is fresh")
    token_args(c)
    c.set_defaults(fn=cmd_check_token)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
