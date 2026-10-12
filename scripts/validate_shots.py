#!/usr/bin/env python
"""
validate_shots.py -- the HOST-side gate for the screenshot archive pulled from the analysis VM.

    python -I scripts/validate_shots.py <batch>.shots.tar --clean-dir D:\\LeJetData\\clean ^
           --clean-jsonl D:\\LeJetData\\clean\\<batch>.clean.jsonl

Nothing from the VM is trusted. The archive is NEVER extracted with tarfile's extract(): every member
is read into memory, checked, and only then written by this script, under a name this script chooses.

Checks (any failure = exit 1 and NOTHING is written):
  * the sha256 sidecar matches the archive;
  * the archive is a plain tar with at most 20 001 members; every member is a REGULAR FILE named
    exactly `<16 hex>.png` or `shots.jsonl` (no directories, links, devices, paths, duplicates);
  * sizes: each PNG <= 4 MB, the manifest <= 5 MB, total <= 600 MB;
  * every PNG passes the strict structural check (lejet.pngcheck): signature, chunk CRCs, whitelisted
    chunks, sane dimensions, IEND last, nothing after it;
  * shots.jsonl: valid JSON lines with EXACTLY the expected keys and types, error codes from the closed
    list, unique url_ids, ok=true rows have a PNG whose sha256/size/width/height equal the manifest,
    and every PNG belongs to an ok=true row;
  * with --clean-jsonl: every url_id must belong to the validated batch data.

Output: <clean-dir>/shots/<batch>/<url_id>.png, shots.jsonl, and sha256 of the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from lejet import pngcheck                                       # noqa: E402
from lejet.stages import shot_stage                              # noqa: E402

MAX_MEMBERS = shot_stage.MAX_SHOTS_TAR_MEMBERS + 1
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 600 * 1024 * 1024
_PNG_NAME = re.compile(r"^[0-9a-f]{16}\.png$")
_ID = re.compile(r"^[0-9a-f]{16}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_KEYS = {"url_id": str, "ok": bool, "error": str, "bytes": int, "sha256": str, "width": int,
                 "height": int, "partial": bool, "blocked_requests": int, "ms": int,
                 "final_url": str, "title": str, "text_chars": int, "rescued": bool}
# page-controlled text: capped, and free of control / bidirectional-override characters
_TEXT_LIMITS = {"final_url": 1100, "title": 200}
_BAD_CHARS = re.compile("[\x00-\x1f\x7f\x80-\x9f‎‏‪-‮⁦-⁩]")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_manifest_line(rec: Any) -> Optional[str]:
    if not isinstance(rec, dict):
        return "not an object"
    if set(rec) != set(EXPECTED_KEYS):
        return f"keys differ: extra {sorted(set(rec) - set(EXPECTED_KEYS))}, missing {sorted(set(EXPECTED_KEYS) - set(rec))}"
    for key, typ in EXPECTED_KEYS.items():
        v = rec[key]
        if typ is int:
            low = -1 if key == "text_chars" else 0                  # -1 = unknown (only for the text count)
            if not _is_int(v) or v < low or v > 10**10:
                return f"{key}: not a plausible integer"
        elif typ is bool:
            if not isinstance(v, bool):
                return f"{key}: not a boolean"
        elif not isinstance(v, str):
            return f"{key}: not a string"
        elif key in _TEXT_LIMITS and (len(v) > _TEXT_LIMITS[key] or _BAD_CHARS.search(v)):
            return f"{key}: too long or contains control characters"
    if not _ID.match(rec["url_id"]):
        return "url_id is not 16 hex characters"
    if rec["error"] not in shot_stage.ERRORS:
        return f"error code not allowed: {rec['error'][:30]!r}"
    if rec["ok"] and rec["error"] != "none":
        return "ok row with an error code"
    if not rec["ok"] and rec["error"] == "none":
        return "failed row without an error code"
    if rec["ok"] and not _SHA.match(rec["sha256"]):
        return "ok row without a sha256"
    return None


def validate(tar_path: str, sidecar: str, batch_ids: Optional[set]) -> Tuple[List[str], Dict[str, bytes], bytes]:
    """Returns (problems, {name: png bytes}, manifest bytes). Never writes anything."""
    problems: List[str] = []
    pngs: Dict[str, bytes] = {}
    manifest = b""

    with open(tar_path, "rb") as fh:
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    try:
        expected = open(sidecar, encoding="ascii").read().split()[0].lower()
    except (OSError, IndexError, UnicodeDecodeError):
        return ["the .sha256 file is missing or unreadable"], pngs, manifest
    if expected != digest:
        return ["sha256 mismatch: the archive was damaged or altered in transit"], pngs, manifest

    try:
        tar = tarfile.open(tar_path, "r:")
    except (tarfile.TarError, OSError) as exc:
        return [f"not a readable tar file: {type(exc).__name__}"], pngs, manifest
    with tar:
        members = []
        try:
            for m in tar:
                members.append(m)
                if len(members) > MAX_MEMBERS:
                    return [f"more than {MAX_MEMBERS} members"], pngs, manifest
        except (tarfile.TarError, EOFError, OSError) as exc:
            return [f"the tar is damaged: {type(exc).__name__}"], pngs, manifest
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            problems.append("duplicate member names")
        total = 0
        for m in members:
            if not m.isreg():
                problems.append(f"member {m.name[:40]!r} is not a regular file")
                continue
            if m.name != "shots.jsonl" and not _PNG_NAME.match(m.name):
                problems.append(f"member name not allowed: {m.name[:40]!r}")
                continue
            limit = MAX_MANIFEST_BYTES if m.name == "shots.jsonl" else pngcheck.MAX_BYTES
            if m.size > limit:
                problems.append(f"{m.name}: {m.size} bytes is over the limit")
                continue
            total += m.size
            if total > MAX_TOTAL_BYTES:
                problems.append("total size over the limit")
                break
            data = tar.extractfile(m).read(limit + 1)        # type: ignore[union-attr]
            if len(data) != m.size:
                problems.append(f"{m.name}: size differs from its header")
                continue
            if m.name == "shots.jsonl":
                manifest = data
            else:
                ok, why, _, _ = pngcheck.check_png(data)
                if not ok:
                    problems.append(f"{m.name}: not a valid PNG ({why})")
                    continue
                pngs[m.name] = data
    if problems:
        return problems, {}, b""
    if not manifest:
        return ["shots.jsonl is missing"], {}, b""

    # the manifest must agree with the files, one-to-one
    seen_ids = set()
    ok_ids = set()
    try:
        lines = manifest.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return ["shots.jsonl is not UTF-8"], {}, b""
    for n, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except (ValueError, RecursionError):
            problems.append(f"shots.jsonl line {n}: not JSON")
            continue
        bad = check_manifest_line(rec)
        if bad:
            problems.append(f"shots.jsonl line {n}: {bad}")
            continue
        uid = rec["url_id"]
        if uid in seen_ids:
            problems.append(f"shots.jsonl line {n}: duplicate url_id")
            continue
        seen_ids.add(uid)
        if batch_ids is not None and uid not in batch_ids:
            problems.append(f"shots.jsonl line {n}: url_id is not in the validated batch")
        if rec["ok"]:
            ok_ids.add(uid)
            data = pngs.get(f"{uid}.png")
            if data is None:
                problems.append(f"{uid}: marked ok but no PNG in the archive")
                continue
            _, _, w, h = pngcheck.check_png(data)
            if (rec["bytes"], rec["width"], rec["height"]) != (len(data), w, h):
                problems.append(f"{uid}: size/dimensions in the manifest differ from the PNG")
            if rec["sha256"] != hashlib.sha256(data).hexdigest():
                problems.append(f"{uid}: sha256 in the manifest differs from the PNG")
    extra = {n[:-4] for n in pngs} - ok_ids
    if extra:
        problems.append(f"{len(extra)} PNG(s) in the archive are not listed as ok in the manifest")
    return problems, ({} if problems else pngs), (b"" if problems else manifest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tar")
    ap.add_argument("--sidecar", default=None, help="default: <tar>.sha256")
    ap.add_argument("--clean-dir", default=r"D:\LeJetData\clean")
    ap.add_argument("--clean-jsonl", default=None, help="validated batch data: every url_id must be in it")
    ap.add_argument("--batch", default=None, help="default: taken from the file name")
    a = ap.parse_args()

    batch = a.batch or os.path.basename(a.tar).split(".")[0]
    if not re.match(r"^[A-Za-z0-9_-]{1,40}$", batch):
        print("bad batch id", file=sys.stderr)
        return 1
    batch_ids = None
    if a.clean_jsonl:
        batch_ids = set()
        with open(a.clean_jsonl, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    batch_ids.add(json.loads(line)["url_id"])

    problems, pngs, manifest = validate(a.tar, a.sidecar or a.tar + ".sha256", batch_ids)
    if problems:
        print("SHOTS VALIDATION FAILED: nothing was written")
        for p in problems[:25]:
            print("  -", p)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        return 1

    out = os.path.join(a.clean_dir, "shots", batch)
    os.makedirs(out, exist_ok=True)
    for name, data in pngs.items():
        with open(os.path.join(out, name), "wb") as fh:
            fh.write(data)
    with open(os.path.join(out, "shots.jsonl"), "wb") as fh:
        fh.write(manifest)
    with open(a.tar, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    n_rows = len(manifest.decode("utf-8").splitlines())
    print(f"SHOTS VALIDATION OK: {len(pngs)} screenshots of {n_rows} attempts, archive sha256 {digest}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
