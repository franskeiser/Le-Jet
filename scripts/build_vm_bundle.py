#!/usr/bin/env python
"""
build_vm_bundle.py -- build the offline, hash-pinned install bundle for the analysis VM
(WEB_INSPECTION_PLAN.md sections 8.3, 11.7). Run it on the HOST:

    python scripts/build_vm_bundle.py [--out D:\\LeJetData\\bundle]

It
  1. downloads, for Python 3.12 / linux x86_64, ONLY binary wheels of the packages in
     requirements/vm.in (+ their dependencies) and of requirements/vm-test.in;
  2. builds a wheel of `nostril` from its pinned git commit (it is not on PyPI under that name);
  3. writes requirements/vm.txt and vm-test.txt WITH SHA-256 HASHES of every wheel;
  4. packs wheels + code + tests + installer into lejet-bundle.tar.gz with a MANIFEST.sha256.

Outputs go to --out (outside the repository, outside OneDrive). Copy vm.txt / vm-test.txt from
there into requirements/ when they change.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOSTRIL = "git+https://github.com/casics/nostril.git@fbc0c91249283a9fbc9036206391ce1138826fd3"
PLATFORMS = ["manylinux2014_x86_64"] + [f"manylinux_2_{n}_x86_64" for n in range(17, 40)]

CODE_FILES = ["src/features/link_features.py", "src/data/clean_web_phishing.py"]
CODE_DIRS = ["src/lejet"]
# tests that can run in the VM (no pandas / scikit-learn / dataset needed)
TEST_FILES = [
    "tests/conftest.py", "tests/fixtures/pages.py", "tests/test_urlnorm.py", "tests/test_safety.py",
    "tests/test_html_features.py", "tests/test_free_hosting.py", "tests/test_external_features.py",
    "tests/test_dns_stage.py", "tests/test_rdap_stage.py", "tests/test_parse_pool.py",
    "tests/test_http_stage.py", "tests/test_pipeline_local.py", "tests/test_storage_export.py",
    "tests/test_cli.py", "tests/test_schema.py", "tests/test_link_features_golden.py",
    "tests/test_ct_stage.py", "tests/test_pipeline_ct.py", "tests/test_pngcheck.py",
    "tests/test_shot_stage.py", "tests/test_shot_browser.py", "tests/test_guard_proxy.py", "tests/test_live_worker.py",
    "tests/test_hannousse_url.py", "tests/test_hannousse_html.py",
    "tests/fixtures/link_features_golden.json",
]
EXTRA_FILES = ["infra/analysis/install_bundle.sh", "infra/analysis/preflight.sh",
               "infra/analysis/lejet-run.sh", "infra/analysis/lejet-export.sh",
               "infra/analysis/helpers/lejet-sudo-preflight.sh", "infra/analysis/helpers/lejet-sudo-install-urls.sh",
               "infra/analysis/helpers/lejet-sudo-batch.sh", "infra/analysis/helpers/lejet-sudo-export.sh",
               "infra/analysis/helpers/lejet-sudo-shots.sh", "infra/analysis/helpers/lejet-sudo-live.sh",
               "infra/analysis/systemd/lejet-gate.service", "infra/analysis/systemd/lejet-gate.timer",
               "infra/analysis/systemd/lejet-live.service",
               "infra/analysis/preflight_sequential.sh", "infra/analysis/install_browser.sh",
               "infra/analysis/helpers/91-lejet-helpers.sudoers",
               "requirements/vm.in", "requirements/vm-test.in"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd[:6]), "..." if len(cmd) > 6 else "")
    subprocess.run(cmd, check=True)


def download(req_file: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "--quiet", "--dest", str(dest), "--only-binary=:all:",
           "--python-version", "3.12", "--implementation", "cp", "--abi", "cp312", "-r", str(req_file)]
    for p in PLATFORMS:
        cmd += ["--platform", p]
    run(cmd)


def wheel_pin(name: str) -> tuple[str, str]:
    m = re.match(r"^([A-Za-z0-9_.]+)-([0-9][^-]*)(?:-\d[^-]*)?-[^-]+-[^-]+-[^-]+\.whl$", name)
    if not m:
        raise SystemExit(f"cannot parse wheel name {name}")
    return re.sub(r"[-_.]+", "-", m.group(1)).lower(), m.group(2)


def requirements_text(wheels: list[Path], header: str) -> str:
    lines = [header]
    for w in sorted(wheels, key=lambda p: wheel_pin(p.name)):
        name, ver = wheel_pin(w.name)
        lines.append(f"{name}=={ver} \\\n    --hash=sha256:{sha256(w)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\LeJetData\bundle")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "wheels-run").mkdir(parents=True)
    wheels = out / "wheels"
    wheels.mkdir()

    print("1/4 downloading wheels for Python 3.12 / linux x86_64")
    run_in = ROOT / "requirements" / "vm.in"
    download(run_in, out / "wheels-run")
    download(ROOT / "requirements" / "vm-test.in", out / "wheels-test")

    print("2/4 building the nostril wheel from its pinned commit")
    run([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "--wheel-dir", str(out / "wheels-run"), NOSTRIL])

    run_wheels = sorted((out / "wheels-run").glob("*.whl"))
    test_only = [w for w in sorted((out / "wheels-test").glob("*.whl"))
                 if w.name not in {r.name for r in run_wheels}]
    for w in run_wheels + test_only:
        shutil.copy2(w, wheels / w.name)
    for w in wheels.glob("*.whl"):
        if not (w.name.endswith("-none-any.whl") or "linux" in w.name):
            print("  WARNING: not a linux/any wheel:", w.name)

    print("3/4 writing hash-pinned requirements")
    head = "# Generated by scripts/build_vm_bundle.py -- do not edit by hand.\n"
    # newline="\n": the files go to a Linux VM and into git; never CRLF
    (out / "vm.txt").write_text(requirements_text(run_wheels, head + "# runtime (offline install)"),
                                encoding="utf-8", newline="\n")
    (out / "vm-test.txt").write_text(
        requirements_text(run_wheels + test_only, head + "# runtime + pytest (offline install)"),
        encoding="utf-8", newline="\n")

    print("4/4 packing the bundle")
    files: dict[str, Path] = {}
    for rel in CODE_FILES:
        files["code/" + rel] = ROOT / rel
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo"):
                files["code/" + p.relative_to(ROOT).as_posix()] = p
    for rel in TEST_FILES:
        if (ROOT / rel).exists():
            files["code/" + rel] = ROOT / rel
    for rel in EXTRA_FILES:
        if (ROOT / rel).exists():
            files[rel if rel.startswith("infra/") else "extra/" + rel] = ROOT / rel
    for w in sorted(wheels.glob("*.whl")):
        files["wheels/" + w.name] = w
    files["requirements/vm.txt"] = out / "vm.txt"
    files["requirements/vm-test.txt"] = out / "vm-test.txt"
    files["install_bundle.sh"] = ROOT / "infra" / "analysis" / "install_bundle.sh"

    # The bytes that go into the archive (text files with LF only: the VM is Linux). The manifest
    # hashes exactly these bytes, so `sha256sum -c MANIFEST.sha256` on the VM passes.
    payload: dict[str, bytes] = {}
    for name, p in sorted(files.items()):
        data = p.read_bytes()
        if name.endswith((".sh", ".txt", ".in", ".py", ".json", ".sudoers")) and not name.startswith("wheels/"):
            data = data.replace(b"\r\n", b"\n")
        payload[name] = data
    manifest = "".join(f"{hashlib.sha256(d).hexdigest()}  {n}\n" for n, d in payload.items())
    bundle = out / "lejet-bundle.tar.gz"
    with tarfile.open(bundle, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        def add(name: str, data: bytes, mode: int = 0o644) -> None:
            info = tarfile.TarInfo("lejet-bundle/" + name)
            info.size, info.mtime, info.mode = len(data), 0, mode
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            tar.addfile(info, io.BytesIO(data))
        for name, data in payload.items():
            add(name, data, 0o755 if name.endswith(".sh") else 0o644)
        add("MANIFEST.sha256", manifest.encode("ascii"))
    print(f"\nbundle: {bundle}  ({bundle.stat().st_size // 1024} KB, {len(files)} files)")
    print("sha256:", sha256(bundle))
    print("pinned runtime wheels:", len(run_wheels), "| test-only wheels:", len(test_only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
