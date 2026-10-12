#!/usr/bin/env python
"""
verify_vm_bundle.py -- check the offline VM bundle BEFORE it is copied anywhere
(WEB_INSPECTION_PLAN.md sections 8.3, 11.7). Run on the host:

    python scripts/verify_vm_bundle.py [--bundle D:\\LeJetData\\bundle\\lejet-bundle.tar.gz]

Checks:
  1. MANIFEST.sha256 matches every file in the archive;
  2. every wheel is pinned in requirements/vm-test.txt with the right hash, and every pin has a
     wheel (so `pip --require-hashes --no-deps` can install all of it, nothing more);
  3. the dependency CLOSURE is complete for Python 3.12 on linux: every Requires-Dist of every
     wheel (markers evaluated for the VM, not for this host) is present among the pins;
  4. wheels are linux x86_64 cp312 or pure-Python;
  5. shell scripts use LF line endings;
  6. the files the VM needs are present.
Exit code 0 only if everything passes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import tarfile
import zipfile

VM_ENV = {
    "python_version": "3.12", "python_full_version": "3.12.3", "sys_platform": "linux",
    "platform_system": "Linux", "os_name": "posix", "implementation_name": "cpython",
    "platform_python_implementation": "CPython", "platform_machine": "x86_64", "extra": "",
}
# Known packaging mistakes that are deliberately ignored: (wheel, requirement).
IGNORED = {("nostril", "pytest")}
REQUIRED_FILES = [
    "code/src/lejet/cli.py", "code/src/lejet/pipeline.py", "code/src/lejet/parse_pool.py",
    "code/src/features/link_features.py", "code/src/data/clean_web_phishing.py",
    "code/src/lejet/features/free_hosting_suffixes.txt", "install_bundle.sh",
    "requirements/vm.txt", "requirements/vm-test.txt", "MANIFEST.sha256",
    "infra/analysis/preflight.sh", "infra/analysis/lejet-run.sh", "infra/analysis/lejet-export.sh",
    "infra/analysis/helpers/lejet-sudo-preflight.sh", "infra/analysis/helpers/lejet-sudo-install-urls.sh",
    "infra/analysis/helpers/lejet-sudo-batch.sh", "infra/analysis/helpers/lejet-sudo-export.sh",
    "infra/analysis/helpers/lejet-sudo-shots.sh", "infra/analysis/install_browser.sh",
    "infra/analysis/helpers/91-lejet-helpers.sudoers", "code/src/lejet/browser_selftest.py",
    "code/src/lejet/stages/shot_stage.py", "code/src/lejet/pngcheck.py", "code/src/lejet/stages/guard_proxy.py",
]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=r"D:\LeJetData\bundle\lejet-bundle.tar.gz")
    args = ap.parse_args()
    try:
        from packaging.markers import Marker
        from packaging.requirements import Requirement
    except ImportError:
        print("this check needs the `packaging` package (pip install packaging)")
        return 2

    problems: list[str] = []
    tar = tarfile.open(args.bundle)
    names = tar.getnames()
    root = "lejet-bundle/"

    def read(name: str) -> bytes:
        return tar.extractfile(root + name).read()

    # 1. manifest
    entries = [l.split("  ", 1) for l in read("MANIFEST.sha256").decode().splitlines() if l.strip()]
    for digest, name in entries:
        if hashlib.sha256(read(name)).hexdigest() != digest:
            problems.append(f"manifest mismatch: {name}")
    listed = {n for _, n in entries}
    actual = {n[len(root):] for n in names if n.startswith(root) and n != root.rstrip("/")
              and tar.getmember(n).isfile() and n != root + "MANIFEST.sha256"}
    if listed != actual:
        problems.append(f"files not in the manifest: {sorted(actual - listed)[:5]} / missing: {sorted(listed - actual)[:5]}")
    print(f"1. manifest: {len(entries)} entries checked")

    # 2. pins vs wheels
    req_text = read("requirements/vm-test.txt").decode()
    pins = {}
    if "\r" in req_text:
        problems.append("requirements file has CRLF line endings")
    for m in re.finditer(r"^([a-z0-9-]+)==(\S+) \\\r?\n\s+--hash=sha256:([0-9a-f]{64})", req_text, re.M):
        pins[norm(m.group(1))] = (m.group(2), m.group(3))
    wheels = {}
    for n in names:
        if n.endswith(".whl"):
            data = tar.extractfile(n).read()
            m = re.match(r"^([A-Za-z0-9_.]+)-([0-9][^-]*)(?:-\d[^-]*)?-[^-]+-[^-]+-[^-]+\.whl$", n.split("/")[-1])
            wheels[norm(m.group(1))] = (m.group(2), hashlib.sha256(data).hexdigest(), n.split("/")[-1], data)
    for name, (ver, digest, fname, _) in wheels.items():
        if name not in pins:
            problems.append(f"wheel without a pin: {fname}")
        elif pins[name] != (ver, digest):
            problems.append(f"pin differs from the wheel: {name}")
    for name in pins:
        if name not in wheels:
            problems.append(f"pin without a wheel: {name}")
    runtime_pins = set(re.findall(r"^([a-z0-9-]+)==", read("requirements/vm.txt").decode(), re.M))
    if not pins or not runtime_pins:
        problems.append("no pins parsed from the requirements files")
    print(f"2. pins: {len(pins)} (runtime {len(runtime_pins)}), wheels: {len(wheels)}")

    # 3. dependency closure (for the VM, with and without the test extras)
    def closure_missing(available: set[str]) -> list[str]:
        out = []
        for name in sorted(available):
            if name not in wheels:
                continue
            z = zipfile.ZipFile(io.BytesIO(wheels[name][3]))
            meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
            for line in z.read(meta).decode("utf-8", "replace").splitlines():
                if not line.startswith("Requires-Dist:"):
                    continue
                req = Requirement(line.split(":", 1)[1].strip())
                if req.marker is not None and not req.marker.evaluate(VM_ENV):
                    continue
                dep = norm(req.name)
                if (name, dep) in IGNORED:
                    continue
                if dep not in available:
                    out.append(f"{name} needs {dep}")
        return out
    for label, avail in (("runtime", runtime_pins), ("runtime+test", set(pins))):
        missing = closure_missing(avail)
        problems += [f"[{label}] {m}" for m in missing]
        print(f"3. closure ({label}): {'complete' if not missing else 'MISSING ' + ', '.join(missing)}")

    # 4. wheel tags
    for name, (_, _, fname, _) in wheels.items():
        linux_x86 = "x86_64" in fname and "manylinux" in fname           # glibc builds (not musllinux)
        if not (fname.endswith("-none-any.whl")
                or ("cp312" in fname and linux_x86)
                or ("-py3-none-" in fname and linux_x86)):                # e.g. playwright (manylinux1)
            problems.append(f"wrong wheel tag for the VM: {fname}")
            continue
        if "musllinux" in fname or "aarch64" in fname:
            problems.append(f"wrong wheel tag for the VM: {fname}")
    print("4. wheel tags checked")

    # 5. line endings
    for n in names:
        if n.endswith((".sh", ".sudoers")) and b"\r" in tar.extractfile(n).read():
            problems.append(f"CRLF in {n}")
    print("5. shell scripts: LF only" if not any("CRLF" in p for p in problems) else "5. CRLF found")

    # 6. required files
    for f in REQUIRED_FILES:
        if root + f not in names:
            problems.append(f"missing in bundle: {f}")
    print("6. required files present" if not any(p.startswith("missing in bundle") for p in problems) else "6. files missing")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("BUNDLE OK  sha256:", hashlib.sha256(open(args.bundle, "rb").read()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
