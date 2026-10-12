"""
Tests for scripts/validate_shots.py (the host-side gate for the screenshot archive) and
scripts/make_gallery.py. The archive comes from the VM and is treated as hostile: these tests build
malicious archives on purpose.

Run with:  python -m pytest tests/test_validate_shots.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import tarfile

import pytest

from lejet.stages import shot_stage
from lejet.stages.shot_stage import ShotResult
from test_pngcheck import png

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vs = load("validate_shots")
gal = load("make_gallery")

A, B, C = "0123456789abcdef", "fedcba9876543210", "aaaaaaaaaaaaaaaa"


def manifest_line(uid, data=None, **kw):
    rec = {"url_id": uid, "ok": data is not None, "error": "none" if data is not None else "goto_timeout",
           "bytes": len(data) if data else 0, "sha256": hashlib.sha256(data).hexdigest() if data else "",
           "width": 8 if data else 0, "height": 8 if data else 0, "partial": False,
           "blocked_requests": 0, "ms": 1234, "final_url": "hxxps://shop[.]example[.]com/", "title": "Example shop",
           "text_chars": 120 if data else -1}
    rec.update(kw)
    return json.dumps(rec, sort_keys=True)


def make_tar(path, files, extra=()):
    """files: list of (name, bytes). extra: raw TarInfo objects with data (for evil members)."""
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for info, data in extra:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    blob = open(path, "rb").read()
    with open(path + ".sha256", "w") as fh:
        fh.write(f"{hashlib.sha256(blob).hexdigest()}  {os.path.basename(path)}\n")
    return path


def good_files():
    p = png(8, 8)
    return [(f"{A}.png", p), ("shots.jsonl", (manifest_line(A, p) + "\n" + manifest_line(B) + "\n").encode())]


def check(tmp_path, files, extra=(), batch_ids=None):
    tar = make_tar(str(tmp_path / "b1.shots.tar"), files, extra)
    return vs.validate(tar, tar + ".sha256", batch_ids)


def test_a_good_archive_passes_and_returns_the_files(tmp_path):
    problems, pngs, manifest = check(tmp_path, good_files(), batch_ids={A, B})
    assert problems == [] and list(pngs) == [f"{A}.png"] and manifest.startswith(b"{")


def test_an_archive_made_by_the_real_packer_passes(tmp_path):
    good = ShotResult(A, ok=True, error="none", bytes=len(png(8, 8)), sha256=hashlib.sha256(png(8, 8)).hexdigest(),
                      width=8, height=8)
    bad = ShotResult(B, ok=False, error="navigation_error")
    (tmp_path / f"{A}.png").write_bytes(png(8, 8))
    tar = str(tmp_path / "b9.shots.tar")
    shot_stage.pack_tar(str(tmp_path), [good, bad], tar)
    open(tar + ".sha256", "w").write(f"{hashlib.sha256(open(tar, 'rb').read()).hexdigest()}  b9.shots.tar\n")
    problems, pngs, _ = vs.validate(tar, tar + ".sha256", {A, B})
    assert problems == [] and len(pngs) == 1


@pytest.mark.parametrize("name", ["../evil.png", "/etc/passwd", "sub/0123456789abcdef.png", "0123456789ABCDEF.png",
                                  "0123456789abcdef.PNG", "0123456789abcdef.png.exe", "short.png",
                                  "0123456789abcdef.html", "index.html", "..", ".", "a" * 95])
def test_member_names_outside_the_pattern_are_refused(tmp_path, name):
    files = good_files() + [(name, png(8, 8))]
    problems, pngs, _ = check(tmp_path, files)
    assert problems and pngs == {}


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE,
                                  tarfile.FIFOTYPE])
def test_links_directories_and_devices_are_refused(tmp_path, kind):
    info = tarfile.TarInfo(f"{C}.png")
    info.type = kind
    info.linkname = "/etc/passwd"
    problems, pngs, _ = check(tmp_path, good_files(), extra=[(info, None)])
    assert any("not a regular file" in p for p in problems) and pngs == {}


def test_duplicate_members_are_refused(tmp_path):
    problems, _, _ = check(tmp_path, good_files() + [(f"{A}.png", png(8, 8))])
    assert any("duplicate" in p for p in problems)


@pytest.mark.parametrize("data", [b"", b"not a png", b"<html><script>alert(1)</script></html>", png(8, 8)[:-4],
                                  png(8, 8) + b"trailing"])
def test_a_bad_png_inside_the_archive_is_refused(tmp_path, data):
    files = [(f"{A}.png", data), ("shots.jsonl", (manifest_line(A, png(8, 8)) + "\n").encode())]
    problems, pngs, _ = check(tmp_path, files)
    assert any("not a valid PNG" in p for p in problems) and pngs == {}


def test_a_wrong_sidecar_hash_is_refused(tmp_path):
    tar = make_tar(str(tmp_path / "b1.shots.tar"), good_files())
    open(tar + ".sha256", "w").write("0" * 64 + "  b1.shots.tar\n")
    problems, _, _ = vs.validate(tar, tar + ".sha256", None)
    assert problems and "sha256 mismatch" in problems[0]


def test_a_missing_or_garbage_sidecar_is_refused(tmp_path):
    tar = make_tar(str(tmp_path / "b1.shots.tar"), good_files())
    os.remove(tar + ".sha256")
    assert vs.validate(tar, tar + ".sha256", None)[0]
    open(tar + ".sha256", "w").write("")
    assert vs.validate(tar, tar + ".sha256", None)[0]


def test_a_file_that_is_not_a_tar_is_refused(tmp_path):
    p = tmp_path / "x.shots.tar"
    p.write_bytes(b"this is not a tar file" * 100)
    open(str(p) + ".sha256", "w").write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  x\n")
    problems, _, _ = vs.validate(str(p), str(p) + ".sha256", None)
    assert problems


def test_a_truncated_tar_is_refused(tmp_path):
    tar = make_tar(str(tmp_path / "b1.shots.tar"), good_files())
    blob = open(tar, "rb").read()
    open(tar, "wb").write(blob[:700])
    open(tar + ".sha256", "w").write(f"{hashlib.sha256(blob[:700]).hexdigest()}  b1.shots.tar\n")
    assert vs.validate(tar, tar + ".sha256", None)[0]


def test_the_manifest_is_required(tmp_path):
    problems, _, _ = check(tmp_path, [(f"{A}.png", png(8, 8))])
    assert problems


@pytest.mark.parametrize("mutate, why", [
    (lambda r: r.pop("ms"), "keys differ"),
    (lambda r: r.update(extra="x"), "keys differ"),
    (lambda r: r.update(url_id="XYZ"), "url_id"),
    (lambda r: r.update(error="rm -rf /"), "error code not allowed"),
    (lambda r: r.update(ok=True, error="goto_timeout"), "ok row with an error code"),
    (lambda r: r.update(bytes="12"), "bytes"),
    (lambda r: r.update(width=-3), "width"),
    (lambda r: r.update(partial="yes"), "partial"),
    (lambda r: r.update(sha256="abc"), "sha256"),
    (lambda r: r.update(title="evil \x1b[31mred"), "control characters"),
    (lambda r: r.update(title="a" * 500), "too long"),
    (lambda r: r.update(final_url="hxxps://x[.]example[.]com/‮exe.txt"), "control characters"),
    (lambda r: r.update(final_url="a" * 5000), "too long"),
    (lambda r: r.update(title=5), "title"),
    (lambda r: r.update(text_chars=-5), "text_chars"),
    (lambda r: r.update(text_chars="12"), "text_chars"),
])
def test_manifest_lines_with_wrong_shape_are_refused(tmp_path, mutate, why):
    p = png(8, 8)
    rec = json.loads(manifest_line(A, p))
    mutate(rec)
    files = [(f"{A}.png", p), ("shots.jsonl", (json.dumps(rec) + "\n").encode())]
    problems, _, _ = check(tmp_path, files)
    assert problems and any(why in x for x in problems), problems


def test_a_manifest_that_lies_about_the_png_is_refused(tmp_path):
    p = png(8, 8)
    files = [(f"{A}.png", p), ("shots.jsonl", (manifest_line(A, p, sha256="b" * 64) + "\n").encode())]
    assert any("sha256" in x for x in check(tmp_path, files)[0])
    files = [(f"{A}.png", p), ("shots.jsonl", (manifest_line(A, p, width=99) + "\n").encode())]
    assert any("dimensions" in x for x in check(tmp_path, files)[0])
    files = [(f"{A}.png", p), ("shots.jsonl", (manifest_line(A, p, bytes=5) + "\n").encode())]
    assert check(tmp_path, files)[0]


def test_ok_without_a_png_and_png_without_ok_are_refused(tmp_path):
    p = png(8, 8)
    files = [("shots.jsonl", (manifest_line(A, p) + "\n").encode())]
    assert any("no PNG" in x for x in check(tmp_path, files)[0])
    files = [(f"{A}.png", p), (f"{B}.png", p), ("shots.jsonl", (manifest_line(A, p) + "\n" + manifest_line(B) + "\n").encode())]
    assert any("not listed as ok" in x for x in check(tmp_path, files)[0])


def test_duplicate_ids_in_the_manifest_are_refused(tmp_path):
    p = png(8, 8)
    files = [(f"{A}.png", p), ("shots.jsonl", (manifest_line(A, p) + "\n" + manifest_line(A, p) + "\n").encode())]
    assert any("duplicate url_id" in x for x in check(tmp_path, files)[0])


def test_ids_that_are_not_in_the_validated_batch_are_refused(tmp_path):
    problems, _, _ = check(tmp_path, good_files(), batch_ids={A})
    assert any("not in the validated batch" in x for x in problems)


@pytest.mark.parametrize("junk", [b"not json\n", b"[1,2,3]\n", b"null\n", b"\xff\xfe\n", b'{"url_id": 5}\n'])
def test_garbage_manifests_are_refused(tmp_path, junk):
    files = [(f"{A}.png", png(8, 8)), ("shots.jsonl", junk)]
    assert check(tmp_path, files)[0]


def test_a_header_that_lies_about_the_size_does_not_make_us_read_a_lot(tmp_path):
    info = tarfile.TarInfo(f"{C}.png")
    info.size = 10**9                                 # claims a gigabyte, carries nothing
    tar = str(tmp_path / "b1.shots.tar")
    # write the header by hand: a truncated archive that claims a huge member
    hdr = info.tobuf(tarfile.USTAR_FORMAT)
    open(tar, "wb").write(hdr + b"\x00" * 1024)
    open(tar + ".sha256", "w").write(f"{hashlib.sha256(open(tar, 'rb').read()).hexdigest()}  b1.shots.tar\n")
    problems, pngs, _ = vs.validate(tar, tar + ".sha256", None)
    assert problems and pngs == {}


def test_main_writes_nothing_when_validation_fails(tmp_path, monkeypatch, capsys):
    files = good_files() + [("../evil.png", png(8, 8))]
    tar = make_tar(str(tmp_path / "b7.shots.tar"), files)
    out = tmp_path / "clean"
    monkeypatch.setattr("sys.argv", ["validate_shots.py", tar, "--clean-dir", str(out)])
    assert vs.main() == 1
    assert not out.exists()


def test_main_writes_the_checked_files_on_success(tmp_path, monkeypatch, capsys):
    tar = make_tar(str(tmp_path / "b7.shots.tar"), good_files())
    clean = tmp_path / "b7.clean.jsonl"
    clean.write_text("\n".join(json.dumps({"url_id": u}) for u in (A, B)) + "\n")
    out = tmp_path / "clean"
    monkeypatch.setattr("sys.argv", ["validate_shots.py", tar, "--clean-dir", str(out), "--clean-jsonl", str(clean)])
    assert vs.main() == 0
    shots = out / "shots" / "b7"
    assert sorted(os.listdir(shots)) == [f"{A}.png", "shots.jsonl"]
    assert (shots / f"{A}.png").read_bytes() == png(8, 8)
    assert "SHOTS VALIDATION OK" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------
# the gallery page must be inert even when the (defanged) text is hostile
# ---------------------------------------------------------------------------------------
def test_the_gallery_is_inert_and_escapes_everything():
    rows = [{"url_id": A, "batch_id": "b7", "final_url": "hxxps://evil[.]com/<script>alert(1)</script>\"onerror=\"x",
             "final_status": 200},
            {"url_id": B, "batch_id": "b7", "final_url": "hxxp://other[.]org/", "final_status": 404}]
    page = gal.build_html(rows, {A}, "b7<img src=x onerror=y>")
    assert "<script" not in page.lower() and "<a " not in page.lower() and "<form" not in page.lower()
    assert "onerror=\"x" not in page and "&lt;script&gt;" in page and "&quot;onerror=&quot;x" in page
    assert "Content-Security-Policy" in page and "img-src 'self'" in page and "default-src 'none'" in page
    assert f"{A}.png" in page and f"{B}.png" not in page
    assert "1 pages without one" in page
