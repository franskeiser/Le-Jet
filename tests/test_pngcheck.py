"""
Tests for src/lejet/pngcheck.py: the strict structural PNG check applied to every screenshot, in the
VM and again on the host. PNGs are built by hand with zlib, so no image library is needed.

Run with:  python -m pytest tests/test_pngcheck.py
"""

from __future__ import annotations

import struct
import zlib

import pytest

from lejet import pngcheck


def chunk(ctype: bytes, body: bytes, crc: int = None) -> bytes:
    if crc is None:
        crc = zlib.crc32(ctype + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", crc)


def ihdr(w=4, h=3, depth=8, colour=2, comp=0, filt=0, interlace=0) -> bytes:
    return chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, colour, comp, filt, interlace))


def idat(w=4, h=3) -> bytes:
    raw = b"".join(b"\x00" + b"\x10\x20\x30" * w for _ in range(h))
    return chunk(b"IDAT", zlib.compress(raw))


def png(w=4, h=3, **kw) -> bytes:
    return pngcheck.SIGNATURE + ihdr(w, h, **kw) + idat(w, h) + chunk(b"IEND", b"")


def test_a_valid_png_passes_and_reports_its_size():
    ok, reason, w, h = pngcheck.check_png(png(1280, 8))
    assert ok and reason == "" and (w, h) == (1280, 8)


def test_harmless_ancillary_chunks_are_accepted():
    data = (pngcheck.SIGNATURE + ihdr() + chunk(b"sRGB", b"\x00") + chunk(b"pHYs", b"\x00" * 9)
            + idat() + chunk(b"IEND", b""))
    assert pngcheck.check_png(data)[0]


@pytest.mark.parametrize("name, data, why", [
    ("empty", b"", "too short"),
    ("text", b"hello world, definitely not a png at all, " * 5, "bad signature"),
    ("html", b"<html><script>alert(1)</script></html>" * 5, "bad signature"),
    ("jpeg", b"\xff\xd8\xff\xe0" + b"\x00" * 100, "bad signature"),
    ("signature only", pngcheck.SIGNATURE + b"\x00" * 10, "too short"),
])
def test_things_that_are_not_pngs_are_refused(name, data, why):
    ok, reason, _, _ = pngcheck.check_png(data)
    assert not ok and why in reason


def test_a_bad_crc_is_refused():
    data = bytearray(png())
    data[20] ^= 0xFF                                  # inside IHDR
    assert pngcheck.check_png(bytes(data)) == (False, "bad CRC", 0, 0)


def test_truncated_file_is_refused():
    data = png()
    for cut in (len(data) - 1, len(data) - 12, len(data) // 2, 40):
        assert not pngcheck.check_png(data[:cut])[0]


def test_data_after_iend_is_refused():
    ok, reason, _, _ = pngcheck.check_png(png() + b"GARBAGE-AFTER-THE-END")
    assert not ok and reason == "data after IEND"


def test_missing_iend_is_refused():
    data = pngcheck.SIGNATURE + ihdr() + idat()
    assert pngcheck.check_png(data) == (False, "no IEND", 0, 0)


def test_a_png_without_image_data_is_refused():
    data = pngcheck.SIGNATURE + ihdr() + chunk(b"IEND", b"")
    ok, reason, _, _ = pngcheck.check_png(data)
    assert not ok and reason == "no image data"


def test_ihdr_must_come_first():
    data = pngcheck.SIGNATURE + chunk(b"gAMA", b"\x00\x00\x00\x01") + ihdr() + idat() + chunk(b"IEND", b"")
    assert pngcheck.check_png(data)[1] == "IHDR missing or wrong size"


def test_second_ihdr_is_refused():
    data = pngcheck.SIGNATURE + ihdr() + ihdr() + idat() + chunk(b"IEND", b"")
    assert pngcheck.check_png(data)[1] == "second IHDR"


@pytest.mark.parametrize("kw", [
    dict(w=0), dict(h=0), dict(w=20000), dict(h=20000),
    dict(depth=3), dict(depth=16, colour=3), dict(colour=1), dict(colour=7),
    dict(comp=1), dict(filt=1), dict(interlace=2),
])
def test_unreasonable_headers_are_refused(kw):
    w = kw.get("w", 4)
    h = kw.get("h", 3)
    data = pngcheck.SIGNATURE + ihdr(**kw) + chunk(b"IDAT", zlib.compress(b"\x00" * 16)) + chunk(b"IEND", b"")
    assert not pngcheck.check_png(data)[0], (w, h)


def test_too_many_pixels_is_refused():
    data = pngcheck.SIGNATURE + ihdr(16000, 16000) + chunk(b"IDAT", zlib.compress(b"\x00" * 16)) + chunk(b"IEND", b"")
    assert pngcheck.check_png(data)[1] == "too many pixels"


@pytest.mark.parametrize("ctype", [b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"acTL", b"fcTL", b"zzzz"])
def test_chunk_types_outside_the_whitelist_are_refused(ctype):
    data = pngcheck.SIGNATURE + ihdr() + chunk(ctype, b"hidden payload") + idat() + chunk(b"IEND", b"")
    ok, reason, _, _ = pngcheck.check_png(data)
    assert not ok and "not allowed" in reason


def test_bad_chunk_type_characters_are_refused():
    data = pngcheck.SIGNATURE + ihdr() + chunk(b"ID\x00T", b"x") + idat() + chunk(b"IEND", b"")
    assert pngcheck.check_png(data)[1] == "bad chunk type"


def test_a_chunk_length_beyond_the_file_is_refused():
    data = bytearray(png())
    data[8:12] = struct.pack(">I", 0x7FFFFFFF)       # IHDR claims 2 GB
    assert pngcheck.check_png(bytes(data))[1] == "chunk length beyond end of file"


def test_oversized_files_are_refused():
    big = pngcheck.SIGNATURE + ihdr() + chunk(b"IDAT", b"\x00" * (pngcheck.MAX_BYTES + 1)) + chunk(b"IEND", b"")
    assert pngcheck.check_png(big)[1] == "too large"


def test_iend_with_data_is_refused():
    data = pngcheck.SIGNATURE + ihdr() + idat() + chunk(b"IEND", b"x")
    assert pngcheck.check_png(data)[1] == "IEND with data"


@pytest.mark.parametrize("junk", [None, "text", 5, [1, 2], {"a": 1}])
def test_non_bytes_never_raise(junk):
    assert pngcheck.check_png(junk)[0] is False


def test_random_junk_never_raises():
    import random
    rng = random.Random(1234)
    good = png()
    for _ in range(300):
        data = bytearray(good)
        for _ in range(rng.randint(1, 6)):
            data[rng.randrange(len(data))] = rng.randrange(256)
        ok, reason, _, _ = pngcheck.check_png(bytes(data))
        assert isinstance(ok, bool) and isinstance(reason, str)
        assert not (ok and bytes(data) != good and reason)
