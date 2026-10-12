"""
pngcheck.py -- a strict structural check of a PNG file, standard library only.

Used twice: inside the VM right after the browser produced a screenshot, and on the HOST again
before a screenshot is accepted into the clean folder. The browser writes the PNG structure itself,
so an attacker's page controls only the PIXELS, never the bytes of the container; this check makes
sure that stays true (and that nothing odd, oversized or truncated gets through).

What is verified (and nothing is decoded: no image library ever touches the data):
  * the 8-byte PNG signature;
  * the first chunk is IHDR, exactly 13 bytes, with a sane width/height, bit depth, colour type,
    and compression/filter/interlace methods 0 / 0 / 0 or 1;
  * every chunk's length fits inside the file and its CRC-32 is correct;
  * chunk types are 4 ASCII letters; only a small whitelist of chunk types is accepted;
  * at least one IDAT, IEND is the last chunk and nothing follows it;
  * total size and chunk count stay under fixed limits.
"""

from __future__ import annotations

import struct
import zlib
from typing import Tuple

SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_BYTES = 4 * 1024 * 1024
MAX_CHUNKS = 5000
MAX_DIMENSION = 16384
MAX_PIXELS = 40_000_000
# critical chunks plus the harmless ancillary ones a browser may write
ALLOWED_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA", b"cHRM", b"sRGB", b"iCCP",
                  b"pHYs", b"sBIT", b"bKGD", b"tIME"}
_VALID_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}


def check_png(data: bytes) -> Tuple[bool, str, int, int]:
    """Return (ok, reason, width, height). `reason` is "" when ok. Never raises."""
    try:
        return _check(data)
    except Exception as exc:                                    # noqa: BLE001 - defensive: junk in, "no" out
        return False, f"internal:{type(exc).__name__}", 0, 0


def _check(data: bytes) -> Tuple[bool, str, int, int]:
    if not isinstance(data, (bytes, bytearray)):
        return False, "not bytes", 0, 0
    n = len(data)
    if n < len(SIGNATURE) + 12 + 13 + 12:
        return False, "too short", 0, 0
    if n > MAX_BYTES:
        return False, "too large", 0, 0
    if data[:8] != SIGNATURE:
        return False, "bad signature", 0, 0

    pos = 8
    chunks = 0
    width = height = 0
    seen_idat = False
    first = True
    while pos < n:
        if pos + 8 > n:
            return False, "truncated chunk header", 0, 0
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = bytes(data[pos + 4:pos + 8])
        if length > n or pos + 12 + length > n:
            return False, "chunk length beyond end of file", 0, 0
        if not all(65 <= b <= 90 or 97 <= b <= 122 for b in ctype):
            return False, "bad chunk type", 0, 0
        if ctype not in ALLOWED_CHUNKS:
            return False, f"chunk type not allowed: {ctype.decode('ascii')}", 0, 0
        body = data[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
            return False, "bad CRC", 0, 0
        chunks += 1
        if chunks > MAX_CHUNKS:
            return False, "too many chunks", 0, 0

        if first:
            if ctype != b"IHDR" or length != 13:
                return False, "IHDR missing or wrong size", 0, 0
            width, height, depth, colour, comp, filt, interlace = struct.unpack(">IIBBBBB", body)
            if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
                return False, "dimensions out of range", 0, 0
            if width * height > MAX_PIXELS:
                return False, "too many pixels", 0, 0
            if colour not in _VALID_DEPTHS or depth not in _VALID_DEPTHS[colour]:
                return False, "bad colour type / bit depth", 0, 0
            if comp != 0 or filt != 0 or interlace not in (0, 1):
                return False, "bad compression / filter / interlace method", 0, 0
            first = False
        elif ctype == b"IHDR":
            return False, "second IHDR", 0, 0
        if ctype == b"IDAT":
            seen_idat = True
        pos += 12 + length
        if ctype == b"IEND":
            if length != 0:
                return False, "IEND with data", 0, 0
            if pos != n:
                return False, "data after IEND", 0, 0
            if not seen_idat:
                return False, "no image data", 0, 0
            return True, "", width, height
    return False, "no IEND", 0, 0
