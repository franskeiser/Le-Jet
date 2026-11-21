"""
screens.py -- read SMS screenshots: picture -> text lines (OCR) -> separate message bubbles.

    load_image(bytes)                        validate + decode one upload (size, type, rotation), raise ImageRejected
    Reader().lines(image)                    RapidOCR on the VM itself (no outside service, nothing is stored)
    split_messages(lines, width, height)     group OCR lines into bubbles, drop phone chrome, tell sent from received

The grouping is a heuristic on the position of each text line, so the page ALWAYS shows its result for a person to
correct (edit, merge, remove) before anything is saved. Links are the weak spot of OCR (l/1, O/0), which is another
reason for that review. Pillow, numpy and rapidocr are imported only where used, so the splitting logic and its tests
need none of them.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from statistics import median
from typing import List, Optional, Sequence

MAX_IMAGES = 6                       # per upload
MAX_IMAGE_BYTES = 5 * 1024 * 1024    # per image
MAX_PIXELS = 40_000_000              # refuse picture bombs before decoding
MAX_SIDE = 2600                      # larger pictures are shrunk: OCR time and memory grow with size
MAX_MESSAGES_PER_IMAGE = 40
MIN_CONF = 0.45                      # OCR lines it is not sure about are dropped
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


class ImageRejected(ValueError):
    """The upload is not a usable screenshot (the text is safe to show the user)."""


@dataclass(frozen=True)
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0


@dataclass
class Bubble:
    text: str
    direction: str          # "in" = received, "out" = looks sent by the person who took the screenshot
    x0: float
    y0: float
    x1: float
    y1: float


# ---------------------------------------------------------------------------------------------- picture and OCR
def load_image(data: bytes):
    """Bytes of an upload -> RGB numpy array. Raises ImageRejected with a message fit for the user."""
    from PIL import Image, ImageOps
    import numpy as np

    try:
        im = Image.open(io.BytesIO(data))
        fmt, (w, h) = im.format, im.size
    except Exception:                                                   # noqa: BLE001
        raise ImageRejected("This file is not a picture.") from None
    if fmt not in ALLOWED_FORMATS:
        raise ImageRejected("Only PNG, JPEG or WebP screenshots are accepted.")
    if w * h > MAX_PIXELS:
        raise ImageRejected("This picture is too large.")
    try:
        im = ImageOps.exif_transpose(im)             # phone photos carry their rotation in the metadata
        im.load()
    except Exception:                                                   # noqa: BLE001
        raise ImageRejected("This picture could not be opened.") from None
    im = im.convert("RGB")
    if max(im.size) > MAX_SIDE:
        scale = MAX_SIDE / max(im.size)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    return np.asarray(im)


class Reader:
    """RapidOCR (ONNX models bundled in its wheel). One instance, used by one thread at a time."""

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR
        self._engine = RapidOCR()

    def lines(self, image) -> List[Line]:
        result, _ = self._engine(image, return_word_box=True)       # a box for every character, see respace()
        out: List[Line] = []
        for item in result or []:
            box, text, conf = item[0], unicodedata.normalize("NFKC", str(item[1])), float(item[2])
            if len(item) >= 5:
                text = respace(image, text, item[3], item[4])
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            out.append(Line(text, float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)), conf))
        return out


# ---------------------------------------------------------------------------------------------- spaces
# The recognizer often leaves out the space between two words ("unlockgreatr3wardsbefore") even though the picture has
# a clear gap. The positions it reports for each character are only estimates, and gaps measured from them are too noisy
# (they split words like "balan ce"), so the gap is measured in the PIXELS instead: a real space is a blank vertical
# strip across the line (no text-coloured pixel in the middle rows, which also keeps link underlines out of the way),
# while letters inside a word leave at most a thin sliver. The character positions only decide WHERE in the text the
# space goes. Spaces the recognizer did find are always kept.
MIN_SPACE_PX = 4
MIN_SPACE_FRAC = 0.27                 # of the line height: real spaces measure about 0.32, gaps inside words 0.1
MIN_CAMEL_FRAC = 0.20                 # lower-case letter then capital ("ReplyYES"): almost never one word, so a smaller gap is enough


def _blank_runs(image, x0: int, x1: int, y0: int, y1: int):
    """(start, end) columns, relative to x0, of the blank strips strictly inside the text of one line."""
    import numpy as np

    H, W = image.shape[0], image.shape[1]
    h = y1 - y0
    ry0, ry1 = max(0, int(y0 + 0.05 * h)), min(H, int(y0 + 0.80 * h))     # capitals' bars to the baseline, above the underline
    x0, x1 = max(0, x0), min(W, x1)
    if ry1 - ry0 < 3 or x1 - x0 < 8:
        return []
    crop = image[ry0:ry1, x0:x1].astype(np.float32)
    if crop.ndim == 2:
        crop = crop[:, :, None]
    background = np.median(crop.reshape(-1, crop.shape[2]), axis=0)
    dist = np.abs(crop - background).sum(axis=2)
    peak = float(np.percentile(dist, 99))
    if peak < 40:                                        # no contrast: nothing to measure
        return []
    ink = (dist > 0.35 * peak).any(axis=0)
    cols = np.flatnonzero(ink)
    if len(cols) < 2:
        return []
    need = max(MIN_SPACE_PX, int(round(MIN_CAMEL_FRAC * h)))       # the smallest gap that can ever be a space
    runs, start = [], None
    for x in range(int(cols[0]), int(cols[-1]) + 1):
        if not ink[x]:
            start = x if start is None else start
        elif start is not None:
            if x - start >= need:
                runs.append((start, x))
            start = None
    return runs


def respace(image, text: str, boxes, chars) -> str:
    """Put back the spaces the OCR dropped. Any doubt about the data -> the text unchanged."""
    try:
        if not boxes or not chars or len(boxes) != len(chars) or "".join(chars).replace(" ", "") != text.replace(" ", ""):
            return text
        spans = [(min(q[0] for q in b), max(q[0] for q in b), min(q[1] for q in b), max(q[1] for q in b)) for b in boxes]
        letters = [(c, s) for c, s in zip(chars, spans) if c != " "]
        if len(letters) < 2:
            return text
        x0 = int(min(s[0] for _, s in letters))
        x1 = int(max(s[1] for _, s in letters)) + 1
        y0 = int(min(s[2] for _, s in letters))
        y1 = int(max(s[3] for _, s in letters)) + 1
        runs = _blank_runs(image, x0, x1, y0, y1)
        line_h = max(1, y1 - y0)
    except (TypeError, IndexError, ValueError):
        return text
    centers = [(s[0] + s[1]) / 2 for _, s in letters]
    # where the OCR already put a space: before letter number k
    had, k = set(), 0
    for c in chars:
        if c == " ":
            had.add(k)
        else:
            k += 1
    add = set()
    for a, b in runs:
        xg = x0 + (a + b) / 2
        k = sum(1 for cx in centers if cx < xg)         # the gap lies before letter number k
        if 0 < k < len(letters) and not ({k - 1, k, k + 1} & had):
            before, after = letters[k - 1][0], letters[k][0]
            camel = before.islower() and after.isupper() and not any(c in "/@" for c, _ in letters)   # not inside a link
            if (b - a) < max(MIN_SPACE_PX, (MIN_CAMEL_FRAC if camel else MIN_SPACE_FRAC) * line_h):
                continue
            # '.' and ',' sit on the baseline below the rows that are measured, so a gap next to one is not a space,
            # unless an uppercase word follows ("Juan. Balance")
            if after in ".," or (before in ".," and not after.isupper()):
                continue
            add.add(k)
    # a space that would leave one stray letter ("PANAL O", "CASHOU T") is almost always a wide capital, not a word:
    # only 'a' and 'I' are one-letter words
    for k in sorted(add):
        cuts = sorted(had | add)
        left = max([0] + [c for c in cuts if c < k])
        right = min([len(letters)] + [c for c in cuts if c > k])
        for word in (letters[left:k], letters[k:right]):
            chars_only = [c for c, _ in word if c.isalnum()]          # "O!" is the single letter O
            if len(chars_only) == 1 and chars_only[0].isalpha() and chars_only[0] not in "aAI":
                add.discard(k)
    out: List[str] = []
    for k, (c, _) in enumerate(letters):
        if k in had or k in add:
            out.append(" ")
        out.append(c)
    return "".join(out).strip()


# ---------------------------------------------------------------------------------------------- phone chrome
# The OCR often drops the spaces of small grey text ("TextMessage SMS", "Sun,Oct11at12:11AM") and reads the bullet
# of "Text Message • SMS" as a replacement character, so all these checks look at the text COMPACTED: lower case,
# without spaces, bullets or apostrophes.
_COMPACT_STRIP = re.compile(r"[\s\ufffd\u00b7\u2022'\u2019\u02bc`]+")
_CLOCK = re.compile(r"^\d{1,2}[:.]\d{2}(\s?[ap]\.?m\.?)?$", re.I)
_PROMPT_START = ("textmessage", "imessage", "rcsmessage", "chatmessage", "typeamessage", "message")
_DELIVERY = re.compile(r"^(delivered|read|sent|seen|sending|notdelivered|failed)")
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DATEISH = re.compile(
    r"^(mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?(?=[a-z0-9])"
    rf"|^({_MONTHS})[a-z]*\.?\d{{1,2}}"
    rf"|^\d{{1,2}}({_MONTHS})[a-z]*"
    r"|^\d{1,2}/\d{1,2}"
)
_LONE_STAMP = re.compile(r"^(\d{1,2}/\d{1,2}(/\d{2,4})?)?\d{1,2}:\d{2}(am|pm)?$")     # '2/38:27PM' under a bubble
_UNREAD = ("unread", "unreadmessages", "unreadmessage", "newmessages", "newmessage")
# phrases that are the app's own, never part of a message: Google Messages' preview button and "can't reply" panel
_LINE_NOTICES = ("taptoloadpreview", "sendercantacceptreplies", "acceptreplies", "contactthemdirectly")
# system notices a phone shows inside the conversation (iPhone's unknown-sender banner and its buttons)
_NOTICE_WORDS = ("unknownsender", "didnotexpect", "markasknown", "reportjunk", "blockthiscaller")
_LINKISH = re.compile(r"[A-Za-z0-9-]\.[A-Za-z]{2,}|/|@")


def _compact(text: str) -> str:
    return _COMPACT_STRIP.sub("", text.lower())


def _is_centered(ln: Line, width: float) -> bool:
    return abs((ln.x0 + ln.x1) / 2 - width / 2) < 0.10 * width


def _is_chrome(ln: Line, width: float, height: float) -> bool:
    """Status bar, app-bar title, clock, date separators, 'Delivered', the message input box and its buttons."""
    t = ln.text.strip()
    c = _compact(t)
    if len(c) <= 1 or not re.search(r"[A-Za-z0-9]", t):         # '+', icons, stray marks
        return True
    if ln.y1 < 0.06 * height:                                  # status bar
        return True
    if ln.y1 < 0.18 * height and len(t) <= 24 and not _LINKISH.search(t):   # contact name / phone number in the app bar
        return True
    if _CLOCK.match(t) or _LONE_STAMP.match(c):
        return True
    if ln.y0 > 0.75 * height and len(c) <= 24 and c.startswith(_PROMPT_START):      # the message input box
        return True
    if len(c) <= 22 and _DELIVERY.match(c):
        return True
    if any(w in c for w in _LINE_NOTICES) or (c.endswith("learnmore") and len(c) <= 28):
        return True
    return _is_separator(ln, width)


def _is_separator(ln: Line, width: float) -> bool:
    """A centered grey line between messages: 'Today 3:12 PM', 'Text Message - SMS', 'Mon, Sep 21 at 11:36 AM', 'Unread'."""
    c = _compact(ln.text)
    if not (_is_centered(ln, width) and len(c) <= 40):
        return False
    if c in _UNREAD or c.startswith(("today", "yesterday", "now", "textmessage", "imessage", "rcsmessage")):
        return True
    return bool(_DATEISH.match(c)) and any(ch.isdigit() for ch in c)


def _is_notice(text: str, x0: float, x1: float, width: float) -> bool:
    c = _compact(text)
    return len(c) <= 160 and abs((x0 + x1) / 2 - width / 2) < 0.10 * width and any(w in c for w in _NOTICE_WORDS)


# ---------------------------------------------------------------------------------------------- links across lines
_TLD = None


def _known_suffix(host: str) -> bool:
    global _TLD
    if _TLD is None:
        import tldextract
        _TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
    return bool(_TLD(host).suffix)


def _slash_misread(token: str) -> bool:
    """'bit.lyl' or 'https://cutt.lyl': the '/' that ends a wrapped link was read as l, I, 1 or |."""
    t = re.sub(r"(?i)^https?://", "", token)
    if len(t) < 5 or t[-1] not in "lI1|":
        return False
    base = t[:-1]
    return bool(re.fullmatch(r"[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+", base)) and _known_suffix(base) and not _known_suffix(t)


_LINK_CONTINUATION = frozenset("-./?=&_%#")


def _linkish_token(token: str) -> bool:
    return "://" in token or "www." in token.lower() or bool(re.search(r"[A-Za-z0-9-]\.[A-Za-z]{2,}", token))


_URL_CHARS = re.compile(r"^[A-Za-z0-9\-._~:/?#@!$&=+%]+$")


def _looks_like_link_piece(token: str) -> bool:
    """A long URL-looking token (a scheme, a path or www.), not a word that happens to contain a dot or a comma."""
    return (len(token) >= 14 and bool(_URL_CHARS.match(token))
            and ("://" in token or "/" in token or token.lower().startswith("www.")))


_ENDS_WITH_SCHEME = re.compile(r"(?i)\bhttps?\s*:\s*[/\\lI|1]{0,2}$")      # "https:" "https:/" "https://" or the OCR's "https:/l"


def _join(lines: Sequence[Line], right_edge: float, width: float) -> str:
    """Join the wrapped lines of one bubble. A link that was cut across two lines is rejoined without a space."""
    text = lines[0].text.strip()
    for prev, ln in zip(lines, lines[1:]):
        last = text.split(" ")[-1] if text else ""
        glue = " "
        if _ENDS_WITH_SCHEME.search(text):                                    # "https://" | "3go.run/324h"
            glue = ""
        elif last.endswith("/") and ("." in last or "http" in last.lower()):  # "bit.ly/" | "ubrewardscard"
            glue = ""
        elif _slash_misread(last):                                            # "bit.lyl" | "ubrewardscard"
            text, glue = text[:-1] + "/", ""
        elif ln.text.lstrip()[:1] in _LINK_CONTINUATION and _linkish_token(last):   # "https://bdo" | "-on.cc"
            glue = ""
        elif last.endswith("-") and len(last) >= 3 and ln.text.lstrip()[:1].isalnum() and ln.text.lstrip()[:1].islower():
            glue = ""                                                         # "pre-" | "approved": no space after a hyphen break
        elif _looks_like_link_piece(last) and prev.x1 >= right_edge - 0.015 * width:   # cut by the bubble's edge
            glue = ""
        text += glue + ln.text.strip()
    text = re.sub(r"(?i)\b(https?)\s*:\s*/\s*[/lI|1]?\s*", r"\1://", text)   # OCR spaces out the scheme or reads a slash as l
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# ---------------------------------------------------------------------------------------------- bubbles
def _pixel(image, x: int, y: int):
    h, w = image.shape[0], image.shape[1]
    x, y = min(max(x, 1), w - 2), min(max(y, 1), h - 2)
    patch = image[y - 1:y + 2, x - 1:x + 2].reshape(-1, image.shape[2] if image.ndim == 3 else 1)
    return [float(v) for v in patch.mean(axis=0)]


def _differs(a, b, tol: float = 14.0) -> bool:
    return sum(abs(x - y) for x, y in zip(a, b)) > tol


def _inside_one_bubble(image, a: Line, b: Line, h: float) -> bool:
    """True if the strip between two text lines is still the bubble's colour (a blank line inside one message), and
    False if it is the page background (two separate messages). Sampled in the bubble's left padding, three times."""
    x = int(max(2, min(a.x0, b.x0) - 0.4 * h))
    for frac in (0.25, 0.5, 0.75):
        y = int(a.y1 + (b.y0 - a.y1) * frac)
        c = _pixel(image, x, y)
        if not (_differs(c, _pixel(image, 2, y)) and _differs(c, _pixel(image, image.shape[1] - 3, y))):
            return False
    return True


def split_messages(lines: Sequence[Line], width: float, height: float, image=None) -> List[Bubble]:
    """`image` (the pixels, optional) lets two text blocks with a gap between them be told apart: one bubble with a blank
    line in it, or two bubbles."""
    kept = [ln for ln in lines if ln.conf >= MIN_CONF and ln.text.strip() and not _is_chrome(ln, width, height)]
    separators = [ln.y0 for ln in lines if ln.conf >= MIN_CONF and _is_separator(ln, width)]
    if separators:                    # a short link-less line above the first separator is the contact name or a phone number
        top = min(separators)
        kept = [ln for ln in kept if not (ln.y1 <= top and len(ln.text.strip()) <= 24 and not _LINKISH.search(ln.text))]
    if not kept:
        return []
    kept.sort(key=lambda ln: (ln.y0, ln.x0))
    h = median(ln.y1 - ln.y0 for ln in kept) or 1.0            # typical text height: all spacing is judged against it

    groups: List[List[Line]] = [[kept[0]]]
    for ln in kept[1:]:
        prev = groups[-1][-1]
        gap = ln.y0 - prev.y1
        # lines of one bubble share a left edge (or, for centered notices, a middle); a different column is the other
        # side of the conversation
        aligned = abs(ln.x0 - prev.x0) <= 1.2 * h or abs((ln.x0 + ln.x1) / 2 - (prev.x0 + prev.x1) / 2) <= 0.04 * width
        far = gap > 0.8 * h
        if far and aligned and image is not None and gap < 5 * h and _inside_one_bubble(image, prev, ln, h):
            far = False
        if far or (not aligned and gap > -0.3 * h):
            groups.append([ln])
        else:
            groups[-1].append(ln)

    bubbles: List[Bubble] = []
    for g in groups[:MAX_MESSAGES_PER_IMAGE]:
        x0, x1 = min(x.x0 for x in g), max(x.x1 for x in g)
        y0, y1 = min(x.y0 for x in g), max(x.y1 for x in g)
        outgoing = x1 > 0.88 * width and x0 > 0.15 * width     # right-aligned bubble = the screenshot owner's own reply
        bubbles.append(Bubble(_join(g, x1, width), "out" if outgoing else "in", x0, y0, x1, y1))
    return [b for b in bubbles if b.text and not _is_notice(b.text, b.x0, b.x1, width)]


def read_screenshot(reader: Reader, data: bytes) -> List[Bubble]:
    """One upload -> its bubbles, top to bottom. Raises ImageRejected."""
    image = load_image(data)
    height, width = image.shape[0], image.shape[1]
    return split_messages(reader.lines(image), float(width), float(height), image)
