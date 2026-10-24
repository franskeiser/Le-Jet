"""
dataset.py -- the shared submissions CSV of the website: 50 columns.

One row per submitted SMS: 6 message columns, 5 link columns, 2 inspection columns and the 37 link/page features. These
are the crowd study's 57 columns (src/data/build_crowd_dataset.COLUMNS) minus seven the website does not collect or
need: category, sender_type, date_received, label_source, consent, inspected_at, hours_to_inspect
(tests/test_webapp_dataset.py checks that the two layouts differ by exactly those seven). Missing values are -1, never
blank. The row rules of that builder (one link domain at most 20 times per class, second-person label check, number
masking) are NOT applied here: this file keeps everything that was submitted.

A file in the previous 57-column layout is converted when the site starts (a copy is kept next to it).

Helpers that exist in the builder (template_id, link_shortened, fetch_status) are copied here because the builder needs
pandas and the VM does not have it; the test compares them with the builder's.

No third-party imports: standard library + the lab's own modules.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from lejet import safety
from src.features.link_features import SHORTENERS, extract_links, normalize_link

MISSING = -1
LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=8))          # Philippine time, as the builder uses
MAX_TEXT = 2000

MESSAGE_COLUMNS = ["message_id", "sms_text", "label", "language", "date_submitted", "label_verified_by"]
LINK_COLUMNS = ["link", "final_url", "link_domain", "template_id", "link_shortened"]
INSPECTION_COLUMNS = ["page_fetched", "fetch_status"]
REMOVED_COLUMNS = ("category", "sender_type", "date_received", "label_source", "consent", "inspected_at", "hours_to_inspect")
# src/unified/prepare_web_37.FEATURES_37 (the test checks this copy against it)
FEATURES_37 = [
    "length_url", "length_hostname", "nb_dots", "nb_slash", "nb_www", "https_token", "ratio_digits_url",
    "ratio_digits_host", "prefix_suffix", "shortening_service", "nb_redirection",
    "length_words_raw", "char_repeat", "shortest_word_host", "shortest_word_path", "longest_words_raw",
    "longest_word_host", "avg_word_host", "phish_hints", "domain_in_brand",
    "nb_hyperlinks", "ratio_intHyperlinks", "nb_extCSS", "login_form", "external_favicon", "links_in_tags",
    "ratio_intMedia", "ratio_extMedia", "iframe", "safe_anchor", "empty_title", "domain_in_title",
    "domain_with_copyright",
    "whois_registered_domain", "domain_registration_length", "domain_age", "page_rank",
]
COLUMNS = MESSAGE_COLUMNS + LINK_COLUMNS + INSPECTION_COLUMNS + FEATURES_37
assert len(COLUMNS) == 50 and len(set(COLUMNS)) == 50
# the layout this site wrote before the seven columns were dropped: such a file is converted on start-up
LEGACY_COLUMNS_57 = (["message_id", "sms_text", "label", "category", "language", "sender_type", "date_received", "date_submitted",
                      "label_source", "label_verified_by", "consent"] + LINK_COLUMNS
                     + ["inspected_at", "hours_to_inspect", "page_fetched", "fetch_status"] + FEATURES_37)
assert len(LEGACY_COLUMNS_57) == 57

LABELS = ("smishing", "legitimate")
LANGUAGES = ("en", "tl", "taglish")                     # English, Tagalog, Taglish (the builder's codes)
# What the page lists for each row (the full text is in the download).
VIEW_COLUMNS = ["message_id", "date_submitted", "label", "language", "link_domain", "fetch_status"]


# ---------------------------------------------------------------------------------------------- text and link
def clean_text(text: str) -> str:
    """The SMS as typed, made inert: line breaks kept, other control characters and bidi tricks removed."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    lines = [safety.strip_dangerous_chars(line) for line in text.split("\n")]
    return "\n".join(lines).strip()[:MAX_TEXT]


def first_link(text: str) -> Optional[Tuple[str, str]]:
    """(raw link as written, normalised link the lab visits) -- the first link only (collection rule 7)."""
    links = extract_links(text)
    if not links:
        return None
    raw = links[0]["raw"]
    return raw, normalize_link(raw)


def defang_in_text(text: str) -> str:
    """The SMS with every link in it made un-clickable (for showing on the page)."""
    out, pos = [], 0
    for link in extract_links(text):
        out.append(text[pos:link["start"]])
        out.append(safety.defang_url(text[link["start"]:link["end"]]))
        pos = link["end"]
    out.append(text[pos:])
    return "".join(out)


def template_key(text: str) -> str:
    for link in reversed(extract_links(text)):
        text = text[:link["start"]] + " <url> " + text[link["end"]:]
    text = re.sub(r"\d[\d,.]*", "0", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def template_id(text: str) -> str:
    return "t" + hashlib.sha256(template_key(text).encode("utf-8")).hexdigest()[:8]


def is_shortened(link: str) -> int:
    links = extract_links(link)
    host = links[0]["host"] if links else ""
    return int(host.removeprefix("www.") in SHORTENERS)


def fetch_status(fail_reason: str, final_differs: Any) -> str:
    """live, redirected, dead, blocked or timeout, from the lab's fail_reason (as in the builder)."""
    if fail_reason == "none":
        return "redirected" if final_differs else "live"
    if "timeout" in fail_reason:
        return "timeout"
    if fail_reason in ("rejected_url", "port_blocked", "redirect_to_blocked", "challenge_page", "blocked_or_empty"):
        return "blocked"
    return "dead"


# ---------------------------------------------------------------------------------------------- one row
def build_row(message_id: str, text: str, label: str, language: str, raw_link: str, lab: Dict[str, Any],
              full_features: Dict[str, Any], submitted: datetime.datetime) -> Dict[str, Any]:
    """`lab` is a cleaned lab row (schema.clean_row), `full_features` is schema.full_features(lab)."""
    if label not in LABELS or language not in LANGUAGES:
        raise ValueError(f"label {label!r} / language {language!r} are not allowed")
    row: Dict[str, Any] = {
        "message_id": message_id,
        "sms_text": text,
        "label": label,
        "language": language,
        "date_submitted": submitted.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
        "label_verified_by": MISSING,           # a second person checks the label later
        "link": raw_link,
        "final_url": lab["final_url"] if lab.get("final_status", MISSING) != MISSING and lab.get("final_url") else MISSING,
        "link_domain": lab.get("final_reg_domain") or lab.get("reg_domain") or MISSING,
        "template_id": template_id(text),
        "link_shortened": is_shortened(raw_link),
        "page_fetched": int(lab.get("page_fetched", 0)),
        "fetch_status": fetch_status(lab.get("fail_reason", "internal_error"), lab.get("final_differs")),
    }
    for name in FEATURES_37:
        row[name] = full_features.get(name, MISSING)       # page_rank is -1 unless the site looked it up (webapp/rank.py)
    assert list(row) == COLUMNS
    return row


def _cell(column: str, value: Any) -> Any:
    """Free text is made safe for spreadsheets (a leading = + - @ would run as a formula)."""
    if column in ("sms_text", "link", "final_url", "link_domain") and isinstance(value, str):
        return safety.csv_safe(value)
    return value


# ---------------------------------------------------------------------------------------------- the file
class Store:
    """The CSV on disk plus what the page needs in memory. NOT thread-safe: the app calls it under one asyncio lock."""

    def __init__(self, path: str):
        self.path = path
        self._seen: Dict[Tuple[str, str], str] = {}          # (sms_text, link) as stored -> message_id
        self._view: List[Dict[str, Any]] = []
        self._next = 1
        self.counts: Dict[str, int] = {label: 0 for label in LABELS}
        self.lang_counts: Dict[str, int] = {lang: 0 for lang in LANGUAGES}
        self._load()

    def _load(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow(COLUMNS)
            return
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            if header == COLUMNS:
                for r in reader:
                    self._remember(r)
                return
            if header != LEGACY_COLUMNS_57:
                raise RuntimeError(f"{self.path} does not have the {len(COLUMNS)}-column layout; move it away or fix it")
            legacy_rows = list(reader)
        # an old 57-column file: keep a copy, then rewrite it without the dropped columns (language of old rows stays -1)
        backup = self.path + ".57cols.bak"
        if os.path.exists(backup):
            backup = f"{self.path}.57cols.{int(time.time())}.bak"
        shutil.copy2(self.path, backup)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(legacy_rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        for r in legacy_rows:
            self._remember(r)

    def _remember(self, r: Dict[str, Any]) -> None:
        self._seen[(r["sms_text"], r["link"])] = r["message_id"]
        self._view.append({c: r[c] for c in VIEW_COLUMNS} | {"sms_text": r["sms_text"], "link": r["link"]})
        self.counts[r["label"]] = self.counts.get(r["label"], 0) + 1
        if r.get("language") in self.lang_counts:
            self.lang_counts[r["language"]] += 1
        m = re.fullmatch(r"web_(\d+)", r["message_id"])
        if m:
            self._next = max(self._next, int(m.group(1)) + 1)

    def next_id(self) -> str:
        return f"web_{self._next:06d}"

    def existing(self, text: str, raw_link: str) -> Optional[str]:
        """message_id of an identical earlier submission (same text and same link), else None."""
        return self._seen.get((_cell("sms_text", text), _cell("link", raw_link)))

    def append(self, row: Dict[str, Any]) -> None:
        stored = {c: _cell(c, row[c]) for c in COLUMNS}
        with open(self.path, "a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n").writerow(stored)
            fh.flush()
            os.fsync(fh.fileno())
        self._remember({c: str(v) for c, v in stored.items()})

    def __len__(self) -> int:
        return len(self._view)

    def newest(self, limit: int) -> List[Dict[str, Any]]:
        out = []
        for v in reversed(self._view[-limit:]):
            out.append({**{c: v[c] for c in VIEW_COLUMNS},
                        "sms_preview": defang_in_text(v["sms_text"])[:240]})
        return out
