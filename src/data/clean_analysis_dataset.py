r"""Clean data/raw/sms/analysisdataset.csv into the labelled SMS-text dataset (words only, links cut out).

The raw file is `label,message` (561 spam, 312 ham, Filipino / Taglish / English). One row per message comes out with:

    message_id              an_000001 ... in file order after cleaning
    sms_text                the message with every link removed, whitespace and escape sequences tidied (what the text model reads)
    sms_text_original       the message as received, whitespace tidied only (nothing is lost)
    label                   smishing / legitimate (the study's labels; LABEL_TO_IDX in src/features/model/train.py)
    label_raw               spam / ham as in the file
    has_url, n_urls         whether and how many links the message held
    dup_after_link_removal  1 if another kept row has the same sms_text (same wording, different link): flagged, not dropped,
                            so a split can keep such rows in one fold

Label mapping: spam -> smishing, ham -> legitimate. The file calls the messages spam / ham; most of its spam is online-casino and
job-offer bait with a link, which is how this study uses "smishing". Say if the mapping should be different.

Cleaning steps, in order:
  0. Cells whose UTF-8 was read as cp1252 ("bÃ‰st" for "bÉst") are put right.
  1. Escape sequences written as text ("\n", "\\r\\n") and real line breaks become one space; non-breaking spaces too.
  2. Links glued to the word before them ("register!http://x.com", "Libre!th108.com") are pulled apart, and links split by the
     sender ("{9y15. com}", "w8bonu. com") are joined back, before detection (only for common link endings, and only a lowercase
     ending, so "now. Com..." stays text).
  3. Links are found with URLExtract. A candidate with a scheme, www. or a path, or an IP address, is a link. A bare domain
     ("11y.life") counts only if its ending is a common link ending (LINK_ENDINGS), written in lower or upper case, and it is not
     the domain of an email address ("x@gmail.com" stays). Anything else is a word ("la.st", "Amazon.com" mid-sentence is judged by
     the same rule: .com is a link ending, so it goes; see the summary's removed_bare_domains for what was cut).
  4. Links are removed, then the gaps tidied: empty brackets and braces, doubled punctuation, cut-off "https:" fragments.
     The words around a link are never rewritten.
  5. Rows with no letters or digits left (after dropping <TAGS> such as <REAL NAME>) or only "<<Content not supported.>>" (an MMS the
     phone could not show) are dropped, and so are exact duplicates of the
     original message (case and spacing ignored, first kept; no duplicate pair has conflicting labels, which is checked).
     The anonymisation tags themselves stay in the text.

Writes data/processed/analysis_text_clean.csv and analysis_text_clean.summary.json (every count, plus the bare domains removed).

Usage:
    python -m src.data.clean_analysis_dataset            # reads data/raw/sms/analysisdataset.csv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from urlextract import URLExtract

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "data" / "raw" / "sms" / "analysisdataset.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "analysis_text_clean.csv"
LABELS = {"spam": "smishing", "ham": "legitimate"}

# Endings a bare domain must have to count as a link. Short country-style endings that are also common letter pairs are left out.
LINK_ENDINGS = {"com", "net", "org", "info", "xyz", "life", "club", "top", "vip", "live", "win", "bet", "site", "online", "world",
                "shop", "store", "fun", "icu", "co", "ph", "buzz", "work", "link", "click", "app", "pro", "today", "games",
                "casino", "asia", "biz"}
_SPACED_ENDINGS = "com|net|org|info|xyz|life|club|top|vip|live|win|bet|site|online|world|shop"
_ENDINGS_RE = "|".join(sorted(LINK_ENDINGS, key=len, reverse=True))
# A link glued to the word before it ("register!http://x.com", "Libre!th108.com", "URL:kt777ad2.com") is invisible to URLExtract.
_GLUED_SCHEME = re.compile(r"(?<=\S)(?=https?://|www\.)", re.IGNORECASE)
_GLUED_DOMAIN = re.compile(rf"([!?,;:])(?=[\w-]+(?:\.[\w-]+)*\.(?:{_ENDINGS_RE})\b|[\w-]+(?:\.[\w-]+)+/)")
_PLACEHOLDER = re.compile(r"^\W*content not supported\W*$", re.IGNORECASE)       # an MMS the phone could not show: no text at all
_SPACED = re.compile(rf"([A-Za-z0-9-]{{2,}})\s*\.\s+(?-i:({_SPACED_ENDINGS}))\b")
_CLEAR_LINK = re.compile(r"^https?:|^www\.|/", re.IGNORECASE)
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_TAG = re.compile(r"<[A-Z ]+>")

_extractor = URLExtract()


def fix_mojibake(text: str) -> str:
    """A cell whose UTF-8 was read as cp1252 ("bÃ‰st" for "bÉst") is put right; any other text is returned unchanged."""
    if not re.search("Ã|â€|Â", text):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def squash(text: str) -> str:
    """Escape sequences written as text, real line breaks and odd spaces -> one space."""
    text = re.sub(r"(\\+[rn])+", " ", text)
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def find_links(text: str) -> list[str]:
    """Every link in `text` (already joined by join_split_domains), longest first."""
    links = []
    for url, (start, _) in _extractor.find_urls(text, get_indices=True):
        if _CLEAR_LINK.search(url) or _IPV4.match(url):
            links.append(url)
            continue
        ending = url.rstrip(".").rsplit(".", 1)[-1]
        in_email = start > 0 and text[start - 1] == "@"
        if ending.lower() in LINK_ENDINGS and (ending.islower() or ending.isupper()) and not in_email and "@" not in url:
            links.append(url)
    return sorted(set(links), key=len, reverse=True)       # longest first, so a short link inside a long one is not cut first


def join_split_domains(text: str) -> str:
    """Pull apart links glued to the word before them and put back links the sender split ("w8bonu. com")."""
    text = _GLUED_DOMAIN.sub(r"\1 ", _GLUED_SCHEME.sub(" ", text))
    return _SPACED.sub(r"\1.\2", text)


def remove_links(text: str, links: list[str]) -> str:
    for link in links:
        text = re.sub(re.escape(link), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"-*\bhttps?:/{0,2}", " ", text, flags=re.IGNORECASE)        # a link cut off in the data: no domain left
    text = re.sub(r"[\(\[\{<]\s*[\)\]\}>]", " ", text)                           # brackets and braces left empty
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)                                 # space before punctuation the link left behind
    text = re.sub(r"([,.;:!?])(\s*[,.;:!?])+", r"\1", text)                      # doubled punctuation
    return squash(text)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    summary = {"rows_in": len(raw), "labels_in": raw["label"].value_counts().to_dict()}
    unknown = set(raw["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"labels other than spam / ham: {sorted(unknown)}")
    original = raw["message"].astype(str).map(fix_mojibake).map(squash)
    joined = original.map(join_split_domains)
    links = [find_links(t) for t in joined]
    summary["removed_bare_domains"] = dict(Counter(l for ls in links for l in ls if not _CLEAR_LINK.search(l) and not _IPV4.match(l)))
    out = pd.DataFrame({
        "sms_text": [remove_links(t, l) for t, l in zip(joined, links)],
        "sms_text_original": original,
        "label": raw["label"].map(LABELS),
        "label_raw": raw["label"],
        "has_url": [int(bool(l)) for l in links],
        "n_urls": [len(l) for l in links],
    })
    no_content = out["sms_text"].map(lambda s: not re.search(r"[A-Za-z0-9]", _TAG.sub("", s)) or bool(_PLACEHOLDER.match(s)))
    summary["dropped_no_letters_or_digits_left"] = int(no_content.sum())
    summary["dropped_rows_were_labelled"] = out.loc[no_content, "label_raw"].value_counts().to_dict()
    out = out[~no_content]
    dup = out["sms_text_original"].str.casefold().duplicated()
    conflicting = (out.assign(k=out["sms_text_original"].str.casefold()).groupby("k")["label"].nunique() > 1).sum()
    if conflicting:
        raise ValueError(f"{conflicting} messages appear with both labels: decide which label is right before cleaning")
    summary["dropped_exact_duplicate_original"] = int(dup.sum())
    out = out[~dup].reset_index(drop=True)
    out.insert(0, "message_id", [f"an_{i:06d}" for i in range(1, len(out) + 1)])
    out["dup_after_link_removal"] = out["sms_text"].duplicated(keep=False).astype(int)
    summary.update(rows_out=len(out), labels_out=out["label"].value_counts().to_dict(),
                   with_url=out.groupby("label")["has_url"].sum().to_dict(),
                   without_url=out.groupby("label")["has_url"].apply(lambda s: int((s == 0).sum())).to_dict(),
                   flagged_same_text_different_link=int(out["dup_after_link_removal"].sum()),
                   median_chars_clean=float(out["sms_text"].str.len().median()))
    return out, summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="source", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    out, summary = clean(pd.read_csv(args.source))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "removed_bare_domains"}, indent=2))
    print(f"bare domains removed: {len(summary['removed_bare_domains'])} distinct; wrote {args.out}")


if __name__ == "__main__":
    main()
