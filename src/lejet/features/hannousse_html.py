"""
hannousse_html.py -- the content-based features of the public web-phishing set (Hannousse & Yahiouche, 2021,
Table 3) that the original 12 webpage features do not cover and that the lab exports.

The content columns the lab exports, in the public set's order, are CONTENT_FEATURES (16 of its 24). Twelve of
them are the existing `html_features.WEBPAGE_FEATURES`; this module adds the other four:

    popup_window   safe_anchor   onmouseover   right_clic

Eight public-set content columns are NOT exported (schema.DROPPED_FROM_FULL): ratio_intRedirection,
ratio_extRedirection, ratio_intErrors and ratio_extErrors (they need a request to every hyperlink, and the lab
never fetches a linked resource); ratio_extHyperlinks (exactly 1 - ratio_intHyperlinks in the lab); and
ratio_nullHyperlinks, submit_email and sfh (0 in every row of the public set, so they cannot be evaluated).

How it stays safe: `collect_facts()` is called from the one walk `html_features.extract_features` already makes over
the parsed page, so the page is parsed once. Script text is only ever searched with plain substring tests on a
capped, whitespace-stripped copy (no regular expressions on attacker-controlled text), and nothing is executed or
fetched.

The public set contains no pages, so unlike the URL features these four cannot be checked against it row by row;
tests/test_hannousse_html.py pins the definitions. See docs/hannousse_87_features.md.
"""

from __future__ import annotations

from typing import Any, Dict, List

# The content columns of the public set that the lab exports, in its order.
CONTENT_FEATURES: List[str] = [
    "nb_hyperlinks", "ratio_intHyperlinks", "nb_extCSS", "login_form", "external_favicon", "links_in_tags",
    "ratio_intMedia", "ratio_extMedia", "iframe", "popup_window", "safe_anchor", "onmouseover", "right_clic",
    "empty_title", "domain_in_title", "domain_with_copyright",
]
# The four this module produces (the other twelve come from html_features.extract_features).
EXTRA_CONTENT_FEATURES = ["popup_window", "safe_anchor", "onmouseover", "right_clic"]

MAX_JS_CHARS = 256 * 1024          # all inline script text together
MAX_JS_PER_SCRIPT = 32 * 1024
_TEXT_INPUT_TYPES = {"text", "email", "password", "tel", "search", "number", "url"}


def collect_facts(el, tag: str, facts: Dict[str, Any]) -> None:
    """Called once per element during the existing walk. Never raises on odd attributes."""
    if tag == "a":
        href = el.get("href")
        if href is not None:
            facts["anchors"] += 1
            low = href.lower()
            # paper: an anchor to '#', 'javascript' or 'mailto' goes nowhere real ("unsafe")
            if "#" in low or "javascript" in low or "mailto" in low:
                facts["unsafe_anchors"] += 1
    elif tag == "input":
        if (el.get("type") or "text").strip().lower() in _TEXT_INPUT_TYPES:
            facts["text_inputs"] += 1
    elif tag == "script" and el.get("src") is None:
        if facts["js_chars"] < MAX_JS_CHARS:
            text = (el.text or "")[:MAX_JS_PER_SCRIPT].lower()
            facts["js_parts"].append(text)
            facts["js_chars"] += len(text)
    attrs = el.attrib
    if attrs:
        if "onmouseover" in attrs:
            facts["onmouseover_attr"] = True
        for name, value in attrs.items():
            if name.startswith("on") and value and _has_event_button_2("".join(value.lower().split())[:2048]):
                facts["event_button_attr"] = True
                break


def _has_event_button_2(compact: str) -> bool:
    return "event.button==2" in compact or "event.button===2" in compact


def extra_features(facts: Dict[str, Any]) -> Dict[str, float]:
    """The four extra content features from the facts collected during the walk."""
    out: Dict[str, float] = {}
    anchors = facts.get("anchors", 0)
    out["safe_anchor"] = round(100.0 * facts.get("unsafe_anchors", 0) / anchors, 6) if anchors else 0.0
    js = "".join("".join(facts.get("js_parts", [])).split())
    out["popup_window"] = int("prompt(" in js or ("window.open(" in js and facts.get("text_inputs", 0) > 0))
    out["onmouseover"] = int(bool(facts.get("onmouseover_attr")))
    out["right_clic"] = int(bool(facts.get("event_button_attr")) or _has_event_button_2(js))
    return {name: out[name] for name in EXTRA_CONTENT_FEATURES}
