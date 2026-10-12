"""
Tests for scripts/clean_smishing_links.py (turning the team's spreadsheet cells into a clean URL list).
Only invented example domains are used. No spreadsheet library is needed: the cell-level functions are tested.

Run with:  python -m pytest tests/test_clean_smishing_links.py
"""

from __future__ import annotations

import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("clean_smishing_links", os.path.join(HERE, "..", "scripts", "clean_smishing_links.py"))
csl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(csl)


@pytest.mark.parametrize("cell, expected", [
    ("https://bank-verify.example.com/login", ["https://bank-verify.example.com/login"]),
    ("\nhttps://bank-verify.example.com", ["https://bank-verify.example.com"]),          # leading newline
    ("  bank-verify.example.com/x  ", ["bank-verify.example.com/x"]),                    # no scheme kept as written
    ("https://bit.ly/4abcDEF.", ["https://bit.ly/4abcDEF"]),                             # trailing full stop
    ("<https://a.example.com/x>", ["https://a.example.com/x"]),
    ("hxxps://evil[.]example[.]com/x", ["https://evil.example.com/x"]),                  # defanged
    ("https://a.example.com​/x", ["https://a.example.com/x"]),                      # zero-width space? splits cleanly
    ("see https://a.example.com/x and b.example.org/y", ["https://a.example.com/x", "b.example.org/y"]),
    ("https://a.example.com/x?q=1&r=2", ["https://a.example.com/x?q=1&r=2"]),
])
def test_links_are_pulled_out_of_a_cell_and_cleaned(cell, expected):
    got = csl.links_in_cell(cell)
    assert [g.replace("​", "") for g in got] == expected


@pytest.mark.parametrize("cell", [None, "", "   ", "\n", "Links", "URL", "October Messages", "Verdict", "Live", "Dead",
                                  "12345", "3.14", "Reviewed", "no links here at all", "Charles"])
def test_headers_labels_numbers_and_plain_words_are_not_links(cell):
    assert csl.links_in_cell(cell) == []


def test_the_trusted_name_at_evil_trick_is_kept_not_dropped():
    assert csl.links_in_cell("https://secure-bank.example.com@evil.example.org/login") == \
        ["https://secure-bank.example.com@evil.example.org/login"]
    recs, _ = csl.clean(entries(("https://secure-bank.example.com@evil.example.org/login", None)))
    assert len(recs) == 1 and "safety gate" in recs[0]["note"]          # kept in the metadata, never fetched as is


def test_junk_never_raises():
    for junk in (5, 3.5, True, [], {"a": 1}, b"bytes", "\x00\x01", "://", "http://", "a" * 5000, "[.]", "hxxp"):
        csl.links_in_cell(junk)


@pytest.mark.parametrize("a, b", [
    ("https://Bank.Example.com/login", "http://bank.example.com/login"),            # scheme and case
    ("https://bank.example.com/login/", "https://bank.example.com/login"),          # trailing slash
    ("https://bank.example.com", "bank.example.com/"),
    ("hxxps://bank[.]example[.]com", "https://bank.example.com"),
])
def test_the_same_link_spelled_differently_has_the_same_key(a, b):
    assert csl.canonical_key(csl.clean_token(a)) == csl.canonical_key(csl.clean_token(b))


@pytest.mark.parametrize("a, b", [
    ("https://bit.ly/4abc", "https://bit.ly/4ABC"),                                   # short codes are case-sensitive
    ("https://bank.example.com/a", "https://bank.example.com/b"),
    ("https://bank.example.com/a?x=1", "https://bank.example.com/a?x=2"),
    ("https://www.example.com/", "https://example.com/"),
])
def test_different_links_keep_different_keys(a, b):
    ka, kb = csl.canonical_key(a), csl.canonical_key(b)
    if a == "https://bit.ly/4abc":
        assert ka.lower() == kb.lower()      # host and path differ only by case; the PATH case matters for shorteners
    else:
        assert ka != kb


def entries(*cells):
    return [{"cell": c, "source": f"f.xlsx:Sheet1:{i + 2}", "manual": m} for i, (c, m) in enumerate(cells)]


def test_duplicates_are_merged_but_every_source_row_and_label_is_kept():
    recs, stats = csl.clean(entries(("https://a.example.com/x", "Live"), ("a.example.com/x/", "Alive"),
                                    ("https://b.example.com", "Dead"), ("https://a.example.com/x", "Unsure")))
    assert [r["raw"] for r in recs] == ["https://a.example.com/x", "https://b.example.com"]
    assert stats["duplicates"] == 2 and stats["links_found"] == 4
    assert len(recs[0]["sources"]) == 3 and recs[0]["manual"] == ["live", "unsure"]
    assert recs[1]["manual"] == ["dead"]


def test_the_first_spelling_is_the_one_kept():
    recs, _ = csl.clean(entries(("bank.example.com/x", None), ("https://BANK.example.com/x/", None)))
    assert recs[0]["raw"] == "bank.example.com/x"


def test_the_safety_gate_refuses_are_noted_and_not_listed():
    recs, stats = csl.clean(entries(("http://10.0.0.5/admin", None), ("http://example.com:8080/", None),
                                    ("http://user:pw@example.com/", None), ("https://ok.example.com", None)))
    notes = {r["raw"]: r["note"] for r in recs}
    assert notes["https://ok.example.com"] == ""
    assert all("safety gate" in notes[k] for k in notes if k != "https://ok.example.com")
    assert stats["refused_by_gate"] == 3


def test_shorteners_are_flagged():
    recs, _ = csl.clean(entries(("https://bit.ly/4abc", None), ("https://tinyurl.com/x", None), ("https://a.example.com", None)))
    assert [r["shortener"] for r in recs] == [True, True, False]


@pytest.mark.parametrize("label, expected", [("Live", "live"), ("Alive", "live"), ("alive ", "live"), ("DEAD", "dead"),
                                              ("Unsure", "unsure"), (None, None), ("", None), ("Reviewed", "reviewed")])
def test_team_labels_are_normalised(label, expected):
    assert csl.normalise_label(label) == expected


def test_stats_add_up():
    cells = [("Links", None), ("", None), ("https://a.example.com", "Live"), ("a.example.com", "Live"),
             ("https://b.example.com", None), (None, None)]
    recs, stats = csl.clean(entries(*cells))
    assert stats["cells"] == 6 and stats["empty_or_header"] == 3 and stats["links_found"] == 3
    assert stats["duplicates"] == 1 and len(recs) == 2
