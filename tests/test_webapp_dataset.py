"""
Tests for webapp/dataset.py: the shared submissions CSV is the crowd builder's 57-column layout minus seven columns.

Run with:  python -m pytest tests/test_webapp_dataset.py
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from lejet import schema
from src.data import build_crowd_dataset as bcd
from src.unified.prepare_web_37 import FEATURES_37
from webapp import dataset as ds

TEXT = "Your GCash account is locked. Verify at https://gcash-verify-ph.xyz/login now."
T0 = datetime.datetime(2026, 11, 3, 6, 0, tzinfo=datetime.timezone.utc)      # 14:00 Philippine time


def _lab(**over):
    row = schema.new_row("0123456789abcdef", "https://gcash-verify-ph.xyz/login", "web")
    row.update(fail_reason="none", html_available=1, page_fetched=1, final_status=200,
               final_url="https://gcash-verify-ph.xyz/login", reg_domain="gcash-verify-ph.xyz",
               final_reg_domain="gcash-verify-ph.xyz", final_differs=0, nb_hyperlinks=7)
    row.update(over)
    return schema.clean_row(row)


def _row(message_id="web_000001", text=TEXT, label="smishing", language="en", **over):
    lab = _lab(**over)
    link, _ = ds.first_link(text)
    return ds.build_row(message_id, text, label, language, link, lab, schema.full_features(lab), T0)


REMOVED = {"category", "sender_type", "date_received", "label_source", "consent", "inspected_at", "hours_to_inspect"}


def test_layout_is_the_builders_minus_seven_columns():
    assert set(ds.REMOVED_COLUMNS) == REMOVED
    assert ds.COLUMNS == [c for c in bcd.COLUMNS if c not in REMOVED]          # same order, same names
    assert set(bcd.COLUMNS) - set(ds.COLUMNS) == REMOVED and len(ds.COLUMNS) == 50
    assert not REMOVED & set(ds.COLUMNS) and "language" in ds.COLUMNS and "label_verified_by" in ds.COLUMNS
    assert ds.FEATURES_37 == list(FEATURES_37)
    assert all(f in schema.FULL_FEATURES for f in ds.FEATURES_37)
    assert ds.LANGUAGES == ("en", "tl", "taglish")


def test_helpers_match_the_builder():
    texts = pd.Series([TEXT, "Visit bit.ly/3xYz for 500 pesos", "no digits https://a.example.com/x"])
    assert [ds.template_id(t) for t in texts] == list(bcd.template_ids(texts))
    for link in ("https://bit.ly/3xYz", "http://www.tinyurl.com/a", "https://gcash-verify-ph.xyz"):
        assert ds.is_shortened(link) == bcd.is_shortened(link)
    for reason in schema.FAIL_REASONS:
        for differs in (0, 1):
            assert ds.fetch_status(reason, differs) == bcd.fetch_status({"fail_reason": reason, "final_differs": differs})


def test_row_has_50_columns_and_no_blanks():
    row = _row(language="taglish")
    assert list(row) == ds.COLUMNS and len(row) == 50
    assert all(v is not None and v != "" for v in row.values())
    assert not REMOVED & set(row)
    assert row["language"] == "taglish" and row["label_verified_by"] == -1
    assert row["date_submitted"] == "2026-11-03 14:00" and row["fetch_status"] == "live" and row["page_fetched"] == 1
    assert row["nb_hyperlinks"] == 7 and row["page_rank"] == -1
    assert row["link"] == "https://gcash-verify-ph.xyz/login" and row["link_domain"] == "gcash-verify-ph.xyz"


def test_page_rank_is_minus_one_unless_the_site_looked_it_up():
    assert _row()["page_rank"] == -1
    lab = _lab()
    features = schema.full_features(lab)
    features["page_rank"] = 7                                                # webapp/rank.py puts the answer here
    row = ds.build_row("web_000001", TEXT, "smishing", "en", ds.first_link(TEXT)[0], lab, features, T0)
    assert row["page_rank"] == 7 and list(row) == ds.COLUMNS


def test_label_and_language_must_be_known():
    for language in ("en", "tl", "taglish"):
        assert _row(language=language)["language"] == language
    with pytest.raises(ValueError):
        _row(language="english")
    with pytest.raises(ValueError):
        _row(language="-1")
    with pytest.raises(ValueError):
        _row(label="maybe")


def test_dead_page_keeps_the_row_with_minus_ones():
    row = _row(fail_reason="dns_nxdomain", html_available=0, page_fetched=0, final_status=-1, final_url="")
    assert row["fetch_status"] == "dead" and row["page_fetched"] == 0 and row["final_url"] == -1


def test_first_link_and_defang():
    raw, norm = ds.first_link(TEXT)
    assert raw.startswith("https://gcash-verify-ph.xyz/login")
    assert norm.startswith("https://gcash-verify-ph.xyz")
    assert ds.first_link("no link here at all") is None
    shown = ds.defang_in_text(TEXT)
    assert "hxxps://gcash-verify-ph[.]xyz/login" in shown and "https://" not in shown


def test_clean_text_keeps_lines_and_drops_tricks():
    assert ds.clean_text("a\r\nb\x00\x1b[31m‮ c\r") == "a\nb c"
    assert len(ds.clean_text("x" * 5000)) == ds.MAX_TEXT


def test_store_append_reload_and_duplicates(tmp_path):
    path = str(tmp_path / "sub" / "s.csv")
    store = ds.Store(path)
    assert len(store) == 0 and store.next_id() == "web_000001"
    row = _row()
    assert store.existing(TEXT, row["link"]) is None
    store.append(row)
    assert store.existing(TEXT, row["link"]) == "web_000001"
    assert store.next_id() == "web_000002" and store.counts["smishing"] == 1 and store.lang_counts["en"] == 1

    again = ds.Store(path)                                   # a restart keeps everything
    assert len(again) == 1 and again.next_id() == "web_000002"
    assert again.existing(TEXT, row["link"]) == "web_000001"
    assert again.newest(5)[0]["sms_preview"].count("hxxps://") == 1 and again.newest(5)[0]["language"] == "en"
    assert again.lang_counts == {"en": 1, "tl": 0, "taglish": 0}

    df = pd.read_csv(path)                                   # and it is a plain CSV the builder's tools can read
    assert list(df.columns) == ds.COLUMNS and len(df) == 1 and df.loc[0, "sms_text"] == TEXT


def test_store_round_trips_multiline_and_neutralises_formulas(tmp_path):
    store = ds.Store(str(tmp_path / "s.csv"))
    text = "=HYPERLINK(\"x\")\nline two, with a comma https://evil-bank.example.com/a"
    store.append(_row(text=text))
    df = pd.read_csv(store.path)
    assert df.loc[0, "sms_text"] == "'" + text               # a leading = is made inert for spreadsheets
    assert store.existing(text, ds.first_link(text)[0]) == "web_000001"


def test_store_refuses_a_file_with_another_layout(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        ds.Store(str(bad))


def _legacy_file(path, rows):
    """A file as the previous version of the site wrote it: 57 columns."""
    import csv
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ds.LEGACY_COLUMNS_57, lineterminator="\n")
        w.writeheader()
        for r in rows:
            full = dict(r, category=-1, sender_type=-1, date_received=-1, label_source="submitter", consent=1,
                        inspected_at="2026-11-03 14:00", hours_to_inspect=0.0, language=-1)
            w.writerow(full)


def test_a_57_column_file_is_converted_and_a_copy_is_kept(tmp_path):
    path = tmp_path / "s.csv"
    _legacy_file(path, [_row("web_000001"), _row("web_000002", text="Pay now https://lbc-track.top/pay", label="legitimate")])
    before = path.read_text(encoding="utf-8")
    store = ds.Store(str(path))
    assert len(store) == 2 and store.next_id() == "web_000003"
    assert store.counts == {"smishing": 1, "legitimate": 1}
    assert store.lang_counts == {"en": 0, "tl": 0, "taglish": 0}                  # old rows have no language (-1)
    assert store.existing(TEXT, "https://gcash-verify-ph.xyz/login") == "web_000001"
    backup = tmp_path / "s.csv.57cols.bak"
    assert backup.read_text(encoding="utf-8") == before                           # the old file, untouched
    df = pd.read_csv(path)
    assert list(df.columns) == ds.COLUMNS and len(df) == 2 and not REMOVED & set(df.columns)
    assert list(df["language"]) == [-1, -1] and df.loc[1, "label"] == "legitimate" and df.loc[0, "sms_text"] == TEXT
    store.append(_row("web_000003", text="Third https://x-prize.top/a", language="tl"))   # and it keeps working
    assert ds.Store(str(path)).lang_counts["tl"] == 1 and len(ds.Store(str(path))) == 3
    assert len(list(tmp_path.glob("*.bak"))) == 1                                 # re-reading does not make more copies


def test_a_second_conversion_does_not_overwrite_the_first_copy(tmp_path):
    path = tmp_path / "s.csv"
    (tmp_path / "s.csv.57cols.bak").write_text("older copy", encoding="utf-8")
    _legacy_file(path, [_row()])
    ds.Store(str(path))
    assert (tmp_path / "s.csv.57cols.bak").read_text(encoding="utf-8") == "older copy"
    assert len(list(tmp_path.glob("s.csv.57cols.*.bak"))) == 1
