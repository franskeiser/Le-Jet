"""
Tests for src/data/add_hunted_to_pilot.py.

Run with:  python -m pytest tests/test_add_hunted_to_pilot.py
"""

from __future__ import annotations

import pandas as pd

from src.data import add_hunted_to_pilot as ah
from src.data import build_crowd_dataset as bcd
from src.unified.prepare_web_37 import FEATURES_37


def _live(urls):
    df = pd.DataFrame({"url": urls})
    for i, col in enumerate(FEATURES_37):
        df[col] = i
    return df


def _pilot():
    first = ah.synthetic_rows(_live(["hxxps://old-one[.]com/"]), first_id=1, neutral_rate=0.0, seed=1)
    return first, first.iloc[0:0].copy()


def test_new_rows_follow_the_57_column_layout_and_continue_the_ids():
    pilot, aside = _pilot()
    kept, _, counts = ah.add_to_pilot(pilot, aside, _live(["hxxps://a-brand[.]com/", "hxxp://b-brand[.]xyz/x"]))
    bcd.check(kept)
    assert list(kept.columns) == bcd.COLUMNS
    assert list(kept["message_id"]) == ["pilot_000001", "pilot_000002", "pilot_000003"]
    assert counts["hunted_live_in"] == 2 and counts["hunted_already_in_pilot"] == 0


def test_hunted_rows_are_smishing_with_a_real_link_inside_the_text():
    pilot, aside = _pilot()
    kept, _, _ = ah.add_to_pilot(pilot, aside, _live(["hxxps://a-brand[.]com/"]))
    row = kept.iloc[-1]
    assert row["label"] == "smishing" and row["link"] == "https://a-brand.com/"
    assert "https://a-brand.com/" in row["sms_text"]


def test_a_link_already_in_the_pilot_is_not_added_twice():
    pilot, aside = _pilot()
    kept, _, counts = ah.add_to_pilot(pilot, aside, _live(["hxxps://old-one[.]com/"]))
    assert len(kept) == 1 and counts["hunted_already_in_pilot"] == 1


def test_the_domain_cap_sets_extra_rows_aside_instead_of_deleting_them():
    pilot, aside = _pilot()
    urls = [f"hxxps://kit{i}[.]same-host[.]com/" for i in range(25)]
    kept, set_aside, _ = ah.add_to_pilot(pilot, aside, _live(urls))
    assert (kept["link_domain"] == "same-host.com").sum() == bcd.MAX_PER_DOMAIN_PER_CLASS
    assert len(set_aside) == 5
