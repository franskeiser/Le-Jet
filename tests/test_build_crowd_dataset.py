"""
Tests for src/data/build_crowd_dataset.py (docs/c_prune_architecture.md sections 4.3, 4.4).

Run with:  python -m pytest tests/test_build_crowd_dataset.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import build_crowd_dataset as bcd
from src.unified.prepare_web_37 import FEATURES_37


def _lean(rows):
    """A tiny fresh_lean-shaped frame: (sms_text, label, link, link_domain) per row."""
    df = pd.DataFrame(rows, columns=["sms_text", "label", "link", "link_domain"])
    df["source"], df["link_status"] = "synthetic", "link"
    for i, col in enumerate(FEATURES_37):
        df[col] = i
    return df


def test_57_columns_in_documented_order():
    assert len(bcd.COLUMNS) == 57
    assert bcd.COLUMNS[:3] == ["message_id", "sms_text", "label"]
    assert bcd.COLUMNS[-37:] == list(FEATURES_37)


@pytest.mark.parametrize("text, expected", [
    ("PayMaya: verify now https://a.xyz", "e-wallet"),
    ("Maya: verify now https://a.xyz", "e-wallet"),
    ("Security Bank: review https://a.xyz", "bank"),
    ("Home Credit: loan https://a.xyz", "loan"),
    ("Your parcel fee https://a.xyz", "other_financial"),
])
def test_category_for(text, expected):
    assert bcd.category_for(text) == expected


def test_language_for():
    assert bcd.language_for("BDO: Mag-check ng account details mo dito https://a.xyz") == "taglish"
    assert bcd.language_for("BDO: Your statement is ready at https://a.xyz") == "en"


def test_template_ids_ignore_links_and_amounts():
    texts = pd.Series(["BDO: P2,200 sent. See https://a.xyz/x",
                       "BDO: P7,900 sent. See https://b.xyz/y",
                       "BPI: P2,200 sent. See https://a.xyz/x"])
    ids = bcd.template_ids(texts)
    assert ids[0] == ids[1] and ids[0] != ids[2]


def test_is_shortened():
    assert bcd.is_shortened("https://bit.ly/abc") == 1
    assert bcd.is_shortened("https://www.bdo.com.ph/") == 0


def test_build_pilot_rules_and_layout(tmp_path):
    rows = [("BDO: verify https://dup.xyz", "smishing", "https://dup.xyz", "dup.xyz")] * 2
    rows += [(f"GCash: verify https://cap.xyz/{i}", "smishing", f"https://cap.xyz/{i}", "cap.xyz") for i in range(22)]
    rows += [(f"GCash: receipt https://cap.xyz/{i}", "legitimate", f"https://cap.xyz/{i}", "cap.xyz") for i in range(3)]
    rows += [("Maya: no link in this text", "legitimate", "https://gone.xyz", "gone.xyz")]
    src = tmp_path / "fresh_lean.csv"
    _lean(rows).to_csv(src, index=False)

    kept, set_aside, summary = bcd.build_pilot(src)
    assert list(kept.columns) == bcd.COLUMNS
    assert summary["row_rules"] == {"input_rows": 28, "dropped_no_link": 1, "dropped_exact_duplicate": 1,
                                    "set_aside_domain_cap": 2}
    assert len(kept) == 24 and len(set_aside) == 2
    # The cap is per domain AND class: the 3 legitimate cap.xyz rows are all kept.
    assert kept.groupby(["link_domain", "label"]).size().to_dict() == {
        ("cap.xyz", "legitimate"): 3, ("cap.xyz", "smishing"): 20, ("dup.xyz", "smishing"): 1}
    assert kept["message_id"].str.match(r"^pilot_\d{6}$").all()
    assert (kept["date_submitted"] == -1).all() and (kept["final_url"] == -1).all()
    assert "not study results" in summary["source"]

    out = tmp_path / "crowd_pilot_synthetic.csv"
    bcd.write(kept, set_aside, summary, out)
    assert len(pd.read_csv(out)) == 24
    assert len(pd.read_csv(out.with_suffix(".set_aside.csv"))) == 2
    assert out.with_suffix(".summary.json").exists()


def test_crowd_mode_not_built_yet():
    with pytest.raises(SystemExit, match="Only --pilot"):
        bcd.main([])
