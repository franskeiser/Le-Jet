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


def test_a_mode_is_required():
    with pytest.raises(SystemExit):
        bcd.main([])


def test_mask_numbers_keeps_the_link():
    text = "Call 0917 123 4567 or +639171234567, acct 1234-5678-9012. Pay: https://x.xyz/0917123456789 via 22565"
    assert bcd.mask_numbers(text) == "Call <NUM> or <NUM>, acct <ACCT>. Pay: https://x.xyz/0917123456789 via 22565"


def test_url_id_matches_the_lab():
    import sys
    sys.path.insert(0, str(bcd.ROOT / "src"))
    from lejet.urlnorm import compute_url_id
    assert bcd.url_id("https://a-wallet.xyz/v") == compute_url_id("https://a-wallet.xyz/v")


@pytest.mark.parametrize("rec,expected", [
    ({"fail_reason": "none", "final_differs": 0}, "live"),
    ({"fail_reason": "none", "final_differs": 1}, "redirected"),
    ({"fail_reason": "dns_nxdomain"}, "dead"),
    ({"fail_reason": "connect_timeout"}, "timeout"),
    ({"fail_reason": "challenge_page"}, "blocked"),
])
def test_fetch_status(rec, expected):
    assert bcd.fetch_status(rec) == expected


def test_build_crowd_joins_submissions_to_the_lab(tmp_path):
    import datetime
    import json

    import make_crowd_template as mct
    raw, clean = tmp_path / "raw", tmp_path / "clean"
    raw.mkdir(), clean.mkdir()
    wb = mct.build()
    ws = wb["submissions"]
    cols = bcd.crowd_to_urls.COLUMNS

    def put(row, **values):
        for name, value in values.items():
            ws.cell(row=row, column=cols.index(name) + 1, value=value)

    base = dict(label="smishing", category="bank", language="en", sender_type="number",
                date_received="2026-10-18", label_source="submitter", consent="1")
    put(3, date_submitted=datetime.datetime(2026, 10, 18, 22, 0), label_verified_by="r02",
        sms_text="BDO: verify now https://bdo-x.top/login or call 09171234567", **base)          # inspected, fetched
    put(4, date_submitted=datetime.datetime(2026, 10, 18, 21, 0), label_verified_by="r02",
        sms_text="LTO fine https://lto-fine.bar", **base)                                       # inspected, dead
    put(5, date_submitted=datetime.datetime(2026, 10, 18, 23, 0), label_verified_by="r02",
        sms_text="New https://not-run-yet.xyz", **base)                                         # not inspected
    put(6, date_submitted=datetime.datetime(2026, 10, 18, 23, 5), label_verified_by=None,
        sms_text="Unchecked https://bdo-x.top/login", **base)                                   # label not checked
    wb.save(raw / "submissions_2026-10-18_r01.xlsx")

    full = {f: 5 for f in bcd.FEATURES_37 if f != "page_rank"}
    recs = [
        {"url_id": bcd.url_id("https://bdo-x.top/login"), "fetch_ts": "2026-10-18T15:30:00Z", "fail_reason": "none",
         "final_differs": 0, "final_status": 200, "final_url": "hxxps://bdo-x[.]top/login", "page_fetched": 1,
         "reg_domain": "bdo-x[.]top", "final_reg_domain": "bdo-x[.]top", "features_full": full},
        {"url_id": bcd.url_id("https://lto-fine.bar"), "fetch_ts": "2026-10-18T15:30:00Z", "fail_reason": "dns_nxdomain",
         "final_differs": 0, "final_status": -1, "final_url": "", "page_fetched": 0,
         "reg_domain": "lto-fine[.]bar", "final_reg_domain": "lto-fine[.]bar",
         "features_full": {**full, "nb_hyperlinks": -1}},
    ]
    (clean / "crowd20261018.clean.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    pd.DataFrame({"url_id": [recs[0]["url_id"]], "page_rank": [3]}).to_csv(clean / "crowd20261018.model_input.csv")

    kept, _, summary = bcd.build_crowd(raw, clean)
    assert list(kept.columns) == bcd.COLUMNS
    assert summary["row_rules"]["left_out_example_row"] == 1
    assert summary["row_rules"]["left_out_not_inspected"] == 1
    assert summary["row_rules"]["left_out_not_verified"] == 1
    assert kept["message_id"].tolist() == ["crowd_000001", "crowd_000002"]    # ordered by date_submitted
    dead, live = kept.iloc[0], kept.iloc[1]
    assert dead["link_domain"] == "lto-fine.bar" and dead["final_url"] == -1 and dead["fetch_status"] == "dead"
    assert dead["page_rank"] == -1 and dead["nb_hyperlinks"] == -1
    assert live["final_url"] == "https://bdo-x.top/login" and live["page_rank"] == 3 and live["page_fetched"] == 1
    assert live["inspected_at"] == "2026-10-18 23:30" and live["hours_to_inspect"] == 1.5
    assert live["sms_text"] == "BDO: verify now https://bdo-x.top/login or call <NUM>"
