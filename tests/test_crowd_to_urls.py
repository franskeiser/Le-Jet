"""
Tests for scripts/crowd_to_urls.py.

Run with:  python -m pytest tests/test_crowd_to_urls.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import crowd_to_urls as ctu                                  # noqa: E402


def _row(text, consent="1"):
    return {"date_submitted": "2026-11-03 14:20", "sms_text": text,
            "submitter_view": "Suspicious or potentially fraudulent", "org_type": "Bank",
            "concern": "Account verification", "label": "smishing",
            "category": "e-wallet", "language": "en", "sender_type": "number",
            "date_received": "2026-11-03", "label_source": "submitter",
            "label_verified_by": "r02", "consent": consent}


def _write(path, rows, bom=False):
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ctu.COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_first_link_only_and_normalised():
    assert ctu.first_link("Pay now: bdo-secure[.]xyz/pay or https://b.test/x") == "http://bdo-secure.xyz/pay"
    assert ctu.first_link("Verify: https://gcash-verify-ph.xyz/login.") == "https://gcash-verify-ph.xyz/login"
    assert ctu.first_link("No link here, call us") is None


def test_build_skips_and_dedups():
    rows = [
        _row(ctu.EXAMPLE_TEXT),
        _row("Locked! https://a-wallet.xyz/v"),
        _row("Again https://a-wallet.xyz/v"),
        _row("No consent https://c.xyz", consent="0"),
        _row("no link at all"),
        _row("Your bill: https://meralco.com.ph/pay"),
    ]
    urls, report = ctu.build([("submissions_2026-11-03.csv", rows)])
    assert urls == ["https://a-wallet.xyz/v", "https://meralco.com.ph/pay"]
    assert report["rows"] == 6 and report["kept"] == 2 and report["duplicates"] == 1
    assert report["example_row"] == 1 and report["no_consent"] == 1 and report["no_link"] == 1
    assert report["no_link_rows"] == ["submissions_2026-11-03.csv:6"]


def test_main_reads_excel_bom_and_writes_outputs(tmp_path):
    src = tmp_path / "submissions_2026-11-03.csv"
    _write(src, [_row('Sabi ng "bank": https://x-bank.xyz/otp, salamat')], bom=True)
    out = tmp_path / "input" / "urls.txt"
    assert ctu.main([str(src), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "https://x-bank.xyz/otp\n"
    assert json.loads((tmp_path / "input" / "urls.txt.report.json").read_text())["kept"] == 1


def test_missing_column_is_an_error(tmp_path):
    src = tmp_path / "bad.csv"
    src.write_text("date_submitted,sms_text\n2026-11-03 14:20,https://a.xyz\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        ctu.read_rows(str(src))


def test_main_reads_xlsx_template_copy(tmp_path):
    import datetime

    import make_crowd_template as mct
    wb = mct.build()
    ws = wb[ctu.SHEET]
    # Row 2 is the example; row 3 a real message with Excel-typed dates and a number for consent.
    ws.cell(row=3, column=1, value=datetime.datetime(2026, 10, 18, 22, 45))
    for name, value in [("sms_text", "Pay toll: https://toll-ph.xyz/pay"), ("label", "smishing"),
                        ("date_received", datetime.datetime(2026, 10, 18)), ("consent", 1)]:
        ws.cell(row=3, column=ctu.COLUMNS.index(name) + 1, value=value)
    src = tmp_path / "submissions_2026-10-18.xlsx"
    wb.save(src)
    rows = ctu.read_rows(str(src))
    assert len(rows) == 2                     # the formatted empty rows below are not read
    assert rows[1]["date_submitted"] == "2026-10-18 22:45" and rows[1]["date_received"] == "2026-10-18"
    assert rows[1]["consent"] == "1"
    out = tmp_path / "urls.txt"
    assert ctu.main([str(src), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "https://toll-ph.xyz/pay\n"


def test_template_has_a_dropdown_per_choice_column():
    import make_crowd_template as mct
    wb = mct.build()
    formulas = {dv.formula1 for dv in wb[ctu.SHEET].data_validations.dataValidation}
    assert formulas == {f"=list_{n}" for n in mct.CHOICES}
    assert set(mct.CHOICES) <= set(ctu.COLUMNS) and set(mct.HELP) == set(ctu.COLUMNS)
    assert len(mct.CHOICES["concern"]) == 18 and len(mct.CHOICES["org_type"]) == 11
