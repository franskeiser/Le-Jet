"""Tests for scripts/build_final_from_master.py: the text repair and the message key used to match the data lead's list to our rows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_final_from_master as M  # noqa: E402


def test_repair_fixes_utf8_read_as_windows_1252():
    assert M.repair("Get a \u00e2\u0082\u00b1") != "Get a \u00e2\u0082\u00b1"     # damaged text is changed
    assert M.repair("Youâ\u0080\u0099ve received") == "You\u2019ve received"


def test_repair_leaves_good_text_alone():
    for t in ["plain ascii", "Get a \u20b150 voucher", "caf\u00e9 promo", "Hi JUAN C., kumusta ka?"]:
        assert M.repair(t) == t


def test_key_ignores_case_spacing_bom_and_the_spreadsheet_guard():
    a = "'-August celebrant?  Get a FREE cake"
    b = "-august celebrant? get a free cake"
    assert M.key_of(a) == M.key_of(b)
    assert M.key_of("\ufeffHello\n  there") == M.key_of("hello there")


def test_different_messages_have_different_keys():
    assert M.key_of("Your OTP is 123456") != M.key_of("Your OTP is 654321")


def test_text_edits_file_keeps_every_link_in_the_new_text():
    """data/final/text_edits.csv rewrites bare-link messages around the SAME link: each new text must contain the link that was in the old one."""
    import re
    from pathlib import Path

    import pandas as pd
    path = Path(__file__).resolve().parents[1] / "data" / "final" / "text_edits.csv"
    if not path.exists():
        return
    edits = pd.read_csv(path, keep_default_na=False, dtype=str)
    assert list(edits.columns) == ["sms_text", "new_text", "reason"]
    assert edits.sms_text.is_unique
    url = re.compile(r"https?://\S+")
    for old, new in zip(edits.sms_text, edits.new_text):
        links = url.findall(old)
        assert links, f"no link in the original text {old!r}"
        assert all(l in new for l in links), f"the new text lost its link: {new!r}"
        assert new.strip() != links[0], "the new text is still only a link"
