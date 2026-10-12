"""
Tests for scripts/build_smishing_workbook.py (joining the team's links, the team's labels and the lab's
results into one workbook). Invented example domains only. The workbook-writing tests need openpyxl (a host
helper kept outside the repo) and are skipped where it is missing.

Run with:  python -m pytest tests/test_smishing_workbook.py
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from lejet import urlnorm

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bsw", os.path.join(HERE, "..", "scripts", "build_smishing_workbook.py"))
bsw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsw)


def meta(raw, manual=(), shortener=False, note=""):
    key = raw.split("://")[-1].lower().rstrip("/")             # the real metadata keys carry no scheme
    return {"raw": raw, "key": key, "sources": ["a.xlsx:Sheet1:3"], "manual": list(manual), "note": note, "shortener": shortener}


def lab(raw, fail="none", features=None, **kw):
    row = {"url_id": urlnorm.compute_url_id(raw), "fail_reason": fail, "final_url": "hxxps://x[.]example[.]com/",
           "final_reg_domain": "x[.]example[.]com", "final_status": 200, "hop_count": 0,
           "features": features if features is not None else {"domain_age": 123}}
    row.update(kw)
    return row


@pytest.mark.parametrize("fail, shot, expected", [
    ("none", True, "live"), ("none", False, "live"),
    ("http_4xx", True, "reachable-blocked"), ("http_4xx", False, "reachable-blocked"),
    ("challenge_page", False, "reachable-blocked"), ("parked_or_default_page", False, "reachable-blocked"),
    ("connect_timeout", True, "reachable-blocked"),          # the browser got a page the plain fetch could not
    ("connect_timeout", False, "not live"), ("dns_nxdomain", False, "not live"), ("dns_timeout", False, "not live"),
    ("too_many_redirects", False, "not live"), ("read_timeout", False, "not live"),
])
def test_lab_class_rules(fail, shot, expected):
    assert bsw.lab_class(fail, shot) == expected


def test_every_lab_failure_reason_has_a_plain_language_text():
    from lejet import schema
    for reason in schema.FAIL_REASONS:
        if reason.startswith("render_") or reason in ("blocked_or_empty",):
            continue
        assert reason in bsw.WHY, reason


def test_team_label_text():
    assert bsw.team_label([]) == "(no label)"
    assert bsw.team_label(["live"]) == "live"
    assert bsw.team_label(["live", "unsure"]) == "live/unsure"


def test_join_gives_each_link_one_record_with_both_sides():
    a, b, c = "https://a.example.com/x", "https://b.example.com/y", "https://c.example.com/z"
    recs = bsw.join([meta(a, ["live"]), meta(b, ["unsure"], shortener=True), meta(c)],
                    [lab(a), lab(b, "http_4xx", final_status=403), lab(c, "dns_nxdomain", features={"domain_age": -1})],
                    {urlnorm.compute_url_id(a): {"ok": True}, urlnorm.compute_url_id(b): {"ok": True}},
                    {urlnorm.compute_url_id(a): "7"})
    assert [r["cls"] for r in recs] == ["live", "reachable-blocked", "not live"]
    assert recs[0]["page_rank"] == "7" and recs[0]["shot"] == "yes" and recs[0]["domain_age"] == 123
    assert recs[1]["shortener"] == "yes" and recs[1]["status"] == 403 and recs[1]["team"] == "unsure"
    assert recs[2]["shot"].startswith("no") and recs[2]["domain_age"] == "" and recs[2]["team"] == "(no label)"
    assert all(r["url"].startswith("hxxp") and "[.]" in r["url"] for r in recs)            # defanged


def test_a_link_missing_from_the_lab_batch_is_kept_and_marked_not_tested():
    recs = bsw.join([meta("http://10.0.0.5/x", note="refused by the safety gate: rejected_url")], [], {}, {})
    assert recs[0]["lab"] == "not tested" and recs[0]["cls"] == "not live" and "safety gate" in recs[0]["why"]


def test_caveats_flag_pages_that_loaded_but_are_probably_not_the_phish():
    short = meta("https://bit.ly/4abc", ["live"], shortener=True)
    own = lab("https://bit.ly/4abc", final_reg_domain="bit[.]ly")                   # ended on the shortener itself
    assert "shortener's own page" in bsw.caveats(short, own, "")
    plain = meta("https://a.example.com/x")
    assert bsw.caveats(plain, lab("https://a.example.com/x"), "") == ""
    assert "free hosting platform (edgeone.dev)" in bsw.caveats(
        plain, lab("https://a.example.com/x", free_hosting_subdomain=1, hosting_platform="edgeone.dev"), "")
    assert "very popular" in bsw.caveats(plain, lab("https://a.example.com/x"), "10")
    assert bsw.caveats(plain, lab("https://a.example.com/x"), "8") == ""
    assert bsw.caveats(plain, lab("https://a.example.com/x"), "not-a-number") == ""
    assert bsw.caveats(short, lab("https://bit.ly/4abc", "dns_nxdomain", final_reg_domain="bit[.]ly"), "10") == ""   # failed: no caveat


@pytest.mark.parametrize("final", ["hxxps://site-abc[.]edgeone[.]dev/", "hxxps://x[.]edgeone[.]app/p", "hxxps://pay[.]eu[.]cc/"])
def test_pages_on_free_platforms_are_flagged_even_when_the_lab_run_predates_the_list(final):
    plain = meta("https://a.example.com/x")
    row = lab("https://a.example.com/x", final_url=final, free_hosting_subdomain=0)
    assert "free hosting platform" in bsw.caveats(plain, row, "")


def test_an_ordinary_final_address_is_not_flagged_as_free_hosting():
    plain = meta("https://a.example.com/x")
    row = lab("https://a.example.com/x", final_url="hxxps://shop[.]example[.]com/", free_hosting_subdomain=0)
    assert "free hosting" not in bsw.caveats(plain, row, "")


@pytest.mark.parametrize("status", [404, 410])
def test_a_removed_shortener_link_is_not_live_and_is_explained(status):
    short = meta("https://bit.ly/4abcDEF", ["live"], shortener=True)
    row = lab("https://bit.ly/4abcDEF", "http_4xx", final_status=status, final_reg_domain="bit[.]ly")
    recs = bsw.join([short], [row], {urlnorm.compute_url_id("https://bit.ly/4abcDEF"): {"ok": True, "bytes": 90000}}, {})
    assert recs[0]["cls"] == "not live" and "case-sensitive" in recs[0]["why"] and str(status) in recs[0]["why"]


def test_a_403_on_an_ordinary_site_stays_reachable_but_blocked():
    plain = meta("https://a.example.com/x")
    row = lab("https://a.example.com/x", "http_4xx", final_status=403)
    recs = bsw.join([plain], [row], {}, {})
    assert recs[0]["cls"] == "reachable-blocked"


def test_a_404_on_an_ordinary_site_is_not_treated_as_a_removed_shortener():
    plain = meta("https://a.example.com/x")            # not a shortener
    row = lab("https://a.example.com/x", "http_4xx", final_status=404)
    assert bsw.join([plain], [row], {}, {})[0]["cls"] == "reachable-blocked"


@pytest.mark.parametrize("status", [520, 521, 522, 523, 524, 525, 530])
def test_a_cloudflare_origin_error_means_the_server_is_down(status):
    plain = meta("https://a.example.com/x")
    recs = bsw.join([plain], [lab("https://a.example.com/x", "http_5xx", final_status=status)], {}, {})
    assert recs[0]["cls"] == "not live" and "server is down" in recs[0]["why"]


def test_a_shortener_that_resolves_to_a_different_site_is_not_touched_by_the_refinement():
    short = meta("https://bit.ly/4abc", ["live"], shortener=True)
    recs = bsw.join([short], [lab("https://bit.ly/4abc", final_reg_domain="x[.]example[.]com", final_status=200)], {}, {})
    assert recs[0]["cls"] == "live"


def test_the_browsers_own_address_title_and_white_screen_flag_come_through():
    a, b = "https://a.example.com/x", "https://b.example.com/y"
    shots = {urlnorm.compute_url_id(a): {"ok": True, "bytes": 90000, "final_url": "hxxps://landing[.]example[.]org/p", "title": "Welcome", "text_chars": 300},
             urlnorm.compute_url_id(b): {"ok": True, "bytes": 4000, "final_url": "hxxps://b[.]example[.]com/y", "title": "", "text_chars": 0}}
    recs = bsw.join([meta(a), meta(b)], [lab(a), lab(b)], shots, {})
    assert recs[0]["browser_url"] == "hxxps://landing[.]example[.]org/p" and recs[0]["title"] == "Welcome" and recs[0]["blank"] == ""
    assert recs[1]["blank"].startswith("yes")


def test_a_rescued_picture_is_explained_and_not_called_a_white_screen():
    a = "https://a.example.com/x"
    shots = {urlnorm.compute_url_id(a): {"ok": True, "bytes": 300000, "final_url": "about:blank", "title": "", "text_chars": 0,
                                         "rescued": True}}
    recs = bsw.join([meta(a)], [lab(a)], shots, {})
    assert recs[0]["blank"].startswith("no: picture taken before the page wiped itself")


def test_the_caveat_reaches_the_record():
    short = meta("https://bit.ly/4abc", ["dead"], shortener=True)
    recs = bsw.join([short], [lab("https://bit.ly/4abc", final_reg_domain="bit[.]ly")], {}, {})
    assert recs[0]["cls"] == "live" and "shortener's own page" in recs[0]["caveat"]


def test_agreement_table_counts_add_up():
    recs = [{"team": "live", "cls": "live"}, {"team": "live", "cls": "not live"}, {"team": "dead", "cls": "not live"},
            {"team": "unsure", "cls": "reachable-blocked"}, {"team": "(no label)", "cls": "live"}]
    table = bsw.agreement(recs)
    assert table[0] == ["Team's label \\ Lab result", "live", "reachable-blocked", "not live", "total"]
    rows = {r[0]: r[1:] for r in table[1:]}
    assert rows["live"] == [1, 0, 1, 2] and rows["dead"] == [0, 0, 1, 1] and rows["unsure"] == [0, 1, 0, 1]
    assert rows["total"] == [2, 1, 2, 5]


def test_the_workbook_has_the_expected_sheets_and_keeps_text_as_text(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    a, b = "https://a.example.com/x", "https://b.example.com/y"
    recs = bsw.join([meta(a, ["live"]), meta(b, ["unsure"])],
                    [lab(a), lab(b, "dns_nxdomain", features={"domain_age": -1})], {}, {})
    recs[0]["why"] = "=cmd|' /C calc'!A0"                                     # hostile text must stay inert
    recs[0]["sources"] = "=HYPERLINK(\"http://evil.example\",\"x\")"
    out = str(tmp_path / "w.xlsx")
    bsw.write_workbook(recs, out, {"Batch": "t1"})
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["Summary", "Cleaned", "Live", "Reachable-blocked", "Not live", "Unsure"]
    assert wb["Cleaned"].max_row == 3 and wb["Live"].max_row == 2 and wb["Not live"].max_row == 2
    assert wb["Unsure"].max_row == 2                                           # the unsure link is listed even though not live
    why = [c for c in wb["Cleaned"][2] if isinstance(c.value, str) and c.value.startswith("=cmd")][0]
    assert why.data_type == "s" and why.value == "=cmd|' /C calc'!A0"
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                assert cell.data_type != "f", (ws.title, cell.coordinate)
        assert not ws._hyperlinks
    flat = " ".join(str(c.value) for ws in wb.worksheets for row in ws.iter_rows() for c in row if c.value is not None)
    assert "https://a.example.com" not in flat and "hxxps://a[.]example[.]com" in flat    # only defanged URLs
