"""
Tests for scripts/openphish_ph_links.py (Philippines links from the OpenPhish history) and scripts/openphish_check.py (which of them
are still alive). No lab, no network and no real phishing link: the history reader runs on a throwaway git repository made here, and
the live/dead judge on invented lab results.

Run with:  python -m pytest tests/test_openphish_scripts.py
"""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


links = load("openphish_ph_links")
check = load("openphish_check")


# ---------------------------------------------------------------------------------------
# the link rules
# ---------------------------------------------------------------------------------------
def test_defang_and_refang_are_exact_opposites():
    for url in ["http://a.example/x?y=1", "https://bpisal.com/ph", "http://203.0.113.9:8080/a.b"]:
        assert links.refang(links.defang(url)) == url
    assert links.defang("https://bpisal.com/ph") == "hxxps://bpisal[.]com/ph"
    assert "." not in links.defang("http://a.b.c/").replace("[.]", "")


@pytest.mark.parametrize("row, strong", [
    ({"brand": "BDO Unibank", "url": "https://bdo-x.com/ph"}, "PH brand"),
    ({"brand": "Land Bank of the Philippines", "url": "http://x.test/"}, "PH brand"),
    ({"brand": "Bank of the Philippine Islands", "tld": "com", "url": "http://x.test/"}, "PH brand"),
    ({"brand": "Generic/Spear Phishing", "tld": "ph", "url": "https://x.ph/"}, ".ph domain"),
    ({"brand": "Generic/Spear Phishing", "country_code": "PH", "url": "http://x.test/"}, "hosted in PH"),
    ({"brand": "BDO Unibank", "tld": "ph", "country_code": "PH", "url": "http://x.ph/"}, "PH brand + .ph domain + hosted in PH"),
])
def test_strong_signals(row, strong):
    assert links.strong_signal(row, links.weak_reasons(row)) == strong


@pytest.mark.parametrize("row", [
    {"brand": "Shopee", "url": "https://shopee-promo.example/", "country_code": "HK"},        # regional brand: weak only
    {"brand": "Meta Platforms, Inc.", "url": "https://3cxs4-0d4-3ppf-ambdo-10-09-2026-hh.pages.dev/"},   # "bdo-" inside a random name
    {"brand": "Roblox", "url": "http://x.test/", "page_language": "tl:0.62"},                  # a language-detector guess
    {"brand": "Itau Unibanco S.A", "url": "http://x.test/", "country_code": "BR"},
])
def test_weak_signals_are_never_strong(row):
    assert links.strong_signal(row, links.weak_reasons(row)) == ""


def test_weak_reasons_find_the_weak_signals_for_the_broad_file():
    assert "ph_brand" in links.weak_reasons({"brand": "Shopee", "url": "http://x.test/"})
    assert "tagalog_page" in links.weak_reasons({"brand": "", "url": "http://x.test/", "page_language": "tl:0.99"})
    assert "ph_words_in_url" in links.weak_reasons({"brand": "", "url": "http://my-bdo-login.test/"})
    assert links.weak_reasons({"brand": "Office365", "url": "http://x.test/", "country_code": "US"}) == []


# ---------------------------------------------------------------------------------------
# the history reader, on a throwaway git repository
# ---------------------------------------------------------------------------------------
HEADER = "url,brand,ip,asn,asn_name,country_code,country_name,tld,discover_time,family_id,host,isotime,page_language,sector\n"


def feed_line(url, brand, cc="US", tld="com", lang=""):
    return f"{url},{brand},1.2.3.4,AS1,Net,{cc},Land,{tld},t,f,h,i,{lang},Financial\n"


def commit(repo, text, when):
    (repo / "feed.csv").write_text(HEADER + text, encoding="utf-8")
    env = dict(os.environ, GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    base = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t.test"]
    subprocess.run(base + ["add", "feed.csv"], check=True, env=env, capture_output=True)
    subprocess.run(base + ["commit", "-m", "snapshot", "--no-gpg-sign"], check=True, env=env, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "academic"
    r.mkdir()
    subprocess.run(["git", "init", "-q", str(r)], check=True, capture_output=True)
    # day 1, two commits: the NEWER one is the day's snapshot (the first one's row is gone by then)
    commit(r, feed_line("http://old-only.example/", "BDO Unibank"), "2026-01-01T08:00:00")
    commit(r, feed_line("http://bdo-a.example/ph", "BDO Unibank") + feed_line("http://other.example/", "Meta"), "2026-01-01T20:00:00")
    # day 2: the BDO link is still there (must keep its FIRST day), plus a new .ph link and a Shopee one
    commit(r, feed_line("http://bdo-a.example/ph", "BDO Unibank") + feed_line("http://x.ph/", "Generic/Spear Phishing", tld="ph")
           + feed_line("http://shop.example/", "Shopee"), "2026-01-02T10:00:00")
    return r


def test_one_snapshot_per_day_and_first_seen_day(repo):
    days = links.daily_snapshots(repo)
    assert sorted(days) == ["2026-01-01", "2026-01-02"]
    rows = {r["url"]: r for r in links.collect(repo)}
    assert "http://old-only.example/" not in rows                       # only in the day's older commit: not a snapshot
    assert rows["http://bdo-a.example/ph"]["first_seen_day"] == "2026-01-01"   # kept once, on its first day
    assert rows["http://x.ph/"]["first_seen_day"] == "2026-01-02"
    assert "http://other.example/" not in rows                          # no Philippines signal at all
    assert len(rows) == len(links.collect(repo))                        # no duplicates


def test_strict_and_broad_files(repo, tmp_path):
    out = tmp_path / "out"
    assert links.main(["--repo", str(repo), "--out", str(out)]) == 0
    strict = list(csv.DictReader(open(out / "openphish_ph_strict.csv", encoding="utf-8")))
    broad = list(csv.DictReader(open(out / "openphish_ph_broad.csv", encoding="utf-8")))
    assert {r["url"] for r in strict} == {"http://bdo-a.example/ph", "http://x.ph/"}
    assert {r["url"] for r in broad} == {"http://bdo-a.example/ph", "http://x.ph/", "http://shop.example/"}   # Shopee is weak: broad only
    assert strict[0]["first_seen_day"] >= strict[-1]["first_seen_day"]  # newest first
    assert all(r["url"].startswith("http://") for r in strict)           # real links by default


def test_defang_option_writes_safe_files(repo, tmp_path):
    out = tmp_path / "out"
    assert links.main(["--repo", str(repo), "--out", str(out), "--defang"]) == 0
    text = open(out / "openphish_ph_strict_defanged.csv", encoding="utf-8").read()
    assert "hxxp://bdo-a[.]example/ph" in text and "http://" not in text


def test_a_folder_that_is_not_a_clone_is_refused(tmp_path):
    assert links.main(["--repo", str(tmp_path), "--out", str(tmp_path / "o")]) == 2


# ---------------------------------------------------------------------------------------
# alive or dead
# ---------------------------------------------------------------------------------------
def shot(title="", chars=500, ok=True, reach="live", reason="ok", head=""):
    return {"ok": ok, "reach": reach, "reason": reason, "title": title, "text_chars": chars, "text_head": head}


@pytest.mark.parametrize("fail, html, s, lab_ok, has, want", [
    ("none", True, shot("LANDBANK iAccess Retail Internet Banking - Login", 1048), True, True, "yes"),
    ("none", True, None, True, True, "yes"),                                              # fetched fine, no picture
    ("dns_nxdomain", False, None, True, True, "no"),
    ("dns_no_answer", False, None, True, True, "no"),
    ("parked_or_default_page", True, shot("Welcome to nginx!", 200), True, True, "no"),
    ("http_4xx", False, shot("", 9), True, True, "no"),                                   # a server that only says "Not Found"
    ("http_5xx", False, shot("Cloudflare Tunnel error | x.example | Cloudflare", 700), True, True, "no"),
    ("http_4xx", False, shot("Bank login", 2500), True, True, "maybe"),                   # error status but a real page
    ("none", True, shot("Account Suspended", 120), True, True, "no"),
    ("challenge_page", False, shot("Just a moment...", 40), True, True, "maybe"),
    ("blocked_or_empty", False, None, True, True, "maybe"),
    ("connect_timeout", False, shot(ok=False, reach="dead", reason="unreachable", chars=-1), True, True, "no"),
    ("connect_refused", False, None, True, True, "no"),
    ("tls_error", False, shot("Secure Login", 900), True, True, "maybe"),
    ("?", False, None, False, False, "unknown"),                                          # the lab run itself failed
])
def test_judge(fail, html, s, lab_ok, has, want):
    alive, why = check.judge(fail, html, s, lab_ok, has)
    assert alive == want and isinstance(why, str) and why


LANDBANK_HEAD = ("About Us | FAQs | Security Policy | Data Privacy Statement | Advisory on ATM Use | iAccess Features | Find Us Welcome to iAccess! "
                 "CLIENT ADVISORY Due to the expected surge in transactions from the simultaneous crediting of year-end bonuses and salaries")


def test_the_real_live_phishing_page_is_still_alive():
    """The one live link of the first run: a Landbank login clone. Its text must not trip the 'gone' phrases."""
    alive, why = check.judge("none", True, shot("LANDBANK iAccess Retail Internet Banking - Login", 1048, head=LANDBANK_HEAD), True)
    assert (alive, why) == ("yes", "page loads")


@pytest.mark.parametrize("head", [
    "This page has been removed.",
    "The page you requested has been taken down by the hosting provider.",
    "Sorry, we couldn't find that page.",
    "Oops! This page doesn't exist.",
    "404 Page Not Found",
    "This content isn't available right now",
    "Your account has been suspended. Please contact your hosting provider.",
    "This website is suspended due to a violation of our terms of service.",
    "Domain has expired. Renew now.",
    "This domain is for sale",
    "Suspected phishing page. This page has been reported.",
    "Access Denied. You don't have permission to access this resource.",
    "This site is temporarily unavailable. Please try again later.",
    "Welcome to nginx! If you see this page, the web server is successfully installed.",
    "Error 1033 Cloudflare Tunnel error",
    "Website under maintenance",
    "Coming soon",
    "Index of / Parent Directory",
])
def test_a_page_that_says_it_is_gone_is_dead_even_when_it_loaded_and_the_title_looks_normal(head):
    alive, why = check.judge("none", True, shot("My Bank - Login", 120, head=head), True)
    assert alive == "no", head
    assert "gone or blocked" in why


def test_the_phrase_that_made_it_dead_is_quoted_in_the_reason():
    alive, why = check.judge("none", True, shot("Welcome", 90, head="Hello. This page has been removed by the owner."), True)
    assert alive == "no" and "has been removed" in why


def test_the_title_alone_is_enough_and_works_without_any_text():
    assert check.judge("none", True, shot("404 Not Found", 800, head=""), True)[0] == "no"
    assert check.judge("none", True, shot("Account Suspended", 800, head=""), True)[0] == "no"


def test_a_page_with_lots_of_text_is_not_killed_by_a_stray_word_in_its_small_print():
    head = "Welcome back. Sign in to your account. Note: forbidden transactions may be reported."
    assert check.judge("none", True, shot("Online Banking", 5000, head=head), True)[0] == "yes"       # real content
    assert check.judge("none", True, shot("Online Banking", 300, head=head), True)[0] == "no"        # a short page with that word


def test_unknown_amount_of_text_counts_as_little():
    assert check.judge("none", True, shot("Hi", -1, head="This page has been removed."), True)[0] == "no"


def test_results_without_text_head_still_work_as_before():
    shot_old = {"ok": True, "reach": "live", "reason": "ok", "title": "Bank login", "text_chars": 900}      # a result from before text was recorded
    assert check.judge("none", True, shot_old, True)[0] == "yes"
    assert check.judge("http_4xx", False, {**shot_old, "text_chars": 9}, True)[0] == "no"


def test_gone_phrase_helper():
    assert check.gone_phrase("", "Page not found", 50)
    assert check.gone_phrase("Page not found", "lots of text " * 300, 9000)       # in the title: always counts
    assert check.gone_phrase("Hello", "page not found", 9000) == ""                # in the text of a big page: ignored
    assert check.gone_phrase("Hello", "all fine here", 50) == ""


def test_dead_reasons_say_what_happened():
    assert check.judge("dns_nxdomain", False, None, True)[1] == "domain does not exist"
    assert "Not Found" in check.judge("http_4xx", False, shot("Not Found", 9), True)[1]
    assert "Tunnel error" in check.judge("http_5xx", False, shot("Cloudflare Tunnel error", 700), True)[1]
    assert "connect_timeout/unreachable" in check.judge("connect_timeout", False, shot(ok=False, reason="unreachable", reach="dead"), True)[1]


# ---------------------------------------------------------------------------------------
# the checker's own behaviour (without a lab)
# ---------------------------------------------------------------------------------------
def make_links_csv(path, rows):
    cols = ["first_seen_day", "why", "signal", "brand", "sector", "country_code", "tld", "lang", "url"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for day, brand, url in rows:
            w.writerow({"first_seen_day": day, "brand": brand, "url": url, "signal": "PH brand"})


def test_dry_run_lists_newest_first_defanged_and_never_calls_the_lab(tmp_path, capsys, monkeypatch):
    make_links_csv(tmp_path / "openphish_ph_strict.csv", [
        ("2026-01-01", "BDO Unibank", "http://older.example/ph"), ("2026-03-01", "BDO Unibank", "https://newest.example/ph"),
        ("2026-02-01", "Shopee", "http://shopee.example/"), ("2026-02-15", "Bank of the Philippine Islands", "https://mid.example/ph")])

    def boom(*a, **k):
        raise AssertionError("the lab must not be called in a dry run")
    monkeypatch.setattr(check.subprocess, "run", boom)
    assert check.main(["--out", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.index("newest") < out.index("mid") < out.index("older")           # newest first
    assert "hxxps://newest[.]example/ph" in out and "https://" not in out and "http://" not in out
    assert "shopee" not in out                                                   # not a Philippine bank brand
    check.main(["--out", str(tmp_path), "--dry-run", "--all-brands"])
    assert "shopee" in capsys.readouterr().out


def test_a_rerun_skips_links_that_already_have_a_result(tmp_path, capsys, monkeypatch):
    make_links_csv(tmp_path / "openphish_ph_strict.csv", [("2026-03-01", "BDO Unibank", "https://a.example/ph"),
                                                          ("2026-02-01", "BDO Unibank", "https://b.example/ph")])
    check.write_results(tmp_path, [{"alive": "no", "why": "x", "first_seen_day": "2026-03-01", "brand": "BDO Unibank",
                                    "url": "https://a.example/ph", "lab_batch": "op0101n01"}])
    check.main(["--out", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "b[.]example" in out and "a[.]example" not in out and "1 already checked" in out
    check.main(["--out", str(tmp_path), "--dry-run", "--again"])
    assert "a[.]example" in capsys.readouterr().out


def test_not_dead_file_holds_only_yes_and_maybe(tmp_path):
    rows = [{"alive": a, "why": "w", "url": f"http://{a}{i}.example/", "brand": "BDO Unibank", "first_seen_day": "d", "lab_batch": f"b{i}"}
            for i, a in enumerate(["yes", "no", "maybe", "unknown", "no"])]
    check.write_results(tmp_path, rows)
    live = list(csv.DictReader(open(tmp_path / "openphish_ph_bank_NOT_DEAD.csv", encoding="utf-8")))
    allrows = list(csv.DictReader(open(tmp_path / "openphish_ph_bank_checked_ALL.csv", encoding="utf-8")))
    assert [r["alive"] for r in live] == ["yes", "maybe"] and len(allrows) == 5


def test_missing_link_list_is_a_clear_error(tmp_path, capsys):
    assert check.main(["--out", str(tmp_path), "--dry-run"]) == 2
    assert "openphish_ph_links.py" in capsys.readouterr().err


def test_again_replaces_only_the_rechecked_rows_and_keeps_the_rest(tmp_path, monkeypatch):
    """--again must never throw away the other results; and a re-checked page that now says it was removed turns dead."""
    import json

    make_links_csv(tmp_path / "openphish_ph_strict.csv", [("2026-03-01", "BDO Unibank", "https://newest.example/ph"),
                                                          ("2026-02-01", "BDO Unibank", "https://other.example/ph")])
    old = [{"alive": "yes", "why": "page loads", "first_seen_day": "2026-03-01", "brand": "BDO Unibank", "url": "https://newest.example/ph",
            "lab_batch": "op0101n01"},
           {"alive": "no", "why": "domain does not exist", "first_seen_day": "2026-02-01", "brand": "BDO Unibank", "url": "https://other.example/ph",
            "lab_batch": "op0101n02"}]
    check.write_results(tmp_path, old)

    def fake_lab(cmd, **kw):                                           # stands in for run_live_azure.ps1: writes what the lab would
        batch, clean = cmd[cmd.index("-Batch") + 1], cmd[cmd.index("-CleanDir") + 1]
        os.makedirs(os.path.join(clean, "shots", batch), exist_ok=True)
        with open(os.path.join(clean, f"{batch}.clean.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"fail_reason": "none", "html_available": True, "final_url": "hxxps://newest[.]example/ph"}) + "\n")
        with open(os.path.join(clean, "shots", batch, "shots.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"ok": True, "reach": "live", "reason": "ok", "title": "My Bank", "text_chars": 90, "width": 412, "height": 900,
                                 "pages": [], "text_head": "Sorry, this page has been removed."}) + "\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(check.subprocess, "run", fake_lab)
    assert check.main(["--out", str(tmp_path), "--again", "--limit", "1"]) == 0
    rows = {r["url"]: r for r in csv.DictReader(open(tmp_path / "openphish_ph_bank_checked_ALL.csv", encoding="utf-8"))}
    assert len(rows) == 2                                                                # nothing was thrown away
    assert rows["https://other.example/ph"]["why"] == "domain does not exist"            # untouched
    assert rows["https://newest.example/ph"]["alive"] == "no"                           # was "yes": the page now says it was removed
    assert "has been removed" in rows["https://newest.example/ph"]["why"]
    assert "has been removed" in rows["https://newest.example/ph"]["page_text_start"]
    live = list(csv.DictReader(open(tmp_path / "openphish_ph_bank_NOT_DEAD.csv", encoding="utf-8")))
    assert live == []                                                                    # and so it leaves the not-dead file


def test_only_dead_rechecks_the_dead_and_skips_the_live(tmp_path, capsys):
    make_links_csv(tmp_path / "openphish_ph_strict.csv", [("2026-03-01", "BDO Unibank", "https://alive.example/ph"),
                                                          ("2026-02-01", "BDO Unibank", "https://dead.example/ph"),
                                                          ("2026-01-01", "BDO Unibank", "https://never-checked.example/ph")])
    base = {"why": "w", "first_seen_day": "d", "brand": "BDO Unibank"}
    check.write_results(tmp_path, [{**base, "alive": "yes", "url": "https://alive.example/ph", "lab_batch": "op0101n01"},
                                   {**base, "alive": "no", "url": "https://dead.example/ph", "lab_batch": "op0101n02"}])
    check.main(["--out", str(tmp_path), "--dry-run", "--again", "--only-dead"])
    out = capsys.readouterr().out
    assert "dead[.]example" in out and "alive[.]example" not in out and "never-checked" not in out
