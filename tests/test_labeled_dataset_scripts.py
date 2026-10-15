"""
scripts/build_labeled_input.py (master smishing links + the same number of legitimate links, one labelled CSV and the
lab input files) and scripts/build_feature_dataset.py (lab results + labels -> `url, 73 features, status`).
No network, no real link is ever opened.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_feature_dataset as bfd                          # noqa: E402
import build_labeled_input as bli                            # noqa: E402
import make_push_file                                        # noqa: E402
from lejet import schema, urlnorm                            # noqa: E402

MASTER = [
    ("https://gcash-verify.top/login", "phishtank", "human-verified"),
    ("http://evil-parcel.xyz/track?id=1", "phishunt", "feed-listed"),
    ("https://paymaya-claim.com/", "manual:list", "smishing-verified"),
    ("http://bit.ly-fake.cc/abc", "openphish", "feed-listed"),
    ("https://shopee-refund.site/x", "phishunt", "feed-listed"),
]
SMS = [
    ("Your parcel: https://www.lbcexpress.com/track now", "legitimate"),
    ("Pay at https://www.bdo.com.ph/pay or https://www.bdo.com.ph/help", "legitimate"),
    ("see https://www.bdo.com.ph/a https://www.bdo.com.ph/b https://www.bdo.com.ph/c https://www.bdo.com.ph/d", "legitimate"),
    ("Hi from https://maya.ph/app", "legitimate"),
    ("click https://gcash-verify.top/other", "legitimate"),            # same domain as a master link: must be excluded
    ("WIN now https://scam-prize.top/", "smishing"),                    # a smishing message: never a legitimate link
    ("Claim https://gcash-10-year.lc/ today", "legitimate"),            # mislabelled in the SMS data: not an allowed sender
    ("no link in this one", "legitimate"),
]
CURATED = ["aaaa\thxxps://www[.]imf[.]org/", "bbbb\thxxps://github[.]com/", "cccc\thxxps://www[.]cdc[.]gov/",
           "dddd\thxxps://www[.]unicef[.]org/", "eeee\thxxps://zoom[.]us/signin", "ffff\thxxps://stackoverflow[.]com/"]


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    # the real allowlist names real senders; the fixture's senders are made up
    monkeypatch.setattr(bli, "SMS_LEGIT_ALLOW", {"lbcexpress.com", "bdo.com.ph", "maya.ph", "gcash-verify.top"})
    monkeypatch.setattr(bli, "DEFAULT_TRANCO_POOL", str(tmp_path / "no_such_pool.txt"))        # never the real pool in tests
    m = tmp_path / "master.csv"
    with open(m, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["link", "source", "trust", "score"])
        for link, source, trust in MASTER:
            w.writerow([link, source, trust, 5])
    s = tmp_path / "sms.csv"
    with open(s, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sms", "label"])
        w.writerows(SMS)
    c = tmp_path / "curated.txt"
    c.write_text("\n".join(CURATED) + "\n", encoding="utf-8")
    return m, s, c, tmp_path / "out"


def run_input(inputs, *extra):
    m, s, c, out = inputs
    rc = bli.main(["--master", str(m), "--sms", str(s), "--curated", str(c), "--out-dir", str(out), *extra])
    files = sorted(os.listdir(out)) if out.exists() else []
    return rc, files, out


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_same_number_of_legitimate_links_and_everything_defanged(inputs):
    rc, files, out = run_input(inputs, "--chunk", "4")
    assert rc == 0
    csvs = [f for f in files if f.endswith(".csv")]
    assert len(csvs) == 1
    rows = read_csv(out / csvs[0])
    pos = [r for r in rows if r["status"] == "phishing"]
    neg = [r for r in rows if r["status"] == "legitimate"]
    assert len(pos) == len(MASTER) == 5 and len(neg) == 5                 # the same number
    assert all(r["url"].startswith(("hxxp://", "hxxps://")) and "." not in r["url"].split("//")[1].split("/")[0].replace("[.]", "")
               for r in rows)
    assert not any(r["url"].lower().startswith(("http://", "https://")) for r in rows)
    assert {r["trust"] for r in pos} == {"human-verified", "feed-listed", "smishing-verified"}
    assert all(r["trust"] == "" for r in neg)
    assert len({r["url_id"] for r in rows}) == len(rows)                  # no duplicates
    assert {r["source"].split(":")[0] for r in neg} <= {"legit_sms_real", "legit_curated_homepage"}


def test_legitimate_links_never_share_a_site_with_a_master_link_nor_come_from_smishing_messages(inputs):
    _, _, out = run_input(inputs)
    rows = read_csv(out / next(f for f in os.listdir(out) if f.endswith(".csv")))
    legit_urls = " ".join(r["url"] for r in rows if r["status"] == "legitimate")
    assert "gcash-verify" not in legit_urls and "scam-prize" not in legit_urls


def test_real_sms_links_from_senders_that_are_not_allowed_are_never_used(inputs):
    _, _, out = run_input(inputs)
    rows = read_csv(out / next(f for f in os.listdir(out) if f.endswith(".csv")))
    assert "gcash-10-year" not in " ".join(r["url"] for r in rows)        # a lookalike the SMS data calls legitimate
    m, s, c, _ = inputs
    assert all("gcash-10-year" not in u for u in bli.sms_legit_links(str(s)))
    assert bli.sms_legit_links(str(s), allow=set()) == []
    assert "lbcexpress.com" in " ".join(bli.sms_legit_links(str(s)))


def test_the_real_allowlist_contains_no_lookalike_and_no_generic_shortener():
    for d in bli.SMS_LEGIT_ALLOW:
        assert not any(t in d for t in ("gcash-", "bdo-", "-ph", "verify", "claim")), d
    assert not {"bit.ly", "tinyurl.com", "goo.gl", "onelink.to", "page.link", "gcashapp.page.link"} & bli.SMS_LEGIT_ALLOW


def test_a_site_is_capped_but_the_cap_is_raised_when_the_pools_are_short(inputs):
    # bdo.com.ph has 5 real links; with --per-domain 1 only one is used while other sites can fill the count
    _, _, out = run_input(inputs, "--per-domain", "1")
    rows = read_csv(out / next(f for f in os.listdir(out) if f.endswith(".csv")))
    assert sum("bdo[.]com[.]ph" in r["url"] for r in rows) == 1
    # a large master needs more than the pools offer: the cap is raised step by step, then the shortfall is reported
    pool_a = [f"http://site{i % 2}.example/p{i}" for i in range(6)]
    got = bli.pick_legit(pool_a, [], 6, 1, random.Random(1), set(), set())
    assert len(got) == 6 and len({u for u, _ in got}) == 6
    short = bli.pick_legit(pool_a, [], 50, 1, random.Random(1), set(), set())
    assert len(short) == 6


def test_every_part_has_both_classes_and_the_parts_cover_every_link_once(inputs):
    _, files, out = run_input(inputs, "--chunk", "5")
    parts = sorted(f for f in files if f.endswith(".txt"))
    assert len(parts) == 2
    lines = [l for p in parts for l in (out / p).read_text(encoding="utf-8").splitlines()]
    rows = read_csv(out / next(f for f in files if f.endswith(".csv")))
    assert sorted(lines) == sorted(r["url"] for r in rows)
    first = set((out / parts[0]).read_text(encoding="utf-8").splitlines())
    labels = {r["url"]: r["status"] for r in rows}
    assert {labels[u] for u in first} == {"phishing", "legitimate"}


def test_same_seed_same_files(inputs, tmp_path):
    _, _, out1 = run_input(inputs)
    m, s, c, _ = inputs
    out2 = tmp_path / "out2"
    bli.main(["--master", str(m), "--sms", str(s), "--curated", str(c), "--out-dir", str(out2)])
    f = next(f for f in os.listdir(out1) if f.endswith(".csv"))
    assert (out1 / f).read_bytes() == (out2 / f).read_bytes()


def test_url_ids_are_the_ones_the_lab_will_use(inputs):
    _, _, out = run_input(inputs)
    for r in read_csv(out / next(f for f in os.listdir(out) if f.endswith(".csv"))):
        assert r["url_id"] == urlnorm.compute_url_id(make_push_file.refang(r["url"]))


# ---------------------------------------------------------------------------------------
# build_feature_dataset.py
# ---------------------------------------------------------------------------------------
def lab_record(url_defanged, fetched=True, ts="2026-10-15T01:00:00Z", **features):
    raw = make_push_file.refang(url_defanged)
    row = schema.new_row(urlnorm.compute_url_id(raw), raw, "feat")
    row.update(fetch_ok=int(fetched), html_available=int(fetched), page_fetched=int(fetched), fetch_ts=ts,
               fail_reason="none" if fetched else "http_4xx", final_reg_domain="example.com")
    if fetched:
        row.update(nb_hyperlinks=7, ratio_intHyperlinks=0.5, hf_safe_anchor=40.0)
    row.update(features)
    return schema.export_record(row)


def label_file(tmp_path, items):
    p = tmp_path / "labels.csv"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["url", "status", "source", "trust", "url_id"])
        for url, status in items:
            w.writerow([url, status, "x", "", urlnorm.compute_url_id(make_push_file.refang(url))])
    return p


def clean_file(tmp_path, recs, name="a.clean.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="ascii")
    return p


ITEMS = [("hxxps://a1[.]top/x", "phishing"), ("hxxps://a2[.]top/x", "phishing"), ("hxxps://a3[.]top/x", "phishing"),
         ("hxxps://b1[.]org/", "legitimate"), ("hxxps://b2[.]org/", "legitimate"), ("hxxps://b3[.]org/", "legitimate")]


def test_the_columns_are_exactly_those_of_the_trimmed_public_set(tmp_path):
    labels = label_file(tmp_path, ITEMS)
    clean = clean_file(tmp_path, [lab_record(u) for u, _ in ITEMS])
    out = tmp_path / "ds.csv"
    assert bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out)]) == 0
    with open(out, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["url"] + schema.FULL_FEATURES + ["status"] and len(rows[0]) == 75
    assert len(rows) == 1 + 6 and all(len(r) == 75 for r in rows)
    trimmed = r"D:\LeJetData\datasets\train-1_73features.csv"
    if os.path.exists(trimmed):                                   # the host has the trimmed public set: same header
        assert rows[0] == next(csv.reader(open(trimmed, encoding="utf-8")))
    head = dict(zip(rows[0], rows[1]))
    assert head["url"].startswith("http") and "[.]" not in head["url"]          # the REAL link, like the public set
    assert head["status"] in ("phishing", "legitimate")
    assert head["nb_hyperlinks"] == "7" and head["safe_anchor"] == "40.0" and head["page_rank"] == "-1"


def test_unfetched_pages_are_left_out_unless_all_is_asked_for(tmp_path):
    labels = label_file(tmp_path, ITEMS)
    recs = [lab_record(u, fetched=(i != 0)) for i, (u, _) in enumerate(ITEMS)]
    clean = clean_file(tmp_path, recs)
    out = tmp_path / "ds.csv"
    bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out)])
    assert len(list(csv.reader(open(out, encoding="utf-8")))) == 1 + 5
    bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out), "--all"])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 6
    unfetched = next(r for r in rows if r["url"] == make_push_file.refang(ITEMS[0][0]))
    assert unfetched["nb_hyperlinks"] == "-1" and unfetched["safe_anchor"] == "-1"       # page columns unknown


def test_balance_makes_the_classes_equal_with_a_fixed_seed(tmp_path):
    items = ITEMS + [("hxxps://a4[.]top/x", "phishing"), ("hxxps://a5[.]top/x", "phishing")]
    labels = label_file(tmp_path, items)
    clean = clean_file(tmp_path, [lab_record(u) for u, _ in items])
    outs = []
    for n in (1, 2):
        out = tmp_path / f"ds{n}.csv"
        bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out), "--balance"])
        outs.append(out)
    rows = list(csv.DictReader(open(outs[0], encoding="utf-8")))
    assert [r["status"] for r in rows].count("phishing") == [r["status"] for r in rows].count("legitimate") == 3
    assert outs[0].read_bytes() == outs[1].read_bytes()


def test_the_newest_result_wins_and_unknown_links_are_ignored(tmp_path):
    labels = label_file(tmp_path, ITEMS[:2])
    old = lab_record(ITEMS[0][0], ts="2026-10-14T01:00:00Z", nb_hyperlinks=1)
    new = lab_record(ITEMS[0][0], ts="2026-10-15T01:00:00Z", nb_hyperlinks=99)
    stranger = lab_record("hxxps://stranger[.]net/")
    c1, c2 = clean_file(tmp_path, [new], "n.clean.jsonl"), clean_file(tmp_path, [old, stranger, lab_record(ITEMS[1][0])], "o.clean.jsonl")
    out = tmp_path / "ds.csv"
    bfd.main(["--labels", str(labels), "--clean", str(c1), str(c2), "--out", str(out)])
    rows = {r["url"]: r for r in csv.DictReader(open(out, encoding="utf-8"))}
    first, second = make_push_file.refang(ITEMS[0][0]), make_push_file.refang(ITEMS[1][0])
    assert set(rows) == {first, second} and rows[first]["nb_hyperlinks"] == "99"


def test_page_rank_is_added_from_the_host_file_and_old_format_results_are_refused(tmp_path):
    labels = label_file(tmp_path, ITEMS[:1])
    rec = lab_record(ITEMS[0][0])
    pr = tmp_path / "pr.csv"
    pr.write_text(f"url_id,page_rank\n{rec['url_id']},6\n", encoding="utf-8")
    out = tmp_path / "ds.csv"
    bfd.main(["--labels", str(labels), "--clean", str(clean_file(tmp_path, [rec])), "--out", str(out), "--page-rank-csv", str(pr)])
    assert next(csv.DictReader(open(out, encoding="utf-8")))["page_rank"] == "6"
    old = {k: v for k, v in rec.items() if k != "features_full"}
    with pytest.raises(SystemExit):
        bfd.main(["--labels", str(labels), "--clean", str(clean_file(tmp_path, [old], "old.clean.jsonl")), "--out", str(out)])


def test_a_file_that_is_not_a_label_file_is_refused(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        bfd.read_labels(str(bad))


def test_the_url_column_is_the_exact_link_the_lab_fetched_and_defanged_only_on_request(tmp_path):
    labels = label_file(tmp_path, ITEMS)
    clean = clean_file(tmp_path, [lab_record(u) for u, _ in ITEMS])
    real, fanged = tmp_path / "real.csv", tmp_path / "fanged.csv"
    bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(real)])
    bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(fanged), "--defanged"])
    real_rows = list(csv.DictReader(open(real, encoding="utf-8")))
    fanged_rows = list(csv.DictReader(open(fanged, encoding="utf-8")))
    assert {r["url"] for r in fanged_rows} == {u for u, _ in ITEMS}                     # hxxps://a1[.]top/x ...
    assert {r["url"] for r in real_rows} == {"https://a1.top/x", "https://a2.top/x", "https://a3.top/x",
                                             "https://b1.org/", "https://b2.org/", "https://b3.org/"}
    # the real link hashes to the lab's url_id: it is exactly what was fetched
    ids = {r["url"]: urlnorm.compute_url_id(r["url"]) for r in real_rows}
    assert set(ids.values()) == {lab_record(u)["url_id"] for u, _ in ITEMS}


def test_lean_drops_constant_and_near_identical_columns_and_keeps_the_rest(tmp_path):
    pytest.importorskip("sklearn")
    items = [(f"hxxps://p{i}[.]top/x", "phishing") for i in range(6)] + [(f"hxxps://l{i}[.]org/", "legitimate") for i in range(6)]
    labels = label_file(tmp_path, items)
    recs = []
    for i, (u, status) in enumerate(items):
        k = 10 + i if status == "phishing" else 100 + 7 * i          # nb_hyperlinks differs between and within the classes
        recs.append(lab_record(u, nb_hyperlinks=k, links_in_tags=float(k) * 2, ratio_intMedia=float(i % 3)))
    out = tmp_path / "ds.csv"
    assert bfd.main(["--labels", str(labels), "--clean", str(clean_file(tmp_path, recs)), "--out", str(out), "--lean"]) == 0
    lean = list(csv.reader(open(tmp_path / "ds_lean.csv", encoding="utf-8")))
    head = lean[0]
    assert head[0] == "url" and head[-1] == "status"
    assert "nb_hyperlinks" in head and "links_in_tags" not in head           # monotone copies: one is kept
    assert "page_rank" not in head and "nb_at" not in head                    # constant (-1 / 0 in every row)
    assert all(len(r) == len(head) for r in lean)
    full = list(csv.reader(open(out, encoding="utf-8")))
    assert len(full[0]) == 75 and len(lean) == len(full)                      # the full file is untouched


def test_lean_also_drops_columns_that_are_almost_always_the_same_unless_min_variation_is_zero(tmp_path):
    pytest.importorskip("sklearn")
    items = [(f"hxxps://p{i}[.]top/x", "phishing") for i in range(30)] + [(f"hxxps://l{i}[.]org/", "legitimate") for i in range(30)]
    labels = label_file(tmp_path, items)
    recs = []
    for i, (u, status) in enumerate(items):
        # nb_hyperlinks varies a lot; iframe is 1 in a single row of 60 (1.7%); login_form is 1 in 10 of 60 (17%)
        recs.append(lab_record(u, nb_hyperlinks=(i * 37) % 101, iframe=int(i == 3), login_form=int(i % 6 == 0)))
    clean = clean_file(tmp_path, recs)
    out = tmp_path / "ds.csv"
    assert bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out), "--lean"]) == 0
    head = next(csv.reader(open(tmp_path / "ds_lean.csv", encoding="utf-8")))
    assert "iframe" not in head and "login_form" in head and "nb_hyperlinks" in head
    out2 = tmp_path / "ds2.csv"
    assert bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out2), "--lean", "--min-variation", "0"]) == 0
    assert "iframe" in next(csv.reader(open(tmp_path / "ds2_lean.csv", encoding="utf-8")))


# ---------------------------------------------------------------------------------------
# the table that grows: feature_input.csv with date_added
# ---------------------------------------------------------------------------------------
def write_master(path, links, live=None):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["link", "source", "trust", "score", "lab_page_fetched"])
        for i, link in enumerate(links):
            w.writerow([link, "phishunt", "feed-listed", 5, "" if live is None else live[i]])


def table(out):
    return read_csv(out / "feature_input.csv")


def parts_of(out):
    return sorted(f for f in os.listdir(out) if f.startswith("feature_input_") and f.endswith(".txt"))


def test_the_first_run_creates_one_table_and_stamps_every_row_with_the_date(inputs):
    rc, files, out = run_input(inputs, "--date", "2026-10-20")
    assert rc == 0 and "feature_input.csv" in files
    rows = table(out)
    assert list(rows[0]) == ["url", "status", "source", "trust", "url_id", "date_added"]
    assert {r["date_added"] for r in rows} == {"2026-10-20"} and len(rows) == 10


def test_a_second_run_adds_only_the_new_links_with_the_new_date_and_leaves_old_rows_alone(inputs, tmp_path):
    m, s, c, out = inputs
    run_input(inputs, "--date", "2026-10-20")
    before = table(out)
    first_parts = parts_of(out)
    write_master(m, [l for l, _, _ in MASTER] + ["https://brand-new-scam.xyz/pay"])
    rc, files, _ = run_input(inputs, "--date", "2026-10-22")
    after = table(out)
    assert rc == 0 and len(after) == len(before) + 2                       # 1 new smishing + 1 new legitimate
    assert after[:len(before)] == before                                   # old rows: same, same order, same dates
    added = after[len(before):]
    assert {r["date_added"] for r in added} == {"2026-10-22"}
    assert sorted(r["status"] for r in added) == ["legitimate", "phishing"]
    assert len({r["url_id"] for r in after}) == len(after)                 # still no duplicates
    new_parts = [p for p in parts_of(out) if p not in first_parts]
    assert len(new_parts) == 1 and len((out / new_parts[0]).read_text().split()) == 2    # the lab gets only the new links


def test_nothing_new_changes_nothing(inputs, capsys):
    run_input(inputs, "--date", "2026-10-20")
    m, s, c, out = inputs
    snapshot = (out / "feature_input.csv").read_bytes()
    parts = parts_of(out)
    rc, _, _ = run_input(inputs, "--date", "2026-10-25")
    assert rc == 0 and (out / "feature_input.csv").read_bytes() == snapshot and parts_of(out) == parts
    assert "Nothing new to add" in capsys.readouterr().out


def test_dry_run_writes_nothing(inputs):
    m, s, c, out = inputs
    rc, files, _ = run_input(inputs, "--dry-run")
    assert rc == 0 and files == []


def test_the_older_per_day_file_becomes_the_start_of_the_table(inputs):
    m, s, c, out = inputs
    out.mkdir()
    legacy = out / "feature_input_20261014.csv"
    with open(legacy, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["url", "status", "source", "trust", "url_id"])
        for link, source, trust in MASTER:
            d = bli.defang(link)
            w.writerow([d, "phishing", "smishing_master:" + source, trust, bli.link_id(link)])
    write_master(m, [l for l, _, _ in MASTER] + ["https://another-scam.top/x"])
    rc, files, _ = run_input(inputs, "--date", "2026-10-20")
    rows = table(out)
    assert rc == 0 and len(rows) == 5 + 2
    assert [r["date_added"] for r in rows[:5]] == ["2026-10-14"] * 5       # taken from the file's name
    assert {r["date_added"] for r in rows[5:]} == {"2026-10-20"}
    assert legacy.exists()                                                  # the old file is left as it was


def test_only_live_hands_over_just_the_links_the_hunters_lab_found_live(inputs):
    m, s, c, out = inputs
    links = [l for l, _, _ in MASTER]
    write_master(m, links, live=["1", "0", "1", "", "1"])
    rc, _, _ = run_input(inputs, "--only-live", "--date", "2026-10-20")
    rows = [r for r in table(out) if r["status"] == "phishing"]
    assert rc == 0 and len(rows) == 3
    assert {r["url"] for r in rows} == {bli.defang(links[0]), bli.defang(links[2]), bli.defang(links[4])}


def test_without_enough_legitimate_links_only_as_many_smishing_links_are_added_and_the_rest_wait(inputs, capsys):
    m, s, c, out = inputs
    write_master(m, [f"https://scam{i}.xyz/p" for i in range(30)])           # far more than the 10 or so legit links
    rc, _, _ = run_input(inputs, "--date", "2026-10-20")
    text = capsys.readouterr().out
    rows = table(out)
    n_pos, n_neg = (sum(r["status"] == k for r in rows) for k in ("phishing", "legitimate"))
    assert rc == 0 and n_pos == n_neg and 0 < n_pos < 30                       # balanced: no lab time wasted
    assert "WAITING" in text and "--curated" in text
    # the waiting links were NOT added, so a later run (with more legitimate links) picks them up
    extra = out.parent / "more_legit.txt"
    extra.write_text("\n".join(f"hxxps://www[.]legit-extra-{i}[.]org/" for i in range(40)) + "\n", encoding="utf-8")
    rc, _, _ = run_input(inputs, "--date", "2026-10-21", "--curated", str(c), str(extra))
    rows = table(out)
    assert sum(r["status"] == "phishing" for r in rows) == 30 == sum(r["status"] == "legitimate" for r in rows)
    assert {r["date_added"] for r in rows if r["status"] == "phishing"} == {"2026-10-20", "2026-10-21"}


def test_one_hosting_platform_is_capped_and_the_selection_takes_the_platforms_in_turn(inputs, capsys):
    m, s, c, out = inputs
    kit = [f"https://kit{i}.pages.dev/" for i in range(12)]                    # one platform, 12 near-copies
    others = ["https://scam-a.xyz/", "https://scam-b.top/", "https://x.github.io/pay"]
    write_master(m, kit + others)
    rc, _, _ = run_input(inputs, "--date", "2026-10-20", "--platform-cap", "3")
    text = capsys.readouterr().out
    pos = [r["url"] for r in table(out) if r["status"] == "phishing"]
    assert rc == 0 and sum("pages[.]dev" in u for u in pos) == 3                # capped at 3
    assert all(any(o.split("//")[1].split("/")[0].replace(".", "[.]") in u for u in pos) for o in others)
    assert "held back by the platform cap" in text and "9 smishing" in text      # 12 - 3 stay in the master


def test_the_cap_counts_what_the_table_already_holds_and_zero_switches_it_off(inputs):
    m, s, c, out = inputs
    write_master(m, [f"https://k{i}.pages.dev/" for i in range(3)])
    run_input(inputs, "--date", "2026-10-20", "--platform-cap", "4")
    write_master(m, [f"https://k{i}.pages.dev/" for i in range(8)])
    run_input(inputs, "--date", "2026-10-21", "--platform-cap", "4")
    assert sum("pages[.]dev" in r["url"] for r in table(out) if r["status"] == "phishing") == 4     # 3 + 1, not 3 + 4
    run_input(inputs, "--date", "2026-10-22", "--platform-cap", "0")
    assert sum("pages[.]dev" in r["url"] for r in table(out) if r["status"] == "phishing") == 8


def test_confirmed_links_are_chosen_before_feed_links_inside_a_platform(inputs):
    m, s, c, out = inputs
    with open(m, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["link", "source", "trust", "score", "lab_page_fetched"])
        for i in range(6):
            w.writerow([f"https://f{i}.pages.dev/", "openphish_feed", "feed-listed", 5, ""])
        w.writerow(["https://mine.pages.dev/", "manual:list", "smishing-verified", 5, ""])
    run_input(inputs, "--date", "2026-10-20", "--platform-cap", "2")
    pos = [r for r in table(out) if r["status"] == "phishing"]
    assert any(r["trust"] == "smishing-verified" for r in pos) and len(pos) == 2


def test_group_key_names_the_platform_or_the_site():
    assert bli.group_key("https://abc.pages.dev/x") == "pages.dev"
    assert bli.group_key("http://x.github.io/a") == "github.io"
    assert bli.group_key("https://paymaya-claim.com/") == "paymaya-claim.com"
    assert bli.group_key("http://1.2.3.4/login") == bli.reg_domain("http://1.2.3.4/login")


def test_when_excel_holds_the_table_the_new_one_goes_to_an_unsaved_copy_and_is_used_next_time(inputs, monkeypatch):
    m, s, c, out = inputs
    run_input(inputs, "--date", "2026-10-20")
    write_master(m, [l for l, _, _ in MASTER] + ["https://scam-a.xyz/x"])
    real_replace = os.replace

    def locked(src, dst):
        if str(dst).endswith("feature_input.csv"):
            raise PermissionError("locked by Excel")
        return real_replace(src, dst)
    monkeypatch.setattr(bli.os, "replace", locked)
    run_input(inputs, "--date", "2026-10-22")
    assert (out / "feature_input_UNSAVED_COPY.csv").exists()
    monkeypatch.setattr(bli.os, "replace", real_replace)
    write_master(m, [l for l, _, _ in MASTER] + ["https://scam-a.xyz/x", "https://scam-b.xyz/x"])
    run_input(inputs, "--date", "2026-10-23")
    rows = table(out)
    assert not (out / "feature_input_UNSAVED_COPY.csv").exists()            # merged back into the table
    assert {r["date_added"] for r in rows} == {"2026-10-20", "2026-10-22", "2026-10-23"} and len(rows) == 10 + 2 + 2


# ---------------------------------------------------------------------------------------
# build_feature_dataset.py: platform cap, stable name, locked file
# ---------------------------------------------------------------------------------------
def test_the_dataset_builder_caps_phishing_rows_per_platform_and_leaves_legitimate_rows_alone(tmp_path):
    items = ([(f"hxxps://k{i}[.]pages[.]dev/", "phishing") for i in range(8)]
             + [("hxxps://scam-a[.]xyz/", "phishing"), ("hxxps://scam-b[.]top/", "phishing")]
             + [(f"hxxps://l{i}[.]pages[.]dev/", "legitimate") for i in range(5)])
    labels = label_file(tmp_path, items)
    clean = clean_file(tmp_path, [lab_record(u) for u, _ in items])
    out = tmp_path / "ds.csv"
    assert bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out), "--platform-cap", "3"]) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    phishing = [r["url"] for r in rows if r["status"] == "phishing"]
    assert sum("pages.dev" in u for u in phishing) == 3 and len(phishing) == 5          # 3 capped + the 2 others
    assert sum(r["status"] == "legitimate" for r in rows) == 5                          # legitimate rows: all kept
    out2 = tmp_path / "ds2.csv"
    bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out2), "--platform-cap", "0"])
    assert sum(r["status"] == "phishing" for r in csv.DictReader(open(out2, encoding="utf-8"))) == 10
    # fixed seed: the same rows every time
    out3 = tmp_path / "ds3.csv"
    bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out3), "--platform-cap", "3"])
    assert out.read_bytes() == out3.read_bytes()


def test_the_default_output_is_one_stable_file_not_one_per_day(tmp_path, monkeypatch):
    labels = label_file(tmp_path, ITEMS)
    clean = clean_file(tmp_path, [lab_record(u) for u, _ in ITEMS])
    monkeypatch.setattr(bfd, "DEFAULT_OUT", str(tmp_path / "datasets"))
    assert bfd.main(["--labels", str(labels), "--clean", str(clean), "--lean"]) == 0
    assert sorted(os.listdir(tmp_path / "datasets")) == ["feature_dataset.csv", "feature_dataset_lean.csv"]


def test_an_open_excel_file_does_not_lose_the_run(tmp_path, monkeypatch, capsys):
    labels = label_file(tmp_path, ITEMS)
    clean = clean_file(tmp_path, [lab_record(u) for u, _ in ITEMS])
    out = tmp_path / "ds.csv"
    out.write_text("old", encoding="utf-8")
    real_open = open

    def locked_open(path, mode="r", *args, **kw):
        if str(path) == str(out) and "w" in mode:
            raise PermissionError("locked by Excel")
        return real_open(path, mode, *args, **kw)
    monkeypatch.setattr("builtins.open", locked_open)
    assert bfd.main(["--labels", str(labels), "--clean", str(clean), "--out", str(out)]) == 0
    monkeypatch.undo()
    assert out.read_text(encoding="utf-8") == "old"                                    # the locked file is untouched
    new = [f for f in os.listdir(tmp_path) if f.startswith("ds_NEW_")]
    assert len(new) == 1 and "open in another program" in capsys.readouterr().out
    assert len(list(csv.reader(open(tmp_path / new[0], encoding="utf-8")))) == 1 + 6


# ---------------------------------------------------------------------------------------
# --extra-phishing: a list of phishing links added as they are (PhishTank verified_online.csv)
# ---------------------------------------------------------------------------------------
def write_phishtank(path, urls):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phish_id", "url", "phish_detail_url", "submission_time", "verified", "verification_time", "online", "target"])
        for i, u in enumerate(urls):
            w.writerow([i, u, "https://example.invalid/d", "2026-10-14T00:00:00+00:00", "yes", "2026-10-14T00:10:00+00:00", "yes", "Other"])


def test_extra_phishing_adds_the_first_n_rows_without_partners_and_reports_what_it_skipped(inputs, tmp_path, capsys):
    m, s, c, out = inputs
    feed = tmp_path / "verified_online.csv"
    urls = ([f"https://kit{i}.pages.dev/login" for i in range(10)]                 # one platform: NOT capped here
            + ["https://has space.example/x", "ftp://nope.example/", "https://scam-a.xyz/"]
            + [f"https://later{i}.top/" for i in range(5)])
    write_phishtank(feed, urls)
    rc, files, _ = run_input(inputs, "--date", "2026-10-20", "--extra-phishing", str(feed), "--limit", "13",
                             "--extra-source", "phishtank_verified_online", "--extra-trust", "human-verified", "--chunk", "4")
    text = capsys.readouterr().out
    rows = [r for r in table(out) if r["source"].startswith("extra_phishing:")]
    assert rc == 0 and len(rows) == 11                                             # first 13 rows, minus 2 unusable
    assert {r["source"] for r in rows} == {"extra_phishing:phishtank_verified_online"}
    assert {r["trust"] for r in rows} == {"human-verified"} and {r["status"] for r in rows} == {"phishing"}
    assert {r["date_added"] for r in rows} == {"2026-10-20"}
    assert not any("later" in r["url"] for r in rows)                              # beyond --limit
    assert sum("pages[.]dev" in r["url"] for r in rows) == 10                      # the user's list is taken as it is
    assert "whitespace" in text and "not an http(s) link" in text and "no legitimate partners" in text
    parts = parts_of(out)
    lines = [l for p in parts for l in (out / p).read_text().split()]
    assert len(parts) >= 3 and set(r["url"] for r in rows) <= set(lines)           # the lab gets them, in chunks of 4


def test_extra_phishing_skips_links_already_in_the_table_and_a_second_run_adds_nothing(inputs, tmp_path):
    m, s, c, out = inputs
    run_input(inputs, "--date", "2026-10-20")
    already = MASTER[0][0]
    feed = tmp_path / "feed.txt"
    feed.write_text("\n".join([already, "https://fresh-one.xyz/a", "https://fresh-two.xyz/b"]) + "\n", encoding="utf-8")
    before = len(table(out))
    run_input(inputs, "--date", "2026-10-21", "--extra-phishing", str(feed))
    assert len(table(out)) == before + 2                                           # the one already known is skipped
    run_input(inputs, "--date", "2026-10-22", "--extra-phishing", str(feed))
    assert len(table(out)) == before + 2


def test_legit_ratio_sets_how_many_legitimate_partners_each_phishing_link_gets(inputs):
    m, s, c, out = inputs
    write_master(m, [f"https://scam{i}.xyz/p" for i in range(10)])
    extra = out.parent / "pool.txt"
    extra.write_text("\n".join(f"hxxps://www[.]ok{i}[.]org/" for i in range(40)) + "\n", encoding="utf-8")
    rc, _, _ = run_input(inputs, "--date", "2026-10-20", "--legit-ratio", "0.5", "--tranco-pool", str(extra))
    rows = table(out)
    assert rc == 0 and sum(r["status"] == "phishing" for r in rows) == 10 and sum(r["status"] == "legitimate" for r in rows) == 5
    write_master(m, [f"https://scam{i}.xyz/p" for i in range(20)])
    run_input(inputs, "--date", "2026-10-21", "--legit-ratio", "1.0", "--tranco-pool", str(extra))
    rows = table(out)
    assert sum(r["status"] == "phishing" for r in rows) == 20 and sum(r["status"] == "legitimate" for r in rows) == 15   # 5 + 10 more


def test_legit_ratio_with_a_short_pool_adds_only_the_phishing_links_it_can_partner(inputs, tmp_path, capsys):
    m, s, c, out = inputs
    write_master(m, [f"https://scam{i}.xyz/p" for i in range(30)])
    run_input(inputs, "--date", "2026-10-20", "--legit-ratio", "1.5")                  # 45 legit links wanted, ~15 exist
    rows = table(out)
    n_pos, n_neg = (sum(r["status"] == k for r in rows) for k in ("phishing", "legitimate"))
    assert 0 < n_pos < 30 and n_neg > 0
    assert n_neg >= 1.5 * n_pos - 1.5 and n_pos <= n_neg / 1.5 + 1                       # each phishing link kept its 1.5 partners
    assert "WAITING" in capsys.readouterr().out
