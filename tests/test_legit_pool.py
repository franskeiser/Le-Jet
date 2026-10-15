"""scripts/build_legit_pool.py: a pool of legitimate-site links sampled from a Tranco-style ranking. No network."""
from __future__ import annotations

import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_labeled_input as bli                            # noqa: E402
import build_legit_pool as blp                               # noqa: E402


def fake_rows(n=30000):
    return [(i, f"site{i}.example{i % 7}.com") for i in range(1, n + 1)]


def test_read_tranco_skips_headers_duplicates_bad_names_and_ranks_beyond_the_limit(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("\n".join(["rank,domain", "1,google.com", "2,Facebook.COM", "3,google.com", "4,not a domain", "5,xn--80ak6aa92e.com",
                            "6,x", "200000,far.example.org"]) + "\n", encoding="utf-8")
    assert blp.read_tranco(str(f), limit=100000) == [(1, "google.com"), (2, "facebook.com"), (5, "xn--80ak6aa92e.com")]


def test_infrastructure_and_free_hosting_platform_roots_are_not_usable():
    for bad in ("a248.e.akamai.net", "d1.cloudfront.net", "ads.doubleclick.net", "fonts.googleapis.com", "pages.dev",
                "github.io", "something.cdn77.org", "static.fbcdn.net"):
        assert not blp.usable(bad), bad
    for abused in ("shop-deals.top", "fresh.xyz", "login-now.icu", "free.tk"):
        assert not blp.usable(abused), abused
    for unwanted in ("xxxgr.net", "bestcasino.com", "freeporn.example", "megabingo.net"):
        assert not blp.usable(unwanted), unwanted
    for good in ("wikipedia.org", "bdo.com.ph", "example.org", "nasa.gov", "sussex.ac.uk", "essex.gov.uk"):
        assert blp.usable(good), good


def test_the_sample_follows_the_band_weights_without_repeats_and_is_reproducible():
    rows = fake_rows()
    pool = blp.sample_pool(rows, 1000, seed=3)
    assert len(pool) == 1000 and len({d for _, d, _ in pool}) == 1000
    counts = {}
    for _r, _d, band in pool:
        counts[band] = counts.get(band, 0) + 1
    assert counts == {"1-5000": 150, "5001-20000": 350, "20001-100000": 500}
    assert pool == blp.sample_pool(rows, 1000, seed=3)
    assert pool != blp.sample_pool(rows, 1000, seed=4)
    assert all(lo <= r <= hi for r, _d, band in pool for lo, hi in [tuple(map(int, band.split("-")))])


def test_a_band_with_too_few_domains_gives_its_share_to_the_others():
    rows = [(i, f"s{i}.example.com") for i in range(1, 3001)]                   # only the first band has domains
    pool = blp.sample_pool(rows, 2000, seed=1)
    assert len(pool) == 2000 and {b for _, _, b in pool} == {"1-5000"}
    assert len(blp.sample_pool(rows, 99999, seed=1)) == 3000                      # never more than exist


def test_main_writes_a_defanged_pool_and_a_csv(tmp_path):
    t = tmp_path / "tranco_top100k_2026-10-15.csv"
    t.write_text("\n".join(f"{r},{d}" for r, d in fake_rows(25000)) + "\n", encoding="utf-8")
    out = tmp_path / "pool.txt"
    assert blp.main(["--tranco", str(t), "--n", "200", "--out", str(out)]) == 0
    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    assert len(lines) == 200 and all(l.startswith("hxxps://") and l.endswith("/") and "[.]" in l for l in lines)
    assert not any(l.lower().startswith("http") for l in lines)
    rows = list(csv.DictReader(open(tmp_path / "pool.csv", encoding="utf-8")))
    assert len(rows) == 200 and set(rows[0]) == {"domain", "rank", "band"}


def test_the_builder_adds_extra_legitimate_links_from_the_large_pool_without_any_phishing_partner(tmp_path):
    # a master with nothing new, a table that already exists, and a pool of 30 sites
    master = tmp_path / "master.csv"
    master.write_text("link,source,trust,score\nhttps://evil-one.xyz/a,phishunt,feed-listed,5\n", encoding="utf-8")
    sms = tmp_path / "sms.csv"
    sms.write_text("sms,label\nhi,legitimate\n", encoding="utf-8")
    curated = tmp_path / "curated.txt"
    curated.write_text("hxxps://www[.]curated-one[.]org/\n", encoding="utf-8")
    pool = tmp_path / "pool.txt"
    pool.write_text("# header\n" + "\n".join(f"hxxps://tr{i}[.]org/" for i in range(30)) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    base = ["--master", str(master), "--sms", str(sms), "--curated", str(curated), "--out-dir", str(out), "--tranco-pool", str(pool)]
    assert bli.main(base + ["--date", "2026-10-20"]) == 0                      # the first run: 1 phishing + 1 partner
    rows = list(csv.DictReader(open(out / "feature_input.csv", encoding="utf-8")))
    assert [r["status"] for r in rows].count("phishing") == 1 and [r["status"] for r in rows].count("legitimate") == 1
    assert bli.main(base + ["--date", "2026-10-21", "--add-legit", "12"]) == 0     # nothing new in the master: legit only
    rows = list(csv.DictReader(open(out / "feature_input.csv", encoding="utf-8")))
    added = [r for r in rows if r["date_added"] == "2026-10-21"]
    assert len(added) == 12 and {r["status"] for r in added} == {"legitimate"}
    assert {r["source"] for r in added} == {"legit_tranco"} and len({r["url_id"] for r in rows}) == len(rows)
    parts = sorted(f for f in os.listdir(out) if f.endswith(".txt"))
    assert any(len((out / p).read_text().split()) == 12 for p in parts)           # the lab gets exactly the 12 new links
    assert bli.main(base + ["--date", "2026-10-22", "--add-legit", "100"]) == 0    # more than the pool has: takes what exists
    rows = list(csv.DictReader(open(out / "feature_input.csv", encoding="utf-8")))
    assert sum(r["source"] == "legit_tranco" for r in rows) == 30


def test_a_site_that_is_also_a_phishing_domain_is_never_used_as_legitimate(tmp_path):
    master = tmp_path / "master.csv"
    master.write_text("link,source,trust,score\nhttps://tr3.org/login,phishunt,feed-listed,5\n", encoding="utf-8")
    sms = tmp_path / "sms.csv"
    sms.write_text("sms,label\nhi,legitimate\n", encoding="utf-8")
    curated = tmp_path / "curated.txt"
    curated.write_text("# nothing here\n", encoding="utf-8")
    pool = tmp_path / "pool.txt"
    pool.write_text("\n".join(f"hxxps://tr{i}[.]org/" for i in range(10)) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    assert bli.main(["--master", str(master), "--sms", str(sms), "--curated", str(curated), "--out-dir", str(out),
                     "--tranco-pool", str(pool), "--add-legit", "20"]) == 0
    rows = list(csv.DictReader(open(out / "feature_input.csv", encoding="utf-8")))
    legit = [r["url"] for r in rows if r["status"] == "legitimate"]
    assert legit and not any("tr3[.]org" in u for u in legit)                    # the phishing site's domain is excluded
    assert len(legit) == 9                                                       # the other nine, the partner included
