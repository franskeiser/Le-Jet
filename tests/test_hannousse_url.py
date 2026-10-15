"""
The 56 URL-based features of the public 87-feature set (lejet.features.hannousse_url).

Two kinds of test:
  * worked examples with hand-checked values (they also run inside the analysis VM);
  * a regression guard against the public set itself: every feature must keep reproducing the dataset's own
    value for at least the share of its 11,430 URLs recorded below (skipped where the dataset is not present,
    e.g. in the VM). The shares are what docs/hannousse_87_features.md reports.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from lejet import schema
from lejet.features import hannousse_url as H

DATASET = Path(__file__).resolve().parents[1] / "data" / "interim" / "hussein-train-and-test-merged.csv"

# Minimum share of the dataset's rows each feature must reproduce. Everything not listed must be >= 0.999.
MIN_SHARE = {
    "random_domain": 0.92,         # the nostril test: the original's exact input is unknown (~92% reproducible)
    "brand_in_subdomain": 0.994,   # the brand list is reconstructed from the dataset (see hannousse_lists.py)
    "brand_in_path": 0.996,
    "abnormal_subdomain": 0.995,   # a fitted rule, not the authors' regular expression
}
NOT_IN_URL_TEXT = {"nb_redirection", "nb_external_redirection"}   # they come from the recorded redirect chain


def test_the_names_are_the_public_sets_first_56_columns_in_order():
    assert H.URL_FEATURES == schema.DATASET_COLUMNS[:56]
    assert len(set(H.URL_FEATURES)) == 56
    # the table exports 53 of them: nb_or (always 0), nb_hyphens (weak) and nb_subdomains (a function of nb_dots)
    # are dropped; all 56 are still computed and checked against the public set
    assert [n for n in H.URL_FEATURES if n in schema.FULL_FEATURES] == [
        n for n in H.URL_FEATURES if n not in ("nb_or", "nb_hyphens", "nb_subdomains")]


def test_every_feature_is_a_finite_number():
    out = H.url_features("https://www.example.com/a/b?x=1&y=2")
    assert list(out) == H.URL_FEATURES
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in out.values())


def test_worked_example_with_a_subdomain_and_a_long_path():
    f = H.url_features("https://www.todayshomeowner.com/how-to-make-homemade-insecticidal-soap-for-plants/")
    assert f["length_url"] == 82 and f["length_hostname"] == 23
    assert f["nb_hyphens"] == 7 and f["nb_slash"] == 4 and f["nb_dots"] == 2 and f["nb_www"] == 1
    assert f["nb_subdomains"] == 2 and f["https_token"] == 0 and f["nb_dslash"] == 0
    # the dataset's own values for this URL
    assert (f["length_words_raw"], f["shortest_words_raw"], f["longest_words_raw"]) == (10, 2, 15)
    assert f["avg_words_raw"] == 6 and (f["shortest_word_host"], f["longest_word_host"], f["avg_word_host"]) == (3, 15, 9)


def test_host_words_are_the_domain_and_sub_domain_labels_without_the_suffix():
    f = H.url_features("http://thapthan.ac.th/information/confirmation/110dpbox/page/486743eb73799618ccfe5f14ca95b9a8")
    assert (f["shortest_word_host"], f["longest_word_host"], f["avg_word_host"]) == (8, 8, 8)
    assert f["length_words_raw"] == 6 and f["longest_words_raw"] == 32 and f["avg_words_raw"] == 12.5


def test_the_query_belongs_to_the_path_words():
    f = H.url_features("http://example.com/a/b?id=7&name=zed")
    assert f["length_words_raw"] == 1 + 2 + 4       # example | a b | id 7 name zed
    assert f["nb_qm"] == 1 and f["nb_and"] == 1 and f["nb_eq"] == 2


def test_a_raw_ip_host_is_flagged_and_its_address_is_counted_twice_in_the_words():
    f = H.url_features("http://148.204.63.249/users/sign_in")
    assert f["ip"] == 1 and f["abnormal_subdomain"] == 1 and f["tld_in_subdomain"] == 1
    assert f["length_words_raw"] == 11 and f["shortest_word_host"] == 2 and f["longest_word_host"] == 3


def test_flags_and_lists():
    f = H.url_features("http://login-secure.paypal.com.evil.top:8080/wp-admin/login.php?x=http://a.b")
    assert f["port"] == 1 and f["prefix_suffix"] == 1 and f["suspecious_tld"] == 1
    assert f["http_in_path"] == 1 and f["phish_hints"] >= 4          # wp, admin, login, login ...
    assert f["tld_in_path"] == 0 and f["punycode"] == 0
    assert H.url_features("https://bit.ly/3abc")["shortening_service"] == 1
    assert H.url_features("https://xn--80ak6aa92e.com/")["punycode"] == 1
    assert H.url_features("http://example.com/payload.exe")["path_extension"] == 1
    assert H.url_features("https://www.google.com/")["domain_in_brand"] == 1


def test_digits_and_spaces_and_repeats():
    f = H.url_features("http://a1b2.example.com/x%20y z")
    assert f["nb_space"] == 2
    assert f["ratio_digits_host"] == pytest.approx(2 / len("a1b2.example.com"))
    assert H.url_features("http://aaa.example.com/")["char_repeat"] >= 3


def test_redirection_features_come_from_the_recorded_chain():
    r = H.redirection_features("http://short.example.com/x", ["short.example.com", "other.org", "t.co"])
    assert r == {"nb_redirection": 3, "nb_external_redirection": 2}
    assert H.redirection_features("http://a.com/", []) == {"nb_redirection": 0, "nb_external_redirection": 0}


def test_odd_input_never_raises():
    for url in ("", "http://", "http://[::1]/", "http://a b/", "https://" + "a." * 80 + "com/" + "x" * 900,
                "http://%00.com/", "javascript:alert(1)"):
        out = H.url_features(url)
        assert list(out) == H.URL_FEATURES


@pytest.mark.skipif(not DATASET.exists(), reason="the public dataset is only on the host")
def test_each_feature_reproduces_the_public_datasets_values():
    rows = list(csv.DictReader(open(DATASET, encoding="utf-8")))
    assert len(rows) == 11430
    hits = {name: 0 for name in H.URL_FEATURES if name not in NOT_IN_URL_TEXT}
    for r in rows:
        mine = H.url_features(r["url"])
        for name in hits:
            if math.isclose(float(r[name]), float(mine[name]), abs_tol=1e-6):
                hits[name] += 1
    failures = {n: round(h / len(rows), 4) for n, h in hits.items() if h / len(rows) < MIN_SHARE.get(n, 0.999)}
    assert not failures, failures
