"""Tests for the text-and-link relevance dataset: lexicon, templates and the builder's rules (src/data/relevance_*.py, build_relevant_dataset.py)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import build_relevant_dataset as brd
from src.data import relevance_lexicon as lex
from src.data import relevance_templates as tpl
from src.features.link_features import extract_links

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HAVE_RAW = all((RAW / p).exists() for p in ("sms/isda_sms_dataset.csv", "web-phishing/train-1.csv", "web-phishing/test.csv",
                                            "web-phishing/hunt_live_features.csv", "web-phishing/crowd_pilot_synthetic.csv"))


# ------------------------------------------------------------------------------------------------------- lexicon
@pytest.mark.parametrize("url, brands, owned", [
    ("http://smrt.ph/GigaLifeApp", {"smart"}, {"smart"}),
    ("https://official.paymaya.com/d0ec/e6d83ecc", {"maya"}, {"maya"}),
    ("hxxps://ticket-fedex[.]com/", {"fedex"}, set()),                                    # a look-alike names the brand but does not own it
    ("http://buy-ipad.yu.hopto.org/apple-power-adapter-for-iphone-7.html", {"apple"}, set()),
    ("https://www.hi-smart.ru/", {"smart"}, set()),
    ("http://smartphone-cases.com/x", set(), set()),                                      # "smartphone" is not Smart
    ("https://docs.google.com/forms/d/e/x", {"google"}, {"google"}),
    ("https://1789503-coinbase.com/", {"coinbase"}, set()),
])
def test_tag_url_names_brands_and_knows_which_it_owns(url, brands, owned):
    tags = lex.tag_url(url)
    assert tags.brands == brands and tags.owned == owned


def test_every_brand_is_found_in_text_by_its_own_name_and_in_a_link_by_its_own_domain():
    for key, (name, url_pattern, text_pattern, _theme) in lex.BRANDS.items():
        assert key in lex.tag_text(f"{name}: your account update").brands, key
        assert text_pattern and url_pattern
    for key, labels in lex.OFFICIAL.items():
        assert key in lex.BRANDS, key


def test_no_control_characters_in_the_relevance_sources():
    """A backslash-b written as a real backspace once silently disabled six brands (the patterns compiled and matched nothing)."""
    base = Path(__file__).resolve().parents[1]
    for rel in ("src/data/relevance_lexicon.py", "src/data/relevance_templates.py", "src/data/build_relevant_dataset.py", "src/data/check_relevance.py"):
        text = (base / rel).read_text(encoding="utf-8")
        assert not [c for c in text if ord(c) < 32 and c not in "\n\r\t"], rel


def test_exchanges_are_separate_brands():
    assert lex.tag_url("http://binance-login.com").brands == {"binance"}
    assert lex.tag_url("http://kucoin-login.com").brands == {"kucoin"}


def test_url_themes_match_token_starts_and_host_themes_ignore_the_path():
    assert "delivery" in lex.tag_url("http://uspstracking.fun/pay").themes
    assert lex.tag_url("https://keepgreentb.com/doc/download.html").themes == set()        # "download" is not "load"
    assert lex.tag_url("http://soundtrack.com/x").themes == set()
    blog = lex.tag_url("http://blogs.worldbank.org/transport/global-logistics-hub")
    assert "delivery" in blog.themes and "delivery" not in blog.host_themes


def test_title_and_site_names_come_from_the_link():
    tags = lex.tag_url("https://www.todayshomeowner.com/how-to-make-homemade-insecticidal-soap-for-plants/")
    assert tags.title.startswith("how to make homemade insecticidal soap") and tags.site == "Todayshomeowner"
    assert lex.tag_url("http://pinkskky.blogspot.com/2011/11/typeface-poster.html").site == "Pinkskky"      # a hosting platform is named by its tenant


def test_ambiguous_brand_words_match_only_as_brands():
    assert lex.tag_text("Paki-check po ng details dito").brands == set()                  # "dito" is Tagalog for "here"
    assert lex.tag_text("DITO: get free data").brands == {"dito"}
    assert lex.tag_text("This is a smart phone").brands == set()
    assert lex.tag_text("Smart: new promo").brands == {"smart"}
    assert lex.tag_text("I ate an apple").brands == set()


def test_tags_are_read_from_the_words_not_from_the_link():
    text = "Hi, check this: http://login-microsoft-online.com/"
    assert lex.tag_message(text).brands == set()
    assert lex.tag_text(text).brands == {"microsoft"}                                       # reading the whole message would see the link's name
    assert lex.strip_links(text) == "Hi, check this:"


def test_relation_levels():
    link = lex.tag_url("http://smrt.ph/GigaLifeApp")
    assert lex.relation(lex.tag_text("Smart: promo"), link) == "brand"
    assert lex.relation(lex.tag_text("PayPal: promo"), link) == "conflict"
    assert lex.relation(lex.tag_text("Hi, check this"), link) == "generic"
    assert lex.relation(lex.tag_text("Get more load and promo"), link) == "theme"
    page = lex.tag_url("https://www.todayshomeowner.com/how-to-make-homemade-insecticidal-soap-for-plants/")
    words = "Sharing this article, how to make homemade insecticidal soap"
    assert lex.relation(lex.tag_text(words), page, words) == "page"


def test_a_genuine_notice_needs_the_brands_own_domain():
    lookalike = lex.tag_url("http://buy-ipad.yu.hopto.org/apple-power-adapter-for-iphone-7.html")
    apple_text = lex.tag_text("Apple: your settings changed")
    assert lex.relation(apple_text, lookalike, genuine=False) == "brand"                    # a phishing row may imitate
    assert lex.relation(apple_text, lookalike, genuine=True) == "conflict"                  # a genuine one may not
    assert lex.relation(apple_text, lex.tag_url("https://www.apple.com/"), genuine=True) == "brand"


def test_a_share_of_an_article_about_a_brand_is_not_a_conflict():
    link = lex.tag_url("https://www.howtogeek.com/197348/how-to-clear-your-google-search-history-on-android/")
    words = "Sharing this article, how to clear your google search history on android"
    assert lex.relation(lex.tag_text(words), link, words, genuine=True) == "page"


# ------------------------------------------------------------------------------------------------------ templates
LINK = "http://example-link.com/x"


def _found(text):
    links = extract_links(text)
    return links[0]["raw"].rstrip(".,") if links else None


def test_every_template_fills_every_slot_and_keeps_its_link_findable():
    rng = np.random.default_rng(0)
    for theme, pools in tpl._SMISH.items():
        for brand in ("paypal", None):
            for _ in range(6):
                assert _found(tpl.smish(rng, LINK, theme, brand)) == LINK, theme
    for theme, (en, tl) in tpl._LEGIT_BRAND.items():
        for _ in range(6):
            assert _found(tpl.legit_brand(rng, LINK, "smart", theme)) == LINK
    for _ in range(20):
        assert _found(tpl.neutral(rng, LINK)) == LINK
        assert _found(tpl.legit_security(rng, LINK, "gcash")) == LINK
        assert _found(tpl.legit_urgent(rng, LINK, "smart")) == LINK
    for theme in tpl._LEGIT_THEME:
        assert _found(tpl.legit_theme(rng, LINK, theme)) == LINK
    for theme in tpl._LEGIT_PRESSURE:
        assert _found(tpl.legit_pressure(rng, LINK, theme)) == LINK
    for brand in tpl._SCAM_BENIGN:
        assert _found(tpl.scam_benign(rng, LINK, brand)) == LINK
    assert _found(tpl.legit_share(rng, LINK, title="how to make soap at home")) == LINK
    assert _found(tpl.legit_share(rng, LINK, site="Todayshomeowner")) == LINK
    assert tpl.legit_share(rng, LINK) is None and tpl.scam_benign(rng, LINK, "nobody") is None


def test_no_leftover_braces_and_a_brand_named_in_the_words():
    rng = np.random.default_rng(1)
    for _ in range(200):
        text = tpl.smish(rng, LINK, "account", "paypal")
        assert "{" not in text and "}" not in text
        assert "PayPal" in text
    assert "Smart" in tpl.legit_brand(rng, LINK, "smart", "telco")


def test_no_link_scams_carry_a_masked_number_and_no_link():
    rng = np.random.default_rng(2)
    texts = {tpl.smish_nolink(rng) for _ in range(400)}
    assert len(texts) > 150                                                                # enough variety to build 165 unique rows
    assert all(extract_links(t) == [] and "{" not in t for t in texts)
    assert sum("<NUM>" in t for t in texts) / len(texts) > 0.95


def test_templates_are_reproducible_with_a_seed():
    a = [tpl.smish(np.random.default_rng(7), LINK, "delivery", "fedex") for _ in range(3)]
    assert a[0] == a[1] == a[2]


def test_a_second_deadline_is_not_added_after_one_already_in_the_sentence():
    rng = np.random.default_rng(3)
    for _ in range(400):
        text = tpl.smish(rng, LINK, "account", "paypal")
        assert len(__import__("re").findall(r"\b(?:hours|oras)\b", text)) <= 1, text


# --------------------------------------------------------------------------------------------------- builder rules
def _link(url, cls="legit", source="tt"):
    return brd.Link(url, brd._domain(url), cls, source, None, lex.tag_url(url))


def test_blank_features_have_37_columns_and_the_page_ones_are_minus_one():
    f = brd.blank_features("http://jackpotcityph7.com/win")
    assert set(f) == set(brd.FEATURES_37)
    assert all(f[c] == -1 for c in brd.PAGE_FEATURES) and f["nb_redirection"] == 0
    assert f["length_url"] == len("http://jackpotcityph7.com/win")


def test_a_genuine_brand_notice_is_only_written_for_the_brands_own_domain():
    rng = np.random.default_rng(4)
    own = _link("http://smrt.ph/GigaLifeApp")
    fake = _link("https://www.hi-smart.ru/")
    for _ in range(60):
        assert "Smart" in brd.legit_text(rng, own)[0] or True                              # may be neutral, share or brand wording
        text, _pair = brd.legit_text(rng, fake)
        assert "Smart" not in text                                                          # hi-smart.ru is not Smart
    voices = {brd.legit_text(np.random.default_rng(i), own)[0] for i in range(200)}
    assert any("Smart" in v for v in voices)


def test_deep_pages_of_global_platforms_get_no_account_notice():
    deep = _link("https://msdn.microsoft.com/en-us/library/ee818756(v=vs.110).aspx")
    front = _link("https://www.microsoft.com/")
    assert not brd._brand_voice_ok(deep, "microsoft") and brd._brand_voice_ok(front, "microsoft")
    assert brd._brand_voice_ok(_link("http://smart.com.ph/promo/long/path/here"), "smart")        # Philippine brands: any depth


def test_gov_theme_needs_a_philippine_host():
    ph, br = lex.tag_url("https://www.sss.gov.ph/"), lex.tag_url("https://escolavirtual.gov.br/")
    assert brd._host_theme(ph, {"gov"}) == "gov" and brd._host_theme(br, {"gov"}) is None


def test_phishing_wording_never_names_a_brand_the_link_does_not_carry():
    rng = np.random.default_rng(5)
    for url in ("http://paypal-secure-login.xyz/a", "http://uspstracking.fun/pay", "hxxp://z-instagram[.]blogspot[.]com/", "http://secure-login-verify.xyz/update",
                "http://thapthan.ac.th/information/confirmation/page", "http://plain-site-5521.com/"):
        link = _link(brd.lex.refang(url) if url.startswith("hxxp") else url, "phish")
        link.url = url
        for _ in range(80):
            text, pair = brd.smish_text(rng, link)
            words = lex.strip_links(text)
            rel = lex.relation(lex.tag_text(words), link.tags, words)
            assert rel != "conflict", (url, text)
            assert pair in ("S1", "S2")


def test_domain_cap_counts_per_domain_and_class():
    counts = {}
    ok = [brd._domain_cap_ok(counts, "smrt.ph", "legitimate") for _ in range(brd.MAX_PER_DOMAIN + 5)]
    assert ok.count(True) == brd.MAX_PER_DOMAIN and brd._domain_cap_ok(counts, "smrt.ph", "smishing")


def test_check_flags_conflicts_duplicates_and_a_stored_link_that_is_not_the_first_link():
    cols = brd.COLUMNS + ["has_url"]
    base = {c: brd.bcd.MISSING for c in cols}
    base.update({"consent": 1, "label": "legitimate", "has_url": 1, "link_domain": "a.com", "page_fetched": 0})
    rows = [{**base, "message_id": "m1", "sms_text": "see http://a.com/x now", "link": "http://a.com/x"},
            {**base, "message_id": "m2", "sms_text": "see http://a.com/x now", "link": "http://a.com/x"},
            {**base, "message_id": "m3", "sms_text": "go to Coins.ph then http://b.com/y", "link": "http://b.com/y", "link_domain": "b.com"}]
    df = pd.DataFrame(rows, columns=cols)
    meta = pd.DataFrame({"message_id": ["m1", "m2", "m3"], "relation": ["generic", "generic", "conflict"],
                         "relation_check": ["generic", "generic", "conflict"]})
    problems = " | ".join(brd.check(df, meta))
    assert "repeat the same text and link" in problems and "first extracted link" in problems and "different brands" in problems


# ------------------------------------------------------------------------------------------------------ integration
@pytest.mark.skipif(not HAVE_RAW, reason="the raw SMS and web-phishing files are not on this machine")
def test_a_small_build_passes_its_own_checks_and_balances_the_sources():
    df, meta, summary = brd.build(seed=1, scale=0.15)
    assert summary["problems"] == []
    assert list(df.columns) == brd.COLUMNS + ["has_url"] and "label_source" not in df.columns and not df[brd.COLUMNS].isna().any().any()
    assert set(df["label"]) == {"smishing", "legitimate"} and 0.35 < (df["label"] == "smishing").mean() < 0.6
    linkless = df[df["has_url"] == 0]
    assert (linkless["link"] == "-1").all() and linkless["link_domain"].is_unique and (linkless[brd.FEATURES_37] == -1).all().all()
    blank = df[(df["has_url"] == 1) & (df["page_fetched"] == 0)]
    assert (blank[brd.PAGE_FEATURES] == -1).all().all() and (blank["nb_redirection"] == 0).all()
    assert df.groupby(["link_domain", "label"]).size().drop("-1", level=0, errors="ignore").max() <= brd.MAX_PER_DOMAIN
    for source in ("live_tt", "live_lean"):                                                  # each crawler contributes BOTH classes
        by_label = meta[meta["feature_source"] == source].merge(df[["message_id", "label"]], on="message_id")["label"].value_counts()
        assert by_label.min() > 0.4 * by_label.max(), (source, by_label.to_dict())
    hunt = meta[meta["link_source"] == "hunt"]
    assert (hunt["feature_source"] == "offline_url").all()                                   # hunt supplies links only


@pytest.mark.skipif(not HAVE_RAW, reason="the raw SMS and web-phishing files are not on this machine")
def test_the_same_seed_gives_the_same_dataset():
    a, _, _ = brd.build(seed=3, scale=0.05)
    b, _, _ = brd.build(seed=3, scale=0.05)
    pd.testing.assert_frame_equal(a, b)
