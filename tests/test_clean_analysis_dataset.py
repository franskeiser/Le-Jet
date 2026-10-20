import pandas as pd
import pytest

pytest.importorskip("urlextract")

from src.data import clean_analysis_dataset as c  # noqa: E402


def prepared(text):
    """What clean() does to one message before cutting links: tidy, pull apart glued links, join split ones."""
    return c.join_split_domains(c.squash(c.fix_mojibake(text)))


def cut(text):
    joined = prepared(text)
    return c.remove_links(joined, c.find_links(joined))


def test_mojibake_is_repaired_and_clean_text_is_left_alone():
    assert c.fix_mojibake("The bÃ‰st deals") == "The bÉst deals"          # UTF-8 of "É" read as cp1252
    assert c.fix_mojibake("SpÉnd more") == "SpÉnd more"                       # a real accented letter
    assert c.fix_mojibake("plain text") == "plain text"


def test_escape_sequences_and_line_breaks_become_one_space():
    assert c.squash("line one\\r\\nline two\nline three\\n  end") == "line one line two line three end"


@pytest.mark.parametrize("text, expected", [
    ("Win big! Register now http://lucky-888.world", "Win big! Register now"),
    ("Get your bonus wp8luck.com/8888WIN", "Get your bonus"),
    ("Join us at 11y.life today", "Join us at today"),                                    # a bare domain with a link ending
    ("Visit https://NTC.GOV.PH or call 1682", "Visit or call 1682"),
])
def test_links_are_removed_and_the_words_stay(text, expected):
    assert cut(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("click to register!http://whgv05.com", "click to register!"),                      # glued to the word before it
    ("Jackpot Pool ay 7777 Libre!th108.com", "Jackpot Pool ay 7777 Libre!"),
    ("search more URL:kt777ad2.com", "search more URL:"),
    ("claim now in Philippines!lnnk.in/iFfo", "claim now in Philippines!"),
    ("Get your bonus at {9y15. com} now", "Get your bonus at now"),                     # split by the sender
    ("sign up at w8bonu. com today", "sign up at today"),
])
def test_links_glued_to_a_word_or_split_by_the_sender_are_still_found(text, expected):
    assert cut(text) == expected


@pytest.mark.parametrize("text", [
    "contact me at someone@gmail.com for details",           # an email's domain
    "Reply to (perssonfamfoundation@gmail.com) for more",
    "Your paymen.t for la.st month was pa.id",               # obfuscated words
    "that was timely.Open it now",                           # two words joined by a full stop
    "Thanks. Com on over",                                   # a sentence, not a link
])
def test_emails_and_ordinary_words_stay_as_text(text):
    assert cut(text) == text


def test_a_link_cut_off_in_the_data_leaves_no_scheme_behind():
    for text in ("CID: CO4125 https:", "confirm at the https:// (copy it)", "THANK YOU --https:/WIN: Send 0"):
        assert "http" not in cut(text).lower()


def raw_frame():
    return pd.DataFrame({
        "label": ["spam", "spam", "spam", "ham", "ham", "spam", "ham"],
        "message": ["Pay now at http://a.example.com/x", "http://b.example.com/only", "Pay now at http://c.example.com/y",
                    "Ok see you at 5", "<REAL NAME>", "<<Content not supported.>>", "OK  see you at 5"],
    })


def test_clean_maps_labels_drops_empty_and_duplicates_and_reports_every_count():
    out, summary = c.clean(raw_frame())
    assert summary["dropped_no_letters_or_digits_left"] == 3            # link only, tag only, MMS placeholder
    assert summary["dropped_exact_duplicate_original"] == 1             # "OK  see you at 5" repeats "Ok see you at 5"
    assert out["message_id"].tolist() == ["an_000001", "an_000002", "an_000003"] and summary["rows_out"] == 3
    assert out["label"].tolist() == ["smishing", "smishing", "legitimate"]
    assert out["label_raw"].tolist() == ["spam", "spam", "ham"]
    assert out["has_url"].tolist() == [1, 1, 0] and out["n_urls"].tolist() == [1, 1, 0]
    assert out["dup_after_link_removal"].tolist() == [1, 1, 0]          # the two "Pay now at" rows differ only by their link
    assert out["sms_text"].iloc[0] == "Pay now at"
    assert out["sms_text_original"].iloc[0].endswith("a.example.com/x")


def test_clean_refuses_an_unknown_label_and_a_message_with_two_labels():
    with pytest.raises(ValueError, match="other than spam / ham"):
        c.clean(pd.DataFrame({"label": ["spam", "maybe"], "message": ["a b c", "d e f"]}))
    with pytest.raises(ValueError, match="both labels"):
        c.clean(pd.DataFrame({"label": ["spam", "ham"], "message": ["Same words here", "same words here"]}))
