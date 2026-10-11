"""
Tests for src/features/text_preprocessing.py: the CNN's <URL> masking uses the same link
finder as has_url, and punctuation splits words except inside words, numbers and tags.

Run with:  pytest tests/test_text_preprocessing.py
"""

import pytest

from src.features.link_features import extract_links
from src.features.text_preprocessing import PAD_TOKEN, preprocess, text_tokens


@pytest.mark.parametrize("text, expected", [
    ("Verify at gcash-verify.xyz/login. Thanks!", ["verify", "at", "<URL>", "thanks", "!"]),
    ("see you tomorrow.call me", ["see", "you", "tomorrow", "call", "me"]),
    ("Username:Rom88 Balance:3,862,587.65", ["username", "rom88", "balance", "386258765"]),
    ("Pa-share sa Pag-IBIG, don't forget 12:00am", ["pashare", "sa", "pagibig", "dont", "forget", "1200am"]),
    ("OTP sent <DATE_TIME>,thanks", ["otp", "sent", "<datetime>", "thanks"]),
    ("Hi <REAL NAME>, ref <REFERENCE NUMBER>", ["hi", "<realname>", "ref", "<referencenumber>"]),
])
def test_text_tokens(text, expected):
    assert text_tokens(text) == expected


@pytest.mark.parametrize("text", [
    "Visit https://www.paymaya.com/deals now",
    "Updates at up.edu.ph",
    "email help@gcash.com today",
    "see you tomorrow.call me",
    "two links: bit.ly/abc and www.usdtjerc.com",
])
def test_url_token_matches_link_finder(text):
    # Every link has_url counts becomes exactly one <URL> token, and nothing else does.
    assert text_tokens(text).count("<URL>") == len(extract_links(text))


def test_preprocess_pads_and_truncates():
    assert preprocess("hello there", 4) == ["hello", "there", PAD_TOKEN, PAD_TOKEN]
    assert preprocess("a b c d e f", 3) == ["a", "b", "c"]
