"""Tests for scripts/financial_category.py (the six financial categories, word rules v2 + hand corrections, 2026-11-22).

Each case is a message that the first version of the rules got wrong, checked by hand.

Run with:  python -m pytest tests/test_financial_category.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import financial_category as fc  # noqa: E402

CASES = [
    # parcel numbers are not gambling ("PH" + digits)
    ("Flash Express: your parcel PH2264810 is stuck at the Pasig hub because the barangay in your address is missing.", "none"),
    ("Lalamove: Driver Rodel has picked up your item. Live tracking link: https://share.lalamove.com/?PH1197334520", "none"),
    # a bank's cashback or rebate is banking, not gambling
    ("BPI Amore Cashback Card: Get up to 4% cashback on groceries, drugstores & utilities! Apply by Dec 15 at https://www.bpi.com.ph", "banking"),
    ("[EWB]You have PHP 5,216 available for cashback. Claim it before it expires today. https://ewbnotice.com/ph", "banking"),
    ("Security Bank: Convert purchases of P5,000+ to 0% installment for 6 months! Enroll via the app.", "banking"),
    # shop, mall and telco promos with "spin", "bonus" or "cashout" are not gambling
    ("Shopee: Spin and win! May chance kang manalo ng P5,000 na voucher, isang ikot lang dito: https://x.com/", "none"),
    ("Smart Signature Plan 1999: Get the iPhone 17 at P0 cashout! Reserve online: smart.com.ph", "none"),
    ("Globe GoSURF599 users: Enjoy 3GB bonus data FREE this weekend! Register at https://www.globe.com.ph", "none"),
    # gambling lures with weak words only
    ("970****419: You received PHP 658.00. Withdraw or playnow. https://bit.ly/4sldpEy", "gambling"),
    ("Mga minamahal, naghanda kami ng P588 na bonus para sa iyo! Magbukas ng account at tanggapin ito nang libre: tp208.com", "gambling"),
    ("Uy, nasubukan mo na ang spin sa Kulay? May chance kang manalo ng hanggang 188P! May rebate din na hanggang 2.7%. Sign up dito", "gambling"),
    ("JUAN C., Elevate your gaming at Winning+! Join for 4888 Game Events and 153% Play Loads", "gambling"),
    # "slot" in an interview or appointment is not gambling
    ("We are hiring technical support representatives. Apply online and get an interview slot within the week.", "none"),
    ("Maxicare: Annual physical exam slots open Jan 5, book early in the app.", "none"),
    # "investigation" is not investment; crypto wallets are crypto
    ("GCash Help Center: we received your report about a missing P3,000. Verify your account so we can start the investigation", "e-wallet"),
    ("Your Ledger Live login has expired. Please sign in again here: https://ledger-live-login-web.square.site/", "investment_crypto"),
    # loans and credit lines
    ("Need extra funds? Get up to P90K with Cashify. Apply now: https://cashify.onelink.me/x", "lending"),
    ("GCash: Pre-qualified ka for GLoan up to P25,000! Check your offer sa app", "lending"),
    # a generic "verify your account" with no money is not banking
    ("Hello, applicant ka ba sa work-from-home? I-verify mo na yung account mo dito para sa interview: http://x.com", "none"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_hand_checked_cases(text, expected):
    assert fc.categorize(text)[0] == expected


def test_dito_in_tagalog_is_not_the_telco():
    # "dito" (here) used to switch off the gambling words; DITO in capitals is still the telco
    assert fc.categorize("Mag-recharge ng P100 para makakuha ng P777 nang libre, sali dito: s17.de")[0] == "gambling"
    assert fc.categorize("Install the app for 28 cash! 110% bonuses and mystery prize. http://px177.com DITO")[0] == "none"


def test_hand_corrections_replace_the_rules_and_a_stale_file_stops(tmp_path):
    texts = ["Account Update: P9,527.80 added to your balance. Play now or withdraw anytime. Tap here: https://bit.ly/x",
             "Flash Express: your parcel PH2264810 is stuck at the Pasig hub."]
    path = tmp_path / "category_decisions.csv"
    pd.DataFrame({"sms_text": [texts[0]], "final_category": ["gambling"], "reason": ["test"]}).to_csv(path, index=False)
    out, n = fc.categorize_all(texts, path)
    assert n == 1 and out[0][0] == "gambling" and "gambling" in out[0][1].split("|") and out[1] == ("none", "none")
    pd.DataFrame({"sms_text": ["a text that is not in the dataset"], "final_category": ["gambling"], "reason": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not in the dataset"):
        fc.categorize_all(texts, path)
    pd.DataFrame({"sms_text": [texts[0]], "final_category": ["sports"], "reason": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown categories"):
        fc.categorize_all(texts, path)


def test_the_dataset_categories_equal_rules_plus_hand_corrections():
    root = Path(__file__).resolve().parents[1]
    data, decisions = root / "data" / "final" / "lejet_final_dataset.csv", root / "data" / "final" / "category_decisions.csv"
    if not data.exists():
        pytest.skip("the final dataset is not on this machine")
    d = pd.read_csv(data, keep_default_na=False, dtype=str)
    out, _ = fc.categorize_all(list(d.sms_text), decisions)
    assert list(d.category) == [c[0] for c in out] and list(d.categories_all) == [c[1] for c in out]
