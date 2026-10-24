"""
Tests for webapp/screens.py: splitting OCR lines into message bubbles. The pure-logic tests need no OCR; the last ones
draw a fake phone screenshot and run the real OCR, and are skipped where RapidOCR / Pillow are not installed.

Run with:  python -m pytest tests/test_webapp_screens.py
"""

from __future__ import annotations

import pytest

from webapp import dataset as ds
from webapp import screens
from webapp.screens import Line

W, H = 1080.0, 2200.0


def L(text, x0, y0, x1=None, h=42, conf=0.98):
    return Line(text, x0, y0, x1 if x1 is not None else x0 + 8 * len(text), y0 + h, conf)


# the OCR output of a drawn thread (taken from a real run of the OCR on tests' fake screenshot)
THREAD = [
    L("3:12", 57, 40, 144), L("LTE 87%", 898, 41, 1047), L("GCash", 140, 133, 305),
    L("Today 3:12 PM", 426, 298, 656),
    L("GCash: Your account will be", 87, 421, 611), L("suspended today. Verify now at", 85, 480, 670),
    L("https://gcash-verify-ph.xyz/login to", 84, 537, 719), L("avoid losing your balance.", 85, 595, 574),
    L("Reminder: act within 24 hours.", 84, 736, 651),
    L("3:40 PM", 475, 852, 608),
    L("BDO Alert: Unusual login detected.", 85, 971, 732), L("Confirm your identity: bit.ly/3xYzAb", 86, 1031, 735),
    L("who is this?", 774, 1175, 995),
    L("Hi! Your parcel is on hold. Pay the 49", 85, 1318, 772), L("peso fee at www.lbc-track.top/pay", 82, 1373, 714),
    L("Text message", 120, 2015, 368),
]


def test_thread_is_split_into_bubbles_and_chrome_is_dropped():
    b = screens.split_messages(THREAD, W, H)
    assert [x.direction for x in b] == ["in", "in", "in", "out", "in"]
    assert b[0].text == "GCash: Your account will be suspended today. Verify now at https://gcash-verify-ph.xyz/login to avoid losing your balance."
    assert b[1].text == "Reminder: act within 24 hours."
    assert b[2].text == "BDO Alert: Unusual login detected. Confirm your identity: bit.ly/3xYzAb"
    assert b[3].text == "who is this?"
    assert b[4].text.endswith("peso fee at www.lbc-track.top/pay")
    joined = " ".join(x.text for x in b)
    for junk in ("3:12", "LTE", "Today", "3:40 PM", "Text message"):
        assert junk not in joined
    assert [bool(ds.extract_links(x.text)) for x in b] == [True, False, True, False, True]


def test_single_message_in_dark_mode():
    single = [L("3:12", 56, 41, 146), L("GCash", 138, 132, 306), L("Tue, Oct 14", 447, 297, 633),
              L("Maya: Your loan of PHP 5,000 is", 85, 419, 690), L("approved! Claim at", 85, 477, 443),
              L("https://maya-loan.cc/claim", 83, 538, 572), L("Text message", 118, 2014, 370)]
    b = screens.split_messages(single, W, H)
    texts = [x.text for x in b]
    assert "Maya: Your loan of PHP 5,000 is approved! Claim at https://maya-loan.cc/claim" in texts
    assert ds.first_link(texts[-1])[0].startswith("https://maya-loan.cc/claim")


def test_tight_crop_of_one_bubble_is_received_not_sent():
    # a screenshot cropped to just the bubble: its text spans nearly the whole width
    crop = [L("Your parcel is held. Pay at https://lbc-track.top/p", 30, 30, 1040), L("and avoid return.", 30, 85, 400)]
    b = screens.split_messages(crop, W, 300.0)
    assert len(b) == 1 and b[0].direction == "in"


def test_link_cut_at_the_bubble_edge_is_rejoined():
    cut = [L("Verify at https://secure-account-verification-gcash", 85, 400, 760),
           L("-ph.top/login now", 85, 458, 400)]
    b = screens.split_messages(cut, W, H)
    assert len(b) == 1 and "https://secure-account-verification-gcash-ph.top/login now" in b[0].text


def test_ocr_spaced_scheme_is_repaired():
    b = screens.split_messages([L("Open https: // gcash-ph.top/x", 85, 400, 700)], W, H)
    assert b[0].text == "Open https://gcash-ph.top/x"


def test_low_confidence_and_empty_input():
    assert screens.split_messages([], W, H) == []
    assert screens.split_messages([L("blurry text", 85, 400, conf=0.2)], W, H) == []


def test_bubbles_are_split_on_gaps_and_on_a_change_of_side():
    lines = [L("first bubble", 85, 400, 400), L("still first", 85, 458, 400),
             L("second bubble", 85, 700, 400),
             L("reply from me", 700, 800, 1000)]
    b = screens.split_messages(lines, W, H)
    assert [x.text for x in b] == ["first bubble still first", "second bubble", "reply from me"]
    assert [x.direction for x in b] == ["in", "in", "out"]


def test_load_image_refuses_junk():
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    with pytest.raises(screens.ImageRejected):
        screens.load_image(b"this is not a picture")
    with pytest.raises(screens.ImageRejected):
        screens.load_image(b"GIF89a" + b"\x00" * 40)                  # a GIF header: wrong type or undecodable


# ---------------------------------------------------------------------------------------------- real OCR
def _draw_thread(path):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=40)
    img = Image.new("RGB", (1080, 1800), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((60, 30), "3:12", font=font, fill=(0, 0, 0))
    y = 250
    for text, right in [("GCash: Your account is locked. Verify at", False), ("https://gcash-verify-ph.xyz/login", False),
                        (None, False),
                        ("Pay the fee at www.lbc-track.top/pay now", False), (None, False),
                        ("who is this?", True)]:
        if text is None:
            y += 90
            continue
        w = d.textlength(text, font=font)
        x = 1080 - 60 - w if right else 60
        d.text((x, y), text, font=font, fill=(0, 0, 0))
        y += 62
    img.save(path)


def test_real_ocr_reads_a_drawn_screenshot(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("rapidocr_onnxruntime")
    path = tmp_path / "thread.png"
    _draw_thread(path)
    bubbles = screens.read_screenshot(screens.Reader(), path.read_bytes())
    links = [ds.first_link(b.text) for b in bubbles if b.direction == "in"]
    found = [l[0] for l in links if l]
    assert any(l.startswith("https://gcash-verify-ph.xyz/login") for l in found), [b.text for b in bubbles]
    assert any("lbc-track.top/pay" in l for l in found), [b.text for b in bubbles]
    assert any(b.direction == "out" for b in bubbles)


# ---------------------------------------------------------------------------------------------- a real iPhone screenshot
# The raw OCR lines of a real 942 x 2048 iPhone "Messages" screenshot of an unknown-sender spam text (positions and text
# exactly as the OCR returned them; note the dropped spaces, the replacement character for the bullet, and 'https:/l').
IPHONE = [
    Line("11:06", 99, 55, 246, 97, 0.98), Line("52", 675, 56, 852, 96, 0.77), Line("BIS 888>", 366, 185, 582, 230, 0.90),
    Line("TextMessage�SMS", 302, 323, 641, 360, 0.97), Line("Sun,Oct11at12:11AM", 287, 373, 657, 407, 0.94),
    Line("6229 ikaw ang napili today!", 68, 466, 656, 520, 0.98), Line("Wild Bounty gives Golden", 68, 522, 622, 576, 0.99),
    Line("Egg Jackp0t Rush + 150%", 71, 580, 636, 631, 0.99), Line("extra bonus for you: https:/l", 68, 635, 661, 689, 0.96),
    Line("3go.run/324h", 69, 692, 366, 743, 0.98),
    Line("If you did notexpect this message", 187, 873, 754, 917, 0.94), Line("from anunknown sender,itmaybe", 182, 919, 758, 956, 0.96),
    Line("spam.", 419, 964, 525, 1000, 1.0),
    Line("Delete", 173, 1056, 324, 1110, 1.0), Line("MarkasKnown", 524, 1061, 864, 1105, 0.98),
    Line("Text Message�SMS", 221, 1906, 666, 1956, 0.97), Line("+", 97, 1912, 130, 1943, 0.89),
]


def test_real_iphone_screenshot_gives_one_message_with_a_repaired_link():
    b = screens.split_messages(IPHONE, 942.0, 2048.0)
    assert [x.text for x in b] == [
        "6229 ikaw ang napili today! Wild Bounty gives Golden Egg Jackp0t Rush + 150% extra bonus for you: https://3go.run/324h"]
    assert b[0].direction == "in"
    assert ds.first_link(b[0].text)[0] == "https://3go.run/324h"


def test_wrapped_scheme_is_joined_without_a_space():
    for tail in ("https://", "https:/", "https:/l", "https: //"):
        lines = [L("Claim your prize: " + tail, 85, 400, 700), L("claim-now.top/x", 85, 458, 400)]
        assert screens.split_messages(lines, W, H)[0].text == "Claim your prize: https://claim-now.top/x", tail
    # a host that really starts with l/1 must not lose its first letter
    keep = [L("Open https://", 85, 400, 500), L("lbc-track.top/p", 85, 458, 400)]
    assert screens.split_messages(keep, W, H)[0].text == "Open https://lbc-track.top/p"
    assert screens.split_messages([L("Open https://1password.com/x", 85, 400, 700)], W, H)[0].text == "Open https://1password.com/x"


def test_centered_notice_is_one_block_and_dropped_but_a_message_that_mentions_spam_is_kept():
    notice = [L("If you did not expect this message", 187, 873, 754), L("from an unknown sender, it may be", 182, 919, 758),
              L("spam.", 419, 964, 525)]
    assert screens.split_messages(notice, 942.0, 2048.0) == []
    real = [L("Reply STOP to unsubscribe from spam. Claim: https://x-prize.top/a", 70, 600, 700)]
    assert len(screens.split_messages(real, 942.0, 2048.0)) == 1


# ---------------------------------------------------------------------------------------------- more real screens
# OCR lines of a real iPhone dark-mode thread (945 x 2048): three messages, and the TOP of the screen shows a fourth one
# scrolling under the translucent header. Wrapped links read as 'bit.lyl' + 'ubrewardscard', and a link followed by
# 'T&Cs apply' on the next line.
MONEYMAX = [
    Line("Applynow!GetNoAnnualFee", 76, 7, 627, 34, .97), Line("11:12", 99, 32, 249, 74, .99), Line("n a Metrobank", 226, 37, 509, 92, .95),
    Line("Instagram", 36, 75, 189, 105, .98), Line("tanium Card via Moneymax.", 114, 95, 610, 135, .91), Line("offer!bit", 216, 141, 371, 181, .98),
    Line("NoMsgs?TxtOFF", 70, 237, 397, 278, .98), Line("22565", 70, 285, 196, 324, 1), Line("Moneymax", 353, 293, 576, 339, 1),
    Line("Mon,Sep21at11:36AM", 317, 414, 627, 442, .95),
    Line("Apply now!Get No Annual Fee", 73, 487, 629, 533, .95), Line("For Lifeona UnionBank", 74, 541, 506, 575, .96),
    Line("Rewards Credit Card via", 72, 585, 513, 625, .97), Line("Moneymax. Limited offer! bit.lyl", 70, 631, 650, 678, .98),
    Line("ubrewardscard", 74, 684, 347, 722, 1), Line("NoMsgs? Txt OFF Moneymax to", 72, 731, 656, 771, .95), Line("22565", 72, 778, 197, 818, 1),
    Line("Fri,Sep25at5:03PM", 332, 905, 615, 933, .96), Line("Apply now! Get No Annual Fee", 73, 980, 629, 1025, .96),
    Line("For Lifeon a UnionBank", 74, 1032, 506, 1066, .95), Line("RewardsCreditCardvia", 75, 1081, 512, 1115, .99),
    Line("Moneymax. Limited offer! bit.lyl", 71, 1123, 650, 1170, 1), Line("ubrewardscard", 75, 1177, 347, 1213, .99),
    Line("NoMsgs? Txt OFF Moneymax to", 71, 1221, 655, 1265, .99), Line("22565", 70, 1269, 198, 1311, 1),
    Line("Today11:12AM", 375, 1396, 571, 1427, .98),
    Line("Apply now!Get No Annual Fee", 73, 1473, 630, 1516, .94), Line("For LifeonaMetrobank", 74, 1525, 506, 1559, .98),
    Line("Titanium Card via Moneymax.", 73, 1570, 613, 1611, .95), Line("Limitedoffer!bit.ly/mbtitanium", 70, 1619, 631, 1658, .99),
    Line("T&Cs apply", 70, 1665, 284, 1710, .99), Line("NoMsgs?Txt OFF Moneymaxto", 71, 1714, 655, 1756, .96),
    Line("22565", 69, 1758, 198, 1803, 1), Line("+", 97, 1913, 134, 1950, .99), Line("Text Message�SMS", 231, 1913, 608, 1956, .96),
]


def test_moneymax_wrapped_links_ghost_text_and_the_false_join():
    b = screens.split_messages(MONEYMAX, 945.0, 2048.0)
    with_link = [x.text for x in b if ds.extract_links(x.text)]
    assert len(with_link) == 3
    assert [ds.first_link(t)[0] for t in with_link] == ["bit.ly/ubrewardscard", "bit.ly/ubrewardscard", "bit.ly/mbtitanium"]
    assert with_link[2].endswith("bit.ly/mbtitanium T&Cs apply NoMsgs?Txt OFF Moneymaxto 22565")   # 'T&Cs' NOT glued to the link
    ghosts = [x for x in b if not ds.extract_links(x.text)]
    # the half-hidden message under the header: link-less fragments (one when the pixels are available, two without them)
    assert 1 <= len(ghosts) <= 2 and ghosts[0].text.startswith("tanium Card via Moneymax.")
    assert not any("Today" in x.text or "Sep" in x.text or "Text Message" in x.text for x in b)


def test_google_messages_and_xiaomi_chrome_is_dropped():
    w, h = 928.0, 2010.0
    lines = [
        L("VKTechPro", 287, 130, 512, h=40), L("Unread", 410, 1075, 512, h=34), L("Tuesday, Sep 15�2:33 PM", 280, 1138, 643, h=34),
        L("Hi 9389246913! Your P3213.00", 62, 1250, 614, h=40), L("balance is ready. Play now and", 62, 1312, 620, h=40),
        L("https://bit.ly/4xjmJJX?6913", 62, 1374, 549, h=40), L("OPT-OUT NO 9982762637", 62, 1500, 543, h=40),
        L("Sender can't accept replies. Contact", 78, 1800, 750, h=40), L("them directly. Learn more", 78, 1860, 551, h=40),
        L("Tap to load preview", 222, 1580, 518, h=36), L("2/38:27PM", 59, 420, 170, h=26), L("directly.Learn more", 70, 1810, 398, h=36),
    ]
    texts = [x.text for x in screens.split_messages(lines, w, h)]
    assert all("Unread" not in t and "Sender" not in t and "preview" not in t and "Learn more" not in t and "8:27" not in t for t in texts)
    assert any("9389246913" in t and "bit.ly/4xjmJJX?6913" in t for t in texts)
    # an ordinary message that says 'learn more' is not chrome
    ok = screens.split_messages([L("Click to learn more: https://x-prize.top/a", 70, 600, 700)], w, h)
    assert len(ok) == 1 and "learn more" in ok[0].text


def test_app_bar_name_below_a_large_avatar_is_dropped():
    # iPhone shows the contact name under a big avatar, at 13% of the height
    assert screens.split_messages([L("LTGPROMO>", 379, 255, 543)], 946.0, 2048.0) == []


# ---------------------------------------------------------------------------------------------- bubble colour and spaces
def _canvas(w=900, h=1000, bg=(0, 0, 0)):
    np = pytest.importorskip("numpy")
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bg
    return img


def test_blank_line_inside_one_bubble_is_merged_but_two_bubbles_stay_apart():
    one = _canvas()
    one[100:520, 20:720] = (40, 40, 45)                                   # a single tall bubble
    two = _canvas()
    two[100:250, 20:720] = (40, 40, 45)                                   # two bubbles with a strip of background between
    two[290:520, 20:720] = (40, 40, 45)
    lines = [L("Your balance is ready, play now:", 60, 150, 600), L("https://bit.ly/4xjmJJX", 60, 190, 500),
             L("OPT-OUT NO 9982762637", 60, 360, 560)]                    # a blank line (about 3 lines) before the last one
    merged = screens.split_messages(lines, 900.0, 1000.0, one)
    assert len(merged) == 1 and "OPT-OUT NO 9982762637" in merged[0].text and "bit.ly/4xjmJJX" in merged[0].text
    assert len(screens.split_messages(lines, 900.0, 1000.0, two)) == 2
    assert len(screens.split_messages(lines, 900.0, 1000.0, None)) == 2   # without pixels: the gap decides, as before


def _draw_words(words, underline=False, bg=(255, 255, 255), gap=0.38):
    """Draw `words` separated by a phone-like gap, return (image, chars, boxes) as the OCR reports them when it DROPS the spaces."""
    pil = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw, ImageFont
    np = pytest.importorskip("numpy")
    size = 44
    font = ImageFont.load_default(size=size)
    img = pil.new("RGB", (900, 140), bg)
    d = ImageDraw.Draw(img)
    x, y0, box_h = 30.0, 40, int(size * 1.1)
    chars, boxes = [], []
    for w, word in enumerate(words):
        if w:
            x += gap * size                                    # the gap of a space (0.38 of the font size)
        for c in word:
            d.text((x, y0), c, font=font, fill=(0, 0, 0))
            chars.append(c)
            boxes.append([[x, y0], [x + font.getlength(c), y0], [x + font.getlength(c), y0 + box_h], [x, y0 + box_h]])
            x += font.getlength(c)
    if underline:
        yy = y0 + int(box_h * 0.9)
        d.line((30, yy, x, yy), fill=(0, 0, 0), width=2)
    return np.asarray(img), chars, boxes


def test_dropped_spaces_are_restored_from_the_picture():
    img, chars, boxes = _draw_words(["Hello", "world", "Titanium", "Card"])
    assert screens.respace(img, "HelloworldTitaniumCard", boxes, chars) == "Hello world Titanium Card"


def test_no_spaces_are_invented_inside_words_or_links():
    img, chars, boxes = _draw_words(["https://bit.ly/ubrewardscard"])
    assert screens.respace(img, "".join(chars), boxes, chars) == "https://bit.ly/ubrewardscard"
    img, chars, boxes = _draw_words(["Rewards", "Credit"])
    assert screens.respace(img, "RewardsCredit", boxes, chars) == "Rewards Credit"
    # an underline running under the whole line (a link) must not hide the gap or create one
    img, chars, boxes = _draw_words(["Verify", "now"], underline=True)
    assert screens.respace(img, "Verifynow", boxes, chars) == "Verify now"


def test_respace_keeps_spaces_the_ocr_found_and_gives_up_on_bad_data():
    img, chars, boxes = _draw_words(["Hello", "world"])
    full = list("Hello world")                                           # the OCR found the space itself
    assert screens.respace(img, "Hello world", boxes[:5] + [boxes[4]] + boxes[5:], full) == "Hello world"
    assert screens.respace(img, "Helloworld", boxes[:-1], chars) == "Helloworld"       # boxes do not match the characters
    assert screens.respace(img, "Different", boxes, chars) == "Different"
    assert screens.respace(img, "x", None, None) == "x"


def test_a_space_that_would_leave_one_stray_letter_is_not_added():
    # wide capitals leave gaps as big as a space: 'PANAL O!' / 'CASHOU T' are rejected, 'on a' (a real word) is kept
    img, chars, boxes = _draw_words(["on", "a", "card"])
    assert screens.respace(img, "onacard", boxes, chars) == "on a card"
    img, chars, boxes = _draw_words(["PANAL", "O!"])
    assert screens.respace(img, "PANALO!", boxes, chars) == "PANALO!"


# ---------------------------------------------------------------------------------------------- Samsung / Android crop / iPhone names
def test_link_wrapped_at_its_hyphen_is_rejoined():
    # Samsung Messages breaks "https://bdo-on.cc" after "bdo" and starts the next line with the hyphen
    lines = [L("Check your offer here: https://https://bdo", 109, 640, 633), L("-on.cc Valid until 07/20/26", 109, 700, 687),
             L("only.", 109, 760, 205)]
    b = screens.split_messages(lines, 1080.0, 1075.0)
    assert b[0].text == "Check your offer here: https://https://bdo-on.cc Valid until 07/20/26 only."
    assert ds.first_link(b[0].text) is not None


def test_a_word_broken_after_a_hyphen_has_no_space_added():
    b = screens.split_messages([L("Check your pre-", 109, 640, 633), L("approved offer today", 109, 700, 687)], 1080.0, 1075.0)
    assert b[0].text == "Check your pre-approved offer today"


def test_a_line_with_a_dot_is_not_mistaken_for_a_link_cut_at_the_edge():
    # ':5159901.com,Responsible' is the widest line but is words, not a cut link: 'Gaming' needs its space
    lines = [L("Big Welcome Gift, PHP88 Free Credit for new", 68, 260, 400), L("users. Reply YES Claim", 68, 320, 370),
             L(":5159901.com,Responsible", 68, 370, 430), L("Gaming", 68, 405, 170)]
    assert screens.split_messages(lines, 720.0, 523.0)[0].text.endswith(":5159901.com,Responsible Gaming")


def test_app_bar_above_the_first_separator_is_dropped_even_in_a_small_crop():
    # a 720 x 523 crop: the contact name sits at 21% of the height, far below the usual app-bar band
    lines = [L("MEGASLOt", 189, 90, 380, h=30), L("Thursday, May 28", 276, 183, 447, h=26),
             L("Big Welcome Gift, PHP88 Free", 68, 262, 400, h=30), L("5159901.com", 75, 296, 200, h=30)]
    texts = [x.text for x in screens.split_messages(lines, 720.0, 523.0)]
    assert texts == ["Big Welcome Gift, PHP88 Free 5159901.com"]
    # but a short message that contains a link above the first separator is kept
    keep = screens.split_messages([L("bit.ly/3xYzAb", 70, 90, 300, h=30), L("Thursday, May 28", 276, 183, 447, h=26)], 720.0, 523.0)
    assert [x.text for x in keep] == ["bit.ly/3xYzAb"]


def test_iphone_contact_pill_under_a_large_avatar_is_dropped():
    lines = [L("SDQL 888>", 355, 272, 560, h=66), L("Text Message�SMS", 335, 392, 585, h=36), L("Sun,Sep13at7:00PM", 321, 425, 598, h=36),
             L("50 FREE SPINS", 70, 497, 306, h=40), L("Login now. https://bit.ly/4gR9dbN", 70, 548, 591, h=40)]
    assert [x.text for x in screens.split_messages(lines, 920.0, 2000.0)] == ["50 FREE SPINS Login now. https://bit.ly/4gR9dbN"]


def test_fullwidth_characters_from_the_ocr_become_ascii():
    np = pytest.importorskip("numpy")
    reader = object.__new__(screens.Reader)
    reader._engine = lambda img, **kw: ([[[[0, 0], [90, 0], [90, 10], [0, 10]], "Claim：5159901.com，Now", 0.9]], None)
    assert reader.lines(np.zeros((20, 100, 3), dtype=np.uint8))[0].text == "Claim:5159901.com,Now"


def test_lowercase_then_capital_needs_a_smaller_gap_but_other_pairs_do_not():
    img, chars, boxes = _draw_words(["Reply", "YES"], gap=0.25)
    assert screens.respace(img, "ReplyYES", boxes, chars) == "Reply YES"
    img, chars, boxes = _draw_words(["reply", "yes"], gap=0.25)               # same gap, lower-case pair: not enough
    assert screens.respace(img, "replyyes", boxes, chars) == "replyyes"
    img, chars, boxes = _draw_words(["Union", "Bank"], gap=0.0)               # a brand name: letters at their normal spacing
    assert screens.respace(img, "UnionBank", boxes, chars) == "UnionBank"
