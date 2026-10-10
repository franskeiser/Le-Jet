# src/data/build_v2_with_synthetic.py
#
# Builds data/labeled/v2withSynthetic_<date>.csv = v2_2026-09-15.csv + synthetic LEGITIMATE
# messages that contain URLs, plus matching split files under data/splits_v2withSynthetic/.
#
# Why: v2 has 4,667 URL-bearing messages but only 433 of them are legitimate, and those are
# almost all telco/bank promos. The model therefore learned "<URL> present => smishing" --
# even a bare "www.facebook.com" was classified as smishing. The earlier 349-row fix
# (notebooks/06b_run_rf_gating_augmented.py) only fed synthetic rows to the RF-gating fit,
# never to the CNN/fusion training, so the bias survived.
#
# Split policy (keeps results comparable with every earlier v2 run):
#   - Real v2 rows keep their EXACT existing assignment from data/splits/seed_<s>/ -- so the
#     real test set is the same 2,833 rows the previous numbers were computed on.
#   - The 349 legacy synthetic rows (test-more-legitimate-urls-message.csv) go to train only;
#     their templates are unknown, so they can't be split leakage-free.
#   - New synthetic rows are split by TEMPLATE (GroupShuffleSplit), ~75/10/15 train/rf_fit/test,
#     so synthetic test rows come from templates the model never trained on.
import argparse
import os
import random
import string
from datetime import date

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import SEEDS
from src.data.split_dataset import ID_COLUMN, ensure_message_id, load_split_ids, save_ids

BASE_DATASET_PATH = "data/labeled/v2_2026-09-15.csv"
BASE_SPLITS_DIR = "data/splits"
LEGACY_SYNTHETIC_PATH = "data/labeled/test-more-legitimate-urls-message.csv"
LEGACY_SYNTHETIC_TAG = "synthetic_legit_url_test"
PROBE_PATH = "experiments/data/url_probe_set.csv"  # hand-written eval set -- never generated

REAL_SOURCE = "v2_2026-09-15"
LEGACY_SOURCE = "synthetic_legit_url_legacy"
NEW_SOURCE = "synthetic_legit_url_v2"
LOOKALIKE_SOURCE = "synthetic_lookalike_smishing"

ROWS_PER_TEMPLATE = 40
SMISHING_ROWS_PER_TEMPLATE = 100  # 16 lookalike templates -> ~1,600 rows

# --------------------------------------------------------------------------------------------
# Fill pools
# --------------------------------------------------------------------------------------------
FIRST_NAMES = [
    "Juan", "Maria", "Jose", "Ana", "Mark", "Angel", "Paolo", "Bea", "Carlo", "Kim", "Miguel",
    "Andrea", "Rafael", "Nicole", "Joshua", "Camille", "Gab", "Trish", "Enzo", "Patricia",
    "Kevin", "Jasmine", "Renz", "Aira", "Christian", "Sofia", "Luis", "Janine", "Marco", "Pia",
]
LAST_NAMES = [
    "santos", "reyes", "cruz", "bautista", "garcia", "mendoza", "torres", "delacruz",
    "ramos", "aquino", "navarro", "villanueva", "castillo", "fernandez", "lim", "tan",
]
GROUPS = [
    "batch2019reunion", "bsit3a", "barkadasquad", "titasofmanila", "cavitefoodies",
    "ustalumni", "manilarunners", "bikecommuterph", "phcatlovers", "plantitasph",
]
SUBREDDITS = ["Philippines", "phcareers", "PHbuildapc", "phinvest", "CasualPH", "programming"]
RESTAURANTS = [
    "Jollibee", "McDonald's", "Chowking", "Mang Inasal", "Greenwich", "KFC", "Max's",
    "Army Navy", "Shakey's", "Yellow Cab", "Starbucks", "Tim Hortons",
]
BRANCHES = ["SM North", "Cubao", "Makati Ave", "Alabang", "Pasig Kapitolyo", "Quezon Ave", "Baguio"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
CITIES = [("MNL", "CEB"), ("MNL", "DVO"), ("CEB", "MNL"), ("MNL", "ILO"), ("CRK", "MNL"),
          ("MNL", "PPS"), ("MNL", "BCD"), ("DVO", "MNL")]
SCHOOLS = [("UP Diliman", "up.edu.ph"), ("Ateneo", "ateneo.edu"), ("DLSU", "dlsu.edu.ph"),
           ("UST", "ust.edu.ph"), ("Mapua", "mapua.edu.ph"), ("PUP", "pup.edu.ph"),
           ("FEU", "feu.edu.ph"), ("Adamson", "adamson.edu.ph")]
TYPHOONS = ["Kristine", "Carina", "Egay", "Paeng", "Odette", "Aghon", "Julian", "Enteng"]
TOPICS = ["Jose_Rizal", "Mayon_Volcano", "Philippine_eagle", "Adobo", "Banaue_Rice_Terraces",
          "Manila", "Machine_learning", "Photosynthesis", "World_War_II", "Python_(programming_language)"]
TIMES = ["7am", "8am", "9am", "10am", "1pm", "2pm", "3pm", "4pm", "6pm", "7pm", "8:30pm"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "tomorrow", "later"]


def _rid(rng, n, chars=string.ascii_letters + string.digits):
    return "".join(rng.choice(chars) for _ in range(n))


def _digits(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def _handle(rng):
    return rng.choice(FIRST_NAMES).lower() + rng.choice(["", ".", "_"]) + rng.choice(LAST_NAMES) + \
        rng.choice(["", _digits(rng, 2)])


def _amount(rng, lo, hi):
    return f"{rng.randint(lo, hi):,}.{rng.choice(['00', '50', _digits(rng, 2)])}"


def _date(rng):
    return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}"


# Social / media links of the kind friends actually send each other.
def social_url(rng):
    kind = rng.randrange(18)
    www = rng.choice(["", "www.", "https://www.", "https://"])
    if kind == 0:
        return f"{www}facebook.com/{_handle(rng)}"
    if kind == 1:
        return f"{www}facebook.com/groups/{rng.choice(GROUPS)}"
    if kind == 2:
        return f"https://www.facebook.com/share/p/{_rid(rng, 16)}/"
    if kind == 3:
        return f"https://m.facebook.com/story.php?story_fbid={_digits(rng, 15)}"
    if kind == 4:
        return f"{www}facebook.com/events/{_digits(rng, 15)}"
    if kind == 5:
        return f"https://www.youtube.com/watch?v={_rid(rng, 11)}"
    if kind == 6:
        return f"https://youtu.be/{_rid(rng, 11)}"
    if kind == 7:
        return f"https://www.instagram.com/p/{_rid(rng, 11)}/"
    if kind == 8:
        return f"{www}instagram.com/{_handle(rng)}"
    if kind == 9:
        return f"https://www.tiktok.com/@{_handle(rng)}/video/{_digits(rng, 19)}"
    if kind == 10:
        return f"https://vt.tiktok.com/{_rid(rng, 9)}/"
    if kind == 11:
        return f"https://x.com/{_handle(rng)}/status/{_digits(rng, 19)}"
    if kind == 12:
        return f"https://open.spotify.com/playlist/{_rid(rng, 22)}"
    if kind == 13:
        return f"https://open.spotify.com/track/{_rid(rng, 22)}"
    if kind == 14:
        return f"https://www.reddit.com/r/{rng.choice(SUBREDDITS)}/comments/{_rid(rng, 6).lower()}/"
    if kind == 15:
        return f"https://en.wikipedia.org/wiki/{rng.choice(TOPICS)}"
    if kind == 16:
        return f"https://www.linkedin.com/in/{_handle(rng).replace('.', '-')}"
    return f"https://fb.watch/{_rid(rng, 10)}/"


def bare_social_domain(rng):
    domain = rng.choice([
        "facebook.com", "instagram.com", "youtube.com", "tiktok.com", "twitter.com", "x.com",
        "reddit.com", "linkedin.com", "messenger.com", "spotify.com", "netflix.com",
        "google.com", "gmail.com", "wikipedia.org", "pinterest.com", "threads.net",
        "shopee.ph", "lazada.com.ph", "canva.com", "github.com",
    ])
    prefix = rng.choice(["", "www.", "https://", "https://www.", "http://www."])
    return prefix + domain + rng.choice(["", "", "/"])


def work_url(rng):
    kind = rng.randrange(10)
    if kind == 0:
        return f"https://meet.google.com/{_rid(rng, 3, string.ascii_lowercase)}-" \
               f"{_rid(rng, 4, string.ascii_lowercase)}-{_rid(rng, 3, string.ascii_lowercase)}"
    if kind == 1:
        return f"https://us{rng.choice(['02', '04', '05', '06'])}web.zoom.us/j/{_digits(rng, 11)}"
    if kind == 2:
        return f"https://docs.google.com/document/d/{_rid(rng, 25)}/edit"
    if kind == 3:
        return f"https://docs.google.com/spreadsheets/d/{_rid(rng, 25)}/edit?usp=sharing"
    if kind == 4:
        return f"https://drive.google.com/drive/folders/{_rid(rng, 20)}"
    if kind == 5:
        return f"https://forms.gle/{_rid(rng, 17)}"
    if kind == 6:
        return f"https://classroom.google.com/c/{_rid(rng, 16)}"
    if kind == 7:
        return f"https://www.canva.com/design/{_rid(rng, 11)}/view"
    if kind == 8:
        return f"https://teams.microsoft.com/l/meetup-join/{_digits(rng, 8)}"
    return f"https://github.com/{_handle(rng).replace('.', '')}/{rng.choice(['thesis', 'capstone', 'portfolio', 'webapp'])}"


def maps_url(rng):
    return rng.choice([
        f"https://maps.app.goo.gl/{_rid(rng, 17)}",
        f"https://www.google.com/maps/place/{rng.choice(BRANCHES).replace(' ', '+')}",
        f"https://waze.com/ul/h{_rid(rng, 9).lower()}",
    ])


def name(rng):
    return rng.choice(FIRST_NAMES)


# --------------------------------------------------------------------------------------------
# Templates: (template_id, callable(rng) -> text). Legitimate only -- no credential requests,
# no threats, no prizes, no urgency-to-click. Mix of English, Tagalog and Taglish.
# --------------------------------------------------------------------------------------------
TEMPLATES = {
    # --- bare / near-bare URLs (the exact www.facebook.com failure case) ---
    "bare_domain": lambda r: bare_social_domain(r),
    "bare_social_link": lambda r: social_url(r),
    "bare_link_emoji": lambda r: f"{social_url(r)} {r.choice(['haha', 'hahaha', 'lol', 'lmao', '😂', '😂😂', '🥹', '❤️'])}",
    "domain_in_sentence": lambda r: r.choice([
        f"just search it on {bare_social_domain(r)}",
        f"it's on {bare_social_domain(r)} naman",
        f"nasa {bare_social_domain(r)} lang yan",
        f"go to {bare_social_domain(r)} and look for it",
    ]),
    # --- casual personal shares ---
    "share_en_checkout": lambda r: f"{r.choice(['Check this out', 'Look at this', 'Omg look', 'You have to see this'])} {social_url(r)}",
    "share_tl_tignan": lambda r: f"{r.choice(['Uy tignan mo to', 'Tingnan mo to beh', 'Pre nakita mo na to?', 'Grabe to oh'])} {social_url(r)}",
    "share_tl_sinasabi": lambda r: f"Ito yung {r.choice(['sinasabi ko sayo', 'kinukwento ko kanina', 'video na sinabi ko', 'post na pinag-uusapan natin'])} {social_url(r)}",
    "share_en_vid": lambda r: f"here's the {r.choice(['vid', 'video', 'clip', 'song', 'post', 'thread'])} {r.choice(['I was talking about', 'from earlier', 'you asked for', 'I mentioned'])} {social_url(r)}",
    "share_like_post": lambda r: f"{r.choice(['Pa-like naman ng post ko', 'Pa-share naman po', 'Pakilike po ng entry ko sa contest', 'Pa-heart naman'])} {social_url(r)} {r.choice(['salamat!', 'thank youuu', 'ty!', ''])}".strip(),
    "share_add_me": lambda r: f"{r.choice(['Add mo ko sa fb', 'Follow mo ko sa ig', 'This is my new account', 'Bago kong account to'])} {social_url(r)}",
    "share_photos": lambda r: f"{r.choice(['Photos from', 'Pics from', 'Uploaded the pictures from', 'Album from'])} {r.choice(['the reunion', 'last night', 'the wedding', 'our trip', 'the birthday party', 'Christmas party'])}: {social_url(r)}",
    "share_family": lambda r: f"{r.choice(['Ma', 'Pa', 'Tita', 'Lola', 'Kuya', 'Ate'])}, {r.choice(['ito yung video ni', 'nandito yung pics ni', 'nakapost na yung graduation ni', 'tingnan niyo po yung post ni'])} {name(r)} {social_url(r)}",
    "share_recipe": lambda r: f"{r.choice(['Yung recipe ng adobo', 'Recipe for the leche flan', 'Here is the sinigang recipe', 'Yung tutorial sa pancit'])} {social_url(r)}",
    "share_music": lambda r: f"{r.choice(['New fave song', 'Pakinggan mo to', 'Our playlist for the road trip', 'This song reminds me of you'])} {social_url(r)}",
    "share_reddit": lambda r: f"{r.choice(['Found this thread helpful', 'Basahin mo tong thread', 'Someone asked the same question here', 'Relevant sa usapan natin'])} {social_url(r)}",
    "share_news": lambda r: f"{r.choice(['Did you see this?', 'Nabasa mo na ba?', 'FYI', 'Grabe yung balita'])} https://www.{r.choice(['rappler.com', 'inquirer.net', 'philstar.com', 'gmanetwork.com/news', 'abs-cbn.com/news', 'bbc.com/news'])}/{r.choice(['nation', 'metro', 'business', 'sports', 'world'])}/{_digits(r, 6)}",
    "share_birthday": lambda r: f"Happy birthday {name(r)}! {r.choice(['Made you a little video', 'Here are our throwback pics', 'Watch this'])} {social_url(r)} 🎂",
    "share_marketplace": lambda r: f"{r.choice(['Is this still available?', 'Magkano last price?', 'Interested ako dito', 'Pwede pa-reserve?'])} https://www.facebook.com/marketplace/item/{_digits(r, 15)}/",
    "share_maps": lambda r: f"{r.choice(['Dito tayo magkita', 'Here is the location', 'Pin ng bahay', 'Meet here later'])} {maps_url(r)} {r.choice(['', 'see you!', 'ingat', 'text mo ko pag malapit ka na'])}".strip(),
    "share_wiki": lambda r: f"{r.choice(['For our report', 'Source for the assignment', 'Basahin mo to para sa quiz'])} https://en.wikipedia.org/wiki/{r.choice(TOPICS)}",
    # --- work / school ---
    "work_meeting": lambda r: f"{r.choice(['Meeting link for', 'Link for our call', 'Zoom link for', 'GMeet for'])} {r.choice(DAYS)} {r.choice(TIMES)}: {work_url(r)}",
    "work_doc": lambda r: f"{r.choice(['Here is the draft', 'Updated file', 'Pakitingnan yung doc', 'Shared the folder'])} {work_url(r)} {r.choice(['', 'pls comment', 'thanks!', 'deadline is Friday'])}".strip(),
    "work_form": lambda r: f"{r.choice(['Please answer our survey for thesis', 'Pasagot naman ng form para sa research namin', 'RSVP form for the reunion', 'Sign-up sheet'])} https://forms.gle/{_rid(r, 17)} {r.choice(['thank you!', 'salamat po', ''])}".strip(),
    "school_classroom": lambda r: f"Good {r.choice(['morning', 'afternoon', 'evening'])} class. {r.choice(['Our online class will be at', 'Join the session at', 'Lecture recording is at'])} {work_url(r)}",
    "school_suspension": lambda r: (lambda s: f"{s[0]} Advisory: Classes are suspended on {_date(r)} due to Typhoon {r.choice(TYPHOONS)}. Stay safe. Updates at {s[1]}")(r.choice(SCHOOLS)),
    "school_enrollment": lambda r: (lambda s: f"{s[0]}: Enrollment for the {r.choice(['1st', '2nd'])} semester starts on {_date(r)}. See the schedule at {s[1]}/registrar")(r.choice(SCHOOLS)),
    # --- transactional notices from official services ---
    "grab_food": lambda r: f"Your GrabFood order from {r.choice(RESTAURANTS)} is on the way! Track it in the Grab app or at grab.com/ph",
    "grab_receipt": lambda r: f"Thanks for riding with Grab! Your fare was P{_amount(r, 90, 650)}. See your receipt at grab.com/ph/help",
    "shopee_shipped": lambda r: f"Shopee: Your order {_digits(r, 6)}{_rid(r, 8).upper()} has been shipped and will arrive in {r.randint(2, 5)} days. Track at shopee.ph",
    "shopee_delivered": lambda r: f"Your Shopee parcel {_rid(r, 12).upper()} was delivered. Rate your purchase in the app. shopee.ph",
    "lazada_delivery": lambda r: f"Lazada: Your package for order {_digits(r, 12)} is out for delivery today. Track at lazada.com.ph",
    "foodpanda": lambda r: f"foodpanda: {r.choice(RESTAURANTS)} accepted your order. Estimated delivery {r.randint(20, 45)} mins. foodpanda.ph",
    "jnt_parcel": lambda r: f"J&T Express: Your parcel {_digits(r, 12)} is out for delivery. Please prepare exact amount for COD. jtexpress.ph",
    "lbc_parcel": lambda r: f"LBC: Your shipment {_digits(r, 12)} has arrived at LBC {r.choice(BRANCHES)}. Track at lbcexpress.com",
    "meralco_bill": lambda r: f"Meralco: Your bill for {r.choice(MONTHS)} is P{_amount(r, 800, 9000)}, due on {_date(r)}. View at meralco.com.ph",
    "maynilad_bill": lambda r: f"Maynilad: Your water bill of P{_amount(r, 250, 2500)} is due on {_date(r)}. Pay via Bayad Center, GCash or maynilad.com.ph",
    "telco_bill": lambda r: f"{r.choice(['Globe', 'PLDT Home', 'Smart', 'Converge'])}: Your statement for {r.choice(MONTHS)} is ready. Amount due P{_amount(r, 999, 3500)}. View at {r.choice(['globe.com.ph', 'pldthome.com', 'smart.com.ph', 'convergeict.com'])}",
    "flight_confirmed": lambda r: (lambda c: f"{r.choice(['Cebu Pacific', 'Philippine Airlines', 'AirAsia'])}: Your flight {c[0]}-{c[1]} on {_date(r)} is confirmed. Booking ref {_rid(r, 6).upper()}. Manage at {r.choice(['cebupacificair.com', 'philippineairlines.com', 'airasia.com'])}")(r.choice(CITIES)),
    "netflix_receipt": lambda r: f"{r.choice(['Netflix', 'Spotify', 'YouTube Premium', 'Disney+'])}: We received your payment of P{_amount(r, 149, 549)}. See billing details at {r.choice(['netflix.com', 'spotify.com', 'youtube.com', 'disneyplus.com'])}",
    "clinic_appt": lambda r: f"Reminder: Your appointment at {r.choice(['St. Luke’s', 'The Medical City', 'Makati Med', 'Asian Hospital'])} is on {_date(r)} at {r.choice(TIMES)}. Directions: {maps_url(r)}",
    "pagasa_advisory": lambda r: f"PAGASA: Tropical Storm {r.choice(TYPHOONS)} is expected to bring heavy rains over {r.choice(['Metro Manila', 'Luzon', 'Bicol', 'Visayas'])}. Updates at pagasa.dost.gov.ph",
    "bank_statement_info": lambda r: f"{r.choice(['BPI', 'BDO', 'Metrobank', 'UnionBank', 'Security Bank'])}: Your {r.choice(MONTHS)} credit card statement is ready. You may view it via the mobile app. Learn more at {r.choice(['bpi.com.ph', 'bdo.com.ph', 'metrobank.com.ph', 'unionbankph.com', 'securitybank.com'])}",
    "church_event": lambda r: f"{r.choice(['Simbang Gabi', 'Sunday Mass', 'Youth fellowship', 'Bible study'])} will be livestreamed at {r.choice(TIMES)}. Watch here: {social_url(r)}",
    "barangay_notice": lambda r: f"Brgy. {r.choice(['San Antonio', 'Poblacion', 'Bagong Silang', 'Santo Nino'])}: Libreng bakuna sa {_date(r)}, {r.choice(TIMES)} sa covered court. Details sa FB page namin {r.choice(['facebook.com', 'www.facebook.com'])}/brgy{r.choice(['sanantonio', 'poblacion', 'bagongsilang', 'santonino'])}",
    "tl_dinner_plans": lambda r: f"{r.choice(['Kain tayo dito', 'Try natin to sa sabado', 'Bagong bukas na resto'])} {r.choice(['https://www.facebook.com/', 'https://www.instagram.com/'])}{r.choice(['cafe', 'eats', 'kitchen', 'grill'])}{_handle(r).replace('.', '')} {r.choice(['mukhang masarap', 'libre mo ko ha', 'g ka?', ''])}".strip(),
    "en_job_post": lambda r: f"Saw this opening, might be good for you {r.choice(['https://www.linkedin.com/jobs/view/', 'https://ph.jobstreet.com/job/'])}{_digits(r, 8)}",
}


# --------------------------------------------------------------------------------------------
# Contrast set: brand-LOOKALIKE smishing (round 2). Adding only legit brand-URL rows taught the
# model "brand name in URL => legitimate", so lookalike domains (brand-verify.shop, brand.com.
# evil.top, typosquats) slipped through. These rows pair the same brands with the real attacker
# patterns so the model has to learn the difference between the brand's own domain and a
# domain that merely contains the brand. Wording deliberately differs from the hand-written
# probe set (experiments/data/url_probe_set.csv) so that set stays an independent check.
# --------------------------------------------------------------------------------------------
LOOKALIKE_BRANDS = {
    # brand display name -> domain stem used in lookalikes
    "Facebook": "facebook", "Meta": "meta", "Instagram": "instagram", "Messenger": "messenger",
    "TikTok": "tiktok", "YouTube": "youtube", "Google": "google", "Gmail": "gmail",
    "GCash": "gcash", "Maya": "maya", "BDO": "bdo", "BPI": "bpi", "Metrobank": "metrobank",
    "UnionBank": "unionbank", "Landbank": "landbank", "Shopee": "shopee", "Lazada": "lazada",
    "Grab": "grab", "Netflix": "netflix", "Spotify": "spotify", "Globe": "globe",
    "Smart": "smart", "PLDT": "pldt", "Meralco": "meralco", "LBC": "lbc", "J&T": "jnt",
    "SSS": "sss", "PhilHealth": "philhealth", "Pag-IBIG": "pagibig", "LTO": "lto",
}
LOOKALIKE_WORDS = ["verify", "secure", "support", "help", "login", "update", "confirm",
                   "recovery", "unlock", "billing", "rewards", "claim", "service", "center",
                   "ph", "online", "auth", "kyc", "promo", "refund"]
BAD_TLDS = ["xyz", "top", "shop", "win", "site", "online", "club", "vip", "live", "link",
            "click", "info", "icu", "cc", "store", "app", "me", "net", "com"]
TYPO_MAP = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5"}


def _typosquat(rng, stem):
    kind = rng.randrange(4)
    idx = [i for i, c in enumerate(stem) if c in TYPO_MAP]
    if kind == 0 and idx:
        i = rng.choice(idx)
        return stem[:i] + TYPO_MAP[stem[i]] + stem[i + 1:]
    if kind == 1 and len(stem) > 3:
        i = rng.randrange(1, len(stem))
        return stem[:i] + stem[i - 1] + stem[i:]  # doubled letter: gcassh
    if kind == 2 and len(stem) > 4:
        i = rng.randrange(1, len(stem) - 1)
        return stem[:i] + stem[i + 1:]  # dropped letter: facebok
    return stem + rng.choice(["-ph", "ph", "app", "s"])


def lookalike_url(rng, stem):
    kind = rng.randrange(6)
    path = rng.choice(["", "/login", "/verify", f"/{_rid(rng, 6)}", "/ph", "/account",
                       f"/?id={_digits(rng, 6)}", "/claim", "/update"])
    scheme = rng.choice(["", "", "https://", "http://", "www."])
    words = rng.sample(LOOKALIKE_WORDS, 2)
    if kind == 0:   # brand-word.tld
        host = f"{stem}-{words[0]}.{rng.choice(BAD_TLDS)}"
    elif kind == 1:  # word-brand-word.tld
        host = f"{words[0]}-{stem}-{words[1]}.{rng.choice(BAD_TLDS)}"
    elif kind == 2:  # brand hidden in subdomain of attacker domain
        host = f"{stem}.com.{words[0]}-{words[1]}.{rng.choice(BAD_TLDS)}"
    elif kind == 3:  # typosquat
        host = f"{_typosquat(rng, stem)}.{rng.choice(['com', 'net', 'ph', 'co', 'xyz', 'top'])}"
    elif kind == 4:  # brandword concatenated
        host = f"{stem}{words[0]}.{rng.choice(BAD_TLDS)}"
    else:           # brand as subdomain of a random host
        host = f"{stem}-{words[0]}.{_rid(rng, 5, string.ascii_lowercase)}.{rng.choice(BAD_TLDS)}"
    return scheme + host + path


def _brand(rng):
    name_, stem = rng.choice(list(LOOKALIKE_BRANDS.items()))
    return name_, stem


def _lk(fn):
    # Wrap a (rng, brand_name, url) -> text template so the brand in the text and the
    # lookalike domain in the URL always agree (real attackers keep them consistent).
    def run(rng):
        b, stem = _brand(rng)
        return fn(rng, b, lookalike_url(rng, stem))
    return run


SMISHING_TEMPLATES = {
    "lk_account_suspended": _lk(lambda r, b, u: f"{b}: {r.choice(['We detected suspicious activity on your account', 'Your account has been temporarily restricted', 'Your account is scheduled for deactivation', 'Multiple failed sign-in attempts were detected'])}. {r.choice(['Restore access at', 'Verify your identity at', 'Secure it now:', 'Review here:'])} {u}"),
    "lk_account_tl": _lk(lambda r, b, u: f"{b} ALERT: {r.choice(['Na-lock ang account mo', 'Masususpinde ang account mo ngayong araw', 'May kahina-hinalang login sa account mo', 'Kailangan i-verify ang account mo'])}. {r.choice(['I-click agad', 'Mag-login dito', 'I-update dito', 'Ayusin dito'])} {u}"),
    "lk_policy_violation": _lk(lambda r, b, u: f"{b} Notice: {r.choice(['Your profile was reported for copyright infringement', 'Your page has received multiple community standards reports', 'Your content breaks our terms of service', 'Your business account is flagged for review'])}. {r.choice(['Submit an appeal within 12 hours at', 'Avoid permanent removal by confirming at', 'File a dispute here'])} {u}"),
    "lk_kyc_update": _lk(lambda r, b, u: f"{b}: {r.choice(['Your KYC details are incomplete', 'Your e-wallet verification has expired', 'Your registered SIM needs re-validation', 'Your account limit will be reduced'])}. {r.choice(['Update before midnight at', 'Complete verification at', 'Re-verify now:'])} {u}"),
    "lk_otp_phish": _lk(lambda r, b, u: f"{b}: A transfer of P{_amount(r, 5000, 49000)} is pending approval. {r.choice(['If you did not request this, cancel it at', 'Not you? Block the transaction at', 'To reverse, sign in at'])} {u}"),
    "lk_prize": _lk(lambda r, b, u: f"{r.choice(['Congratulations', 'CONGRATS', 'Good news', 'Binabati ka namin'])}! {r.choice(['You were selected as', 'Ikaw ang napiling', 'Your number won as'])} {b}'s {r.choice(['lucky winner', 'loyalty raffle winner', 'anniversary grand prize winner'])} of P{_amount(r, 10000, 150000)}. {r.choice(['Claim it here', 'Kunin dito', 'Redeem now'])} {u}"),
    "lk_voucher": _lk(lambda r, b, u: f"{b}: {r.choice(['You have an unclaimed', 'Here is your exclusive', 'Last day to use your'])} P{r.choice(['500', '1,000', '2,000', '5,000'])} {r.choice(['voucher', 'cashback', 'gift card', 'e-coupon'])}. {r.choice(['Expires in 2 hours', 'Limited slots only', 'Valid today only'])}! {u}"),
    "lk_points_expiry": _lk(lambda r, b, u: f"{b} Rewards: {r.randint(1500, 9999)} points {r.choice(['will expire today', 'are expiring tonight', 'mawawala na bukas'])}. {r.choice(['Convert them to cash at', 'Redeem gadgets at', 'I-redeem na sa'])} {u}"),
    "lk_parcel_fee": _lk(lambda r, b, u: f"{b}: {r.choice(['Your parcel could not be delivered', 'Hindi ma-deliver ang package mo', 'Your package is held at our warehouse'])} {r.choice(['due to an unpaid customs fee', 'dahil sa maling address', 'pending a P' + str(r.randint(35, 199)) + ' redelivery fee'])}. {r.choice(['Settle it at', 'Update details at', 'Reschedule at'])} {u}"),
    "lk_billing_failed": _lk(lambda r, b, u: f"{b}: {r.choice(['We could not process your payment', 'Your subscription payment was declined', 'Your auto-debit failed'])}. {r.choice(['Update your card within 24 hours to avoid disconnection', 'Service will be cut off tonight unless you update billing', 'Re-enter payment details to keep your plan'])}: {u}"),
    "lk_refund": _lk(lambda r, b, u: f"{b}: {r.choice(['You are eligible for a refund of', 'Overpayment detected. Refund amount:', 'May refund ka na'])} P{_amount(r, 800, 12000)}. {r.choice(['Enter your bank details at', 'Claim your refund at', 'I-claim sa'])} {u}"),
    "lk_gov_ayuda": _lk(lambda r, b, u: f"{b}: {r.choice(['Kwalipikado ka sa', 'You qualified for the', 'Approved na ang'])} P{r.choice(['5,000', '6,500', '8,000', '10,000'])} {r.choice(['cash assistance', 'ayuda', 'calamity loan', 'pension bonus'])}. {r.choice(['Mag-register bago ang deadline', 'Register before slots run out', 'I-verify ang pangalan mo'])}: {u}"),
    "lk_password_expiry": _lk(lambda r, b, u: f"{b}: {r.choice(['Your password will expire in 24 hours', 'Your 2-step verification was turned off', 'Your recovery email was changed'])}. {r.choice(['Keep your current password at', 'If this was not you, go to', 'Re-enable protection at'])} {u}"),
    "lk_job_offer": _lk(lambda r, b, u: f"{r.choice(['Hi!', 'Hello po!', 'Good day!'])} {b} is hiring {r.choice(['online reviewers', 'part-time product testers', 'work-from-home encoders'])}. Earn P{r.choice(['1,500', '3,000', '5,000'])}/day. {r.choice(['Apply at', 'Register here', 'Sign up now'])} {u}"),
    "lk_bare_urgent": _lk(lambda r, b, u: f"{r.choice(['URGENT', 'Final notice', 'Action required', 'Paalala'])}: {b} {r.choice(['account verification', 'security check', 'account renewal'])} {u}"),
    "lk_friend_hacked": _lk(lambda r, b, u: f"{r.choice(['Uy ikaw ba to sa video?', 'Is this you in this video??', 'Grabe nakita kita dito', 'Someone posted your photo here'])} {u}"),
}


def generate_synthetic(rng: random.Random, existing_texts: set[str]) -> pd.DataFrame:
    rows, seen = [], set(existing_texts)
    for templates, label, source, per_template in [
        (TEMPLATES, "legitimate", NEW_SOURCE, ROWS_PER_TEMPLATE),
        (SMISHING_TEMPLATES, "smishing", LOOKALIKE_SOURCE, SMISHING_ROWS_PER_TEMPLATE),
    ]:
        for template_id, fn in templates.items():
            made, attempts = 0, 0
            while made < per_template and attempts < per_template * 50:
                attempts += 1
                text = fn(rng)
                if text in seen:
                    continue
                seen.add(text)
                rows.append({"sms_text": text, "label": label, "source": source,
                             "template_id": template_id})
                made += 1
    return pd.DataFrame(rows)


def _grouped_split(part: pd.DataFrame, seed: int):
    # test ~15%, then rf_fit ~10% of the total (== 10/85 of what's left), grouped by template
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    rest_idx, test_idx = next(gss.split(part, groups=part["template_id"]))
    rest, s_test = part.iloc[rest_idx], part.iloc[test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.10 / 0.85, random_state=seed)
    tr_idx, rf_idx = next(gss2.split(rest, groups=rest["template_id"]))
    return rest.iloc[tr_idx], rest.iloc[rf_idx], s_test


def build(out_csv: str, out_splits_dir: str, gen_seed: int = 42) -> None:
    real = ensure_message_id(pd.read_csv(BASE_DATASET_PATH))
    real = real.assign(source=REAL_SOURCE, template_id="")

    legacy = pd.read_csv(LEGACY_SYNTHETIC_PATH)
    legacy = legacy[legacy["source"] == LEGACY_SYNTHETIC_TAG][["sms_text", "label"]]
    legacy = legacy[~legacy["sms_text"].isin(set(real["sms_text"]))]
    legacy = legacy.assign(source=LEGACY_SOURCE, template_id="legacy")

    probe_texts = set(pd.read_csv(PROBE_PATH)["sms"]) if os.path.exists(PROBE_PATH) else set()
    new = generate_synthetic(random.Random(gen_seed),
                             set(real["sms_text"]) | set(legacy["sms_text"]) | probe_texts)

    synth = ensure_message_id(pd.concat([legacy, new], ignore_index=True))
    df = pd.concat([real, synth], ignore_index=True)
    assert not df["sms_text"].duplicated().any()
    assert not df[ID_COLUMN].duplicated().any()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df[["sms_text", "label", "source", "template_id"]].rename(columns={"sms_text": "sms"}).to_csv(
        out_csv, index=False, encoding="utf-8",
    )
    print(f"wrote {out_csv}: {len(df)} rows "
          f"(real={len(real)} legacy_synth={len(legacy)} "
          f"new_legit={int((new['label'] == 'legitimate').sum())} ({len(TEMPLATES)} templates) "
          f"lookalike_smishing={int((new['label'] == 'smishing').sum())} ({len(SMISHING_TEMPLATES)} templates))")
    print(df["label"].value_counts().to_dict())

    legacy_ids = synth[synth["source"] == LEGACY_SOURCE]
    for seed in SEEDS:
        # Split legit and lookalike-smishing templates separately so both labels are
        # represented in every synthetic split (a single grouped split over 62 templates can
        # otherwise put zero smishing templates in test).
        parts = [_grouped_split(synth[synth["source"] == src], seed)
                 for src in (NEW_SOURCE, LOOKALIKE_SOURCE)]
        s_train, s_rf, s_test = (pd.concat([p[i] for p in parts]) for i in range(3))

        splits = {}
        for split_name, s_part in [("train", pd.concat([s_train, legacy_ids])),
                                   ("rf_fit", s_rf), ("test", s_test)]:
            real_ids = load_split_ids(seed, split_name, BASE_SPLITS_DIR)
            ids = pd.concat([real[real[ID_COLUMN].isin(real_ids)][[ID_COLUMN]],
                             s_part[[ID_COLUMN]]])
            splits[split_name] = ids
            save_ids(ids, os.path.join(out_splits_dir, f"seed_{seed}", f"{split_name}_ids.txt"))
        sets = {k: set(v[ID_COLUMN]) for k, v in splits.items()}
        assert not (sets["train"] & sets["rf_fit"]) and not (sets["train"] & sets["test"]) \
            and not (sets["rf_fit"] & sets["test"])
        assert sum(len(s) for s in sets.values()) == len(df)
        print(f"seed={seed} train={len(sets['train'])} rf_fit={len(sets['rf_fit'])} "
              f"test={len(sets['test'])} | synthetic test templates: "
              f"{sorted(s_test['template_id'].unique())}")


def main():
    parser = argparse.ArgumentParser(description="Build the v2withSynthetic dataset + splits.")
    parser.add_argument("--output", default=f"data/labeled/v2withSynthetic_{date.today().isoformat()}.csv")
    parser.add_argument("--splits-dir", default="data/splits_v2withSynthetic")
    args = parser.parse_args()
    build(args.output, args.splits_dir)


if __name__ == "__main__":
    main()
