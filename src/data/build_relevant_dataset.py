r"""Build a synthetic SMS + link dataset in which the text and the link are about the same thing (a Smart text carries a Smart link).

Output: data/processed/crowd_relevant_synthetic.csv   the crowd schema (57 columns) + `has_url`
        data/processed/crowd_relevant_synthetic.meta.csv   one provenance row per message (where the text, link and features came from)
        data/processed/crowd_relevant_synthetic.summary.json
It is a SYNTHETIC dataset for building and rehearsing the pipeline, never study evidence. The meta file's `text_source` says which rows have
real wording (isda_real) and which are written by templates (synthetic).

SOURCES
    text   data/raw/sms/isda_sms_dataset.csv        real Philippine SMS (Ham 1,501 / smishing-type Spam 1,259); the 310 spam rows that
                                                    are not smishing (ads, job offers) are left out: their label is ambiguous
    page   data/raw/web-phishing/train-1.csv, test.csv   (tt)    11,430 links with their 37 features, 50 / 50
           data/raw/web-phishing/crowd_pilot_synthetic.csv (lean) links of the earlier pilot, with their features (631 legitimate, 636 phishing)
           data/raw/web-phishing/hunt_live_features.csv   (hunt) 960 live phishing links from feeds (links only: see below)

HOW A TEXT AND A LINK ARE MADE TO FIT
    * A real ISDA message keeps the link it already contains. Its 20 URL features are computed from that link (the repo's own
      offline extractor, lejet.features.hannousse_url). Its 17 page / domain features come from a fetched page of the same domain or
      brand when one exists (a smrt.ph text borrows a smrt.ph page), otherwise the page is "not fetched".
    * A synthetic message is written AFTER its link is chosen, from what the link says about itself (relevance_lexicon.tag_url): its
      brand, its theme, the title in its path, its site name (relevance_templates). A link naming PayPal never gets a Smart text.
    Relation recorded per row (meta): own_link / brand / theme / generic. A conflict (text names one brand, link another) is refused.

THE DESIGN (cells) - so that the words alone, or the link alone, do not decide the label
    S1 scam wording + phishing link          S2 neutral wording + phishing link         S3 scam wording + a clean brand page (Facebook, Google..)
    S0 smishing with no link                 L1 genuine notice + its own link          L2 genuine but alarming notice + its own link
    L0 legitimate with no link               (neutral wording is used for BOTH classes)

TWO CONTROLS AGAINST SHORTCUTS
    1. Every pool of page features has its own crawler signature: a classifier tells two pools of the same class apart with AUC 0.986
       to 0.998, and hunt vs anything with AUC 1.000 (hunt never looked up page_rank, whois or domain age). So the model could learn
       "which pool" instead of "which link". Hence (a) hunt contributes LINKS ONLY (its page features are not used; --hunt-pages
       turns that off), and (b) tt and lean each contribute BOTH classes in near-equal numbers.
    2. Rows whose page is "not fetched" get the same treatment whatever the class: the 17 page / domain features are -1 and
       nb_redirection is 0, URL features recomputed from the link text. About a third of smishing rows and under a fifth of
       legitimate rows are like this (dead smishing pages are real; the gap is reported, not hidden).

Deviations from the crowd schema, all deliberate: rows without a link are kept (has_url = 0, link = -1, link_domain = nolink_<n> so each is its own
group in a grouped split); the per-domain cap is
MAX_PER_DOMAIN (60), not 20, because a brand's own domain legitimately carries many messages.

Usage:
    python -m src.data.build_relevant_dataset [--seed 42] [--scale 1.0] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))                       # lejet.features.* are imported by their package name

from lejet.features.hannousse_url import url_features  # noqa: E402

from src.data import build_crowd_dataset as bcd  # noqa: E402
from src.data import clean_analysis_dataset as cad  # noqa: E402
from src.data import relevance_lexicon as lex  # noqa: E402
from src.data import relevance_templates as tpl  # noqa: E402
from src.features.link_features import extract_links  # noqa: E402
from src.unified.prepare_web_37 import FEATURES_37, PAGE_FEATURES, URL_FEATURES  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "crowd_relevant_synthetic.csv"
MAX_PER_DOMAIN = 60
COLUMNS = [c for c in bcd.COLUMNS if c != "label_source"]      # the crowd schema without label_source; where a row came from is in the meta file
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)

# Rows wanted per group at --scale 1.0 (about 6,500 in all). Real ISDA groups are capped by what exists.
QUOTA = {
    "isda_smish_link": 800, "isda_ham_link": 400, "isda_ham_nolink": 480, "isda_smish_nolink": 150, "smish_nolink_synth": 165,
    "smish_tt": 1300, "smish_lean": 330, "smish_hunt": 330, "smish_s3": 200,
    "legit_tt": 1500, "legit_lean": 330, "legit_blank": 900,
}
# The wording shares below trade RELEVANCE against AMBIGUITY (measured on the default build, 5-fold CV grouped by link domain):
#   --smish-neutral 0.35 --smish-legit-style 0.40 --legit-neutral 0.45   61% of link rows name a brand / page / theme, text-only AUC 0.936
#   defaults (0.20, 0.40, 0.30)                                           70%                                          text-only AUC 0.960
#   --smish-neutral 0.12 --smish-legit-style 0.45 --legit-neutral 0.20    77%                                          text-only AUC 0.975
# Neutral wording ("Please see the details here") fits any link: compatible, but not relevant. It is also the only wording that is
# truly uninformative (text-only AUC on those rows 0.49 to 0.52).
# How a PHISHING link is worded (S1 scam wording / S2 anything that does not sound like a scam), as shares of the rows of that kind.
# S2 is what keeps the words from deciding the label: a neutral line, or a notice written like a genuine one (a receipt, a bill)
# whose link is the phishing one. Likewise for a CLEAN link: neutral wording and urgent-but-genuine notices (L2) overlap the scams.
SMISH_NEUTRAL, SMISH_LEGIT_STYLE = 0.20, 0.40
LEGIT_NEUTRAL, LEGIT_URGENT_SHARE, LEGIT_SECURITY_SHARE, LEGIT_PRESSURE_SHARE = 0.30, 0.25, 0.20, 0.45
S3_ONLY_IF = {"google": r"forms|docs|drive|sites", "microsoft": r"sharepoint|onedrive"}      # a clean page that can plausibly carry a scam's call to action
BRAND_CAP = {"smish": 70, "legit": 45, "hunt": 60}
SMISH_THEME_FOR_BRAND = {"whatsapp": "account", "instagram": "social", "linkedin": "job", "zoom": "account", "dhl": "delivery", "fedex": "delivery",
                         "usps": "delivery", "ups": "delivery", "lbc": "delivery", "jnt": "delivery", "alibaba": "shopping", "binance": "crypto",
                         "roblox": "prize", "steam": "prize", "spotify": "subscription", "netflix": "subscription", "amazon": "shopping",
                         "paypal": "payment", "facebook": "social", "google": "account", "microsoft": "account", "apple": "account",
                         "docusign": "account", "smart": "telco", "globe": "telco", "dito": "telco", "gcash": "payment", "maya": "payment",
                         "sss": "gov", "pagibig": "gov", "philhealth": "gov", "ntc": "gov", "meralco": "payment", "dost": "gov"}
# Brands whose genuine texts point at an app or account page of their own, at any depth. A global platform (Google, Microsoft, Apple,
# Facebook...) sends account notices to its front pages, not to a docs article or a store listing, so a deep link on one of those is
# written as a share of that page instead.
PH_BRANDS = {"smart", "globe", "dito", "gcash", "maya", "bdo", "bpi", "metrobank", "unionbank", "landbank", "gotyme", "seabank", "coinsph",
             "meralco", "sss", "pagibig", "philhealth", "ntc", "dost", "lbc", "jnt", "grab", "foodpanda", "lazada", "shopee"}
THEME_PRIORITY = ["gambling", "delivery", "crypto", "job", "telco", "gov", "shopping", "payment", "account", "social", "prize"]


@dataclass
class Link:
    url: str                       # as written in the SMS (defanged for hunt)
    domain: str
    cls: str                       # "phish" or "legit"
    source: str                    # tt / lean / hunt
    features: dict | None          # the 37 features of a fetched page, or None
    tags: lex.LinkTags = field(default_factory=lex.LinkTags)

    @property
    def norm(self) -> str:
        return lex.normalise_url(self.url)


# ------------------------------------------------------------------------------------------------------------ pools
def _clean_features(row: pd.Series) -> dict:
    values = {f: float(row[f]) for f in FEATURES_37}
    for f in ("domain_age", "domain_registration_length"):
        if values[f] < 0:
            values[f] = -1.0
    return values


def _domain(url: str) -> str:
    e = _EXTRACT(lex.normalise_url(url).split("//", 1)[-1].split("/")[0].split(":")[0])
    return (e.domain + "." + e.suffix).lower() if e.suffix else e.domain.lower()


def _usable(url: str) -> bool:
    """Only links the pipeline's own extractor finds again, whole, inside a sentence."""
    found = extract_links(f"see {url} now")
    return bool(found) and found[0]["raw"].rstrip(".,") == url and " " not in url and len(url) <= 120


def load_pools() -> dict[str, list[Link]]:
    tt = pd.concat([pd.read_csv(RAW / "web-phishing" / n, low_memory=False) for n in ("train-1.csv", "test.csv")], ignore_index=True)
    tt = tt.drop_duplicates("url")
    pilot = pd.read_csv(RAW / "web-phishing" / "crowd_pilot_synthetic.csv", keep_default_na=False)
    hunt = pd.read_csv(RAW / "web-phishing" / "hunt_live_features.csv", keep_default_na=False)
    hunted = set(hunt["url"].map(bcd._refang))
    pools: dict[str, list[Link]] = {"tt_phish": [], "tt_legit": [], "lean_phish": [], "lean_legit": [], "hunt": []}
    for _, r in tt.iterrows():
        if _usable(r["url"]):
            cls = "phish" if r["status"] == "phishing" else "legit"
            pools[f"tt_{cls}"].append(Link(r["url"], _domain(r["url"]), cls, "tt", _clean_features(r), lex.tag_url(r["url"])))
    for _, r in pilot.iterrows():
        if not _usable(r["link"]):
            continue
        if r["label"] == "legitimate":
            pools["lean_legit"].append(Link(r["link"], _domain(r["link"]), "legit", "lean", _clean_features(r), lex.tag_url(r["link"])))
        elif r["link"] not in hunted:
            pools["lean_phish"].append(Link(r["link"], _domain(r["link"]), "phish", "lean", _clean_features(r), lex.tag_url(r["link"])))
    for url in hunt["url"]:
        plain = bcd._refang(url)
        if _usable(url) and _usable(plain):                                        # keep it defanged in the text, as the lab does
            pools["hunt"].append(Link(url, _domain(plain), "phish", "hunt", None, lex.tag_url(plain)))
    return pools


def blank_features(url_norm: str) -> dict:
    """A page that was not fetched: the 20 URL features read off the link, the 17 page / domain features -1, no redirect count."""
    computed = url_features(url_norm)
    out = {f: float(computed[f]) for f in URL_FEATURES}
    out["nb_redirection"] = 0.0
    out.update({f: -1.0 for f in PAGE_FEATURES})
    return out


# ------------------------------------------------------------------------------------------------------- row assembly
@dataclass
class Row:
    text: str
    link: str                      # "" when there is no link
    label: str
    features: dict
    page_fetched: int
    text_source: str               # isda_real / synthetic
    link_source: str               # isda_own / tt / lean / hunt / none
    feature_source: str            # live_tt / live_lean / offline_url / lean_borrowed / none
    pair_type: str
    domain: str = "-1"
    sender: str = "unknown"
    theme: str = ""


def _category(text: str, link_tags: lex.LinkTags | None = None) -> str:
    themes = lex.tag_message(text).themes | (link_tags.themes if link_tags else set())
    for t in THEME_PRIORITY:
        if t in themes:
            return t
    return "other"


def _theme_of_link(tags: lex.LinkTags, allowed) -> str | None:
    for t in THEME_PRIORITY:
        if t in tags.themes and t in allowed:
            return t
    return None


def _first_brand(tags: lex.LinkTags) -> str | None:
    """A brand the link names (a look-alike counts): for phishing links."""
    return sorted(tags.brands)[0] if tags.brands else None


def _own_brand(tags: lex.LinkTags) -> str | None:
    """A brand whose own domain the link is on: the only kind a genuine notice may speak for."""
    return sorted(tags.owned)[0] if tags.owned else None


def _brand_voice_ok(link: Link, brand: str) -> bool:
    """A genuine notice in the brand's voice needs the brand's own domain, and either a Philippine brand or a front page."""
    if brand in PH_BRANDS:
        return True
    path = re.sub(r"^https?://[^/]+", "", link.norm.split("#")[0])
    return path.strip("/") == "" or (path.count("/") <= 2 and "?" not in path)


def _host_theme(tags: lex.LinkTags, allowed) -> str | None:
    """A theme the HOST of a clean link names. A path such as /logistics-blog says nothing about what the site is."""
    for t in THEME_PRIORITY:
        if t == "gov" and not tags.host.endswith(".ph"):                  # these texts are Philippine: a Brazilian or US agency is not their sender
            continue
        if t in tags.host_themes and t in allowed:
            return t
    return None


def smish_text(rng, link: Link) -> tuple[str, str] | None:
    """(text, pair_type) for a phishing link: S1 scam wording that fits it, or S2 wording that does not sound like a scam."""
    brand = _first_brand(link.tags)
    roll = rng.random()
    if brand:
        theme = SMISH_THEME_FOR_BRAND.get(brand, lex.default_theme(brand))
        if not tpl.has_smish(theme):
            theme = "account"
        if roll < SMISH_NEUTRAL:
            return tpl.neutral(rng, link.url), "S2"
        if roll < SMISH_NEUTRAL + SMISH_LEGIT_STYLE and tpl.has_legit_brand(lex.default_theme(brand)):
            return tpl.legit_brand(rng, link.url, brand, lex.default_theme(brand)), "S2"       # a copy of a genuine notice, wrong link
        return tpl.smish(rng, link.url, theme, brand), "S1"
    theme = _theme_of_link(link.tags, set(tpl._SMISH))
    if theme:
        if roll < SMISH_NEUTRAL:
            return tpl.neutral(rng, link.url), "S2"
        if roll < SMISH_NEUTRAL + SMISH_LEGIT_STYLE and tpl.has_legit_theme(theme):
            return tpl.legit_theme(rng, link.url, theme), "S2"
        return tpl.smish(rng, link.url, theme), "S1"
    if roll < 0.3:
        return tpl.smish(rng, link.url, "account"), "S1"                              # a plain link: a generic notice contradicts nothing
    return tpl.neutral(rng, link.url), "S2"


def legit_text(rng, link: Link) -> tuple[str, str] | None:
    """(text, pair_type) for a clean link: a genuine notice from its brand (L1), an urgent or security-minded one (L2), a notice on
    its theme, a share naming the page, or neutral wording."""
    brand = _own_brand(link.tags)
    roll = rng.random()
    if brand and _brand_voice_ok(link, brand):
        theme = lex.default_theme(brand)
        if tpl.has_legit_brand(theme):
            if roll < LEGIT_URGENT_SHARE:
                return tpl.legit_urgent(rng, link.url, brand), "L2"
            if roll < LEGIT_URGENT_SHARE + LEGIT_SECURITY_SHARE:
                return tpl.legit_security(rng, link.url, brand), "L2"
            return tpl.legit_brand(rng, link.url, brand, theme), "L1"
    if roll < LEGIT_NEUTRAL:
        return tpl.neutral(rng, link.url), "L1"
    theme = _host_theme(link.tags, set(tpl._LEGIT_THEME))
    if theme and tpl.has_legit_pressure(theme) and rng.random() < LEGIT_PRESSURE_SHARE:
        return tpl.legit_pressure(rng, link.url, theme), "L2"                         # sounds like a scam, is a real sale or deadline
    if theme and rng.random() < 0.5:
        return tpl.legit_theme(rng, link.url, theme), "L1"
    shared = tpl.legit_share(rng, link.url, link.tags.title, link.tags.site)
    return (shared, "L1") if shared else (tpl.neutral(rng, link.url), "L1")


def scam_benign_text(rng, link: Link) -> tuple[str, str] | None:
    """S3: scam wording for a clean brand page that can plausibly carry a call to action (a Facebook page, a Google form)."""
    for brand in sorted(link.tags.owned):
        if brand in tpl._SCAM_BENIGN and (brand not in S3_ONLY_IF or re.search(S3_ONLY_IF[brand], link.norm.lower())):
            text = tpl.scam_benign(rng, link.url, brand)
            if text:
                return text, "S3"
    return None


def _balanced(links: list[Link], n: int, rng, cap_key: str) -> list[Link]:
    """n links, brand-tagged ones first (at most BRAND_CAP[cap_key] per brand), then theme-tagged, then the rest; all shuffled within tier."""
    order = rng.permutation(len(links))
    shuffled = [links[i] for i in order]
    per_brand: dict[str, int] = {}
    tier1, tier2, tier3 = [], [], []
    for link in shuffled:
        brand = _first_brand(link.tags)
        if brand and per_brand.get(brand, 0) < BRAND_CAP[cap_key]:
            per_brand[brand] = per_brand.get(brand, 0) + 1
            tier1.append(link)
        elif link.tags.themes:
            tier2.append(link)
        elif not brand:
            tier3.append(link)
    rest = max(n - len(tier1), 0)
    n2 = min(len(tier2), int(rest * 0.75))
    picked = tier1[:n] + tier2[:n2] + tier3[: max(rest - n2, 0)]
    return picked[:n]


def _domain_cap_ok(counts: dict, domain: str, label: str) -> bool:
    key = (domain, label)
    if counts.get(key, 0) >= MAX_PER_DOMAIN:
        return False
    counts[key] = counts.get(key, 0) + 1
    return True


def link_row(link: Link, text: str, label: str, pair_type: str, features_live: bool) -> Row:
    live = features_live and link.features is not None
    return Row(text=text, link=link.url, label=label,
               features=link.features if live else blank_features(link.norm), page_fetched=int(live), text_source="synthetic",
               link_source=link.source, feature_source=(f"live_{link.source}" if live else "offline_url"), pair_type=pair_type,
               domain=link.domain, theme=_category(text, link.tags))


# --------------------------------------------------------------------------------------------------------- ISDA rows
def load_isda(rng, pools: dict[str, list[Link]], counts: dict, quota: dict) -> list[Row]:
    isda = pd.read_csv(RAW / "sms" / "isda_sms_dataset.csv", keep_default_na=False)
    isda = isda[(isda["SMISH"] == 1) | (isda["CLASSIFICATION"] == "Ham")].copy()          # the 310 spam rows that are not smishing are left out
    isda["label"] = np.where(isda["SMISH"] == 1, "smishing", "legitimate")
    isda["text"] = (isda["MESSAGE"].str.replace("\ufffd", "", regex=False).map(cad.fix_mojibake).map(cad.squash)
                    .map(cad.join_split_domains).map(bcd.mask_numbers))
    isda = isda[isda["text"].str.len() >= 8]
    isda = isda[~isda["text"].str.casefold().duplicated()].sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    lean_by_domain: dict[str, list[Link]] = {}
    lean_by_brand: dict[str, list[Link]] = {}
    for link in pools["lean_legit"]:
        lean_by_domain.setdefault(link.domain, []).append(link)
        for b in link.tags.brands:
            lean_by_brand.setdefault(b, []).append(link)
    rows: list[Row] = []
    used = {"isda_smish_link": 0, "isda_ham_link": 0, "isda_ham_nolink": 0, "isda_smish_nolink": 0}
    for r in isda.itertuples():
        links = extract_links(r.text)
        sender = "alphanumeric" if re.search(r"[A-Za-z]{3}", str(r.ADDRESS)) else "mobile"
        if not links:
            key = "isda_smish_nolink" if r.label == "smishing" else "isda_ham_nolink"
            if used[key] < quota[key]:
                used[key] += 1
                rows.append(Row(r.text, "", r.label, {f: -1.0 for f in FEATURES_37}, 0, "isda_real", "none", "none",
                                "S0" if r.label == "smishing" else "L0", sender=sender, theme=_category(r.text)))
            continue
        link = links[0]["raw"].rstrip(".,")
        key = "isda_smish_link" if r.label == "smishing" else "isda_ham_link"
        domain = _domain(link)
        if used[key] >= quota[key] or not _domain_cap_ok(counts, domain, r.label):
            continue
        used[key] += 1
        tags = lex.tag_url(link)
        features = blank_features(lex.normalise_url(link))
        feature_source, fetched = "offline_url", 0
        if r.label == "legitimate":                                                   # a genuine message: borrow a fetched page of that domain / brand
            donors = lean_by_domain.get(domain) or next((lean_by_brand[b] for b in sorted(tags.brands) if b in lean_by_brand), [])
            if donors:
                donor = donors[int(rng.integers(len(donors)))]
                features = {**{f: features[f] for f in URL_FEATURES}, **{f: donor.features[f] for f in PAGE_FEATURES}}
                features["nb_redirection"] = donor.features["nb_redirection"]
                feature_source, fetched = "lean_borrowed", 1
        rows.append(Row(r.text, link, r.label, features, fetched, "isda_real", "isda_own", feature_source,
                        "S1" if r.label == "smishing" else ("L2" if _is_alarming(r.text) else "L1"),
                        domain=domain, sender=sender, theme=_category(r.text, tags)))
    return rows


def _is_alarming(text: str) -> bool:
    return bool(re.search(r"\b(otp|password|scam|suspicious|never (?:share|click|ask)|beware|mag-?ingat|babala|huwag)\b", text, re.IGNORECASE))


# --------------------------------------------------------------------------------------------------------- synthetic
def synthetic_rows(rng, pools: dict[str, list[Link]], counts: dict, quota: dict, hunt_pages: bool) -> list[Row]:
    rows: list[Row] = []
    seen_pairs: set = set()

    def add(link: Link, made: tuple[str, str] | None, label: str, live: bool) -> bool:
        if made is None or (made[0], link.url) in seen_pairs or not _domain_cap_ok(counts, link.domain, label):
            return False
        seen_pairs.add((made[0], link.url))
        rows.append(link_row(link, made[0], label, made[1], live))
        return True

    # phishing links: tt and lean pages are live; hunt is links only (see the module doc)
    for pool, key, cap in (("tt_phish", "smish_tt", "smish"), ("lean_phish", "smish_lean", "smish"), ("hunt", "smish_hunt", "hunt")):
        for link in _balanced(pools[pool], quota[key], rng, cap):
            add(link, smish_text(rng, link), "smishing", live=hunt_pages or pool != "hunt")

    # clean links, labelled legitimate; the first `legit_blank` of the tt ones get a page that was not fetched
    tt_legit = _balanced(pools["tt_legit"], quota["legit_tt"] + quota["legit_blank"] + quota["smish_s3"], rng, "legit")
    lean_legit = _balanced(pools["lean_legit"], quota["legit_lean"], rng, "legit")
    s3_pool = [l for l in tt_legit + lean_legit if any(b in tpl._SCAM_BENIGN for b in l.tags.owned)]
    s3_ids, s3_rows = set(), 0
    for _ in range(2):                                                              # S3: scam wording, a real clean platform page; label smishing
        for link in s3_pool:                                                        # each page may carry two different scam texts
            if s3_rows < quota["smish_s3"] and add(link, scam_benign_text(rng, link), "smishing", live=True):
                s3_ids.add(id(link))
                s3_rows += 1
    blank_ids = {id(l) for l in rng.permutation([l for l in tt_legit if id(l) not in s3_ids])[: quota["legit_blank"]]}
    for link in tt_legit + lean_legit:
        if id(link) in s3_ids:
            continue
        add(link, legit_text(rng, link), "legitimate", live=id(link) not in blank_ids)
    seen_s0, tries = set(), 0
    while len(seen_s0) < quota["smish_nolink_synth"] and tries < 50 * max(quota["smish_nolink_synth"], 1):   # S0: a number to call or text, no link
        tries += 1
        text = tpl.smish_nolink(rng)
        if text not in seen_s0:
            seen_s0.add(text)
            rows.append(Row(text, "", "smishing", {f: -1.0 for f in FEATURES_37}, 0, "synthetic", "none", "none", "S0"))
    return rows


# ---------------------------------------------------------------------------------------------------------- output
def to_frame(rows: list[Row], rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    out, meta = [], []
    for i, r in enumerate(rows, start=1):
        has_url = int(bool(r.link))
        fetched = r.page_fetched
        base = {
            "message_id": f"rel_{i:06d}", "sms_text": r.text, "label": r.label, "category": r.theme or "other",
            "language": bcd.language_for(r.text), "sender_type": r.sender, "date_received": bcd.MISSING, "date_submitted": bcd.MISSING,
            "label_verified_by": bcd.MISSING, "consent": 1,
            "link": r.link if has_url else str(bcd.MISSING), "final_url": bcd.MISSING,
            "link_domain": r.domain if has_url else f"nolink_{i:06d}",                  # one group per row, so a grouped split cannot pile them into one fold
            "template_id": "", "link_shortened": bcd.is_shortened(r.link) if has_url else 0, "inspected_at": bcd.MISSING,
            "hours_to_inspect": bcd.MISSING, "page_fetched": fetched,
            "fetch_status": "no_link" if not has_url else ("live" if fetched else "dead"),
        }
        out.append({**base, **r.features, "has_url": has_url})
        words = lex.strip_links(r.text)                                              # tags come from the words, not from the link inside them
        tt = lex.tag_text(words)
        lt = lex.tag_url(r.link) if has_url else lex.LinkTags()
        genuine = r.label == "legitimate" or r.pair_type == "S3"                      # the link is a clean one: its brand must be its own
        relation_words = "none" if not has_url else lex.relation(tt, lt, words, genuine)
        relation = "none" if not has_url else ("own_link" if r.link_source == "isda_own" else relation_words)
        meta.append({"message_id": base["message_id"], "pair_type": r.pair_type, "text_source": r.text_source, "link_source": r.link_source,
                     "feature_source": r.feature_source, "relation": relation,
                     "relation_check": relation_words,
                     "text_brands": "|".join(sorted(tt.brands)), "link_brands": "|".join(sorted(lt.brands)),
                     "text_themes": "|".join(sorted(tt.themes)), "link_themes": "|".join(sorted(lt.themes))})
    df = pd.DataFrame(out, columns=COLUMNS + ["has_url"])
    for f in FEATURES_37:                                                              # counts and flags are whole numbers, as in the crowd schema
        if (df[f] % 1 == 0).all():
            df[f] = df[f].astype(int)
    df["template_id"] = bcd.template_ids(df["sms_text"]).values
    return df, pd.DataFrame(meta)


def scaled(quota: dict, scale: float) -> dict:
    return {k: int(round(v * scale)) for k, v in quota.items()}


def build(seed: int = 42, scale: float = 1.0, hunt_pages: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    quota = scaled(QUOTA, scale)
    pools = load_pools()
    counts: dict = {}
    rows = load_isda(rng, pools, counts, quota) + synthetic_rows(rng, pools, counts, quota, hunt_pages)
    df, meta = to_frame(rows, rng)
    problems = check(df, meta)
    summary = summarize(df, meta, problems, seed, scale, {k: len(v) for k, v in pools.items()})
    return df, meta, summary


def check(df: pd.DataFrame, meta: pd.DataFrame) -> list[str]:
    problems = []
    if not df["message_id"].is_unique:
        problems.append("message ids repeat")
    if set(df["label"]) - {"smishing", "legitimate"}:
        problems.append("unknown label")
    if df[COLUMNS].isna().any().any():
        problems.append("blank values: missing must be -1")
    link = df[df["has_url"] == 1]
    first = link["sms_text"].map(lambda t: (extract_links(t) or [{"raw": None}])[0]["raw"])
    if link.duplicated(["sms_text", "link"]).any():
        problems.append(f"{int(link.duplicated(['sms_text', 'link']).sum())} rows repeat the same text and link")
    if not (first.str.rstrip(".,") == link["link"]).all():
        problems.append(f"{int((first.str.rstrip('.,') != link['link']).sum())} rows whose first extracted link is not the stored link")
    written = meta[meta["relation"] != "own_link"]                          # a real message keeps whatever link it came with
    if (written["relation_check"] == "conflict").any():
        problems.append(f"{int((written['relation_check'] == 'conflict').sum())} written rows where text and link name different brands")
    if (df.groupby(["link_domain", "label"]).size().drop(bcd.MISSING, level=0, errors="ignore") > MAX_PER_DOMAIN).any():
        problems.append("a domain exceeds the per-domain cap")
    return problems


def summarize(df, meta, problems, seed, scale, pool_sizes) -> dict:
    m = meta.merge(df[["message_id", "label", "has_url", "page_fetched"]], on="message_id")
    return {"source": "SYNTHETIC relevance dataset: pipeline rehearsal, not study results", "seed": seed, "scale": scale,
            "rows": len(df), "label": df["label"].value_counts().to_dict(), "has_url": df["has_url"].value_counts().to_dict(),
            "pair_type": m.groupby(["pair_type", "label"]).size().unstack(fill_value=0).to_dict(),
            "text_source": m.groupby(["text_source", "label"]).size().unstack(fill_value=0).to_dict(),
            "feature_source": m.groupby(["feature_source", "label"]).size().unstack(fill_value=0).to_dict(),
            "relation": m.groupby(["relation", "label"]).size().unstack(fill_value=0).to_dict(),
            "page_missing_share_of_link_rows": {k: round(float(v), 4) for k, v in
                                                m[m["has_url"] == 1].groupby("label")["page_fetched"].apply(lambda s: 1 - s.mean()).items()},
            "domains": int(df.loc[df["has_url"] == 1, "link_domain"].nunique()), "templates": int(df["template_id"].nunique()),
            "pool_sizes_after_link_check": pool_sizes, "problems": problems}


def main(argv=None) -> None:
    global SMISH_NEUTRAL, SMISH_LEGIT_STYLE, LEGIT_NEUTRAL
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--hunt-pages", action="store_true", help="use hunt's own page features (they let a model tell the pool from the label)")
    ap.add_argument("--smish-neutral", type=float, default=SMISH_NEUTRAL, help="share of phishing links given neutral wording")
    ap.add_argument("--smish-legit-style", type=float, default=SMISH_LEGIT_STYLE, help="share given wording copied from a genuine notice")
    ap.add_argument("--legit-neutral", type=float, default=LEGIT_NEUTRAL, help="share of clean links given neutral wording")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    SMISH_NEUTRAL, SMISH_LEGIT_STYLE, LEGIT_NEUTRAL = a.smish_neutral, a.smish_legit_style, a.legit_neutral
    df, meta, summary = build(a.seed, a.scale, a.hunt_pages)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8")
    meta.to_csv(a.out.with_suffix(".meta.csv"), index=False, encoding="utf-8")
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {a.out} ({len(df)} rows)")
    if summary["problems"]:
        sys.exit("PROBLEMS: " + "; ".join(summary["problems"]))


if __name__ == "__main__":
    main()
