r"""Brands and themes, read from a link and from an SMS, so that a text and its link can be checked for relevance.

Used by build_relevant_dataset.py (to choose and write texts that fit their links) and check_relevance.py (to audit the result
with the same rules). Pure functions, no files, no network.

A BRAND is an organisation a message can speak for (Smart, GCash, Netflix...). A THEME is what the message is about (a parcel,
a login, a promo...). Both are read the same way from the two sides:
    from a link   host and path tokens ("my-bdo-online.com" -> bdo; "uspstracking.fun/pay" -> usps, delivery)
    from a text   the brand's name or the theme's words
Two things are then comparable: a text and its link AGREE on a brand when both name it, and CONFLICT when both name a brand and
the sets are disjoint (a Smart text with a PayPal link). A link that names no brand never conflicts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import tldextract

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)   # x.blogspot.com is its own site

# key -> (display name, URL pattern, text pattern, default theme)
# URL patterns run on the lower-case "host path" string; short names must stand alone as a label or token so that "smartphone",
# "mayan" or "grabcad" do not count. Text patterns run case-insensitively with word boundaries.
BRANDS: dict[str, tuple[str, str, str, str]] = {
    # Philippine telcos
    "smart":     ("Smart", r"(?:^|[\W_])(?:smart|smrt|gigalife|smartcommunications)(?:$|[\W_]|\.)|(?:^|[\W_])smart-?ph(?:$|[\W_]|\.)|tntph|(?:^|[\W_])tnt(?:$|[\W_])",
                  r"(?-i:\b(?:Smart|SMART|TNT|Tnt)\b)|\b(?:gigalife|giga\s?life|giga\s?stories|doble\s?giga)\b", "telco"),
    "globe":     ("Globe", r"(?:^|[\W_])(?:globe|glbe|gomo|globe\.com\.ph)(?:$|[\W_])|globeph", r"(?-i:\b(?:Globe|GLOBE|GOMO|Gomo)\b)", "telco"),
    "dito":      ("DITO", r"(?:^|[\W_])dito(?:telecom|\.ph)?(?:$|[\W_])", r"(?-i:\bDITO\b)", "telco"),
    # Philippine banks and wallets
    "gcash":     ("GCash", r"gcash", r"\bg-?cash\b", "payment"),
    "maya":      ("Maya", r"paymaya|(?:^|[\W_])maya(?:$|[\W_]|\.)|mayaph|maya-ph", r"\b(?:pay\s?maya|maya|mayarewards)\b", "payment"),
    "bdo":       ("BDO", r"(?:^|[\W_])bdo(?:$|[\W_]|[a-z]*online)|my-?bdo|bdo-?online|bankodeoro", r"\b(?:bdo|banco de oro)\b", "bank"),
    "bpi":       ("BPI", r"(?:^|[\W_])bpi(?:$|[\W_]|online)|bpionline|bankofthephilippineislands", r"\b(?:bpi|bank of the philippine islands)\b", "bank"),
    "metrobank": ("Metrobank", r"metrobank", r"\bmetrobank\b", "bank"),
    "unionbank": ("UnionBank", r"unionbank", r"\bunion\s?bank\b", "bank"),
    "landbank":  ("Landbank", r"landbank|lbp-?online", r"\bland\s?bank\b", "bank"),
    "gotyme":    ("GoTyme", r"gotyme", r"\bgo\s?tyme\b", "bank"),
    "seabank":   ("SeaBank", r"seabank", r"\bsea\s?bank\b", "bank"),
    "coinsph":   ("Coins PH", r"coins\.ph|coinsph|coins-ph", r"\bcoins(?:\.|\s)?ph\b", "crypto"),
    # Shopping, food, rides
    "shopee":    ("Shopee", r"shopee", r"\bshopee\b", "shopping"),
    "lazada":    ("Lazada", r"lazada", r"\blazada\b", "shopping"),
    "grab":      ("Grab", r"(?:^|[\W_])grab(?:food|pay|taxi)?(?:$|[\W_]|\.)", r"(?-i:\b(?:Grab|GRAB)(?:Food|Pay|FOOD|PAY)?\b)", "shopping"),
    "foodpanda": ("foodpanda", r"foodpanda", r"\bfood\s?panda\b", "shopping"),
    "amazon":    ("Amazon", r"amazon|amzn", r"\bamazon\b", "shopping"),
    "alibaba":   ("Alibaba", r"alibaba", r"\balibaba\b", "shopping"),
    "aliexpress": ("AliExpress", r"aliexpress", r"\baliexpress\b", "shopping"),
    # Global platforms
    "facebook":  ("Facebook", r"facebook|(?:^|[\W_])fb(?:$|[\W_]|\.com)|fbcdn|messenger\.com", r"\bfacebook\b|(?-i:\bFB\b)", "social"),
    "instagram": ("Instagram", r"instagram|(?:^|[\W_])ig(?:$|[\W_])", r"\binstagram\b", "social"),
    "whatsapp":  ("WhatsApp", r"whatsapp|wa\.me", r"\bwhats\s?app\b", "social"),
    "telegram":  ("Telegram", r"telegram|(?:^|[\W_])t\.me(?:$|/)", r"\btelegram\b", "social"),
    "tiktok":    ("TikTok", r"tiktok", r"\btik\s?tok\b", "social"),
    "google":    ("Google", r"google|gmail|goo\.gl|(?:^|[\W_])forms\.gle", r"\b(?:google|gmail)\b", "account"),
    "apple":     ("Apple", r"(?:^|[\W_])apple(?:id)?(?:$|[\W_]|\.com)|icloud|appleid", r"(?-i:\b(?:Apple|APPLE)\b)|\b(?:icloud|apple id)\b", "account"),
    "microsoft": ("Microsoft", r"microsoft|outlook|office365|(?:^|[\W_])office(?:$|[\W_])|onedrive|sharepoint", r"\b(?:microsoft|outlook|office\s?365)\b", "account"),
    "netflix":   ("Netflix", r"netflix", r"\bnetflix\b", "subscription"),
    "spotify":   ("Spotify", r"spotify", r"\bspotify\b", "subscription"),
    "paypal":    ("PayPal", r"paypal|paypa1", r"\bpay\s?pal\b", "payment"),
    "roblox":    ("Roblox", r"rob[l1]ox|robux", r"\b(?:roblox|robux)\b", "gaming"),
    "steam":     ("Steam", r"steam(?:community|powered)", r"(?-i:\b(?:Steam|STEAM)\b)", "gaming"),
    "binance":   ("Binance", r"binance", r"\bbinance\b", "crypto"),
    "coinbase":  ("Coinbase", r"coinbase", r"\bcoinbase\b", "crypto"),
    "kucoin":    ("KuCoin", r"kucoin", r"\bkucoin\b", "crypto"),
    "kraken":    ("Kraken", r"kraken", r"\bkraken\b", "crypto"),
    "metamask":  ("MetaMask", r"metamask", r"\bmetamask\b", "crypto"),
    "ledger":    ("Ledger", r"ledger(?:live|wallet)|ledger-", r"(?-i:\bLedger\b)", "crypto"),
    "linkedin":  ("LinkedIn", r"linkedin|lnkd\.in", r"\blinked\s?in\b", "job"),
    "zoom":      ("Zoom", r"(?:^|[\W_])zoom(?:$|[\W_]|\.us)", r"(?-i:\b(?:Zoom|ZOOM)\b)", "social"),
    "docusign":  ("DocuSign", r"docusign", r"\bdocu\s?sign\b", "account"),
    # Delivery
    "fedex":     ("FedEx", r"fedex", r"\bfed\s?ex\b", "delivery"),
    "dhl":       ("DHL", r"(?:^|[\W_])dhl(?:$|[\W_])", r"\bdhl\b", "delivery"),
    "usps":      ("USPS", r"usps", r"\busps\b", "delivery"),
    "ups":       ("UPS", r"(?:^|[\W_])ups(?:$|[\W_]|\.com)", r"(?-i:\bUPS\b)", "delivery"),
    "lbc":       ("LBC", r"(?:^|[\W_])lbc(?:$|[\W_])", r"(?-i:\bLBC\b)", "delivery"),
    "jnt":       ("J&T Express", r"(?:^|[\W_])jnt(?:express)?(?:$|[\W_])|jtexpress", r"\bj\s?&\s?t\b|\bjnt\b", "delivery"),
    # Government and utilities
    "sss":       ("SSS", r"(?:^|[\W_])sss(?:$|[\W_]|\.gov)", r"(?-i:\bSSS\b)", "gov"),
    "pagibig":   ("Pag-IBIG", r"pagibig|pag-ibig", r"\bpag-?ibig\b", "gov"),
    "philhealth": ("PhilHealth", r"philhealth", r"\bphil\s?health\b", "gov"),
    "ntc":       ("NTC", r"(?:^|[\W_])ntc(?:$|[\W_]|\.gov)", r"(?-i:\bNTC\b)", "gov"),
    "meralco":   ("Meralco", r"meralco", r"\bmeralco\b", "utility"),
    "dost":      ("DOST", r"(?:^|[\W_])dost(?:$|[\W_]|\.gov)", r"\bdost\b", "gov"),
}

# Registrable-domain labels (or full domains, when they contain a dot) that a brand really owns. A GENUINE notice from a brand must
# link to one of these; a link that only mentions the brand in its host or path ("buy-ipad.yu.hopto.org/apple-power-adapter") is a
# look-alike, which only a phishing row may carry.
OFFICIAL: dict[str, set] = {
    "smart": {"smart", "smrt", "tntph", "gigalife"}, "globe": {"globe", "glbe", "gomo"}, "dito": {"dito"}, "gcash": {"gcash"},
    "maya": {"maya", "paymaya"}, "bdo": {"bdo"}, "bpi": {"bpi"}, "metrobank": {"metrobank"}, "unionbank": {"unionbank", "unionbankph"},
    "landbank": {"landbank"}, "gotyme": {"gotyme"}, "seabank": {"seabank"}, "coinsph": {"coins.ph"}, "shopee": {"shopee"},
    "lazada": {"lazada"}, "grab": {"grab"}, "foodpanda": {"foodpanda"}, "amazon": {"amazon"}, "alibaba": {"alibaba"}, "aliexpress": {"aliexpress"},
    "facebook": {"facebook", "fb", "messenger"}, "instagram": {"instagram"}, "whatsapp": {"whatsapp", "wa.me"},
    "telegram": {"telegram", "t.me"}, "tiktok": {"tiktok"}, "google": {"google", "goo", "gmail"}, "apple": {"apple", "icloud"},
    "microsoft": {"microsoft", "live", "office", "outlook", "onedrive", "sharepoint", "microsoftonline"}, "netflix": {"netflix"},
    "spotify": {"spotify"}, "paypal": {"paypal"}, "roblox": {"roblox"}, "steam": {"steampowered", "steamcommunity"},
    "binance": {"binance"}, "coinbase": {"coinbase"}, "kucoin": {"kucoin"}, "kraken": {"kraken"}, "metamask": {"metamask"},
    "ledger": {"ledger"}, "linkedin": {"linkedin", "lnkd"}, "zoom": {"zoom"}, "docusign": {"docusign"}, "fedex": {"fedex"},
    "dhl": {"dhl"}, "usps": {"usps"}, "ups": {"ups"}, "lbc": {"lbcexpress", "lbc"}, "jnt": {"jtexpress", "jnt"}, "sss": {"sss"},
    "pagibig": {"pagibigfund"}, "philhealth": {"philhealth"}, "ntc": {"ntc"}, "meralco": {"meralco"}, "dost": {"dost"},
}

# theme -> (URL pattern on host+path, text pattern). In a URL a plain word counts only at the start of a token
# ("track" in "tracking", not in "soundtrack"; "load" alone, not in "download"); long distinctive words may sit inside a name
# ("jackpotcityph7", "uspstracking").
_T = r"(?<![a-z])"
_E = r"(?![a-z])"
THEMES: dict[str, tuple[str, str]] = {
    "gambling": (rf"casino|slot|jackpot|poker|bingo|lotto|sabong|jili|roulette|{_T}bet{_E}|bet\d|\dbet|betfil|betph|sbet|ubet|jpc|bonus|{_T}spin{_E}|win\d|\dwin|777|888|wplus|{_T}fc\d",
                 r"casino|slots?|jackpot|poker|sabong|bingo|lotto|roulette|bonus|\bbet\b|betting|jili|deposit|cashback|wagering"),
    "delivery": (rf"parcel|{_T}track|deliver|shipp?ing|{_T}post(?:al|office)?{_E}|courier|{_T}package|freight|logistic|cargo|customs|fedex|usps|dhl|lbc|jnt|{_T}ups{_E}",
                 r"parcel|package|deliver|courier|track(?:ing)?|shipment|shipping|customs|waybill|rider"),
    "account": (rf"login|log-?in|signin|sign-in|verif|{_T}account|{_T}secure|security|{_T}update|{_T}confirm|{_T}auth|password|myaccount|webmail|unlock|restore|suspend",
                r"account|login|log in|sign.?in|verify|password|\botp\b|locked|suspended|unusual activity|security|confirm your"),
    "payment": (rf"{_T}pay(?:ment)?{_E}|wallet|{_T}bank|{_T}card{_E}|{_T}cash|credit|loan|billing|invoice|refund|transfer|remit",
                r"pay(?:ment)?\b|wallet|bank|\bcard\b|balance|cash.?in|cash.?out|loan|bill(?:ing)?|invoice|refund|transaction|transfer"),
    "crypto": (rf"crypto|{_T}coin|bitcoin|btc|usdt|{_T}token{_E}|{_T}swap|nft|defi|airdrop|staking|mining|{_T}exchange",
               r"crypto|bitcoin|btc|usdt|\btokens?\b|airdrop|staking|mining|withdrawal"),
    "shopping": (rf"{_T}shop|{_T}store{_E}|{_T}deal|{_T}sale{_E}|{_T}mall|{_T}order|{_T}cart|outlet|discount|coupon|voucher|{_T}market|{_T}buy",
                 r"\border\b|\bcart\b|\bsale\b|discount|voucher|coupon|promo|\bshop|checkout|flash sale"),
    "job": (rf"{_T}job|career|hiring|vacanc|recruit|employ|{_T}apply|workfrom|workathome|talent",
            r"hiring|\bjobs?\b|vacanc|apply|salary|work from home|part-?time|recruit|per day"),
    "telco": (rf"{_T}promo|{_T}load{_E}|giga|unli|prepaid|postpaid|{_T}sim{_E}|wifi",
              r"\bdata\b|\bload\b|promo|giga|unli|prepaid|postpaid|\bsim\b|balance|subscription"),
    "gov": (r"\.gov|(?<![a-z])sss(?![a-z])|philhealth|pagibig|(?<![a-z])ntc(?![a-z])|dost|(?<![a-z])bir(?![a-z])|(?<![a-z])lto(?![a-z])|dole|dfa|barangay|municipal",
            r"government|\bgov\b|benefit|ayuda|contribution|\btax\b|permit|license|registration|\bsss\b|philhealth|pag-?ibig"),
    "social": (rf"facebook|instagram|whatsapp|telegram|tiktok|messenger|{_T}group|{_T}chat|{_T}follow|{_T}profile|zoom|{_T}meet{_E}",
               r"facebook|instagram|whatsapp|telegram|tiktok|\bgroup\b|\bpage\b|profile|follow|zoom|meeting|\bchat\b"),
    "prize": (rf"prize|reward|{_T}gift|giveaway|raffle|{_T}claim|{_T}free{_E}|survey|lucky",
              r"prize|reward|\bgift|\bwon\b|\bwin(?:ner|s)?\b|lucky|giveaway|raffle|\bclaim|\bfree\b|survey|congratulations|selected"),
}

HOSTING = {"blogspot", "wordpress", "github", "githubusercontent", "appspot", "weebly", "wixsite", "herokuapp", "netlify", "vercel", "web",
           "000webhostapp", "sites", "yolasite", "tumblr", "godaddysites", "surge", "pages", "workers", "azurewebsites", "firebaseapp"}
_HEXISH = re.compile(r"^(?:[0-9a-f]{8,}|\d+)$")


@dataclass
class LinkTags:
    host: str = ""
    domain: str = ""                       # registrable domain, lower case
    brands: set = field(default_factory=set)          # named anywhere in host or path (a look-alike counts)
    owned: set = field(default_factory=set)           # the registrable domain really is the brand's (OFFICIAL)
    themes: set = field(default_factory=set)          # named anywhere in host or path
    host_themes: set = field(default_factory=set)     # named in the host alone (a path like /logistics-blog proves nothing about the site)
    title: str = ""                        # readable words from the path ("how to make homemade soap"), "" if none
    site: str = ""                         # a readable site name from the domain ("Todayshomeowner")


@dataclass
class TextTags:
    brands: set = field(default_factory=set)
    themes: set = field(default_factory=set)


def refang(url: str) -> str:
    return re.sub(r"(?i)^hxxp", "http", str(url).replace("[.]", ".").replace("(.)", "."))


def normalise_url(url: str) -> str:
    """The link with a scheme (http:// if it had none), defanging undone: what the feature code expects."""
    url = refang(url.strip())
    return url if re.match(r"(?i)^https?://", url) else "http://" + url


def _slug_title(path: str) -> str:
    segments = [s for s in re.split(r"/+", path.lower()) if s]
    for segment in reversed(segments):
        segment = re.sub(r"\.(?:html?|php|aspx?|jsp)$", "", segment)
        words = [w for w in re.split(r"[-_+]", segment) if w]
        if len(words) >= 3 and all(w.isalpha() for w in words) and not any(_HEXISH.match(w) for w in words):
            return " ".join(words[:12])
    return ""


def tag_url(url: str) -> LinkTags:
    parsed = urlparse(normalise_url(url))
    host = (parsed.hostname or "").lower()
    ext = _EXTRACT(host)
    domain = ".".join(p for p in (ext.domain, ext.suffix) if p).lower()
    haystack = f"{host} {parsed.path.lower()}"
    tags = LinkTags(host=host, domain=domain, title=_slug_title(parsed.path))
    for key, (_, url_pattern, _, _) in BRANDS.items():
        if re.search(url_pattern, haystack):
            tags.brands.add(key)
    for key, (url_pattern, _) in THEMES.items():
        if re.search(url_pattern, haystack):
            tags.themes.add(key)
        if re.search(url_pattern, host):
            tags.host_themes.add(key)
    tags.owned = {k for k in tags.brands if ext.domain.lower() in OFFICIAL.get(k, ()) or domain in OFFICIAL.get(k, ())}
    label = ext.domain
    if label in HOSTING and ext.subdomain:                                   # "pinkskky.blogspot.ru" is the site pinkskky, not Blogspot
        label = ext.subdomain.split(".")[-1]
    if label and not re.search(r"\d{3,}", label) and 3 <= len(label) <= 22:
        tags.site = label.capitalize()
    return tags


def tag_text(text: str) -> TextTags:
    tags = TextTags()
    for key, (_, _, text_pattern, _) in BRANDS.items():
        if re.search(text_pattern, text, re.IGNORECASE):
            tags.brands.add(key)
    low = text.lower()
    for key, (_, text_pattern) in THEMES.items():
        if re.search(text_pattern, low):
            tags.themes.add(key)
    return tags


def strip_links(text: str) -> str:
    """The message with its link(s) cut out, so that tags are read from the words and never from the link itself."""
    from src.features.link_features import extract_links
    for link in reversed(extract_links(text)):
        text = text[: link["start"]] + " " + text[link["end"]:]
    return " ".join(text.split())


def tag_message(text: str) -> TextTags:
    return tag_text(strip_links(text))


def brand_name(key: str) -> str:
    return BRANDS[key][0]


def default_theme(key: str) -> str:
    return BRANDS[key][3]


def names_the_page(text: str, link_tags: LinkTags) -> bool:
    """The words mention the page itself: its site name, or at least three words of the title in its path."""
    low = text.lower()
    if link_tags.site and len(link_tags.site) >= 4 and re.search(r"(?<![a-z])" + re.escape(link_tags.site.lower()) + r"(?![a-z])", low):
        return True
    words = set(link_tags.title.split())
    return len(words) >= 3 and len({w for w in words if len(w) > 2 and w in low}) >= 3


def relation(text_tags: TextTags, link_tags: LinkTags, text: str = "", genuine: bool = False) -> str:
    """'conflict' (the text names a brand the link does not carry), 'brand' (a shared brand), 'page' (the words name the page: pass `text`,
    links cut out), 'theme' (a shared theme) or 'generic' (compatible, nothing shared).
    genuine=True is for a clean link: the brand must be the link's OWN (OFFICIAL domain), so "Apple: ..." pointing at a random site that
    merely mentions apple is a conflict. A phishing link may only imitate a brand, so genuine=False counts a mention."""
    link_brands = link_tags.owned if genuine else link_tags.brands
    if text_tags.brands & link_brands:
        return "brand"
    if text and names_the_page(text, link_tags):                    # a share of an article ABOUT a brand is the page's topic, not a brand's voice
        return "page"
    if text_tags.brands and (link_brands or (genuine and link_tags.brands)):
        return "conflict"
    if text_tags.themes & link_tags.themes:
        return "theme"
    return "generic"
