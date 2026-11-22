r"""The six financial subcategories of the study, assigned to a message by word and brand rules.

The six (Chapter 3 inclusion criteria): banking, e-wallet, lending, payments_billing, investment_crypto, gambling.
`categorize(text)` returns (primary, all_matched):
  * primary: the ONE category the message is mainly about; "none" if no rule matches (many legitimate chit-chat or telco-promo messages,
    and smishing that is not financial, such as Microsoft / Google account lures, job offers or parcel notices);
  * all_matched: every category a rule matched, "|"-joined (a message can touch several: the criteria say "at least one").
Primary is chosen so that a message that only MENTIONS a bank or wallet as a way to pay is not counted as banking / e-wallet:
gambling, then investment_crypto, then lending win when they match STRONGLY; otherwise the category with most hits (ties: banking,
e-wallet, payments_billing).

Version 2 (2026-11-22): gambling, investment_crypto and lending have STRONG words (casino, sabong, loan, crypto...) and WEAK words
(withdraw, cash out, rebate, interest, profit...). A weak word alone no longer wins over a bank or wallet; weak words count only next to a
strong word of the same category, or (two or more of them) when no other category matched. `ph` + digits (parcel numbers such as
"PH2264810") was dropped from gambling. Hand-checked corrections are in data/final/category_decisions.csv (applied by
build_final_from_master.py after these rules).

THIS IS AN ASSIGNMENT BY RULES plus hand corrections, not an independent annotation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

CATEGORIES = ["banking", "e-wallet", "lending", "payments_billing", "investment_crypto", "gambling"]

RULES: Dict[str, str] = {
    # ---- gambling: STRONG words decide alone; WEAK words need two or more and no shop / mall / telco brand in the message
    "gambling": (
        r"casino|(?<!interview )(?<!exam )(?<!time )(?<!your )\bslots?\b(?! ?(open|available|left|remaining))|jackpot|sabong|bingo|\bjili|color ?game|scatter|pagcor|\bbets\b|\bbetting\b|\bbet\d|\w*bet(ph)?\d*\.(com|ph|net|cc|vip|xyz)|"
        r"\bu ?b ?e ?t\b|unibet|lotto|lottery|poker|baccarat|roulette|pusoy|tongits|sportsbook|wager|super ?ace|fortune ?gems|\bpogo\b|arena ?plus|"
        r"applus|nustar|okbet|\bplayzone|free ?spins?|\d+ ?spins\b|deposit bonus|welcome bonus|(first|1st) (deposit|top ?up)|unang deposito|free chips|"
        r"sugal|\bw\+|w['.]? ?plus\b|\bkt777|fc777|pwin\d|\bace365|ggpanalo|winning ?(plus|\+)|winplus|\bjpc\b|\bj-pc\b|sugar ?(play|bet)|lolliplay|baraha|pinata ?wins|\bsbet|jpc ?ph|bett?fil|panalob|\bpin77|"
        r"\bcc ?(bet|games?)\b|\bph ?(win|bet)\b|winhub|lucky ?ph|table ?game|game ?room|online casino|place (a |your )?bets?|responsible gaming|"
        r"play responsibly|\bgcazh|gc@sh|\bbonos\b"
    ),
    "gambling_weak": (
        r"\bspins?\b|rebate|free (bonus|credit)|lucky ?(game|wheel|draw|spin)|\bgame(s)? (bonus|reward)|\bonline game|prize ?pool|\bcash ?out|"
        r"\bplay\b|\blaro\b|maglaro|laruin|paglalaro|panalo|manalo|nanalo|nananalo|swerte|\bwins?\b|\bvip\b|\bdepo\b|top ?up|\blibre(ng)?\b|"
        r"play ?now|first play|\bbounty\b|\bgame ?(app|site)\b|paldo|\btalo\b|nilalaro|papremyo|\b[1-9]\d{1,2}x\b|payout|recharge|"
        r"mystery prize|rescue cash|withdraw|red envelope|pulang sobre|\b\d{1,3}p\b|\bbonus(es)?\b(?! ?(points?|perks|data))|"
        r"[\w-]*(bonus|bet|casino|slots?|jili|panalo|win|laro|swerte|spin)[\w-]*\.(com|ph|net|cc|vip|xyz|win|live|site|club|onl|shop|co|today|ltd|one|run|app|bet)\b"
    ),
    "investment_crypto": (
        r"crypto|bitcoin|\bbtc\b|\busdt\b|ethereum|\beth\b|binance|coinbase|kraken|kucoin|metamask|\bledger\b|trust ?wallet|seed phrase|recovery phrase|\bnft\b|blockchain|\bforex\b|invest(ing|ment|ments|or|ors)?\b|stock market|stocks? (trading|investing)|"
        r"\btrading\b|\btrader\b|\bmutual fund|\bipo\b|airdrop|\bbitget|\bbybit|\bokx\b|\bgcrypto|\bdouble your|\bdividends?\b|\bmp2\b|passive income"
    ),
    "investment_crypto_weak": (
        r"mining|profit|\broi\b|\bhigh return|\breturns\b|\bearn \$|\busd\b|\$ ?\d{3,}"
    ),
    "lending": (
        r"\bloans?\b|\butang\b|pautang|\bcash ?loan|online lending|\bborrow|\blend(er|ing)?\b|\bhome ?credit|\bbillease|\bkviku|\btala\b|"
        r"\bdigido|\bcashalo|\bspay ?later|\bgcredit|\bggives|\bgloan|\bmaya credit|easy credit|\bsalary loan|\batome\b|\bcashbee|\bpesoloan|cash2go|"
        r"juanhand|pondopeso|crezu|cashify|flexi ?fund|instant approval|repayment|cash advance|(quick|fast|instant|easy|immediate) cash|"
        r"cash ?(aid|assistance) .*\binterest|\binterest\b.*\bapply|\bapply\b.*\binterest|credit line|\bguarantor|ma-?disburse"
    ),
    "lending_weak": (
        r"\binterest\b|\bcredit limit|\bhulog|\bamortiz|\binstallment|\bpay later\b|need cash|credit score|"
        r"\bcash\b.*\bapply\b|\bapply\b.*\bcash\b|pass rate|disburs|emergency (funds|cash)|great credit|extra funds|need funds|\bfunds?\b.*\bapply"
    ),
    "banking": (
        r"\bbank\b|\bbanking\b|\bbdo|\bbpi\b|metrobank|unionbank|\bubp\b|landbank|\blbp|security ?bank|\brcbc\b|\bpnb\b|china ?bank|east ?west|\bpbcom|\bseabank|"
        r"\bgotyme|\btonik\b|\bcimb\b|\bmaribank|\batm\b|savings account|checking account|\bdebit|\bcredit card|\bcard (blocked|number|ending)|\baccount (number|ending|no)|"
        r"\bonline banking|\bbsp\b|\bbancnet|\binstapay|\bpesonet|\bkyc\b|\bewb\b|banco de oro|citibank|\bciti\b|\bhsbc\b|credit limit increase|\bnew card\b|card upgrade|"
        r"\bcredit cards?\b.*\binstallment|\binstallment\b.*\bcredit cards?\b|\bbalance transfer\b"
    ),
    "banking_generic": (       # only counts when no telco is named (a TNT or Globe "account" is not a bank account)
        r"\b(account|acct|a/c)\b.*\b(locked|suspend|block|verif|deactivat|restrict|validat|limited|compromis)|"
        r"\b(locked|suspend|block|verif|deactivat|restrict|validat|compromis)\w*\b.*\b(account|acct|a/c)\b"
    ),
    "e-wallet": (
        r"\bgcash\b|g-cash|\bg cash\b|pay ?maya|\bmaya\b|coins\.ph|\bgrab ?pay|\bshopee ?pay|\bgoogle ?pay|\bapple ?pay|\bpaypal|\bwallet\b|\bcash ?in\b|\bqr ?ph|\b(new|total|your|current) balance\b|balance update|successfully credited|\bcash alert\b|express send|g-c-?ash|\bcoins (alert|support|wallet|account)"
    ),
    "payments_billing": (
        r"\bbills?\b|\bbilling\b|\bpayments?\b|\bpaid\b|\bpay(ing)? (now|your|the|for)|\binvoice|\brefund|\boverdue|\bdue date|\bunpaid|\bfines?\b|\bpenalt|\bmmda\b|"
        r"\btoll\b|\bnotice of violation|\btraffic|\bregistration fee|\bprocessing fee|\bdelivery fee|\bshipping fee|\bredelivery|\bcustoms|\bremit|\bpadala|\bmeralco|"
        r"\bwater bill|\bpldt|\bconverge|\breceipt|\bcontribution|\bref(erence)?\.? ?(no|#)|\btransaction|\btransfer(red)?\b|\bbir\b|\btax\b|"
        r"\bebill|\bbayad|\bbayaran|\bamount due|\b(unpaid|overdue|fee|fine)\b.*\b(php|p|₱) ?\d|\b(php|p|₱) ?\d[\d,.]*.*\b(unpaid|overdue|fee|fine)\b"
    ),
}
_COMPILED = {k: re.compile(v, re.I) for k, v in RULES.items()}
_TELCO = re.compile(r"\b(tnt|smart|globe|sun|gomo|gigalife|giga|gigamania|talk ?n ?text)\b|\*123#", re.I)
_DITO = re.compile(r"\bDITO\b")   # the telco, in capitals; "dito" in lower case is Tagalog for "here"
_NONFIN = re.compile(r"google|gmail|microsoft|outlook|office ?365|whatsapp|facebook|\bfb\b|instagram|tiktok|\bgrab\b|shopee|lazada|lazmall|zalora|amazon|apple|icloud|netflix|spotify|fedex|\bdhl\b|j&t|lbc|\bsf express|seller|driver|linkedin|puregold|robinsons|\bmalls?\b|go ?rewards|kumu|genshin|hoyo", re.I)
_MONEY = re.compile(r"(php|\bp|₱) ?\d|\bfunds?\b|\bmoney\b|\bpera\b|\bbalance\b|\bcards?\b|\bsavings\b|\bdeposit", re.I)   # generic "account locked": banking only with money words
_STRONG = ["gambling", "investment_crypto", "lending"]       # these win over a bank or wallet that is only named as the way to pay
_TIE = ["banking", "e-wallet", "payments_billing"]
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s"})


def _unleet(text: str) -> str:
    """'SP1NS', 'b0nus', 'W!NS' -> 'spins', 'bonus', 'wins': only inside words that already contain two or more letters."""
    def fix(m: "re.Match[str]") -> str:
        w = m.group(0)
        return w.translate(_LEET).replace("!", "i") if sum(c.isalpha() for c in w) >= 2 else w
    return re.sub(r"[A-Za-z0-9@$!]+", fix, text)


def categorize(text: str) -> Tuple[str, str]:
    plain = _unleet(text)
    hits: Dict[str, int] = {k: max(len(rx.findall(text)), len(rx.findall(plain))) for k, rx in _COMPILED.items()}
    generic = hits.pop("banking_generic")
    weak = {k: hits.pop(k + "_weak") for k in _STRONG}
    # weak words count once each (the same word twice is still one word)
    weak = {k: len({m.group(0).lower() for t in (text, plain) for m in _COMPILED[k + "_weak"].finditer(t)}) for k in _STRONG}
    brand_or_telco = bool(_NONFIN.search(text) or _TELCO.search(text) or _DITO.search(text))
    if not brand_or_telco and not hits["e-wallet"] and _MONEY.search(text):
        hits["banking"] += generic         # a wallet brand named in the message decides it, not the generic "account locked" wording
    strong = {k: hits[k] for k in _STRONG}
    if brand_or_telco:
        weak["gambling"] = 0               # "spin and win" at Shopee, Globe "bonus data", Smart "P0 cashout" are not gambling
    for k in _STRONG:
        if strong[k]:
            hits[k] += weak[k]             # weak words support a strong one
    # weak only: gambling needs two weak words; investment / lending need two, or one when no other category matched
    tie_or_strong = any(hits[k] for k in _TIE) or any(strong.values())
    # with a bank / wallet / bill already matched, three weak gambling words are needed (a bank's "cash rebate" is not gambling)
    needed = 3 if any(hits[k] for k in _TIE) else 2
    if not strong["gambling"] and weak["gambling"] >= needed and not strong["investment_crypto"] and not strong["lending"]:
        hits["gambling"] = weak["gambling"]
        return "gambling", "|".join(k for k in CATEGORIES if hits[k])
    for k in ("investment_crypto", "lending"):
        if not strong[k] and (weak[k] >= 2 or (weak[k] == 1 and not tie_or_strong)):
            hits[k] = weak[k]
    matched: List[str] = [k for k in CATEGORIES if hits[k]]
    for k in _STRONG:
        if strong[k]:
            return k, "|".join(matched)
    for k in _STRONG:
        if hits[k]:
            return k, "|".join(matched)
    rest = [k for k in _TIE if hits[k]]
    if not rest:
        return "none", ""
    best = max(rest, key=lambda k: (hits[k], -_TIE.index(k)))
    return best, "|".join(matched)


def categorize_all(texts: List[str], decisions_path=None) -> Tuple[List[Tuple[str, str]], int]:
    """categorize() for every text, then the hand-checked corrections of `decisions_path` (columns sms_text, final_category, reason;
    keyed by the exact message text). Returns ([(primary, all_matched or "none")...], number of texts set by hand). A correction whose
    text is not among `texts` stops the build, so a stale file cannot pass silently."""
    import pandas as pd
    out = [categorize(t) for t in texts]
    out = [(p, a or "none") for p, a in out]
    if not decisions_path or not Path(decisions_path).exists():
        return out, 0
    dec = pd.read_csv(decisions_path, keep_default_na=False, dtype=str)
    unknown = sorted(set(dec.final_category) - set(CATEGORIES) - {"none"})
    if unknown or dec.sms_text.duplicated().any():
        raise ValueError(f"category_decisions: unknown categories {unknown} or a text listed twice")
    by_text = dict(zip(dec.sms_text, dec.final_category))
    missing = set(by_text) - set(texts)
    if missing:
        raise ValueError(f"category_decisions: {len(missing)} texts are not in the dataset, e.g. {next(iter(missing))[:80]!r}")
    n = 0
    for i, t in enumerate(texts):
        if t in by_text:
            final = by_text[t]
            matched = [] if out[i][1] == "none" else out[i][1].split("|")
            merged = [k for k in CATEGORIES if k in matched or k == final]
            out[i] = (final, "|".join(merged) if final != "none" else "none")
            n += 1
    return out, n
