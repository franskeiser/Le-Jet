r"""The six financial subcategories of the study, assigned to a message by word and brand rules.

The six (Chapter 3 inclusion criteria): banking, e-wallet, lending, payments_billing, investment_crypto, gambling.
`categorize(text)` returns (primary, all_matched):
  * primary: the ONE category the message is mainly about; "none" if no rule matches (many legitimate chit-chat or telco-promo messages,
    and smishing that is not financial, such as Microsoft / Google account lures, job offers or parcel notices);
  * all_matched: every category a rule matched, "|"-joined (a message can touch several: the criteria say "at least one").
Primary is chosen so that a message that only MENTIONS a bank or wallet as a way to pay is not counted as banking / e-wallet:
gambling, then investment_crypto, then lending win when they match; otherwise the category with most hits (ties: banking, e-wallet, payments_billing).

THIS IS A DRAFT ASSIGNMENT BY RULES, not a human annotation: check a sample before using the numbers in the thesis.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

CATEGORIES = ["banking", "e-wallet", "lending", "payments_billing", "investment_crypto", "gambling"]

RULES: Dict[str, str] = {
    "gambling": (
        r"casino|\bslots?\b|jackpot|sabong|bingo|\bjili|color ?game|\bspins?\b|scatter|pagcor|\bbets\b|\bbetting\b|\bbet\d|\w*bet(ph)?\d*\.(com|ph|net|cc|vip|xyz)|\bubet\b|"
        r"lotto|lottery|poker|baccarat|roulette|pusoy|tongits|sportsbook|wager|rebate|super ?ace|fortune ?gems|\bpogo\b|arena ?plus|applus|nustar|okbet|"
        r"\bplayzone|lucky ?(game|wheel|draw|spin)|free (bonus|credit|chips)|deposit bonus|welcome bonus|first deposit|"
        r"\bgame(s)? (bonus|reward)|\bonline game|sugal|prize ?pool|game ?room|\bcash ?out\b|\bw\+|\bwin ?\d{3,}|\bcc ?(bet|games?)\b|play now|first play|"
        r"\bbounty\b|\bplay (and|&) win|\bgame ?(app|site)\b|lucky ?ph|\bph ?(win|bet|\d{2,})\b|paldo|\btalo\b|nilalaro|papremyo|\b\d{2,}x\b|payout|"
        r"mystery prize|rescue cash|withdraw|winhub|winning\+|big win|table game|red envelope|play loads|recharge"
    ),
    "gambling_bonus": r"\bbonus(es)?\b|\d+ ?bonus",       # counts as gambling only when no shop or telco is named
    "investment_crypto": (
        r"crypto|bitcoin|\bbtc\b|\busdt\b|ethereum|\beth\b|binance|\bnft\b|blockchain|\bforex\b|\btrading\b|\btrader\b|invest|stock market|stocks? (trading|investing)|mining|\bdividend|"
        r"passive income|profit|\bmutual fund|\bipo\b|airdrop|\broi\b|capital|\bbitget|\bbybit|\bokx\b|\bcoinbase|\bgcrypto|\bhigh return|\bdouble your|\bearn \$|\busd\b|\$ ?\d{3,}"
    ),
    "lending": (
        r"\bloans?\b|\butang\b|pautang|\bcredit limit|\bcash ?loan|online lending|\bborrow|\blend(er|ing)?\b|\binterest\b|\bhome ?credit|\bbillease|\bkviku|\btala\b|"
        r"\bdigido|\bcashalo|\bspay ?later|\bgcredit|\bggives|\bcredit line|\bhulog|\bamortiz|\binstallment|\bpay later\b|\bmaya credit|\bsalary loan|\batome\b|"
        r"\bcashbee|\bpesoloan|need cash|credit score|cash advance|(quick|fast|instant|easy) cash|\bcash\b.*\bapply\b|\bapply\b.*\bcash\b|pass rate|cash2go|juanhand|disburs|instant approval|repayment|emergency (funds|cash)|great credit"
    ),
    "banking": (
        r"\bbank\b|\bbanking\b|\bbdo\b|\bbpi\b|metrobank|unionbank|\bubp\b|landbank|\blbp|security ?bank|\brcbc\b|\bpnb\b|china ?bank|east ?west|\bpbcom|\bseabank|"
        r"\bgotyme|\btonik\b|\bcimb\b|\bmaribank|\batm\b|savings account|checking account|\bdebit|\bcredit card|\bcard (blocked|number|ending)|\baccount (number|ending|no)|"
        r"\bonline banking|\bbsp\b|\bbancnet|\binstapay|\bpesonet|\bkyc\b"
    ),
    "banking_generic": (       # only counts when no telco is named (a TNT or Globe "account" is not a bank account)
        r"\b(account|acct|a/c)\b.*\b(locked|suspend|block|verif|deactivat|restrict|validat|limited|compromis)|"
        r"\b(locked|suspend|block|verif|deactivat|restrict|validat|compromis)\w*\b.*\b(account|acct|a/c)\b"
    ),
    "e-wallet": (
        r"\bgcash\b|g-cash|\bg cash\b|pay ?maya|\bmaya\b|coins\.ph|\bgrab ?pay|\bshopee ?pay|\bgoogle ?pay|\bapple ?pay|\bpaypal|\bwallet\b|\bcash ?in\b|\bqr ?ph|\b(new|total|your|current) balance\b|balance update|successfully credited|\bcash alert\b"
    ),
    "payments_billing": (
        r"\bbills?\b|\bbilling\b|\bpayments?\b|\bpaid\b|\bpay(ing)? (now|your|the|for)|\binvoice|\brefund|\boverdue|\bdue date|\bunpaid|\bfines?\b|\bpenalt|\bmmda\b|"
        r"\btoll\b|\bnotice of violation|\btraffic|\bregistration fee|\bprocessing fee|\bdelivery fee|\bshipping fee|\bredelivery|\bcustoms|\bremit|\bpadala|\bmeralco|"
        r"\bwater bill|\bpldt|\bconverge|\breceipt|\bcontribution|\bref(erence)?\.? ?(no|#)|\btransaction|\btransfer(red)?\b|\bbir\b|\btax\b|"
        r"\bebill|\bbayad|\bbayaran|\bamount due|\b(unpaid|overdue|fee|fine)\b.*\b(php|p|₱) ?\d|\b(php|p|₱) ?\d[\d,.]*.*\b(unpaid|overdue|fee|fine)\b"
    ),
}
_COMPILED = {k: re.compile(v, re.I) for k, v in RULES.items()}
_TELCO = re.compile(r"\b(tnt|smart|globe|sun|dito|gomo|gigalife|giga|talk ?n ?text)\b|\*123#", re.I)
_NONFIN = re.compile(r"google|gmail|microsoft|outlook|office ?365|whatsapp|facebook|\bfb\b|instagram|tiktok|\bgrab\b|shopee|lazada|lazmall|zalora|amazon|apple|icloud|netflix|spotify|fedex|\bdhl\b|j&t|lbc|\bsf express|seller|driver", re.I)
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
    bonus = hits.pop("gambling_bonus")
    if not _NONFIN.search(text) and not _TELCO.search(text):
        hits["gambling"] += bonus
    if not _TELCO.search(text) and not _NONFIN.search(text) and not hits["e-wallet"]:
        hits["banking"] += generic         # a wallet brand named in the message decides it, not the generic "account locked" wording
    matched: List[str] = [k for k in CATEGORIES if hits[k]]
    for k in _STRONG:
        if hits[k]:
            return k, "|".join(matched)
    rest = [k for k in _TIE if hits[k]]
    if not rest:
        return "none", ""
    best = max(rest, key=lambda k: (hits[k], -_TIE.index(k)))
    return best, "|".join(matched)
