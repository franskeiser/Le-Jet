"""Turn data/processed/feature_dataset_lean.csv into a 37-feature table with SYNTHETIC SMS text (placeholder fresh set).

1. Features: keeps the final 37 (src/unified/prepare_web_37.FEATURES_37). Drops avg_words_raw, longest_word_path and
   shortest_words_raw, and adds shortest_word_path, computed from the URL path + query (matches the raw web set on 98.2%
   of rows). Negative domain_age / domain_registration_length become -1, as in prepare_web_37.
2. Text: each row gets a templated financial SMS that contains its own link. Per class, `--neutral-rate` of the messages
   use wording shared by both classes, so the text alone does not decide the label. Language mix: English / Taglish.

The text is invented. Scores on this file are a pipeline smoke test, NOT study results; replace `sms_text` with the real
SMS when the group has it (docs/progress.md, section 6, first decision).

Writes data/processed/fresh_lean_synthetic.csv (message_id, sms_text, label, source, link, link_status, link_domain,
37 features).

Usage:
    python -m src.unified.prepare_lean_synthetic [--neutral-rate 0.25] [--seed 42]
"""
import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract

from src.unified.prepare_web_37 import FEATURES_37, PAGE_FEATURES

ROOT = Path(__file__).resolve().parents[2]
LEAN = ROOT / "data" / "processed" / "feature_dataset_lean.csv"
OUT = ROOT / "data" / "processed" / "fresh_lean_synthetic.csv"

BRANDS = ["GCash", "Maya", "BDO", "BPI", "Metrobank", "UnionBank", "Landbank", "Security Bank", "GoTyme", "Coins.ph",
          "PayMaya", "Home Credit", "Atome", "SeaBank"]

# Each template: {b} = brand, {u} = link, {n} = amount/number.
PHISH = [
    "{b}: Your account has been suspended due to unusual activity. Verify now to avoid permanent closure: {u}",
    "{b} ALERT: P{n} was sent from your account. If this was not you, secure your account here {u}",
    "Dear {b} customer, your account will be locked in 24 hours. Confirm your details at {u}",
    "Congratulations! You won P{n} from {b} promo. Claim your reward today: {u}",
    "{b}: Na-detect ang suspicious login sa account mo. I-verify agad para hindi ma-deactivate: {u}",
    "Pre-approved ka sa loan na P{n} sa {b}! Mag-apply na, limited slots lang: {u}",
    "{b}: Your OTP was requested from a new device. Cancel the request at {u}",
    "Your {b} wallet is limited. Update your KYC within 12 hours to restore access {u}",
    "FINAL NOTICE from {b}: unpaid balance P{n}. Settle now to avoid legal action {u}",
]
LEGIT = [
    "{b}: You received P{n}. View your transaction details in the app or at {u}",
    "{b}: Your monthly statement is ready. You can view it at {u}",
    "{b}: Thank you for your payment of P{n}. Receipt available at {u}",
    "{b} reminder: your bill of P{n} is due in 3 days. Pay through the app or {u}",
    "{b}: Nag-login ka sa bagong device. Kung ikaw ito, wala ka nang gagawin. Para sa tulong: {u}",
    "{b}: We will never ask for your OTP or password. Learn more about staying safe at {u}",
    "{b} advisory: scheduled maintenance tonight 12AM-3AM. Details: {u}",
    "URGENT from {b}: a P{n} charge on your card needs your attention. Review it at {u}",
    "{b}: Salamat sa pag-apply. Ang status ng loan application mo ay makikita sa {u}",
]
# Wording used for both classes.
NEUTRAL = [
    "{b}: Please review your account activity here {u}",
    "{b}: Update available for your account. See details {u}",
    "Hi, this is {b}. Kindly check your transaction of P{n}: {u}",
    "{b}: Mag-check ng account details mo dito {u}",
    "{b} notice: action may be needed on your account. {u}",
]


def shortest_word_path(url: str) -> int:
    p = urlparse(url)
    words = [w for w in re.split(r"[^A-Za-z0-9]+", p.path + (" " + p.query if p.query else "")) if w]
    return min(map(len, words)) if words else 0


def brand_for(url: str, rng: np.random.Generator) -> str:
    """Use the link's own brand when its domain names one, else a random one."""
    host = urlparse(url if "//" in url else "http://" + url).netloc.lower()
    for b in BRANDS:
        if re.sub(r"[^a-z]", "", b.lower()) in re.sub(r"[^a-z]", "", host):
            return b
    return str(rng.choice(BRANDS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral-rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = pd.read_csv(LEAN)
    d["shortest_word_path"] = d["url"].map(shortest_word_path)
    for col in ("domain_age", "domain_registration_length"):
        d.loc[d[col] < 0, col] = -1
    d["label"] = d["status"].map({"phishing": "smishing", "legitimate": "legitimate"})
    assert d["label"].notna().all()

    texts = []
    for url, label in zip(d["url"], d["label"]):
        pool = NEUTRAL if rng.random() < a.neutral_rate else (PHISH if label == "smishing" else LEGIT)
        tpl = str(rng.choice(pool))
        texts.append(tpl.format(b=brand_for(url, rng), u=url, n=f"{int(rng.integers(2, 80)) * 100:,}"))

    ext = d["url"].map(lambda u: tldextract.extract(u))
    out = pd.DataFrame({
        "message_id": [f"lean_{i}" for i in range(len(d))],
        "sms_text": texts,
        "label": d["label"],
        "source": "synthetic",
        "link": d["url"],
        "link_status": "link",
        "link_domain": [e.registered_domain or e.domain for e in ext],
    })
    out = pd.concat([out, d[FEATURES_37]], axis=1)
    out.to_csv(OUT, index=False)
    print(f"{OUT}: {len(out)} rows, {len(FEATURES_37)} features ({len(PAGE_FEATURES)} page/outside), "
          f"{out['link_domain'].nunique()} domains, {out['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
