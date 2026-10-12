"""Synthetic demo data: real URLs from the public web set wrapped in made-up SMS texts.

PROTOTYPE ONLY. Everything here exists so the defense tool has something to train on. The
texts are invented and the class balance, the share of ambiguous wording and the share of dead
pages are my own choices, so no score measured on this data is a study result.

What is real: the URLs and their page, domain and WHOIS features (public web phishing set,
Hannousse & Yahiouche 2021). What is made up: the SMS text around each URL, the "no link"
messages, and which pages count as unreachable.

Leak guard: the web set is split by domain. One part trains the demo RF extractors; the other
part (the "pool") supplies the URLs for the SMS rows. The extractors therefore never saw the
URLs they score, as will be true for live links.

Ambiguous wording: about half of the linked messages use neutral wording ("Please check this:
<link>") that both classes share, so the text alone cannot decide them and the link evidence
has to. That is a design choice that lets the demo show the mechanism. It says nothing about how
often such messages occur in practice.

Usage (from the project root):
    python -m src.demo.synthetic
"""
import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.features.link_features import LINK_TEXT_FEATURES, extract_links, link_text_features, no_link_features, normalize_link
from src.features.model.rf_extractor import LABEL_TO_IDX, VARIANTS, fit_rf, load_dataset

ROOT = Path(__file__).resolve().parents[2]
DEMO_DATA = ROOT / "data" / "demo" / "synthetic_demo.csv"
DEMO_MODEL_DIR = ROOT / "models" / "demo"
EXTRACTOR_INPUTS = VARIANTS["full"]
PAGE_COLUMNS = EXTRACTOR_INPUTS[len(LINK_TEXT_FEATURES):]  # page_rank, 12 webpage, 4 external
SEED = 42
EXTRACTOR_PARAMS = {"n_estimators": 200, "min_samples_leaf": 2, "max_features": "sqrt"}

SMISH_LINK = [
    "GCash: Your account has been limited. Verify your identity within 24 hours to avoid permanent suspension: {url}",
    "BDO ALERT: Unusual login detected on your online banking. Secure your account now {url}",
    "Maya: You received P{amt}. Claim it here before it expires {url}",
    "LBC: Your parcel could not be delivered. Pay the P{small} redelivery fee here {url}",
    "SSS: Your pension benefits are on hold. Update your records today {url}",
    "Congratulations! You won P{amt} in the Globe promo. Claim your prize {url}",
    "BPI: Your card was used for a P{amt} purchase. If this wasn't you, cancel here {url}",
    "Shopee: Your order is on hold because of an address problem. Confirm details {url}",
    "Kumpirmahin ang iyong account para hindi ma-deactivate. Mag-click dito: {url}",
    "PhilHealth: Your contribution is overdue. Settle now to avoid penalties {url}",
]
LEGIT_LINK = [
    "Reminder: your appointment is tomorrow at {time}. Details: {url}",
    "Hi! Here is the article I mentioned: {url}",
    "Meeting notes from today are posted here {url}",
    "Salamat sa pagbili! Narito ang resibo mo: {url}",
    "Class schedule update posted: {url}",
    "Check out this recipe, you'll love it {url}",
    "Your {month} bill is ready. You can view it here: {url}",
    "Here's the link to the photos from the weekend {url}",
    "Hey, sharing the project folder with you: {url}",
    "Mag-register na for the seminar on Saturday: {url}",
]
AMBIGUOUS_LINK = [
    "Hi, please check this when you can: {url}",
    "Your package update: {url}",
    "Important notice about your account: {url}",
    "Reminder: please confirm your details {url}",
    "See details here {url}",
    "Payment reminder, please review: {url}",
    "Action required: review your request {url}",
    "Pakitingnan ito: {url}",
    "Your {month} statement is ready {url}",
    "Update regarding your order: {url}",
    "Please see the attached notice {url}",
    "FYI {url}",
]
SMISH_NOLINK = [
    "Congratulations! You won P{amt}. Reply YES to claim your prize.",
    "Your GCash account will be closed today. Call 09{n9} now to keep it active.",
    "URGENT: your BDO card is blocked. Text BLOCK to {code5} to unblock.",
    "You have been selected for a P{amt} cash grant. Send your full name and birthday to 09{n9}.",
]
LEGIT_NOLINK = [
    "Your OTP is {otp}. Do not share it with anyone.",
    "Hi! Are we still on for lunch tomorrow?",
    "Reminder: your appointment is tomorrow at {time}.",
    "Salamat po! Natanggap na namin ang bayad mo.",
    "Pls buy rice on your way home, thanks!",
    "Your table for 4 is confirmed for tonight at {time}.",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def fill(template: str, rng: random.Random, url: str = "") -> str:
    values = {
        "{url}": url, "{amt}": f"{rng.randrange(1, 90) * 500:,}", "{small}": str(rng.choice([25, 35, 49, 99])),
        "{month}": rng.choice(MONTHS), "{n9}": "".join(rng.choices("0123456789", k=9)),
        "{otp}": "".join(rng.choices("0123456789", k=6)), "{code5}": "".join(rng.choices("0123456789", k=5)),
        "{time}": rng.choice(["9:00 AM", "10:30 AM", "1:00 PM", "3:30 PM", "6:00 PM"]),
    }
    for key, value in values.items():
        template = template.replace(key, value)
    return template


def url_matches_text(text: str, url: str) -> bool:
    """The serving code finds the link in the text on its own; keep only rows where it finds
    exactly the URL we wrapped, so training and serving see the same link features."""
    links = extract_links(text)
    return bool(links) and normalize_link(links[0]["raw"]) == normalize_link(url)


def train_extractors(web_train: pd.DataFrame) -> dict[str, dict]:
    bundles = {}
    for variant, features in VARIANTS.items():
        rf = fit_rf(web_train[features], web_train["label"].to_numpy(), SEED, **EXTRACTOR_PARAMS)
        bundles[variant] = {"model": rf, "features": features, "variant": variant, "seed": SEED,
                            "params": EXTRACTOR_PARAMS, "label_to_idx": LABEL_TO_IDX}
    return bundles


def linked_row(rng: random.Random, web_row: pd.Series, kind_roll: float, fetched: bool) -> dict | None:
    url = web_row["url"]
    smishing = web_row["status"] == "phishing"
    if kind_roll < 0.5:
        kind, pool = "class_specific", SMISH_LINK if smishing else LEGIT_LINK
    else:
        kind, pool = "ambiguous", AMBIGUOUS_LINK
    text = fill(rng.choice(pool), rng, url)
    if not url_matches_text(text, url):
        return None
    row = {"sms_text": text, "label": "smishing" if smishing else "legitimate", "link": url,
           "text_kind": kind, "has_url": 1, "page_fetched": int(fetched), "domain": web_row["domain"],
           **link_text_features(normalize_link(url))}
    for column in PAGE_COLUMNS:
        row[column] = web_row[column] if fetched else -1
    return row


def no_link_row(rng: random.Random, smishing: bool, index: int) -> dict:
    text = fill(rng.choice(SMISH_NOLINK if smishing else LEGIT_NOLINK), rng)
    row = {"sms_text": text, "label": "smishing" if smishing else "legitimate", "link": "",
           "text_kind": "no_link", "has_url": 0, "page_fetched": 0, "domain": f"nolink:{index}",
           **no_link_features()}
    row.update({column: -1 for column in PAGE_COLUMNS})
    return row


def assign_splits(df: pd.DataFrame, seed: int) -> pd.Series:
    """train 70% / rf_fit 10% / test 20%, grouped by domain so a domain is never in two parts."""
    groups = df["domain"].to_numpy()
    rest, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(df, groups=groups))
    train_rel, rf_rel = next(GroupShuffleSplit(1, test_size=0.125, random_state=seed)
                             .split(rest, groups=groups[rest]))
    split = np.empty(len(df), dtype=object)
    split[test], split[rest[train_rel]], split[rest[rf_rel]] = "test", "train", "rf_fit"
    return pd.Series(split, index=df.index)


def build(out_csv: Path = DEMO_DATA, model_dir: Path = DEMO_MODEL_DIR, seed: int = SEED,
          no_link_share: float = 0.4, page_fetched_share: float = 0.7) -> pd.DataFrame:
    rng = random.Random(seed)
    web = load_dataset()
    ext_idx, pool_idx = next(GroupShuffleSplit(1, test_size=0.45, random_state=seed)
                             .split(web, groups=web["domain"].to_numpy()))
    bundles = train_extractors(web.iloc[ext_idx])
    (model_dir / "extractors").mkdir(parents=True, exist_ok=True)
    for variant, bundle in bundles.items():
        joblib.dump(bundle, model_dir / "extractors" / f"{variant}.joblib")

    rows = []
    for _, web_row in web.iloc[pool_idx].iterrows():
        row = linked_row(rng, web_row, rng.random(), rng.random() < page_fetched_share)
        if row:
            rows.append(row)
    n_no_link = int(len(rows) * no_link_share)
    rows += [no_link_row(rng, i % 2 == 0, i) for i in range(n_no_link)]

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "message_id", [f"demo_{i:05d}" for i in range(len(df))])
    df["split"] = assign_splits(df, seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEMO_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEMO_MODEL_DIR)
    args = parser.parse_args()
    df = build(args.out, args.model_dir)
    print(f"{len(df)} synthetic messages -> {args.out}")
    print(df.groupby(["split", "label"]).size().unstack())
    print(df.groupby(["text_kind", "label"]).size().unstack())


if __name__ == "__main__":
    main()
