"""Add synthetic LEGITIMATE messages with links so the SMS classes are the same size.

Why: in v2_2026-09-15.csv only 6.1% of legitimate messages contain a link (or the <URL>
placeholder) against 67.7% of smishing messages. The CNN sees every link as the same <URL>
token, so it can learn "has a link => smishing". Adding legitimate messages with links
until both classes have 10,955 rows raises the legitimate link share to about 32%.

The texts come from the 46 legitimate templates already written for SmishGuard
(src/data/build_v2_with_synthetic.py): social shares, work/school links, delivery, bill
and advisory notices, in English, Tagalog and Taglish. Rows are spread evenly over the
templates; each row keeps its template_id so splits can be grouped by template later
(synthetic test rows should come from templates the model never trained on).

Usage (from the project root):
    python -m src.data.add_synthetic_legit_links
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from src.data.build_v2_with_synthetic import CITIES, MONTHS, TEMPLATES as BASE_TEMPLATES
from src.data.build_v2_with_synthetic import _amount, _date, _rid

# The SmishGuard versions of these four templates pick the brand and the website
# independently ("UnionBank: ... Learn more at securitybank.com"). A brand that does not
# match its domain is a smishing sign, so these versions always keep the pair together.
BANKS = [("BPI", "bpi.com.ph"), ("BDO", "bdo.com.ph"), ("Metrobank", "metrobank.com.ph"),
         ("UnionBank", "unionbankph.com"), ("Security Bank", "securitybank.com")]
TELCOS = [("Globe", "globe.com.ph"), ("PLDT Home", "pldthome.com"), ("Smart", "smart.com.ph"),
          ("Converge", "convergeict.com")]
SUBSCRIPTIONS = [("Netflix", "netflix.com"), ("Spotify", "spotify.com"),
                 ("YouTube Premium", "youtube.com"), ("Disney+", "disneyplus.com")]
AIRLINES = [("Cebu Pacific", "cebupacificair.com"), ("Philippine Airlines", "philippineairlines.com"),
            ("AirAsia", "airasia.com")]


def _pair(rng, pairs):
    return rng.choice(pairs)


MATCHED_TEMPLATES = {
    "bank_statement_info": lambda r: (lambda b: f"{b[0]}: Your {r.choice(MONTHS)} credit card statement is ready. You may view it via the mobile app. Learn more at {b[1]}")(_pair(r, BANKS)),
    "telco_bill": lambda r: (lambda t: f"{t[0]}: Your statement for {r.choice(MONTHS)} is ready. Amount due P{_amount(r, 999, 3500)}. View at {t[1]}")(_pair(r, TELCOS)),
    "netflix_receipt": lambda r: (lambda s: f"{s[0]}: We received your payment of P{_amount(r, 149, 549)}. See billing details at {s[1]}")(_pair(r, SUBSCRIPTIONS)),
    "flight_confirmed": lambda r: (lambda a, c: f"{a[0]}: Your flight {c[0]}-{c[1]} on {_date(r)} is confirmed. Booking ref {_rid(r, 6).upper()}. Manage at {a[1]}")(_pair(r, AIRLINES), r.choice(CITIES)),
}
TEMPLATES = {**BASE_TEMPLATES, **MATCHED_TEMPLATES}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "data" / "raw" / "sms" / "v2_2026-09-15.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "sms_v2_balanced.csv"

REAL_SOURCE = "v2_2026-09-15"
SYNTHETIC_SOURCE = "synthetic_legit_url"
MAX_ATTEMPTS_PER_ROW = 50


def generate(n_rows: int, existing: set[str], seed: int) -> pd.DataFrame:
    # Round-robin over the templates, one new unique text per template per round, so every
    # template gets the same share. A template that runs out of unique texts (some have
    # small fill pools) drops out and the others cover the rest.
    rng = random.Random(seed)
    seen = set(existing)
    active = list(TEMPLATES)
    rows = []
    while len(rows) < n_rows and active:
        for template_id in list(active):
            if len(rows) == n_rows:
                break
            for _ in range(MAX_ATTEMPTS_PER_ROW):
                text = TEMPLATES[template_id](rng)
                if text not in seen:
                    seen.add(text)
                    rows.append({"sms": text, "label": "legitimate",
                                 "source": SYNTHETIC_SOURCE, "template_id": template_id})
                    break
            else:
                active.remove(template_id)
    if len(rows) < n_rows:
        raise RuntimeError(f"only {len(rows)} unique texts possible, {n_rows} requested")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    real = pd.read_csv(args.input).assign(source=REAL_SOURCE, template_id="")
    counts = real["label"].value_counts()
    n_add = int(counts["smishing"] - counts["legitimate"])

    synthetic = generate(n_add, set(real["sms"].astype(str)), args.seed)
    out = pd.concat([real, synthetic], ignore_index=True)
    assert not out["sms"].duplicated().any()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Added {len(synthetic)} synthetic legitimate messages with links "
          f"({synthetic['template_id'].nunique()} templates) -> {args.output}")
    print("Rows per class:", out["label"].value_counts().to_dict())
    per_template = synthetic["template_id"].value_counts()
    print(f"Rows per template: {per_template.min()} to {per_template.max()}")


if __name__ == "__main__":
    main()
