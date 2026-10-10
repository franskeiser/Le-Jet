# Add keyword-based financial candidate tags to SMS data.

# The tags produced here are suggestions for manual review. They are not final
# labels and do not determine whether a message should be kept or excluded.

# Just for type annotations
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

# For the file path 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "standardized" / "master_standardized.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "standardized"
    / "master_candidate_tagged.csv"
)
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_RANDOM_SEED = 42


# Keep this vocabulary easy to inspect and revise after manual review.
# A keyword may appear in more than one category when the domains overlap.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "banking": [
        "bank",
        "banking",
        "bank account",
        "savings account",
        "checking account",
        "online banking",
        "mobile banking",
        "account balance",
        "account suspended",
        "account locked",
        "debit card",
        "credit card",
        "atm",
        "otp",
        "one-time password",
        "bank pin",
        "atm pin",
        "card pin",
        "bank transfer",
        "funds transfer",
        "withdrawal",
        "beneficiary",
        "unauthorized transaction",
        "bdo",
        "bpi",
        "metrobank",
        "unionbank",
        "security bank",
        "chinabank",
        "rcbc",
        "landbank",
        "pnb",
        "hsbc",
        "citibank",
        "chase",
        "wells fargo",
        "bank of america",
        "bmo",
        "online account",
        "account statement",
        "sbi",
        "yono",
        "hdfc",
        "navy federal",
        "navyfed",
        "boa",
        "bofa",
        "firstbank",
        "credit union",
        "netbanking",
        "caixabank",
        "santander",
        "bancolombia",
        "credit agricole",
        "prepaid card",
        "withdrawn",
        "savings interest",
        "interest earned",
        "new recipient",
        "suspicious transaction",
        "account disabled",
        "account blocked",
        "account restricted",
        "debit on hold",
        "commbank",
        "revolut",
        "kotak",
        "lloyds",
        "fnb",
        "new payee",
        "account credited",
        "credited to account",
        "blocked card",
        "card blocked",
        "account deactivated",
        "account verification",
        "anz",
        "westpac",
        "ocbc",
        "scotiabank",
        "dbs",
        "barclays",
        "natwest",
        "capital one",
        "wire transfer",
    ],
    "e_wallet": [
        "gcash",
        "maya",
        "paymaya",
        "paypal",
        "payoneer",
        "venmo",
        "cash app",
        "skrill",
        "alipay",
        "wechat pay",
        "m-pesa",
        "apple pay",
        "google pay",
        "samsung pay",
        "digital wallet",
        "mobile wallet",
        "e-wallet",
        "ewallet",
        "cash in",
        "cash out",
        "wallet balance",
        "wallet account",
        "wallet verification",
        "wallet transaction",
        "mobile load",
        "prepaid load",
        "paytm",
        "paytm kyc",
        "wallet kyc",
        "cashback in your wallet",
        "wallet suspended",
        "wallet blocked",
        "wallet on hold",
        ],
    "lending": [
        "loan",
        "loans",
        "lending",
        "lender",
        "borrower",
        "borrow money",
        "borrow load",
        "loan application",
        "loan approval",
        "approved loan",
        "loan offer",
        "loan balance",
        "instant loan",
        "cash loan",
        "personal loan",
        "payday loan",
        "online loan",
        "mortgage",
        "cash advance",
        "repayment",
        "loan repayment",
        "overdue loan",
        "installment",
        "interest rate",
        "disbursement",
        "credit limit",
        "credit line",
        "emi",
        "emis",
        "gloan",
        "ggives",
        "gcredit",
    ],
    "payments_billing": [
        "payment",
        "pay now",
        "make a payment",
        "payment failed",
        "payment received",
        "payment due",
        "invoice",
        "utility bill",
        "electricity bill",
        "water bill",
        "phone bill",
        "billing statement",
        "bill payment",
        "amount due",
        "balance due",
        "past due",
        "unpaid amount",
        "unpaid balance",
        "outstanding amount",
        "payment notice",
        "settlement",
        "debt collection",
        "collection notice",
        "service fee",
        "processing fee",
        "delivery fee",
        "shipping fee",
        "toll",
        "traffic fine",
        "parking fine",
        "unpaid fine",
        "fine payment",
        "penalty",
        "tax",
        "refund",
        "reimbursement",
        "money back",
        "subscription charge",
        "bill reminder",
        "traffic violation",
        "violation fine",
        "violation fines",
        "energy bill",
        "delivery fee",
        "pay fee",
        "billing information",
        "billing details",
        "direct debit",
        "customs fee",
        "customs charge",
        "redelivery fee",
    ],
    "investment_crypto": [
        "investment",
        "invest",
        "investing",
        "investor",
        "stock trading",
        "stock investment",
        "stocks",
        "stock market",
        "shares",
        "trading",
        "trader",
        "forex",
        "cryptocurrency",
        "crypto",
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "usdt",
        "binance",
        "coinbase",
        "metamask",
        "crypto wallet",
        "trading account",
        "portfolio",
        "dividend",
        "staking",
        "defi",
        "coin login",
        "crypto login",
        "crypto balance",
        "usdt balance",
        "coin balance",
        "property investment",
        "property deal",
        "investment property",
        "oil investment",
        "trust wallet",
        "digital currency",
    ],
    "gambling": [
        "gambling",
        "casino",
        "online casino",
        "bet",
        "betting",
        "sports betting",
        "sportsbook",
        "wager",
        "jackpot",
        "casino slots",
        "online slots",
        "slot machine",
        "slot game",
        "free spins",
        "casino bonus",
        "betting bonus",
        "deposit bonus",
        "welcome bonus",
        "odds",
        "payout",
        "winnings",
        "withdraw winnings",
        "claim winnings",
        "win big",
        "vip casino",
        "spin",
        "free spin",
        "lucky wheel",
        "winning",
        "panalo",
        "nanalo",
        "tumaya",
        "deposito",
        "maglaro",
        "magparehistro",
        "premyo",
        "b0nus",
        "first deposit",
        "unang deposito",
        "1st deposit",
        "sign up bonus",
        "lucky draw",
        "raffle ticket",
        "magdeposito",
        "mag deposit",
        "mag-recharge",
        "recharge",
        "new member bonus",
        "free bonus",
        "daily bonus",
        "play now",
        "paglalaro",
        "pagwawagi",
        "epicwin",
        "gojackpot",
        "pinoyslot",
        "winningplus",
        "winpnow",
        "panaloka",
        "jili747",
        "co777",
        "sbet",
        "slot machines",
        "magrehistro",
        "w19 games",
        "casin0",
        "sl0t",
        "super slots",
        "deposit promo",
        "registration bonus",
        "play and win",
        "baraha",
        "bets",
        "free bets",
    ],
}

# Turning messy text into a more predictable matching form
def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return " ".join(normalized.split())

# Making sure keyword exists as its own and not sandwiched by other characters
def keyword_is_present(normalized_text: str, keyword: str) -> bool:
    escaped_keyword = re.escape(keyword)
    # Spaces may be punctuation or whitespace in an SMS. A few common linking
    # words cover audited phrases such as "account has been disabled" without
    # allowing the keyword words to be fused together.
    separator = r"[^\w]+(?:(?:is|is[^\w]+now|has[^\w]+been)[^\w]+)?"
    phrase_pattern = escaped_keyword.replace(r"\ ", separator)
    pattern = rf"(?<!\w){phrase_pattern}(?!\w)"
    return re.search(pattern, normalized_text) is not None

# Checking the message against every category and collecting all matches
def match_categories(text: object) -> tuple[list[str], list[str]]:
    normalized_text = normalize_text(text)
    candidate_tags: list[str] = []
    matched_keywords: list[str] = []

    # Going through every category and its corresponding keyword list
    for category, keywords in CATEGORY_KEYWORDS.items():
        category_matches = [
            keyword
            for keyword in keywords
            if keyword_is_present(normalized_text, keyword)
        ]

        # Adding the category if at least one of its keywords matched
        if category_matches:
            candidate_tags.append(category)
            for keyword in category_matches:
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)

    return candidate_tags, matched_keywords

# Applying the category matching to every message in the dataframe
def tag_dataframe(dataframe: pd.DataFrame, text_column: str) -> pd.DataFrame:
    if text_column not in dataframe.columns:
        available = ", ".join(str(column) for column in dataframe.columns)
        raise ValueError(
            f"Text column {text_column!r} was not found. Available columns: {available}"
        )

     # Creating a copy so the original dataframe is not directly modified
    tagged_dataframe = dataframe.copy()
    matches = tagged_dataframe[text_column].apply(match_categories)

    # Saving the matched categories into one column separated by |
    tagged_dataframe["candidate_tags"] = matches.apply(
        lambda result: "|".join(result[0])
    )

    # Saving the keywords that caused the matches into another column
    tagged_dataframe["matched_keywords"] = matches.apply(
        lambda result: "|".join(result[1])
    )
    return tagged_dataframe

def load_ph_spam_marketing_csv(input_path: Path) -> pd.DataFrame:
    """Read this source's multiline CSV and repair its known category overflow."""
    columns = ["date-received", "date-read", "sender", "category", "text"]
    rows = []
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        if next(reader, None) != columns:
            raise ValueError("Unexpected PH spam/marketing CSV header.")

        for row in reader:
            # Two source records contain unquoted ads,notifs before <REDACTED>.
            # Interpret those as one source category, not as extra SMS columns.
            # csv.reader handles quoted commas and embedded newlines normally.
            if len(row) == 6 and row[3:] == ["ads", "notifs", "<REDACTED>"]:
                warnings.warn(
                    f"{input_path}: physical line {reader.line_num}: "
                    "interpreting unquoted ads,notifs as one category; row retained.",
                    stacklevel=2,
                )
                row = row[:3] + ["ads,notifs", row[5]]
            elif len(row) != len(columns):
                raise ValueError(
                    f"{input_path}: record ending at physical line {reader.line_num} "
                    f"has {len(row)} fields; expected {len(columns)}. No rows skipped."
                )
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def load_csv(input_path: Path) -> pd.DataFrame:
    """Keep the known source-specific repair separate from keyword matching."""
    standardized_path = PROJECT_ROOT / "data" / "standardized"
    if input_path.resolve().is_relative_to(standardized_path.resolve()):
        # Keeping standardized fields as strings preserves blank metadata as blanks.
        return pd.read_csv(input_path, dtype=str, keep_default_na=False)
    marketing_path = PROJECT_ROOT / "data" / "raw" / "ph-spam-marketing" / "text-messages.csv"
    if input_path.resolve() == marketing_path.resolve():
        return load_ph_spam_marketing_csv(input_path)
    spam_collection_path = PROJECT_ROOT / "data" / "raw" / "spam-collection" / "spam.csv"
    if input_path.resolve() == spam_collection_path.resolve():
        # This source is not valid UTF-8. Latin-1 preserves every byte, including
        # existing garbled text; repairing that text is outside this loader's scope.
        # Keep all columns (including unnamed extras), literal strings, and blanks.
        return pd.read_csv(
            input_path, encoding="latin-1", dtype=str, keep_default_na=False
        )
    return pd.read_csv(input_path)


# Reading the CSV, optionally sampling rows, tagging them, and saving the output
def tag_csv(
    input_path: Path,
    output_path: Path,
    text_column: str,
    sample_size: int | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output path must be different from the input path.")

    dataframe = load_csv(input_path)

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("Sample size must be greater than zero.")
        if sample_size > len(dataframe):
            raise ValueError(
                f"Sample size {sample_size} exceeds the {len(dataframe)} input rows."
            )

        # Sorting the sampled indices restores source-file order for easier review.
        dataframe = dataframe.sample(
            n=sample_size,
            random_state=random_seed,
        ).sort_index()

    tagged_dataframe = tag_dataframe(dataframe, text_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tagged_dataframe.to_csv(output_path, index=False)
    return len(tagged_dataframe)

# Reading the options given when running the script from the command line
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add financial candidate tags to SMS messages."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Separate tagged CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help=f"SMS text column (default: {DEFAULT_TEXT_COLUMN})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Randomly sample this many input rows before tagging (default: all rows)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Seed used for reproducible sampling (default: {DEFAULT_RANDOM_SEED})",
    )
    return parser.parse_args()

# Controlling the overall flow when the script is executed
def main() -> None:
    args = parse_args()
    row_count = tag_csv(
        args.input,
        args.output,
        args.text_column,
        args.sample_size,
        args.random_seed,
    )
    print(f"Tagged {row_count} rows and wrote: {args.output}")


if __name__ == "__main__":
    main()
