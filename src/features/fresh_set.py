"""Load and check a fresh-set CSV against docs/fresh_set_spec.md.

Usage (from the project root):
    python -m src.features.fresh_set data/fresh/fresh_2026-11-01.csv
Prints every problem found and exits non-zero if there are any.
"""
import sys

import pandas as pd

from src.features.link_features import LINK_TEXT_FEATURES
from src.features.model.rf_extractor import VARIANTS

EXTRACTOR_INPUTS = VARIANTS["full"]  # the 31 columns the full extractor reads
MESSAGE_COLUMNS = ["message_id", "sms_text", "label", "source", "link", "link_status"]
RECORD_COLUMNS = [
    "page_fetched", "fetch_status", "final_url", "final_domain", "n_redirects",
    "received_at", "visited_at", "hours_to_visit",
]
REQUIRED_COLUMNS = MESSAGE_COLUMNS + ["has_url"] + EXTRACTOR_INPUTS + RECORD_COLUMNS
NUMERIC_COLUMNS = EXTRACTOR_INPUTS + ["has_url", "page_fetched", "n_redirects"]
FETCH_STATUSES = {"live", "redirected", "dead", "blocked", "timeout", "not_attempted"}
LABELS = {"smishing", "legitimate"}


def validate(df: pd.DataFrame) -> list[str]:
    """Every way `df` breaks the spec, as readable sentences (empty list = fine)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    errors = []

    bad_numeric = [c for c in NUMERIC_COLUMNS if pd.to_numeric(df[c], errors="coerce").isna().any()]
    if bad_numeric:
        errors.append(f"blank or non-numeric values (use -1 for unknown) in: {bad_numeric}")
        return errors  # the checks below need numbers
    num = df[NUMERIC_COLUMNS].apply(pd.to_numeric)

    if df["message_id"].duplicated().any():
        errors.append("message_id is not unique")
    if not df["label"].isin(LABELS).all():
        errors.append(f"label must be one of {sorted(LABELS)}")
    if not df["fetch_status"].isin(FETCH_STATUSES).all():
        errors.append(f"fetch_status must be one of {sorted(FETCH_STATUSES)}")
    if not num["has_url"].isin([0, 1]).all() or not num["page_fetched"].isin([0, 1]).all():
        errors.append("has_url and page_fetched must be 0 or 1")
        return errors

    no_link = num["has_url"] == 0
    if (df.loc[no_link, "link_status"] != "none").any() or (df.loc[~no_link, "link_status"] != "link").any():
        errors.append("link_status must be 'none' exactly when has_url = 0 (never 'redacted')")
    if (num.loc[no_link, "page_fetched"] == 1).any():
        errors.append("a message with no link has page_fetched = 1")
    if (num.loc[no_link, EXTRACTOR_INPUTS] != -1).any().any():
        errors.append("a message with no link has extractor inputs other than -1")

    fetched = num["page_fetched"] == 1
    if (~df.loc[fetched, "fetch_status"].isin(["live", "redirected"])).any():
        errors.append("page_fetched = 1 requires fetch_status 'live' or 'redirected'")
    if (df.loc[fetched, "final_url"].fillna("").astype(str).str.strip() == "").any():
        errors.append("page_fetched = 1 requires a non-empty final_url")

    hours = pd.to_numeric(df["hours_to_visit"], errors="coerce")
    if (hours < 0).any():
        errors.append("hours_to_visit is negative (visited_at before received_at)")

    link_rows = df[~no_link]
    for label in LABELS:
        if not (link_rows["label"] == label).any():
            errors.append(f"no '{label}' message has a link; the has-a-link shortcut would go untested")
    return errors


def load_fresh(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    errors = validate(df)
    if errors:
        raise ValueError("fresh set breaks docs/fresh_set_spec.md:\n  - " + "\n  - ".join(errors))
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric)
    return df


def main() -> None:
    df = pd.read_csv(sys.argv[1], keep_default_na=False, na_values=[""])
    errors = validate(df)
    for e in errors:
        print("-", e)
    print(f"{len(df)} rows, {len(errors)} problem(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
