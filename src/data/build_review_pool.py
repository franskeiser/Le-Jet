"""Prepare a review workload, without assigning any human annotation values."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "standardized" / "master_candidate_tagged.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "annotation"
INPUT_COLUMNS = [
    "message_id", "source", "source_row_id", "text", "source_label",
    "source_category", "language", "sender", "url", "duplicate_group_id",
    "candidate_tags", "matched_keywords",
]
CONTEXT_COLUMNS = [
    "message_id", "source", "text", "source_label", "source_category",
    "candidate_tags", "matched_keywords",
]
HUMAN_COLUMNS = ["human_label", "subcategory", "review_status", "labeler", "notes"]
REVIEW_COLUMNS = CONTEXT_COLUMNS + HUMAN_COLUMNS
FILTER_MISS_COLUMNS = [
    "message_id", "source", "text", "source_label", "source_category",
    "language", "candidate_tags", "matched_keywords", "duplicate_group_id",
    "exclusion_reason",
]
MAP_COLUMNS = ["message_id", "duplicate_group_id", "canonical_message_id", "is_canonical"]
SOURCES = {
    "mendeley", "spam_collection", "ncsu", "fishing_for_smishing",
    "ph_spam_marketing", "ph_spam_scam", "filipino_spam_sms_yissuh",
    "isda_smishing_detector",
}
ALLOWED_LANGUAGES = {"english", "tagalog", "filipino", "taglish"}
# Unknown is missing language information, not evidence of a foreign language.
UNSPECIFIED_LANGUAGES = {"", "unknown"}
# These are the explicit unsupported languages found in the current input.
# An unfamiliar new value raises an error so it cannot be excluded by guessing.
UNSUPPORTED_LANGUAGES = set(
    "spanish|dutch|french|german|italian|indonesian|portuguese|japanese|hindi|malay|"
    "polish|danish|turkish|greek|urdu|czech|thai|arabic|swedish|chinese|norwegian|"
    "romanian|catalan|korean|swahili|hungarian|finnish|slovenian|russian|estonian|"
    "kannada|latvian|persian|hebrew|igbo|bengali|ukrainian|serbian|javanese|"
    "afrikaans|croatian|gujarati|icelandic|slovak|zulu|sinhala|lithuanian|punjabi|"
    "albanian|kinyarwanda|bulgarian|maltese|divehi|telugu|tamil|uzbek|vietnamese|"
    "mixed (dutch/french)|welsh|yoruba|nl_fr|mongolian|ganda|marathi".split("|")
)


def validate_input(master: pd.DataFrame) -> None:
    if list(master.columns) != INPUT_COLUMNS:
        raise ValueError(f"Expected input columns in this order: {INPUT_COLUMNS}")
    if master.isna().any().any():
        raise ValueError("Input contains NaN values instead of preserved strings.")
    if not master["message_id"].is_unique or master["message_id"].str.strip().eq("").any():
        raise ValueError("Input message_id values must be nonblank and unique.")
    if master["duplicate_group_id"].str.strip().eq("").any():
        raise ValueError("Every input row must have an existing duplicate_group_id.")
    if not set(master["source"]).issubset(SOURCES):
        raise ValueError("Input contains an unexpected source.")
    # Check the existing groups, but do not create or normalize duplicate groups.
    if master.groupby("duplicate_group_id")["text"].nunique().gt(1).any():
        raise ValueError("An existing duplicate group contains different SMS texts.")


def eligible_mask(master: pd.DataFrame) -> pd.Series:
    # Stripping here only tests usability; the stored message text stays untouched.
    text = master["text"].str.strip()
    language = master["language"].str.strip().str.lower()
    unfamiliar = set(language) - ALLOWED_LANGUAGES - UNSPECIFIED_LANGUAGES - UNSUPPORTED_LANGUAGES
    if unfamiliar:
        raise ValueError(f"Language metadata needs human clarification: {sorted(unfamiliar)}")
    empty = text.eq("")
    redacted = text.eq("<REDACTED>")
    usable = ~empty & ~redacted
    unsupported = language.isin(UNSUPPORTED_LANGUAGES)
    # Keep blank and unknown language metadata without trying to detect language.
    eligible = usable & ~unsupported
    return eligible


def known_legitimate_routes(canonical: pd.DataFrame) -> dict[str, pd.Series]:
    source = canonical["source"]
    label = canonical["source_label"].str.casefold()
    category = canonical["source_category"].str.casefold()
    # Using source labels as review context instead of final SmishGuard labels.
    return {
        "Mendeley ham candidates": source.eq("mendeley") & label.eq("ham"),
        "Spam Collection ham candidates": source.eq("spam_collection") & label.eq("ham"),
        "Yissuh ham candidates": source.eq("filipino_spam_sms_yissuh") & label.eq("ham"),
        "ISDA ham candidates": source.eq("isda_smishing_detector") & label.eq("ham"),
        "PH Marketing notifs candidates": source.eq("ph_spam_marketing") & category.eq("notifs"),
        "PH Marketing OTP candidates": source.eq("ph_spam_marketing") & category.eq("otp"),
        "PH Marketing ads candidates": source.eq("ph_spam_marketing") & category.eq("ads"),
        "PH Marketing gov candidates": source.eq("ph_spam_marketing") & category.eq("gov"),
    }


def build_duplicate_map(master: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    # Mapping every original row back to one duplicate representative.
    # Fully ineligible groups have a first-original representative, never a review row.
    first_original = master.drop_duplicates("duplicate_group_id", keep="first")
    representatives = first_original.set_index("duplicate_group_id")["message_id"].copy()
    first_eligible = canonical.set_index("duplicate_group_id")["message_id"]
    representatives.loc[first_eligible.index] = first_eligible
    result = master[["message_id", "duplicate_group_id"]].copy()
    result["canonical_message_id"] = result["duplicate_group_id"].map(representatives)
    result["is_canonical"] = result["message_id"].eq(result["canonical_message_id"])
    return result


def build_review_pool(master: pd.DataFrame):
    # Checking that the input follows the expected structure.
    validate_input(master)
    # Removing unusable rows before choosing one copy per duplicate group.
    eligible = eligible_mask(master)
    eligible_rows = master.loc[eligible]
    # Keeping one eligible copy of each exact message for review.
    canonical = eligible_rows.drop_duplicates("duplicate_group_id", keep="first")
    duplicate_map = build_duplicate_map(master, canonical)
    # Finding messages that already passed the financial keyword filter.
    keyword_tagged = canonical["candidate_tags"].str.strip().ne("")
    # Marking source-known legitimate messages without assigning a human label.
    routes = known_legitimate_routes(canonical)
    known_legitimate = pd.Series(False, index=canonical.index)
    for route in routes.values():
        known_legitimate |= route
    # Selecting financial candidates or known legitimate messages.
    selected = keyword_tagged | known_legitimate
    review = canonical.loc[selected, CONTEXT_COLUMNS].copy()
    # Adding blank fields for manual annotation.
    for field in HUMAN_COLUMNS:
        review[field] = ""
    review = review.reset_index(drop=True)

    # Keeping unselected usable messages in a separate recovery pool.
    filter_miss = canonical.loc[~selected, FILTER_MISS_COLUMNS[:-1]].copy()
    filter_miss["exclusion_reason"] = "untagged_not_known_legitimate"
    filter_miss = filter_miss.reset_index(drop=True)
    duplicate_map = duplicate_map.reset_index(drop=True)
    validate_outputs(master, review, filter_miss, duplicate_map, eligible)
    return review, filter_miss, duplicate_map


def validate_outputs(master, review, filter_miss, duplicate_map, eligible) -> None:
    if (
        list(review.columns) != REVIEW_COLUMNS
        or list(filter_miss.columns) != FILTER_MISS_COLUMNS
        or list(duplicate_map.columns) != MAP_COLUMNS
    ):
        raise ValueError("Output columns do not match the required schemas.")
    if not review["message_id"].is_unique or not review["message_id"].isin(master["message_id"]).all():
        raise ValueError("Review IDs must be unique IDs from the input master.")
    if not review[HUMAN_COLUMNS].eq("").all().all():
        raise ValueError("Human annotation fields must all start blank.")
    original = master.set_index("message_id").loc[review["message_id"]].reset_index()
    if not original[CONTEXT_COLUMNS].equals(review[CONTEXT_COLUMNS]):
        raise ValueError("Review context differs from the original canonical rows.")
    expected_canonical = master.loc[eligible].drop_duplicates("duplicate_group_id", keep="first")
    if not review["message_id"].isin(expected_canonical["message_id"]).all():
        raise ValueError("A review row is not the first eligible copy of its group.")
    if original["duplicate_group_id"].duplicated().any():
        raise ValueError("A duplicate group appears more than once in the review pool.")
    keyword_tagged = expected_canonical["candidate_tags"].str.strip().ne("")
    known_legitimate = pd.Series(False, index=expected_canonical.index)
    for route in known_legitimate_routes(expected_canonical).values():
        known_legitimate |= route
    expected_ids = expected_canonical.loc[keyword_tagged | known_legitimate, "message_id"]
    if review["message_id"].tolist() != expected_ids.tolist():
        raise ValueError("Review pool does not follow keyword-tagged OR known-legitimate selection.")
    # Checking that the recovery pool is exactly the unselected canonical remainder.
    if not filter_miss["message_id"].is_unique:
        raise ValueError("Filter-miss message IDs must be unique.")
    if filter_miss["duplicate_group_id"].duplicated().any():
        raise ValueError("A duplicate group appears more than once in the filter-miss pool.")
    if not filter_miss["candidate_tags"].str.strip().eq("").all():
        raise ValueError("Filter-miss rows must have blank candidate tags.")
    miss_legitimate = pd.Series(False, index=filter_miss.index)
    for route in known_legitimate_routes(filter_miss).values():
        miss_legitimate |= route
    if miss_legitimate.any():
        raise ValueError("A known-legitimate row was placed in the filter-miss pool.")
    if not filter_miss["exclusion_reason"].eq("untagged_not_known_legitimate").all():
        raise ValueError("Filter-miss exclusion reasons are incorrect.")
    expected_miss = expected_canonical.loc[~(keyword_tagged | known_legitimate)]
    original_miss_columns = FILTER_MISS_COLUMNS[:-1]
    if not filter_miss[original_miss_columns].equals(expected_miss[original_miss_columns].reset_index(drop=True)):
        raise ValueError("Filter-miss rows differ from the unselected canonical messages.")
    review_ids = set(review["message_id"])
    miss_ids = set(filter_miss["message_id"])
    canonical_ids = set(expected_canonical["message_id"])
    if review_ids & miss_ids or review_ids | miss_ids != canonical_ids:
        raise ValueError("Review and filter-miss pools must partition all canonical messages.")
    # This also ensures unselected/unsupported source rows remain in the technical map.
    if not duplicate_map[["message_id", "duplicate_group_id"]].equals(
        master[["message_id", "duplicate_group_id"]].reset_index(drop=True)
    ):
        raise ValueError("Duplicate map must preserve every original ID and group in order.")
    if not duplicate_map["canonical_message_id"].isin(master["message_id"]).all():
        raise ValueError("Duplicate map contains a canonical ID absent from the master.")
    groups = duplicate_map.groupby("duplicate_group_id")
    if groups["canonical_message_id"].nunique().ne(1).any() or groups["is_canonical"].sum().ne(1).any():
        raise ValueError("Each duplicate group must map to exactly one canonical row.")
    if not duplicate_map["is_canonical"].equals(
        duplicate_map["message_id"].eq(duplicate_map["canonical_message_id"])
    ):
        raise ValueError("Duplicate map has incorrect is_canonical flags.")
    expected_map = build_duplicate_map(master, expected_canonical).reset_index(drop=True)
    if not duplicate_map.equals(expected_map):
        raise ValueError("Duplicate map does not use the first eligible representatives.")


def main() -> None:
    paths = [
        OUTPUT_DIR / "review_pool.csv",
        OUTPUT_DIR / "filter_miss_pool.csv",
        OUTPUT_DIR / "duplicate_map.csv",
    ]
    # Do not accidentally erase decisions after humans have started annotating.
    if any(path.exists() for path in paths):
        raise FileExistsError("Annotation output already exists. Back it up or move it before rebuilding.")
    master = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
    validate_input(master)
    review, filter_miss, duplicate_map = build_review_pool(master)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Saving the three output files.
    for frame, path in zip([review, filter_miss, duplicate_map], paths):
        frame.to_csv(path, index=False, encoding="utf-8")
        saved = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not saved.equals(frame.astype(str)):
            raise ValueError(f"{path}: CSV round-trip changed output values.")


if __name__ == "__main__":
    main()
