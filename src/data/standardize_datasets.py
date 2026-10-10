#Standardize source schemas without cleaning, labeling, or dropping messages.
#Outputs are reproducible intermediate data, not a manually labeled dataset.


from __future__ import annotations

import csv
import warnings
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "standardized"

# Target structure man
COMMON_COLUMNS = [
    "message_id", "source", "source_row_id", "text", "source_label",
    "source_category", "language", "sender", "url", "duplicate_group_id",
]

# Dict for our seven dataset sources
SOURCE_FILES = {
    "mendeley": "mendeley/Dataset_5971.csv",
    "fishing_for_smishing": "fishing-for-smishing/final_dataset_output.csv",
    "ncsu": "ncsu/phishing_messages.csv",
    "ph_spam_marketing": "ph-spam-marketing/text-messages.csv",
    "ph_spam_scam": "ph-spam-scam/SPAM_SMS.csv",
    "spam_collection": "spam-collection/spam.csv",
    "filipino_spam_sms_yissuh": "filipino-spam-sms-yissuh/merged-set.csv",
    "isda_smishing_detector": "isda-smishing-detector/isda_sms_dataset.csv",
}

# Checks ealy the expected headers for every dataset
SOURCE_HEADERS = {
    "mendeley": ["LABEL", "TEXT", "URL", "EMAIL", "PHONE"],
    "fishing_for_smishing": [
        "time", "sender_id", "telephone_number_type", "original_network_name",
        "original_network_country", "text", "translation", "language",
        "url_shortener", "named_entity", "scam_type", "lure_principles",
    ],
    "ncsu": [
        "messageID", "objectID", "destination number", "message", "time", "error in time",
    ],
    "ph_spam_marketing": ["date-received", "date-read", "sender", "category", "text"],
    "ph_spam_scam": ["masked_celphone_number", "hashed_cellphone_number", "date", "text", "carrier"],
    "spam_collection": ["v1", "v2", "", "", ""],
    "filipino_spam_sms_yissuh": ["label", "message"],
    "isda_smishing_detector": ["ADDRESS", "MESSAGE", "CLASSIFICATION", "CLASS", "SMISH"],
}

# Common field on the left, original source field on the right
# Fields without a mapping stay blank
SOURCE_MAPPINGS = {
    "mendeley": {"text": "TEXT", "source_label": "LABEL", "url": "URL"},
    "fishing_for_smishing": {
        "text": "text", "source_category": "scam_type",
        "language": "language", "sender": "sender_id",
    },
    "ncsu": {"text": "sms_body", "sender": "sender_entity"},
    "ph_spam_marketing": {
        "text": "text", "source_category": "category", "sender": "sender",
    },
    "ph_spam_scam": {"text": "text", "sender": "masked_celphone_number"},
    "spam_collection": {"text": "v2", "source_label": "v1"},
    "filipino_spam_sms_yissuh": {"text": "message", "source_label": "label"},
    "isda_smishing_detector": {"text": "MESSAGE", "sender": "ADDRESS"},
}


def derive_isda_source_labels(raw: pd.DataFrame) -> pd.Series:
    """Map ISDA source evidence without creating final SmishGuard labels."""
    classification = raw["CLASSIFICATION"].str.strip().str.casefold()
    source_class = raw["CLASS"].str.strip()
    smish = raw["SMISH"].str.strip()
    labels = pd.Series("", index=raw.index, dtype=str)
    labels.loc[classification.eq("ham")] = "ham"
    labels.loc[classification.eq("spam") & smish.eq("1")] = "smishing"
    labels.loc[classification.eq("spam") & smish.eq("0")] = "spam"

    recognized = classification.isin({"ham", "spam"}) & source_class.isin({"0", "1"}) & smish.isin({"0", "1"})
    consistent_class = (classification.eq("ham") & source_class.eq("0")) | (
        classification.eq("spam") & source_class.eq("1")
    )
    unexpected = ~recognized | ~consistent_class | (classification.eq("ham") & ~smish.eq("0"))
    if unexpected.any():
        combinations = (
            raw.loc[unexpected, ["CLASSIFICATION", "CLASS", "SMISH"]]
            .value_counts(dropna=False)
            .to_dict()
        )
        warnings.warn(
            f"ISDA contains unexpected CLASSIFICATION/CLASS/SMISH combinations: {combinations}. "
            "Rows are preserved and labels follow the explicit CLASSIFICATION/SMISH mapping when possible.",
            stacklevel=2,
        )
    return labels


def load_source(source: str, path: Path) -> pd.DataFrame:
    encoding = "latin-1" if source == "spam_collection" else "utf-8-sig"
    columns = SOURCE_HEADERS[source].copy()
    if source == "ncsu":
        # The header is missing the sender column, (6 columns but has 7 record fields)
        # Explicit names avoid pandas silently shifting values into the wrong columns
        columns = [
            "messageID", "objectID", "destination_number", "sms_body",
            "sender_entity", "time", "error_in_time",
        ]
    elif source == "spam_collection":
        # Keep the extra fields while loading. They are not part of our common schema
        # Latin-1 preserves the existing garbled characters; we do not repair spelling
        columns = ["v1", "v2", "extra_1", "extra_2", "extra_3"]

    rows = []
    with path.open(encoding=encoding, newline="") as handle:
        # newline="" preserves newlines inside quoted SMS text.
        reader = csv.reader(handle, strict=True)
        try:
            if next(reader, None) != SOURCE_HEADERS[source]:
                raise ValueError(f"{path}: unexpected CSV header.") # Just checks the header using the dictionary
            for row in reader:
                if (
                    source == "ph_spam_marketing"
                    and len(row) == 6
                    and row[3:] == ["ads", "notifs", "<REDACTED>"]
                ):
                   # Broken records have six columns so we'll slice the list and add to repair it
                    warnings.warn(
                        f"{path}: physical line {reader.line_num}: joining ads,notifs "
                        "into one source category; retaining the row.",
                        stacklevel=2,
                    )
                    row = row[:3] + ["ads,notifs", row[5]]
                if len(row) != len(columns):
                    raise ValueError(
                        f"{path}: record ending at physical line {reader.line_num} "
                        f"has {len(row)} fields; expected {len(columns)}. No rows skipped."
                    )
                rows.append(row)
        except csv.Error as error:
            raise ValueError(
                f"{path}: CSV error near physical line {reader.line_num}: {error}"
            ) from error

    # csv.reader leaves blanks, leading zeroes, and strings such as 'NA' unchanged.
    return pd.DataFrame(rows, columns=columns, dtype=str)


def standardize_source(source: str, raw: pd.DataFrame) -> pd.DataFrame:
    # Recreating the raw file in df and creates stable IDs
    result = pd.DataFrame("", index=raw.index, columns=COMMON_COLUMNS)
    positions = range(1, len(raw) + 1)
    result["message_id"] = [f"{source}_{position:06d}" for position in positions]
    result["source"] = source

    # Row positions are equal to the position in csv files. in termsf fr ncsu, we'll retain its message ID  
    result["source_row_id"] = [str(position) for position in positions]
    if source == "ncsu":
        if raw["messageID"].eq("").any() or raw["messageID"].duplicated().any():
            raise ValueError("NCSU messageID must be nonempty and unique.")
        result["source_row_id"] = raw["messageID"]

    # Tells what to copy from the dictionary by passing each of it
    for common_field, original_field in SOURCE_MAPPINGS[source].items():
        result[common_field] = raw[original_field]

    if source == "isda_smishing_detector":
        result["source_label"] = derive_isda_source_labels(raw)
        # Preserve both original numeric classification fields as source context.
        result["source_category"] = "class=" + raw["CLASS"] + "|smish=" + raw["SMISH"]

    # Mendeley's URL field contains Yes/No-style indicators. Preserve those literal
    # source values as requested; do not pretend they are extracted URL strings.
    # Fishing's translation/shortener and PH carrier are deliberately not mapped.
    return result


def add_duplicate_groups(master: pd.DataFrame) -> pd.DataFrame:
    # Exact equality only: no case, punctuation, or whitespace normalization.
    # Every distinct text gets a group, even a singleton or the empty string.
    result = master.copy()
    codes, _ = pd.factorize(result["text"], sort=False)

    # Converts repeated values into numeric codes then
    result["duplicate_group_id"] = [f"dup_{code + 1:06d}" for code in codes]
    return result 


def empty_text_count(frame: pd.DataFrame) -> int:
    # Whitespace-only messages count as empty for reporting, but remain unchanged
    return int(frame["text"].str.strip().eq("").sum())

# Security chckpoint
def validate_master(master: pd.DataFrame, raw_counts: dict[str, int]) -> None:
    if list(master.columns) != COMMON_COLUMNS:
        raise ValueError("Master column order does not match the common schema.")
    if master["message_id"].eq("").any() or not master["message_id"].is_unique:
        raise ValueError("Master message IDs are empty or duplicated.")
    if len(master) != sum(raw_counts.values()):
        raise ValueError("Master row count differs from the raw source total.")
    if set(raw_counts) != set(SOURCE_FILES):
        raise ValueError("A required source is missing or an unexpected source was added.")
    if master["source"].value_counts().to_dict() != raw_counts:
        raise ValueError("Rows per source do not match the raw source counts.")
    if any(Path(name).name == "phishing_campaigns.csv" for name in SOURCE_FILES.values()):
        raise ValueError("Campaign metadata must not be included.")
    if master.groupby("text")["duplicate_group_id"].nunique().gt(1).any():
        raise ValueError("Identical text received different duplicate groups.")
    if master.groupby("duplicate_group_id")["text"].nunique().gt(1).any():
        raise ValueError("Different texts share a duplicate group.")


def write_and_check(frame: pd.DataFrame, path: Path) -> None:
    if list(frame.columns) != COMMON_COLUMNS:
        raise ValueError(f"{path}: incorrect standardized column order.")
    frame.to_csv(path, index=False, encoding="utf-8")

    # Reading back checks that CSV quoting preserves every field, including newlines.
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        if next(reader) != COMMON_COLUMNS:
            raise ValueError(f"{path}: written header changed.")
        if list(reader) != frame.values.tolist():
            raise ValueError(f"{path}: written rows differ from the standardized data.")


def main() -> None:
    frames = []
    raw_counts = {}

    # Iterates through the surce files dict then reads, converts to common schema, then checks number rows
    for source, filename in SOURCE_FILES.items():
        raw = load_source(source, RAW_DIR / filename)
        standardized = standardize_source(source, raw)
        if len(raw) != len(standardized):
            raise ValueError(f"{source}: standardization changed the row count.")
        raw_counts[source] = len(raw)
        frames.append(standardized)
        print(
            f"{source}: raw={len(raw)}, standardized={len(standardized)}, "
            f"empty_text={empty_text_count(standardized)}, "
            f"source_labels={standardized['source_label'].ne('').sum()}, "
            f"source_categories={standardized['source_category'].ne('').sum()}"
        )

    # Assigning groups after combining also finds duplicates across different sources.
    master = add_duplicate_groups(pd.concat(frames, ignore_index=True))
    validate_master(master, raw_counts)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Selects specific source dataset from the combined master
    for source in SOURCE_FILES:
        write_and_check(master.loc[master["source"].eq(source)], OUTPUT_DIR / f"{source}.csv")
    # Writing the master file
    write_and_check(master, OUTPUT_DIR / "master_standardized.csv")

    counts = master["text"].value_counts(dropna=False)
    print(f"Master: total={len(master)}, unique_message_ids={master['message_id'].nunique()}")
    print(f"Empty text: {empty_text_count(master)}")

    # Duplicate rows means extra occurrences beyond the first, not all group members.
    print(f"Exact duplicate rows beyond first: {master['text'].duplicated().sum()}")
    print(f"Duplicate groups with 2+ rows: {counts.gt(1).sum()}")
    print(f"Rows belonging to duplicate groups: {counts[counts.gt(1)].sum()}")
    print(f"Rows per source: {raw_counts}")
    print(f"Validated all {len(SOURCE_FILES) + 1} CSV outputs in {OUTPUT_DIR}; no rows skipped.")


if __name__ == "__main__":
    main()
