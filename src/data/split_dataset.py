# src/data/split_dataset.py
import argparse
import hashlib
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATASET_PATH, LABEL_COLUMN, SEEDS, SPLITS_DIR

ID_COLUMN = "message_id"

# DATASET_PATH / SPLITS_DIR now live in src/config.py (currently v2withSynthetic).

# Task 1.9 only produces train/rf_fit/test. Phase A (Task 2.2) still needs a
# validation slice for early stopping -- carved out of the Training split ONLY,
# so rf_fit/test stay untouched and Section 1's disjointness guarantee holds.
VAL_FRACTION_OF_TRAIN = 0.15


def resolve_label_column(df: pd.DataFrame) -> str:
    # Section 4: config.LABEL_COLUMN points at "subcategory," which isn't in
    # the checked-in labeled export yet. Prefer it the moment it exists;
    # fall back to the binary "label" column, which is what's real today.
    if "label" in df.columns:
        return "label"
    if LABEL_COLUMN in df.columns:
        return LABEL_COLUMN
    raise ValueError(
        f"Neither '{LABEL_COLUMN}' nor 'label' found in input columns: {df.columns.tolist()}"
    )


def ensure_message_id(df: pd.DataFrame) -> pd.DataFrame:
    # Section 0: data/labeled/*.csv doesn't carry message_id through from the
    # annotation pipeline yet. Prefer a real message_id if the input already
    # has one (true once the upstream export is fixed — see Section 10);
    # otherwise derive a stable content-hash ID from sms_text, and refuse to
    # proceed if sms_text has duplicates, since a hash ID would silently
    # merge two distinct rows into one sample in that case.
    if "sms" in df.columns and "sms_text" not in df.columns:
        df = df.rename(columns={"sms": "sms_text"})
    if ID_COLUMN in df.columns:
        return df
    dupes = df["sms_text"][df["sms_text"].duplicated()]
    if not dupes.empty:
        raise ValueError(
            "No message_id column, and sms_text has "
            f"{len(dupes)} duplicate value(s) — a content-hash ID would silently "
            "merge distinct rows. Fix upstream (thread message_id through the "
            "labeled export) instead of deriving IDs here. First duplicate: "
            f"{dupes.iloc[0]!r}"
        )
    df = df.copy()
    df[ID_COLUMN] = df["sms_text"].apply(
        lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    )
    return df


def three_way_split(
    df: pd.DataFrame, label_col: str, seed: int, test_size: float, rf_fit_size: float,
):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed,
    )

    # Section 2.2: rf_fit_size is meant as a fraction of the ORIGINAL total,
    # but train_val is only (1 - test_size) of the original. Re-express the
    # target fraction relative to train_val's own size before the second call
    # — passing rf_fit_size directly here would carve it out of the smaller
    # remainder pool instead of the original total.
    rf_fit_fraction_of_remainder = rf_fit_size / (1 - test_size)

    train, rf_fit = train_test_split(
        train_val,
        test_size=rf_fit_fraction_of_remainder,
        stratify=train_val[label_col],
        random_state=seed,
    )
    return train, rf_fit, test


def assert_disjoint(train: pd.DataFrame, rf_fit: pd.DataFrame, test: pd.DataFrame):
    train_ids, rf_fit_ids, test_ids = (
        set(train[ID_COLUMN]), set(rf_fit[ID_COLUMN]), set(test[ID_COLUMN]),
    )
    assert not (train_ids & rf_fit_ids), "train/rf_fit overlap"
    assert not (train_ids & test_ids), "train/test overlap"
    assert not (rf_fit_ids & test_ids), "rf_fit/test overlap"


def save_ids(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[[ID_COLUMN]].to_csv(path, index=False, header=False)


def run(input_path: str, seed: int, output_dir: str, test_size: float, rf_fit_size: float):
    df = pd.read_csv(input_path)
    df = ensure_message_id(df)
    label_col = resolve_label_column(df)

    train, rf_fit, test = three_way_split(df, label_col, seed, test_size, rf_fit_size)
    assert_disjoint(train, rf_fit, test)
    assert len(train) + len(rf_fit) + len(test) == len(df), "rows lost during split"

    save_ids(train, os.path.join(output_dir, "train_ids.txt"))
    save_ids(rf_fit, os.path.join(output_dir, "rf_fit_ids.txt"))
    save_ids(test, os.path.join(output_dir, "test_ids.txt"))

    print(f"seed={seed}  train={len(train)}  rf_fit={len(rf_fit)}  test={len(test)}")
    for name, split in [("train", train), ("rf_fit", rf_fit), ("test", test)]:
        rates = (split[label_col].value_counts(normalize=True) * 100).round(2).to_dict()
        print(f"  {name:7s} {rates}")


def load_split_ids(seed: int, split_name: str, splits_dir: str = SPLITS_DIR) -> set[str]:
    path = os.path.join(splits_dir, f"seed_{seed}", f"{split_name}_ids.txt")
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_split_df(
    seed: int, split_name: str, dataset_path: str = DATASET_PATH, splits_dir: str = SPLITS_DIR,
) -> pd.DataFrame:
    df = ensure_message_id(pd.read_csv(dataset_path))
    ids = load_split_ids(seed, split_name, splits_dir)
    return df[df[ID_COLUMN].isin(ids)].reset_index(drop=True)


def load_phase_a_train_val(
    seed: int,
    val_fraction: float = VAL_FRACTION_OF_TRAIN,
    dataset_path: str = DATASET_PATH,
    splits_dir: str = SPLITS_DIR,
):
    # Carve validation out of Task 1.9's Training split only -- rf_fit and test
    # are never touched here, so this can't leak into either of them.
    train_full = load_split_df(seed, "train", dataset_path, splits_dir)
    label_col = resolve_label_column(train_full)
    train_df, val_df = train_test_split(
        train_full, test_size=val_fraction, stratify=train_full[label_col], random_state=seed,
    )
    return train_df, val_df, label_col


def main():
    parser = argparse.ArgumentParser(description="Task 1.9 — three-way disjoint split.")
    parser.add_argument("--input", required=True, help="path to a labeled CSV")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True, help="e.g. data/splits/seed_42")
    parser.add_argument(
        "--test-size", type=float, default=0.15,
        help="Test set fraction of the ORIGINAL total (Section 3's recommended default: 0.15)",
    )
    parser.add_argument(
        "--rf-fit-size", type=float, default=0.10,
        help="RF-fitting set fraction of the ORIGINAL total (recommended default: 0.10)",
    )
    args = parser.parse_args()
    run(args.input, args.seed, args.output, args.test_size, args.rf_fit_size)


if __name__ == "__main__":
    main()
