# src/data/split_dataset.py
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.config import DATASET_PATH, LABEL_COLUMN, LINK_FEATURES_PATH, SEEDS, SPLITS_DIR

ID_COLUMN = "message_id"

# DATASET_PATH / LINK_FEATURES_PATH / SPLITS_DIR live in src/config.py.
# Usage (from the project root): python -m src.data.split_dataset

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


# Splits are built from 20 equal folds: 3 for Test (15%), 2 for RF-Fitting (10%), the
# other 15 for Training (75%).
N_FOLDS = 20
TEST_FOLDS = 3
RF_FIT_FOLDS = 2


def split_groups(df: pd.DataFrame) -> pd.Series:
    # Rows that share a group always land in the same split:
    #   synthetic rows   -> their template, so synthetic test rows come from templates the
    #                       model never trained on
    #   real with a link -> the link's registrable domain, so one scam domain (e.g. 1,168
    #                       messages for usdtjerc.com) is never in both Training and Test
    #   other real rows  -> on their own
    synthetic = df["source"].str.startswith("synthetic")
    has_link = df["link_status"] == "link"
    return pd.Series(
        np.select(
            [synthetic, has_link],
            ["template:" + df["template_id"].astype(str), "domain:" + df["link_domain"].astype(str)],
            "message:" + df[ID_COLUMN].astype(str),
        ),
        index=df.index,
    )


def three_way_split(df: pd.DataFrame, label_col: str, seed: int):
    groups = split_groups(df)
    fold = np.empty(len(df), dtype=int)
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for i, (_, idx) in enumerate(folds.split(df, df[label_col], groups)):
        fold[idx] = i
    test = df[fold < TEST_FOLDS]
    rf_fit = df[(fold >= TEST_FOLDS) & (fold < TEST_FOLDS + RF_FIT_FOLDS)]
    train = df[fold >= TEST_FOLDS + RF_FIT_FOLDS]
    for a, b in [(train, rf_fit), (train, test), (rf_fit, test)]:
        assert not set(groups[a.index]) & set(groups[b.index]), "a group spans two splits"
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


def split_report(name: str, part: pd.DataFrame, label_col: str) -> dict:
    smishing = part[label_col] == "smishing"
    real = ~part["source"].str.startswith("synthetic")
    with_link = part["has_url"] == 1
    return {
        "split": name, "rows": len(part), "smishing_share": round(smishing.mean(), 4),
        "real_rows": int(real.sum()), "synthetic_rows": int((~real).sum()),
        "legit_with_link_share": round(with_link[~smishing].mean(), 4),
        "smishing_with_link_share": round(with_link[smishing].mean(), 4),
        "redacted_rows": int((part["link_status"] == "redacted").sum()),
    }


def run(dataset_path: str, features_path: str, seeds: list[int], splits_dir: str) -> pd.DataFrame:
    df = ensure_message_id(pd.read_csv(dataset_path))
    label_col = resolve_label_column(df)
    features = pd.read_csv(features_path, keep_default_na=False)
    df = df.merge(features[[ID_COLUMN, "link_status", "link_domain", "has_url"]], on=ID_COLUMN)

    rows = []
    for seed in seeds:
        train, rf_fit, test = three_way_split(df, label_col, seed)
        assert_disjoint(train, rf_fit, test)
        assert len(train) + len(rf_fit) + len(test) == len(df), "rows lost during split"
        for name, part in [("train", train), ("rf_fit", rf_fit), ("test", test)]:
            save_ids(part, os.path.join(splits_dir, f"seed_{seed}", f"{name}_ids.txt"))
            rows.append({"seed": seed, **split_report(name, part, label_col)})
    return pd.DataFrame(rows)


def load_split_ids(seed: int, split_name: str, splits_dir: str = SPLITS_DIR) -> set[str]:
    path = os.path.join(splits_dir, f"seed_{seed}", f"{split_name}_ids.txt")
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_split_df(
    seed: int, split_name: str, dataset_path: str = DATASET_PATH, splits_dir: str = SPLITS_DIR,
    exclude_redacted: bool = True, features_path: str = LINK_FEATURES_PATH,
) -> pd.DataFrame:
    # exclude_redacted drops messages whose link was replaced by the literal "<URL>"
    # placeholder: their URL-branch pattern (has_url = 1, features -1) is 98.6% smishing
    # and never occurs in real messages. Keep it True for training and reported results.
    # See docs/training_rules.md, rule 1.
    df = ensure_message_id(pd.read_csv(dataset_path))
    ids = load_split_ids(seed, split_name, splits_dir)
    df = df[df[ID_COLUMN].isin(ids)]
    if exclude_redacted:
        status = pd.read_csv(features_path, usecols=[ID_COLUMN, "link_status"])
        redacted = set(status.loc[status["link_status"] == "redacted", ID_COLUMN])
        df = df[~df[ID_COLUMN].isin(redacted)]
    return df.reset_index(drop=True)


def load_phase_a_train_val(
    seed: int,
    val_fraction: float = VAL_FRACTION_OF_TRAIN,
    dataset_path: str = DATASET_PATH,
    splits_dir: str = SPLITS_DIR,
    exclude_redacted: bool = True,
):
    # Carve validation out of Task 1.9's Training split only -- rf_fit and test
    # are never touched here, so this can't leak into either of them.
    train_full = load_split_df(seed, "train", dataset_path, splits_dir, exclude_redacted)
    label_col = resolve_label_column(train_full)
    train_df, val_df = train_test_split(
        train_full, test_size=val_fraction, stratify=train_full[label_col], random_state=seed,
    )
    return train_df, val_df, label_col


def main():
    parser = argparse.ArgumentParser(description="Grouped Training / RF-Fitting / Test split for every seed.")
    parser.add_argument("--input", default=DATASET_PATH, help="labeled SMS CSV")
    parser.add_argument("--features", default=LINK_FEATURES_PATH, help="output of build_sms_link_features")
    parser.add_argument("--output", default=SPLITS_DIR, help="writes seed_<s>/{train,rf_fit,test}_ids.txt")
    parser.add_argument("--report", default="experiments/splits/split_report.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    report = run(args.input, args.features, args.seeds, args.output)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    report.to_csv(args.report, index=False)
    pd.set_option("display.width", 200)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
