"""Audit and conservatively deduplicate manually labeled SMS templates."""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "annotation" / "review_pool_label_here_v2.csv"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "data" / "labeled" / "audit"
VALID_LABELS = {"smishing", "legitimate"}

# Typed placeholders keep their semantic class. Unknown placeholders keep a
# normalized form of their original name instead of becoming interchangeable.
PLACEHOLDER_PATTERN = re.compile(r"<\s*([^<>\r\n]{1,80}?)\s*>")
PLACEHOLDER_ALIASES = {
    "date_time": "datetime",
    "datetime": "datetime",
    "date": "datetime",
    "time": "datetime",
    "phone_number": "phone",
    "mobile_number": "phone",
    "tel_number": "phone",
    "telephone_number": "phone",
    "phone": "phone",
    "mobile": "phone",
    "tel": "phone",
    "telephone": "phone",
    "url": "url",
    "link": "url",
    "website": "url",
    "domain": "url",
    "email": "email",
    "email_address": "email",
    "e_mail": "email",
    "e_mail_address": "email",
    "amount": "amount",
    "balance": "amount",
    "price": "amount",
    "money": "amount",
    "currency": "amount",
    "account": "account_id",
    "account_id": "account_id",
    "account_number": "account_id",
    "account_no": "account_id",
    "acct": "account_id",
    "acct_id": "account_id",
    "acct_number": "account_id",
    "acct_no": "account_id",
    "reference": "reference_id",
    "reference_id": "reference_id",
    "reference_number": "reference_id",
    "reference_no": "reference_id",
    "ref": "reference_id",
    "ref_id": "reference_id",
    "ref_number": "reference_id",
    "ref_no": "reference_id",
    "tracking": "tracking_id",
    "tracking_id": "tracking_id",
    "tracking_number": "tracking_id",
    "tracking_no": "tracking_id",
    "transaction": "transaction_id",
    "transaction_id": "transaction_id",
    "transaction_number": "transaction_id",
    "transaction_no": "transaction_id",
    "txn": "transaction_id",
    "txn_id": "transaction_id",
    "txn_number": "transaction_id",
    "txn_no": "transaction_id",
    "confirmation": "confirmation_id",
    "confirmation_id": "confirmation_id",
    "confirmation_number": "confirmation_id",
    "confirmation_no": "confirmation_id",
    "order": "order_id",
    "order_id": "order_id",
    "order_number": "order_id",
    "order_no": "order_id",
}

# Ordered from specific formats to broad identifiers so real variable campaign
# values become stable typed placeholders without changing the original SMS.
VARIABLE_PATTERNS = (
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I), " <email> "),
    (re.compile(r"\b(?:https?://|www\.)\S+", re.I), " <url> "),
    (re.compile(r"(?<![@\w])(?:[a-z0-9-]+\.)+(?:com|net|org|ph|info|biz|co|io|me|ly|app|site|online|xyz|top|click|link)(?:/\S*)?", re.I), " <url> "),
    (re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)"), " <phone> "),
    (re.compile(r"(?<!\w)(?:php|p|usd|\$|\u20b1|eur|gbp)\s*\d[\d,]*(?:\.\d{1,2})?(?!\w)", re.I), " <amount> "),
    (re.compile(r"(?<!\w)\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})(?!\w)"), " <datetime> "),
    (re.compile(r"(?<!\w)\d{1,2}:\d{2}(?:\s?[ap]m)?(?!\w)", re.I), " <datetime> "),
    (re.compile(r"\b(?:account|acct)\s*(?:no\.?|number|id|#|:|-)\s*[a-z0-9-]{4,}\b", re.I), " <account_id> "),
    (re.compile(r"\b(?:reference|ref)\s*(?:no\.?|number|id|#|:|-)\s*[a-z0-9-]{4,}\b", re.I), " <reference_id> "),
    (re.compile(r"\btracking\s*(?:no\.?|number|id|#|:|-)\s*[a-z0-9-]{4,}\b", re.I), " <tracking_id> "),
    (re.compile(r"\b(?:transaction|txn)\s*(?:no\.?|number|id|#|:|-)\s*[a-z0-9-]{4,}\b", re.I), " <transaction_id> "),
    (re.compile(r"\bconfirmation\s*(?:no\.?|number|id|#|:|-)\s*[a-z0-9-]{4,}\b", re.I), " <confirmation_id> "),
    (re.compile(r"\border\s*(?:no\.?|number|id|#|:|-)\s*[a-z0-9-]{4,}\b", re.I), " <order_id> "),
    (re.compile(r"\b(?=[a-z0-9-]{8,}\b)(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]+\b", re.I), " <identifier> "),
    (re.compile(r"(?<!\w)\d{5,}(?:[.,]\d+)?(?!\w)"), " <number> "),
)


def normalized_value(value: object) -> str:
    """Normalize an annotation value for eligibility checks."""
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def normalize_placeholder(match: re.Match[str]) -> str:
    """Canonicalize a typed placeholder while preserving unknown identities."""
    name = unicodedata.normalize("NFKC", match.group(1)).casefold().strip()
    name = re.sub(r"[\s-]+", "_", name)
    name = re.sub(r"[^\w]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        return " <unknown_placeholder> "
    canonical = PLACEHOLDER_ALIASES.get(name)
    if canonical is None:
        numbered_base = re.sub(r"_\d+$", "", name)
        canonical = PLACEHOLDER_ALIASES.get(numbered_base, name)
    return f" <{canonical}> "


def normalize_template(text: object) -> str:
    """Return a conservative comparison-only SMS template."""
    value = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    value = PLACEHOLDER_PATTERN.sub(normalize_placeholder, value)
    for pattern, replacement in VARIABLE_PATTERNS:
        value = pattern.sub(replacement, value)
    # Treat punctuation and layout as superficial while preserving placeholder words.
    value = re.sub(r"[^\w<>]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def run_normalization_assertions() -> None:
    """Guard the typed-placeholder behavior before processing annotation data."""
    original = "Call <PHONE_NUMBER>"
    assert normalize_template(original) == normalize_template("Call +639171234567")
    assert normalize_template("Expires <DATE_TIME>") != normalize_template(
        "Expires <PHONE_NUMBER>"
    )
    assert normalize_template("<DATE_TIME>") != normalize_template("<PHONE_NUMBER>")
    assert normalize_template("<US_BANK_NUMBER>") != normalize_template("<NAMED_ENTITY>")
    variants = {"<Some Field>", "<SOME-FIELD>", "<some_field>"}
    assert {normalize_template(value) for value in variants} == {"<some_field>"}
    typed_variants = {"<PHONE NUMBER>", "<phone-number>", "<PHONE_NUMBER_1>"}
    assert {normalize_template(value) for value in typed_variants} == {"<phone>"}
    assert original == "Call <PHONE_NUMBER>"


def cluster_id(template: str) -> str:
    """Create a stable identifier from the complete normalized template."""
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
    return f"template_{digest}"


def clarity_key(row: pd.Series) -> tuple[int, int, int, int, str]:
    """Rank representatives deterministically without using source labels.

    Prefer nonempty text, fewer corruption markers, fewer redaction placeholders,
    and more readable alphanumeric content. Ties use stable input order/message ID.
    """
    text = str(row["text"])
    compact = re.sub(r"\s+", " ", text).strip()
    empty_penalty = int(not compact or not re.search(r"\w", compact, re.UNICODE))
    corruption_penalty = sum(compact.count(mark) for mark in ("\ufffd", "Ã", "Â", "â€", "ðŸ"))
    placeholder_penalty = len(re.findall(r"<[^>]+>", compact))
    readable_chars = len(re.findall(r"\w", compact, re.UNICODE))
    return (
        empty_penalty,
        corruption_penalty,
        placeholder_penalty,
        -readable_chars,
        f"{int(row['_input_order']):012d}:{row['message_id']}",
    )


def load_eligible(path: Path) -> tuple[pd.DataFrame, int]:
    """Load the annotation source and retain only final-model-eligible rows."""
    # The Excel-maintained worksheet contains mixed legacy bytes. Latin-1 maps
    # every byte losslessly and matches the existing v1 export behavior.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
    required = {"message_id", "source", "text", "human_label", "review_status", "subcategory"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if frame["message_id"].duplicated().any():
        duplicates = frame.loc[frame["message_id"].duplicated(False), "message_id"].unique()
        raise ValueError(f"message_id must be unique; duplicates include {duplicates[:5].tolist()}")

    labels = frame["human_label"].map(normalized_value)
    statuses = frame["review_status"].map(normalized_value)
    eligible = frame.loc[statuses.eq("reviewed") & labels.isin(VALID_LABELS)].copy()
    eligible["human_label"] = labels.loc[eligible.index]
    eligible["review_status"] = statuses.loc[eligible.index]
    eligible["_input_order"] = range(len(eligible))
    return eligible, len(frame)


def choose_representatives(eligible: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    """Generate clusters and select one deterministic representative per label."""
    work = eligible.copy()
    work["normalized_template"] = work["text"].map(normalize_template)
    # Punctuation-only SMS cannot be compared safely, so each remains its own
    # singleton instead of being collapsed into one artificial empty cluster.
    empty_mask = work["normalized_template"].eq("")
    work.loc[empty_mask, "normalized_template"] = work.loc[empty_mask, "message_id"].map(
        lambda message_id: f"<empty_{message_id}>"
    )
    work["cluster_id"] = work["normalized_template"].map(cluster_id)
    collisions = work.groupby("cluster_id")["normalized_template"].nunique()
    if collisions.gt(1).any():
        raise RuntimeError("A cluster ID hash collision occurred")

    representatives: dict[tuple[str, str], str] = {}
    for (template_cluster, label), group in work.groupby(
        ["cluster_id", "human_label"], sort=True
    ):
        representative = min((row for _, row in group.iterrows()), key=clarity_key)
        representatives[(template_cluster, label)] = str(representative["message_id"])
    return work, representatives


def load_resolutions(path: Path, conflicts: set[str], work: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Validate explicit human decisions for every cross-label cluster."""
    decisions = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"cluster_id", "resolved_label", "representative_message_id"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"Resolution file is missing required columns: {missing}")
    if decisions["cluster_id"].duplicated().any():
        raise ValueError("Resolution file contains duplicate cluster_id values")

    decisions["resolved_label"] = decisions["resolved_label"].map(normalized_value)
    supplied = set(decisions["cluster_id"])
    if supplied != conflicts:
        missing_ids = sorted(conflicts - supplied)
        extra_ids = sorted(supplied - conflicts)
        raise ValueError(f"Resolutions must match all conflicts; missing={missing_ids}, extra={extra_ids}")

    result: dict[str, tuple[str, str]] = {}
    for row in decisions.itertuples(index=False):
        if row.resolved_label not in VALID_LABELS:
            raise ValueError(f"Invalid resolved label for {row.cluster_id}: {row.resolved_label}")
        members = work.loc[work["cluster_id"].eq(row.cluster_id), "message_id"]
        if row.representative_message_id not in set(members):
            raise ValueError(
                f"Representative {row.representative_message_id} is not in {row.cluster_id}"
            )
        result[row.cluster_id] = (row.resolved_label, row.representative_message_id)
    return result


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically so a failed run cannot leave a partial audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def build_audits(
    work: pd.DataFrame,
    representatives: dict[tuple[str, str], str],
    audit_dir: Path,
    resolutions: dict[str, tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Write the complete traceability map and all cross-label conflicts."""
    sizes = work.groupby("cluster_id")["message_id"].transform("size")
    label_counts = work.groupby("cluster_id")["human_label"].transform("nunique")
    conflict_mask = label_counts.gt(1)
    conflict_ids = set(work.loc[conflict_mask, "cluster_id"])

    map_rows = []
    for row, size, is_conflict in zip(work.itertuples(index=False), sizes, conflict_mask):
        if is_conflict and not resolutions:
            # This is only an interim representative within the existing label;
            # it does not resolve which label the cross-label cluster should use.
            representative_id = representatives[(row.cluster_id, row.human_label)]
            representative_scope = "same_label_interim"
        elif is_conflict:
            representative_id = resolutions[row.cluster_id][1]
            representative_scope = "resolved_cluster"
        else:
            representative_id = representatives[(row.cluster_id, row.human_label)]
            representative_scope = "resolved_cluster"
        map_rows.append(
            {
                "cluster_id": row.cluster_id,
                "member_message_id": row.message_id,
                "representative_message_id": representative_id,
                "human_label": row.human_label,
                "cluster_size": int(size),
                "is_representative": bool(representative_id and row.message_id == representative_id),
                "normalized_template": row.normalized_template,
                "source": row.source,
                "cluster_has_label_conflict": bool(is_conflict),
                "representative_scope": representative_scope,
            }
        )
    cluster_map = pd.DataFrame(map_rows)

    conflicts = work.loc[
        conflict_mask,
        [
            "cluster_id", "message_id", "source", "text", "human_label",
            "review_status", "subcategory", "normalized_template",
        ],
    ].copy()
    conflicts["cluster_size"] = sizes.loc[conflicts.index].astype(int)
    conflicts = conflicts.sort_values(["cluster_size", "cluster_id", "message_id"], ascending=[False, True, True])

    write_csv(cluster_map, audit_dir / "template_cluster_map.csv")
    write_csv(conflicts, audit_dir / "template_conflicts.csv")
    return cluster_map, conflicts, conflict_ids


def export_resolved_dataset(
    work: pd.DataFrame,
    representatives: dict[tuple[str, str], str],
    conflicts: set[str],
    resolutions: dict[str, tuple[str, str]],
    output: Path,
) -> None:
    """Export v2 only after every cross-label cluster has an explicit decision."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing labeled dataset: {output}")
    selected: list[dict[str, str]] = []
    for cluster, group in work.groupby("cluster_id", sort=True):
        if cluster in conflicts:
            label, message_id = resolutions[cluster]
        else:
            label = str(group["human_label"].iloc[0])
            message_id = representatives[(cluster, label)]
        row = group.loc[group["message_id"].eq(message_id)].iloc[0]
        selected.append({"sms": row["text"], "label": label})
    result = pd.DataFrame(selected, columns=["sms", "label"])
    if set(result["label"]) - VALID_LABELS:
        raise RuntimeError("Final output contains an invalid label")
    write_csv(result, output)


def report_statistics(work: pd.DataFrame, total_rows: int, conflicts: pd.DataFrame) -> None:
    """Print validation counts and examples without creating an interim dataset."""
    cluster_sizes = work.groupby("cluster_id").size()
    by_label_templates = work.groupby("human_label")["normalized_template"].nunique()
    cross_ids = set(conflicts["cluster_id"])

    print(f"Total annotation rows: {total_rows:,}")
    print(f"Eligible reviewed rows: {len(work):,}")
    print(f"Smishing eligible rows: {(work['human_label'] == 'smishing').sum():,}")
    print(f"Legitimate eligible rows: {(work['human_label'] == 'legitimate').sum():,}")
    print(f"Template clusters: {len(cluster_sizes):,}")
    print(f"Singleton clusters: {(cluster_sizes == 1).sum():,}")
    print(f"Multi-row template clusters: {(cluster_sizes > 1).sum():,}")
    print(f"Rows in multi-row clusters: {cluster_sizes[cluster_sizes > 1].sum():,}")
    print(f"Rows removed by same-label collapsing: {len(work) - int(by_label_templates.sum()):,}")
    print(f"Projected smishing count: {int(by_label_templates.get('smishing', 0)):,}")
    print(f"Projected legitimate count: {int(by_label_templates.get('legitimate', 0)):,}")
    print(f"Projected total: {int(by_label_templates.sum()):,}")
    print(f"Cross-label clusters: {len(cross_ids):,}")
    print(f"Rows in cross-label clusters: {len(conflicts):,}")
    print(f"Smishing members in conflicts: {(conflicts['human_label'] == 'smishing').sum():,}")
    print(f"Legitimate members in conflicts: {(conflicts['human_label'] == 'legitimate').sum():,}")
    print("Largest template clusters:")
    for cluster, size in cluster_sizes.sort_values(ascending=False).head(10).items():
        examples = work.loc[work["cluster_id"].eq(cluster), "text"].head(2).tolist()
        print(f"  {cluster} ({size} rows): {examples}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--resolutions",
        type=Path,
        help="CSV with cluster_id,resolved_label,representative_message_id for every conflict",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Final v2 path; defaults to data/labeled/v2_YYYY-MM-DD.csv with resolutions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_normalization_assertions()
    input_path = args.input.resolve()
    audit_dir = args.audit_dir.resolve()
    work, total_rows = load_eligible(input_path)
    work, representatives = choose_representatives(work)

    # Detect agreement before selecting across labels; conflicts remain unresolved by default.
    conflict_ids = set(
        work.groupby("cluster_id")["human_label"].nunique().loc[lambda values: values.gt(1)].index
    )
    resolutions = None
    if args.resolutions:
        resolutions = load_resolutions(args.resolutions.resolve(), conflict_ids, work)

    cluster_map, conflicts, conflict_ids = build_audits(
        work, representatives, audit_dir, resolutions
    )
    if len(cluster_map) != len(work) or cluster_map["member_message_id"].nunique() != len(work):
        raise RuntimeError("Audit map is incomplete")
    report_statistics(work, total_rows, conflicts)

    if conflict_ids and resolutions is None:
        print("Final v2 intentionally withheld: cross-label clusters require manual adjudication.")
        return
    if args.output and resolutions is None and conflict_ids:
        raise ValueError("--output requires a complete --resolutions file while conflicts exist")

    if resolutions is not None or not conflict_ids:
        output = (args.output or (PROJECT_ROOT / "data" / "labeled" / f"v2_{date.today().isoformat()}.csv")).resolve()
        export_resolved_dataset(work, representatives, conflict_ids, resolutions or {}, output)
        print(f"Final v2 created: {output}")


if __name__ == "__main__":
    main()
