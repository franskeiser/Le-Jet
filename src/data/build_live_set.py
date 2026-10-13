"""Build the LIVE link set: links the lab really visited, with their real page / domain features.

Two sets come out of the same inspected links, and they must never mix (docs/training_rules.md rule 2,
docs/fresh_set_spec.md):

    live_train  rows the head may train on, so it meets page evidence (the "mixed" training of Phase C).
    fresh_test  the test for the web-inspection claim. It never enters training, gating or tuning.

How leakage is prevented (every point is checked by `check_leaks`, which the builder runs and the tests exercise):
  * the split is by DOMAIN GROUP: no group is in both sets. A group is the registrable domain of the link, or
    of the FINAL domain when the link is a shortener (otherwise all bit.ly links would land in one set);
  * fresh_test never holds a domain that appears in the public web set (the extractors train on it) or in the
    old SMS data (the CNN trains on it). Such rows are moved to live_train and flagged `overlap_moved`;
  * both classes have links in both sets, and in each class the test share is close to the requested one;
  * labels are ground truth by construction (team smishing links; generated legitimate links) and are NEVER
    changed because a page was dead or looked clean (plan section 6);
  * missing values are -1, never blank or 0; page_fetched is 1 only when the HTML really was parsed.

Inputs (all produced by the lab, validated on the host):
    <clean>/<batch>.clean.jsonl         the lab's validated rows (defanged URLs)
    <clean>/<batch>.model_input.csv     adds page_rank (scripts/lookup_page_rank.py)
Output: live_links.csv (+ live_report.json). No raw phishing URL is ever written: links are DEFANGED, and the
serving store looks a link up by the sha256 of its normalised form.

Usage:
    python -m src.data.build_live_set --batch smish03:smishing --batch benign02:legitimate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from src.features.model.rf_extractor import (
    EXTERNAL_FEATURES, PAGE_RANK, URL_TEXT_FEATURES, VARIANTS, WEBPAGE_FEATURES, registrable_domain,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN = Path(r"D:\LeJetData\clean")
DEFAULT_OUT = ROOT / "data" / "live"
WEB_SET = ROOT / "data" / "processed" / "web_phishing_clean.csv"
SMS_SET = ROOT / "data" / "processed" / "sms_v2_balanced.csv"

FULL_INPUTS = VARIANTS["full"]                      # 14 link-text + page_rank + 12 webpage + 4 external
LABELS = ("smishing", "legitimate")
KNOWN_SHORTENERS = {"bit.ly", "cutt.ly", "tinyurl.com", "t.co", "rb.gy", "is.gd", "ow.ly", "shorturl.at", "s.id",
                    "tiny.cc", "goo.gl", "rebrand.ly", "t.ly", "zip9.run", "naku.run", "luckyw.run"}

_FETCH_STATUS = {
    "none": "live", "http_4xx": "blocked", "http_5xx": "blocked", "challenge_page": "blocked",
    "parked_or_default_page": "blocked", "blocked_or_empty": "blocked", "non_html": "blocked", "too_large": "blocked",
    "read_timeout": "timeout", "connect_timeout": "timeout", "render_timeout": "timeout",
    "rejected_url": "not_attempted", "port_blocked": "not_attempted", "redirect_to_blocked": "not_attempted",
}   # everything else (dns_*, connect_refused, tls_error, too_many_redirects, ...) = "dead"


def refang(text: str) -> str:
    return str(text).replace("[.]", ".").replace("hxxp", "http")


def fetch_status(fail_reason: str, hop_count: int) -> str:
    if fail_reason == "none":
        return "redirected" if hop_count and hop_count > 0 else "live"
    return _FETCH_STATUS.get(fail_reason, "dead")


def link_key(url: str) -> str:
    """The serving store's lookup key: sha256 of the normalised link (no raw URL is stored)."""
    from src.features.link_features import normalize_link
    return hashlib.sha256(normalize_link(refang(url)).encode("utf-8")).hexdigest()


def group_of(row: Dict, final_domain: str) -> str:
    """Domain group for the split. A shortener is grouped by where it LEADS (its own domain is shared by every
    shortened link), a failed shortener by the link itself (nothing else is known about it)."""
    own = registrable_domain(refang(row["url"])).lower()
    if own in KNOWN_SHORTENERS:
        return final_domain if final_domain and final_domain != own else f"link:{row['url_id']}"
    return own


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_batch(clean_dir: Path, batch: str, label: str, source: str) -> List[Dict]:
    """The lab's rows of one batch as live-set records (without the split yet)."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    rows = read_jsonl(clean_dir / f"{batch}.clean.jsonl")
    ranks: Dict[str, int] = {}
    mi = clean_dir / f"{batch}.model_input.csv"
    if mi.exists():
        frame = pd.read_csv(mi, dtype={"url_id": str})
        for _, r in frame.iterrows():
            if int(r.get("page_rank_ok", 0)) == 1:
                ranks[str(r["url_id"])] = int(r["page_rank"])
    out: List[Dict] = []
    for r in rows:
        feats = r["features"]
        fetched = int(bool(r["page_fetched"]))
        rec: Dict = {
            "link_id": r["url_id"], "label": label, "source": source, "batch_id": batch,
            "link": r["url"], "has_url": 1, "page_fetched": fetched,
            "fetch_status": fetch_status(r["fail_reason"], r.get("hop_count", 0)),
            "fail_reason": r["fail_reason"], "final_url": r["final_url"] if fetched else "",
            "final_domain": r["final_reg_domain"] if fetched else "",
            "n_redirects": r.get("hop_count", -1) if fetched else -1,
            "visited_at": r["fetch_ts"], "free_hosting": int(r.get("free_hosting_subdomain", 0)),
            "hosting_platform": r.get("hosting_platform", ""), "key": link_key(r["url"]),
        }
        for name in URL_TEXT_FEATURES:
            rec[name] = feats.get(name, -1)
        for name in WEBPAGE_FEATURES + EXTERNAL_FEATURES:
            rec[name] = feats.get(name, -1) if fetched or name in EXTERNAL_FEATURES else -1
        rec[PAGE_RANK] = ranks.get(r["url_id"], -1)
        if not fetched:                                     # spec: page not fetched => the 12 webpage columns are -1
            for name in WEBPAGE_FEATURES:
                rec[name] = -1
        rec["shortener"] = int(registrable_domain(refang(r["url"])).lower() in KNOWN_SHORTENERS
                               or bool(feats.get("shortened_url_indicator", 0) == 1))
        rec["group"] = group_of(r, refang(r["final_reg_domain"]).lower() if fetched else "")
        out.append(rec)
    return out


def _safe_domain(url: str) -> str:
    """registrable_domain, but a malformed address in real SMS text (e.g. a stray "[") gives "" instead of an error."""
    try:
        return registrable_domain(url).lower()
    except ValueError:
        return ""


def known_domains(web_csv: Path = WEB_SET, sms_csv: Path = SMS_SET) -> Set[str]:
    """Registrable domains the extractors (web set) and the CNN (SMS data) have already seen."""
    seen: Set[str] = set()
    if web_csv.exists():
        for url in pd.read_csv(web_csv, usecols=["url"])["url"]:
            seen.add(_safe_domain(str(url)))
    if sms_csv.exists():
        from src.features.link_features import extract_links
        header = pd.read_csv(sms_csv, nrows=0).columns
        column = next(c for c in ("sms_text", "sms") if c in header)       # the raw file calls it "sms"
        for text in pd.read_csv(sms_csv, usecols=[column], keep_default_na=False)[column]:
            for link in extract_links(str(text)):
                seen.add(_safe_domain(link["raw"]))
    seen.discard("")
    return seen


def assign_split(records: List[Dict], seed: int = 42, test_share: float = 0.4,
                 blocked_domains: Optional[Set[str]] = None) -> List[Dict]:
    """Add `split` ('live_train' / 'fresh_test') and `overlap_moved`. Groups are shuffled per class with a fixed
    seed and handed to fresh_test until its share of that class is reached; a group that holds both classes, or
    whose domain the models have already seen, goes to live_train (never to the test set)."""
    blocked_domains = blocked_domains or set()
    by_group: Dict[str, List[Dict]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)
    forced_train: Set[str] = set()
    for g, rows in by_group.items():
        labels = {r["label"] for r in rows}
        domains = {registrable_domain(refang(r["link"])).lower() for r in rows} | \
                  {refang(r["final_domain"]).lower() for r in rows if r["final_domain"]}
        if len(labels) > 1 or (domains & blocked_domains) or g in blocked_domains:
            forced_train.add(g)
    rng = random.Random(seed)
    test_groups: Set[str] = set()
    for label in LABELS:
        groups = sorted(g for g, rows in by_group.items() if rows[0]["label"] == label and g not in forced_train)
        rng.shuffle(groups)
        total = sum(len(by_group[g]) for g in groups) + sum(len(by_group[g]) for g in forced_train
                                                           if by_group[g][0]["label"] == label)
        target, got = test_share * total, 0
        for g in groups:
            if got >= target:
                break
            test_groups.add(g)
            got += len(by_group[g])
    for r in records:
        r["split"] = "fresh_test" if r["group"] in test_groups else "live_train"
        r["overlap_moved"] = int(r["group"] in forced_train)
    return records


def check_leaks(records: Sequence[Dict], blocked_domains: Optional[Set[str]] = None,
                test_share: float = 0.4) -> List[str]:
    """Every rule above as a check. Returns a list of problems (empty = clean)."""
    blocked_domains = blocked_domains or set()
    problems: List[str] = []
    ids = [r["link_id"] for r in records]
    if len(ids) != len(set(ids)):
        problems.append("duplicate link_id")
    if len({r["key"] for r in records}) != len(records):
        problems.append("the same link appears twice (same normalised form)")
    split_of: Dict[str, Set[str]] = {}
    for r in records:
        split_of.setdefault(r["group"], set()).add(r["split"])
    both = [g for g, s in split_of.items() if len(s) > 1]
    if both:
        problems.append(f"{len(both)} domain group(s) in both sets")
    for r in records:
        if r["split"] == "fresh_test":
            doms = {registrable_domain(refang(r["link"])).lower(), refang(r["final_domain"]).lower()} - {""}
            if doms & blocked_domains:
                problems.append("a fresh_test domain is already known to the extractors or the CNN")
                break
    for split in ("live_train", "fresh_test"):
        for label in LABELS:
            if not any(r["split"] == split and r["label"] == label and r["has_url"] == 1 for r in records):
                problems.append(f"{split} has no {label} link")
    for r in records:
        if r["label"] not in LABELS:
            problems.append("bad label")
            break
        if r["page_fetched"] == 1 and not (r["final_url"] and r["fetch_status"] in ("live", "redirected")):
            problems.append("page_fetched=1 without a final_url / live status")
            break
        if r["page_fetched"] == 0 and any(r[c] != -1 for c in WEBPAGE_FEATURES):
            problems.append("page not fetched but a webpage feature is not -1")
            break
        if any(pd.isna(r[c]) or r[c] == "" for c in FULL_INPUTS):
            problems.append("a feature is blank or NaN (use -1)")
            break
    for label in LABELS:
        total = [r for r in records if r["label"] == label]
        share = sum(1 for r in total if r["split"] == "fresh_test") / max(1, len(total))
        if total and abs(share - test_share) > 0.2:
            problems.append(f"{label}: test share {share:.2f} is far from the requested {test_share}")
    return problems


def build(batches: Sequence[Tuple[str, str, str]], clean_dir: Path = DEFAULT_CLEAN, seed: int = 42,
          test_share: float = 0.4, blocked: Optional[Set[str]] = None) -> Tuple[pd.DataFrame, Dict]:
    records: List[Dict] = []
    for batch, label, source in batches:
        records += load_batch(clean_dir, batch, label, source)
    # the same link seen in two batches counts once (first batch wins)
    seen_keys: Set[str] = set()
    unique = []
    for r in records:
        if r["key"] not in seen_keys:
            seen_keys.add(r["key"])
            unique.append(r)
    blocked = known_domains() if blocked is None else blocked
    unique = assign_split(unique, seed=seed, test_share=test_share, blocked_domains=blocked)
    problems = check_leaks(unique, blocked, test_share)
    frame = pd.DataFrame(unique)
    report = {
        "rows": len(frame), "problems": problems,
        "by_split_label": {f"{s}/{l}": int(((frame.split == s) & (frame.label == l)).sum())
                           for s in ("live_train", "fresh_test") for l in LABELS},
        "page_fetched_by_split_label": {f"{s}/{l}": int(((frame.split == s) & (frame.label == l)
                                                         & (frame.page_fetched == 1)).sum())
                                        for s in ("live_train", "fresh_test") for l in LABELS},
        "overlap_moved": int(frame["overlap_moved"].sum()), "seed": seed, "test_share": test_share,
        "shortener_share_by_label": {l: round(float(frame[frame.label == l]["shortener"].mean()), 3) for l in LABELS},
    }
    return frame, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", action="append", required=True, metavar="NAME:LABEL[:SOURCE]",
                    help="a validated lab batch and its ground-truth label (smishing / legitimate)")
    ap.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-share", type=float, default=0.4)
    args = ap.parse_args()
    batches = []
    for spec in args.batch:
        parts = spec.split(":")
        batches.append((parts[0], parts[1], parts[2] if len(parts) > 2 else parts[0]))
    frame, report = build(batches, args.clean_dir, args.seed, args.test_share)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "live_links.csv", index=False)
    (args.out / "live_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["problems"]:
        raise SystemExit("LEAK CHECK FAILED: " + "; ".join(report["problems"]))
    print(f"wrote {args.out / 'live_links.csv'}")


if __name__ == "__main__":
    main()
