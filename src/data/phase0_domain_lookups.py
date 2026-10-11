"""Phase 0: find the links in the SMS set and measure how often domain lookups fail.

Plan: data/dataset_and_training_plan.md section 3. The measured failure rate sets how many
training rows the RF extractor should train with WHOIS/DNS masked.

No page is ever fetched. The only network contact is:
    DNS  -- Cloudflare DNS-over-HTTPS (1.1.1.1, a non-filtering resolver; plan section 8)
    RDAP -- the domain's registry, found through the IANA bootstrap file
Neither request reaches the scam site itself.

Usage (from the project root):
    python -m src.data.phase0_domain_lookups            # extract + sample + look up
    python -m src.data.phase0_domain_lookups --per-class 200 --seed 42
"""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.features.link_features import SHORTENERS, extract_links

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMS = ROOT / "data" / "raw" / "sms" / "v2_2026-09-15.csv"
LINKS_OUT = ROOT / "data" / "interim" / "sms_links.csv"
LOOKUPS_OUT = ROOT / "data" / "interim" / "phase0_domain_lookups.csv"
REPORT_DIR = ROOT / "experiments" / "phase0"

DOH_URL = "https://cloudflare-dns.com/dns-query"
IANA_BOOTSTRAP = "https://data.iana.org/rdap/dns.json"
USER_AGENT = "Le-Jet thesis research (Phase 0 domain liveness)"
TIMEOUT = 20
PAUSE = 0.5  # seconds between lookups, to stay well under registry rate limits

def build_links(sms_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sms = pd.read_csv(sms_path)
    rows = []
    for idx, (text, label) in enumerate(zip(sms["sms"].astype(str), sms["label"])):
        for link in extract_links(text):
            rows.append({"message_idx": idx, "label": label, **link})
    links = pd.DataFrame(rows, columns=["message_idx", "label", "raw", "host", "domain"])
    links["shortener"] = links["domain"].isin(SHORTENERS).astype(int)
    sms["has_placeholder"] = sms["sms"].astype(str).str.contains("<URL>", regex=False)
    return sms, links


def load_rdap_bootstrap(session: requests.Session) -> dict[str, str]:
    services = session.get(IANA_BOOTSTRAP, timeout=TIMEOUT).json()["services"]
    return {tld: urls[0] for tlds, urls in services for tld in tlds}


def dns_lookup(session: requests.Session, domain: str) -> str:
    # Plan section 13: dns_record = 1 if A/AAAA/NS records exist, 0 on NXDOMAIN or no
    # records, -1 on resolver error. Returned here as a status word.
    for rtype in ("A", "AAAA", "NS"):
        try:
            r = session.get(DOH_URL, params={"name": domain, "type": rtype},
                            headers={"accept": "application/dns-json"}, timeout=TIMEOUT)
            data = r.json()
        except (requests.RequestException, ValueError):
            return "error"
        if data.get("Status") == 3:
            return "nxdomain"
        if data.get("Status") != 0:
            return "error"
        if any(a.get("type") in (1, 28, 2) for a in data.get("Answer", [])):
            return "resolves"
    return "no_records"


def rdap_lookup(session: requests.Session, bootstrap: dict[str, str], domain: str) -> dict:
    tld = domain.rsplit(".", 1)[-1]
    base = bootstrap.get(tld)
    if base is None:
        return {"rdap_status": "no_rdap_for_tld", "registered": None}
    url = f"{base.rstrip('/')}/domain/{domain}"
    for attempt in range(3):
        try:
            r = session.get(url, timeout=TIMEOUT, headers={"accept": "application/rdap+json"})
        except requests.RequestException:
            return {"rdap_status": "error", "registered": None}
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        break
    if r.status_code == 404:
        return {"rdap_status": "not_found", "registered": None}
    if r.status_code != 200:
        return {"rdap_status": f"http_{r.status_code}", "registered": None}
    try:
        events = r.json().get("events", [])
    except ValueError:
        return {"rdap_status": "error", "registered": None}
    registered = next((e.get("eventDate") for e in events if e.get("eventAction") == "registration"), None)
    return {"rdap_status": "ok" if registered else "no_registration_date", "registered": registered}


def sample_domains(links: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    # Unique registrable domains per class. A domain seen in both classes is kept in both.
    unique = links.drop_duplicates(["label", "domain"])[["label", "domain", "shortener"]]
    shuffled = unique.sample(frac=1, random_state=seed)
    return shuffled.groupby("label").head(per_class).reset_index(drop=True)


def run_lookups(sample: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    # Resumable: domains already in out_path are not looked up again.
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["domain"])
    cached = set(done["domain"])
    session = requests.Session()
    session.headers["user-agent"] = USER_AGENT
    bootstrap = load_rdap_bootstrap(session)

    todo = sample[~sample["domain"].isin(cached)].drop_duplicates("domain")
    rows = []
    for i, domain in enumerate(todo["domain"], 1):
        rows.append({
            "domain": domain, "dns_status": dns_lookup(session, domain),
            **rdap_lookup(session, bootstrap, domain),
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} looked up")
            pd.concat([done, pd.DataFrame(rows)]).to_csv(out_path, index=False)
        time.sleep(PAUSE)
    results = pd.concat([done, pd.DataFrame(rows)]) if rows else done
    return sample.merge(results, on="domain", how="left")


def summarize(sms: pd.DataFrame, links: pd.DataFrame, looked_up: pd.DataFrame) -> str:
    lines = ["Phase 0: SMS links and domain lookups", f"Generated {datetime.now():%Y-%m-%d %H:%M}", ""]

    per_class = pd.DataFrame({
        "messages": sms.groupby("label").size(),
        "with_link": links.groupby("label")["message_idx"].nunique(),
        "with_<URL>_placeholder": sms.groupby("label")["has_placeholder"].sum(),
        "unique_domains": links.groupby("label")["domain"].nunique(),
        "shortener_links": links.groupby("label")["shortener"].sum(),
    }).fillna(0).astype(int)
    per_class["share_with_link"] = (per_class["with_link"] / per_class["messages"]).round(3)
    lines += ["1. Links per class", per_class.to_string(), ""]

    lu = looked_up.copy()
    lu["dns_ok"] = lu["dns_status"] == "resolves"
    lu["rdap_ok"] = lu["rdap_status"] == "ok"
    # Domain features usable = the domain resolves AND RDAP gave a registration date
    # (domain_age and domain_registration_length both need it).
    lu["domain_features_ok"] = lu["dns_ok"] & lu["rdap_ok"]
    rates = lu.groupby("label").agg(
        sampled=("domain", "size"), dns_resolves=("dns_ok", "mean"),
        rdap_ok=("rdap_ok", "mean"), domain_features_ok=("domain_features_ok", "mean"),
    )
    rates.loc["all"] = [len(lu), lu["dns_ok"].mean(), lu["rdap_ok"].mean(), lu["domain_features_ok"].mean()]
    rates["missing_rate_X"] = 1 - rates["domain_features_ok"]
    lines += ["2. Lookup success on sampled domains (share of domains)", rates.round(3).to_string(), ""]

    lines += ["3. DNS status counts", pd.crosstab(lu["dns_status"], lu["label"]).to_string(), ""]
    lines += ["4. RDAP status counts", pd.crosstab(lu["rdap_status"], lu["label"]).to_string(), ""]

    no_short = lu[lu["shortener"] == 0]
    lines += [
        "5. Excluding shorteners (their lookup describes the shortener, not the destination)",
        no_short.groupby("label")["domain_features_ok"].mean().rename("domain_features_ok").round(3).to_string(),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sms", type=Path, default=DEFAULT_SMS)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sms, links = build_links(args.sms)
    LINKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    links.to_csv(LINKS_OUT, index=False)
    print(f"{len(links)} links in {links['message_idx'].nunique()} messages -> {LINKS_OUT}")

    sample = sample_domains(links, args.per_class, args.seed)
    print(f"Looking up {sample['domain'].nunique()} unique domains ...")
    looked_up = run_lookups(sample, LOOKUPS_OUT)
    looked_up.to_csv(REPORT_DIR / "sampled_domains.csv", index=False)

    report = summarize(sms, links, looked_up)
    (REPORT_DIR / "summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
