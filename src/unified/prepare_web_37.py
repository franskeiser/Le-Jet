"""Attach the final 37 link/page features to the web set, keeping the ids, domains and splits of the 31-feature run.

The 37 features (docs/progress.md, section 3) come from data/raw/web-phishing/{train-1,test}.csv under their raw
column names. data/processed/web_phishing_linked.csv already has message ids, domains and labels, and
data/splits_web_phishing/ holds the domain-grouped splits built on those ids. Rows are matched by URL (the one URL
that appears twice is matched by order of appearance and checked by label), so every message keeps its id and
therefore its split. That makes the 31 vs 37 comparison use identical train, gate and test rows.

Lookup failures (negative domain_age / domain_registration_length) are set to the single sentinel -1, as in
docs/web_phishing_cleaning.md, decision 8.

Writes data/processed/web_phishing_37_linked.csv. No new split files are needed.

Usage:
    python -m src.unified.prepare_web_37
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "web-phishing"
LINKED = ROOT / "data" / "processed" / "web_phishing_linked.csv"
OUT = ROOT / "data" / "processed" / "web_phishing_37_linked.csv"

URL_SHAPE = ["length_url", "length_hostname", "nb_dots", "nb_slash", "nb_www", "https_token", "ratio_digits_url",
             "ratio_digits_host", "prefix_suffix", "shortening_service", "nb_redirection"]
WORDS = ["length_words_raw", "char_repeat", "shortest_word_host", "shortest_word_path", "longest_words_raw",
         "longest_word_host", "avg_word_host", "phish_hints", "domain_in_brand"]
PAGE = ["nb_hyperlinks", "ratio_intHyperlinks", "nb_extCSS", "login_form", "external_favicon", "links_in_tags",
        "ratio_intMedia", "ratio_extMedia", "iframe", "safe_anchor", "empty_title", "domain_in_title",
        "domain_with_copyright"]
OUTSIDE = ["whois_registered_domain", "domain_registration_length", "domain_age", "page_rank"]

URL_FEATURES = URL_SHAPE + WORDS            # 20: read from the link
PAGE_FEATURES = PAGE + OUTSIDE              # 17: need the page or an outside service; blanked when the page is not reached
FEATURES_37 = URL_FEATURES + PAGE_FEATURES
KEEP = ["message_id", "url", "label", "link_status", "link_domain", "source", "template_id"]


def main() -> None:
    assert len(FEATURES_37) == 37 and len(set(FEATURES_37)) == 37
    raw = pd.concat([pd.read_csv(RAW / "train-1.csv", low_memory=False), pd.read_csv(RAW / "test.csv", low_memory=False)],
                    ignore_index=True)
    linked = pd.read_csv(LINKED, low_memory=False, keep_default_na=False, na_values=[""])
    linked["occ"] = linked.groupby("url").cumcount()
    raw["occ"] = raw.groupby("url").cumcount()
    table = linked[KEEP + ["occ"]].merge(raw[["url", "occ", "status"] + FEATURES_37], on=["url", "occ"], how="left", validate="1:1")
    assert table[FEATURES_37].notna().all().all(), "some linked rows found no match in the raw files"
    want = table["label"].map({"smishing": "phishing", "legitimate": "legitimate"})
    assert (want == table["status"]).all(), "labels disagree between the linked file and the raw files"
    for col in ("domain_age", "domain_registration_length"):
        table.loc[table[col] < 0, col] = -1
    table = table.drop(columns=["occ", "status"])
    table.to_csv(OUT, index=False)
    print(f"{OUT}: {len(table)} rows, {len(URL_FEATURES)} URL features + {len(PAGE_FEATURES)} page/domain features")


if __name__ == "__main__":
    main()
