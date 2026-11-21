r"""Step 2 of the experimental testing guide: preprocess the SMS text and write the URL-feature files, from the frozen-candidate final dataset.

    .venv\Scripts\python.exe scripts/build_step2_files.py [--data data/final] [--out data/final/step2]

Reads  <data>/lejet_final_dataset.csv  (the final dataset: ONE file, 56 columns).
Writes (all keyed by message_id, same order as the dataset):
  Preprocessed_SMS_Text.csv       guide steps 9-14: each link becomes the <URL> placeholder (the link itself goes to the URL branch),
                                  lowercase, punctuation normalised (! and ? kept as tokens), tokenised, cut or padded to 50 tokens.
                                  Columns: message_id, label, text_tokens (before padding), n_tokens, truncated_at_50, has_url_token, has_url.
                                  NOT here: the vocabulary and the fastText vectors (steps 15-16): they are fitted per fold on the TRAINING part only,
                                  after the splits exist (src/features/build_crowd_embeddings.py).
  URL_Structural_Features.csv     message_id, has_url, page_fetched, page_missing (= 1 - page_fetched, as src/unified/link_lane.py), the 37 features.
  Feature_Extraction_Log.csv      per message: link, final_url, fetch_status, page_fetched, page_missing, where its features came from, and what is missing.
  Final_Model_Input.csv           ONLY the 46 columns the models read + text_tokens + n_tokens + page_missing.
  Step2_Check_Report.json         the guide's "before proceeding" checks (ids, counts, missing-value rules) and the recomputation of the URL features.
The URL features are the lab's (and ISDA's) own extraction. This script only RE-COMPUTES the 19 offline URL-string features from each message's
link with the repo's offline extractor to prove that every row carries the features of its own link.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from webapp import dataset                                          # noqa: E402
from src.features.text_preprocessing import text_tokens             # noqa: E402
from src.features.link_features import normalize_link               # noqa: E402
from lejet.features.hannousse_url import url_features               # noqa: E402

SEQ_LEN = 50
URL19 = [f for f in dataset.FEATURES_37[:20] if f != "nb_redirection"]      # nb_redirection needs a real fetch: not re-computable offline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "final"))
    ap.add_argument("--out", default=str(ROOT / "data" / "final" / "step2"))
    a = ap.parse_args()
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(d / "lejet_final_dataset.csv", keep_default_na=False)
    x = t                                                  # one file: the link, fetch_status ... columns are in it too
    rep: dict = {"rows": len(t)}

    # ---- A. text
    toks = [text_tokens(s) for s in t.sms_text]
    txt = pd.DataFrame({"message_id": t.message_id, "label": t.label, "text_tokens": [" ".join(k) for k in toks],
                        "n_tokens": [len(k) for k in toks]})
    txt["truncated_at_50"] = (txt.n_tokens > SEQ_LEN).astype(int)
    txt["has_url_token"] = [int("<URL>" in k) for k in toks]
    txt["has_url"] = t.has_url
    txt.to_csv(out / "Preprocessed_SMS_Text.csv", index=False, encoding="utf-8", lineterminator="\n")
    rep["text"] = {"empty_after_cleaning": int((txt.n_tokens == 0).sum()), "median_tokens": float(txt.n_tokens.median()),
                   "p95_tokens": float(txt.n_tokens.quantile(0.95)), "truncated_at_50": int(txt.truncated_at_50.sum()),
                   "has_url_but_no_<URL>_token": int(((txt.has_url == 1) & (txt.has_url_token == 0)).sum()),
                   "<URL>_token_but_has_url_0": int(((txt.has_url == 0) & (txt.has_url_token == 1)).sum())}

    # ---- B. URL features
    page_missing = 1 - t.page_fetched
    uf = pd.concat([t[["message_id", "has_url", "page_fetched"]], page_missing.rename("page_missing"), t[dataset.FEATURES_37]], axis=1)
    uf.to_csv(out / "URL_Structural_Features.csv", index=False, encoding="utf-8", lineterminator="\n")

    def came_from(r):
        return {"isda": "ISDA table (lab extractor)", "web_group": "website inspection (lab)", "web_legit_lab": "Azure lab batches (leg*/leb*)",
                "web_synthetic": "link hunts, features copied from the synthetic file"}[r]
    log = pd.DataFrame({"message_id": t.message_id, "has_url": t.has_url, "link": x.link, "final_url": x.final_url, "fetch_status": x.fetch_status,
                        "page_fetched": t.page_fetched, "page_missing": page_missing, "features_from": [came_from(s) for s in t.source],
                        "page_rank_filled": (t.page_rank != -1).astype(int),
                        "whois_domain_age_filled": (t.domain_age != -1).astype(int)})
    log["note"] = ""
    log.loc[log.has_url == 0, "note"] = "no URL: all 37 features -1"
    log.loc[(log.has_url == 1) & (log.page_fetched == 0), "note"] = "page not loaded: URL features kept, page-content features -1"
    log.loc[log.fetch_status == "blocked", "note"] = "page exists but refuses automated visitors (HTTP 401/403/406/429/503 or challenge page)"
    log.to_csv(out / "Feature_Extraction_Log.csv", index=False, encoding="utf-8", lineterminator="\n")

    # ---- the model input
    fmi = pd.concat([t[["message_id", "sms_text", "label", "source", "has_url", "link_domain", "template_id", "dup_group", "page_fetched"]],
                     txt[["text_tokens", "n_tokens"]], page_missing.rename("page_missing"), t[dataset.FEATURES_37]], axis=1)
    fmi.to_csv(out / "Final_Model_Input.csv", index=False, encoding="utf-8", lineterminator="\n")

    # ---- checks: ids, counts, missing-value rules
    chk = {"message_id unique": bool(t.message_id.is_unique),
           "same ids in every file": bool((txt.message_id == t.message_id).all() and (uf.message_id == t.message_id).all() and (fmi.message_id == t.message_id).all()),
           "every label is smishing or legitimate": bool(t.label.isin(["smishing", "legitimate"]).all()),
           "no URL -> has_url 0 and all 37 features -1": bool((t[t.has_url == 0][dataset.FEATURES_37] == -1).all().all()),
           "URL, page not loaded -> page-content features -1": bool((t[(t.has_url == 1) & (t.page_fetched == 0)][dataset.FEATURES_37[20:33]] == -1).all().all()),
           "page loaded -> has_url 1": bool((t[t.page_fetched == 1].has_url == 1).all()),
           "URL rows always have the 19 offline URL features": bool((t[t.has_url == 1][URL19] != -1).all().all())}
    rep["checks"] = chk
    rep["counts"] = {"with_url": int((t.has_url == 1).sum()), "without_url": int((t.has_url == 0).sum()),
                     "accessible": int((t.page_fetched == 1).sum()), "inaccessible": int(((t.has_url == 1) & (t.page_fetched == 0)).sum())}

    # ---- the features belong to the message's own link: recompute the 19 offline URL features from the link and compare
    rows = t[t.has_url == 1].index
    bad = {f: 0 for f in URL19}
    bad_rows, errors = set(), 0
    for i in rows:
        try:
            calc = url_features(normalize_link(str(x.link[i])))
        except Exception:                                   # a link the offline extractor cannot read
            errors += 1
            continue
        for f in URL19:
            if abs(float(calc.get(f, -2)) - float(t[f][i])) > 1e-6:
                bad[f] += 1
                bad_rows.add(i)
    rep["url_feature_recomputation"] = {"rows_checked": int(len(rows)), "extractor_errors": errors, "rows_with_any_difference": len(bad_rows),
                                        "share_identical": round(1 - (len(bad_rows) + errors) / max(1, len(rows)), 4), "differences_by_feature": {k: v for k, v in bad.items() if v}}
    if bad_rows:
        diff = pd.DataFrame({"message_id": t.message_id[sorted(bad_rows)], "source": t.source[sorted(bad_rows)], "link": x.link[sorted(bad_rows)]})
        diff.to_csv(out / "URL_Feature_Recheck_Differences.csv", index=False, encoding="utf-8", lineterminator="\n")
    # ---- and the link belongs to the message: it appears in the message's own text
    import re

    def _bare(u: str) -> str:
        return re.sub(r"^(https?://)?(www\.)?", "", u.strip().lower()).rstrip("/")

    not_in_text = []
    for i in rows:
        txt_ = re.sub(r"\s+", "", t.sms_text[i].lower().replace("hxxp", "http"))
        b = _bare(str(x.link[i]))
        if b not in txt_ and b.split("/")[0] not in txt_:
            not_in_text.append(t.message_id[i])
    rep["link_found_in_its_own_message"] = {"rows_checked": int(len(rows)), "not_found": len(not_in_text), "message_ids": not_in_text[:20]}
    rep["checks"]["every link appears in its own message text"] = not not_in_text
    rep["known_issue"] = ("Links that are a raw IP address (http://8.209.246.69/) are not recognised by the text masker, so they stay in the text as number tokens: "
                          f"{rep['text']['has_url_but_no_<URL>_token']} message(s). The link finder is src/features/link_features.extract_links.")
    (out / "Step2_Check_Report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
