#!/usr/bin/env python
"""
openphish_score.py -- features and the model's verdict for every link that scripts/openphish_check.py opened in the lab.

    .venv\\Scripts\\python.exe scripts\\openphish_score.py [--out D:\\LeJetData\\openphish] [--kind film_scratch] [--dataset crowd_relevant_synthetic]

Reads ONLY files the lab already saved (<out>\\run\\clean) plus the ALL file from openphish_check.py; it opens no link and needs no lab.
Writes <out>\\openphish_ph_bank_features_and_verdict.csv, one row per link, with:

  alive, why, first_seen_day, brand, url, lab_batch, fail_reason      from the check
  page_seen_by_model          1 if the lab saw the page (so the page features are real numbers), 0 if the page was dead (they are -1)
  link_rf_P_phishing          the Link Random Forest's probability that the link is phishing (the model's per-link verdict)
  link_rf_uncertainty_u       how unsure the forest is (0 = sure)
  link_rf_verdict             phishing / legitimate  (P > 0.5)
  link_rf_P_spelling_only     the same forest if the page were hidden: only how the link is spelt + domain facts
  page_effect_on_P            P minus spelling-only P: how much seeing the page moved the score
  film_P_smishing_text_is_link_only   the full FiLM text model, with the link itself as the message text. These links have no SMS, so
                              this number only shows the machinery works; it is NOT a smishing verdict.
  model_note                  says which model this is
  rf37_*                      the 37 numbers the Link RF read (20 link spelling, 13 page, 4 lookups; -1 = not available)
  lab_*  /  full_*            the lab's compact and full (Hannousse) feature sets

THE MODELS ON DISK ARE REHEARSAL MODELS trained on synthetic messages (docs/film_architecture_plan.md: "a rehearsal, never a result").
A verdict from them is real model output but says nothing about how well Le-Jet detects smishing. When the crowd-trained model exists,
pass --dataset crowd_smishing_v1.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(r"D:\LeJetData\openphish")


def refang(url: str) -> str:
    return re.sub(r"\[\.\]|\(\.\)", ".", re.sub(r"^hxxp", "http", url.strip(), flags=re.I))


def defang(url: str) -> str:
    return re.sub(r"\.", "[.]", re.sub(r"^http", "hxxp", url.strip(), flags=re.I))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", default="crowd_relevant_synthetic", help="the FiLM model's training set (the tool serves this one by default)")
    ap.add_argument("--kind", default="film_scratch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from src.data.build_live_set import LABELS, load_batch          # heavy imports only when actually scoring
    from src.features.link_features import normalize_link
    from src.serving.film_model import FilmModel

    clean = a.out / "run" / "clean"
    all_csv = a.out / "openphish_ph_bank_checked_ALL.csv"
    if not all_csv.exists():
        print(f"{all_csv} not found: run scripts/openphish_check.py first", file=sys.stderr)
        return 2
    dataset = REPO / "data" / "processed" / f"{a.dataset}.csv"
    if not FilmModel.available(dataset, a.kind, a.seed, a.fold):
        print(f"the {a.kind} model trained on {a.dataset} (seed {a.seed}, fold {a.fold}) is not on this PC", file=sys.stderr)
        return 2
    film = FilmModel(dataset, a.kind, a.seed, a.fold)
    print("model:", film.info, flush=True)
    note = ("REHEARSAL model trained on synthetic messages; not a validated result" if film.info.get("synthetic")
            else f"model trained on {a.dataset}")

    rows: List[Dict[str, Any]] = []
    for chk in csv.DictReader(open(all_csv, encoding="utf-8")):
        batch = chk["lab_batch"]
        path = clean / f"{batch}.clean.jsonl"
        if not path.exists():
            continue
        recs = load_batch(clean, batch, LABELS[0], "on_demand")
        if len(recs) != 1:
            continue
        stored = recs[0]
        raw: Dict[str, Any] = json.loads(open(path, encoding="utf-8").readline())
        url = refang(raw["url"])
        norm = normalize_link(url)

        feats, page_used = film.link_features(norm, stored)                       # the 37 numbers, with the page the lab saw
        p, u = film.link_numbers(feats)
        feats0, _ = film.link_features(norm, None)                                # the same link with the page hidden
        p0, _u0 = film.link_numbers(feats0)
        link = film.link_tensor(p, u, 0 if page_used else 1, 1)
        film_p = film.p_smishing(film.token_ids(url), link)                       # text = the link only

        row: Dict[str, Any] = {"alive": chk["alive"], "why": chk["why"], "first_seen_day": chk["first_seen_day"], "brand": chk["brand"],
                               "url": url, "lab_batch": batch, "fail_reason": raw.get("fail_reason"), "page_seen_by_model": int(page_used),
                               "link_rf_P_phishing": round(p, 4), "link_rf_uncertainty_u": round(u, 4),
                               "link_rf_verdict": "phishing" if p > 0.5 else "legitimate", "link_rf_P_spelling_only": round(p0, 4),
                               "page_effect_on_P": round(p - p0, 4), "film_P_smishing_text_is_link_only": round(film_p, 4), "model_note": note}
        for name in feats.columns:
            row[f"rf37_{name}"] = float(feats.iloc[0][name])
        for k, v in sorted((raw.get("features") or {}).items()):
            if isinstance(v, (int, float)) and f"rf37_{k}" not in row:
                row[f"lab_{k}"] = v
        for k, v in sorted((raw.get("features_full") or {}).items()):
            if isinstance(v, (int, float)):
                row[f"full_{k}"] = v
        rows.append(row)

    if not rows:
        print("no finished batches to score", file=sys.stderr)
        return 1
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = a.out / "openphish_ph_bank_features_and_verdict.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} links x {len(cols)} columns")
    for r in rows:
        print(f'{r["lab_batch"]}  alive={r["alive"]:5} page_seen={r["page_seen_by_model"]}  P={r["link_rf_P_phishing"]:<6} u={r["link_rf_uncertainty_u"]:<6} '
              f'-> {r["link_rf_verdict"]:10} {defang(r["url"])[:50]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
