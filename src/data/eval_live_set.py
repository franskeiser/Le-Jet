"""Score the models on the FRESH TEST part of the live set (links the lab inspected, never used to train, gate or tune).

Two questions, answered separately and never mixed with the thesis headline (which stays on real messages):
  1. Link level: do the frozen RF extractors tell smishing links from legitimate ones? `full` where the lab
     reached the page, `url_only` everywhere (the spelling-only view a user without inspection gets).
  2. Message level: with text that gives nothing away ("See details here <link>"), does the link/page evidence
     decide? The final model is run with the page shown (stored inspection) and with the page hidden (url_only),
     next to the CNN alone, which only sees the same ambiguous text.

Numbers on this set are small-sample: every rate comes with a 95% Wilson interval. Shorteners are nearly all
smishing here (a known confound, plan 29.8), so the link level is also reported with shorteners left out.
Nothing is opened or fetched: page evidence is read from data/live/live_links.csv only.

Usage:
    python -m src.data.eval_live_set [--live data/live/live_links.csv] [--out experiments/live_eval]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import ARTIFACTS_ROOT, SEEDS
from src.data.build_live_set import refang
from src.demo.synthetic import AMBIGUOUS_LINK, fill
from src.features.model.rf_extractor import extract_p_u, load_extractor
from src.serving.lejet_service import CNN_ID, FUSION_ID, LejetService, LinkStore

DEFAULT_LIVE = ARTIFACTS_ROOT / "data" / "live" / "live_links.csv"
DEFAULT_OUT = ARTIFACTS_ROOT / "experiments" / "live_eval"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def rate(hit: np.ndarray) -> dict:
    n, k = int(len(hit)), int(np.sum(hit))
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": round(k / n, 4) if n else None, "ci95": [lo, hi]}


def summarize(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    """y, pred: 1 = smishing. Recall per class with intervals, balanced accuracy, AUC when both classes exist."""
    out = {"n": int(len(y)), "smishing": int(y.sum()), "legitimate": int((1 - y).sum())}
    if len(y) == 0:
        return out
    out["smishing_recall"] = rate(pred[y == 1] == 1)
    out["legitimate_recall"] = rate(pred[y == 0] == 0)
    rs, rl = out["smishing_recall"]["rate"], out["legitimate_recall"]["rate"]
    if rs is not None and rl is not None:
        out["balanced_accuracy"] = round((rs + rl) / 2, 4)
    if score is not None and len(set(y)) == 2:
        out["auc"] = round(float(roc_auc_score(y, score)), 4)
    return out


def link_level(test: pd.DataFrame, seeds: list[int]) -> dict:
    y = (test["label"] == "smishing").astype(int).to_numpy()
    fetched = (test["page_fetched"] == 1).to_numpy()
    result: dict = {}
    for seed in seeds:
        url_only, full = load_extractor("url_only", seed), load_extractor("full", seed)
        view = test.copy()
        view["has_url"] = 1
        p_url = extract_p_u(url_only, view)["P"].to_numpy()
        p_full = np.full(len(test), np.nan)
        if fetched.any():
            p_full[fetched] = extract_p_u(full, view[fetched])["P"].to_numpy()
        p_routed = np.where(fetched, p_full, p_url)
        no_short = (test["shortener"] == 0).to_numpy()
        result[str(seed)] = {
            "url_only_all": summarize(y, (p_url >= 0.5).astype(int), p_url),
            "full_page_reached": summarize(y[fetched], (p_full[fetched] >= 0.5).astype(int), p_full[fetched]),
            "url_only_same_rows_as_full": summarize(y[fetched], (p_url[fetched] >= 0.5).astype(int), p_url[fetched]),
            "routed_all": summarize(y, (p_routed >= 0.5).astype(int), p_routed),
            "routed_without_shorteners": summarize(y[no_short], (p_routed[no_short] >= 0.5).astype(int), p_routed[no_short]),
            "url_only_without_shorteners": summarize(y[no_short], (p_url[no_short] >= 0.5).astype(int), p_url[no_short]),
        }
    return result


def message_level(test: pd.DataFrame, live_csv: Path, seed: int, rng_seed: int = 42, fusion_dir: Path | None = None) -> tuple[dict, pd.DataFrame]:
    """Fusion model on ambiguous texts wrapped around each fresh-test link: page shown vs page hidden vs CNN alone."""
    shown = LejetService(seed=seed, fusion_dir=fusion_dir, store=LinkStore(live_csv))
    hidden = shown                                                       # same weights; its store is swapped per call
    full_store, empty_store = shown.store, LinkStore(Path("__no_such_file__"))
    rng = random.Random(rng_seed)
    rows = []
    for _, r in test.iterrows():
        msg = fill(rng.choice(AMBIGUOUS_LINK), rng, refang(r["link"]))
        shown.store = full_store
        a = shown.predict(msg, FUSION_ID, explain=False)
        c = shown.predict(msg, CNN_ID, explain=False)
        shown.store = empty_store
        b = shown.predict(msg, FUSION_ID, explain=False)
        shown.store = full_store
        rows.append({"y": int(r["label"] == "smishing"), "page_fetched": int(r["page_fetched"]),
                     "shortener": int(r["shortener"]),
                     "shown": int(a["label"] == "smishing"), "hidden": int(b["label"] == "smishing"),
                     "cnn": int(c["label"] == "smishing"),
                     "p_shown": a["smishing_probability"] / 100, "p_hidden": b["smishing_probability"] / 100,
                     "p_cnn": c["smishing_probability"] / 100,
                     "state_shown": a["inspection"]["state"]})
    d = pd.DataFrame(rows)
    y = d["y"].to_numpy()
    fetched = (d["page_fetched"] == 1).to_numpy()
    out = {
        "n_with_stored_page": int((d["state_shown"] == "inspected").sum()),
        "fusion_page_shown_all": summarize(y, d["shown"].to_numpy(), d["p_shown"].to_numpy()),
        "fusion_page_hidden_all": summarize(y, d["hidden"].to_numpy(), d["p_hidden"].to_numpy()),
        "cnn_only_all": summarize(y, d["cnn"].to_numpy(), d["p_cnn"].to_numpy()),
        "fusion_page_shown_reached_only": summarize(y[fetched], d["shown"].to_numpy()[fetched], d["p_shown"].to_numpy()[fetched]),
        "fusion_page_hidden_reached_only": summarize(y[fetched], d["hidden"].to_numpy()[fetched], d["p_hidden"].to_numpy()[fetched]),
        "cnn_only_reached_only": summarize(y[fetched], d["cnn"].to_numpy()[fetched], d["p_cnn"].to_numpy()[fetched]),
    }
    return out, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", type=Path, default=DEFAULT_LIVE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fusion-dir", type=Path, default=None, help="head to score (default models/fusion/rf_vec)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    live = pd.read_csv(args.live, keep_default_na=False)
    test = live[live["split"] == "fresh_test"].reset_index(drop=True)
    train = live[live["split"] == "live_train"]
    assert set(test["group"]).isdisjoint(set(train["group"])), "a domain group is in both live sets"
    counts = {"fresh_test": {l: int((test["label"] == l).sum()) for l in ("smishing", "legitimate")},
              "page_reached": int((test["page_fetched"] == 1).sum()),
              "shorteners": {l: int(((test["label"] == l) & (test["shortener"] == 1)).sum()) for l in ("smishing", "legitimate")}}
    msg, per_row = message_level(test, args.live, args.seeds[0], fusion_dir=args.fusion_dir)
    report = {"counts": counts, "seeds": args.seeds, "link_level": link_level(test, args.seeds), "message_level_seed": args.seeds[0], "fusion_dir": str(args.fusion_dir or "models/fusion/rf_vec"),
              "message_level": msg,
              "notes": ["Small sample: read every rate with its interval.",
                        "Legitimate links are generated well-known sites, smishing links come from the team's collection; "
                        "the two groups differ in more than spelling, so this is a sanity check, not a headline result.",
                        "Shorteners are almost all smishing here (confound, plan 29.8): see the *_without_shorteners rows."]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "live_eval.json").write_text(json.dumps(report, indent=2))
    per_row.to_csv(args.out / "message_level_rows.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
