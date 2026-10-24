"""Check that the FiLM model the backend serves gives the labels of the evaluation, message by message (plan step 7.8).

Same idea as the SOP 2 measurement's parity check: the served pipeline and the evaluation pipeline must not disagree on any
message of the fold's test part. Two comparisons:

  A. SERVED vs EVALUATION, same inputs. The 37 features come from the dataset's own columns, so the only difference is the code
     path (FilmModel: link numbers -> link tensor -> text ids -> probability, against film.predict_messages). Every label must be
     the same and the probabilities equal to 1e-6. This is the hard check; the script exits 1 if it fails.
  B. FULL PATH FROM THE SMS TEXT ALONE (LejetService.predict): the link is found in the message, its 20 URL features are
     recomputed from its spelling, and nothing else is known (the link is not in the store), against the evaluation on the same
     rows with the 17 page / lookup columns blanked and no redirect count. The URL features can differ a little from the dataset's
     (the dataset kept a trailing "!" or ")" that the app strips from a link), so agreement is reported, not demanded; the hard
     part is that no message crashes and the no-link bypass gives P = 0, u = 0.

Usage (project root; --pilot reads LEJET_PILOT_DATASET like the other tools):
    set LEJET_PILOT_DATASET=data/processed/crowd_relevant_synthetic.csv
    python scripts/check_film_serving.py --pilot [--seed 42] [--fold 0] [--limit 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.split_crowd import load_fold  # noqa: E402
from src.serving.film_model import DEFAULT_KIND, FilmModel  # noqa: E402
from src.serving.lejet_service import FILM_ID, LejetService, LinkStore  # noqa: E402
from src.unified import film as evaluation  # noqa: E402
from src.unified import text_cnn as tc  # noqa: E402
from src.unified.prepare_web_37 import FEATURES_37, PAGE_FEATURES, URL_FEATURES  # noqa: E402


def served_from_features(film: FilmModel, rows) -> np.ndarray:
    """Probability of smishing for each row, one message at a time, through FilmModel (the backend's pieces)."""
    out = []
    has_url = rows["has_url"].astype(int).to_numpy() if "has_url" in rows.columns else np.ones(len(rows), dtype=int)
    page_ok = rows["page_fetched"].astype(int).to_numpy()
    for i in range(len(rows)):
        p = u = 0.0
        if has_url[i]:
            p, u = film.link_numbers(rows.iloc[[i]][FEATURES_37])
        link = film.link_tensor(p, u, 1 - page_ok[i], has_url[i])
        out.append(film.p_smishing(film.token_ids(str(rows["sms_text"].iloc[i])), link))
    return np.array(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--kind", default=DEFAULT_KIND)
    parser.add_argument("--limit", type=int, default=None, help="only the first N test messages (quick check)")
    args = parser.parse_args(argv)

    dataset = tc.dataset_path(args.pilot)
    film = FilmModel(dataset, args.kind, args.seed, args.fold)
    rows = load_fold(args.seed, args.fold, pilot=args.pilot)["test"]
    if args.limit:
        rows = rows.iloc[: args.limit].reset_index(drop=True)
    print(f"{film.info['model']} trained on {film.info['dataset']}: {args.kind} seed {args.seed} fold {args.fold}, {len(rows)} test messages")

    # A. same inputs, two code paths
    reference = evaluation.predict_messages(film.net, film.rf, film.word_to_idx, film.seq_len, rows)
    served = served_from_features(film, rows)
    same_label = int(((served >= 0.5).astype(int) == reference["pred"].to_numpy()).sum())
    worst = float(np.abs(served - reference["p_smishing"].to_numpy()).max())
    ok_a = same_label == len(rows) and worst < 1e-6
    print(f"A. served vs evaluation, same features: {same_label}/{len(rows)} labels equal, largest probability gap {worst:.2e}  "
          f"-> {'PASS' if ok_a else 'FAIL'}")

    # B. the whole path from the SMS text, no page known
    service = LejetService(seed=args.seed, store=LinkStore(Path("__no_such_store__.csv"), []), film=film)
    unseen = rows.copy()                       # a link the lab never saw: no page, no lookups, no redirect count (blank_features)
    unseen[PAGE_FEATURES] = -1.0
    unseen["nb_redirection"] = 0.0
    unseen["page_fetched"] = 0
    hidden = evaluation.predict_messages(film.net, film.rf, film.word_to_idx, film.seq_len, unseen)
    answers = [service.predict(str(t), FILM_ID, explain=False) for t in rows["sms_text"]]
    labels = np.array([int(a["label"] == "smishing") for a in answers])
    probs = np.array([a["smishing_probability"] / 100 for a in answers])
    agree = int((labels == hidden["pred"].to_numpy()).sum())
    linked = rows["has_url"].astype(int).to_numpy() == 1 if "has_url" in rows.columns else np.ones(len(rows), dtype=bool)
    bypass_ok = all(a["timings"]["has_link"] is False and a["link_effect"]["p"] == 0.0 and a["link_effect"]["mean_abs_gamma_shift"] == 0.0
                    for a, has in zip(answers, linked) if not has)
    # how far the recomputed URL features are from the dataset's columns (the cause of any disagreement)
    from src.serving.film_model import _url_features
    from src.features.link_features import extract_links, normalize_link
    diffs, checked = 0, 0
    for text, (_, r) in zip(rows["sms_text"], rows.iterrows()):
        links = extract_links(str(text))
        if links and (r.get("has_url", 1) == 1):
            got = _url_features(normalize_link(links[0]["raw"]))
            checked += 1
            diffs += int(any(abs(got[name] - float(r[name])) > 1e-6 for name in URL_FEATURES))
    print(f"B. full path from the text, link never seen by the lab: {agree}/{len(rows)} labels equal the evaluation on the same rows with their page and lookup columns blanked "
          f"({100 * agree / len(rows):.2f}%), mean probability gap {np.abs(probs - hidden['p_smishing'].to_numpy()).mean():.4f}; "
          f"no-link bypass {'ok' if bypass_ok else 'BROKEN'}; {diffs} of {checked} links have a URL feature that differs from the dataset's")
    return 0 if (ok_a and bypass_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
