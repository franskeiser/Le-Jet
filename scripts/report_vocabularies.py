r"""Check and report the per-fold vocabularies and fastText matrices (guide step 2, items 15-16), and export the splits (guide step 3 file).

    .venv\Scripts\python.exe scripts/report_vocabularies.py [--dataset lejet_final_dataset]

For every seed x fold it proves that the vocabulary is fitted on the TRAINING part only:
  * the vocabulary is exactly <PAD>, <URL> and the tokens of the training messages (nothing from validation or test);
  * how many validation / test word types and word occurrences the vocabulary does not know (they are dropped at encoding time, as for a new message);
  * the matrix has one 300-d row per token, the <PAD> row is zero, there is no NaN, and how many words took their vector from English,
    Tagalog or the English sub-word guess.
Writes into experiments/film/<dataset>/data_prep/:
    vocabulary_report.csv / .json     one row per seed and fold
    split_report.json                 a copy of data/splits_crowd/<dataset>/split_report.json
    Cross_Validation_Splits.xlsx      the guide's step 3 file: message_id, label, and the part (train / val / test) in each of the 15 splits
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_EMBEDDINGS_DIR, CROWD_N_FOLDS, CROWD_SPLITS_DIR, SEEDS      # noqa: E402
from src.features.text_preprocessing import PAD_TOKEN, text_tokens                       # noqa: E402

URL_TOKEN = "<URL>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lejet_final_dataset")
    ap.add_argument("--data", default=str(ROOT / "data" / "final" / "lejet_final_dataset.csv"))
    a = ap.parse_args()
    df = pd.read_csv(a.data, keep_default_na=False)
    toks = [text_tokens(s) for s in df.sms_text]
    out = ROOT / "experiments" / "film" / a.dataset / "data_prep"
    out.mkdir(parents=True, exist_ok=True)
    splits = ROOT / CROWD_SPLITS_DIR / a.dataset
    emb = ROOT / CROWD_EMBEDDINGS_DIR / a.dataset

    rows, ok_all = [], True
    cv = df[["message_id", "label"]].copy()
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            part = pd.read_csv(splits / f"seed{seed}_fold{fold}.csv")
            assert (part.message_id == df.message_id).all()
            cv[f"seed{seed}_fold{fold}"] = part.part.values
            vocab = json.loads((emb / f"seed{seed}_fold{fold}" / "vocab.json").read_text(encoding="utf-8"))
            matrix = np.load(emb / f"seed{seed}_fold{fold}" / "embedding_matrix.npy")
            train_types = {w for t, p in zip(toks, part.part) if p == "train" for w in t} - {PAD_TOKEN, URL_TOKEN}
            vset = set(vocab) - {PAD_TOKEN, URL_TOKEN}
            train_only = vset == train_types and vocab[:2] == [PAD_TOKEN, URL_TOKEN] and len(vocab) == len(set(vocab))
            row = {"seed": seed, "fold": fold, "vocab_size": len(vocab), "train_word_types": len(train_types), "vocab_is_exactly_train_words": bool(train_only)}
            for name in ("val", "test"):
                occ = Counter(w for t, p in zip(toks, part.part) if p == name for w in t if w not in (PAD_TOKEN, URL_TOKEN))
                unk = {w for w in occ if w not in vset}
                row[f"{name}_word_types"] = len(occ)
                row[f"{name}_types_not_in_vocab"] = len(unk)
                row[f"{name}_occurrences_not_in_vocab_pct"] = round(100 * sum(occ[w] for w in unk) / max(1, sum(occ.values())), 2)
            row.update(matrix_shape=f"{matrix.shape[0]}x{matrix.shape[1]}", matrix_rows_match_vocab=bool(matrix.shape[0] == len(vocab) and matrix.shape[1] == 300),
                       pad_row_is_zero=bool(not matrix[0].any()), no_nan=bool(np.isfinite(matrix).all()),
                       rows_all_zero_besides_pad=int((~matrix[1:].any(axis=1)).sum()))
            ok_all &= row["vocab_is_exactly_train_words"] and row["matrix_rows_match_vocab"] and row["pad_row_is_zero"] and row["no_nan"]
            rows.append(row)
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "vocabulary_report.csv", index=False)
    summary = {"dataset": a.dataset, "folds": len(rep), "all_checks_pass": bool(ok_all),
               "vocab_size_min_max": [int(rep.vocab_size.min()), int(rep.vocab_size.max())],
               "val_occurrences_not_in_vocab_pct_mean": round(float(rep.val_occurrences_not_in_vocab_pct.mean()), 2),
               "test_occurrences_not_in_vocab_pct_mean": round(float(rep.test_occurrences_not_in_vocab_pct.mean()), 2),
               "rows_all_zero_besides_pad_max": int(rep.rows_all_zero_besides_pad.max())}
    (out / "vocabulary_report.json").write_text(json.dumps({"summary": summary, "folds": rows}, indent=2), encoding="utf-8")
    shutil.copy(splits / "split_report.json", out / "split_report.json")
    with pd.ExcelWriter(out / "Cross_Validation_Splits.xlsx", engine="openpyxl") as w:
        cv.to_excel(w, sheet_name="splits", index=False)
        readme = pd.DataFrame({"Note": ["Stratified group 5-fold cross-validation, seeds 42, 123 and 2024 (15 splits). Each column is one split; the value is the part the message is in.",
                                        "test = the held-out fold (every message is tested exactly once per seed); val = about 15% of the other four folds, for early stopping; train = the rest.",
                                        "Rows of one group are never in two parts. Group = link domain, template id and near-duplicate group joined together; a message with no link is its own group.",
                                        f"Dataset: {a.dataset} ({len(df)} messages). Rule and sizes: split_report.json."]})
        readme.to_excel(w, sheet_name="README", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
