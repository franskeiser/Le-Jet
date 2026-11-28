r"""FiLM fine-tuning round 2 (2026-11-25), judged on VALIDATION rows only. Nothing here reads a test row's text, label or prediction.

    .venv\Scripts\python.exe scripts/film_tune_validation.py [--variants D1 G3 ...]

Why: the validation diagnosis of 2026-11-25 found D1 (the adopted Proposed model) ties CNN-only. It catches more URL smishing but
adds false alarms on legitimate URL messages where the Link RF is unsure (P 0.2-0.4): the generator turns a weak text signal plus
a weak link signal into "smishing". The candidates below only change the Proposed model's OWN parts (the generator and how it
reads the link), so the shared recipe (Adam 1e-3, patience 3, batch 64, modality dropout 0.25 / 0.10) and the baselines stay as
recorded and no baseline needs retraining.

Candidates, fixed BEFORE running (all on top of D1 = text_dropout "input"):
  D1      replay of the adopted model; must reproduce film_upgrade/cells.csv exactly (replay gap 0)
  G3      generator weight decay 1e-3 (D1: 1e-4): keeps gamma near 1 and beta near 0 unless the link earns it
  G2      generator weight decay 1e-2 (stronger version of G3)
  V1      gamma only, no beta (use_beta False): the link can scale what the text found but cannot add evidence of its own
  V3      the generator reads the link numbers only, not the Conv average (use_average False): stops the weak-text x
          weak-link interaction
  M       mask_padding + average_after_relu: the Conv average no longer depends on message length or on cancelling scores
  V1G3    V1 + G3

Projected McNemar (proxy for Step 10, as film_upgrade_validation.py): per seed, b = rows the candidate gets right and the baseline
wrong, c = the reverse, summed over the 5 folds' validation rows. Baselines = the saved CNN-only and Concatenation validation
predictions (validation_study/validation_predictions.csv, cut-off 0.5).

Keep rule (fixed before running): mean validation balanced accuracy above D1's, AND the worst-seed b - c (over both baselines)
above D1's worst-seed b - c. Best = largest worst-seed b - c. Caveat to disclose: validation rows also pick each run's best epoch,
and picking among 6 candidates on them is optimistic; one confirmatory test read is still needed.

Writes models/film/<dataset>/film_tune/ and experiments/film/<dataset>/film_tune/:
  cells.csv, val_predictions.csv, mcnemar_projection.csv, summary.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_N_FOLDS, SEEDS                              # noqa: E402
from src.data.split_crowd import load_fold                               # noqa: E402
from src.unified import film                                             # noqa: E402
from src.unified import fit_model as fm                                  # noqa: E402
from src.unified import text_cnn as tc                                   # noqa: E402
from src.unified.film_data import part_inputs                            # noqa: E402

D1 = {"text_dropout": "input"}
# name: (FiLMTextCNN options, fit() options)
VARIANTS = {
    "D1": (D1, {}),
    "G3": (D1, {"generator_weight_decay": 1e-3}),
    "G2": (D1, {"generator_weight_decay": 1e-2}),
    "V1": ({**D1, "use_beta": False}, {}),
    "V3": ({**D1, "use_average": False}, {}),
    "M": ({**D1, "mask_padding": True, "average_after_relu": True}, {}),
    "V1G3": ({**D1, "use_beta": False}, {"generator_weight_decay": 1e-3}),
    # Round 3 (fixed 2026-11-25 BEFORE round 2's results were seen; run only if round 2 keeps nothing significant).
    # These change how the Proposed model's link lane is trained (modality dropout, gamma form, generator depth); the
    # baselines' recipe is untouched. A kept round-3 candidate that changes p_text_hidden / p_page_hidden must be reported
    # as a Proposed-only training choice.
    "T0": (D1, {"p_text_hidden": 0.0}),                   # never hide the words: the link cannot learn to stand in for text
    "T5": (D1, {"p_text_hidden": 0.05}),
    "P10": (D1, {"p_page_hidden": 0.10}),                 # hide the page less often (more page evidence while training)
    "GF": ({**D1, "gamma_mode": "free"}, {}),             # unbounded gamma, as in the FiLM paper
    "H32G3": ({**D1, "generator_hidden": 32}, {"generator_weight_decay": 1e-3}),
    # Added 2026-11-25 after round 2 (G3 kept) and a partial round 3 (GF ahead on balanced accuracy; its b - c not yet seen).
    "G3GF": ({**D1, "gamma_mode": "free"}, {"generator_weight_decay": 1e-3}),
}
ROUND2 = ["D1", "G3", "G2", "V1", "V3", "M", "V1G3"]
ROUND3 = ["D1", "T0", "T5", "P10", "GF", "H32G3"]
BASELINES = {"CNN-only": "CNN-only", "Concatenation Fusion": "Concat"}


def mcnemar_p(b: int, c: int) -> float:
    return float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1)) if b + c else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=ROUND2, choices=list(VARIANTS))
    ap.add_argument("--tag", default="film_tune", help="output folder name (round 3: film_tune_r3)")
    args = ap.parse_args()
    names = args.variants
    if "D1" not in names:
        names = ["D1"] + names
    dataset = tc.dataset_path(False)
    out = fm.out_dir(dataset, args.tag)
    mdir = fm.model_dir(dataset) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    recorded = pd.read_csv(fm.out_dir(dataset, "film_upgrade") / "cells.csv")
    recorded = recorded[recorded["variant"] == "D1"].set_index(["seed", "fold"])["val_balanced_accuracy"]
    base_pred = pd.read_csv(fm.out_dir(dataset, "validation_study") / "validation_predictions.csv")
    base_pred = base_pred[base_pred["model"].isin(BASELINES)]

    # Resume: every finished run is appended to partial_*.csv, so a killed run restarts where it stopped (training is deterministic).
    part_cells, part_preds = out / "partial_cells.csv", out / "partial_preds.csv"
    cells = pd.read_csv(part_cells).to_dict("records") if part_cells.exists() else []
    preds = [pd.read_csv(part_preds)] if part_preds.exists() else []
    done = {(r["variant"], r["seed"], r["fold"]) for r in cells}
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            parts = load_fold(seed, fold)
            assert not (set(parts["val"]["message_id"]) & set(parts["test"]["message_id"]))
            for name in names:
                if (name, seed, fold) in done:
                    continue
                options, fit_options = VARIANTS[name]
                model, w2i, seq_len = film.build_film(seed, fold, False, None, **options)
                train, val = (part_inputs(seed, fold, p, w2i, seq_len) for p in ("train", "val"))
                assert list(val.message_id) == list(parts["val"]["message_id"])
                result = fm.fit(model, (train.token_ids, train.link_full, train.link_blank, train.labels),
                                (val.token_ids, val.link_full, val.labels), seed, mdir / f"{name}_seed{seed}_fold{fold}.pt",
                                **fit_options)
                with torch.no_grad():
                    p = torch.softmax(model(val.token_ids, val.link_full), dim=1)[:, 1].numpy()
                y, pred = val.labels.numpy(), (p >= 0.5).astype(int)
                legit = y == 0
                row = {"variant": name, "seed": seed, "fold": fold, "n_val": len(y),
                       "val_balanced_accuracy": balanced_accuracy_score(y, pred), "val_f1": f1_score(y, pred),
                       "false_alarms": int((pred[legit] == 1).sum()), "misses": int((pred[~legit] == 0).sum()),
                       "best_epoch": result.best_epoch, "epochs_run": result.epochs_run,
                       "trainable": film.trainable_counts(model)[0]}
                if name == "D1":
                    row["replay_gap"] = abs(row["val_balanced_accuracy"] - recorded.loc[(seed, fold)])
                cells.append(row)
                frame = pd.DataFrame({"variant": name, "seed": seed, "fold": fold, "message_id": val.message_id,
                                      "label": y, "p_smishing": p})
                preds.append(frame)
                pd.DataFrame([row]).to_csv(part_cells, mode="a", header=not part_cells.exists(), index=False)
                frame.to_csv(part_preds, mode="a", header=not part_preds.exists(), index=False)
                print(f"seed {seed} fold {fold} {name:5}: val bal-acc {row['val_balanced_accuracy']:.4f} | FA {row['false_alarms']} "
                      f"miss {row['misses']} | best epoch {row['best_epoch']} of {row['epochs_run']}", flush=True)

    cells = pd.DataFrame(cells)
    preds = pd.concat(preds, ignore_index=True)
    cells.to_csv(out / "cells.csv", index=False)
    preds.to_csv(out / "val_predictions.csv", index=False)

    preds["ok"] = ((preds["p_smishing"] >= 0.5).astype(int) == preds["label"]).astype(int)
    base_pred = base_pred.assign(ok=((base_pred["p_smishing"] >= 0.5).astype(int) == base_pred["label"]).astype(int))
    key = ["seed", "fold", "message_id"]
    mc = []
    for name in names:
        mine = preds[preds["variant"] == name][key + ["ok"]]
        for bname, short in BASELINES.items():
            both = mine.merge(base_pred[base_pred["model"] == bname][key + ["ok"]], on=key, suffixes=("", "_base"),
                              validate="one_to_one")
            assert len(both) == len(mine)
            for seed, g in both.groupby("seed"):
                b = int(((g["ok"] == 1) & (g["ok_base"] == 0)).sum())
                c = int(((g["ok"] == 0) & (g["ok_base"] == 1)).sum())
                mc.append({"variant": name, "baseline": short, "seed": seed, "b": b, "c": c, "b_minus_c": b - c,
                           "p_proxy": round(mcnemar_p(b, c), 4)})
    mc = pd.DataFrame(mc)
    mc.to_csv(out / "mcnemar_projection.csv", index=False)

    d1 = cells[cells["variant"] == "D1"].set_index(["seed", "fold"])["val_balanced_accuracy"]
    d1_worst = int(mc[mc["variant"] == "D1"]["b_minus_c"].min())
    lines = [f"D1 replay: largest gap to film_upgrade D1 = {cells['replay_gap'].max():.6f} (must be 0)"]
    table = []
    for name in names:
        g = cells[cells["variant"] == name].set_index(["seed", "fold"])
        m = mc[mc["variant"] == name]
        worst = int(m["b_minus_c"].min())
        kept = name != "D1" and g["val_balanced_accuracy"].mean() > d1.mean() and worst > d1_worst
        table.append({"variant": name, "val_bal_acc": round(g["val_balanced_accuracy"].mean(), 4),
                      "val_f1": round(g["val_f1"].mean(), 4),
                      "wins_vs_D1": int((g["val_balanced_accuracy"] > d1).sum()) if name != "D1" else 0,
                      "FA_per_fold": round(g["false_alarms"].mean(), 1), "miss_per_fold": round(g["misses"].mean(), 1),
                      "sum_b_minus_c": int(m["b_minus_c"].sum()), "worst_seed_b_minus_c": worst, "kept": kept})
    table = pd.DataFrame(table)
    pd.set_option("display.width", 250)
    lines += ["", table.to_string(index=False), "", "projected McNemar b - c per seed (validation rows, proxy):",
              mc.pivot_table(index=["variant", "baseline"], columns="seed", values="b_minus_c").to_string()]
    kept = table[table["kept"]]
    lines += ["", "KEPT: " + (", ".join(kept.sort_values("worst_seed_b_minus_c", ascending=False)["variant"]) if len(kept)
                              else "none (no candidate passes the keep rule)")]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
