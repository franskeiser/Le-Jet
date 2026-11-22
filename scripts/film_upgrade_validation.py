r"""FiLM upgrade study, judged on VALIDATION rows only. Nothing here reads a test row's text, label or prediction.

    .venv\Scripts\python.exe scripts/film_upgrade_validation.py            (after scripts/validation_predictions.py)

Goal (Franz, 2026-11-22): the Proposed model significantly better than every baseline in the Step 10 McNemar tests, all 3 seeds.
Plan: C:\Users\Franz\.claude\plans\can-you-determine-why-vectorized-lynx-of-*.md. Candidates fixed BEFORE running:
  control  the unchanged Proposed model (from scratch, has_url bypass), replayed; must reproduce the recorded validation metrics
  D1       text_dropout="input": modality dropout hides the WORDS (Conv input) instead of the pooled numbers, so the link's gamma and
           beta still reach the head on those rows (as the baselines' text dropout keeps their link numbers)
  D1F2     D1 + the generator also reads the Link RF's 37 contributions (link_dim 40)
  D1F3     D1 + a two-layer generator (hidden 64)
  D1F23    D1 + both
  D1F1     D1 + [P, u, page_missing, has_url] also attached to the head (the head no longer sees only text: adviser decision)
Same data, seeds, folds, recipe (fit_model.fit) and early stopping on validation balanced accuracy as the recorded Proposed model.

Projected McNemar (a proxy for Step 10): for each seed, b = rows the candidate gets right and the baseline wrong, c = the reverse, summed
over the 5 folds' validation rows (validation rows of different folds can overlap, so this is a proxy, not a test). Baselines = the saved
CNN-only and Concatenation models' validation predictions (validation_study/validation_predictions.csv, cut-off 0.5).

Keep rule (fixed before running): better than control in mean validation balanced accuracy AND in >= 11 of 15 seed-folds, AND in every
seed b - c > 0 against both CNN-only and Concatenation. Among those kept, the best is the one with the largest worst-seed b - c.

Writes models/film/<dataset>/film_upgrade/ and experiments/film/<dataset>/film_upgrade/:
  cells.csv (one row per variant x seed x fold), val_predictions.csv, mcnemar_projection.csv, summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_DATASET_PATH, CROWD_N_FOLDS, SEEDS          # noqa: E402
from src.data.split_crowd import load_fold                               # noqa: E402
from src.unified import film                                             # noqa: E402
from src.unified import fit_model as fm                                  # noqa: E402
from src.unified import link_lane as ll                                  # noqa: E402
from src.unified import text_cnn as tc                                   # noqa: E402
from src.unified import voting_baseline as vb                            # noqa: E402
from src.unified.film_data import part_inputs                            # noqa: E402

C37 = tuple(ll.CONTRIBUTIONS)
VARIANTS = {
    "control": ({}, ()),
    "D1": ({"text_dropout": "input"}, ()),
    "D1F2": ({"text_dropout": "input", "link_dim": 40}, C37),
    "D1F3": ({"text_dropout": "input", "generator_hidden": 64}, ()),
    "D1F23": ({"text_dropout": "input", "link_dim": 40, "generator_hidden": 64}, C37),
    "D1F1": ({"text_dropout": "input", "head_link": True}, ()),
}
BASELINES = {"CNN-only": "CNN-only", "Concatenation Fusion": "Concat"}
MIN_WINS = 11


def mcnemar_p(b: int, c: int) -> float:
    """Continuity-corrected chi-square p-value, as scripts/build_step10_files.py."""
    return float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1)) if b + c else 1.0


def main() -> int:
    dataset = tc.dataset_path(False)
    out = fm.out_dir(dataset, "film_upgrade")
    mdir = fm.model_dir(dataset) / "film_upgrade"
    out.mkdir(parents=True, exist_ok=True)
    recorded = pd.read_csv(fm.out_dir(dataset, "train_scratch") / "train_summary.csv").set_index(["seed", "fold"])
    base_pred = pd.read_csv(fm.out_dir(dataset, "validation_study") / "validation_predictions.csv")
    base_pred = base_pred[base_pred["model"].isin(BASELINES)]
    source = pd.read_csv(ROOT / CROWD_DATASET_PATH, usecols=["message_id", "source"]).set_index("message_id")["source"]

    cells, preds = [], []
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            parts = load_fold(seed, fold)
            frame = parts["val"]
            test_ids = set(parts["test"]["message_id"])
            assert not (set(frame["message_id"]) & test_ids)
            words = frame["sms_text"].map(lambda t: len(vb.strip_links(t).split())).to_numpy()
            has_url = frame["has_url"].astype(int).to_numpy()
            for name, (options, extra) in VARIANTS.items():
                model, w2i, seq_len = film.build_film(seed, fold, False, None, **options)
                assert film.trainable_counts(model)[0] == film.expected_trainable(
                    link_dim=model.link_dim, generator_hidden=model.generator_hidden, head_link=model.head_link)
                train, val = (part_inputs(seed, fold, p, w2i, seq_len, extra_link_columns=extra) for p in ("train", "val"))
                assert list(val.message_id) == list(frame["message_id"])
                result = fm.fit(model, (train.token_ids, train.link_full, train.link_blank, train.labels),
                                (val.token_ids, val.link_full, val.labels), seed, mdir / f"{name}_seed{seed}_fold{fold}.pt")
                with torch.no_grad():
                    p = torch.softmax(model(val.token_ids, val.link_full), dim=1)[:, 1].numpy()
                y, pred = val.labels.numpy(), (p >= 0.5).astype(int)
                legit = y == 0
                lab = legit & (frame["source"].to_numpy() == "web_legit_lab")
                bare = legit & (words <= 3) & (has_url == 1)
                row = {"variant": name, "seed": seed, "fold": fold, "n_val": len(y),
                       "val_balanced_accuracy": balanced_accuracy_score(y, pred), "val_f1": f1_score(y, pred),
                       "recorded_val_balanced_accuracy": result.metrics["val_balanced_accuracy"],
                       "false_alarms": int((pred[legit] == 1).sum()), "misses": int((pred[~legit] == 0).sum()),
                       "legit_lab_false_alarms": int((pred[lab] == 1).sum()), "legit_lab_rows": int(lab.sum()),
                       "bare_link_false_alarms": int((pred[bare] == 1).sum()), "bare_link_rows": int(bare.sum()),
                       "best_epoch": result.best_epoch, "epochs_run": result.epochs_run,
                       "trainable": film.trainable_counts(model)[0]}
                if name == "control":
                    row["replay_gap"] = abs(result.metrics["val_balanced_accuracy"] - recorded.loc[(seed, fold), "val_balanced_accuracy"])
                cells.append(row)
                preds.append(pd.DataFrame({"variant": name, "seed": seed, "fold": fold, "message_id": val.message_id,
                                           "label": y, "p_smishing": p}))
                print(f"seed {seed} fold {fold} {name:8}: val bal-acc {row['val_balanced_accuracy']:.4f} | false alarms "
                      f"{row['false_alarms']} (lab {row['legit_lab_false_alarms']}) misses {row['misses']} | best epoch "
                      f"{row['best_epoch']} of {row['epochs_run']}", flush=True)

    cells = pd.DataFrame(cells)
    preds = pd.concat(preds, ignore_index=True)
    for (seed, fold), g in preds.groupby(["seed", "fold"]):          # only val rows were written
        parts = load_fold(seed, fold)
        assert set(g["message_id"]) == set(parts["val"]["message_id"])
        assert not (set(g["message_id"]) & set(parts["test"]["message_id"]))
    cells.to_csv(out / "cells.csv", index=False)
    preds.to_csv(out / "val_predictions.csv", index=False)

    # ---------------------------------------------------------------- projected McNemar per seed
    preds["ok"] = ((preds["p_smishing"] >= 0.5).astype(int) == preds["label"]).astype(int)
    base_pred = base_pred.assign(ok=((base_pred["p_smishing"] >= 0.5).astype(int) == base_pred["label"]).astype(int))
    key = ["seed", "fold", "message_id"]
    mc = []
    for name in VARIANTS:
        mine = preds[preds["variant"] == name][key + ["ok"]]
        for bname, short in BASELINES.items():
            theirs = base_pred[base_pred["model"] == bname][key + ["ok"]]
            both = mine.merge(theirs, on=key, suffixes=("", "_base"), validate="one_to_one")
            assert len(both) == len(mine)
            for seed, g in both.groupby("seed"):
                b = int(((g["ok"] == 1) & (g["ok_base"] == 0)).sum())
                c = int(((g["ok"] == 0) & (g["ok_base"] == 1)).sum())
                folds_ahead = int(sum(((h["ok"] == 1) & (h["ok_base"] == 0)).sum() > ((h["ok"] == 0) & (h["ok_base"] == 1)).sum()
                                      for _, h in g.groupby("fold")))
                mc.append({"variant": name, "baseline": short, "seed": seed, "b": b, "c": c, "b_minus_c": b - c,
                           "p_proxy": round(mcnemar_p(b, c), 4), "folds_ahead": folds_ahead})
    mc = pd.DataFrame(mc)
    mc.to_csv(out / "mcnemar_projection.csv", index=False)

    # ---------------------------------------------------------------- keep rule
    control = cells[cells["variant"] == "control"].set_index(["seed", "fold"])["val_balanced_accuracy"]
    lines = []
    replay = cells.loc[cells["variant"] == "control", "replay_gap"].max()
    lines.append(f"control replay: largest gap to the recorded Proposed validation balanced accuracy = {replay:.6f} (must be 0)")
    base_ba = {short: base_pred[base_pred["model"] == bname].groupby(["seed", "fold"]).apply(
        lambda g: balanced_accuracy_score(g["label"], (g["p_smishing"] >= 0.5).astype(int))).mean()
        for bname, short in BASELINES.items()}
    lines.append("baselines' mean validation balanced accuracy: " + ", ".join(f"{k} {v:.4f}" for k, v in base_ba.items()))
    table = []
    for name in VARIANTS:
        g = cells[cells["variant"] == name].set_index(["seed", "fold"])
        wins = int((g["val_balanced_accuracy"] > control).sum()) if name != "control" else 0
        m = mc[mc["variant"] == name]
        worst = int(m["b_minus_c"].min())
        kept = name != "control" and g["val_balanced_accuracy"].mean() > control.mean() and wins >= MIN_WINS and worst > 0
        table.append({"variant": name, "val_bal_acc": round(g["val_balanced_accuracy"].mean(), 4),
                      "val_f1": round(g["val_f1"].mean(), 4), "wins_vs_control": wins,
                      "false_alarms_per_fold": round(g["false_alarms"].mean(), 1), "misses_per_fold": round(g["misses"].mean(), 1),
                      "legit_lab_FA": int(g["legit_lab_false_alarms"].sum()), "bare_link_FA": f"{int(g['bare_link_false_alarms'].sum())}/{int(g['bare_link_rows'].sum())}",
                      "worst_seed_b_minus_c": worst, "kept": kept})
    table = pd.DataFrame(table)
    pd.set_option("display.width", 250)
    lines += ["", table.to_string(index=False), "", "projected McNemar per seed (validation rows, proxy):",
              mc.pivot_table(index=["variant", "baseline"], columns="seed", values="b_minus_c").to_string()]
    kept = table[table["kept"]]
    lines.append("")
    lines.append("KEPT: " + (", ".join(kept.sort_values("worst_seed_b_minus_c", ascending=False)["variant"]) if len(kept) else
                             "none (no candidate passes the keep rule; stop and report)"))
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
