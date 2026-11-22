r"""B3 of docs/validation_only_improvement_plan.md, judged on VALIDATION rows only. Nothing here reads a test row.

    .venv\Scripts\python.exe scripts/b3_variants_validation.py

Why this is not "start FiLM near do-nothing": the generator already starts at zero (gamma = 1, beta = 0) in every Proposed model, so that
change was already in place. The finding of phase A is about messages whose text is (almost) empty: the generator reads the average of the
Conv scores over all 50 positions, which for a bare link is almost only padding. Two variants the model already supports, fixed before running:
  control        the unchanged Proposed model (from scratch, has_url bypass), replayed here; must reproduce the recorded validation metrics
  C1_mask        mask_padding=True: the average only counts positions whose window holds a real word (message length no longer matters)
  C2_link_only   use_average=False: the generator reads the link numbers only, never the text (variant V3 of the plan)
Same data, seeds, folds, recipe (fit_model.fit), early stopping on validation balanced accuracy. Models are written to
models/film/<dataset>/b3_variants/, results to experiments/film/<dataset>/b3_variants/.

Keep rule (from the plan): a variant is kept only if its mean validation balanced accuracy beats the control's AND it is better in at least
11 of the 15 seed-fold pairs. Secondary, descriptive only: false-alarm rate on legitimate bare-link messages (3 words or fewer without the link)
and on the longer legitimate messages. Reference points: Concatenation's validation balanced accuracy from its own training summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CROWD_N_FOLDS, SEEDS                              # noqa: E402
from src.data.split_crowd import load_fold                               # noqa: E402
from src.unified import film                                             # noqa: E402
from src.unified import fit_model as fm                                  # noqa: E402
from src.unified import text_cnn as tc                                   # noqa: E402
from src.unified import voting_baseline as vb                            # noqa: E402
from src.unified.film_data import part_inputs                            # noqa: E402

VARIANTS = {"control": {}, "C1_mask": {"mask_padding": True}, "C2_link_only": {"use_average": False}}


def main() -> int:
    dataset = tc.dataset_path(False)
    out = fm.out_dir(dataset, "b3_variants")
    mdir = fm.model_dir(dataset) / "b3_variants"
    out.mkdir(parents=True, exist_ok=True)
    rows, preds = [], []
    for seed in SEEDS:
        for fold in range(CROWD_N_FOLDS):
            frame = load_fold(seed, fold)["val"]
            words = frame["sms_text"].map(lambda t: len(vb.strip_links(t).split())).to_numpy()
            has_url = frame["has_url"].astype(int).to_numpy()
            for name, options in VARIANTS.items():
                model, w2i, seq_len = film.build_film(seed, fold, False, None, **options)
                train, val = (part_inputs(seed, fold, p, w2i, seq_len) for p in ("train", "val"))
                assert list(val.message_id) == list(frame["message_id"])
                result = fm.fit(model, (train.token_ids, train.link_full, train.link_blank, train.labels),
                                (val.token_ids, val.link_full, val.labels), seed, mdir / f"{name}_seed{seed}_fold{fold}.pt")
                with torch.no_grad():
                    p = torch.softmax(model(val.token_ids, val.link_full), dim=1)[:, 1].numpy()
                y, pred = val.labels.numpy(), (p >= 0.5).astype(int)
                short = (words <= 3) & (has_url == 1)
                rows.append({"variant": name, "seed": seed, "fold": fold, "best_epoch": result.best_epoch,
                             "val_balanced_accuracy": balanced_accuracy_score(y, pred), "val_f1": f1_score(y, pred),
                             "recorded_balanced_accuracy": result.metrics["val_balanced_accuracy"],
                             "false_alarms": int(((y == 0) & (pred == 1)).sum()), "misses": int(((y == 1) & (pred == 0)).sum()),
                             "bare_link_legit_rows": int(((y == 0) & short).sum()), "bare_link_false_alarms": int(((y == 0) & short & (pred == 1)).sum()),
                             "other_legit_rows": int(((y == 0) & ~short).sum()), "other_false_alarms": int(((y == 0) & ~short & (pred == 1)).sum())})
                preds.append(pd.DataFrame({"variant": name, "seed": seed, "fold": fold, "message_id": val.message_id, "label": y, "p_smishing": p}))
            r = [x for x in rows if x["seed"] == seed and x["fold"] == fold]
            print(f"seed {seed} fold {fold}: " + " | ".join(f"{x['variant']} {x['val_balanced_accuracy']:.4f}" for x in r), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "b3_results.csv", index=False)
    pd.concat(preds).to_csv(out / "b3_validation_predictions.csv", index=False)

    rec = pd.read_csv(out.parent / "train_scratch" / "train_summary.csv")[["seed", "fold", "val_balanced_accuracy"]]
    c = res[res.variant == "control"].merge(rec, on=["seed", "fold"], suffixes=("", "_recorded_by_group"))
    gap = float((c.val_balanced_accuracy - c.val_balanced_accuracy_recorded_by_group).abs().max())
    cat = pd.read_csv(out.parent / "concat_scratch" / "train_summary.csv")["val_balanced_accuracy"].mean()
    print(f"\ncontrol replay vs the group's recorded validation balanced accuracy: largest difference {gap:.6f}")
    print(f"Concatenation (recorded) mean validation balanced accuracy: {cat:.4f}\n")
    ctl = res[res.variant == "control"].set_index(["seed", "fold"])
    for name in ["control", "C1_mask", "C2_link_only"]:
        v = res[res.variant == name].set_index(["seed", "fold"])
        better = int((v.val_balanced_accuracy > ctl.val_balanced_accuracy).sum()) if name != "control" else 0
        print(f"{name:13s} val bal-acc {v.val_balanced_accuracy.mean():.4f} | F1 {v.val_f1.mean():.4f} | better than control in {better}/15 | "
              f"false alarms/fold {v.false_alarms.mean():.1f} misses/fold {v.misses.mean():.1f} | bare-link false alarms "
              f"{int(v.bare_link_false_alarms.sum())}/{int(v.bare_link_legit_rows.sum())} | other-legit false alarms "
              f"{v.other_false_alarms.sum() / v.other_legit_rows.sum():.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
