"""Two tests of whether E2 really uses the link (plan docs/film_architecture_plan.md, steps 5.9 and 5.10). Nothing is trained here: they score
models that already exist, on the test part, so on the real data they are a test read (--read-test, once, logged).

Step 5.9, link-shuffle control. At test time every message is given ANOTHER message's link numbers: the link rows of the messages that
have a link are permuted among themselves (all link columns, `has_url` stays with its message). If a model uses the link, its accuracy
falls toward the text-only model's; if accuracy barely changes, it was not using the link. Several permutations per fold are averaged
(the permutations are the same for every model, so every model sees the same wrong links). The text-only CNN ignores the link, so it is
the floor, and its accuracy does not change. Reported per model: accuracy with the right link, with a shuffled link, and the share of
the link's benefit that is lost,
    lost = (accuracy right - accuracy shuffled) / (accuracy right - accuracy of the text-only CNN).
1 means all of the link's benefit depended on the right link; 0 means the link was not used.

Step 5.10, shared-wording stress test. A message is "shared wording" when its template (template_id) occurs in BOTH classes in the dataset
and the rarer class holds at least `min_share` of the template's messages (default 0.2): the words are then no guide to the class, so
the text-only CNN should sit near chance and only the link can help. E2 can use the link there only through beta, so this is where
FiLM is most likely to lose to concatenation. Reported per model: accuracy on the shared-wording messages and on the rest, complete
pages and page hidden, and exact McNemar on the shared-wording messages (E2 against each other model, Holm per seed).
The definition comes from the dataset's own `template_id`, so it works on the real data too if the crowd file has that column.

Models scored are those whose checkpoints exist for every seed and fold: the text-only CNN, E2 (from scratch), E2 without the has_url
bypass, the concatenation control (from scratch), the raw-feature CNN, and any film_ablations variants (--with-ablations).

Writes experiments/film/<dataset>/controls/:
    shuffle_summary.csv        per model: accuracy / F1 with the right and with a shuffled link, mean and std over seeds, share lost
    shuffle_per_seed.csv       the same per seed
    shared_wording_summary.csv per model and condition: accuracy / F1 on the shared-wording messages and on the rest
    shared_wording_mcnemar.csv E2 against each other model on the shared-wording messages
    shared_wording_ids.csv     the message ids in the subset (so the group can inspect them)
    test_predictions.csv       complete and page-hidden predictions (shuffled ones are not kept)
    test_reads.log             one line per run

Usage:
    python -m src.unified.film_controls --pilot
    python -m src.unified.film_controls --pilot --with-ablations --shuffles 5 --min-share 0.2
    python -m src.unified.film_controls --read-test                  # the real crowd dataset
"""
from __future__ import annotations

import argparse
import datetime
import sys
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch

from src.config import CROWD_N_FOLDS, SEEDS
from src.unified import film_ablations as fa
from src.unified import film_compare as fc
from src.unified import fit_model as fm
from src.unified import raw_cnn_control as rc
from src.unified import text_cnn as tc
from src.unified.film_data import PartData, part_inputs

MODELS = [fc.BASELINE, fc.E2_SCRATCH, fc.E2_NOBYPASS, fc.CONTROL_SCRATCH, fc.RAW_CNN]
SHUFFLES = 5
MIN_SHARE = 0.2


# ------------------------------------------------------------------ the two controls, as plain functions
def shuffle_link(link: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """`link` (n, k) with every column but the last given to another message. The last column is has_url: it stays with its message,
    and only the messages that have a link (has_url = 1) trade numbers, so a no-link message is never handed someone's link."""
    out = link.clone()
    linked = (link[:, -1] == 1).nonzero().squeeze(1)
    if len(linked) > 1:
        donors = linked[torch.randperm(len(linked), generator=generator)]
        out[linked, :-1] = link[donors, :-1]
    return out


def shared_wording_templates(templates: pd.Series, labels: pd.Series, min_share: float = MIN_SHARE) -> set:
    """The template ids that occur in both classes, the rarer class holding at least `min_share` of the template's messages."""
    table = pd.crosstab(templates, labels)
    if table.shape[1] < 2:
        return set()
    share = table.div(table.sum(axis=1), axis=0).min(axis=1)
    return set(table.index[(table.min(axis=1) > 0) & (share >= min_share)])


def link_benefit_lost(acc_right: float, acc_shuffled: float, acc_text_only: float) -> float:
    """The share of the link's benefit that disappears when the link is shuffled; nan when the model gains nothing from the link."""
    gain = acc_right - acc_text_only
    return float("nan") if gain <= 0 else (acc_right - acc_shuffled) / gain


# ------------------------------------------------------------------ which models are scored
def model_specs(pilot: bool, seeds: List[int], folds: List[int], with_ablations: bool) -> Dict[str, dict]:
    """name -> {"load": (seed, fold, pilot) -> (model, vocab, seq_len), "part": (seed, fold, part, vocab, seq_len, pilot) -> PartData}."""
    dataset = tc.dataset_path(pilot)
    specs: Dict[str, dict] = {}
    for name in MODELS:
        kind, load = fc.LOADERS[name]
        if all(fm.checkpoint_path(dataset, kind, s, f).exists() for s in seeds for f in folds):
            part = rc.raw_part_inputs if name == fc.RAW_CNN else part_inputs
            specs[name] = {"load": load, "part": part}
    if with_ablations:
        for variant in fa.VARIANTS:
            if all(fm.checkpoint_path(dataset, fa.kind_of(variant), s, f).exists() for s in seeds for f in folds):
                specs[fa.label_of(variant)] = {
                    "load": (lambda s, f, p, v=variant: fa.load_variant(v, s, f, p)[:3]),
                    "part": (lambda s, f, part, vocab, n, p, v=variant: fa.variant_part(v, s, f, part, vocab, n, p))}
    for needed in (fc.BASELINE, fc.E2_SCRATCH):
        if needed not in specs:
            raise FileNotFoundError(f"{needed}: no trained models for every seed and fold; train them first")
    return specs


def probabilities(model, test: PartData, link: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(test.token_ids, link), dim=1)[:, 1].numpy()


# ------------------------------------------------------------------ one pass over the test part
def collect(pilot: bool, specs: Dict[str, dict], seeds: List[int], folds: List[int], shuffles: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(predictions for complete / page hidden, predictions with shuffled links), one row per message, model and condition."""
    kept, shuffled = [], []
    for seed in seeds:
        for fold in folds:
            loaded = {name: spec["load"](seed, fold, pilot) for name, spec in specs.items()}
            vocab, length = loaded[fc.BASELINE][1], loaded[fc.BASELINE][2]
            assert all(v == vocab and n == length for _, v, n in loaded.values()), "the models disagree on the fold's vocabulary"
            base = part_inputs(seed, fold, "test", vocab, length, pilot)
            meta = {"seed": seed, "fold": fold, "message_id": base.message_id, "label": base.labels.numpy(),
                    "has_url": base.link_full[:, 3].numpy().astype(int), "page_live": (1 - base.link_full[:, 2]).numpy().astype(int)}
            for name, spec in specs.items():
                model = loaded[name][0]
                test = spec["part"](seed, fold, "test", vocab, length, pilot)
                assert np.array_equal(test.message_id, base.message_id), f"{name}: the test rows are not in the same order"
                for condition, link in (("complete", test.link_full), ("page hidden", test.link_blank)):
                    p = probabilities(model, test, link)
                    kept.append(pd.DataFrame({**meta, "model": name, "condition": condition, "p_smishing": p, "pred": (p >= 0.5).astype(int)}))
                for k in range(shuffles):
                    generator = torch.Generator().manual_seed(1000 * seed + 10 * fold + k)       # the same permutation for every model
                    p = probabilities(model, test, shuffle_link(test.link_full, generator))
                    shuffled.append(pd.DataFrame({**meta, "model": name, "shuffle": k, "p_smishing": p, "pred": (p >= 0.5).astype(int)}))
            print(f"seed {seed} fold {fold}: {len(specs)} models scored", flush=True)
    kept, shuffled = pd.concat(kept, ignore_index=True), pd.concat(shuffled, ignore_index=True)
    for (seed, model, condition), group in kept.groupby(["seed", "model", "condition"]):
        assert group["message_id"].is_unique, f"seed {seed}: a message was tested twice by {model}"
    return kept, shuffled


# ------------------------------------------------------------------ 5.9 summary
def shuffle_tables(kept: pd.DataFrame, shuffled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (seed, model), g in kept[kept["condition"] == "complete"].groupby(["seed", "model"], sort=False):
        right = fc.scores(g["label"].to_numpy(), g["pred"].to_numpy())
        wrong = [fc.scores(s["label"].to_numpy(), s["pred"].to_numpy())
                 for _, s in shuffled[(shuffled["seed"] == seed) & (shuffled["model"] == model)].groupby("shuffle")]
        rows.append({"seed": seed, "model": model, "n": len(g), "shuffles": len(wrong),
                     "accuracy_right": right["accuracy"], "accuracy_shuffled": float(np.mean([w["accuracy"] for w in wrong])),
                     "f1_right": right["f1"], "f1_shuffled": float(np.mean([w["f1"] for w in wrong]))})
    per_seed = pd.DataFrame(rows)
    floor = per_seed[per_seed["model"] == fc.BASELINE].set_index("seed")["accuracy_right"]
    per_seed["accuracy_text_only"] = per_seed["seed"].map(floor)
    per_seed["link_benefit_lost"] = [link_benefit_lost(r.accuracy_right, r.accuracy_shuffled, r.accuracy_text_only)
                                     for r in per_seed.itertuples()]
    per_seed["accuracy_drop"] = per_seed["accuracy_right"] - per_seed["accuracy_shuffled"]
    cols = ["accuracy_right", "accuracy_shuffled", "accuracy_drop", "f1_right", "f1_shuffled", "link_benefit_lost"]
    summary = per_seed.groupby("model", sort=False)[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return per_seed, summary.reset_index()


# ------------------------------------------------------------------ 5.10 summary
def shared_wording_ids(pilot: bool, min_share: float = MIN_SHARE) -> set:
    data = pd.read_csv(tc.dataset_path(pilot), keep_default_na=False, usecols=["message_id", "template_id", "label"])
    shared = shared_wording_templates(data["template_id"], data["label"], min_share)
    return set(data.loc[data["template_id"].isin(shared), "message_id"])


def shared_tables(kept: pd.DataFrame, ids: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept = kept.assign(shared=kept["message_id"].isin(ids))
    rows = []
    for (seed, model, condition, shared), g in kept.groupby(["seed", "model", "condition", "shared"], sort=False):
        rows.append({"seed": seed, "model": model, "condition": condition, "subset": "shared wording" if shared else "the rest",
                     "n": len(g), "smishing_share": float(g["label"].mean()), **fc.scores(g["label"].to_numpy(), g["pred"].to_numpy())})
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(["subset", "condition", "model"], sort=False)[["n", "smishing_share", "accuracy", "f1"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    subset = kept[(kept["condition"] == "complete") & kept["shared"]]
    others = [m for m in subset["model"].unique() if m != fc.E2_SCRATCH]
    mc = []
    for seed, g in subset.groupby("seed"):
        wide = {m: gm.sort_values("message_id") for m, gm in g.groupby("model")}
        results = [fc.mcnemar(wide[fc.E2_SCRATCH]["label"].to_numpy(), wide[fc.E2_SCRATCH]["pred"].to_numpy(), wide[o]["pred"].to_numpy())
                   for o in others]
        for other, r, p in zip(others, results, fc.holm([r["p_exact"] for r in results])):
            mc.append({"seed": seed, "n": len(wide[other]), "model_a": fc.E2_SCRATCH, "model_b": other, **r, "p_holm": p,
                       "significant_0.05": p < 0.05})
    return summary, pd.DataFrame(mc)


# ------------------------------------------------------------------ run
def run_controls(pilot: bool, read_test: bool = False, seeds: List[int] = SEEDS, folds: List[int] | None = None,
                 shuffles: int = SHUFFLES, min_share: float = MIN_SHARE, with_ablations: bool = False) -> dict:
    dataset = tc.dataset_path(pilot)
    folds = list(range(CROWD_N_FOLDS)) if folds is None else folds
    if not pilot and not read_test:
        raise PermissionError("the test folds of the real data are read once, after everything is fixed: pass --read-test")
    specs = model_specs(pilot, seeds, folds, with_ablations)
    out = fm.out_dir(dataset, "controls")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_reads.log", "a", encoding="utf-8") as log:
        log.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} dataset={dataset.name} seeds={seeds} folds={folds} "
                  f"read_test_flag={read_test} shuffles={shuffles} min_share={min_share} models={list(specs)}\n")
    kept, shuffled = collect(pilot, specs, seeds, folds, shuffles)
    shuffle_per_seed, shuffle_summary = shuffle_tables(kept, shuffled)
    ids = shared_wording_ids(pilot, min_share)
    shared_summary, shared_mcnemar = shared_tables(kept, ids)
    kept.to_csv(out / "test_predictions.csv", index=False)
    shuffle_per_seed.to_csv(out / "shuffle_per_seed.csv", index=False)
    shuffle_summary.to_csv(out / "shuffle_summary.csv", index=False)
    shared_summary.to_csv(out / "shared_wording_summary.csv", index=False)
    shared_mcnemar.to_csv(out / "shared_wording_mcnemar.csv", index=False)
    pd.DataFrame({"message_id": sorted(ids)}).to_csv(out / "shared_wording_ids.csv", index=False)
    return {"shuffle_summary": shuffle_summary, "shuffle_per_seed": shuffle_per_seed, "shared_summary": shared_summary,
            "shared_mcnemar": shared_mcnemar, "ids": ids, "models": list(specs)}


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a rehearsal; its numbers are not study results)")
    ap.add_argument("--read-test", action="store_true", help="required for the real data: reads the test folds")
    ap.add_argument("--with-ablations", action="store_true", help="also score the film_ablations variants that are trained")
    ap.add_argument("--shuffles", type=int, default=SHUFFLES, help="permutations per fold (default %(default)s)")
    ap.add_argument("--min-share", type=float, default=MIN_SHARE, help="the rarer class's least share of a shared-wording template")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    args = ap.parse_args(argv)
    try:
        r = run_controls(args.pilot, args.read_test, args.seeds, args.folds, args.shuffles, args.min_share, args.with_ablations)
    except PermissionError as error:
        print(error)
        sys.exit(2)
    pd.set_option("display.width", 220)
    show = r["shuffle_summary"][["model", "accuracy_right_mean", "accuracy_shuffled_mean", "accuracy_drop_mean", "link_benefit_lost_mean",
                                 "link_benefit_lost_std"]]
    print("\nLink-shuffle control (accuracy; share of the link's benefit lost):")
    print(show.round(4).to_string(index=False))
    print(f"\nShared-wording messages: {len(r['ids'])} in the dataset")
    s = r["shared_summary"]
    for subset in ("shared wording", "the rest"):
        print(f"\n{subset} (mean accuracy over seeds)")
        print(s[s["subset"] == subset].pivot(index="model", columns="condition", values="accuracy_mean").round(3).to_string())
        print(f"n per seed about {s[s['subset'] == subset]['n_mean'].iloc[0]:.0f}, smishing share {s[s['subset'] == subset]['smishing_share_mean'].iloc[0]:.2f}")
    print("\nMcNemar on the shared-wording messages (exact, complete pages), E2 against each model, Holm per seed:")
    print(r["shared_mcnemar"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
