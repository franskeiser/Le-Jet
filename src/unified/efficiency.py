"""SOP 2: the computational efficiency of the trained models (plan docs/film_architecture_plan.md, D23 and steps 4.3a to 4.3e).

The group's instrument ("SOP 2 Instrument, Computational Efficiency", 2026-10-20) rewritten for this code (PyTorch, not Keras). For each
trained model it measures, in a NEW Python process, on the test part of one fold, one message at a time, CPU only, one thread:
    2.1 model size           MB, the sum of the files a deployment needs (Form B)
    2.2 peak memory          MB, the highest resident memory while the model is loaded and the test part classified once, minus the
                             memory before the model was loaded; psutil RSS sampled every 5 ms (Form C)
    2.3 inference runtime    ms per message, mean / median / SD, overall and with / without a link, after 10 untimed warm-up messages (Form D)
Not measured: training, and the URL / page inspection (network speed, not the model); the 37 features are extracted before the clock starts.
Answered descriptively: mean +- SD over the 3 seeds, no significance test (Form E).

Rules this study adds to the instrument (step 4.3a):
  * Which models: fold 0 of seeds 42, 123, 2024 (`--fold`), trained from scratch (D7a). 3 runs each, so 7 models x 3 seeds x 3 runs = 63 runs.
  * The Link RF is timed on P and u only for every model that uses it (the generator reads only P, u and page_missing), so that the FiLM model and the
    concatenation control, which read the same inputs, differ only by the generator. `proposed_contrib` is the FiLM model with the forest also
    computing the 37 contributions (D21), listed apart so the cost of that choice is visible.
  * No calibrator (raw P, D9). The voting model has no URL classifier here (voting_baseline.py), so its files are the TF-IDF vectorizer and the vote.
  * A model is loaded as a deployment would: the architecture from its sidecar, the weights from the .pt file (the trained embedding is in
    them; the starting fastText matrix is not loaded), the vocabulary from vocab.json.
  * Same libraries imported by every process before the memory baseline is read, so library overhead is the same for every model.

Models (key: what, files counted):
    proposed          the FiLM model (film_scratch): .pt + sidecar .json + vocab.json + Link RF .joblib
    proposed_contrib  the same files; the forest also computes the 37 contributions (extra row)
    cnn_only          the text-only CNN: .pt + .json + vocab.json
    rf_only           the Link RF .joblib alone; messages with a link only
    concat            the concatenation control (concat_scratch): .pt + .json + vocab.json + Link RF .joblib
    raw_cnn           CNN-only with the raw 37 features (cnn_raw): .pt + .json + vocab.json (extra row)
    voting            KNN + RF + ExtraTrees vote on TF-IDF: one .joblib (made by `save-voting`)

Writes experiments/film/<dataset>/efficiency/: form_b_model_size.csv, form_c_peak_memory.csv, form_d_runtime.csv,
descriptive_details.csv (parameter counts, forest nodes), form_a_environment.json, form_e_summary.csv and .md.

Usage (leave the machine alone while `run-all` runs: other programs change the timings):
    python -m src.unified.efficiency save-voting --pilot                 # the voting model is not saved by voting_baseline.py
    python -m src.unified.efficiency describe --pilot --model proposed --seed 42     # parameters, forest nodes (own process)
    python -m src.unified.efficiency measure --pilot --model proposed --seed 42 --run 1
    python -m src.unified.efficiency run-all --pilot                     # every model x seed x run, each in a new process, random order
    python -m src.unified.efficiency run-all --pilot --resume            # after an interruption: only the jobs not done yet
    python -m src.unified.efficiency summarize --pilot
    add --limit 300 to measure / run-all for a quick check on the first 300 test messages
"""
from __future__ import annotations

import os

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")             # CPU only for every model (run-all sets it for each process)
for _var in THREAD_VARIABLES:
    os.environ.setdefault(_var, "1")                            # one thread, set before numpy / torch are imported

import argparse
import csv
import datetime
import gc
import json
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.config import CROWD_N_FOLDS, SEEDS
from src.unified import text_cnn as tc

MB = 1024 ** 2
RUNS = (1, 2, 3)
WARMUP = 10
SAMPLE_SECONDS = 0.005
HIDDEN_PAGE_CUTOFF = 0.5

# key -> label, checkpoint kind (fit_model.checkpoint_path), whether it reads the Link RF, extra row?
MODELS: Dict[str, dict] = {
    "proposed": {"label": "Proposed (CNN–RF + FiLM)", "kind": "film_scratch", "forest": True, "extra": False},
    "cnn_only": {"label": "CNN-only (text)", "kind": "cnn", "forest": False, "extra": False},
    "rf_only": {"label": "RF-only (Link RF)", "kind": None, "forest": True, "extra": False},
    "concat": {"label": "Concatenation fusion", "kind": "concat_scratch", "forest": True, "extra": False},
    "voting": {"label": "KNN-RF-ETC voting (text only)", "kind": None, "forest": False, "extra": False},
    "proposed_contrib": {"label": "Proposed, forest also computes the 37 contributions", "kind": "film_scratch", "forest": True, "extra": True},
    "raw_cnn": {"label": "CNN-only, text + raw 37 features", "kind": "cnn_raw", "forest": False, "extra": True},
}


# ------------------------------------------------------------------ where things are
def out_dir(dataset: Path) -> Path:
    from src.unified import fit_model as fm
    return fm.out_dir(dataset, "efficiency")


def voting_path(dataset: Path, seed: int, fold: int) -> Path:
    from src.unified import fit_model as fm
    return fm.model_dir(dataset) / f"voting_seed{seed}_fold{fold}.joblib"


def component_files(model: str, dataset: Path, seed: int, fold: int) -> List[tuple]:
    """[(component, path)] a deployment of `model` needs: everything inference reads, nothing it does not (no dataset, log, sidecar hashes
    of other files, or starting embedding matrix)."""
    from src.unified import film
    from src.unified import fit_model as fm
    spec = MODELS[model]
    if model == "voting":
        return [("TF-IDF vectorizer + KNN/RF/ExtraTrees vote", voting_path(dataset, seed, fold))]
    files = []
    if spec["kind"]:
        checkpoint = fm.checkpoint_path(dataset, spec["kind"], seed, fold)
        files += [("weights", checkpoint), ("configuration", checkpoint.with_suffix(".json")),
                  ("vocabulary", tc.embeddings_dir(dataset, seed, fold) / "vocab.json")]
    if spec["forest"]:
        files.append(("Link RF", film.link_rf_path(dataset, seed, fold)))
    return files


# ------------------------------------------------------------------ the test messages, features pre-extracted
def load_messages(pilot: bool, seed: int, fold: int, limit: int | None = None) -> List[dict]:
    """The fold's test part as one dict per message, in file order. Everything the pipelines need is prepared here, before the clock
    starts and before the memory baseline: the SMS text, has_url, the 37 features as a one-row frame, page_missing, the raw-feature row."""
    import torch
    from src.data.split_crowd import load_fold
    from src.unified.prepare_web_37 import FEATURES_37
    from src.unified.raw_cnn_control import raw_link_matrix
    rows = load_fold(seed, fold, pilot=pilot)["test"]
    if limit:
        rows = rows.iloc[:limit].reset_index(drop=True)
    has_url = rows["has_url"].astype(int).to_numpy() if "has_url" in rows.columns else np.ones(len(rows), dtype=int)
    page_missing = 1 - rows["page_fetched"].astype(int).to_numpy()
    raw = torch.tensor(raw_link_matrix(rows))
    return [{"text": rows["sms_text"].iloc[i], "has_url": int(has_url[i]), "page_missing": int(page_missing[i]),
             "features": rows.iloc[[i]][FEATURES_37], "raw": raw[i:i + 1], "label": int(rows["label"].iloc[i] == "smishing")}
            for i in range(len(rows))]


# ------------------------------------------------------------------ the pipelines (what one message goes through)
def import_all_libraries() -> None:
    """Import every library any model needs, so each process has the same baseline memory whatever it runs."""
    import joblib  # noqa: F401
    import sklearn  # noqa: F401
    import torch
    from src.features.model import jain_baseline  # noqa: F401
    from src.unified import cnn_baseline, concat_control, film, raw_cnn_control, voting_baseline  # noqa: F401
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:                                  # torch allows this once per process; a second call (or parallel work already
        pass                                              # started) is fine only if it is already 1
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("torch is not limited to one thread: the timings would not be comparable")


def deployed_torch_model(model: str, dataset: Path, seed: int, fold: int):
    """(network in eval mode, word_to_idx, sequence length, sidecar): the architecture from the sidecar, the weights from the .pt file."""
    import torch
    import torch.nn as nn
    from src.features.model.concat_cnn import ConcatTextCNN
    from src.features.model.embedding import PAD_TOKEN
    from src.features.model.film_cnn import FiLMTextCNN
    from src.features.model.plain_cnn import PlainTextCNN
    from src.features.model.raw_cnn import RawFeatureCNN
    from src.unified import fit_model as fm
    kind = MODELS[model]["kind"]
    checkpoint = fm.checkpoint_path(dataset, kind, seed, fold)
    sidecar = fm.read_sidecar(checkpoint)
    vocab = json.loads((tc.embeddings_dir(dataset, seed, fold) / "vocab.json").read_text(encoding="utf-8"))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    embedding = nn.Embedding(len(vocab), 300, padding_idx=word_to_idx[PAD_TOKEN])         # the trained weights come with the checkpoint
    if kind == "film_scratch":
        net = FiLMTextCNN(embedding, sidecar["num_filters"], sidecar["kernel_size"], sidecar["link_dim"], sidecar["dense_units"],
                          **sidecar["options"])
    elif kind == "cnn":
        net = PlainTextCNN(embedding, sidecar["num_filters"], sidecar["kernel_size"], freeze_embedding=sidecar["freeze_embedding"])
    elif kind == "concat_scratch":
        net = ConcatTextCNN(embedding, sidecar["num_filters"], sidecar["kernel_size"], sidecar["link_dim"], sidecar["dense_units"],
                            freeze_embedding=sidecar["freeze_embedding"])
    else:
        net = RawFeatureCNN(embedding, num_filters=sidecar["num_filters"], kernel_size=sidecar["kernel_size"],
                            branch_units=sidecar["branch_units"], dense_units=sidecar["dense_units"],
                            freeze_embedding=sidecar["freeze_embedding"])
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return net.eval(), word_to_idx, sidecar["seq_len_l"], sidecar


def build_classifier(model: str, pilot: bool, seed: int, fold: int) -> Callable[[dict], int]:
    """classify(message) -> 0 / 1, loading everything the model needs. This is what is timed message by message."""
    import joblib
    import torch
    from src.features.model.fusion_data import encode_batch_tolerant
    from src.features.model.rf_extractor import p_and_u
    from src.unified import film
    from src.unified import link_lane as ll
    dataset = tc.dataset_path(pilot)

    def forest():
        rf = film.load_link_rf(dataset, seed, fold)
        rf.n_jobs = 1
        return rf

    if model == "voting":
        from src.unified import voting_baseline as vb
        bundle = joblib.load(voting_path(dataset, seed, fold))
        tfidf, vote = bundle["tfidf"], bundle["text_clf"]
        for est in vote.estimators_:
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1
        return lambda msg: int(vote.predict(tfidf.transform([vb._prep(msg["text"])]))[0])

    if model == "rf_only":
        rf = forest()
        return lambda msg: int(p_and_u(rf, msg["features"])[0][0] >= HIDDEN_PAGE_CUTOFF)

    net, word_to_idx, seq_len, _ = deployed_torch_model(model, dataset, seed, fold)
    no_link = torch.tensor([[0.0, 0.0, 1.0, 0.0]])                       # P, u, page_missing, has_url of a message with no link

    def tokens(msg):
        return encode_batch_tolerant([msg["text"]], word_to_idx, seq_len)

    if model == "cnn_only":
        def classify(msg):
            with torch.no_grad():
                return int(net(tokens(msg), None).argmax(dim=1))
        return classify
    if model == "raw_cnn":
        def classify(msg):
            with torch.no_grad():
                return int(net(tokens(msg), msg["raw"]).argmax(dim=1))
        return classify

    rf = forest()
    with_contributions = model == "proposed_contrib"

    def classify(msg):
        ids = tokens(msg)
        if msg["has_url"]:                                             # a message with no link never reaches the forest (the bypass)
            if with_contributions:
                scored = ll.score(rf, msg["features"])                 # P, u and the 37 contributions
                p, u = float(scored["P"].iloc[0]), float(scored["u"].iloc[0])
            else:
                p_arr, u_arr = p_and_u(rf, msg["features"])
                p, u = float(p_arr[0]), float(u_arr[0])
            link = torch.tensor([[p, u, float(msg["page_missing"]), 1.0]])
        else:
            link = no_link
        with torch.no_grad():
            return int(net(ids, link).argmax(dim=1))
    return classify


# ------------------------------------------------------------------ the voting model is saved here
def save_voting(pilot: bool, seeds: List[int] = SEEDS, folds: List[int] | None = None) -> List[Path]:
    """Fit the voting model on each fold's TRAIN part (as voting_baseline.py does) and save TF-IDF + the vote, the files it needs to predict."""
    import joblib
    from src.data.split_crowd import load_fold
    from src.unified import voting_baseline as vb
    dataset = tc.dataset_path(pilot)
    folds = [0] if folds is None else folds
    paths = []
    for seed in seeds:
        for fold in folds:
            detector = vb.fit_detector(load_fold(seed, fold, pilot=pilot)["train"], seed)
            path = voting_path(dataset, seed, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"tfidf": detector.tfidf, "text_clf": detector.text_clf, "seed": seed, "fold": fold}, path)
            paths.append(path)
            print(f"seed {seed} fold {fold}: saved {path.name} ({path.stat().st_size / MB:.2f} MB)", flush=True)
    return paths


# ------------------------------------------------------------------ one measurement (one model, one seed, one run, one process)
class PeakRSS:
    """Background thread that keeps the highest resident memory the process reaches (psutil, every 5 ms)."""

    def __init__(self, interval: float = SAMPLE_SECONDS):
        import psutil
        self.process, self.interval = psutil.Process(), interval
        self.peak, self._on = self.process.memory_info().rss, False

    def _loop(self):
        while self._on:
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self._on = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._on = False
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def descriptive_details(model: str, dataset: Path, seed: int, fold: int) -> dict:
    """Optional detail beside the size (the instrument's 2.1 note): parameters of the network, trees and nodes of the forest."""
    import joblib
    from src.unified import film
    row = {"model": model, "seed": seed, "fold": fold}
    if MODELS[model]["kind"]:
        net, _, _, sidecar = deployed_torch_model(model, dataset, seed, fold)
        row["parameters_all"] = sum(p.numel() for p in net.parameters())
        row["parameters_trainable_without_embedding"] = sidecar.get("trainable_without_embedding", "")
    if MODELS[model]["forest"]:
        rf = film.load_link_rf(dataset, seed, fold)
        row["forest_trees"] = len(rf.estimators_)
        row["forest_nodes"] = int(sum(t.tree_.node_count for t in rf.estimators_))
    if model == "voting":
        vote = joblib.load(voting_path(dataset, seed, fold))["text_clf"]
        row["forest_trees"] = sum(len(e.estimators_) for e in vote.estimators_ if hasattr(e, "estimators_"))
    return row


def measure(model: str, pilot: bool, seed: int, fold: int, run: int, limit: int | None = None) -> dict:
    """Steps 2 to 10 of the instrument's procedure in this process. Returns the three rows it appended."""
    import torch
    import psutil
    dataset = tc.dataset_path(pilot)
    out = out_dir(dataset)
    files = component_files(model, dataset, seed, fold)

    # 2.1 model size, from the files on disk (it does not change between runs, so it is written on run 1). Nothing is loaded here:
    # loading a model before the baseline is read would leave its memory in the process and hide it from 2.2.
    if run == 1:
        for component, path in files:
            append_row(out / "form_b_model_size.csv", {"model": model, "seed": seed, "fold": fold, "component": component,
                                                        "file": path.name, "size_mb": round(path.stat().st_size / MB, 4)})

    import_all_libraries()
    messages = load_messages(pilot, seed, fold, limit)
    if model == "rf_only":                                                 # the forest cannot speak for a message with no link
        messages = [m for m in messages if m["has_url"]]
    torch.sum(torch.ones(8, 8))                                            # the framework's one-time set-up is part of the baseline
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss

    # 2.2 peak memory: load the model and classify every test message once
    with PeakRSS() as monitor:
        classify = build_classifier(model, pilot, seed, fold)
        for message in messages:
            classify(message)
    memory = {"model": model, "seed": seed, "fold": fold, "run": run, "n": len(messages),
              "rss_baseline_mb": round(baseline / MB, 2), "rss_peak_mb": round(monitor.peak / MB, 2),
              "peak_memory_mb": round((monitor.peak - baseline) / MB, 2)}
    append_row(out / "form_c_peak_memory.csv", memory)

    # 2.3 runtime, monitor off: 10 untimed warm-up messages, then every message one at a time
    for message in messages[:WARMUP]:
        classify(message)
    times, linked = [], []
    for message in messages:
        start = time.perf_counter()
        classify(message)
        times.append((time.perf_counter() - start) * 1000.0)
        linked.append(bool(message["has_url"]))
    t, link = np.array(times), np.array(linked)
    runtime = {"model": model, "seed": seed, "fold": fold, "run": run, "n": len(t), "total_s": round(t.sum() / 1000, 3),
               "mean_ms": round(t.mean(), 3), "median_ms": round(float(np.median(t)), 3), "p95_ms": round(float(np.quantile(t, 0.95)), 3),
               "sd_ms": round(t.std(ddof=1), 3),
               "mean_link_ms": round(t[link].mean(), 3) if link.any() else "",
               "mean_nolink_ms": round(t[~link].mean(), 3) if (~link).any() else ""}
    append_row(out / "form_d_runtime.csv", runtime)
    return {"memory": memory, "runtime": runtime}


# ------------------------------------------------------------------ all runs, in random order, each in a new process
def job_order(models: List[str], seeds: List[int], runs=RUNS, rng_seed: int = 0) -> List[tuple]:
    """(model, seed, run) for every combination: all run-1 jobs first, then run 2, then run 3, shuffled within each run, so no model always
    goes first and every model is measured once before any is measured twice."""
    rng = random.Random(rng_seed)
    order = []
    for run in runs:
        batch = [(m, s, run) for m in models for s in seeds]
        rng.shuffle(batch)
        order += batch
    return order


def environment(pilot: bool) -> dict:
    """Form A: the machine and the library versions, plus what can be checked automatically of the fairness list."""
    import numpy
    import pandas
    import psutil
    import sklearn
    import torch
    battery = psutil.sensors_battery()
    return {"measured_at": datetime.datetime.now().isoformat(timespec="seconds"), "dataset": tc.dataset_path(pilot).name,
            "cpu": platform.processor(), "cores_logical": psutil.cpu_count(), "cores_physical": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1), "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "scikit_learn": sklearn.__version__, "psutil": psutil.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "checked_by_the_script": {"gpu_disabled": not torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
                                      "env_thread_variables": {v: os.environ.get(v) for v in
                                                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                                      "forests_n_jobs": 1, "battery_present": battery is not None,
                                      "plugged_in": None if battery is None else bool(battery.power_plugged)},
            "to_tick_by_hand": ["laptop plugged in and the same power mode for all runs", "other applications closed; no downloads or updates",
                                "same Python environment for every model", "same test messages in the same order for every model"]}


def describe(model: str, pilot: bool, seed: int, fold: int) -> dict:
    """Append the descriptive details of one model (its own process: it loads the model, which would taint a memory measurement)."""
    row = descriptive_details(model, tc.dataset_path(pilot), seed, fold)
    append_row(out_dir(tc.dataset_path(pilot)) / "descriptive_details.csv", row)
    return row


def child_environment() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.update({v: "1" for v in THREAD_VARIABLES})
    return env


def completed_jobs(out: Path) -> set:
    """(model, seed, run) of every measurement already in the logs: a job counts as done only when memory AND runtime were written."""
    done = []
    for name in ("form_c_peak_memory.csv", "form_d_runtime.csv"):
        path = out / name
        if not path.exists():
            return set()
        table = pd.read_csv(path)
        done.append(set(zip(table["model"], table["seed"].astype(int), table["run"].astype(int))))
    return done[0] & done[1]


def run_all(pilot: bool, models: List[str], seeds: List[int], fold: int, limit: int | None, runs=RUNS, resume: bool = False) -> int:
    """Every job in a new process. resume=True keeps the logs and runs only the jobs not done yet (for a run that was interrupted); the
    jobs that are left then run later than the others, under whatever else the computer was doing, so a clean full run is better."""
    dataset = tc.dataset_path(pilot)
    out = out_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        for name in ("form_b_model_size.csv", "form_c_peak_memory.csv", "form_d_runtime.csv", "descriptive_details.csv"):
            if (out / name).exists():
                (out / name).unlink()                                     # a new run replaces the old logs
    jobs = job_order(models, seeds, runs)
    if resume:
        done = completed_jobs(out)
        jobs = [job for job in jobs if job not in done]
        print(f"resuming: {len(done)} jobs already done, {len(jobs)} to run", flush=True)
    root = Path(__file__).resolve().parents[2]
    for model in ([] if resume else models):                              # the descriptive details, one process each, before any measurement
        for seed in seeds:
            subprocess.run([sys.executable, "-m", "src.unified.efficiency", "describe", "--model", model, "--seed", str(seed),
                            "--fold", str(fold)] + (["--pilot"] if pilot else []), check=True, cwd=root, env=child_environment())
    for number, (model, seed, run) in enumerate(jobs, 1):
        command = [sys.executable, "-m", "src.unified.efficiency", "measure", "--model", model, "--seed", str(seed),
                   "--fold", str(fold), "--run", str(run)] + (["--pilot"] if pilot else []) + (["--limit", str(limit)] if limit else [])
        started = time.perf_counter()
        subprocess.run(command, check=True, cwd=root, env=child_environment())
        print(f"[{number}/{len(jobs)}] {model} seed {seed} run {run}: {time.perf_counter() - started:.0f} s", flush=True)
    (out / "form_a_environment.json").write_text(json.dumps(environment(pilot), indent=2), encoding="utf-8")
    return len(jobs)


# ------------------------------------------------------------------ Form E
def summarize(pilot: bool) -> pd.DataFrame:
    """Per model and seed: size summed over its files, peak memory and runtimes averaged over the runs; then mean and SD over the seeds."""
    out = out_dir(tc.dataset_path(pilot))
    size = pd.read_csv(out / "form_b_model_size.csv").groupby(["model", "seed"])["size_mb"].sum().rename("size_mb")
    memory = pd.read_csv(out / "form_c_peak_memory.csv").groupby(["model", "seed"])["peak_memory_mb"].mean()
    runtime = pd.read_csv(out / "form_d_runtime.csv").groupby(["model", "seed"])[["mean_ms", "mean_link_ms", "mean_nolink_ms"]].mean()
    per_seed = pd.concat([size, memory, runtime], axis=1)
    per_seed.to_csv(out / "form_e_per_seed.csv")
    summary = per_seed.groupby("model", sort=False).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reindex([m for m in MODELS if m in summary.index])
    summary.to_csv(out / "form_e_summary.csv")
    (out / "form_e_summary.md").write_text(format_markdown(summary), encoding="utf-8")
    return summary


def pm(mean: float, sd: float, digits: int = 2) -> str:
    if pd.isna(mean):
        return "-"
    return f"{mean:.{digits}f} ± {0.0 if pd.isna(sd) else sd:.{digits}f}"


def format_markdown(summary: pd.DataFrame) -> str:
    lines = ["| Model | Model size (MB) | Peak memory (MB) | Runtime, all (ms/msg) | Runtime, with link | Runtime, no link |",
             "|---|---|---|---|---|---|"]
    for key, r in summary.iterrows():
        label = MODELS[key]["label"] + (" (extra)" if MODELS[key]["extra"] else "")
        lines.append(f"| {label} | {pm(r.size_mb_mean, r.size_mb_std)} | {pm(r.peak_memory_mb_mean, r.peak_memory_mb_std, 1)} | "
                     f"{pm(r.mean_ms_mean, r.mean_ms_std, 3)} | {pm(r.mean_link_ms_mean, r.mean_link_ms_std, 3)} | "
                     f"{pm(r.mean_nolink_ms_mean, r.mean_nolink_ms_std, 3)} |")
    return "\n".join(lines) + "\n\nValues are mean ± SD over the seeds (runs averaged within each seed).\n"


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["save-voting", "describe", "measure", "run-all", "summarize"])
    ap.add_argument("--pilot", action="store_true", help="the synthetic pilot (a dry run; its numbers are not study results)")
    ap.add_argument("--model", choices=list(MODELS), help="measure: which model")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=None, help="run-all: default every model")
    ap.add_argument("--seed", type=int, default=None, help="measure: which seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--fold", type=int, default=0, help="the fold whose models and test part are measured (default 0)")
    ap.add_argument("--run", type=int, default=1, help="measure: repetition number")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test messages (a quick check; the real run uses all)")
    ap.add_argument("--resume", action="store_true", help="run-all: keep the logs and run only the jobs that are not done yet")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    if args.step == "save-voting":
        save_voting(args.pilot, args.seeds, [args.fold])
    elif args.step in ("describe", "measure"):
        if not args.model or args.seed is None:
            ap.error(f"{args.step} needs --model and --seed")
        if args.step == "describe":
            print(json.dumps(describe(args.model, args.pilot, args.seed, args.fold)))
            return
        result = measure(args.model, args.pilot, args.seed, args.fold, args.run, args.limit)
        print(json.dumps(result))
    elif args.step == "run-all":
        count = run_all(args.pilot, args.models or list(MODELS), args.seeds, args.fold, args.limit, resume=args.resume)
        print(f"{count} runs done")
        print(summarize(args.pilot).round(3).to_string())
    else:
        summary = summarize(args.pilot)
        print((out_dir(tc.dataset_path(args.pilot)) / "form_e_summary.md").read_text(encoding="utf-8"))
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
