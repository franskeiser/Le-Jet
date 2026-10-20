"""One training loop for the text-only baseline and for E2, so both get exactly the same treatment (plan docs/film_architecture_plan.md,
steps 3.3, 3.4 and 3.6).

fit() trains a model with the shared recipe (the same one for every model of the study):
  * Adam, learning rate 1e-3, up to 30 epochs, batches of 64, patience 3 on validation balanced accuracy, the best epoch's
    weights kept (saved to `checkpoint` and loaded back);
  * cross-entropy with label smoothing 0.1 and inverse-frequency class weights from the training labels;
  * weight decay on Dense 64's weights, and on the generator's weights for E2 (the model's parameter_groups());
  * modality dropout, training only, redrawn every epoch from a generator seeded with the seed: a message's page is hidden with
    p = 0.25 (its link numbers are swapped for the page-hidden version), and its text numbers are hidden with p = 0.10. The
    text-only baseline passes 0 and 0, since it has one input and ignores the link.
Train rows carry out-of-fold link numbers (no row is scored by a forest that saw it); validation rows carry the final
forest's, page as inspected, with no dropout. The test part is never read here.

For E2, every epoch also records how far gamma and beta moved on the validation messages (film_stats), and the best epoch's
per-filter table is returned (step 3.4).

The helpers below the loop write and verify the JSON sidecar that sits next to every checkpoint (hashes of what it was built from).
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, NamedTuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, precision_recall_fscore_support
from torch import nn

from src.config import ARTIFACTS_ROOT, DENSE_WEIGHT_DECAY, LABEL_SMOOTHING
from src.features.model.fusion_data import class_weights
from src.unified import text_cnn as tc
from src.unified.text_cnn import BATCH_SIZE, EPOCHS, LR, PATIENCE

P_PAGE_HIDDEN, P_TEXT_HIDDEN = 0.25, 0.10
VAL_METRICS = ["val_accuracy", "val_balanced_accuracy", "val_precision", "val_recall", "val_f1", "val_log_loss"]
MOVED = 0.05          # a filter counts as moved when its mean |gamma - 1| or mean |beta| on the validation messages exceeds this


# ------------------------------------------------------------------ where things are written
def model_dir(dataset: Path) -> Path:
    return ARTIFACTS_ROOT / "models" / "film" / dataset.stem


def out_dir(dataset: Path, step: str) -> Path:
    return ARTIFACTS_ROOT / "experiments" / "film" / dataset.stem / step


def checkpoint_path(dataset: Path, kind: str, seed: int, fold: int) -> Path:
    """kind: "cnn" (the text-only baseline), "film" (E2), "film_scratch" (E2 without the warm start)."""
    return model_dir(dataset) / f"{kind}_seed{seed}_fold{fold}.pt"


# ------------------------------------------------------------------ evaluation
def evaluate(model: nn.Module, token_ids: torch.Tensor, link: torch.Tensor, labels: torch.Tensor, batch: int = 512) -> dict:
    """Validation metrics, eval mode, no modality dropout. Smishing (1) is the positive class."""
    model.eval()
    with torch.no_grad():
        p = torch.cat([torch.softmax(model(token_ids[i:i + batch], link[i:i + batch]), dim=1)[:, 1]
                       for i in range(0, len(labels), batch)]).numpy()
    y, pred = labels.numpy(), (p >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"val_accuracy": accuracy_score(y, pred), "val_balanced_accuracy": balanced_accuracy_score(y, pred),
            "val_precision": precision, "val_recall": recall, "val_f1": f1,
            "val_log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])}


def film_stats(model, token_ids: torch.Tensor, link: torch.Tensor) -> tuple[dict, pd.DataFrame]:
    """How far gamma and beta moved, and what the link did to each filter, over `token_ids` (E2 only; eval mode, no gradients).

    Returns (summary, per-filter table). In the table each row is one filter: the mean and spread of gamma and beta over the
    messages, the mean lift (value with the link minus value without it), and how often its winning window held no words.
    """
    model.eval()
    e = model.explain(token_ids, link)
    deviation = (e.gamma - 1).abs()
    table = pd.DataFrame({"filter": np.arange(e.gamma.shape[1]), "gamma_mean": e.gamma.mean(0).numpy(),
                          "gamma_std": e.gamma.std(0).numpy(), "beta_mean": e.beta.mean(0).numpy(),
                          "beta_std": e.beta.std(0).numpy(), "gamma_dev_mean": deviation.mean(0).numpy(),
                          "beta_abs_mean": e.beta.abs().mean(0).numpy(), "lift_mean": e.lift.mean(0).numpy(),
                          "lift_abs_mean": e.lift.abs().mean(0).numpy(),
                          "no_phrase_share": 1.0 - e.window_has_words.float().mean(0).numpy()})
    summary = {"gamma_dev_mean": float(deviation.mean()), "gamma_dev_max": float(deviation.max()),
               "beta_abs_mean": float(e.beta.abs().mean()), "beta_abs_max": float(e.beta.abs().max()),
               "filters_moved": int(((table["gamma_dev_mean"] > MOVED) | (table["beta_abs_mean"] > MOVED)).sum()),
               "lift_abs_mean": float(e.lift.abs().mean()), "no_phrase_share": float(table["no_phrase_share"].mean())}
    return summary, table


# ------------------------------------------------------------------ training
class FitResult(NamedTuple):
    metrics: dict                    # the best epoch's validation metrics
    best_epoch: int
    epochs_run: int
    seconds: float                   # wall-clock training time (RQ2)
    log: pd.DataFrame                # one row per epoch
    filters: pd.DataFrame | None     # E2 only: the best epoch's per-filter table (step 3.4)


def fit(model: nn.Module, train: tuple, val: tuple, seed: int, checkpoint: Path, epochs: int = EPOCHS, lr: float = LR,
        patience: int = PATIENCE, weight_decay: float = DENSE_WEIGHT_DECAY, generator_weight_decay: float | None = None,
        p_page_hidden: float = P_PAGE_HIDDEN, p_text_hidden: float = P_TEXT_HIDDEN) -> FitResult:
    """Trains `model` in place. train = (token ids, full link numbers, page-hidden link numbers, labels); val = (token ids,
    link numbers, labels). The best epoch is saved to `checkpoint` and loaded back."""
    ids, link_full, link_blank, y = train
    torch.manual_seed(seed)                                   # the dropout layers' draws
    gen = torch.Generator().manual_seed(seed)                 # modality dropout and the batch order
    optimizer = torch.optim.Adam(model.parameter_groups(weight_decay, generator_weight_decay), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y), label_smoothing=LABEL_SMOOTHING)
    is_film = hasattr(model, "explain")
    best, best_state, best_epoch, best_metrics, bad, epoch, rows = -1.0, None, -1, None, 0, -1, []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        page_hidden = torch.rand(len(y), generator=gen) < p_page_hidden
        link = torch.where(page_hidden[:, None], link_blank, link_full)
        text_keep = (torch.rand(len(y), generator=gen) >= p_text_hidden).float()
        order = torch.randperm(len(y), generator=gen)
        loss_sum = 0.0
        for first in range(0, len(order), BATCH_SIZE):
            batch = order[first:first + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(ids[batch], link[batch], text_keep[batch]), y[batch])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        metrics = evaluate(model, *val)
        row = {"epoch": epoch, "train_loss": loss_sum / len(y), **metrics}
        if is_film:
            row.update(film_stats(model, val[0], val[1])[0])
        rows.append(row)
        if metrics["val_balanced_accuracy"] > best:
            best, best_state, best_epoch, best_metrics, bad = metrics["val_balanced_accuracy"], copy.deepcopy(model.state_dict()), epoch, metrics, 0
        else:
            bad += 1
            if bad >= patience:
                break
    seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)
    filters = film_stats(model, val[0], val[1])[1] if is_film else None
    return FitResult(best_metrics, best_epoch, epoch + 1, seconds, pd.DataFrame(rows), filters)


# ------------------------------------------------------------------ sidecars
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sidecar_files(dataset: Path, seed: int, fold: int, checkpoint: Path, extra: Dict[str, Path] | None = None) -> Dict[str, Path]:
    """The files a model is built from, by name: the fold's vocabulary and starting embedding matrix, the checkpoint, and `extra`."""
    emb_dir = tc.embeddings_dir(dataset, seed, fold)
    return {"vocab.json": emb_dir / "vocab.json", "embedding_matrix.npy": emb_dir / "embedding_matrix.npy",
            checkpoint.name: checkpoint, **(extra or {})}


def hashes(paths: Dict[str, Path]) -> Dict[str, str]:
    return {name: sha256(path) for name, path in paths.items()}


def check_hashes(expected: Dict[str, str], paths: Dict[str, Path]) -> None:
    """Stops if any file named in `expected` has changed since the sidecar was written."""
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"{paths[name]} changed since the model was trained")


def write_sidecar(checkpoint: Path, payload: dict) -> dict:
    checkpoint.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def read_sidecar(checkpoint: Path) -> dict:
    return json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
