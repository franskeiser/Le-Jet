# src/features/model/train.py
import torch
import torch.nn as nn

from src.config import LABEL_SMOOTHING
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.features.text_preprocessing import preprocess

def encode_batch(texts: list[str], word_to_idx: dict[str, int], seq_len: int) -> torch.Tensor:
    # Deliberately NOT using .get(token, some_fallback) here. Every token should already
    # be in word_to_idx, since the vocab was built from this same corpus (Task 1.8). If a
    # token is ever missing, that's a real bug (vocab/data mismatch) that should crash
    # loudly with a KeyError, not get silently masked by a wrong fallback token.
    all_ids = []
    for text in texts:
        tokens = preprocess(text, seq_len=seq_len)
        all_ids.append([word_to_idx[t] for t in tokens])
    return torch.tensor(all_ids, dtype=torch.long)

# smishing=1 is a deliberate choice, not arbitrary — Appendix A's metric definitions treat
# "Smishing" as the positive class (TP = correctly predicted Smishing), and
# sklearn's precision_recall_fscore_support(average='binary') defaults to pos_label=1.
# Getting this backwards wouldn't crash — it would silently swap precision and recall.
LABEL_TO_IDX = {'legitimate': 0, 'smishing': 1}

class SmishDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, labels: torch.Tensor):
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]

def train_phase_a(model, train_loader, val_loader, epochs, lr, patience=3,
                  checkpoint_path='models/phase_a_checkpoint.pt'):
    # checkpoint_path defaults to the deployed Phase-A path; notebooks/15_run_multiseed.py
    # overrides it so extra seeds never overwrite the deployed seed-42 checkpoint.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)  # raw logits in, Section 1.3
    best_val_f1 = -1.0
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        for token_ids, labels in train_loader:
            optimizer.zero_grad()
            logits = model(token_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for token_ids, labels in val_loader:
                logits = model(token_ids)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0,
        )
        print(f"epoch {epoch}: val_acc={acc:.4f} val_precision={precision:.4f} "
              f"val_recall={recall:.4f} val_f1={f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_metrics = {'val_accuracy': acc, 'val_precision': precision, 'val_recall': recall, 'val_f1': f1}
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)  # Section 6
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # best_metrics, not the last loop iteration's -- the saved checkpoint is the best
    # epoch's weights, so the returned/logged metrics must describe that same epoch,
    # not whatever later epoch the patience countdown happened to stop on.
    return best_metrics


class FusionDataset(Dataset):
    # SmishDataset only carries (token_ids, labels) -- Phase C's forward() needs a
    # same-row-aligned URL feature batch too (TASK_2_4 guide, Section 0.8).
    def __init__(self, token_ids: torch.Tensor, url_features: torch.Tensor, labels: torch.Tensor):
        assert len(token_ids) == len(url_features) == len(labels), "row count mismatch"
        self.token_ids = token_ids
        self.url_features = url_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.url_features[idx], self.labels[idx]


def train_phase_c(model, train_loader, val_loader, epochs, lr, checkpoint_path, patience=3):
    # Deliberately NOT reusing train_phase_a() as-is -- its checkpoint path is hardcoded
    # to 'models/phase_a_checkpoint.pt' and would silently overwrite the frozen Phase-A
    # baseline the results table depends on. checkpoint_path is a required argument here
    # specifically so that mistake is structurally impossible (TASK_2_4 guide, Section 0.9).
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best_val_f1 = -1.0
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        for token_ids, url_features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(token_ids, url_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for token_ids, url_features, labels in val_loader:
                logits = model(token_ids, url_features)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0,
        )  # F1 stays primary here too -- same 97.6%-smishing imbalance as Phase A.
        print(f"epoch {epoch}: val_acc={acc:.4f} val_precision={precision:.4f} "
              f"val_recall={recall:.4f} val_f1={f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_metrics = {'val_accuracy': acc, 'val_precision': precision, 'val_recall': recall, 'val_f1': f1}
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # best_metrics, not the last loop iteration's -- see train_phase_a's comment above.
    return best_metrics