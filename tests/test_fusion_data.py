import numpy as np
import pytest
import torch

from src.config import NUM_FILTERS
from src.features.model.fusion_classifier import FusionClassifier
from src.features.model.fusion_data import AUX_COLUMNS, class_weights, load_embedding_layer, load_seed_inputs
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.rf_gating_module import FIXED_GATE_COLUMNS, fit_and_score_url_group
from src.features.model.train import FusionDataset, train_phase_c
from torch.utils.data import DataLoader


def test_aux_columns_are_14_link_features_then_has_url_p_u():
    assert len(AUX_COLUMNS) == 17
    assert AUX_COLUMNS[-3:] == ["has_url", "P", "u"]


def test_class_weights_equalise_total_loss():
    labels = torch.tensor([0] * 30 + [1] * 10)
    w = class_weights(labels)
    assert w[0] * 30 == pytest.approx(w[1] * 10)


def test_fixed_columns_keep_gate_one_and_nan_importance():
    rng = np.random.default_rng(0)
    n = 400
    names = ["a", "b", "has_url", "P", "u"]
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n), np.ones(n), rng.random(n), rng.random(n)])
    y = (X[:, 0] > 0).astype(int)
    idx = np.arange(n)
    out = fit_and_score_url_group(X, names, y, idx[:280], idx[280:], seed=1, n_repeats=2)
    for i, name in enumerate(names):
        if name in FIXED_GATE_COLUMNS:
            assert out["gates"][i] == 1.0 and np.isnan(out["perm_mean"][i])
        else:
            assert not np.isnan(out["perm_mean"][i])


def test_real_split_inputs_and_one_training_step(tmp_path):
    token_ids, aux, labels = load_seed_inputs(42, "test")
    assert token_ids.shape[0] == aux.shape[0] == labels.shape[0] > 0
    assert aux.shape[1] == len(AUX_COLUMNS)
    no_link = aux[:, AUX_COLUMNS.index("has_url")] == 0
    assert (aux[no_link][:, AUX_COLUMNS.index("P"):] == -1).all()  # P = u = -1 without a link
    assert (aux[no_link][:, :14] == -1).all()
    linked = ~no_link
    assert ((aux[linked][:, AUX_COLUMNS.index("P")] >= 0) & (aux[linked][:, AUX_COLUMNS.index("P")] <= 1)).all()

    embedding, _ = load_embedding_layer()
    encoder = TextEncoder(embedding, NUM_FILTERS)
    model = FusionClassifier(encoder, torch.ones(NUM_FILTERS), torch.ones(len(AUX_COLUMNS)),
                             url_dim=len(AUX_COLUMNS), dense_units=64)
    n = 256
    loader = DataLoader(FusionDataset(token_ids[:n], aux[:n], labels[:n]), batch_size=64)
    metrics = train_phase_c(model, loader, loader, epochs=1, lr=1e-3,
                            checkpoint_path=str(tmp_path / "ckpt.pt"), class_weights=class_weights(labels))
    assert "val_balanced_accuracy" in metrics
