# src/features/model/fusion_classifier.py
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import DENSE_UNITS, NUM_FILTERS, SEEDS
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.embedding import build_torch_embedding_layer


def load_gates(path: str, cnn_dim: int, url_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    # np.save'd gates come back float64 (permutation_importance's importances_mean is
    # float64); the model's own dtype is float32 -- cast here, once, rather than letting
    # torch's tensor-tensor type promotion silently upcast the fusion vector and crash
    # three layers later at the first nn.Linear. See guide Section 0.7.
    gates = np.load(path).astype(np.float32)
    assert gates.shape[0] == cnn_dim + url_dim, (
        f"gate array length {gates.shape[0]} != cnn_dim+url_dim {cnn_dim + url_dim} -- "
        f"F changed, or the URL feature width changed, since this file was exported (guide Section 4)"
    )
    return torch.from_numpy(gates[:cnn_dim]), torch.from_numpy(gates[cnn_dim:])


def load_phase_a_encoder_weights(model: TextEncoder, checkpoint_path: str) -> None:
    # Discard temp_head explicitly (Task 2.2's throwaway Phase-A-only classification
    # head) rather than a blanket strict=False, which would also silently swallow a
    # genuine key mismatch (e.g. wrong F between checkpoint and this model) with no
    # warning -- the encode_batch convention elsewhere in this codebase (train.py:9-18)
    # is "a real mismatch should crash loudly," not get masked.
    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
    encoder_state = {k: v for k, v in checkpoint_state.items() if not k.startswith('temp_head.')}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    assert not unexpected, f"unexpected keys not matching TextEncoder: {unexpected}"
    expected_missing = {'temp_head.weight', 'temp_head.bias'}
    assert set(missing) == expected_missing, (
        f"expected only {expected_missing} to be missing (temp_head is intentionally "
        f"never loaded here), got: {missing}"
    )


class FusionClassifier(nn.Module):
    def __init__(self, encoder: TextEncoder, cnn_gates: torch.Tensor, url_gates: torch.Tensor,
                 url_dim: int, dense_units: int):
        super().__init__()
        self.encoder = encoder  # embedding + 1 conv branch; temp_head is never attached here
        # register_buffer, not a plain attribute or a bare requires_grad=False tensor --
        # never in .parameters() (optimizer structurally cannot touch it), saved/reloaded
        # with the model, moves with .to(device) automatically. See guide Section 2.1.
        self.register_buffer('cnn_gates', cnn_gates)
        self.register_buffer('url_gates', url_gates)

        fusion_dim = cnn_gates.shape[0] + url_dim  # never hardcode F+12 -- guide Section 0.4/0.6
        self.dense1 = nn.Linear(fusion_dim, dense_units)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_units, 2)

    def forward(self, token_ids: torch.Tensor, url_features: torch.Tensor) -> torch.Tensor:
        cnn_features = self.encoder.encode(token_ids)              # (batch, F) -- NOT .forward()
        gated_cnn = cnn_features * self.cnn_gates                   # elementwise, guide Section 1 step 2
        gated_url = url_features * self.url_gates
        fused = torch.cat([gated_cnn, gated_url], dim=1)            # CNN-group first, guide Section 1 step 3
        h = torch.relu(self.dense1(fused))
        h = self.dropout(h)
        return self.dense2(h)                                        # raw logits -- no Softmax here,
                                                                        # same convention as TextEncoder.forward()

    @torch.no_grad()
    def predict_proba(self, token_ids: torch.Tensor, url_features: torch.Tensor) -> torch.Tensor:
        # forward() must keep returning raw logits -- nn.CrossEntropyLoss (train_phase_c,
        # src/features/model/train.py) applies log_softmax internally, so baking a Softmax
        # into forward() would double up and silently wreck the training loss. This is the
        # inference-only path: (batch, 2) probabilities that sum to 1 per row, column 1 =
        # P(smishing) per LABEL_TO_IDX = {'legitimate': 0, 'smishing': 1}. Multiply by 100
        # for a percentage.
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(token_ids, url_features)
            return torch.softmax(logits, dim=1)
        finally:
            self.train(was_training)


def load_model(model_dir: str = "models", seed: int = SEEDS[0]) -> FusionClassifier:
    with open(f"{model_dir}/word_to_idx.json") as f:
        word_to_idx = json.load(f)
    embedding_matrix = np.load(f"{model_dir}/embedding_matrix.npy")
    embedding_layer = build_torch_embedding_layer(embedding_matrix, word_to_idx)
    encoder = TextEncoder(embedding_layer, num_filters=NUM_FILTERS)

    cnn_dim = NUM_FILTERS
    gates_path = f"{model_dir}/gates_seed{seed}.npy"
    url_dim = np.load(gates_path).shape[0] - cnn_dim  # guide Section 2, point 3
    cnn_gates, url_gates = load_gates(gates_path, cnn_dim, url_dim)

    model = FusionClassifier(encoder, cnn_gates, url_gates, url_dim=url_dim, dense_units=DENSE_UNITS)

    checkpoint_path = f"{model_dir}/phase_c_checkpoint_seed{seed}.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # strict=True (the default) deliberately, unlike load_phase_a_encoder_weights' partial
    # load above -- this checkpoint IS the whole model (encoder + gates + dense head, guide
    # Section 0), so every key should match exactly. Any mismatch here means a real
    # architecture drift between this function and whatever produced the checkpoint, and
    # should crash loudly rather than silently loading a partial/wrong model.
    model.load_state_dict(state_dict)
    model.eval()  # serving-only from here on -- no further training, ever, on this instance
    return model


class UngatedFusionClassifier(nn.Module):
    # Task 2.5 -- the RQ2 ("does gating help at all?") comparison point. Deliberately a
    # SEPARATE class from FusionClassifier, not a use_gates=False flag on it -- this class
    # has no cnn_gates/url_gates buffers anywhere, so it can never accidentally inherit a
    # gate-loading bug from FusionClassifier. See TASK_2_5 guide, Section 2.
    def __init__(self, encoder: TextEncoder, cnn_dim: int, url_dim: int, dense_units: int):
        super().__init__()
        self.encoder = encoder  # a SEPARATE TextEncoder instance from FusionClassifier's --
                                 # each model fine-tunes its own copy of the Phase-A weights
                                 # independently (guide Section 3), never shared.
        fusion_dim = cnn_dim + url_dim  # never hardcode 64+11 -- same discipline as FusionClassifier
        self.dense1 = nn.Linear(fusion_dim, dense_units)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_units, 2)

    def forward(self, token_ids: torch.Tensor, url_features: torch.Tensor) -> torch.Tensor:
        cnn_features = self.encoder.encode(token_ids)            # (batch, F) -- NOT .forward()
        fused = torch.cat([cnn_features, url_features], dim=1)   # raw concat, no gate multiply --
                                                                    # CNN-group first, same order as
                                                                    # FusionClassifier for consistency
        h = torch.relu(self.dense1(fused))
        h = self.dropout(h)
        return self.dense2(h)                                     # raw logits, same convention

    @torch.no_grad()
    def predict_proba(self, token_ids: torch.Tensor, url_features: torch.Tensor) -> torch.Tensor:
        # Mirrors FusionClassifier.predict_proba exactly -- same semantics, so the two
        # models' outputs are directly comparable in Ablation 1 (Phase 5).
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(token_ids, url_features)
            return torch.softmax(logits, dim=1)
        finally:
            self.train(was_training)


class PrunedFusionClassifier(UngatedFusionClassifier):
    # C-prune (docs/c_prune_architecture.md 2.1, Phase 6): the k-filter CNN from cnn_encoder.prune_encoder +
    # the 40 link numbers -> Concatenate (k + 40) -> Dense 64 (weight decay) -> ReLU -> dropout 0.5 -> 2.
    # The gating is static and already done: the removed filters no longer exist, so there are no gate
    # buffers, and the forward pass is UngatedFusionClassifier's. What it adds:
    #   * keep_idx, a buffer (saved with the weights): kept position i is original filter keep_idx[i], for
    #     the explanations (Phase 8) and to refuse a checkpoint of another k on load;
    #   * the Phase A temporary head is dropped from the encoder (set to None, so the encoder passed in loses
    #     it): the fused model never calls it, and it must not count as a trainable parameter (section 5.3);
    #   * parameter_groups: weight decay on Dense 64's weights only;
    #   * forward's optional text_keep, for modality dropout while fine-tuning.
    def __init__(self, encoder: TextEncoder, keep_idx, url_dim: int, dense_units: int = DENSE_UNITS):
        keep = torch.as_tensor(np.asarray(keep_idx), dtype=torch.long)
        if len(keep) != encoder.conv.out_channels:
            raise ValueError(f"keep_idx has {len(keep)} filters, the encoder {encoder.conv.out_channels}")
        super().__init__(encoder, cnn_dim=len(keep), url_dim=url_dim, dense_units=dense_units)
        self.encoder.temp_head = None
        self.register_buffer('keep_idx', keep)

    def forward(self, token_ids: torch.Tensor, url_features: torch.Tensor,
                text_keep: torch.Tensor | None = None) -> torch.Tensor:
        # UngatedFusionClassifier's forward, plus modality dropout's text half for training only: text_keep is
        # (batch,) 1 = text kept, 0 = hidden (its k filter scores set to 0, as run_fused_lean.fit_head does).
        # Never passed at validation, test or live time.
        cnn_features = self.encoder.encode(token_ids)
        if text_keep is not None:
            cnn_features = cnn_features * text_keep[:, None]
        h = torch.relu(self.dense1(torch.cat([cnn_features, url_features], dim=1)))
        return self.dense2(self.dropout(h))

    def parameter_groups(self, weight_decay: float) -> list[dict]:
        # For torch.optim.Adam(groups, lr=...): Adam's weight_decay is L2 added to the gradient. Only the
        # trainable parameters are listed (the frozen embedding is left out).
        decayed = [self.dense1.weight]
        rest = [p for p in self.parameters() if p.requires_grad and all(p is not d for d in decayed)]
        return [{"params": decayed, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]


def build_url_feature_tensor(url_feature_dicts: list[dict], expected_columns: list[str]) -> torch.Tensor:
    # expected_columns is an explicit contract, not inferred -- must be the SAME column
    # order Task 2.3's build_url_feature_matrix produced when the loaded gates were fit,
    # or url_gates[i] silently lands on the wrong feature (guide Section 4). Passing it
    # explicitly is the cheapest available guard until Task 2.3 exports a sidecar file
    # recording the fit-time column order.
    df = pd.DataFrame(url_feature_dicts)
    assert list(df.columns) == expected_columns, (
        f"URL feature column order {list(df.columns)} != expected {expected_columns} -- "
        f"gates would be silently misapplied to the wrong features"
    )
    return torch.tensor(df.values, dtype=torch.float32)


@torch.no_grad()
def no_url_smoke_check(model: FusionClassifier, word_to_idx: dict[str, int], seq_len: int) -> torch.Tensor:
    # Build plan step 6 -- confirm the sentinel-valued no-URL vector (Task 1.6's
    # convention: measurement features = -1, counts = 0, has_url = 0) flows through gate
    # multiplication + fusion + the Dense head without producing NaN/inf.
    from src.features.model.train import encode_batch
    from src.features.url_features import no_url_features

    model.eval()
    token_ids = encode_batch(["Your package could not be delivered."], word_to_idx, seq_len)
    url_dict = no_url_features()
    url_tensor = torch.tensor([[url_dict[k] for k in url_dict]], dtype=torch.float32)

    logits = model(token_ids, url_tensor)
    assert torch.isfinite(logits).all(), f"non-finite output on the no-URL sentinel path: {logits}"
    return logits
