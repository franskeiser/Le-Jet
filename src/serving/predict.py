import json
import math
import threading
import time
from functools import lru_cache

import joblib
import numpy as np
import torch

from src.config import DENSE_UNITS, NUM_FILTERS, SEEDS, SEQ_LEN_L
from src.features.model.cnn_encoder import TextEncoder
from src.features.model.embedding import build_torch_embedding_layer
from src.features.model.fusion_classifier import UngatedFusionClassifier, load_model
from src.features.text_preprocessing import (
    mask_urls, space_out_placeholder_tags, lowercase, normalize_punctuation, tokenize,
    pad_or_truncate, primary_url,
)
from src.features.url_features import get_url_feature_vector
from src.serving.explain_text import explain_message
from src.serving.safe_browsing_client import check_hard_bypass, get_url_verdict
from src.serving.scope import assess_scope

LABEL_FROM_IDX = {0: 'legitimate', 1: 'smishing'}  # inverse of train.py's LABEL_TO_IDX -- keep
                                                     # this file readable without importing train.py
                                                     # just for the one dict

URL_FEATURE_COLUMNS = [
    'url_length', 'https_presence', 'number_of_hyphens', 'number_of_slashes',
    'suspicious_keyword_count', 'shortened_url_indicator', 'has_ip_address',
    'subdomain_count', 'special_char_count', 'suspicious_url_token_count', 'has_url',
]  # must match get_url_feature_vector()'s key order exactly


class _StubFusionModel:
    """
    Fake model. Matches FusionClassifier.predict_proba's exact shape (same method name,
    same arguments, same (batch, 2) output) so Task 4.1's real model can swap in later
    with a one-line change, not a rewrite.

    Deliberately NOT random -- always "60% legitimate" for every input, so tests are
    reproducible.
    """
    def predict_proba(self, token_ids: torch.Tensor, url_features: torch.Tensor) -> torch.Tensor:
        batch = token_ids.shape[0]
        return torch.tensor([[0.6, 0.4]] * batch)


with open('models/word_to_idx.json') as f:
    _WORD_TO_IDX = json.load(f)  # loaded once at import time, not per-request

_MODEL = load_model()  # Task 4.3 -- was _StubFusionModel() (Task 3.3)

# Temperature scaling (Guo et al., "On Calibration of Modern Neural Networks", 2017) --
# _MODEL is trained with label smoothing (src/config.py's LABEL_SMOOTHING), which keeps
# softmax output off the 0/1 extremes; T then corrects whatever over- or under-confidence
# remains on the val split, without touching which class wins. Fitted offline by notebooks/09_fit_temperature.py on Phase C's held-out
# val split (never used to update a weight, only for early stopping / this) and saved
# alongside the checkpoint it was fit against -- same load_model() default seed (SEEDS[0]).
# Falls back to T=1.0 (no-op, today's exact uncalibrated behavior) if that file hasn't been
# generated yet, e.g. in an environment that ships models/ without also running the notebook.
try:
    with open(f'models/temperature_seed{SEEDS[0]}.json') as f:
        _TEMPERATURE = json.load(f)['temperature']
except FileNotFoundError:
    _TEMPERATURE = 1.0

# Research Mode's model comparison (app/app.py's VALID_MODEL_IDS). Same ids
# notebooks/10_evaluate_test_set.py (and notebooks/20 for jain_replication) report results under,
# and the same checkpoints they evaluate, so a live prediction here matches the thesis tables.
DEFAULT_MODEL_ID = 'fusion_gated'
JAIN_MODEL_ID = 'jain_replication'  # Jain et al. (2022) ensemble baseline -- not a torch model
MODEL_IDS = ('cnn_only_baseline', 'fusion_ungated', 'fusion_gated', JAIN_MODEL_ID)

# One shared instance per model is reused across requests, and text_reasons_for_message()
# runs a backward pass that writes .grad on it -- two overlapping requests (Research Mode's
# "Compare All Models" fires three at once) would race on that .grad. Inference is a few
# ms on CPU, so serializing it costs nothing noticeable.
_INFERENCE_LOCK = threading.Lock()


def _load_baseline_embedding(model_dir: str) -> torch.nn.Embedding:
    embedding_matrix = np.load(f"{model_dir}/embedding_matrix.npy")
    return build_torch_embedding_layer(embedding_matrix, _WORD_TO_IDX)


@lru_cache(maxsize=None)
def _load_baseline(model_id: str, model_dir: str = "models", seed: int = SEEDS[0]) -> torch.nn.Module:
    # Loaded on first use, not at import -- Standard Mode never touches the baselines, so
    # the server (and every test importing this module) doesn't pay for two extra
    # checkpoints it may never need. Mirrors notebooks/10_evaluate_test_set.py's loading.
    encoder = TextEncoder(_load_baseline_embedding(model_dir), num_filters=NUM_FILTERS)
    if model_id == 'cnn_only_baseline':
        # Phase A's checkpoint IS the CNN-only baseline: TextEncoder + its temp_head.
        model = encoder
        model.load_state_dict(torch.load(f"{model_dir}/phase_a_checkpoint.pt", map_location="cpu"))
    elif model_id == 'fusion_ungated':
        model = UngatedFusionClassifier(
            encoder, cnn_dim=NUM_FILTERS, url_dim=len(URL_FEATURE_COLUMNS), dense_units=DENSE_UNITS,
        )
        model.load_state_dict(torch.load(f"{model_dir}/ungated_checkpoint_seed{seed}.pt", map_location="cpu"))
    else:
        raise ValueError(f"unknown baseline model_id: {model_id!r}")
    model.eval()
    return model


@lru_cache(maxsize=None)
def _load_jain(model_dir: str = "models", seed: int = SEEDS[0]):
    # Loaded on first use, like the other baselines. Saved by notebooks/20_run_jain_replication.py
    # -- the exact fitted detector the thesis tables scored (src/features/model/jain_baseline.py).
    # Single-core predict: one message at a time, as timed by notebooks/21 (same predictions).
    return joblib.load(f"{model_dir}/jain_replication_seed{seed}.joblib").use_single_core_inference()


def _get_model(model_id: str) -> torch.nn.Module:
    if model_id == DEFAULT_MODEL_ID:
        return _MODEL
    return _load_baseline(model_id)


@torch.no_grad()
def _predict_proba(model: torch.nn.Module, token_ids: torch.Tensor, url_tensor: torch.Tensor) -> torch.Tensor:
    # TextEncoder (CNN-only) never sees URL features and has no predict_proba() of its own;
    # its forward() returns raw logits through temp_head (cnn_encoder.py).
    if isinstance(model, TextEncoder):
        return torch.softmax(model(token_ids), dim=1)
    return model.predict_proba(token_ids, url_tensor)


def _apply_temperature(prob_smishing: float, temperature: float) -> float:
    # A 2-class softmax reduces to a sigmoid of the logit difference d = z_smishing -
    # z_legitimate: prob_smishing = sigmoid(d). softmax(logits / T) is then exactly
    # sigmoid(d / T) -- so d/T can be recovered straight from the already-softmaxed
    # probability via the inverse sigmoid (logit), with no need to reach back into
    # FusionClassifier.forward()'s raw logits (predict_proba() deliberately never
    # exposes them -- see its own comment on why forward() must stay Softmax-free).
    prob_smishing = min(max(prob_smishing, 1e-12), 1 - 1e-12)  # guard log(0)/log(inf) at the extremes
    logit_diff = math.log(prob_smishing / (1 - prob_smishing))
    return 1.0 / (1.0 + math.exp(-logit_diff / temperature))


def _tokenize_for_serving(text: str) -> list[str]:
    # Same first 4 steps as text_preprocessing.preprocess(), stopped BEFORE pad_or_truncate --
    # we need to drop unknown words between tokenizing and padding, which preprocess()
    # doesn't have a hook for.
    text = mask_urls(text)
    text = space_out_placeholder_tags(text)
    text = lowercase(text)
    text = normalize_punctuation(text)
    return tokenize(text)


def _tokens_for_message(text: str, word_to_idx: dict[str, int], seq_len: int) -> list[str]:
    # Unlike training, an unknown word here is expected, not a bug -- a live SMS can contain
    # a typo, new slang, or a brand name the vocabulary was never built from. We drop the
    # unknown word entirely (rather than replacing it with a mid-sequence <PAD>) so the
    # remaining real words stay next to each other.
    tokens = _tokenize_for_serving(text)
    known_tokens = [t for t in tokens if t in word_to_idx]
    return pad_or_truncate(known_tokens, seq_len)


def encode_for_serving(texts: list[str], word_to_idx: dict[str, int], seq_len: int) -> torch.Tensor:
    all_ids = [
        [word_to_idx[t] for t in _tokens_for_message(text, word_to_idx, seq_len)]
        for text in texts
    ]
    return torch.tensor(all_ids, dtype=torch.long)


def predict_sms(raw_text: str, model_id: str = DEFAULT_MODEL_ID, skip_safe_browsing_bypass: bool = False) -> dict:
    # model_id picks one of MODEL_IDS (Research Mode); Standard Mode always uses the default.
    # skip_safe_browsing_bypass=True (Research Mode) leaves Safe Browsing out entirely, so every
    # model -- the proposed one included -- is compared on its own prediction alone.
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown model_id: {model_id!r}")

    # Safe Browsing runs only for the proposed system in Standard Mode. The baselines (CNN-only,
    # ungated fusion, Jain et al.) are bare models, and Research Mode compares models without it,
    # so neither ever calls it: no lookup, no bypass, and safe_browsing_verdict=None.
    uses_safe_browsing = model_id == DEFAULT_MODEL_ID and not skip_safe_browsing_bypass

    # ONE Safe Browsing call per message (if there's a URL at all) -- check_hard_bypass
    # reuses this same verdict instead of re-checking, and the normal path below now
    # reports the real clean/error result instead of always hardcoding None.
    verdict = get_url_verdict(raw_text) if uses_safe_browsing else None
    # check_hard_bypass() would look the URL up itself if handed verdict=None, so baselines
    # must skip it entirely rather than just passing no verdict.
    bypass_result = check_hard_bypass(raw_text, verdict=verdict) if uses_safe_browsing else None
    if bypass_result is not None:
        # Task 3.2's dict already has label/confidence/source/detected_url/
        # safe_browsing_verdict -- add url_features=None so both return paths share the
        # same key set. text_reasons=[] too: the bypass path deliberately never runs the
        # CNN (safe_browsing_client.py's check_hard_bypass docstring), so there's no
        # forward pass to explain.
        # scope=None: a link Safe Browsing confirmed malicious needs no out-of-scope caveat.
        return {**bypass_result, 'model_id': model_id, 'url_features': None, 'text_reasons': [],
                'text_reason_spans': [], 'cnn_evidence': None, 'url_evidence': None, 'scope': None}

    if model_id == JAIN_MODEL_ID:
        return _predict_jain(raw_text, get_url_feature_vector(raw_text), verdict)

    # inference_ms (Research Mode's per-message latency badge): this model's own input preparation
    # (URL features + tokenization) plus its forward pass -- excluding the Safe Browsing call above,
    # the lazy first-call model load, any wait on _INFERENCE_LOCK, and the explanation pass below.
    model = _get_model(model_id)
    prep_start = time.perf_counter()
    url_dict = get_url_feature_vector(raw_text)
    url_tensor = torch.tensor([[url_dict[k] for k in URL_FEATURE_COLUMNS]], dtype=torch.float32)
    tokens = _tokens_for_message(raw_text, _WORD_TO_IDX, SEQ_LEN_L)
    token_ids = torch.tensor([[_WORD_TO_IDX[t] for t in tokens]], dtype=torch.long)
    prep_seconds = time.perf_counter() - prep_start
    # The temperature was fit against the gated model only (notebooks/09_fit_temperature.py),
    # so the baselines report their raw, uncalibrated softmax (T=1).
    temperature = _TEMPERATURE if model_id == DEFAULT_MODEL_ID else 1.0
    with _INFERENCE_LOCK:
        forward_start = time.perf_counter()
        probs = _predict_proba(model, token_ids, url_tensor)  # (1, 2), T=1 (uncalibrated)
        # Temperature-scale prob_smishing (index 1, per LABEL_TO_IDX in train.py), then derive
        # everything else from it -- T>0 is monotonic, so the winning class (pred_idx) never
        # changes, only how extreme its reported confidence is.
        prob_smishing = _apply_temperature(float(probs[0, 1].item()), temperature)
        pred_idx = 1 if prob_smishing > 0.5 else 0
        confidence = prob_smishing if pred_idx == 1 else 1.0 - prob_smishing
        inference_ms = (prep_seconds + time.perf_counter() - forward_start) * 1000
        explanation = explain_message(
            model, token_ids, url_tensor, tokens, pred_idx, raw_text=raw_text, url_names=URL_FEATURE_COLUMNS,
        )

    return {
        'label': LABEL_FROM_IDX[pred_idx],
        'confidence': confidence,
        'model_id': model_id,
        'source': 'fusion_model',
        'detected_url': primary_url(raw_text),
        'safe_browsing_verdict': verdict,  # real clean/error verdict now, not hardcoded None
        'url_features': url_dict,
        'text_reasons': explanation['reasons'],
        # text_reason_spans[i]: where text_reasons[i] sits in raw_text ({'start', 'end'},
        # code-point offsets) so the frontend can highlight it; None if it couldn't be located.
        'text_reason_spans': explanation['spans'],
        # Research Mode's exact CNN extraction: the individual filters (window, activation,
        # gate, contribution) behind this verdict -- src/serving/explain_text.py's explain_message().
        'cnn_evidence': explanation['cnn_evidence'],
        # Same attribution for the 11 URL features (None for the CNN-only baseline).
        'url_evidence': explanation['url_evidence'],
        'inference_ms': inference_ms,
        'scope': assess_scope(raw_text, _tokenize_for_serving(raw_text), _WORD_TO_IDX, LABEL_FROM_IDX[pred_idx]),
    }


def _predict_jain(raw_text: str, url_dict: dict, verdict: dict | None) -> dict:
    # Jain et al.'s own routing (URL present -> URL classifier, else text classifier) on the raw
    # SMS. Its confidence is the routed classifier's own score (jain_baseline.predict_one), not
    # temperature-calibrated. text_reasons=[]: explain_text.py's attribution reads the CNN's
    # filters, and this model has no CNN. url_dict is still returned so app.py's generic
    # link heuristics (url_reasons) work the same way they do for the CNN-only baseline.
    detector = _load_jain()  # lazy first-call load stays outside the timed span
    start = time.perf_counter()
    out = detector.predict_one(raw_text)  # its own preprocessing (UrlExtract, stemming) + prediction
    inference_ms = (time.perf_counter() - start) * 1000
    prob_smishing = out['prob_smishing']
    pred_idx = out['label']
    confidence = prob_smishing if pred_idx == 1 else 1.0 - prob_smishing
    return {
        'label': LABEL_FROM_IDX[pred_idx],
        'confidence': confidence,
        'model_id': JAIN_MODEL_ID,
        'source': 'fusion_model',  # "a model ran" (vs. safe_browsing_hard_bypass) -- app/frontend contract
        'detected_url': primary_url(raw_text),
        'safe_browsing_verdict': verdict,
        'url_features': url_dict,
        'text_reasons': [],
        'text_reason_spans': [],
        'cnn_evidence': None,  # no CNN in this model
        'url_evidence': None,
        'jain_route': out['route'],
        'inference_ms': inference_ms,
        # Same check as the CNN models: the vocabulary part still reads the CNN's vocabulary, so it
        # says whether the proposed model could read this message, not Jain et al.'s own pipeline.
        'scope': assess_scope(raw_text, _tokenize_for_serving(raw_text), _WORD_TO_IDX, LABEL_FROM_IDX[pred_idx]),
    }
