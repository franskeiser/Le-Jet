# src/serving/explain_text.py
"""
Real, model-grounded explanations for WHICH phrases in a message's text drove
its verdict. Unlike app/app.py's url_reasons (simple threshold checks on
url_features -- "https_presence == 0" etc.), there's no equivalent simple rule
for text: the CNN doesn't expose a "why" without asking the model itself.

Method: gradient x input attribution (Simonyan et al. 2013 -- a standard,
widely-used saliency technique for differentiable models) over TextEncoder's
max-pooled CNN features. Each of the 3*NUM_FILTERS pooled features is one
filter's adaptive-max-pool output; FusionClassifier.encoder.encode_with_positions()
also hands back WHICH window (start index + kernel size) each filter's max came
from, so a feature's attribution score maps straight back to the literal words
that produced it.

Read-only with respect to the model: this module never calls .step()/optimizer
code and never touches forward()/encode()/predict_proba(), so it cannot change
what the model predicts -- it only asks the already-computed graph "which of
your inputs mattered."
"""
import re

import torch

from src.config import NUM_FILTERS
from src.features.text_preprocessing import (
    URL_PATTERN, PLACEHOLDER_TAG_RE, mask_urls, space_out_placeholder_tags, lowercase,
    normalize_punctuation, tokenize,
)

KERNEL_SIZES = (3, 4, 5)
PAD_TOKEN = '<PAD>'
# Overlapping windows are merged into one phrase, but never past this many tokens -- otherwise
# a message where every window points the same way collapses into one reason that is the
# whole message.
MAX_PHRASE_TOKENS = 8
LEADING_PUNCT = set('!?.,:;-')


def _prettify_token(token: str) -> str:
    # Placeholder tags (<URL>, <PHONE_NUMBER>, ...) read oddly verbatim in a
    # sentence -- render them as a bracketed label instead: "<PHONE_NUMBER>" -> "[phone number]".
    if token.startswith('<') and token.endswith('>'):
        name = token[1:-1].lower().replace('_', ' ')
        return '[link]' if name == 'url' else f'[{name}]'
    return token


def _token_char_spans(raw_text: str, tokens: list[str]) -> list[tuple[int, int]] | None:
    # Maps each real (non-<PAD>) model token back to the (start, end) characters of the
    # whitespace-separated chunk of raw_text it came from, so a reason can quote the user's
    # own words (casing, punctuation, hyphens) instead of the normalized tokens.
    # Each chunk goes through the same steps as predict.py's _tokenize_for_serving() --
    # equivalent to running them on the whole text, since a URL or tag never spans whitespace.
    # Tokens the vocabulary dropped are simply skipped: a dropped word is never in the
    # vocabulary, so it can never be mistaken for the next kept token. Returns None if the
    # walk doesn't reproduce `tokens` exactly; callers then fall back to the token text.
    real = [t for t in tokens if t != PAD_TOKEN]
    spans, k = [], 0
    for chunk in re.finditer(r'\S+', raw_text):
        text = normalize_punctuation(lowercase(space_out_placeholder_tags(mask_urls(chunk.group(0)))))
        for tok in tokenize(text):
            if k < len(real) and tok == real[k]:
                spans.append((chunk.start(), chunk.end()))
                k += 1
    return spans if k == len(real) else None


def _render_phrase(
    raw_text: str | None, char_spans, tokens: list[str], start: int, end: int,
) -> tuple[str, tuple[int, int] | None]:
    # Returns (phrase, (char_start, char_end) of that phrase in raw_text). The span is what the
    # frontend highlights in the original message; None when there's no raw text to point into.
    if char_spans is None:
        return ' '.join(_prettify_token(t) for t in tokens[start:end]), None
    span_start, span_end = char_spans[start][0], char_spans[end - 1][1]
    # a span opening on a lone "!" chunk reads as a typo -- skip leading punctuation/space
    while span_start < span_end and (raw_text[span_start] in LEADING_PUNCT or raw_text[span_start].isspace()):
        span_start += 1
    text = raw_text[span_start:span_end]
    text = URL_PATTERN.sub('[link]', text)
    text = PLACEHOLDER_TAG_RE.sub(lambda m: _prettify_token(m.group(0)), text)
    text = ' '.join(text.split())  # line breaks/double spaces inside the span -> one space
    # "…" marks an excerpt, so a phrase cut mid-sentence doesn't read as the whole message.
    if raw_text[:span_start].strip():
        text = '…' + text
    if raw_text[span_end:].strip():
        text = text + '…'
    return text, (span_start, span_end)


def _merge_windows(windows: list[tuple[float, int, int]], max_groups: int) -> list[tuple[int, int]]:
    # windows: (impact, start, end), strongest first. Each new window joins the group it
    # overlaps (if the result stays within MAX_PHRASE_TOKENS) or starts a new group; groups
    # come back in the order their strongest window appeared, i.e. most influential first.
    groups: list[list[int]] = []
    for _, start, end in windows:
        overlapping = [g for g in groups if start < g[1] and g[0] < end]
        if overlapping:
            g = overlapping[0]
            if max(end, g[1]) - min(start, g[0]) <= MAX_PHRASE_TOKENS:
                g[0], g[1] = min(start, g[0]), max(end, g[1])
            continue
        if len(groups) >= max_groups:
            break
        groups.append([start, end])
    return [(g[0], g[1]) for g in groups]


def _logits_from_cnn_features(model, cnn_features: torch.Tensor, url_tensor: torch.Tensor) -> torch.Tensor:
    # Re-runs everything AFTER the encoder for whichever of the three serving models this
    # is, so the gradient flows back to cnn_features. Must mirror each model's forward().
    if not hasattr(model, 'encoder'):
        return model.temp_head(cnn_features)                   # TextEncoder (CNN-only baseline)
    if hasattr(model, 'cnn_gates'):
        fused = torch.cat([cnn_features * model.cnn_gates, url_tensor * model.url_gates], dim=1)
    else:
        fused = torch.cat([cnn_features, url_tensor], dim=1)   # UngatedFusionClassifier
    hidden = torch.relu(model.dense1(fused))
    return model.dense2(hidden)


def text_reasons_for_message(
    model, token_ids: torch.Tensor, url_tensor: torch.Tensor, tokens: list[str],
    predicted_idx: int, max_reasons: int = 3, raw_text: str | None = None,
) -> list[str]:
    return explain_message(model, token_ids, url_tensor, tokens, predicted_idx, max_reasons, raw_text)['reasons']


def _evidence_item(rank, branch_i, filt_i, start, tokens, char_spans, activation, gate, impact, total):
    kernel_size = KERNEL_SIZES[branch_i]
    window = tokens[start:start + kernel_size]
    real_end = start + sum(t != PAD_TOKEN for t in window)  # a window may run into padding
    span = None
    if char_spans is not None and real_end > start:
        span = {'start': char_spans[start][0], 'end': char_spans[real_end - 1][1]}
    return {
        'rank': rank,
        'feature_index': branch_i * NUM_FILTERS + filt_i,   # position in the 192-dim CNN vector
        'kernel_size': kernel_size,
        'filter': filt_i,                                     # filter number within its branch
        'tokens': window,                                     # exactly what the filter read
        'token_start': start,
        'span': span,
        'activation': activation,                             # max-pooled ReLU output
        'gate': gate,                                         # RF gate (gated model only)
        'contribution': impact,                               # gradient x activation
        'share': round(impact / total, 4) if total else 0.0,  # of all same-sign contributions
    }


def _url_evidence(model, url_in: torch.Tensor, url_names: list[str], text_total: float) -> dict | None:
    # Same gradient x input attribution as the CNN filters, on the 11 URL features: how much
    # each one moved the predicted label's score for THIS message. url_in is the raw
    # (un-gated) feature row, so grad x raw already includes the gate: d(score)/d(raw) =
    # d(score)/d(gated) * gate. The CNN-only baseline never reads URL features -> None.
    if url_in.grad is None or not url_names:
        return None
    raw = url_in.detach()[0]
    contribution = (url_in.grad[0] * raw).tolist()
    gates = model.url_gates.tolist() if hasattr(model, 'url_gates') else None
    url_total = sum(contribution)
    magnitude = sum(abs(c) for c in contribution)
    features = [
        {
            'feature': name,
            'value': round(raw[i].item(), 4),
            'gate': round(gates[i], 4) if gates else None,
            'gated_value': round(raw[i].item() * gates[i], 4) if gates else None,
            'contribution': round(contribution[i], 4),
            'share': round(abs(contribution[i]) / magnitude, 4) if magnitude else 0.0,
        }
        for i, name in enumerate(url_names)
    ]
    features.sort(key=lambda f: -abs(f['contribution']))
    paths = abs(url_total) + abs(text_total)
    return {
        'features': features,                   # strongest effect first (either direction)
        'url_total': round(url_total, 4),       # net push of the whole URL path
        'text_total': round(text_total, 4),     # net push of all 192 CNN filters
        'url_path_share': round(abs(url_total) / paths, 4) if paths else 0.0,
    }


def explain_message(
    model, token_ids: torch.Tensor, url_tensor: torch.Tensor, tokens: list[str],
    predicted_idx: int, max_reasons: int = 3, raw_text: str | None = None, max_evidence: int = 10,
    url_names: list[str] | None = None,
) -> dict:
    """
    model: any of the three serving models -- FusionClassifier, UngatedFusionClassifier,
        or a bare TextEncoder (the CNN-only baseline, explained through its temp_head).
    token_ids/url_tensor: EXACTLY what predict_sms() passed to model.predict_proba().
    tokens: the padded token list encode_for_serving() built token_ids from, same order
        (src/serving/predict.py's _tokens_for_message()).
    predicted_idx: the LABEL_TO_IDX index predict_sms() actually chose -- reasons explain
        THAT verdict (whichever label won), not the alternative one.
    raw_text: the original message. When given, each reason quotes the user's own words
        (overlapping windows merged into one phrase); otherwise the normalized tokens.

    Returns:
      'reasons': readable phrases for users (merged, original wording).
      'spans': spans[i] is {'start', 'end'} -- reasons[i]'s character offsets in raw_text
          (Python string indices, i.e. code points), or None when it can't be located.
      'cnn_evidence': the unmerged, unprettified CNN extraction for Research Mode -- the
          individual filters behind the verdict, each with the exact token window it
          max-pooled, its activation, gate and gradient x activation contribution.
          'supporting' = the top filters pushing toward the predicted label, 'opposing' = the
          top ones pushing against it. Padding windows are kept here (<PAD> in 'tokens')
          because they're genuinely part of what the model used.
      'url_evidence': the same attribution for the URL path (url_names gives the column
          order of url_tensor): per-feature value, gate, gated value and contribution, plus
          the URL path's net push next to the text path's. None for the CNN-only baseline.
    """
    model.eval()
    encoder = getattr(model, 'encoder', model)
    cnn_features, positions = encoder.encode_with_positions(token_ids)
    cnn_features.retain_grad()
    url_in = url_tensor.detach().clone().requires_grad_(True)  # a leaf, so its .grad is kept

    logits = _logits_from_cnn_features(model, cnn_features, url_in)

    model.zero_grad(set_to_none=True)
    logits[0, predicted_idx].backward()
    grad = cnn_features.grad[0]                        # (3*NUM_FILTERS,)
    contribution = grad * cnn_features.detach()[0]      # gradient x input, same shape
    model.zero_grad(set_to_none=True)                   # don't leave .grad sitting on a
                                                          # model instance reused across requests
    activations = cnn_features.detach()[0]
    gates = model.cnn_gates if hasattr(model, 'cnn_gates') else None

    filters = []  # (impact, branch_i, filt_i, start) for every one of the 3*NUM_FILTERS filters
    for branch_i in range(len(KERNEL_SIZES)):
        branch_positions = positions[branch_i][0]        # (NUM_FILTERS,)
        for filt_i in range(NUM_FILTERS):
            impact = contribution[branch_i * NUM_FILTERS + filt_i].item()
            filters.append((impact, branch_i, filt_i, int(branch_positions[filt_i].item())))

    windows = []  # (impact, start, end) -- only n-grams that pushed TOWARD the predicted label
    for impact, branch_i, _, start in filters:
        end = start + KERNEL_SIZES[branch_i]
        if impact <= 0 or PAD_TOKEN in tokens[start:end]:
            continue  # padding window -- no real words to show a user
        windows.append((impact, start, end))
    windows.sort(key=lambda w: w[0], reverse=True)

    char_spans = _token_char_spans(raw_text, tokens) if raw_text else None
    reasons, spans = [], []
    for start, end in _merge_windows(windows, max_reasons):
        text, span = _render_phrase(raw_text, char_spans, tokens, start, end)
        phrase = f'"{text}"'
        if phrase not in reasons:
            reasons.append(phrase)
            spans.append({'start': span[0], 'end': span[1]} if span else None)

    def evidence(sign: int, limit: int) -> list[dict]:
        picked = sorted((f for f in filters if sign * f[0] > 0), key=lambda f: -sign * f[0])
        total = sum(f[0] for f in picked)
        return [
            _evidence_item(
                rank, b, f, start, tokens, char_spans,
                round(activations[b * NUM_FILTERS + f].item(), 4),
                round(gates[b * NUM_FILTERS + f].item(), 4) if gates is not None else None,
                round(impact, 4), total,
            )
            for rank, (impact, b, f, start) in enumerate(picked[:limit], start=1)
        ]

    cnn_evidence = {
        'supporting': evidence(+1, max_evidence),
        'opposing': evidence(-1, 3),
        'supporting_total': round(sum(f[0] for f in filters if f[0] > 0), 4),
        'opposing_total': round(sum(f[0] for f in filters if f[0] < 0), 4),
        'n_filters': len(filters),
        'n_supporting': sum(1 for f in filters if f[0] > 0),
    }
    url_evidence = _url_evidence(model, url_in, url_names, sum(f[0] for f in filters))
    return {'reasons': reasons, 'spans': spans, 'cnn_evidence': cnn_evidence, 'url_evidence': url_evidence}
