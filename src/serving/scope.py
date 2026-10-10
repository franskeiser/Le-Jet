import re

from src.data.keyword_filter import match_categories

# Out-of-scope warnings for the frontend. Serving-layer only: these never change the label or
# confidence, they only tell the user when a verdict rests on weak ground.
#
# Cutoffs measured on the real seed-42 test split (2,833 messages) plus a small probe set:
#   - Text the model can't read (Chinese, gibberish, emoji-only) leaves 0 known words, and the
#     model then returns the same all-padding output for all of them (p_smishing=0.230).
#   - "0 known, or under half known" flags ~0.04% of in-scope test messages, so short
#     legitimate texts ("ok thanks") are not flagged.
#   - 80.7% of legitimate test messages match no financial keyword (ordinary chat), but only
#     0.12% of smishing ones do -- so non_financial is only raised on a smishing verdict, where
#     it marks a scam outside the thesis scope (e.g. a parcel-delivery scam).
#   - A peso amount also counts as financial: the keyword list has no money pattern, so lending
#     and transfer messages like "Get up to P90K. Apply now" or "A transfer of P15,315.77 is
#     pending approval" matched no category. In the dataset, 328 smishing messages with no
#     keyword match carry a peso amount.
MIN_KNOWN_RATIO = 0.5

# Placeholder tags (<url>, ...) and the kept !/? marks aren't words the user wrote.
_NON_WORD_RE = re.compile(r'^(<[a-z_]+>|[!?])$', re.IGNORECASE)

# A peso amount: ₱500, PHP 1,000, P15,315.77, P90K. The trailing \b keeps "p2p" from matching.
_PESO_AMOUNT_RE = re.compile(r'(?:₱|\bphp|\bp)\s?\d[\d,.]*\s?k?\b', re.IGNORECASE)


def assess_scope(raw_text: str, tokens: list[str], word_to_idx: dict[str, int], label: str) -> dict:
    # tokens: the message's serving tokens BEFORE unknown words are dropped
    # (predict.py's _tokenize_for_serving).
    words = [t for t in tokens if not _NON_WORD_RE.match(t)]
    known_count = sum(1 for t in words if t in word_to_idx)
    ratio = known_count / len(words) if words else 0.0
    categories, _ = match_categories(raw_text)
    has_peso_amount = bool(_PESO_AMOUNT_RE.search(raw_text))
    has_link = any(t.lower() == '<url>' for t in tokens)

    # Only a link (or a link plus words the model can't read) judged legitimate: no warning. The
    # verdict rests on the link's URL features and a clean Safe Browsing check, not on missing text.
    link_only_legitimate = known_count == 0 and has_link and label == 'legitimate'

    flags = []
    if (known_count == 0 or ratio < MIN_KNOWN_RATIO) and not link_only_legitimate:
        flags.append('insufficient_text')
    if label == 'smishing' and not categories and not has_peso_amount:
        flags.append('non_financial')

    return {
        'flags': flags,
        'known_word_count': known_count,
        'word_count': len(words),
        'known_word_ratio': round(ratio, 4),
        'financial_categories': categories,
        'has_peso_amount': has_peso_amount,
    }
