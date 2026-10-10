import re

URL_PATTERN = re.compile(
    r'(?:https?://|www\.)'
    r'[^\s<>"{}|\\^\[\]`]+'
    r'|'
    r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+'
    r'(?:com|net|org|info|io|co|gov|edu|ph|'
    r'de|uk|eu|ca|nl|so|fyi|xyz|mx|ch|tw|'
    r'ai|lc|is|wang|dev|app|online|site|'
    r'store|shop|click|link|live|club|vip|'
    r'tv|me|cc|biz|mobi|pro|int|gg|bet|win)\b'
    r'(?:/[^\s]*)?',
    re.IGNORECASE
)

def mask_urls(text: str) -> str:
    text = URL_PATTERN.sub('<URL>', text)
    text = re.sub(r'<URL>([.,!?)"\'\]]+)', r'<URL>\1 ', text)
    return text

def extract_urls(text: str) -> list[str]:
    return [m.group(0) for m in URL_PATTERN.finditer(text)]

def primary_url(text: str) -> str | None:
    # Decision: when a message contains more than one URL, only the first
    # one found feeds the 12-dim URL structural-feature vector (Section 2's
    # fixed schema has room for one URL's worth of features per message).
    urls = extract_urls(text)
    return urls[0] if urls else None

def lowercase(text: str) -> str:
    # .lower() also lowercases the <URL> placeholder itself — restore it so
    # <URL>/<PAD> stay visually distinct from real word tokens downstream.
    return text.lower().replace('<url>', '<URL>')

# Placeholder tags (<URL>, and pre-existing PII-redaction tags like <PHONE_NUMBER>,
# <DATE_TIME>, <US_DRIVER_LICENSE>) sit directly against neighboring words/punctuation
# in the raw text with no separating whitespace. Force a boundary before
# normalize_punctuation strips the only thing keeping a tag from fusing into
# "frm<usdriverlicense>" as a single token.
PLACEHOLDER_TAG_RE = re.compile(r'<[a-zA-Z_]+>')

def space_out_placeholder_tags(text: str) -> str:
    return PLACEHOLDER_TAG_RE.sub(lambda m: f' {m.group(0)} ', text)

# Keep letters, digits, whitespace, and !/? — drop everything else.
# IGNORECASE so the uppercase letters inside the <URL> placeholder survive.
PUNCT_RE = re.compile(r'[^a-z0-9\s!?<>]', re.IGNORECASE)

def normalize_punctuation(text: str) -> str:
    return PUNCT_RE.sub('', text)

# Matches whole word tokens (including the <URL> placeholder, since < and >
# are in the character class) or a single ! or ? as its own token — so
# "now!!" becomes ["now", "!", "!"], not one glued-together token.
# IGNORECASE so the uppercase <URL> placeholder still matches as one token.
TOKEN_RE = re.compile(r'[a-z0-9<>]+|[!?]', re.IGNORECASE)

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)

PAD_TOKEN = '<PAD>'

def pad_or_truncate(tokens: list[str], seq_len: int) -> list[str]:
    # Decision: when a message is longer than seq_len, keep the FRONT
    # (tokens[:seq_len]) and drop the tail — not the other way around.
    if len(tokens) >= seq_len:
        return tokens[:seq_len]
    return tokens + [PAD_TOKEN] * (seq_len - len(tokens))

def preprocess(text: str, seq_len: int) -> list[str]:
    text = mask_urls(text)
    text = space_out_placeholder_tags(text)
    text = lowercase(text)
    text = normalize_punctuation(text)
    tokens = tokenize(text)
    return pad_or_truncate(tokens, seq_len)