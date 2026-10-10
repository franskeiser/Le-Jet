# src/features/correlation_check.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.features.url_features import get_url_feature_vector

FEATURE_COLUMNS = [
    # number_of_dots dropped 2026-09-08: 97% redundant with subdomain_count
    # (constant +1 offset from the base domain's own dot), and the only
    # divergent cases (e.g. www.Nationwide.co.uk: dots=3, subdomains=1) show
    # it picking up multi-part-TLD noise that subdomain_count was purpose-
    # built to filter out (Task 1.5's .com.ph fix). subdomain_count kept.
    'url_length', 'https_presence', 'number_of_hyphens',
    'number_of_slashes', 'suspicious_keyword_count', 'shortened_url_indicator',
    'has_ip_address', 'subdomain_count', 'special_char_count',
    'suspicious_url_token_count',
]

def build_feature_dataframe(messages: list[str]) -> pd.DataFrame:
    rows = [get_url_feature_vector(text) for text in messages]
    return pd.DataFrame(rows)

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # has_url == 1 only — including no-URL sentinel rows fakes correlation
    # between every feature that shares the same "-1 means no URL" value.
    url_only = df[df['has_url'] == 1]
    return url_only[FEATURE_COLUMNS].corr()

def zero_variance_columns(df: pd.DataFrame) -> list[str]:
    url_only = df[df['has_url'] == 1]
    return [c for c in FEATURE_COLUMNS if url_only[c].std() == 0]

def flag_high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.85) -> list[dict]:
    flagged = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):  # upper triangle only: skip diagonal + mirrored dupes
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > threshold:  # NaN (zero-variance) never flags here on purpose
                flagged.append({'feature_a': cols[i], 'feature_b': cols[j], 'r': round(float(r), 3)})
    return sorted(flagged, key=lambda d: -abs(d['r']))

def plot_heatmap(corr: pd.DataFrame, out_path: str = 'experiments/url_feature_correlation.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('URL structural feature correlation (has_url=1 rows only)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_flagged_report(flagged: list[dict], out_path: str = 'experiments/correlation_flagged_pairs.csv'):
    # empty decision columns for the group to fill in during the review session
    report = pd.DataFrame(flagged)
    for col in ['decision', 'reasoning', 'decided_by', 'date']:
        report[col] = ''
    report.to_csv(out_path, index=False)
    return report

def run(labeled_csv_path: str, text_col: str = 'message'):
    df_raw = pd.read_csv(labeled_csv_path)
    features_df = build_feature_dataframe(df_raw[text_col].tolist())

    zero_var = zero_variance_columns(features_df)
    if zero_var:
        print(f'WARNING: zero-variance features (correlation undefined): {zero_var}')

    corr = correlation_matrix(features_df)
    flagged = flag_high_correlation_pairs(corr)

    plot_heatmap(corr)
    save_flagged_report(flagged)

    print(f'{len(flagged)} pair(s) flagged with |r| > 0.85 — see experiments/correlation_flagged_pairs.csv')
    return corr, flagged
