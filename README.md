# Le-Jet

**Thesis:** A Random Forest-Guided Static Gating Framework for Financial-Related Smishing Classification Using SMS Text

Le-Jet classifies SMS messages as smishing or legitimate. A CNN reads the message text. For messages with a link, a Random Forest **feature extractor** looks at the link's URL, web page and domain data and outputs two numbers, P and u. These are concatenated with the CNN and URL features, re-weighted by static gates, and passed to the final classifier.

```
Public web set (11,430 pages) ──> train RF extractor ──> freeze
                                                          │
SMS link ──> inspector (safe) ──> page + URL features ────┤
                                                          v
                                                  P and u (or -1)
                                                          │
SMS text ──> fastText 300-d ──> single-kernel CNN ──┐     │
URL features + has_url + domain popularity ─────────┼─> gates ──> dense ──> softmax
                                                    └─────┘
```

The full plan is in [`data/dataset_and_training_plan.md`](data/dataset_and_training_plan.md). For background on how this repo differs from SmishGuard, read [`CONTEXT.md`](CONTEXT.md).

## Status

| Step | Status |
|---|---|
| Clean the public web phishing set (32 features) | Done |
| Train the RF feature extractor | Done (results being reviewed) |
| Web inspector for SMS links | Planned ([`WEB_INSPECTION_PLAN.md`](WEB_INSPECTION_PLAN.md)) |
| CNN, gating and fusion classifier | Code carried over from SmishGuard, not yet adapted |

## Setup

You need Python 3.13. From the project root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

**Keep the versions in `requirements.txt`.** The trained models are saved with joblib, and they may not load (or may behave differently) under another scikit-learn version. If you need a new package, install it in the venv and add it to `requirements.txt` with its version.

Run every command below from the project root with the venv active.

## 1. Clean the web phishing dataset

```powershell
python src/data/clean_web_phishing.py
```

| | |
|---|---|
| Input | `data/interim/hussein-train-and-test-merged.csv`: the Hannousse & Yahiouche set (train + test merged), 11,430 rows × 89 columns |
| Output | `data/processed/web_phishing_clean.csv`: 11,430 rows × 34 columns (`url`, 32 features, `status`) |
| Labels | 5,715 legitimate, 5,715 phishing |

The 32 features:

- **URL structural (16):** `url_length`, `https_presence`, `number_of_hyphens`, `number_of_slashes`, `subdomain_count`, `special_char_count`, `suspicious_url_token_count`, `shortened_url_indicator`, `has_ip_address`, `suspicious_keyword_count`, `has_url`, `page_rank` (Open PageRank), `path_length`, `random_domain`, `length_hostname`, `tld_in_path`
- **Webpage (12):** `nb_hyperlinks`, `ratio_intHyperlinks`, `nb_extCSS`, `login_form`, `external_favicon`, `links_in_tags`, `ratio_intMedia`, `ratio_extMedia`, `iframe`, `empty_title`, `domain_in_title`, `domain_with_copyright`
- **External (4):** `whois_registered_domain`, `domain_registration_length`, `domain_age`, `dns_record`

Each feature's definition and every cleaning decision are in [`docs/web_phishing_cleaning.md`](docs/web_phishing_cleaning.md).

## 2. Train the RF feature extractor

```powershell
python -m src.features.model.rf_extractor
```

This takes about 6 minutes. For each seed (42, 123, 2024), it:

1. Splits the data 70/15/15 into train, validation and test. **The split is by domain**, so no domain appears in more than one part.
2. Tunes a Random Forest on the validation part (16 settings).
3. Trains two versions:
   - `url_only`: the 14 features computed from the link text (no `page_rank`). **Use this for the old SMS data.** Page, domain and `page_rank` lookups made today mostly show that an old scam domain has died, which is a time shortcut, not evidence (Phase 0: 82% of smishing domains no longer resolve).
   - `full`: all 32 features minus `has_url`, plus `page_available`. Use it only for links inspected within hours of arrival (the fresh set, a live demo).

   `has_url` is not an input because it is always 1 here.
4. Measures accuracy, precision, recall, F1, ROC-AUC and calibration on the test part. It also runs the shortcut ablations and permutation importance.
5. Saves the frozen extractors.

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--page-mask-frac` | `0.3` | Share of training rows made to look like a dead page (webpage features set to -1, `page_available` = 0). `0` turns it off. |
| `--top-k` | off | Retrain on only the top-k features by validation permutation importance. |
| `--seeds` | `42 123 2024` | Seeds to run. |

**Outputs**

| Path | Contents |
|---|---|
| `models/rf_extractor/{url_only,full}_seed{N}.joblib` | Frozen extractors (about 165 MB in total). **Do not commit these**; anyone can regenerate them with the command above. |
| `experiments/rf_extractor/metrics_summary.csv` | Validation and test scores, mean and standard deviation across seeds |
| `experiments/rf_extractor/ablations_summary.csv` | Shortcut ablation results |
| `experiments/rf_extractor/importance_summary.csv` | Permutation and Gini importance per feature |
| `experiments/rf_extractor/*_per_seed.csv`, `tuning_grid.csv`, `splits.csv`, `run_config.json` | Per-seed details and the exact run settings |

**Using a trained extractor**

```python
from src.features.model.rf_extractor import load_extractor, extract_p_u

bundle = load_extractor("full", seed=42)
pu = extract_p_u(bundle, features_df)  # DataFrame with columns P and u
```

`features_df` needs the extractor's input columns plus `has_url`. Rows with `has_url == 0` get P = u = -1. For a link whose page could not be fetched, pass `page_available = 0` and set the webpage features to -1.

## Tests

```powershell
python -m pytest tests
```

## Repository layout

```
data/
  raw/                  original downloads
  interim/              merged, not yet cleaned
  processed/            cleaned, ready for training
  dataset_and_training_plan.md
docs/                   write-ups of decisions (e.g. dataset cleaning)
src/
  config.py             shared seeds and settings
  data/                 dataset cleaning and splitting scripts
  features/             URL features and text preprocessing
  features/model/       RF extractor, CNN, gating and fusion classifier
  lejet/                web inspector code (URL normalization, safety checks)
  serving/              prediction code (from SmishGuard)
tests/
models/                 trained models (generated, not committed)
experiments/            training reports (generated)
CONTEXT.md              project background and how it differs from SmishGuard
WEB_INSPECTION_PLAN.md  detailed plan for safely visiting SMS links
```

## Notes

- `CONTEXT.md` and `WEB_INSPECTION_PLAN.md` still describe an earlier 79-feature scope. The 32 features above are the current ones.
- `sfh` and `ratio_nullHyperlinks` were dropped because they are 0 in every row of the dataset.
- `dns_record` and `whois_registered_domain` use `1` for the suspicious case (no DNS record; not in WHOIS), as in the dataset. The inspector must follow this direction.
- Some SmishGuard files (`src/features/model/train.py`, `src/features/url_features.py`, and others) need `torch` and `tldextract`. These are not in `requirements.txt` yet; they will be added when that code is adapted.
- Never open a suspicious link on your own computer. All link visits go through the isolated inspector described in `WEB_INSPECTION_PLAN.md`.
