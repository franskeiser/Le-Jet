# Le-Jet

**Thesis:** A Random Forest-Guided Static Gating Framework for Financial-Related Smishing Classification Using SMS Text

Le-Jet classifies SMS messages as smishing or legitimate. A CNN reads the message text. For messages with a link, a Random Forest **feature extractor** looks at the link's URL, web page and domain data and outputs two numbers, P and u. These are concatenated with the CNN and URL features, re-weighted by static gates, and passed to the final classifier.

```
Public web set (11,430 pages) ──> train 2 RF extractors (url_only, full) ──> freeze

SMS link ──┬─ old SMS data ──────────────────────────> url_only ──┐
           └─ fresh link ──> inspector ─┬─ page loaded ──> full ──┤
                                        └─ page failed ──> url_only ┤
                                                                  v
                                                          P and u (or -1)
                                                                  │
SMS text ──> fastText 300-d ──> CNN ──────────────┐               │
14 URL text features + has_url ───────────────────┼─> gates <─────┘
                                                  v
                                          dense ──> softmax
```

The full plan is in [`data/dataset_and_training_plan.md`](data/dataset_and_training_plan.md). For background on how this repo differs from SmishGuard, read [`CONTEXT.md`](CONTEXT.md).

## Status

| Step | Status |
|---|---|
| Clean the public web phishing set (32 features) | Done |
| Train the RF feature extractor | Done (Option C: two extractors, no masking; see [`docs/decision_web_inspection.md`](docs/decision_web_inspection.md)) |
| Synthetic legitimate SMS with links | Done |
| Link features for every SMS + grouped splits | Done |
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

This takes about 4 minutes. For each seed (42, 123, 2024), it:

1. Splits the data 70/15/15 into train, validation and test. **The split is by domain** (exact registrable domain via `tldextract`), so no domain appears in more than one part.
2. Tunes a Random Forest on the validation part (16 settings).
3. Trains two versions:
   - `url_only`: the 14 features computed from the link text (no `page_rank`). **Use this for the old SMS data.** Page, domain and `page_rank` lookups made today mostly show that an old scam domain has died, which is a time shortcut, not evidence (Phase 0: 82% of smishing domains no longer resolve).
   - `full`: all 32 features minus `has_url`, trained on complete pages only. Use it only for fresh links whose page was fetched within hours of arrival (the fresh set, a live demo).

   There is no masking: each extractor only sees the kind of data it was trained on. A fresh link whose page fails goes to `url_only`.

   `has_url` is not an input because it is always 1 here.
4. Measures accuracy, precision, recall, F1, ROC-AUC and calibration on the test part. It also runs the shortcut ablations and permutation importance.
5. Saves the frozen extractors.

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--top-k` | off | Retrain on only the top-k features by validation permutation importance. |
| `--seeds` | `42 123 2024` | Seeds to run. |

**Outputs**

| Path | Contents |
|---|---|
| `models/rf_extractor/{url_only,full}_seed{N}.joblib` | Frozen extractors (about 176 MB in total). **Do not commit these**; anyone can regenerate them with the command above. |
| `experiments/rf_extractor/metrics_summary.csv` | Validation and test scores, mean and standard deviation across seeds |
| `experiments/rf_extractor/ablations_summary.csv` | Shortcut ablation results |
| `experiments/rf_extractor/importance_summary.csv` | Permutation and Gini importance per feature |
| `experiments/rf_extractor/*_per_seed.csv`, `tuning_grid.csv`, `splits.csv`, `run_config.json` | Per-seed details and the exact run settings |

**Using a trained extractor**

```python
from src.features.model.rf_extractor import load_extractor, extract_p_u, route_p_u

url_only = load_extractor("url_only", seed=42)
full = load_extractor("full", seed=42)

# Old SMS data: link text only.
pu = extract_p_u(url_only, sms_features)          # columns P, u

# Fresh, inspected links: full where the page was fetched, url_only otherwise.
pu = route_p_u(url_only, full, fresh_features)    # columns P, u, extractor
```

Both need the extractor's input columns plus `has_url`; rows with `has_url == 0` get P = u = -1. `route_p_u` also needs `page_fetched` (1 or 0).

## 3. Add synthetic legitimate messages with links

```powershell
python -m src.data.add_synthetic_legit_links
```

Writes `data/processed/sms_v2_balanced.csv` (21,910 rows, 10,955 per class). Details: [`docs/sms_synthetic_legit_links.md`](docs/sms_synthetic_legit_links.md).

## 4. Compute link features for every SMS

```powershell
python -m src.data.build_sms_link_features
```

Finds the first link in each message and computes `has_url` plus the 14 link-text features the `url_only` extractor uses. Writes `data/processed/sms_link_features.csv` (joined to the messages by `message_id`). No network access.

- 9 of the 14 features came pre-computed in the public dataset, so `src/features/link_features.py` re-implements the dataset's own definitions. Recomputed from the dataset's `url` column, 13 features match 99.95–100% of rows; `random_domain` matches 92.4% (the original's exact input is unknown).
- `link_status` is `link`, `redacted` (the text only has the `<URL>` placeholder, so the link itself is lost) or `none`.

## 5. Split the SMS data

```powershell
python -m src.data.split_dataset
```

For each seed (42, 123, 2024), writes `data/splits/seed_<s>/{train,rf_fit,test}_ids.txt`: about 75% Training, 10% RF-Fitting and 15% Test, balanced by class. Rows are grouped so a group never spans two splits:

- synthetic rows by template, so synthetic test rows come from templates the model never trained on;
- real rows with a link by the link's domain (one scam domain has 1,168 messages);
- other real rows on their own.

Sizes and link shares per split: `experiments/splits/split_report.csv`.

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

- **Before writing any training or evaluation code, read [`docs/training_rules.md`](docs/training_rules.md).** In particular, messages whose link was replaced by `<URL>` are excluded (the split loaders do this by default), and the class imbalance this leaves must be fixed.
- `CONTEXT.md` and `WEB_INSPECTION_PLAN.md` still describe an earlier 79-feature scope. The 32 features above are the current ones.
- `sfh` and `ratio_nullHyperlinks` were dropped because they are 0 in every row of the dataset.
- `dns_record` and `whois_registered_domain` use `1` for the suspicious case (no DNS record; not in WHOIS), as in the dataset. The inspector must follow this direction.
- Some SmishGuard files (`src/features/model/train.py`, `src/features/url_features.py`, and others) need `torch` and `tldextract`. These are not in `requirements.txt` yet; they will be added when that code is adapted.
- Never open a suspicious link on your own computer. All link visits go through the isolated inspector described in `WEB_INSPECTION_PLAN.md`.
