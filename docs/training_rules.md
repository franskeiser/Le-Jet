# Rules for training and evaluating the SMS model

**Read this before writing or changing any training or evaluation code** (CNN, gates, fusion head, experiments). Each rule fixes a known problem in our data. Breaking one silently inflates the scores.

## 1. Exclude messages whose link was replaced by `<URL>`

**Rule:** drop every row with `link_status == "redacted"` (in `data/processed/sms_link_features.csv`) from training **and** evaluation, in every split.

**Why:** 3,056 messages (3,014 smishing, 42 legitimate) had their link removed upstream and replaced with the literal text `<URL>`. For them the URL branch would get `has_url = 1` with all 14 features `-1`, a pattern that is 98.6% smishing. Real messages never contain a literal `<URL>`, so the model would learn a dataset artifact, not a scam signal.

**In code:** `load_split_df()` and `load_phase_a_train_val()` in `src/data/split_dataset.py` already drop these rows by default (`exclude_redacted=True`). Do not turn that off for training or reported results.

**Follow-up (do not forget):** after excluding them, the classes are uneven:

| | Rows |
|---|---|
| Smishing | 7,941 |
| Legitimate (incl. 3,025 synthetic) | 10,913 |

Fix it in one of two ways, decided by the group, and say which in the thesis:
- **Class weights** in the loss (keeps all data), or
- **Fewer synthetic rows** (regenerate with `src/data/add_synthetic_legit_links.py` so the classes match again).

Also re-check the per-class link shares after excluding them (`experiments/splits/split_report.csv` reports them with the redacted rows still in).

## 2. Old SMS data uses only the link text

**Rule:** for the SMS dataset, P and u come from the `url_only` extractor (`extract_p_u`). No page, WHOIS, DNS or `page_rank` values for these messages.

**Why:** 82% of old smishing domains no longer exist, so anything looked up today mostly means "the domain died", not "this is a scam". See `docs/decision_web_inspection.md`. The `full` extractor is only for fresh links inspected within hours (`route_p_u`).

## 3. Use the grouped splits as they are

**Rule:** always load splits with `src/data/split_dataset.py` (`data/splits/seed_<s>/`). Never re-split with a plain random split.

**Why:** synthetic rows are grouped by template and real rows with links by domain. A random split would put the same scam domain (e.g. 1,168 messages for `usdtjerc.com`) or the same synthetic template in both Training and Test.

## 4. Report the main scores on real messages only

**Rule:** the headline Test scores use rows with `source == "v2_2026-09-15"` only. Synthetic rows may be reported separately, labelled as synthetic.

**Why:** synthetic messages are a training aid. Scores on them show how well the model learned our templates, not real-world performance.

## 5. Report with-link and without-link results separately

**Rule:** besides the overall Test score, report scores for messages with a link (`has_url == 1`) and without one (`has_url == 0`).

**Why:** even with the synthetic rows, smishing messages have links far more often than legitimate ones, so a model can still lean on "has a link means smishing". The split makes that visible.

## 6. Gates and extractors

- Gates are fitted **only** on the RF-Fitting split; `has_url`, P and u keep a fixed gate of 1.0.
- **The RF-Fitting split is uneven after rule 1** (seed 42: 1,096 legitimate vs 637 smishing). The SmishGuard gating code (`src/features/model/rf_gating_module.py`) scores permutation importance with F1 by default, which shifts with class balance. When adapting it, fit the gating forests with `class_weight="balanced"` and score with `balanced_accuracy`.
- The RF extractors are frozen; never retrain them on SMS data.
- Use all three seeds (42, 123, 2024) and report mean and standard deviation.

## 7. `random_domain` is kept at 92% parity (decided 2026-10-11)

Recomputed from the public dataset's URLs, `random_domain` matches the dataset's values on 92.4% of rows; the other 13 link-text features match 99.95–100% (`tests/test_link_features.py`). The original extractor's exact input to the `nostril` gibberish test is unknown, so it cannot be reproduced exactly. It is kept because it is a low-importance feature (12th of 14 in `url_only`), so the mismatch adds a little noise, not a bias. State this in the thesis's limitations.
