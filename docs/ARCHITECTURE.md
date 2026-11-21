# Le-Jet architecture: the one current reference

**Read this first. If another file disagrees with this one, this one wins.** Older design files are kept as history only (list at the end).
Last updated: 2026-11-21. Change this file in the same commit as any change to the model, the data or the experiment plan.

**Title:** Financial-Related Smishing Detection Using a Hybrid CNN–Random Forest Model with FiLM
**Model name in the code:** E2 (`FiLMTextCNN`). **Experiment order:** `docs/final 4&5.md` (the 10-step testing guide).

---

## 1. The idea in one sentence

**The text model makes the decision; the link evidence changes how closely it reads the words.**

A CNN reads the words. A Random Forest (the Link RF) reads the link and the page behind it. The RF does not vote. It produces
a "volume knob" setting per CNN filter (FiLM): a risky link makes the filters that look for warning signs louder, a clean link
leaves them alone or quiets them. The final classifier sees only text numbers.

## 2. The picture

```
 SMS (with or without a link)
        │  split: link cut out, replaced by the <URL> token; has_url = 1 if a link was found
        ├───────────────────────────────────────────────┐
        ▼                                               ▼
 TEXT PREPROCESSING                              URL STRING FEATURES (20)  +  WEBPAGE / DOMAIN FEATURES (17)
 lowercase, normalise punctuation,               = the 37 features (section 4)
 word tokens, pad / cut to 50 (§7.1)              │
        ▼                                               ▼
 fastText 300-d (EN → TL → EN-subword),          LINK RF: 200 trees (§7.2), frozen, fit on every training message with a link (§7.3)
 vocabulary from the fold's TRAIN part only,       → P            phishing probability (mean over trees)
 fine-tuned during training                        → u            disagreement between trees (SD)
        ▼                                          → page_missing 1 if the page did not load
 Conv1D, 64 filters, width 3 → x                   → c1…c37       per-feature contributions (computed, saved,
        │                                                         NOT read by the generator; explanations only)
        ├── Conv average (64) ──────┐                   │
        │                           ▼                   ▼
        │                 GENERATOR: Dense 67 → 128 (P, u, page_missing + 64 averages)
        │                 Δγ, Δβ (64 each);  γ = 1 + tanh(Δγ),  β = Δβ;  starts at zero (γ=1, β=0)
        │                           │
        │                 has_url = 1 ? ── no ──► γ = 1, β = 0 (no link: filters unchanged)
        ▼                           │ yes
 FiLM:  x' = γ·x + β   (before ReLU, every position)
        ▼
 ReLU → max-pool → 64 numbers → dropout 0.2
        ▼
 Dense 64 → ReLU → dropout 0.5 → Output 2 → softmax → SMISHING / LEGITIMATE   (sees text numbers only)
```

## 3. Each part, in plain words

| # | Part | What it does | Trained? | Code |
|---|---|---|---|---|
| 1 | Split | Cuts the link out of the text (it becomes `<URL>`) and sets `has_url`. The same link finder does both, so they always agree. | no | `src/features/text_preprocessing.py`, `src/features/link_features.py` |
| 2 | Text preprocessing | Lowercase, normalise repeated punctuation (`!` and `?` kept as tokens), word tokens, pad / cut to 50 tokens (§7.1). | no | `text_preprocessing.py` |
| 3 | Embedding | Each word → 300 numbers. Lookup order: English fastText, then Tagalog, then an English sub-word guess. The vocabulary is built from each fold's **training part only**; unknown words are dropped. | **yes, fine-tuned** | `src/features/build_crowd_embeddings.py` |
| 4 | Conv1D | 64 filters, each reads 3 neighbouring words. | yes | `src/features/model/film_cnn.py` |
| 5 | Conv average | Mean of each filter over the message. Tells the generator *what the text says*, so the same link can affect different messages differently. | no weights | `film_cnn.py` |
| 6 | 37 link features | 20 from the URL string, 17 from the page and domain (section 4). | no | the lab inspector |
| 7 | Link RF | Turns the 37 features into P, u, page_missing (and c1…c37). Fitted once per fold, then frozen. Never trained together with the CNN. | **separately, then frozen** | `src/unified/link_lane.py` |
| 8 | Generator | One Dense layer, 67 → 128 numbers = one γ and one β per filter, per message. Starts at zero, so training begins as a plain text CNN. | yes | `film_cnn.py` |
| 9 | No-link bypass | `has_url = 0` → γ = 1, β = 0: the message is read as plain text. | no | `film_cnn.py` |
| 10 | FiLM | `x' = γ·x + β` before ReLU. γ > 1 louder, < 1 quieter; β pushes up or down. | no weights | `film_cnn.py` |
| 11 | Head | ReLU → max-pool → dropout 0.2 → Dense 64 → ReLU → dropout 0.5 → Output 2. | yes | `film_cnn.py` |

**Message conditions** (guide Step 2):

| Condition | Features | Model |
|---|---|---|
| No URL | has_url = 0, all 37 features = -1 | generator skipped: γ = 1, β = 0 |
| URL, page did not load | URL + domain features kept; 13 page features = -1; page_missing = 1 | RF scores what it has |
| URL, page loaded | all 37; page_missing = 0 | RF scores all 37 |

## 4. The 37 features (names exactly as in the code, `src/unified/prepare_web_37.FEATURES_37`)

**URL string (20):** length_url, length_hostname, nb_dots, nb_slash, nb_www, https_token, ratio_digits_url, ratio_digits_host,
longest_word_host, phish_hints, prefix_suffix, shortening_service, nb_redirection, length_words_raw, char_repeat,
shortest_word_host, shortest_word_path, longest_words_raw, avg_word_host, domain_in_brand.

**Webpage and domain (17):** 13 page features, set to -1 when the page did not load: nb_hyperlinks, ratio_intHyperlinks, nb_extCSS,
login_form, external_favicon, links_in_tags, ratio_intMedia, ratio_extMedia, iframe, safe_anchor, empty_title, domain_in_title,
domain_with_copyright. 4 outside features, kept when the page did not load: page_rank, whois_registered_domain,
domain_registration_length, domain_age.

## 5. Size

| | Text-only CNN | E2 |
|---|---|---|
| Conv1D | 57,664 | 57,664 |
| Generator | – | 8,704 |
| Dense 64 + Output | 4,290 | 4,290 |
| **Trainable, excluding the embedding** | **61,954** | **70,658 (+14%)** |

The embedding adds vocabulary size × 300 to both. The real speed cost of E2 is visiting the page (seconds), not the model (under 1 ms).

## 6. How it is trained and evaluated

**Data:** `data/final/lejet_final_dataset.csv`. **6,716 messages: 3,091 smishing, 3,625 legitimate. 5,828 with a URL (3,624 page
loaded, 2,204 not loaded), 888 without.** These are the numbers to quote; an older table in the guide (6,718 / 3,097 / 3,621) is out of date.

**Folds (guide Step 3):** Stratified Group 5-fold, seeds 42, 123, 2024 = 15 runs (`src/data/split_crowd.py`, files in
`data/splits_crowd/lejet_final_dataset/`). Related messages (same link domain, same template, or ≥ 90% same wording) are one group and never
cross parts. In each run: test = one fold; validation = about 15% of the other four (grouped); train = the rest. Per-message list:
`experiments/film/lejet_final_dataset/data_prep/Cross_Validation_Splits.xlsx`; checks: `Cross_Validation_Checks.xlsx` in the same folder.
**The test fold is never used for training, early stopping, tuning or any of the decisions in section 7.**

**Order per run:**
1. Vocabulary + fastText matrix from the training part (`python -m src.features.build_crowd_embeddings`).
2. Link RF: settings tuned by grouped 3-fold CV on the training part (`link_lane tune`). Training messages get **out-of-fold** P / u
   (each scored by a forest that never saw it or its group: `link_lane oof`); validation and test messages are scored by one forest
   fit on the whole training part (`link_lane final`).
3. E2 trained **from scratch** (decision D7): `python -m src.unified.film train` (default since 2026-11-21; `--warm-start baseline` is an ablation).

**Recipe (identical for every neural model, `src/unified/fit_model.py`):** Adam, lr 0.001, batch 64, up to 30 epochs, early stop
after 3 epochs without improvement in **validation balanced accuracy** (best epoch kept); cross-entropy with class weights and label
smoothing 0.1; weight decay 1e-4 on Dense 64 and the generator. **Modality dropout** (training only): page hidden for 25% of
messages (their RF numbers replaced by the page-hidden version), text hidden for 10%.

**Models (guide Step 4) and component checks (guide Step 6):**

| Guide name | What it is | Code |
|---|---|---|
| Proposed model | E2, from scratch | `src/unified/film.py` |
| CNN-only | text + the 37 URL features in one two-branch network, no RF (§6.1) | `src/unified/raw_cnn_control.py` |
| Concatenation fusion | same CNN + same RF numbers attached before Dense 64, no FiLM | `src/unified/concat_control.py` |
| KNN-RF-ETC | Jain et al. (2022) with its routing: no URL → text vote, URL → URL classifier (§6.2) | `src/unified/voting_baseline.py` |
| Check: text-only CNN | E2 without any link | `src/unified/cnn_baseline.py` |
| Check: FiLM neutralised | the trained E2 scored with γ = 1, β = 0 for every message | **not built yet** (§8) |
| Check: shuffled RF outputs | every message gets another message's link numbers | `src/unified/film_controls.py` |
| Check: webpage hidden | page features blanked, page_missing = 1 | `film_compare.py` "page hidden" condition |

### 6.1 The CNN-only baseline (text + 37 URL features), rebuilt 2026-11-21

Asked for by the adviser: a CNN whose input is the text **and** the 37 URL features, with no Random Forest.
`src/features/model/raw_cnn.py` (`RawFeatureCNN`).

```
 SMS text                                      37 URL / page / domain features
    │                                                     │
 fastText 300-d (fine-tuned)                  PREPARE (fitted on the training part only, saved in the model):
 Conv1D 64 × width 3                            1. log(1+x) on 18 long-tailed counts / lengths
 ReLU → max-pool → dropout 0.2                  2. standardise with the mean / SD of REAL values only
    │                                           3. missing (-1) → 0 ("average"); clip to ±5
    │                                           + page_missing, 5 missing flags (nb_redirection, WHOIS,
    │                                             registration length, domain age, page rank), has_url  = 44 inputs
    │                                        Dense 64 → ReLU → dropout 0.2
    ▼ 64 text numbers                                     ▼ 64 URL numbers
    └──────────────────── concatenate (128) ──────────────┘
              Dense 64 → ReLU → dropout 0.5 → Output 2 → softmax
```

**Why this design:** it is a fair comparison with E2. The text path, the 37 features, the head, the recipe (including modality
dropout) and the size (68,930 vs 70,658 trainable parameters) are the same; only two things differ, the two the thesis is about:
who reads the 37 features (a Dense layer here, the Link RF in E2) and how the link evidence is combined (side by side here, FiLM in E2).
**Why the preparation:** -1 means "missing" in 38–49% of linked messages (page and WHOIS features) and would otherwise sit next to real
zeros; counts such as `length_url` (up to 1,134) and `nb_hyperlinks` (up to 1,666) would squash everyone else. WHOIS, registration
length, domain age and page rank are often missing **even when the page loaded**, so each gets its own flag.
**Rejected:** convolution over the 37 features (their order is arbitrary, so "neighbouring features" means nothing); adding them to the
text convolution; a character-level CNN on the URL string (a different input from the 37 features); a deeper URL branch (about 3,700
linked training messages per run).
**Smoke test** (seed 42 fold 0, validation only; outputs deleted so Step 4 starts clean): trains in 9 s, rebuilds with a gap of 0;
prepared inputs average about 0 with SD 0.9, and 0.07% of values hit the clip.

### 6.2 KNN-RF-ETC (Jain et al. 2022), with its URL routing (decided 2026-11-21, guide item 37)

The paper's method, as the guide describes it ("text ensemble + separate URL classifier"). Recipes unchanged from the paper
(`src/features/model/jain_baseline.py`); routing and training data adapted to the study (`src/unified/voting_baseline.py`).

```
 SMS ──► has_url? ── no ──► TEXT classifier: lowercase, tokens, no stopwords, Porter stem → TF-IDF (3,000) → SMOTE
                   │                         → hard vote of KNN + Random Forest + Extra Trees (scikit-learn defaults)
                   └─ yes ─► URL classifier: letter tokens of the link, Snowball stem → CountVectorizer → Logistic Regression
                                             (the message text is not read, as in the paper)
```

**Adapted to the study:**
- **Trained on each run's training part** (the paper used separate SMS and URL corpora), and tested on the same test messages as every
  other model, so McNemar's test can pair them. Nothing is tuned; the validation part is not used.
- **Routed by the dataset's own `has_url` and `link`**, the same link finder every other model uses, so "no URL / page not loaded /
  page loaded" are the same messages for all four models (needed for Step 8).
- **Text classifier fitted on all training messages, links removed** (the no-URL messages alone would give about 50 smishing examples
  per run). **URL classifier fitted on the training part's links**, each labelled with its message's label, duplicate (link, label)
  pairs removed (as in the paper).
- p_smishing = Logistic Regression's probability (URL route) or the share of the three voters saying smishing (text route). The
  predictions file records each message's `route`.

**What to expect, and say in Chapter 4:** 87% of messages have a URL, so most verdicts come from the link's words alone, and the grouped
folds always test on domains the URL classifier never saw. Smoke test (seed 42 fold 0, validation, nothing saved): accuracy 0.772
overall, 0.751 on the URL route (654 messages), 0.894 on the text route (113). That is a fair result for the paper's method under a
split that keeps domains apart, not a flaw in the reimplementation.

## 7. Decisions tested on the final dataset (2026-11-21)

All three use only the training and validation parts. Scripts and outputs are named so anyone can rerun them.

### 7.1 Message length: fixed 50 tokens, or the fold's 95th percentile (~32)? → **fixed 50 tokens** (decided 2026-11-21)

**In the code:** `SEQ_LEN_L = 50` (`src/config.py`), returned by `text_cnn.fold_text_setup` for every fold, so E2, the text-only CNN,
the concatenation control and the CNN-only control all use it. Before this, the code used each fold's 95th percentile (31–33); that
number is still written in `split_report.json` for reference only. Matches the guide (Step 2) and the diagram.

**Test:** E2 from scratch, both lengths, the same 15 runs (same folds, forests, recipe); validation parts only.
Results kept in `experiments/film/lejet_final_dataset/length_test/{fold_length_95th_percentile,fixed_50}/train_summary.csv`.

| Validation, mean of 15 | Per-fold (~32) | Fixed 50 | Difference | 50 better in |
|---|---|---|---|---|
| Accuracy | 0.961 | 0.963 | +0.25 pt | 9 / 15 (Wilcoxon p = 0.06) |
| F1 | 0.958 | 0.961 | +0.26 pt | 9 / 15 (p = 0.07) |
| Log loss | 0.153 | 0.148 | −0.005 | 14 / 15 (p = 0.005) |
| Training time per model | 10.5 s | 12.2 s | +1.7 s | |

**Reading:** 50 is never worse: accuracy is level to slightly better (not significant), and probabilities are slightly better in 14 of
15 runs (about 5% of messages are longer than ~32 tokens and lose their ending at the per-fold length). It also matches the guide, the
diagram and Chapter 3, and one fixed number is simpler to report. **The group picked 50.** After the change, E2 was retrained with the
default command and gave weights identical to the test's fixed-50 models in 15 of 15 runs (training is reproducible).

### 7.2 Number of trees in the Link RF → **200 trees** (decided 2026-11-21)

**Test:** on each of the 15 training parts, grouped 3-fold CV over the messages with a link; for every one of the 8 other settings
(depth, leaf size, features per split) a 1,000-tree forest was fit and its first 25…1,000 trees scored (the first k trees of a seeded
forest are exactly the k-tree forest; asserted in the script). Per part, the setting the tuner would pick at each count.
`python -m src.unified.link_rf_study trees` → `experiments/c_prune/lejet_final_dataset/link_rf_study/trees_summary.csv`.

| Trees | 25 | 50 | 100 | 200 | 300 | 500 | 750 | 1,000 |
|---|---|---|---|---|---|---|---|---|
| F1, mean of 15 | 0.908 | 0.911 | 0.910 | 0.911 | 0.911 | 0.911 | 0.912 | 0.912 |
| Log loss (lower is better) | 0.267 | 0.261 | 0.262 | **0.259** | 0.259 | 0.261 | 0.261 | 0.263 |

**Reading:** from 50 trees up the curve is flat: all counts are within 0.2 F1 points, while the 15 parts themselves vary by ±1.5 points.
Only 25 trees is clearly worse. The rule written before the run ("smallest count within 0.001 F1 of the best") picks 50, but there the
rule is deciding on noise (100 scores below 50, which cannot be a real effect). **200 was chosen** because (1) P and u enter the
generator as numbers, so how well P is calibrated (log loss) matters more than F1 at 0.5, and 200 has the lowest log loss; (2) u is the
spread across trees and gets noisier with fewer trees; (3) 200 costs about a quarter of 1,000 trees for the same accuracy (SOP 2).
The other settings (depth, leaf size, features per split) are still tuned per fold. Set in `src/unified/link_lane.py` (`LINK_TREES`).

### 7.3 Which messages the Link RF learns from → **every training message that has a link** (decided 2026-11-21)

Including the ones whose page did not load (their 13 page features are -1). Not "complete pages only".

**Test:** per run, two forests with the same settings (200 trees, the fold's tuned depth / leaf / features), (A) fit on every training
message with a link, (B) fit only on those whose page loaded; both score the validation messages with a link.
`python -m src.unified.link_rf_study rows --trees 200` → `link_rf_study/rows_summary.csv`. Mean of 15 runs:

| Forest trained on | All linked: bal. acc. | Page loaded: bal. acc. | Page not loaded: bal. acc. | Page not loaded: log loss |
|---|---|---|---|---|
| **A. every linked message (chosen)** | **0.927** | 0.947 | **0.870** | **0.26** |
| B. complete pages only | 0.880 | 0.952 | 0.733 | 0.64 |

**Reading:** on pages that loaded the two are tied (within the ±5-point spread across folds). On pages that did not load (about 37% of
linked messages) B is far worse: it never saw a blanked page while training, so its answers there are close to guesses. The forest has
to score those messages at test time and in the app, so it must learn from them. **Limitation to state in Chapter 5:** with A, the
forest can use "the page did not load" itself as evidence (in this dataset 1,303 of the 2,204 not-loaded links are smishing). That is
realistic (scam pages are taken down quickly) but depends on when the pages were visited.

## 7b. Progress on the testing guide

| Step | Status (2026-11-21) | Files |
|---|---|---|
| 1–2 | done | `data/final/`, `data/final/step2/` |
| 3 | done, all checks pass | `experiments/film/lejet_final_dataset/data_prep/Cross_Validation_{Splits,Checks}.xlsx` |
| 4 | **done**: 4 models × 15 runs (+ text-only CNN for Step 6); every neural model rebuilds from its saved files with a gap of 0 | `experiments/film/lejet_final_dataset/step4/Training_Records.xlsx` (`scripts/build_step4_files.py`) |
| 5 | **done** (2026-11-22): test-fold predictions, 5 models × 3 seeds × 6,716 messages, every check passed (each message once per seed, fold matches Step 3, same messages for every model); the test read is logged | `experiments/film/lejet_final_dataset/step5/` (`scripts/build_step5_files.py --read-test`) |
| 6–10 | not started | |

Step 4 validation F1 (mean ± SD of 15; validation picked the best epoch, so these are a training record, **not results**):
E2 0.961 ± 0.012, CNN-only 0.961 ± 0.015, Concatenation 0.964 ± 0.013, KNN-RF-ETC 0.813 ± 0.035, text-only CNN 0.936 ± 0.017.

## 8. Open items (not decided; do not assume either way)

| Item | Status |
|---|---|
| FiLM-neutralised check: no script yet (a test-time switch on the trained E2: treat every message as `has_url = 0`) | to build before Step 6 |
| McNemar multiple-comparison correction (the code applies Holm per seed in `rf_only_compare.py`) | to confirm with the adviser (guide item 39) |
| Diagram wording: "Map full vocabulary" → vocabulary from each fold's training part; "200/500 trees" → 200 trees; "trained on complete pages" → trained on every training message with a link | being fixed by the diagram owner |

**Folder name note:** the Link RF's outputs live in `experiments/c_prune/` and `models/c_prune/`. The name is left over from the
dropped "prune" design (branch `feat/prune-architecture`, decision D16) and is hard-coded in `link_lane.py`. **Nothing is pruned in E2.**
The only pruning-like settings are the forest's own `max_depth` and `min_samples_leaf`, tuned per fold to stop trees memorising.

## 9. Older files (history only, do not follow)

`docs/final_model.md` (static gates, two extractors), `docs/unified_architecture.md` (one RF, static gates),
`docs/progress.md` and `docs/walkthrough.md` (reliability-gate design), `docs/film_architecture_plan.md` (the E2 plan and its decision log
D1–D24; the reasons behind decisions live there, but its pilot numbers are rehearsals on synthetic data and its "100 trees" is replaced by §7.2).
