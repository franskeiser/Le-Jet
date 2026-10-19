# Le-Jet: C-prune architecture plan

**Branch:** `feat/prune-architecture` · **Written:** 2026-10-18 · **Status:** plan only, no code changed yet

**Thesis title (unchanged):** A Random Forest-Guided Static Gating Framework for Financial-Related Smishing Classification Using SMS Text

**Source of the design:** the "C-prune: RF-pruned Conv layer" card of the architecture comparison page (`lejet_all_architectures.html`, version 20). This file turns that card into a build plan.

---

## 1. The idea in plain words

The CNN has 64 **filters**. Each filter is a phrase detector, for example one that fires on "will be locked". Not every filter is useful. Some fire on words that appear in every kind of message, such as "today".

In C-prune a **Random Forest decides which filters the CNN keeps**:

1. Train the text CNN normally (64 filters).
2. Give a **Gate RF** the 64 filter scores of every training message and let it learn smishing vs legitimate.
3. Ask the Gate RF how much each filter helps. Shuffle one filter's scores and measure how much worse the RF gets (**permutation importance**).
4. **Keep the top k filters and delete the rest**. For example, keep 32 of the 64.
5. **Fine-tune** the smaller CNN.
6. **Throw the Gate RF away.** It is not part of the final model.

The result is a normal, smaller CNN whose filters were **chosen by a Random Forest**. Every filter gets a fixed gate: 1 = kept, 0 = deleted. The gate is the same for every message, so it is **static gating**, and it is **RF-guided**.

How it meets each requirement:

| Who | Requirement | How C-prune meets it |
|---|---|---|
| Panel | RF inside or replacing a CNN layer | The Gate RF decides which filters exist in the Conv layer |
| Group leader | RF must not be the classifier | The CNN's Dense + Output layers decide; no RF votes on the verdict |
| Adviser | No extra layer when an existing one can carry it | No new layer. The Conv layer just gets smaller |

---

## 2. The architecture

### 2.1 The final model (what runs for every message)

```
SMS arrives: "Your GCash account will be locked today. Verify here: https://gcash-verify-ph.xyz"
        │
        ▼
Link splitter ─────────────────────────────┬───────────────────────────────────────┐
        │ text without link                 │ URL                                   │
        ▼                                   ▼                                       │
Tokenise + pad (50 words)            Web inspector (at submission time):           │
        │                            redirects, page HTML, WHOIS/RDAP, page rank   │
        ▼                                   │                                       │
Embedding: fastText 300-d, FROZEN           ▼                                       │
        │  50 × 300                  Feature extractor: 20 URL + 17 page/domain    │
        ▼                                   │  37                                   │
Pruned Conv1D, width 3                      ▼                                       │
k filters kept by the Gate RF        Link Random Forest (200 trees)                │
        │  48 × k                    → P, u, 37 contributions                      │
        ▼                                   │  39                                   │
ReLU                                        ▼                                       │
        │                            + page_missing flag → 40 numbers              │
        ▼                                   │                                       │
Global max-pool + dropout 0.2               │                                       │
        │  k                                │  40                                   │
        └──────────────┬────────────────────┘                                       │
                       ▼                                                            │
              Concatenate (k + 40)                                                  │
                       ▼                                                            │
              Dense 64 + weight decay · ReLU · dropout 0.5                          │
                       ▼                                                            │
              Output: 2 neurons + softmax (legitimate / smishing)                   │
                       ▼                                                            │
              Verdict + reasons (P(smishing) ≥ 0.5 → smishing) ◄────────────────────┘
```

`k` is the number of kept filters. Three sizes are tested: **48, 32 and 16** (75%, 50% and 25% of 64). The final `k` is chosen on the **validation** split, never on test.

The two RFs at a glance:

| | Link RF | Gate RF |
|---|---|---|
| Reads | the 37 link features of a message | the 64 pooled filter scores of the trained CNN |
| Job | turns the link into 40 numbers for the head | ranks the 64 filters so the weak ones can be deleted |
| When it runs | for **every message**, training and live | **once**, while building the model |
| In the final model? | yes | **no**, discarded after step 5 |

### 2.2 How the kept filters are chosen (done once, while building)

```
1 · Train the text CNN as usual (all 64 filters)
        │ trained CNN
        ▼
2 · Record the 64 pooled filter scores (eval mode, no dropout)
        │ table: training messages × 64
        ▼
3 · Train the Gate Random Forest (5 folds, grouped by domain)
        │ input: the 64 filter scores · target: smishing or legitimate
        ▼
4 · Permutation importance for each filter, on each held-out fold
        │ shuffle one filter's column → how much worse is the log-loss?
        │ average over the 5 folds (this is the existing "oof_logloss" setting)
        ▼
5 · Rank the 64 filters and keep the top k
        │ kept: g = 1 · removed: g = 0
        ▼
6 · Delete the removed filters, then fine-tune
        │ their rows leave the Conv weights, their columns leave Dense 64
        ▼
Result: a normal, smaller CNN with k filters. The Gate RF is discarded.
```

### 2.3 What "delete a filter" means in the weights

| Layer | Shape before | Shape after | What is removed |
|---|---|---|---|
| Conv1D weight | 64 × 300 × 3 | **k** × 300 × 3 | the rows of the removed filters |
| Conv1D bias | 64 | **k** | the same positions |
| Dense 64 weight | 64 × (64 + 40) | 64 × (**k** + 40) | the columns that read the removed filters |

Before fine-tuning, the pruned model gives **exactly** the same output as the full model with the removed filters' gates set to 0, because a deleted filter would only ever have sent a 0. A unit test checks this (section 6, Phase 5).

### 2.4 Where the static gating is: the RF gate mask

Without link gates, the static gating lives **inside the Conv layer**. The pruning **is** the gating.

A gate is a number that controls how much of a signal gets through. The Gate RF gives every one of the 64 filters a gate, and together those 64 gates form the **gate mask**:

```
Filter   What it learned to spot (example)     Gate-RF rank   Gate g
#17      "will be locked", "account suspended"      1          1   open: kept
#40      "verify your details"                      5          1   open: kept
#29      "click here"                              30          1   open: kept (if k ≥ 30)
#51      "thank you"                               52          0   closed: deleted
#58      "today" (in every kind of message)        64          0   closed: deleted

Gate mask m = [g₁, g₂, …, g₆₄], with exactly k ones
```

- **g = 1:** the filter's signal passes through.
- **g = 0:** the filter's signal is blocked for good. Because a closed filter would only ever send 0, it is deleted from the weights (section 2.3). The pruned CNN is the mask made permanent.

Applied to the Conv layer, the mask does this:

```
Conv output of filter j = m_j × ReLU(W_j · x + b_j)      m_j ∈ {0, 1}, fixed

m_j = 0  →  row j is removed from the Conv weights and column j from Dense 64
m_j = 1  →  filter j is kept as it is
```

How each word of the title maps to this mask:

| Title word | Where it is in C-prune |
|---|---|
| Random Forest-Guided | the Gate RF's permutation importance decides which gates are 1 and which are 0 |
| Static | the mask is set once, while building, and is **the same for every message**. Filter #58 is closed for a GCash scam and for a real bank notice alike. The opposite, dynamic gating, changes the gates per message (as FiLM does) |
| Gating | every filter has a gate, open (1) or closed (0), inside the Conv layer |

**Current model vs C-prune:** the current model gates in two separate steps **after** the CNN (`× cnn_gates`, `× rf_gates`). C-prune gates **once, inside the CNN**, which is what the panel asked for.

**How to show it in the thesis:** draw the 64 filters with a 1 or 0 next to each, then show the 0s removed. Name it the **"RF-guided static gate mask on the Conv filters"**. Save the mask per seed (`gate_mask_seed*.json`, the same content as `keep_idx`), so it can be printed in the results.

**If the panel says "deleting is not gating":** variant **C** uses the same Gate RF, but its gates are graded (0 = off, 0.5–3.0 = weaker or stronger), merged into the Conv weights as W ← g·W, b ← g·b. Phases 4–5 already produce the importance ranking it needs, so C can be reported as a variant without rebuilding anything.

### 2.5 Why it should work (short version; studies on the comparison page)

- Trained networks have spare filters, and deleting the least important ones after training, then fine-tuning, keeps or improves accuracy: HRank [S9], CHIP [S10, main study], Li et al. 2017 [F6].
- Importance-based removal beats random removal: Li et al. 2017 [F6], Molchanov et al. 2019 [F3].
- Some text-CNN filters are nearly unused: Jacovi et al. 2018 [F1].
- **Your contribution:** no study ranks text-CNN filters with an RF's permutation importance. The tests in Phase 7 are what prove it helped.

---

## 3. What changes from the current model

### 3.1 Architecture

| Part | Current model (`docs/final_model.md`, `run_fused_lean.py`) | C-prune |
|---|---|---|
| Conv filters | 64, all kept | **k** (48, 32 or 16), chosen by the Gate RF |
| Text gate | separate `× cnn_gates` (64 numbers, 0.1–3.0) after pooling | **removed**. The gating now happens inside the Conv layer (kept or deleted) |
| Link gate | separate `× rf_gates` (0.1–3.0) on the contributions | **removed** (decided 2026-10-18). Tested once as an ablation (Phase 7) |
| Modality dropout | page hidden 25%, text hidden 10% while training | **unchanged**, same for every model (decided 2026-10-18) |
| Link lane | `has_url`, P, u, contributions, page_missing | **P, u, 37 contributions, page_missing = 40**. `has_url` is dropped because every message has a link |
| Dense 64 | no weight decay | **weight decay** (start at 1e-4, tune on validation) |
| Gate RF | kept its 64 gate numbers in the model | used once, then **discarded** |
| Embedding | fastText 300-d, fine-tuned (6,582,000 trainable parameters) | fastText 300-d, **frozen** for every model (decided 2026-10-18), so pruning shows in the parameter count |
| Model size | 64,514 trainable parameters (Conv + Dense + Output, embedding frozen) | 49,074 / 33,634 / 18,194 for k = 48 / 32 / 16 |

### 3.2 Data

| | Current | C-prune study |
|---|---|---|
| Datasets | v2 SMS + public web-phishing pages + fresh_lean (+ synthetic parts) | **one dataset only: crowdsourced SMS messages with a URL, plus that URL's features** |
| Link features | from a different public dataset, matched by URL | measured **live by the web inspector** when the message is submitted |
| Messages without a link | handled with sentinels | **none**. Every message has a link |
| Synthetic text | used in some sets | **none** |

The old files stay on disk (they are not deleted). They are simply not used for training, gating, tuning or testing in this study.

### 3.3 Code (planned, not done yet)

| File | Change |
|---|---|
| `src/config.py` | point `DATASET_PATH` to `data/processed/crowd_smishing_v1.csv` and `SPLITS_DIR` to `data/splits_crowd`; add `PRUNE_KEEP = [48, 32, 16]`, `DENSE_WEIGHT_DECAY`; re-measure `SEQ_LEN_L` on the crowd data |
| `src/data/build_crowd_dataset.py` (**new**) | crowd submissions + lab inspection output → `data/processed/crowd_smishing_v1.csv` + `.summary.json` (sections 4.2–4.4) |
| `src/data/split_dataset.py` | reuse: 5-fold cross-validation (or 70/15/15 at 4,000+) **grouped by link domain and template**, seeds 42, 123, 2024, written to `data/splits_crowd/` |
| `src/features/model/cnn_encoder.py` | add `prune_encoder(encoder, keep_idx)`, which copies the kept Conv rows into a new k-filter `TextEncoder` |
| `src/features/model/rf_gating_module.py` | reuse the importance code; add `select_top_k(importance, k)` next to `gate_transform` |
| `src/features/model/fusion_classifier.py` | add a `PrunedFusionClassifier` (no gate buffers; `cnn_dim = k`, `url_dim = 40`), or reuse `UngatedFusionClassifier` with `cnn_dim = k`; `load_model` reads `k` and `keep_idx` from a sidecar file instead of `NUM_FILTERS` |
| `src/features/model/train.py` | `train_phase_c` takes `weight_decay` for Dense 64 |
| `src/unified/unified_rf.py` | reuse: Link RF with out-of-fold P, u, contributions; drop `has_url` |
| `src/unified/run_prune.py` (**new**) | the full study runner: link lane → Phase A CNN → Gate RF → prune → fine-tune → controls → test |
| `src/serving/lejet_service.py`, `predict.py`, `explain_text.py` | build the model with `k` filters; map kept filters back to their original ids for the explanations |
| `tests/test_prune.py` (**new**) | pruning exactness, shapes, sidecar round-trip |

---

## 4. The dataset (crowdsourcing only)

### 4.1 What one row is

**One crowdsourced SMS message that contains a URL, plus that URL's features measured by the web inspector at submission time.** Nothing else: no public datasets, no synthetic messages.

Collection rules:

1. **Scope:** financial-related messages only (banks, e-wallets such as GCash and Maya, loans, payments, deliveries asking for payment), to match the title. Both classes must be financial-related, so the model can't learn "finance words = smishing".
2. **Both classes:** smishing **and** legitimate financial messages with links (real bank, e-wallet and payment notices). Legitimate messages with links are the hardest to collect, so plan for them from day one.
3. **Inspect fast:** smishing pages die within hours or days. The web inspector must run **at submission time**, or within a few hours. Record `inspected_at`.
4. **Labels:** the submitter gives a label, and **a second person checks it** (two-person rule). Record `label_verified_by`. Never relabel a message because its page was dead or looked clean.
5. **Privacy and consent:** consent checkbox on the form; names, numbers and account details are removed from `sms_text` before storage (same rule as `docs/fresh_set_spec.md`).
6. **Duplicates:** near-identical templates (same text, different link) are marked with a `template_id`, so they stay on one side of the split.
7. **One link per message:** if a message has several links, use the first one (same rule as before).

### 4.2 Files and names

| File | What it holds | Where | Committed? |
|---|---|---|---|
| `submissions_<YYYY-MM-DD>_<code>.xlsx` (one per contributor per day) | the raw submissions, **before** personal details are removed; a copy of `_template_submissions.xlsx` (made by `scripts/make_crowd_template.py`) with dropdowns, including the form's own `submitter_view`, `org_type` and `concern` answers. These 3 columns are raw-file only for now: not in the 57 columns of section 4.3 | `data/crowd/raw/` (in the repo, so contributors can add files) | yes, **private repo only** (it has personal data) |
| `urls.txt` | the first link of each message, made by `scripts/crowd_to_urls.py`; the inspector's input | `D:\LeJetData\input\` | no (real phishing links) |
| `<batch>.clean.jsonl` | the web inspector's output for that day's links (written by `pull_results.ps1`) | `D:\LeJetData\clean\` | no |
| `crowd_smishing_draft.csv` (+ `.set_aside.csv`, `.summary.json`) | the same 57 columns, rebuilt from all raw files by `build_crowd_dataset.py --crowd` after each lab run; message ids may still shift | `data/processed/` | yes, **private repo only** (so contributors can see the features) |
| **`crowd_smishing_v1.csv`** | **the study dataset**: one row per message, the columns of section 4.3 | `data/processed/` | yes, once frozen |
| `crowd_smishing_v1.summary.json` | counts per class, category, domain, `page_fetched`, written by the builder | `data/processed/` | yes |
| `seed<S>_fold<F>.csv` or `seed<S>_{train,val,test}.csv` | the message ids of each split | `data/splits_crowd/` | yes |

File rules for `crowd_smishing_v1.csv`:
- UTF-8, comma-separated, one header row, columns in the exact order of section 4.3.
- **Written once and never edited.** More data makes a new version (`crowd_smishing_v2.csv`). The thesis names the version it used.
- Built only by `src/data/build_crowd_dataset.py`, never by hand, so it can be rebuilt from the raw files.

### 4.3 Columns (in this order)

**Message (11)**

| Column | Type | Allowed values / format | Example |
|---|---|---|---|
| `message_id` | text | unique, prefix `crowd_` + 6 digits | `crowd_000127` |
| `sms_text` | text | the message with the link **kept**; personal details replaced by `<NAME>`, `<NUM>`, `<ACCT>` | `Your GCash account will be locked today. Verify here: https://gcash-verify-ph.xyz` |
| `label` | text | `smishing` or `legitimate` | `smishing` |
| `category` | text | `bank`, `e-wallet`, `credit_card`, `loan`, `bill_payment`, `delivery_fee`, `government_aid`, `investment`, `other_financial` | `e-wallet` |
| `language` | text | `en`, `tl`, `taglish` | `en` |
| `sender_type` | text | `number` (11-digit mobile), `shortcode`, `alphanumeric` (e.g. "GCash"), `unknown` | `number` |
| `date_received` | date | `YYYY-MM-DD` (when the person got the SMS) | `2026-11-03` |
| `date_submitted` | datetime | `YYYY-MM-DD HH:MM` local time (UTC+8) | `2026-11-03 14:20` |
| `label_source` | text | `submitter`, `bank_confirmed`, `reviewer` | `submitter` |
| `label_verified_by` | text | reviewer code (the second person) | `r02` |
| `consent` | 0/1 | must be 1, rows with 0 are dropped | `1` |

**Link (5)**

| Column | Type | Allowed values / format | Example |
|---|---|---|---|
| `link` | text | the first link exactly as it appears in the SMS | `https://gcash-verify-ph.xyz` |
| `final_url` | text | the URL after redirects; `-1` if not reached | `https://gcash-verify-ph.xyz/login` |
| `link_domain` | text | registrable domain of `final_url` (or of `link` if not reached); the **split group** | `gcash-verify-ph.xyz` |
| `template_id` | text | same id for near-identical texts (`deduplicate_labeled_templates.py`) | `t0311` |
| `link_shortened` | 0/1 | 1 if `link` is a shortener (bit.ly, tinyurl, …) | `0` |

**Inspection (4)**

| Column | Type | Allowed values / format | Example |
|---|---|---|---|
| `inspected_at` | datetime | `YYYY-MM-DD HH:MM` local time | `2026-11-03 14:31` |
| `hours_to_inspect` | number | `inspected_at − date_submitted`, in hours | `0.18` |
| `page_fetched` | 0/1 | 1 only if the page HTML was retrieved and parsed. `page_missing = 1 − page_fetched` is computed in code | `1` |
| `fetch_status` | text | `live`, `redirected`, `dead`, `blocked`, `timeout` | `live` |

**20 URL features** (computed from `link` by the lab code; never reimplemented)

`length_url`, `length_hostname`, `nb_dots`, `nb_slash`, `nb_www`, `https_token`, `ratio_digits_url`, `ratio_digits_host`, `prefix_suffix`, `shortening_service`, `nb_redirection`, `length_words_raw`, `char_repeat`, `shortest_word_host`, `shortest_word_path`, `longest_words_raw`, `longest_word_host`, `avg_word_host`, `phish_hints`, `domain_in_brand`

**17 page/domain features** (13 from the page, 4 from outside services)

`nb_hyperlinks`, `ratio_intHyperlinks`, `nb_extCSS`, `login_form`, `external_favicon`, `links_in_tags`, `ratio_intMedia`, `ratio_extMedia`, `iframe`, `safe_anchor`, `empty_title`, `domain_in_title`, `domain_with_copyright`, `whois_registered_domain`, `domain_registration_length`, `domain_age`, `page_rank`

That is **57 columns** in total (11 + 5 + 4 + 37). Units are the same as the lab's current output: ratios in percent, ages and lengths in days.

**Missing values are always `-1`, never blank, NaN or 0.**
- Page not fetched: the 13 page columns are `-1`. The 4 outside columns are still filled if their lookups worked.
- A lookup failed (WHOIS/RDAP, Open PageRank): that column is `-1`.

The lab also exports a wider 73-column table (`features_full`). The study uses these 37, so it matches the design. The wider table is kept for the thesis appendix.

### 4.4 Row rules (the builder drops or flags rows that break them)

| Rule | Action |
|---|---|
| no link in `sms_text` | drop (this dataset is URL messages only) |
| `consent` ≠ 1 | drop |
| `label` not verified by a second person | keep out until verified |
| not financial-related | drop |
| exact duplicate (same `sms_text` and same `link`) | keep the first, drop the rest |
| `hours_to_inspect` > 24 | keep, but flag it in the summary. Aim for under 6 hours: phishing domains often survive "less than 24 hours before being blocked" (Pritom et al. 2026) |
| more than 20 rows of one `link_domain` in one class | keep the first 20 by `date_submitted`, set the rest aside |
| personal details still in `sms_text` | send back for cleaning |

### 4.5 How much data is needed

| Level | Total messages (smishing / legitimate) | How to evaluate | What it lets you claim |
|---|---|---|---|
| **Minimum** | **2,000** (1,000 / 1,000) | **5-fold grouped cross-validation**: every message is tested once, by a model that never saw it, so all 2,000 are test predictions | the model works, and comparisons between models can be tested (paired McNemar on 2,000 predictions) |
| **Recommended** | **4,000** (2,000 / 2,000) | 5-fold grouped cross-validation, or one 70 / 15 / 15 split (about 600 test messages) | stronger comparisons. With a single split, about 600 test messages can show a gap of about **3 points** between two models |

Below 4,000, always use cross-validation. One 15% test split of 2,000 would be only 300 messages.

**The basis for these numbers:**

| Basis | What it says | What it means for Le-Jet |
|---|---|---|
| Other smishing datasets (Pritom et al. 2026 review; Mishra & Soni 2021, 2022; Goel et al. 2024) | most have **362–638 smishing messages**. The largest labelled smishing set has only **223 messages with a URL**. The review: *"existing smishing datasets are often limited in scale, diversity, and temporal coverage"* | **2,000 URL messages is already larger** than the URL portion of any public smishing dataset found |
| The only crowdsourced smishing dataset (Timko & Rahman, CODASPY 2024, SmishTank) | **1,062** user-submitted smishing messages, with **no legitimate ones**. The review says smishing-only data *"prevents the training of robust binary classifiers"* | crowdsourcing is slow. Collect legitimate messages on purpose. 2,000 is a realistic floor; 4,000 is a stretch goal |
| Text CNN (Kim 2014) | TextCNN with pretrained word vectors worked on datasets of **3,775** (CR) and **5,952** (TREC) sentences. Pretrained vectors help *"in the absence of a large supervised training set"* | **4,000 is inside the proven range** for TextCNN with pretrained vectors (Le-Jet uses fastText). 2,000 is below it, which is why it is only the minimum |
| Testing one classifier (Beleites et al. 2013) | *"75 – 100 samples will usually be needed to test a good but not perfect classifier"* | either level has far more than that |
| Comparing two classifiers (Beleites et al. 2013) | *"often requires hundreds of statistically independent test samples"* | the C-prune claim **is** a comparison (RF-ranked vs random removal), so the evaluation needs hundreds of test predictions: cross-validation at 2,000, or about 600+ in a single test split at 4,000 |
| Own calculation | 95% margin = 1.96 × √(p(1−p)/n). At p ≈ 0.95: ±2.5 points for n = 300, ±1.7 for 600, ±1.0 for 2,000. Paired McNemar, models disagreeing on about 5–6% of messages: about **520** test predictions for a 3-point gap, about **980** for a 2-point gap | 2,000 cross-validated predictions can detect a gap of about 2 points |

**Composition targets (as important as the total):**
- **Balance:** aim for 50 / 50. Never go below 40% legitimate.
- **Domain variety:** at most **20 messages per `link_domain` per class**. Aim for **at least 100 different legitimate domains**, or the CNN learns domain names instead of smishing.
- **Categories:** no single `category` above 40% of the data, and each category should have both classes where possible.
- **Fetched pages:** at least **70%** of rows with `page_fetched = 1`. Otherwise the page features are mostly `-1`.
- **Learning-curve check** (Phase 7): train on 25%, 50%, 75% and 100% of the training data. If accuracy is still climbing at 100%, collect more.

### 4.6 Example row (shortened)

```
message_id,sms_text,label,category,language,sender_type,date_received,date_submitted,label_source,label_verified_by,consent,link,final_url,link_domain,template_id,link_shortened,inspected_at,hours_to_inspect,page_fetched,fetch_status,length_url,...,domain_age,page_rank
crowd_000127,"Your GCash account will be locked today. Verify here: https://gcash-verify-ph.xyz",smishing,e-wallet,en,number,2026-11-03,2026-11-03 14:20,submitter,r02,1,https://gcash-verify-ph.xyz,https://gcash-verify-ph.xyz/login,gcash-verify-ph.xyz,t0311,0,2026-11-03 14:31,0.18,1,live,27,...,2,0
```

---

## 5. Research questions and how each is measured

### 5.1 The questions and the models compared

| # | Research question (statement of the problem) | Measured by | Phase |
|---|---|---|---|
| RQ1 | Classification performance of the proposed RF-guided static gating framework and the baselines: **accuracy, precision, recall, F1** | section 5.2: confusion matrix of the test predictions, smishing = positive class, mean ± std over seeds 42 / 123 / 2024 | 7 |
| RQ2 | Computational efficiency: **number of trainable parameters, inference latency per message, training time** | sections 5.2 and 5.3 | 7 |
| RQ3 | Is there a significant difference between the classification outcomes of the proposed framework and the baselines? **McNemar's test** | section 5.4 | 7 |

In RQ1–RQ3, **"features"** in "static feature gating" means the CNN's filter features: each filter gives one feature after max-pooling, and the gate mask keeps or removes it (section 2.4). State this in Chapter 3.

**Models in every comparison:**

| Role | Model |
|---|---|
| Proposed | **C-prune** (final k chosen on validation; k = 48, 32, 16 all reported) |
| Baseline | Plain CNN (text only) |
| Baseline | CNN_CONCAT (64 text features + raw link features, no RF, no gates) |
| Baseline | UNGATED (64 text features + 40 Link RF numbers, no gates) |
| Baseline | Current GATED model (separate gates after the CNN) |
| Control | Random removal at the same k |
| Control | k-filter CNN trained from scratch |
| Variant / ablation | C (graded gates), C-prune + link gates |

All of them use the same data, splits, seeds, recipe and **frozen embedding**.

### 5.2 Where each measurement is taken in the architecture

Every measurement has a **fixed place** in the architecture. The diagram below is the final model of section 2.1, drawn in the order the application runs it, with the measuring points marked:

- **⏱ t0 … t4**: timestamps taken with `time.perf_counter()`. The latency is computed from them.
- **[P]**: trainable layer. Its parameters are counted for RQ2.
- **[F]**: frozen. Its numbers are reported but not counted as trainable.
- **[RF]**: a Random Forest. It has no trainable parameters; reported as trees and nodes.
- **★**: the point where the verdict is compared with the true label, for RQ1 and RQ3.

```
                        ⏱ t0   message submitted to the application
                               │
                               ▼
              ┌─────────────────────────────────────────┐
  stage 1     │ Web inspector                           │   network: redirects, page HTML,
              │ (lab pipeline: duration_ms, stage_ms_*) │   WHOIS/RDAP, Open PageRank
              └─────────────────────────────────────────┘
                        ⏱ t1   inspection finished                 inspection time = t1 − t0
                               │
                               ▼
              ┌─────────────────────────────────────────┐
  stage 2     │ Feature extractor: 20 URL + 17 page     │
              └─────────────────────────────────────────┘
                        ⏱ t2   37 features ready                   feature time   = t2 − t1
                               │
                               ▼
              ┌─────────────────────────────────────────┐
  stage 3     │ [RF] Link Random Forest, 200 trees      │
              │ → P, u, 37 contributions + page_missing │
              └─────────────────────────────────────────┘
                        ⏱ t3   40 link numbers ready               Link RF time   = t3 − t2
                               │
                               ▼
              ┌─────────────────────────────────────────┐
  stage 4     │ Tokenise SMS text (link removed)        │
              │ [F] Embedding, fastText 300-d (frozen)  │
              │ [P] Pruned Conv1D, k filters            │   ← the RF gate mask (section 2.4)
              │     ReLU → global max-pool              │
              │     Concatenate (k + 40)                │
              │ [P] Dense 64                            │
              │ [P] Output, 2 neurons + softmax         │
              └─────────────────────────────────────────┘
                        ⏱ t4   verdict shown                       text model time = t4 − t3
                               │
                               ▼
                        ★  verdict vs true label → confusion matrix (RQ1) and McNemar pairs (RQ3)

  inference latency per message = t4 − t0
```

The building steps (done once, before any test message) are timed the same way, for **training time**:

```
⏱ s0 ─ Link RF training + out-of-fold outputs ─ ⏱ s1 ─ Phase A: text CNN ─ ⏱ s2
     ─ Gate RF + permutation importance ─ ⏱ s3 ─ pruning ─ ⏱ s4 ─ fine-tuning (Phase C) ─ ⏱ s5

training time = s5 − s0   (each step reported too: s1 − s0, s2 − s1, …)
```

**Definition of every metric:**

| Metric | RQ | Definition | Formula | Unit | Taken at |
|---|---|---|---|---|---|
| Accuracy | 1 | share of test messages classified correctly | (TP + TN) / (TP + TN + FP + FN) | % | ★ |
| Precision | 1 | of the messages flagged smishing, the share that really are | TP / (TP + FP) | % | ★ |
| Recall | 1 | of the real smishing messages, the share that were caught | TP / (TP + FN) | % | ★ |
| F1-score | 1 | balance of precision and recall | 2 × Precision × Recall / (Precision + Recall) | % | ★ |
| Trainable parameters | 2 | count of the numbers the optimiser updates | Σ size of every [P] layer (formulas in section 5.3) | count | [P] layers |
| Inference latency | 2 | wall-clock time from submission to verdict, for one message | t4 − t0 (and each stage separately) | ms | t0 … t4 |
| Training time | 2 | wall-clock time to build the model from the training data | s5 − s0 (and each step separately) | s | s0 … s5 |
| McNemar's test | 3 | do two models make **different** mistakes on the same messages more than chance? | b, c = messages only one model got right; exact p = 2 × Σ(i = 0 … min(b, c)) C(b + c, i) × 0.5^(b + c) | p-value | ★, paired |

TP, TN, FP, FN come from the confusion matrix at ★, with **smishing as the positive class**:

| | Predicted smishing | Predicted legitimate |
|---|---|---|
| **Truly smishing** | TP (caught) | FN (missed) |
| **Truly legitimate** | FP (false alarm) | TN (correctly passed) |

**Measurement procedure, step by step:**

1. Build every model on the training folds (timers s0 … s5 → training time).
2. Count each model's trainable parameters in code (`requires_grad = True`) and check them against section 5.3.
3. Inspect every test link once with the web inspector, and save its stage 1 time and features.
4. For each model, run every test message one at a time (after 50 warm-up messages) and save t0 … t4.
5. Save each verdict next to the true label (★).
6. From the saved verdicts: the confusion matrix → accuracy, precision, recall, F1.
7. From the saved timings: median, 95th percentile and mean of t4 − t0 and of each stage.
8. From the paired verdicts: b and c for each pair of models → exact McNemar, Holm-corrected.
9. Repeat for seeds 42, 123 and 2024, and report mean ± std.

**Result tables for Chapter 4** (filled in after Phase 7):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| C-prune (k = …) | | | | |
| Plain CNN | | | | |
| CNN_CONCAT | | | | |
| UNGATED | | | | |
| Current GATED | | | | |
| Random removal (same k) | | | | |

| Model | Trainable parameters | Latency median (ms) | Latency 95th pct (ms) | Text model median (ms) | Training time (s) |
|---|---|---|---|---|---|
| C-prune (k = …) | | | | | |
| Plain CNN | | | | | |
| CNN_CONCAT | | | | | |
| UNGATED | | | | | |
| Current GATED | | | | | |

| Comparison | Seed | b | c | p (exact) | p (Holm) | Significant? |
|---|---|---|---|---|---|---|
| C-prune vs Plain CNN | 42 | | | | | |
| C-prune vs Random removal | 42 | | | | | |
| … | | | | | | |

### 5.3 Computational efficiency (RQ2)

**Number of trainable parameters.** The fastText embedding (6,582,000 numbers) is **frozen for every model**, so it is not trainable and is reported separately as "frozen parameters". With it frozen, the text model's trainable parameters are:

| Part | Formula | 64 filters | k = 48 | k = 32 | k = 16 |
|---|---|---|---|---|---|
| Conv1D | 300 × 3 × k + k | 57,664 | 43,248 | 28,832 | 14,416 |
| Dense 64 | (k + 40) × 64 + 64 | 6,720 | 5,696 | 4,672 | 3,648 |
| Output | 64 × 2 + 2 | 130 | 130 | 130 | 130 |
| **Total trainable** | | **64,514** | **49,074** | **33,634** | **18,194** |
| Reduction vs 64 filters | | — | 24% | 48% | 72% |

- Count in code with `sum(p.numel() for p in model.parameters() if p.requires_grad)` and check it against this table.
- The Link RF has no trainable parameters in the neural-network sense. Report it separately: 200 trees, total number of nodes, file size.
- The Gate RF is discarded, so it adds nothing to the final model. It only adds training time.
- Baselines without the link lane have different Dense sizes (e.g. plain CNN: 64 × 64 + 64). Each model's count is taken from its own code.

**Inference latency per message.** It is the **total wall-clock time in the application**, from the moment a message is submitted to the moment the verdict is shown, **including the web inspection**. It is wall-clock time, not CPU time: most of the inspection is waiting for the network (the website, DNS, WHOIS, Open PageRank), and CPU time would hide that waiting.

```
submit ─┬─ [1] web inspection      lab pipeline: duration_ms, stage_ms_*        network, seconds
        ├─ [2] feature extraction  the 37 link features                         CPU, milliseconds
        ├─ [3] Link RF             200 trees → 40 numbers                       CPU, milliseconds
        ├─ [4] text model          tokenise → embedding → Conv → Dense → Output  CPU, milliseconds
        └─ verdict
total latency = [1] + [2] + [3] + [4]
```

Each stage is timed with `time.perf_counter()` (already used in `src/serving/predict.py` and `lejet_service.py`; the lab already records `duration_ms` and `stage_ms_*` in `src/lejet/pipeline.py`).

The fairness problem: the same link can take 2 seconds now and 8 seconds later, because websites slow down or time out. If every model inspected live on its own, the differences would come from the internet, not the models. So **each test link is inspected once, and every model shares that inspection**:

1. Inspect each test link once. Record its stage [1] time and its features.
2. Run every model on the same messages and features, and time only its own stages.
3. **Total for each model = the shared [1] + its own [2] + [3] + [4].**

| Model | Uses the web inspection? | Total latency |
|---|---|---|
| Plain CNN | no | [4] |
| CNN_CONCAT | yes | [1] + [2] + [4] |
| UNGATED, current GATED, C-prune, controls | yes | [1] + [2] + [3] + [4] |

The plain CNN will be much faster because it skips the inspection. That is a real trade-off (faster, but blind to the link) and is reported as such.

Measurement rules (written in Chapter 3):
- **One machine for all models.** Record the CPU model, RAM and OS. **CPU only**, no GPU, because that is how the application runs.
- **Batch size 1:** one message at a time, as in the application.
- **Warm-up:** run about 50 messages first and do not count them.
- **Fixed threads:** `torch.set_num_threads(1)` and Link RF `n_jobs = 1`, stated in the paper.
- **Report median, 95th percentile and mean**, per stage and in total. The median is the headline: one slow website pulls the mean up.
- No other heavy programs running during the benchmark.
- Save the per-message times (`latency_seed<S>.csv`: `message_id`, model, `ms_inspect`, `ms_features`, `ms_link_rf`, `ms_text_model`, `ms_total`).

Expected result: the total is dominated by stage [1] (seconds against milliseconds). Pruning mainly speeds up stage [4]. Report the total, which answers RQ2 honestly, **and** the stage breakdown, which shows where pruning helps.

**Training time.** Wall-clock time per step, on the same machine, with the same rules:

| Step | Which models have it |
|---|---|
| Link RF training (incl. out-of-fold outputs) | every model with the link lane |
| Phase A: text CNN | every model |
| Gate RF + permutation importance (5 folds × 30 repeats) | C-prune, C |
| Pruning | C-prune, random removal |
| Fine-tuning / Phase C | every model |
| **Total** | every model |

C-prune **will take longer to train** than the baselines, because of the Gate RF and the permutation importance. Report this honestly as a **one-time building cost**, exchanged for a smaller model at prediction time.

### 5.4 McNemar's test (RQ3)

- **Paired:** both models predict **the same test messages**. With cross-validation, every message is a test case once, by models that never saw it. Assert the message ids are identical before testing (as `run_fused_lean.py` already does).
- **Exact McNemar** on the disagreements: b = messages C-prune got right and the baseline got wrong, c = the reverse. The test asks whether b and c differ by more than chance. Existing code: `src/features/model/mcnemar.py`.
- **Per seed:** run it for each of the 3 seeds and report all three (b, c, p-value).
- **Several comparisons:** C-prune is compared with several baselines, so the p-values are corrected with **Holm's method**, otherwise some "significant" results may appear by luck.
- **Significance level:** α = 0.05 after correction.

---

## 6. Phases and checklists

Tick each box when done. **No code is run until the group agrees to start a phase.**

**Order of work (agreed 2026-10-19):**
1. Finish the architecture on the synthetic pilot first: the rest of Phase 5, Phase 6 and the Phase 7 controls. The ranking rule does not need deciding for this, because every later step only reads `keep_idx`. Changing the rule later means swapping one file and re-running, not rebuilding.
2. Collect the crowd data meanwhile (Phase 1), legitimate messages first.
3. When the real data is in, run the whole pipeline, then `python -m src.unified.gate_rf compare` on **validation**.
4. Lock the ranking rule there (Phase 6, "Lock the ranking rule"), **once**. Do not change it again after that.
5. Run Phase 7, and read the test folds **once**, last.

### Phase 0: Decisions and setup
- [x] Group agrees on C-prune as the main model (this file)
- [x] **Link gates: decided 2026-10-18.** No link gates in the main model. The static gating is the RF gate mask on the Conv filters (section 2.4). "C-prune + link gates" is tested once as an ablation (Phase 7)
- [x] **Modality dropout: decided 2026-10-18.** Kept as in the current recipe, for **every** model: while training, the page features are hidden in 25% of messages (`page_missing = 1`) and the text features in 10%. Never applied at validation, test or live time
- [x] **Embedding: decided 2026-10-18.** fastText 300-d is **frozen** for every model (no model fine-tunes it). Trainable parameters are then Conv + Dense + Output only (section 5.3)
- [x] **Latency: decided 2026-10-18.** Total wall-clock time in the application, from submission to verdict, **including the web inspection**, measured stage by stage with one shared inspection per message (section 5.3)
- [x] Freeze the shared recipe: Adam 1e-3, 30 epochs, patience 3, label smoothing 0.1, class weights, modality dropout 0.25 / 0.10, seeds 42 / 123 / 2024
- [x] Ethics and consent form approved before collection starts

### Phase 1: Crowdsourced dataset
- [ ] Submission form: SMS text, label, category, date received, consent
- [ ] Web inspector runs at submission time and records `inspected_at`, `page_fetched`, `fetch_status`
- [ ] Personal details removed from `sms_text`
- [ ] Second person verifies every label
- [ ] `template_id` assigned to near-duplicates (`deduplicate_labeled_templates.py`)
- [ ] `build_crowd_dataset.py` writes `data/processed/crowd_smishing_v1.csv` (57 columns, section 4.3) and `crowd_smishing_v1.summary.json`, applying the row rules of section 4.4
- [ ] Data check: balance ≥ 40% legitimate, ≤ 20 per domain per class, ≥ 70% pages fetched, every row has a link, `-1` for missing values
- [ ] Reach at least the minimum (2,000), aim for 4,000

### Phase 2: Splits and text preparation
- [x] *(pilot, grouped by domain only)* Splits grouped by `link_domain` and `template_id`, for each of the 3 seeds: **5-fold cross-validation** below 4,000 messages; 5-fold or 70 / 15 / 15 at 4,000+ (section 4.5). Inside each training part, 15% is held out as validation for early stopping. "Training split" below means the training folds
- [x] *(pilot)* Write the split files to `data/splits_crowd/`
- Pilot (2026-10-18, synthetic `crowd_pilot_synthetic.csv`, pipeline test only): `src/data/split_crowd.py` and `src/features/build_crowd_embeddings.py` run end to end. The strict domain + template rule was refused (one joined group held 73% of the rows, because the generator reuses templates across random domains), so the pilot is grouped by domain only. About 94% of its test rows share a template with training. **Open decision for the real data:** if real campaigns also chain domains and templates into one giant group, choose between domain-only grouping with the template overlap reported, or template-only grouping
- [ ] Check that no domain and no template appears in two splits *(pilot: domains checked, 0 overlap; templates overlap, see the note above)*
- [x] *(pilot)* Remove the link from `sms_text` (the link goes to the link lane only). Done at encoding time: the link becomes the `<URL>` token, as in the live app
- [x] *(pilot: 18)* Re-measure `SEQ_LEN_L` (95th percentile of token counts) on the crowd training split. Per fold, in `split_report.json`
- [x] *(pilot)* Build the vocabulary from the **training split only**. Same fastText 300-d embeddings as now

### Phase 3: Link lane
- [x] *(pilot)* Link RF (200 trees) tuned with grouped cross-validation on the training split. `python -m src.unified.link_lane --pilot tune`: the 16-setting grid of `rf_extractor.PARAM_GRID` (200 or 500 trees), 3 grouped folds of the train part only, highest F1 then lowest log loss. Pilot: CV F1 0.88–0.94
- [x] *(pilot)* Out-of-fold P, u and 37 contributions for every training row (no row is scored by a forest that saw it). `python -m src.unified.link_lane --pilot oof`: 5 grouped folds inside the train part, step 1 settings. Pilot: out-of-fold F1 0.89–0.93
- [x] *(pilot)* One final Link RF on all training rows scores validation and test. `python -m src.unified.link_lane --pilot final`; forests saved to `models/c_prune/<dataset>/`. Only val is summarised (pilot val F1 0.87–0.98); the test scores are written but not read until Phase 7
- [x] *(pilot)* Add `page_missing` → 40 numbers. `has_url` removed. Every row is also scored with its page hidden (13 page columns -1, the 4 outside columns kept, same forest), for modality dropout in Phase 6. `link_lane.load_lane(seed, fold)` returns both versions. Pilot: hiding the page lowers val F1 by about 0.02
- [x] *(pilot)* Check: P = bias + sum of contributions (existing test). Also checked on the written outputs: `python -m src.unified.link_lane --pilot check` (max gap 2e-15 over all 15 seed-folds)

### Phase 4: Text CNN (Phase A)
- [x] *(pilot)* Train the full 64-filter CNN on the training split with early stopping on validation, **embedding frozen** (`requires_grad = False`). `python -m src.unified.text_cnn --pilot train`: shared recipe, per-fold vocabulary and `SEQ_LEN_L`, 57,794 trainable parameters (Conv + temporary head). Pilot: val F1 0.85–0.92
- [x] *(pilot)* Save the checkpoint per seed (and fold): `phase_a_seed<S>_fold<F>.pt` + a `.json` sidecar (filters, kernel, `SEQ_LEN_L`, embedding folder, sha256 of the vocabulary, matrix and weights, val metrics). `python -m src.unified.text_cnn --pilot check` rebuilds every CNN from its sidecar alone and repeats its val metrics exactly

### Phase 5: Gate RF and pruning
- [x] *(pilot)* Record the 64 pooled filter scores for the training messages (eval mode). `python -m src.unified.gate_rf --pilot scores`: each fold's Phase A CNN rebuilt from its sidecar, dropout off, train part only (val and test unseen); `filter_scores_seed<S>_fold<F>.csv` = `message_id`, `label`, `filter_00`–`filter_63` (original filter ids). Pilot: 872–873 rows per fold, 0 dead filters
- [x] *(pilot)* Gate RF with 5 folds grouped by domain. Permutation importance scored by log-loss on each held-out fold, 30 repeats, averaged (`oof_logloss`). `python -m src.unified.gate_rf --pilot importance`: 200 trees, sklearn defaults; `importance_seed<S>_fold<F>.csv` = `filter`, `importance` (mean rise in log-loss), `spread` (standard error over the 5 folds), one column per fold. Pilot (12 min for 15 seed-folds): the Gate RF learns (out-of-fold log-loss 0.23–0.43 against 0.69 for the prior, F1 0.84–0.89), but single filters barely matter: only 0–10 of 64 are clearly above zero, and in 2 seed-folds (42/0, 123/3) none is above zero. The filters overlap, so shuffling one is covered by the others. This makes the next step (shuffle correlated filters together) worth doing before ranking
- [x] *(pilot)* Optional: shuffle highly correlated filters together, so similar filters don't split their importance. `python -m src.unified.gate_rf --pilot groups`: filters grouped by Spearman |rho| ≥ 0.8 on the train messages (average linkage, no labels used), each group shuffled by one shared row order, 30 repeats, on step 2's folds and forests; `group_importance_seed<S>_fold<F>.csv` = `group`, `size`, `filters`, `importance`, `spread`, one column per fold. Pilot (6 min): 21–35 groups per seed-fold (largest 7–23 filters). Groups clearly above zero: 0–7 per seed-fold, covering 7–34 filters, against 0–10 single filters in step 2. Three seed-folds (42/0, 123/3, 123/4) still have none: there, even whole groups cover for each other. How the group numbers make one ranking of the 64 filters is decided in step 4
- [x] *(pilot)* Save the ranking per seed (`filter_importance_seed*.csv`). **Rule, decided 2026-10-19:** one filter per group in turns. Groups go from most to least important (step 3). Round 1 takes each group's best filter (step 2 importance), round 2 each group's second best, and so on. Dead filters go last. So copies of one pattern are removed first, and the RF decides which patterns stay. `python -m src.unified.gate_rf --pilot rank` (per seed and fold, since each fold has its own CNN)
- [x] *(pilot)* Check the order against the others on **val** (no ranking reads it), before fine-tuning: `python -m src.unified.gate_rf --pilot compare`. Removed filters send 0. Orders: this rule, step 2 alone, whole groups first, reverse, 20 random sets. Scored two ways: the Phase A head as is, and a head refit on the train messages' kept scores (a stand-in for Phase 6 fine-tuning). Pilot result, **not better**: with the head as is, this rule is the worst (it keeps one copy of each pattern, and the old head still expects all the copies' votes). With the head refit, every order ties: val log-loss 0.199–0.211 against 0.198 for all 64 filters, even 16 random filters reach 0.214. Only at k = 16 does it beat whole groups and reverse (12 of 15 seed-folds). The synthetic pilot is so redundant that any filters do the job, so no order can win on it. **Re-run `compare` on the real crowd data.** The order is kept unless step 2 alone has the lower mean refit val log-loss there, in which case step 2 alone is used. Never decided on test
- [x] *(pilot)* `select_top_k` gives `keep_idx` for k = 48, 32, 16. `select_top_k(ranking, k)` in `rf_gating_module.py`: the top k of the step 4 ranking, in ascending id order (kept position i = original filter `keep_idx[i]`). It refuses a ranking that is not every filter once, or an impossible k. `PRUNE_KEEP = [48, 32, 16]` in `src/config.py`. `python -m src.unified.gate_rf --pilot keep` writes `keep_summary.csv` and checks that each smaller set lies inside the larger ones; `gate_rf.load_keep_idx(seed, fold, k)` gives one set to the later phases. Pilot: k = 16 holds one filter from each of 16 groups, k = 32 and 48 hold every group, 0 dead filters kept
- [x] *(pilot)* `prune_encoder` builds the k-filter CNN. Dense 64 drops the matching columns. In `cnn_encoder.py`: `prune_encoder(encoder, keep_idx)` copies the kept Conv rows and biases into a new k-filter `TextEncoder` (embedding copied and still frozen, source untouched); `prune_input_columns(linear, keep_idx, cnn_dim)` drops the removed filters' columns from any layer reading [filters | rest] and keeps the rest (the 40 link numbers) unchanged. Used on the Phase A temporary head now. In the current code the fusion Dense 64 is created fresh in Phase C, so in Phase 6 it is either built with k + 40 inputs or cut with `prune_input_columns` if a trained one is reused. `python -m src.unified.gate_rf --pilot prune` builds all 45 pruned CNNs (`gate_rf.pruned_phase_a` gives one to Phase 6) and checks the shapes, the frozen embedding and the trainable parameters: Conv 43,248 / 28,832 / 14,416 for k = 48 / 32 / 16, equal to section 5.3
- [x] *(pilot)* **Unit test:** before fine-tuning, the pruned model's output equals the full model's output with the removed filters set to 0. "Set to 0" = the gate mask applied to the full model, W ← g·W, b ← g·b, with g = 0 for removed filters. `tests/test_prune.py`: the removed filters send exactly 0, the kept scores are equal bit for bit, and the output is equal up to float rounding (the head adds k numbers instead of 64 with zeros among them), for the CNN alone and for a Dense 64 reading [filters | 40 link numbers]. Also checked on the real pilot checkpoints by `python -m src.unified.gate_rf --pilot prune`, on each fold's val messages: all 45 pruned CNNs, score gap 0, largest output gap 2.4e-7
- [x] *(pilot)* **Unit test:** with k = 64 (keep all), the output is unchanged. `tests/test_prune.py`: a shuffled ranking → `select_top_k(…, 64)` → `prune_encoder` gives the same weights and the same output **bit for bit** (same sizes, same weights, same order, so there is nothing to round); a Dense 64 reading [filters | 40 link numbers] is unchanged too. Also checked on all 15 real pilot checkpoints by `python -m src.unified.gate_rf --pilot prune`, on each fold's val messages: identical
- [x] *(pilot)* Save `keep_idx` (the gate mask, `gate_mask_seed*.json`) in a sidecar JSON next to the checkpoint. `python -m src.unified.gate_rf --pilot mask` writes `models/c_prune/<dataset>/gate_mask_seed<S>_fold<F>.json` next to each Phase A checkpoint (per seed and fold, since each fold has its own CNN). Per k = 48 / 32 / 16 it holds `keep_idx` and the gate mask (64 numbers: 1 kept, 0 removed). It also records the full ranking, the ranking rule (`RANKING_RULE` in `gate_rf.py`), the Gate RF settings, and the sha256 of the Phase A checkpoint and the ranking files. `gate_rf.load_gate_mask(seed, fold, k)` refuses the file if any of those changed (a filter number only means something for the CNN it was made from), or if the mask and `keep_idx` disagree. Every later step reads `keep_idx` from here (`pruned_phase_a` included). Pilot: 15 files written and read back, `prune` re-run from them

### Phase 6: Fusion and fine-tuning (Phase C)
- [x] *(pilot)* Pruned CNN + 40 link numbers → Concatenate → Dense 64 (weight decay) → 2 + softmax. `PrunedFusionClassifier` in `fusion_classifier.py`, a subclass of `UngatedFusionClassifier` with `cnn_dim = k`, as planned. It keeps `keep_idx` as a buffer saved with the weights, drops the Phase A temporary head (unused, and not counted), and its `parameter_groups(weight_decay)` puts the decay on Dense 64's weights only. `DENSE_WEIGHT_DECAY = 1e-4` in `src/config.py` (start value, tuned on validation). Dense 64 and the output layer are new, initialised from the seed. `python -m src.unified.fusion --pilot build` builds all 45 models (`fusion.build_pruned_fusion`), with each message lined up with its own 40 link numbers by `message_id` (`fusion.fusion_inputs`, which stops on a missing or extra row). It checks the CNN inside is the pruned Phase A CNN bit for bit, and that the forward pass works on the val messages. Trainable parameters 49,074 / 33,634 / 18,194 for k = 48 / 32 / 16, equal to section 5.3
- [x] *(pilot)* Fine-tune everything except the Link RF with the shared recipe. `python -m src.unified.fusion --pilot train` (`fusion.fine_tune`): Conv + Dense 64 + output train, the embedding stays frozen and the 40 link numbers are fixed inputs. Adam 1e-3 with weight decay 1e-4 on Dense 64's weights only, up to 30 epochs, batches of 64, patience 3 on val balanced accuracy (best epoch kept), label smoothing 0.1, class weights. Modality dropout, train only, redrawn each epoch from the seed: page hidden with p = 0.25 (the 40 numbers swapped for the blank-page version from Phase 3), text hidden with p = 0.10 (the k filter scores set to 0, via `PrunedFusionClassifier.forward(…, text_keep)`). Train rows use out-of-fold link numbers, val uses the final forest's, test is not read. Checkpoints: `models/c_prune/<dataset>/c_prune_seed<S>_fold<F>_k<K>.pt`; `train_summary.csv` records val metrics, epochs and training seconds (RQ2). Pilot (synthetic, pipeline test only): val F1 0.979 / 0.979 / 0.985 for k = 48 / 32 / 16 (range 0.92–1.00), against 0.88 for the text CNN alone (Phase 4) and 0.94 for the Link RF alone (Phase 3). Before fine-tuning (new Dense 64), val F1 was 0.22–0.47. Best epoch 0–15, about 1 second per model
- [ ] **Real data only. Lock the ranking rule** before choosing k: run `python -m src.unified.gate_rf compare` and read `ranking_compare_summary.csv`. Keep "one filter per group in turns" unless "single" (step 2 alone) has the lower mean `val_log_loss_refit`; then switch to single: set `RANKING_RULE = "single"` in `src/unified/gate_rf.py`, run `python -m src.unified.gate_rf mask` and `prune` again (the masks then record "single"), and re-run fine-tuning. Write the choice and its numbers here. Decided once, on validation only, never changed after the test is read. **Tool ready (2026-10-19):** `python -m src.unified.gate_rf lock` applies this rule to `ranking_compare.csv` and writes `ranking_rule_decision.json` (date, rule, numbers). It refuses to run a second time, and the `mask` step then refuses any `RANKING_RULE` other than the locked one. On the pilot (`--pilot lock`) it is a dry run that locks nothing. Pilot dry run: group_turns 0.20383, single 0.20370 (single lower in 24 of 45). It would say "single", but that is a coin flip on synthetic data, so nothing is changed
- [x] *(pilot)* Choose the final k on validation. **Rule, decided 2026-10-19: the one-standard-error rule** (Breiman et al. 1984; Hastie et al., *The Elements of Statistical Learning*, section 7.10). Per k, the mean val F1 over every seed and fold and its standard error (std / √15). The best k has the highest mean (a tie goes to the smaller k), and the final k is the **smallest** k whose mean is at least the best mean minus the best k's standard error. So the smaller, faster model wins when the gap is within noise (RQ2). One k for every seed and fold. `python -m src.unified.fusion --pilot choose` writes `final_k.json` (rule, per-k numbers, chosen k); `fusion.final_k()` gives it to the later steps. Re-run after `train` changes (e.g. a new ranking rule). Pilot: **k = 16** (mean val F1 0.985 ± 0.002, against 0.979 for k = 32 and k = 48), the best and the smallest at once
- [x] *(pilot)* Save checkpoint, `keep_idx` sidecar and Link RF per seed. Per seed, fold and k: the checkpoint `c_prune_seed<S>_fold<F>_k<K>.pt` (from `train`), the Link RF `link_rf_seed<S>_fold<F>.joblib` (from Phase 3), and now the sidecar `c_prune_seed<S>_fold<F>_k<K>.json` next to them. It holds `keep_idx`, the ranking rule, `final` (this k is the chosen one), the architecture (incl. the 40 link inputs in order), the recipe, the trainable parameters, the val metrics, and the name + sha256 of every file the model is built from: checkpoint, gate mask, Link RF, Phase A checkpoint, vocabulary, embedding. `python -m src.unified.fusion --pilot save`. `fusion.load_c_prune(seed, fold, k)` rebuilds model + Link RF from the sidecar alone (k defaults to the final k) and refuses the bundle if any file changed. `fusion.predict_messages` runs raw rows (text + 37 link features) through both, as the app will. Checked on all 45 pilot models: rebuilt from the sidecar and fed the **raw** val rows, each repeats its val metrics exactly (gap 0)

### Phase 7: Controls and tests
Every model uses the same data, splits, seeds, recipe and frozen embedding. The research questions and how each one is measured are in section 5.

- [ ] **Baselines:** plain CNN (text only) · CNN_CONCAT (64 text + raw link features) · UNGATED (64 text + 40 link numbers, no gates) · current GATED (separate gates)
  - *(pilot, 2026-10-19)* **Plain CNN done.** `python -m src.unified.baselines --pilot plain_cnn`: the same Phase A CNN, all 64 filters kept, no link input, the same code (`PrunedFusionClassifier`) and the same recipe (`fusion.fine_tune`). Modality dropout is off, since the text is its only input. 61,954 trainable parameters (section 5.3). `python -m src.unified.baselines --pilot compare` scores both on the test folds (every message once per seed, ids asserted identical; C-prune from the **raw** rows through the saved Link RF). It writes RQ1 metrics, the exact McNemar test, and stage 3/4 latency to `experiments/c_prune/<dataset>/baselines/`. On the real data it refuses to run without `--read-test` and logs every read in `test_reads.log`. **Pilot rehearsal (synthetic test folds, pipeline test only):** F1 C-prune k = 16 0.975 ± 0.002 vs plain CNN 0.865 ± 0.009; McNemar b / c = 159 / 11, 150 / 9, 166 / 10 (p < 1e-33 every seed); text model median 0.35 ms (C-prune, 18,194 parameters) vs 0.42 ms (plain CNN, 61,954); **the Link RF takes about 213 ms per message** (Python Saabas contributions over the trees), about 600× the text model. Look at this before the latency benchmark
- [ ] **C-prune** at k = 48, 32, 16
- [ ] **Random removal** at the same k, with the same fine-tuning. The RF ranking must beat this [S11]
- [ ] **From scratch:** a k-filter CNN trained from the start on the same schedule. The RF choice must beat simply training a smaller CNN [S13]
- [ ] **Reverse ranking** (keep the k least important). It should do worst, which shows the ranking means something
- [ ] **The other ranking rule** at the same k, with the same fine-tuning. If the locked rule is "one per group in turns", run "single" (step 2 alone), and the other way round. It is cheap (both orders already exist) and shows the panel how the rule not chosen would have done
- [ ] **Ablation:** C-prune + link gates (`rf_gates`, 0.1–3.0 on the 37 contributions) put back. Shows whether link-side gating adds anything
- [ ] **Variant C:** graded gates (0 or 0.5–3.0) merged into the Conv weights, from the same Gate RF ranking
- [ ] Print the gate mask (which filters are 1 and 0, and what each filter fires on) for the final model
- [ ] RQ1: accuracy, precision, recall, F1 (smishing = positive) for every model, mean ± std over the 3 seeds
- [ ] RQ2: trainable parameters for every model and every k (counted in code, checked against section 5.3)
- [ ] RQ2: latency benchmark (section 5.3): inspect each test link once, then time every model on the same messages; median, 95th percentile and mean per stage and in total
- [ ] RQ2: training time per step (Phase A, Link RF, Gate RF + permutation importance, pruning, fine-tuning) and in total
- [ ] RQ3: exact McNemar per seed, C-prune vs each baseline, Holm-corrected; message ids asserted identical
- [ ] Learning curve: 25 / 50 / 75 / 100% of the training split
- [ ] Test folds (or the test split) read **once**, after everything above is fixed

### Phase 8: Serving and demo
- [ ] `lejet_service.py` and `predict.py` load the pruned model (k from the sidecar)
- [ ] `explain_text.py` maps kept filters to their original ids, so the explanations still name the right phrases
- [ ] Remove `has_url` from the live feature vector
- [ ] Run the existing serving tests and the demo

### Phase 9: Writing
- [ ] Methodology: dataset, collection rules, the architecture diagrams of section 2
- [ ] Results: the tables of Phase 7
- [ ] Discussion: whether RF-ranked pruning beat random removal and from-scratch training, and what the deleted filters had been detecting
- [ ] Limitations: crowdsourced data size, legitimate-link scarcity, no prior text-CNN study with RF-ranked pruning

---

## 7. Risks to watch

| Risk | Why it matters | What to do |
|---|---|---|
| Too few legitimate messages with links | the model learns "has a link = smishing" style shortcuts | start legitimate collection first; keep ≥ 40% |
| Domain memorisation | test scores look better than real life | ≤ 20 per domain per class; split grouped by domain |
| Dead pages | page features become `-1` | inspect at submission time; `page_missing` flag |
| RF ranking ≈ random | the RF's role can't be shown | that is a valid finding; report it honestly. The Phase 7 controls make it visible |
| Small test set | real differences don't reach significance | aim for 4,000 messages; report confidence intervals |
| Link RF too strong | the CNN is ignored and the model just copies P | compare with UNGATED and plain CNN; report the text-only and link-only scores |

---

## 8. References

Full notes on every study (what was read, advantages, limitations) are on the architecture comparison page: https://claude.ai/artifact/VKVWvsVYyhZciNH9UmwayQ (C-prune card).

**Architecture (pruning)**
- [S9] Lin et al. (2020). HRank: Filter Pruning using High-Rank Feature Map. CVPR. arxiv.org/abs/2002.10179
- [S10] Sui et al. (2021). CHIP: CHannel Independence-based Pruning for Compact Neural Networks. NeurIPS. arxiv.org/abs/2110.13981 (**main study**)
- [S11] Blalock, Gonzalez Ortiz, Frankle & Guttag (2020). What is the State of Neural Network Pruning? MLSys. arxiv.org/abs/2003.03033
- [S12] He & Xiao (2023). Structured Pruning for Deep Convolutional Neural Networks: A Survey. IEEE TPAMI. arxiv.org/abs/2303.00566
- [S13] Wang et al. (2023). Why is the State of Neural Network Pruning so Confusing? arxiv.org/abs/2301.05219
- [S14] Singh & Plumbley (2022). A Passive Similarity based CNN Filter Pruning for Efficient Acoustic Scene Classification. arxiv.org/abs/2203.15751
- [S15] Harshit & Mounvik (2025). Accuracy-Constrained CNN Pruning for Efficient and Reliable EEG-Based Seizure Detection. Preprint. arxiv.org/abs/2509.05190
- [S16] Frankle, Dziugaite, Roy & Carbin (2020). Linear Mode Connectivity and the Lottery Ticket Hypothesis. ICML. arxiv.org/abs/1912.05671
- [S17] Frankle, Dziugaite, Roy & Carbin (2021). Pruning Neural Networks at Initialization: Why are We Missing the Mark? ICLR. arxiv.org/abs/2009.08576
- [S18] Wang, Qin, Zhang & Fu (2021). Neural Pruning via Growing Regularization. ICLR. arxiv.org/abs/2012.09243
- [F1] Jacovi, Sar Shalom & Goldberg (2018). Understanding Convolutional Neural Networks for Text Classification. BlackboxNLP @ EMNLP. arxiv.org/abs/1809.08037
- [F2] Liu et al. (2017). Learning Efficient Convolutional Networks through Network Slimming. ICCV. arxiv.org/abs/1708.06519
- [F3] Molchanov, Mallya, Tyree, Frosio & Kautz (2019). Importance Estimation for Neural Network Pruning. CVPR. arxiv.org/abs/1906.10771
- [F6] Li, Kadav, Durdanovic, Samet & Graf (2017). Pruning Filters for Efficient ConvNets. ICLR. arxiv.org/abs/1608.08710

**Dataset size and evaluation**
- Timko & Rahman (2024). Smishing Dataset I: Phishing SMS Dataset from Smishtank.com. CODASPY '24. doi.org/10.1145/3626232.3653282 · arxiv.org/abs/2402.18430
- Pritom et al. (2026). Short Message Service (SMS) Phishing Attacks and Defenses: A Systematic Review. Preprint. arxiv.org/abs/2604.11429
- Mishra & Soni (2021). DSmishSMS - A System to Detect Smishing SMS. Neural Computing and Applications. doi.org/10.1007/s00521-021-06305-y
- Goel, Ahmad, Jain & Goel (2024). Machine Learning Driven Smishing Detection Framework for Mobile Security. Preprint. arxiv.org/abs/2412.09641
- Kim (2014). Convolutional Neural Networks for Sentence Classification. EMNLP. arxiv.org/abs/1408.5882
- Beleites, Neugebauer, Bocklitz, Krafft & Popp (2013). Sample Size Planning for Classification Models. Analytica Chimica Acta 760, 25–33. doi.org/10.1016/j.aca.2012.11.007 (abstract and first part read in full; later sections from a summary — check before quoting)
