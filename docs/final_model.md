# Le-Jet: final model

**Written 2026-10-12.** Every number below was read from a saved file in this repo. Figures from the synthetic demo are labeled; they show how the model behaves and are **not study results**.

**Thesis:** A Random Forest-Guided Static Gating Framework for Financial-Related Smishing Classification Using SMS Text

---

## 1. The model in one picture

```
                                   ┌─ Random Forest feature extractor (frozen) ───────────────────┐
SMS link ─► 14 link-spelling       │  url_only forest: reads the 14 link features                  │
            features               │  full forest:     reads those 14 + page_rank + 12 webpage     │
            (+ page, domain,       │                   + 4 domain/external features (31 inputs)    │
            external features      │  page reached → full,  otherwise → url_only                   │
            when a page was        │  output per link:                                             │
            reached)               │    P  = mean tree probability       u = spread across trees   │
                                   │    c_<feature> × 31 = how many probability points each        │
                                   │      feature pushed P up or down (0 if the forest did not    │
                                   │      read it, so page slots are 0 when no page was reached)  │
                                   └───────────────────────────────┬───────────────────────────────┘
                                                                   │  has_url, P, u, c_×31  (34 values)
SMS text ─► tokens (max 50) ─► fastText embedding, 300-d (fine-tuned)               ▼
         ─► Conv1d: 64 filters, kernel 3 ─► ReLU ─► max-pool ─► dropout 0.2   × RF-output gates (34)
         ─► 64 text features ─► × CNN gates (64) ─────────────────┐                │
                                                                  ▼                ▼
                                                        concatenate (64 + 34 = 98)
                                                        ─► Dense 64 ─► ReLU ─► dropout 0.5
                                                        ─► Dense 2 ─► softmax (legitimate / smishing)
```

**The four required parts**
1. **RF as feature extractor:** the forest turns each link into a verdict (P, u) and a per-feature breakdown (the 31 `c_` values). No other model reads the link.
2. **CNN features:** 64 max-pooled filters of width 3 over the fastText-embedded message.
3. **URL features in the same RF as the page features:** the link-spelling features and the page/domain/external features are inputs to one forest (`full`), so they share one verdict and one breakdown. The head never sees the raw link features.
4. **Gating:** static gates between 0.1 and 3.0 on the 64 CNN features and on the RF outputs.

**Gates**
| Gated inputs | Gate comes from |
|---|---|
| 64 CNN features | a 200-tree forest fitted on the CNN features of the RF-Fitting split, permutation importance, rescaled to 0.1–3.0 |
| 31 `c_` slots | that feature's permutation importance in the `full` forest (validation part of the public web set for the real-data runs; the RF-Fitting rows with a reached page for the demo), rescaled to 0.1–3.0 |
| `has_url`, `P`, `u` | fixed at 1.0 (P and u already are a forest's output; ranking them against its own inputs would be circular) |

Why the `c_` values: they are in probability units (−1 to 1), so they arrive on the same scale as P and u, with no need to standardize `url_length` or `domain_age`. A feature the forest did not read contributes exactly 0, so one 31-slot vector serves both extractors.

**Size.** 6,646,130 trainable parameters in the forward pass: embedding 6,582,000, conv 57,664, `Dense 64` 6,336, `Dense 2` 130 (plus an unused 130-parameter temporary head left over from Phase A). The extractors and the gates are fixed.

**Code:** `cnn_encoder.py`, `fusion_classifier.py`, `rf_gating_module.py`, `rf_extractor.py`, `rf_feature_extractor.py` in `src/features/model/`.

### What the `c_` breakdown is
Walking one decision tree from root to leaf, every split changes the node's phishing fraction. That change is credited to the feature that was split on. Summed over a path and averaged over the 200 trees:

`P = bias + Σ c_feature` (checked to 1e-5 in `tests/test_rf_feature_extractor.py`; `bias` is the roots' average phishing fraction, 0.47 for the real `url_only` forest).

This is the Saabas path decomposition. It is not a Shapley value, so it can depend on the order in which a tree splits.

### The two extractors (Random Forest, 200 trees each)
| | Inputs | Used for |
|---|---|---|
| `url_only` | the 14 link-spelling features | every old SMS link; any live link whose page failed |
| `full` | 31: the 14 + `page_rank` + 12 webpage + 4 external (`whois_registered_domain`, `domain_registration_length`, `domain_age`, `dns_record`) | live links whose page was fetched |

Page status picks the extractor. It is never an input to the head. Two extractors (not one with masking) because old SMS links are mostly dead: a lookup made today would teach "dead domain = scam" (`docs/decision_web_inspection.md`).

---

## 2. Why this architecture (decided 2026-10-12)

Three ways of giving the head the link information were trained on the same CNN, data and recipe:

| Recipe | Head receives besides text |
|---|---|
| `link_pu` (first design) | 14 raw link features + `has_url` + P + u |
| `pu_only` | `has_url` + P + u (link features only inside the forest) |
| `rf_vec` (**final**) | `has_url` + P + u + 31 per-feature contributions |

**Real SMS data, 3 seeds, balanced accuracy, real test messages** (`experiments/fusion/`):

| Group | CNN only | `link_pu` | `pu_only` | `rf_vec` |
|---|---|---|---|---|
| All | 0.962 ± 0.008 | 0.960 ± 0.005 | 0.961 ± 0.007 | 0.962 ± 0.007 |
| No link | 0.939 ± 0.008 | 0.936 ± 0.002 | 0.935 ± 0.002 | 0.935 ± 0.009 |
| With link | 0.942 ± 0.024 | 0.904 ± 0.016 | 0.926 ± 0.028 | 0.931 ± 0.019 |

Exact McNemar tests per seed (`experiments/fusion/mcnemar.csv`), all real messages: `rf_vec` vs CNN-only p = 0.47, 0.90, 1.00; `rf_vec` vs `link_pu` p = 0.052, 0.64, 0.49. **No significant difference anywhere.** `rf_vec` is the closest of the gated designs to the CNN on messages with links, and `link_pu` the furthest.

**Synthetic demo data, 3 training seeds, accuracy** (`experiments/demo/recipes_summary.csv`):

| Group | `link_pu` | `pu_only` | `rf_vec` |
|---|---|---|---|
| All | 0.949 ± 0.010 | 0.964 ± 0.002 | 0.962 ± 0.001 |
| Neutral wording, page reached | 0.890 ± 0.042 | 0.931 ± 0.006 | 0.920 ± 0.002 |
| Neutral wording, page missing | 0.766 ± 0.011 | 0.809 ± 0.004 | 0.816 ± 0.004 |

**The decision, stated plainly**
- Keeping the link features **inside the forest** (`pu_only`, `rf_vec`) beat passing them raw (`link_pu`) on the synthetic data by about 1.3 points with much less variation across seeds, and was no worse on the real data.
- **`pu_only` and `rf_vec` are tied on accuracy.** I chose `rf_vec` because it is the only one where gating applies to the RF's outputs (in `pu_only` the RF side has nothing to gate), which the thesis requires, and because its per-feature breakdown explains each verdict. That is a choice on requirements and interpretability, not a win on accuracy.

---

## 3. Training phases: which part of the architecture each one trains

| # | Phase | Trains | Stays fixed | Data it reads |
|---|---|---|---|---|
| 0 | **Extractors** | the two Random Forests | – | web-page rows only (§4) |
| 1 | **RF outputs** | nothing (inference) | extractors | each message's link (and page) features → P, u, 31 `c_` |
| 2 | **Phase A: CNN alone** | embedding, conv layer, a temporary 2-way head (discarded afterwards) | extractors | Training split |
| 3 | **Gates** | nothing in the network; one forest is fitted on the CNN features, read and discarded; RF-slot gates computed from the extractor's importances | CNN from Phase A (frozen) | RF-Fitting split; public web set (RF-slot gates) |
| 4 | **Phase C: fusion** | embedding and conv (fine-tuned from Phase A), `Dense 64`, `Dense 2` | gates, extractors | Training split (same rows as Phase A) |
| 5 | **Test** | nothing | everything | Test split, scored once |

**Recipe (Phases A and C, `src/features/model/train.py`)**
- Adam, learning rate 1e-3, batch 64, up to 20 epochs in the demo (30 for the real-data runs).
- Cross-entropy with label smoothing 0.1 and inverse-frequency class weights computed from the training labels.
- Early stopping with patience 3 on validation **balanced accuracy**; the best epoch's weights are kept.
- Validation is 15% of the Training split, carved out of it, so the RF-Fitting and Test splits are never touched.

**Mixed training in Phase C (`MixedAuxDataset`).** When training rows have page evidence (the demo), about half of the reads show the head the `url_only` forest's output for that row instead of the `full` one, so it learns both. On synthetic data this lifted accuracy on page-reached messages from 0.83 (trained on `url_only` output only) to 0.89 (`docs/training_rules.md`). **The real-data runs cannot do this**: the old SMS data has no page features, so Phase C there trains on `url_only` output only. Mixed training for the real model needs a live-links training set.

**CNN gates (Phase 3).** Pass the RF-Fitting messages through the frozen Phase A CNN; split those rows 70/30; fit a class-balanced 200-tree forest on the 70%; score each feature by permutation importance on the 30% (balanced accuracy); fall back to Gini importance if the forest cannot beat guessing the majority class; rescale to 0.1–3.0.

---

## 4. What data was used, file by file

### 4.1 The demo model (`models/demo/`, the one the web page loads)

| File | What it is | Used for |
|---|---|---|
| `data/demo/synthetic_demo.csv` | **7,487 synthetic messages × 40 columns**, built by `python -m src.demo.synthetic` | Phases 1–5 |
| `data/processed/web_phishing_clean.csv` | 11,430 real web pages × 34 columns (5,715 phishing, 5,715 legitimate) | the source of the URLs and page features in the synthetic file, **and** the training data for the demo extractors (Phase 0) |
| `models/embeddings/embedding_matrix.npy` + `vocab.json` | 21,940 words × 300 dimensions | starting word vectors for the CNN (fine-tuned in Phases A and C) |
| `pretrained/cc.en.300.bin`, `cc.tl.300.bin` | pretrained fastText vectors (English, Tagalog) | only to build the embedding matrix (done once) |

**How `synthetic_demo.csv` is built (`src/demo/synthetic.py`)**
1. Split the 11,430 web pages **by domain**: about 55% go to extractor training (6,009 rows), the rest form a pool of URLs. A domain is never in both.
2. Wrap each pool URL in an invented SMS: scam wording, legitimate wording, or **neutral wording** shared by both classes (so the text alone cannot decide those). Add 2,139 invented messages with no link.
3. The message label is the URL's label (phishing → smishing). About 30% of linked rows are marked "page unreachable" and their page features set to `-1`.
4. Split by domain: Training 5,470, RF-Fitting 686, Test 1,331.

| Split | Messages | Smishing | Legitimate | With link | Without link |
|---|---|---|---|---|---|
| Training | 5,470 | 2,938 | 2,532 | 3,938 | 1,532 |
| RF-Fitting | 686 | 341 | 345 | 487 | 199 |
| Test | 1,331 | 665 | 666 | 923 | 408 |

**Outputs of the demo run:** `models/demo/extractors/{url_only,full}.joblib`, `phase_a.pt`, `phase_c.pt`, `gates_seed42.npy` and `gates.csv`, `meta.json` (recipe `rf_vec`, train seed 42, mixing probability 0.5).

### 4.2 The real-data pipeline (`run_experiment.py --variant rf_vec`)

| File | What it is |
|---|---|
| `data/raw/sms/v2_2026-09-15.csv` | **18,885 real SMS** (10,955 smishing, 7,930 legitimate) |
| `data/processed/sms_v2_balanced.csv` | the above + 3,025 synthetic legitimate messages with links = 21,910 (10,955 per class), built by `add_synthetic_legit_links.py` |
| `data/processed/sms_link_features.csv` | 14 link features + `has_url` + `link_status` for every message, built by `build_sms_link_features.py` |
| `data/splits/seed_{42,123,2024}/{train,rf_fit,test}_ids.txt` | grouped Training / RF-Fitting / Test lists per seed, built by `split_dataset.py` |
| `data/raw/web-phishing/{train-1,test}.csv` → `data/interim/hussein-train-and-test-merged.csv` → `data/processed/web_phishing_clean.csv` | the public web set, merged (11,430 rows × 89 columns) and cleaned to 32 features (`clean_web_phishing.py`) |
| `models/rf_extractor/{url_only,full}_seed{42,123,2024}.joblib` | extractors trained on the web set, grouped by domain 70/15/15 per seed |
| `experiments/rf_extractor/importance_per_seed.csv` | permutation importances of the `full` forest, the source of the 31 RF-slot gates |

Rules applied: the 3,056 messages whose link was replaced by the text `<URL>` are excluded from training and evaluation (`docs/training_rules.md` rule 1), which leaves 7,941 smishing and 10,913 legitimate. Headline scores use real messages only (`source == "v2_2026-09-15"`). Outputs: `models/fusion/rf_vec/`, `experiments/fusion/rf_vec/`.

### 4.3 Not used for training
- No live or fresh pages (the fresh set doesn't exist yet).
- The old SMS corpus did **not** train the demo model's weights. It is used for one thing: the 21,940-word vocabulary and the fastText starting vectors were built from it (`build_embeddings.py`, reading `sms_v2_balanced.csv` without the redacted rows).

---

## 5. Results of the final model

### 5.1 Real evidence: the extractors on the public web set (3 seeds, test split)
| Extractor | Accuracy | F1 | ROC AUC |
|---|---|---|---|
| `url_only` (link spelling) | 0.815 ± 0.021 | 0.835 | 0.903 |
| `full` (+ page, domain, external) | 0.921 ± 0.015 | 0.930 | 0.978 |

Page and domain evidence adds about 10 accuracy points. **Caveat:** these are PhishTank and Alexa pages, not SMS links (`experiments/rf_extractor/`).

### 5.2 Real evidence: the full pipeline on the old SMS data
Table in §2. **On the old data the final architecture matches the CNN alone** (McNemar not significant in any seed). About 82% of old scam domains are dead, nearly all real linked messages are smishing, and the CNN is already about 99% accurate on them, so there is little room to improve.

### 5.3 Synthetic demo (mechanism check only; one seed, 1,331 test messages)
| Group | n | Text-only CNN | Final model |
|---|---|---|---|
| All | 1,331 | 0.828 | 0.962 |
| No link | 408 | 1.000 | 1.000 |
| Link, class-specific wording | 461 | 1.000 | 1.000 |
| Link, neutral wording | 462 | 0.504 | 0.892 |
| ...and the page was reached | 328 | 0.485 | 0.921 |
| ...and the page was missing | 134 | 0.552 | 0.821 |

The neutral-wording rows are built so the text carries no signal, which is why the text-only model sits at chance there. The gain is designed in: it shows the link evidence reaches the verdict, and says nothing about how often such messages occur.

---

## 6. What is real, what is synthetic

| | Real | Synthetic or invented |
|---|---|---|
| Web pages, URLs and their page/domain features | ✔ public set (Hannousse & Yahiouche, 2021) | |
| Demo SMS text, "no link" messages | | ✔ written by me |
| Which pages count as unreachable in the demo | | ✔ random 30% |
| Page inspection in the demo app | | ✔ simulated from a stored table; no link is ever opened |
| Extractor scores (§5.1) | ✔ on web pages | |
| Old SMS results (§2, §5.2) | ✔ | 3,025 of the 21,910 rows are synthetic legitimate messages (excluded from headline scores) |
| Demo scores (§5.3) | | ✔ not study results |

## 7. Claims you can and can't make

**Can:** the architecture trains end to end with all four required parts; the forest's per-feature breakdown adds up exactly to its verdict; page and domain evidence adds about 10 points for the extractor on web pages; the CNN cannot see the link or the page (links are replaced by `<URL>`), so only the extractor can use that information; on the old SMS data the final model is statistically the same as the CNN alone.

**Can't (yet):** that the extractor, the breakdown or the gates improve accuracy on real SMS; anything about live links. That test needs the real fresh set. Mixed training for the real model needs a live-links training set.

## 8. Known weaknesses
- **The gates' contribution is untested.** No run compares the final model with the same model without gates, so I cannot say whether the RF-slot or CNN gates change anything beyond what the dense layer would learn.
- **Gates and fine-tuning drift.** CNN gates are computed from the Phase A CNN, but Phase C fine-tunes the encoder (conv weights moved 77% in the earlier demo model; same-filter features correlate 0.78 between phases). Freezing the encoder in Phase C would keep the gates valid; not tested.
- **In the demo the 64 CNN gates are all exactly 1.0.** The synthetic templates are so easy to separate that permuting any single CNN feature changes nothing, so the code falls back to neutral. Only the 31 RF-slot gates are active in the demo; the real-data runs have varied CNN gates (0.1–3.0).
- **Saabas contributions** depend on split order and are not Shapley values.
- Link features come from a re-implementation; `random_domain` matches the original on 92% of rows.
- One seed for the demo; three for the real-data results.

## 9. Reproduce
```powershell
# Demo model (about 10 minutes)
python -m src.demo.synthetic      # data/demo/synthetic_demo.csv + demo extractors
python -m src.demo.train_demo     # final recipe (rf_vec) -> models/demo/
python -m src.demo.app            # http://127.0.0.1:8000

# Real-data pipeline (3 seeds)
python -m src.features.model.rf_extractor
python -m src.features.model.run_experiment --variant rf_vec
python -m src.features.model.mcnemar

# Recipe comparison (§2)
python -m src.features.model.run_experiment --variant pu_only
python -m src.demo.compare_recipes
```
