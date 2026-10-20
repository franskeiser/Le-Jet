# Relevance dataset: SMS text and link that are about the same thing

Built 2026-10-20. **Synthetic. A rehearsal set for the pipeline, never study evidence.** The meta file's `text_source` says which rows have
real wording (`isda_real`) and which are written by templates (`synthetic`).

Files, all in `data/processed/` (gitignored):

| File | What |
|---|---|
| `crowd_relevant_synthetic.csv` | 6,681 rows: the crowd schema's 57 columns minus `label_source`, plus `has_url` (57) |
| `crowd_relevant_synthetic.meta.csv` | one provenance row per message: pair type, where the text, link and features came from, relation, brands and themes |
| `crowd_relevant_synthetic.summary.json` | counts of everything below |
| `crowd_relevant_synthetic.audit.json` | the audit numbers below |

Rebuild (deterministic, same seed gives the same file):
`python -m src.data.build_relevant_dataset` then `python -m src.data.check_relevance`. Code: `src/data/build_relevant_dataset.py`,
`relevance_lexicon.py` (brands and themes read from a link and from words), `relevance_templates.py` (wording), `check_relevance.py` (audit).
Tests: `tests/test_relevant_dataset.py` (32).

## 1. What was asked, and what was done

Asked: 6,000 to 7,000 rows from `isda_sms_dataset.csv` (text) and `hunt_live_features.csv` (page features), secondary `train-1.csv`,
`test.csv`, and the synthetic pilot for legitimate rows; the text and the link must be relevant to each other (a Smart text carries a Smart
link), with variety (a suspicious text with a clean link, a legitimate-looking text with a bad link, and so on).

| Source | Used for |
|---|---|
| ISDA (3,070 messages) | real wording. The 310 "spam but not smishing" rows (ads, job offers) are left out: their label is ambiguous. 1,639 rows used |
| train-1 + test (11,430 links, 50 / 50) | page features of live links (`tt`) |
| crowd pilot (631 legitimate, 626 phishing links) | page features of live links (`lean`), and Smart / Maya / GCash pages for brand pairs |
| hunt (932 live phishing links) | **links only** (see 3.1) |

## 2. How a text and a link are made to fit

* **A real ISDA message keeps the link it contains.** Its 20 URL features are computed from that link by the repo's offline extractor
  (`lejet.features.hannousse_url`). Its 17 page / domain features come from a fetched page of the same domain or brand when one exists (a
  `smrt.ph` text borrows a `smrt.ph` page), otherwise the page is "not fetched".
* **A synthetic message is written after its link is chosen**, from what the link says about itself: its brand, its theme, the title in its
  path, its site name. A link naming PayPal never gets a Smart text.
* **A genuine notice may only speak for a brand whose own domain it links to.** `hi-smart.ru` is not Smart; a deep Microsoft docs page does
  not get an "account security alert"; a government text only links to a `.ph` host. A phishing row may imitate a brand (that is what
  phishing does), so only phishing rows may carry a look-alike.

Relation recorded per row, re-derived by the audit from the words and the link alone (links cut out of the words first):

| Relation | Meaning | Rows with a link |
|---|---|---|
| `own_link` | a real ISDA message with the link it came with | 1,054 |
| `brand` | text and link name the same brand | 856 |
| `page` | the words name the page (its title or site) | 1,699 |
| `theme` | the words and the link share a theme (delivery, login, job...) | 555 |
| `generic` | neutral wording that fits any link; **compatible, not relevant** | 1,767 |
| `conflict` | text names one brand, link another | **0** |

So 70% of link rows are relevant in one of the four senses, 30% carry neutral wording on purpose (see 5).

## 3. The design: the cells

| Cell | Wording | Link | Label | Rows |
|---|---|---|---|---|
| S1 | scam | phishing, in the same voice or theme | smishing | 1,577 |
| S2 | neutral, or copied from a genuine notice (a receipt, a bill) | phishing | smishing | 1,173 |
| S3 | scam (job, giveaway) | a real clean platform page (Facebook, Instagram, Google form, LinkedIn, Zoom) | smishing | 93 |
| S0 | scam with a number to call or text | none | smishing | 270 |
| L1 | genuine notice, a share naming the page, or neutral | clean, its own | legitimate | 2,998 |
| L2 | genuine but urgent or alarming (OTP warnings, an overdue bill, a sale ending tonight, a filing deadline) | clean, its own | legitimate | 90 |
| L0 | genuine, no link | none | legitimate | 480 |

Totals: **3,568 legitimate, 3,113 smishing**; 5,931 rows with a link, 750 without (`has_url = 0`). English 4,584, Taglish 2,097.
Neutral wording is used for **both** classes, so on those rows the text alone says nothing (text-only AUC 0.51).

### 3.1 Two deliberate departures from "use hunt's page features"

Every pool of page features has its own crawler signature. A random forest tells two pools **of the same class** apart with AUC 0.986 to
0.998, and hunt against anything with **AUC 1.000** (hunt never looked up `page_rank`, whois or domain age; removing those four still leaves
0.974). Hunt is all phishing, so using its pages would let a model learn "which crawler" instead of "which link". Hence:

1. **Hunt supplies links only.** Its page features are not used (`--hunt-pages` turns that back on). Its links are stored defanged
   (`hxxps://x[.]com`), which the pipeline's link extractor reads. Its 20 URL features are computed from the link text like every other
   offline row.
2. **Each live crawler contributes both classes in similar numbers:** `tt` 1,369 phishing / 1,636 legitimate, `lean` 344 / 485.

Rows whose page is "not fetched" are treated alike in both classes: the 17 page / domain features are -1, `nb_redirection` is 0, the URL
features are recomputed from the link. 39.8% of smishing link rows and 31.3% of legitimate link rows are like this (dead smishing pages are
real; the gap is reported, not hidden). `fetch_status` is `dead` for them and `no_link` for rows without a link.

## 4. Audit (5-fold cross-validation grouped by link domain, so a domain is never in train and test; AUC, 0.5 = nothing)

| Check | Result | Reading |
|---|---|---|
| Contradictions (written rows) | **0** | |
| Source labels alone (crawler, text source, has_url, page_fetched) | **0.553** | the data's origin barely predicts the label |
| Text alone (link cut out) | 0.960 | real ISDA text alone: 0.993; synthetic wording alone: 0.945 |
| Text alone on neutral-wording rows (1,433) | 0.513 | the neutral wording really is neutral |
| Link alone (37 features + page_missing) | 0.971 | inside one crawler: tt 0.971, lean 0.952, not-fetched 0.971 (the genuine link signal) |
| Text + link | 0.994 | |
| Mean text-only P(smishing) by cell | L1 0.17, L2 0.34, S2 0.64, S3 0.77, S1 0.92, S0 0.86 | the cross-evidence cells (L2, S2, S3) sit in the middle |
| Exact duplicate texts / templates | 0 / 4,338 | |
| Distinct link domains; max rows per domain per class | 4,187; 60 | |

Brand coverage (texts naming the brand that carry a link naming it too): Smart 54 of 86 legitimate texts with a link (the rest are real ISDA
messages linking to `onelink.to/smartapp` or `maya.ph`), Maya 82 of 96. GCash-named smishing (122 texts) has **no** GCash link: those are
real casino spam messages that list GCash as a deposit method.

## 5. Knobs and what they trade

Neutral wording fits any link, so it is compatible but not relevant. More of it makes the words less decisive; less of it makes more rows
relevant. Measured on whole builds:

| Setting (`--smish-neutral --smish-legit-style --legit-neutral`) | Relevant link rows | Text-only AUC |
|---|---|---|
| 0.35 0.40 0.45 | 61% | 0.936 |
| **0.20 0.40 0.30 (default)** | **70%** | **0.960** |
| 0.12 0.45 0.20 | 77% | 0.975 |

The real ISDA text is nearly separable on its own (0.993), which sets a floor on the text-only score. The link signal is the real Hannousse
features, so it is as strong as on the real benchmark.

## 6. Differences from the crowd schema

* Rows without a link are kept (`has_url = 0`, `link = -1`, `link_domain = nolink_<n>` so each is its own group in a grouped split; with
  one shared `-1` they formed a single group of 750 and the splitter put them all in one test fold).
* The per-domain cap is 60, not 20: a brand's own domain legitimately carries many messages. `bcd.check()` enforces 20 and is not used here.
* Counts and flags are integers, ratios are floats. The pipeline's own splitter accepts the file (`split_crowd.run`, group by domain: 5 folds,
  about 1,336 test rows each, smishing share 0.466 in every fold, largest group 84 rows).
* `label_source` is dropped (it was `isda` or `synthetic`; that provenance is `text_source` in the meta file).
* `sender_type` is `alphanumeric` or `mobile` for ISDA rows (from the sender field) and `unknown` for written rows; `category` is the theme.

## 7. Limits to remember

* **Synthetic.** 5,042 of 6,681 texts (75%) are written by templates around real links; 1,639 are real ISDA messages. The thesis rule "no synthetic text in the real study"
  still holds: this set is for building and rehearsing.
* **The no-link rows are not wired into the models yet.** E2's `has_url` bypass exists in `FiLMTextCNN`, but training, evaluation and
  `film.predict_messages` do not pass `has_url`, and the web inspector and forest are not skipped for these rows.
* **Obfuscated links count as no link.** Some ISDA smishing hides its link (`tinyur!.com/ 42m8rjbv`, `{9y1 5.c om}`). The extractor cannot
  find them, so they are `S0` rows.
* **Real links are stored.** ISDA and train/test links are as in their sources; only hunt links are defanged. Do not open them in a browser.
* **Near-duplicate wording.** 1,946 rows sit in templates used five or more times (mostly the shared neutral lines). A split that groups by
  link domain does not group by wording; for the CNN, consider also grouping on `template_id`.
* **Smishing page-missing is higher than legitimate** (39.8% vs 31.3%), so `page_missing` carries a little class information, as it does in
  real data.

## 8. Retrain on this dataset, with has_url in the pipeline (2026-10-20)

Run with `LEJET_PILOT_DATASET=data/processed/crowd_relevant_synthetic.csv` and `--pilot`; outputs sit under `crowd_relevant_synthetic` in
`experiments/film/`, `experiments/c_prune/`, `models/` and `data/splits_crowd/` (the earlier pilot's results are untouched). Order: `split_crowd
--group-by domain`, `build_crowd_embeddings`, `link_lane tune / oof / final / check`, `film_inputs`, `cnn_baseline train`, `concat_control train
--warm-start none`, `film train --warm-start none` (E2, the main model) and `... --no-bypass` (the ablation), then `film_compare`, `rf_only_compare`,
`voting_baseline`. 3 seeds x 5 domain-grouped folds, 6,681 messages, about 1,336 test messages per seed, each tested once per seed.

How has_url is carried: the link tensor is `[P, u, page_missing, has_url]`. The forest is tuned, fitted and scored on messages with a link only; a
no-link message gets zeros and `page_missing = 1`. E2 skips its generator for `has_url = 0` (gamma = 1, beta = 0). The concatenation control reads
has_url as a fourth input (62,210 trainable parameters). The ablation ignores has_url. 15 of 15 models of every kind rebuild from their sidecars with
a gap of exactly 0, and the app path agrees.

F1 on the test folds, mean of 3 seeds (smishing positive; RF-only abstains on a no-link message, so it predicts legitimate there):

| Model | All, complete | All, page hidden | With a link | No link |
|---|---|---|---|---|
| KNN-RF-ETC voting (text only) | 0.861 | 0.861 | 0.858 | 0.885 |
| CNN-only | 0.864 | 0.864 | 0.856 | 0.945 |
| RF-only | 0.865 | 0.713 | 0.906 | 0.000 |
| Concatenation (with has_url) | 0.954 | 0.914 | 0.954 | 0.954 |
| **E2, has_url bypass (main)** | **0.953** | **0.914** | **0.954** | **0.942** |
| E2 without the bypass | 0.953 | 0.912 | 0.954 | 0.941 |

Reading: the link evidence is what lifts the model from about 0.86 to 0.95 (McNemar, Holm-corrected: significant in all 3 seeds against CNN-only).
**E2 does not beat the concatenation control** (p = 1.0, 1.0, 0.93), and **the has_url bypass adds nothing measurable** over E2 without it (p = 1.0, 1.0,
0.054 in the one seed where it is close, with the no-bypass model slightly ahead on some groups). On no-link messages the text-only CNN is as good as
any link-aware model (F1 0.945). So on this dataset the FiLM claim and the bypass claim are both unsupported; the dataset is synthetic and a pilot.
