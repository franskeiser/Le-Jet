# Le-Jet progress and next steps

**Updated 2026-10-15. Read this first, then `walkthrough.md` and `rf_link_expert_architecture.md` for background.**
Those two files still describe the earlier reliability-gate design. This file and the architecture page below are the current plan.

- Current architecture page (static gating): https://claude.ai/artifact/8aL4np4p2Ta2sVRbJybeiR
- Earlier proposal (reliability gate), kept for comparison: https://claude.ai/artifact/Sypzgr9MdQGo9VAHyFL5ii

Always report **accuracy, precision, recall and F1**, mean ± std over seeds 42, 123, 2024.
Run Python with `.venv/Scripts/python.exe` (the global Python has no torch).

---

## 1. The design in five lines

Title constraint: *A Random Forest-Guided Static Gating Framework for Financial-Related Smishing Classification Using SMS Text.* So the gates must be **static** (fixed numbers computed once, never trained per message).

1. **Text path:** SMS text (link text removed) → CNN, trained on v2, then frozen → 64 text features → 64 static gates.
2. **Link path:** the link and its page (up to 87 features) → **one** Random Forest, trained on complete and blanked pages → calibrated `P_cal`, `u`, and one contribution per feature → static gates on the contributions.
3. **Static gates:** permutation importance from a forest on rows held out from the model it scores, rescaled to 0.1–3.0 with `gate = 0.1 + (importance − min) / (max − min) × 2.9`, saved as fixed numbers.
4. **Head:** gated text features, gated contributions, and `P_cal`, `u`, `page_missing`, `has_url` ungated → dense layer → verdict.
5. **Baselines:** **CNN** (text only, no gates), **CNN_CONCAT** (CNN text features + the 37 raw link/page features, no forest, no gates; added 2026-10-15) and **UNGATED** (same inputs as the proposed model, every gate 1.0). All are scored on the same `fresh_test`.

Dropped from the design: the learned per-message reliability gate (it is dynamic, so it conflicts with "static"). It can be future work.

---

## 2. What is done

| Done | Where | Result |
|---|---|---|
| RF vs 1D CNN on link/page features, no text | `src/unified/run_features_only.py`, `experiments/web_phishing/features_only/` | RF wins on 31 features |
| One forest (complete + blanked pages) vs two forests | `src/unified/run_single_rf.py`, `experiments/web_phishing/single_rf/` | One forest is enough |
| Web set prepared with domain-grouped splits | `src/unified/prepare_web_phishing.py` | `data/processed/web_phishing_linked.csv`, `data/splits_web_phishing/` (75/10/15) |
| Removed the RF-only and MLP-only fusion baselines | `src/unified/` | `run_tabular_only.py` deleted |
| Architecture, dataset and training plan | the artifact above | |

Real-page results on `web_phishing_clean.csv` (11,430 pages, domain-grouped splits, 3 seeds):

| Features | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 31 | RF | 0.911 ± 0.009 | 0.930 ± 0.014 | 0.875 ± 0.007 | 0.902 ± 0.009 |
| 31 | 1D CNN | 0.868 ± 0.020 | 0.866 ± 0.027 | 0.849 ± 0.022 | 0.858 ± 0.021 |
| 14 (URL only) | RF | 0.813 ± 0.010 | 0.816 ± 0.027 | 0.774 ± 0.014 | 0.795 ± 0.007 |
| 14 (URL only) | 1D CNN | 0.791 ± 0.009 | 0.813 ± 0.014 | 0.718 ± 0.007 | 0.763 ± 0.009 |

Single forest: F1 0.909 on complete pages, 0.781 with the page columns blanked (a dedicated URL-only forest gets 0.795).

How to claim this: the RF-over-neural lead is clear with 31 features and small with URL only. Do not claim RF superiority on URL-only. The 1D CNN is a weak stand-in for a neural model on a table, so an MLP comparison is still open.

Not a study result: everything under `experiments/every_sms*` used placeholder page features (an unrelated page paired with each SMS). On that data text alone scored about 0.95 and the gated model scored 0.948 against 0.951 ungated. It shows nothing about fusion.

**Repo state:** branch `new-model`. The `web_phishing` scripts and results are staged, not committed. `docs/walkthrough.md`, `docs/rf_link_expert_architecture.md` and this file are untracked. `data/processed/sms_link_features.csv` and `unified_dataset.csv` were deleted and not restored; the old unified trainers (`run_unified`, `run_nogates`, `run_baseline`, `compare_models`) depend on them and on placeholder data, so do not run them as they are.

### 2b. Trained on 2026-10-15 (no fresh data needed)

Run with `python -m src.unified.prepare_v2 --build-embeddings` then `python -m src.unified.train_static_gating` (stages `cnn`, `forest`, `gates`; 3 seeds). Outputs: `models/static_gating/` (checkpoints, forests, calibrators, gates) and `experiments/static_gating/` (metrics, gate values). The web set uses the 31-feature cleaned file, which is the feature set the inspector already produces; 87 features is not tried yet.

| Part | Data | Result (mean ± std, 3 seeds) |
|---|---|---|
| v2 prepared: links removed, 70/10/10/10 split by domain | `data/processed/v2_text.csv`, `data/splits_v2/` | 18,883 messages. Train is 59.8% smishing; val, gate and test about 53% (the biggest scam domains all go to train). New vocabulary of 22,880 tokens in `models/embeddings_v2/`. |
| Text CNN on v2, link text removed | `phase_a_seed*.pt` | v2_test: accuracy 0.952 ± 0.005, precision 0.950 ± 0.005, recall 0.959 ± 0.009, F1 0.955 ± 0.005. Early stopping fired at epoch 4-5. |
| One forest, complete + blanked pages | `forest_seed*.joblib` | web_test complete: F1 0.909 ± 0.007. Blanked: F1 0.781 ± 0.006. |
| Calibration (isotonic and Platt) | `calibrators_web_seed*.joblib` | Complete pages: isotonic cuts the calibration error from 0.059 to 0.035. Blanked pages: no improvement (0.045 raw vs 0.054 isotonic). To be rechecked on `fresh_dev`. |
| Static gates, 64 on the CNN and 31 on the forest | `cnn_gates_seed*.npy`, `rf_gates_seed*.npy` | Held-out balanced accuracy of the CNN gate forest 0.948 ± 0.004 vs 0.5 baseline. |

What the gates showed (these back the improvement ideas in the artifact):
- **RF gates are unstable.** Rank correlation of the importances across seeds is only 0.44-0.59. Between 8 and 15 of the 31 features have negative importance (noise) in each seed.
- **One feature dominates.** `page_rank` gets 3.00 in all three seeds, so min-max scaling pushes the rest into a narrow band (most land between 0.1 and 1.3). A rank-based transform and out-of-fold importance should help.
- **CNN gates** are 64 gates over learned dimensions that differ per seed, so they cannot be compared across seeds. Only about 2% sit at the floor.
- **Text alone is strong.** Even with links removed, the CNN scores F1 0.955 on v2_test, so the sealed fresh test has to be hard for fusion to show anything.

### 2c. Final feature set: 37 features (decided 2026-10-15)

The final link/page features are 37 of the raw 87: 20 from the link (11 URL shape + 9 word features) and 17 from the page and outside services (13 page + 4 outside-service). The list is in `src/unified/prepare_web_37.py`. Everything in 2b that used 31 features is superseded by the 37-feature run below; the 31-feature files are kept beside it (suffix `_f37` marks the 37 run).

Run: `python -m src.unified.prepare_web_37`, then `python -m src.unified.train_static_gating --features 37 --stages forest rfgates`. It reuses the same message ids and domain-grouped splits as the 31 run, so the two are directly comparable. Negative `domain_age` and `domain_registration_length` values are set to -1 (lookup failures).

One forest, complete and blanked pages (mean over 3 seeds, 1,515 test pages):

| Test pages | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Complete | 31 | 0.917 | 0.937 | 0.881 | 0.909 ± 0.007 |
| Complete | **37** | 0.915 | 0.886 | 0.938 | 0.911 ± 0.007 |
| 17 page columns blanked | 31 | 0.801 | 0.803 | 0.761 | 0.781 ± 0.006 |
| 17 page columns blanked | **37** | 0.827 | 0.784 | 0.870 | 0.825 ± 0.004 |

- With the page, the two feature sets are level (F1 0.909 vs 0.911, accuracy 0.917 vs 0.915). The 37 trades precision for recall.
- Without the page, 37 is better: accuracy +2.6 points, F1 +4.4 points, driven by recall (0.761 → 0.870). This is where the extra URL and word features help.
- The shift toward recall comes partly from tuning the forest by F1 (the repo's rule) with a 0.5 threshold. If precision matters more than recall for this study, tune by another score.
- Calibration (isotonic) on complete pages cuts the calibration error from 0.074 to 0.038; on blanked pages it does not help (0.064 raw vs 0.066).
- Forest gates with 37 inputs: 37 gates, `page_rank` 3.00 in all seeds, rank correlation across seeds 0.35-0.56, 13-22 of 37 importances negative per seed. Still unstable, same fixes apply.
- Head size: 64 + 37 + 4 = 105 inputs.
- Open: can the inspector produce all 37 for every message? `nb_redirection` needs a network request; `domain_in_brand` and `shortening_service` need lookup lists.

### 2d. Gate fixes for the forest gates (2026-10-15)

Script: `src/unified/improve_gates.py` (`python -m src.unified.improve_gates`, then `--export oof_logloss rankfloor`). Results: `experiments/static_gating/gate_stability_f37.csv`, all variants in `gate_variants_f37.csv`. The chosen gates are saved as `models/static_gating/rf_gates_seed*_f37_fix.npy` (37 gates per seed). The CNN gates are unchanged.

Why `page_rank` dominates: it is real signal in this dataset. 45% of phishing pages have rank 0 against 1.8% of legitimate; alone it gives F1 0.818; its importance (0.11) is about 3x the next feature and equals all other positive importances combined; dropping it lowers the forest's F1 from 0.898 to 0.870. It is not a bug. What needed fixing was the min-max rescale, which let it squeeze every other feature into a narrow band. Risk to test on the fresh set: the web set's legitimate URLs come from popular-site lists, so SMS links from small legitimate senders may have low rank and look suspicious.

Stability across the 3 seeds (Spearman of the gate vectors):

| Importance | Min-max | Rank | Rank + noise floor | Features not clearly above zero |
|---|---|---|---|---|
| One slice, F1 (first fit) | 0.486 | 0.486 | 0.636 | 29 of 37 |
| One slice, log-loss | 0.959 | 0.959 | 0.942 | 6 |
| 5-fold average, F1 | 0.893 | 0.893 | 0.796 | 26 |
| **5-fold average, log-loss** | 0.992 | 0.992 | **0.995** | 4 |

- Most of the instability came from scoring by F1 (a threshold). Log-loss alone fixes most of it; averaging over 5 held-out folds fixes the rest.
- Chosen: 5-fold log-loss importance (10 repeats per fold, forests trained like the main forest on complete + blanked pages), rank-based rescale of 0.5-3.0, and 0.1 for features not clearly above zero (importance minus 2 standard errors is not positive). `page_rank` stays at 3.0; the next are `nb_www` 2.92, `nb_hyperlinks` 2.84, `phish_hints` 2.76. Four features sit at 0.1 in every seed: `domain_in_brand`, `iframe`, `login_form`, `shortening_service`.
- This shows the gates are stable, not that they help. A feature with near-zero importance can still carry information that a correlated feature also holds (for example `login_form`), so 0.1 is a bet. Grouping correlated features before permuting is not done.

### 2e. Do the static gates help? Check on each side alone (2026-10-15)

Script: `src/unified/gating_check.py` (`python -m src.unified.gating_check`). Results: `experiments/static_gating/gating_check_{summary,by_seed,per_seed}.csv`.

A small head (Dense 64, ReLU, Dropout 0.5, 2 outputs) is trained per side with the features z-scored first and then multiplied by the gates. Text side: 64 frozen-CNN features on v2. Link side: the 37 forest contributions on the web set, with out-of-fold contributions for training rows. Variants: ungated (1.0), gated, and the same gate values shuffled onto the wrong features (5 shuffles per seed, averaged). 3 seeds.

| Side | Ungated F1 | Gated F1 | Shuffled F1 |
|---|---|---|---|
| Text, 64 CNN features (v2 test) | 0.954 ± 0.005 | 0.954 ± 0.005 | 0.954 ± 0.004 |
| Link, 37 contributions (web test) | 0.919 ± 0.005 | 0.913 ± 0.011 (first-fit gates 0.916 ± 0.007) | 0.913 ± 0.016 |
| Link, contributions + P, u ungated | 0.914 ± 0.013 | 0.915 ± 0.015 (first-fit 0.918 ± 0.004) | 0.912 ± 0.015 |

Accuracy, precision and recall move the same way (full table in the summary csv). Per-seed F1 differences from ungated range from -0.013 to +0.014, with no consistent sign.

**Reading:** the gates are stable but do not measurably help on either side, and the real gates do no better than shuffled ones, so the importance information is not what is acting. Likely reasons, not tested: a trained first layer rescales its own inputs; both sides are near their ceiling; no regularisation, so scaling matters little.

**What this does not test:** gating inside the fused head, where one branch can dominate the other, and small training sets (the 700-message pilot), where a feature prior could matter more. These are the cases the title's claim needs. The thesis should not claim gating helps until one of them shows it. Possible follow-ups: repeat this check with training sets of about 700 rows; test gating on the fused head once fresh data exists; compare against a plain gating alternative such as feature selection by importance.

### 2g. Lean fresh set and the fused pipeline, end to end (2026-10-15)

**Data.** `data/processed/feature_dataset_lean.csv` is 1,284 pages (642 per class, 1,141 distinct hosts) with 39 link/page features, no SMS text. It differs from the final 37 by name: it has `avg_words_raw`, `longest_word_path`, `shortest_words_raw` and lacks `shortest_word_path`. `python -m src.unified.prepare_lean_synthetic` converts it: drops the three extras, computes `shortest_word_path` from the URL path and query (agrees with the raw web set on 98.2% of rows), sets negative `domain_age` / `domain_registration_length` to -1, and writes **synthetic** SMS text. Output: `data/processed/fresh_lean_synthetic.csv` (message id, `sms_text`, label, `link`, `link_domain`, 37 features). The text is templated English/Taglish financial wording containing the row's own link; 25% of messages per class use wording shared by both classes (`--neutral-rate`), and there are urgent legitimate and calm smishing templates. **It is invented, so nothing below is a study result.** Replace `sms_text` with the real SMS when the group has it (decision 1 in section 6).

**Pipeline.** `python -m src.unified.run_fused_lean` (3 seeds, about 2 minutes). Per seed: at most 20 rows per domain per class, then 70% `fresh_dev` / 30% `fresh_test` grouped by domain, and `fresh_dev` cut again into head-train / head-val (grouped). Frozen v2 CNN gives 64 text features (links stripped; words outside the v2 vocabulary are dropped, 5% of test tokens). Frozen web forest (`forest_seed*_f37`) gives 37 contributions, P and u for complete and blanked pages; P is Platt-calibrated on `fresh_dev` (the forest never saw those rows). Static gates are the saved ones (`cnn_gates_seed*`, `rf_gates_seed*_f37_fix`), applied after z-scoring; P_cal, u, `page_missing`, `has_url` are never gated. All three models share one head (Dense 64, ReLU, Dropout 0.5, 2) and one recipe. CNN = 64 text features only. UNGATED = 105 inputs with every gate 1.0. GATED = the same inputs with the static gates. Fused heads train with modality dropout (page replaced by its blanked version p=0.25, text zeroed p=0.10; same draws for UNGATED and GATED). Checks in the code: gate shapes and values, UNGATED gates all 1.0, identical test message ids for every model, no NaN. Outputs in `experiments/fused_lean/` (`summary.csv`, `per_seed.csv`, `mcnemar.csv`, `predictions.csv` with message id and seed, `split_sizes.csv`).

Test = about 381 messages per seed from about 278 domains (1 row sharing an exact URL with the web set is dropped; 172 of the lean domains also appear in the web set and 82 in v2, mostly large sites, flagged not removed). Mean ± std over 3 seeds, complete pages:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| CNN (text only) | 0.861 ± 0.005 | 0.892 ± 0.012 | 0.825 ± 0.024 | 0.857 ± 0.008 |
| CNN + raw link features (CNN_CONCAT) | 0.941 ± 0.009 | 0.969 ± 0.018 | 0.913 ± 0.018 | 0.940 ± 0.010 |
| UNGATED | 0.960 ± 0.012 | 0.965 ± 0.016 | 0.955 ± 0.016 | 0.960 ± 0.012 |
| GATED (static) | 0.965 ± 0.011 | 0.974 ± 0.015 | 0.957 ± 0.006 | 0.965 ± 0.010 |
| *frozen v2 CNN's own classifier* | 0.752 ± 0.037 | 0.715 ± 0.012 | 0.844 ± 0.090 | 0.773 ± 0.046 |

Pages blanked (page not reached): UNGATED F1 0.935 ± 0.002, GATED 0.943 ± 0.018 (the CNN is unchanged). With one branch zeroed at test time: GATED link zeroed F1 0.861, text zeroed 0.938; UNGATED 0.850 and 0.927.

**Added baseline, CNN_CONCAT (2026-10-15):** the frozen CNN's 64 text features concatenated with the 37 *raw* link/page features (signed log1p, then z-score), plus `page_missing` and `has_url` (103 inputs). No forest, no gates, same head, same recipe, same modality dropout and random draws. A blanked page has its 17 page columns set to the training mean. Pages blanked: F1 0.924 ± 0.019. Per-seed F1 CNN_CONCAT / UNGATED / GATED: 0.930 / 0.971 / 0.976 (seed 42), 0.949 / 0.948 / 0.963 (123), 0.942 / 0.961 / 0.956 (2024).
- CNN_CONCAT beats text-only by about 8.3 F1 points (McNemar p <= 0.0006 in every seed), so most of the gain comes from simply giving the head the link features.
- The forest adds a smaller amount on top: UNGATED is 2.0 F1 points above CNN_CONCAT, GATED 2.5. McNemar GATED vs CNN_CONCAT: p = 0.0005 (seed 42), 0.38 (123), 0.36 (2024); UNGATED vs CNN_CONCAT: p = 0.0026, 1.00, 0.19. Significant in one seed of three, so **the forest's benefit over raw concatenation is suggestive, not shown**. The forest's own input here is the web-trained model, so part of any edge may be the 9K extra training pages it saw, not the contributions.

**Reading (honest):**
- Fusing the link evidence beats text alone by about 10 F1 points; McNemar is significant in every seed (p < 0.001, about 45 messages fixed against 6 broken). This shows the pipeline works and that the head reads the link branch.
- GATED is 0.5 F1 points above UNGATED on average, but the per-seed difference changes sign (+0.005, +0.015, -0.005) and McNemar is not significant (p 0.11 to 0.69). **No evidence that static gating helps**, same as 2e. Do not claim it.
- The size of the text-only gap is set by my template design (the neutral-rate and the hard templates), not by real text; a different `--neutral-rate` gives a different number. The link side carries most of the signal here (text zeroed: F1 0.93), partly because the forest was trained on the web set, which shares 172 domains with the lean set.
- Only about 380 test messages per seed: accuracy uncertainty is about ±2 points, which is also why 0.5 points between GATED and UNGATED cannot be resolved.
- Not done: the shuffled-gate control inside the fused head, the optional page-state rule, the 700-row small-data check, a fresh forest refit on `fresh_dev` (the frozen web forest is used instead, so its saved gates still apply), and the "same recipe" choice of the static branch scale by cross-validation.

### 2f. Next session (decided 2026-10-15)

The title stays: Random Forest-guided static gating. The gating check (2e) showed no gain on either side alone, so the next work tests gating where it might still help. Nothing is running in the background.

1. **Repeat the gating check with about 700 training rows** (`src/unified/gating_check.py` with a training-size option, several random draws, same variants including shuffled gates). Tests gating in the small-data regime of the fresh pilot. Not written yet.
2. **Build the fused pipeline end to end** (head with modality dropout, CNN and UNGATED baselines under one recipe, the "same message ids" assertion, the zeroing checks) and smoke-test it on placeholder data. Scores from that test mean nothing. **Done 2026-10-15, see 2g** (`src/unified/run_fused_lean.py`, synthetic text).
3. **Static branch scale:** one fixed scale per branch, chosen by cross-validation on `fresh_dev` with "no scaling" as an option. Needs fused data.
4. **Needed from the group first:** the 700-message numbers (class split, how many with a link, how many live, distinct domains, whether each row has the SMS text, financial or not), the inspector check for the 37 features, and the adviser's answer on the fresh-set dev/test split.
5. **Commit** the work: the scripts, results and docs from 2026-10-15 are untracked.

---

## 3. The datasets

| Dataset | File | Rows | Balance | Used for | Split |
|---|---|---|---|---|---|
| Web phishing set, 87 features | `data/raw/web-phishing/train-1.csv` (7,658) + `test.csv` (3,772); cleaned 31-feature version is `data/processed/web_phishing_clean.csv` | 11,430 | 5,715 / 5,715 | Forest, forest gates | Merge the two files, then 75% train / 10% gate slice / 15% test, grouped by domain |
| v2 SMS | `data/raw/sms/v2_2026-09-15.csv` | 18,885 | 10,955 / 7,930 | Text CNN, CNN gates | 70% train / 10% val / 10% gate slice / 10% test, stratified; domains with 100+ messages go to train |
| Fresh inspected SMS | to collect | target 5,000 (floor 3,000; **700 so far**) | aim 50/50 | Forest refit, calibration, head, baselines, final test | 70% `fresh_dev` / 30% `fresh_test`, grouped by domain |

Facts about v2 you need:
- Only 4,846 messages have a real link (4,405 smishing, **441 legitimate**), from 1,446 domains. Three scam domains hold 2,095 of them.
- 3,056 messages have the `<URL>` placeholder instead of a link (3,014 smishing, 42 legitimate).
- So "has a link" predicts smishing about 94% of the time. **Remove link text from v2 before training the CNN.**
- v2 is not filtered to financial messages. Scope is undecided (section 6).
- The old `sms_v2_balanced.csv` added synthetic legitimate messages. Use the raw v2 file.

The web set and v2 are not paired (no SMS has page features). Only the fresh set pairs text, link and page features for the same message.

---

## 4. What to do now (no fresh data needed)

Tasks 2, 3, 4, 5 and 6 were first done on 2026-10-15 (see section 2b). Task 6 uses the plain min-max gates; the improvements listed in it are still to try. Task 1 (87 features) and tasks 7-10 are open.

Owner column is blank on purpose: fill it in.

| # | Task | Owner | Notes |
|---|---|---|---|
| 1 | Re-split the web set from the **raw 87-feature files** (merge `train-1.csv` + `test.csv`), 75/10/15, domain-grouped | | `prepare_web_phishing.py` currently reads the 31-feature clean file. Needs the feature-set decision (section 6). |
| 2 | Split v2: 70/10/10/10, remove link text, class weights, domain-grouped | | New script. Reuse `src/data/split_dataset.py` helpers. |
| 3 | Train the text CNN on v2 (Phase A) and freeze it | | `build_embeddings.py --data` and `LEJET_EMBEDDINGS_DIR` support a per-dataset vocab. Without it, new words raise `KeyError` on purpose. |
| 4 | Move blank-augmented training into the forest trainer | | Logic is in `run_single_rf.py`. Add `page_missing`. |
| 5 | Add forest calibration (isotonic or Platt) on out-of-fold predictions, and check it on the web set | | Not built. Niculescu-Mizil & Caruana (ICML 2005) is the reference. |
| 6 | Fit the static gates: CNN gates from the v2 gate slice, RF gates from the web gate slice | | `urf.gate_importance` and `gate_transform` exist. Improvements to consider: out-of-fold importance, a rank-based transform, log-loss scoring, grouping correlated features. |
| 7 | MLP-only vs RF on the real pages | | Open. Makes the neural-vs-RF claim fair. |
| 8 | Check the domain shift: forest trained on web pages, read with page columns blanked, scored on the 4,846 v2 links | | The 441 legitimate links limit precision and recall. |
| 9 | Related-work pass | | Gated multimodal units (Arevalo et al. 2017, arXiv 1702.01992), ModDrop (Neverova et al. 2016), calibration (Niculescu-Mizil & Caruana 2005). No source found for tree disagreement as a trust signal. Check how much of the RF+CNN idea is already published before wording a novelty claim. |
| 10 | Update `walkthrough.md` and `rf_link_expert_architecture.md` to the static-gating design | | Both still describe the reliability gate. |

Extra ablations, **only after the group agrees** (each adds a model): CNN-gates-only vs RF-gates-only vs both vs neither; a shuffled-gates control (same values on the wrong features); an optional page-state rule (a fixed gate value, e.g. 0.5, chosen from {0.3, 0.5, 0.7, 1.0} by grouped CV on `fresh_dev` when the page is dead).

---

## 5. What waits for fresh data

Refit the forest with `fresh_dev`, calibrate, train the head, train the CNN and UNGATED baselines with the same recipe, test once on `fresh_test`. Details and the exact data fed to each model are in the artifact.

**Same test set for every model:**
- One `fresh_test` per seed; all three models are scored on exactly those messages.
- The test part never chooses gates, thresholds, checkpoints or hyperparameters.
- Save predictions with `message_id` and seed; assert the three files hold identical ids before comparing; use a paired McNemar test per seed.
- The three models share the same frozen CNN, forest, calibrator, dev split, class weights and recipe.

**Fresh set rules:**
- Both classes need links. Keep dead and blocked pages. Record `hours_to_visit`, `fetch_status`, `received_at`, `visited_at`.
- Financial themes only (banks, e-wallets, payments, loans, crypto), including hard legitimate messages such as urgent real bank and e-wallet notices.
- At most 20 messages per domain per class. Collect over at least four weeks and use the latest batch as `fresh_test` if possible.
- No overlap with v2 or the web set; flag any shared domain.
- At least 500 live-page messages per class overall and 150 per class in the test part.
- Missing values in `fresh_set_spec.md` are written as `-1`; the forest trained on blanked pages uses NaN. Convert `-1` to NaN in the page columns when loading.

**With only 700 messages (pilot mode):** a sealed test would be about 210 messages, which is too small (accuracy uncertainty about ±4 points at 90%). Detecting a 2-point difference needs about 1,500 test messages, assuming the models disagree on about 8% of messages; that assumption is a guess. So:
- Do not train the full head (about 6,500 weights). Use a tiny logistic regression on a few inputs (a text logit, `P_cal`, `u`, `page_missing`, `has_url`).
- Evaluate with grouped 5-fold cross-validation over the 700, 3 seeds, fixed hyperparameters. Label it a pilot.
- Treat the 700 as the start of `fresh_dev` and collect a separate sealed `fresh_test` later.
- Do not use v2's linked messages as paired training data (about 231 legitimate vs 2,289 smishing after capping at 20 per domain).
- How the static gates sit upstream of a tiny head is not designed yet.

**Please send the group these numbers about the 700:** class split, how many have a link, how many pages were live, distinct domains, whether each row includes the SMS text, and whether they are financial-themed.

---

## 6. Open decisions

| Decision | Why it matters |
|---|---|
| Does each fresh row include the SMS text, or only link features? | Without text the head cannot be trained or tested. |
| Adviser approval to split the fresh set into dev and test | `fresh_set_spec.md` currently says test-only. |
| How many of the 87 features the inspector can produce at runtime | Sets the number of RF gates and head inputs (99 inputs with 31 features, 155 with 87). The artifact currently draws "31". |
| Filter v2 to financial messages, or keep it as a general pretraining corpus | The title scopes the study to financial smishing. |
| Title wording | Page and domain features come from fetching the link, not from the SMS text itself. |
| Should the CNN baseline get the CNN-side static gates? | Currently it has none, so the chain is CNN → +forest → +gates. |
| Add the optional page-state rule and the gate ablations? | Each is a new model. |

---

## 7. Useful commands

```
# prepare the web set and its splits (31-feature clean file today)
.venv/Scripts/python.exe -m src.unified.prepare_web_phishing

# RF vs 1D CNN, no text
.venv/Scripts/python.exe -m src.unified.run_features_only --data data/processed/web_phishing_linked.csv --splits-dir data/splits_web_phishing --out experiments/web_phishing

# one forest vs two forests, complete vs blanked pages
.venv/Scripts/python.exe -m src.unified.run_single_rf --data data/processed/web_phishing_linked.csv --splits-dir data/splits_web_phishing --out experiments/web_phishing
```

Each run takes about 8-10 minutes for 3 seeds. `data/` is git-ignored, so regenerate it with the first command.
