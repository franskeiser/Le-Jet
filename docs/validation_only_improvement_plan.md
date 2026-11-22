# Validation-only plan to improve (or fairly judge) the Proposed model

Draft, 2026-11-22. **Nothing in this plan has been run.** It follows the error analysis of the Step 5 predictions
(Proposed F1 .941 vs Concatenation .949 and CNN-only .946; McNemar significant against the Proposed model in 5 of 6 tests).

## 0. What we learned that this plan is based on

| Finding (from the saved out-of-fold predictions) | Evidence |
|---|---|
| The Proposed model ranks messages about as well as the baselines | AUC .9878 vs .9896 (Concatenation), .9882 (CNN-only) |
| It loses through **false alarms**, not misses | per seed, false alarms 237 vs 190 / 148; missed smishing 123 vs 120 / 172 |
| Most lost messages are close calls | 77% have a smishing probability between 0.3 and 0.7 |
| The extra false alarms are concentrated | `web_legit_lab` messages linking big sites (instagram, youtube, spotify, netflix, tiktok; page rank 10): +117 vs Concatenation |
| Gambling operators are hard for every model | 20-53% false alarms on casinoplus / bingoplus / arenaplus |
| A higher cut-off would close most of the gap | best single cut-off 0.58-0.66 gives F1 .943-.950 (found with test labels, **so it is evidence only, not a setting we may use**) |

Hypotheses, not yet tested: (H1) the Proposed model's cut-off sits too low because the FiLM path is over-confident toward smishing;
(H2) the Link RF's score is pushed toward smishing on big-brand pages; (H3) the baselines partly use the style of the lab-made legitimate
messages (short, bare link, casual text), a shortcut the Proposed model does not.

## 1. Ground rules (written before anything is run)

1. **Test folds are not used to choose anything.** They have been read twice. Every choice below is made on the validation part of the same fold (the `val` rows of `data/splits_crowd/...`; for example seed 42 fold 0 has 4,605 train / 767 val / 1,343 test).
2. **Equal effort.** Every rule applied to the Proposed model is applied in the same way to CNN-only, Concatenation Fusion and KNN-RF-ETC.
3. **Few, fixed candidates.** At most the candidates listed in section 3. Each one is logged, including those that do not help. No candidate is added after seeing a result.
4. **One validation criterion**, fixed now: balanced accuracy on the fold's validation part (the criterion the training loop already uses for early stopping), with F1 reported beside it.
5. **Per-fold choices only.** A cut-off for fold k is chosen from fold k's validation rows only. Validation rows of other folds may be test rows of fold k, so pooling them would leak test labels.
6. **Honest reporting.** The current results stay in the thesis as the first results. Anything that follows is labelled as a second version made after a diagnostic, with the reason.
7. **The final claim is made on a fresh holdout** (section 5), run once.

## 2. Phase A, diagnose on validation data only (no retraining, about half a day)

A1. **Export validation predictions.** The saved model weights (`models/film/<dataset>/*.pt` with their `.json` sidecars) are reloaded and score each fold's `val` rows, for all four models. This step never touches the `test` part. New script, for example `scripts/validation_predictions.py`, output `experiments/film/<dataset>/validation_study/`.

A2. **Check H1 (cut-off).** On validation rows: how many false alarms and misses per model at 0.5, and the validation balanced accuracy over a fixed grid of cut-offs (0.30 to 0.70, step 0.02). If the Proposed model's best cut-off is clearly above 0.5 on validation too, H1 is supported without looking at test.

A3. **Check H2 (Link RF).** On validation rows only, list the Link RF's smishing score by source and by page rank (big sites vs others), and compare with the Proposed model's final score on the same rows. If the RF already gives high scores to the big-brand legitimate pages, the problem is upstream of FiLM.

A4. **Check H3 (text-style shortcut).** On validation rows, split messages into "bare link or very short text" and "longer text". Compare each model's false alarms in each group. If CNN-only has about none in the first group while the Proposed model does, H3 is supported.

A5. **Stop rule.** If none of H1-H3 shows on validation data, we stop and report the results as they are (Chapter 5 limitation). We do not search further for a reason.

## 3. Phase B, candidate fixes (only those whose phase A check supports them)

| Candidate | What changes | Retraining? | Applied to baselines how |
|---|---|---|---|
| **B1. Validation-chosen cut-off** (supported by H1) | Replace 0.5 by the cut-off with the best validation balanced accuracy in that fold (grid above; ties go to the value nearest 0.5; smoothed by taking the middle of the cut-offs within 0.002 of the best, because a fold's validation set has only about 770 rows) | No, only a different cut-off on saved probabilities | Same rule, each model with its own validation cut-off |
| **B2. Better Link RF score** (supported by H2) | Calibrate the RF's score (Platt or isotonic fitted on the validation rows of the fold), or give FiLM the RF's uncertainty beside its score | RF calibration: cheap; FiLM change: retrain the Proposed model, 15 folds | Concatenation and CNN-only already get the raw link features, so they get no change; this asymmetry is stated in the thesis |
| **B3. FiLM starts near "do nothing"** (only if A3 is not decisive) | Initialise the FiLM scale at 1 and shift at 0, so the first epochs equal the plain concatenation head | Retrain the Proposed model | Concatenation unchanged (it is the same head without FiLM) |

Order: B1 first (free). B2 and B3 only if B1 does not close the gap on validation. Each candidate is judged by validation balanced accuracy and F1 averaged over the 15 fold-seed pairs, against the same number for the unchanged model. A candidate is kept only if it improves the average **and** improves at least 11 of the 15 pairs.

Not on the list, on purpose: tuning learning rate, layer sizes, dropout and the like for the Proposed model alone. Those are many small choices, each one a chance to win by luck.

## 4. Data points to settle in parallel (correctness, not model tuning)

- Open label doubts: fbmplay.com (1/1 split), taguigcom.cam "FREE SP1NS", Netflix receipt, HSBC promo, contis.ph cake promo.
- Gambling operators: confirm applus.ph and the group's rule with the adviser. Optionally check the official PAGCOR licensee list.
- Source shortcut: `web_legit_lab` is 100% legitimate (2,466 messages, 2,453 different templates) and `web_synthetic` is 100% smishing. Ask the adviser whether to report results with and without these sources.
- Any label change means a full rerun of Steps 3-10 (about 35 minutes of training plus the step scripts, as in the last rerun), so do all label changes in one batch.

## 5. The fresh holdout (needed for any claim that the changed model beats the others)

- Collect new messages that nobody has used for training, validation or the earlier tests: realistic legitimate messages with links (not lab-templated), new smishing, and some with no link. Sources: the Azure submission form, group-collected messages, a new ISDA slice if one is unused.
- Freeze the holdout before training anything: save the file, hash it, write down its size and class counts.
- Size: the current differences are about 0.5 F1 points, which is about 2% of messages disagreeing. To detect a difference that small with McNemar we need on the order of 2,000 or more new messages. If we cannot collect that many, the fair statement is "no detectable difference".
- Run each of the four models once, with the cut-off rule fixed in phase B, and run Step 10 (McNemar) on the result. Report whatever comes out.

## 6. What we do not promise

- We do not know that phase B will bring the Proposed model above Concatenation. The honest outcomes are: (a) it closes the gap, (b) it ties, (c) it still trails. All three are reportable.
- A validation cut-off applied to all four models can help the baselines as much as the Proposed model.
- Time: phase A about half a day; B1 about an hour after A; B2/B3 about one to two hours of training each plus checks; section 5 depends on how fast messages can be collected.

## 7. Items to confirm with the adviser before phase B

- Is a second model version after a diagnostic acceptable, reported as such?
- McNemar adjustment (guide item 39: none, Holm, Bonferroni).
- Whether the report may exclude or separately report the two single-source sets (`web_legit_lab`, `web_synthetic`).

---

## Phase A results (run 2026-11-22; validation rows only)

Scripts: `scripts/validation_predictions.py` (A1) and `scripts/validation_study_analysis.py` (A2-A4, decision rules written inside it before the run).
Outputs: `experiments/film/lejet_final_dataset/validation_study/` (`validation_predictions.csv`, `A2_cutoffs.csv`, `A3_link_rf.csv`, `A4_short_messages.csv`, `summary.txt`).
Check: the reloaded models reproduce the recorded validation balanced accuracy of training exactly (difference 0.0 in all 60 model-fold runs). No test row was written or used.

| Hypothesis | Result | Evidence |
|---|---|---|
| **H1** cut-off too low for the Proposed model | **Supported** | best validation cut-off above 0.5 in 13 of 15 runs (mean 0.577); CNN-only 0.407 and Concatenation 0.460. At 0.5 the Proposed model makes 27.7 false alarms per fold against 16.3 (Concatenation) and 11.9 (CNN-only), with about the same misses as Concatenation (17.9 vs 18.3) |
| **H2** Link RF over-scores big-brand pages | **Not supported** | the RF flags only 0.9% of legitimate page-rank-10 messages, yet the Proposed model gives them 25.8% false alarms (CNN-only 1.6%, Concatenation 9.0%). The false alarms are not coming from the RF's score |
| **H3** text-style shortcut (short messages) | **Supported, with a twist** | on legitimate messages of 3 words or fewer once the link is removed, false-alarm rate: Proposed 37.6%, CNN-only 0.9%, Concatenation 1.3% (longer messages: 5.4% / 2.9% / 3.9%) |

Further look at H3 (exploratory, validation and training rows only):
- These short messages are almost all **bare links** (an Instagram reel, a YouTube or TikTok link and nothing else). The Proposed model calls 80 of 81 legitimate bare-link rows smishing (98.8%); the baselines call about none.
- There are only **17 such messages in the whole dataset** (9 legitimate, 8 smishing; 4 Instagram, 3 YouTube). Because the splits keep a domain together, the validation part of a fold can hold 15 of them while its training part holds 1 (seed 42 fold 0). So the models are tested on a message type they have almost never trained on. The baselines default to "legitimate" there; the Proposed model defaults to "smishing".
- This one type accounts for **83 of the Proposed model's 170 extra false alarms** (versus Concatenation) on the validation rows. The other 87 come from longer messages, where the Proposed model is still a little worse (5.4% vs 3.9% false alarms).

What this means for phase B: B1 (validation-chosen cut-off) is supported and cheap. B2 (calibrating the RF) is not supported and is dropped. B3 (FiLM starting near "do nothing") stays a candidate because the problem is in how the Proposed model combines text and link when the text is empty, but it needs its own check on whether FiLM, not the data, causes the bare-link behaviour. Separately, the data has too few bare-link messages (17) for any model to learn them, so the fresh holdout should include more realistic short and bare-link legitimate messages.

### B1 estimate on validation rows only (2026-11-22, `validation_study/B1_cutoffs.csv`)

Cut-off by the B1 rule (middle of the cut-offs within 0.002 of the best validation balanced accuracy), one per fold, 15 folds:
Proposed 0.567 (range 0.47-0.66), CNN-only 0.401 (0.30-0.50), Concatenation 0.470 (0.36-0.56), KNN-RF-ETC 0.637 (0.53-0.70).
The cut-off moves a lot from fold to fold (SD about 0.05-0.07), because one fold's validation part has only 767 messages.

Honest gain estimate (the cut-off is picked on one half of a fold's validation messages, split by duplicate group, and scored on the other half; 20 splits x 15 folds):

| Model | balanced accuracy at 0.5 | with the chosen cut-off | change |
|---|---|---|---|
| Proposed Model | .9422 | .9394 | **-0.0028** |
| CNN-only | .9412 | .9469 | +0.0057 |
| Concatenation Fusion | .9537 | .9530 | -0.0007 |
| KNN-RF-ETC | .8303 | .8678 | +0.0375 |

Reading: a validation-chosen cut-off does **not** help the Proposed model (or Concatenation); it helps CNN-only a little and KNN-RF-ETC a lot. The earlier "best single cut-off gives F1 .943-.950" came from the test labels and is an in-sample best case, not an expected gain. Caveat: choosing on half a fold's validation is noisier than choosing on all of it, so the real gain could be somewhat better, but not by an amount that changes the picture. B1 alone will not close the gap; the Proposed model's false alarms on bare links (probabilities 0.6-0.78) are not removed by a global cut-off that varies this much between folds.

---

## B3 result (run 2026-11-22; validation rows only; `scripts/b3_variants_validation.py`, `experiments/film/lejet_final_dataset/b3_variants/`)

**Correction to this plan.** B3 as written ("start FiLM at gamma = 1, beta = 0") is already how every Proposed model is built (the generator's weights start at zero), so it could not be tested. It was replaced, before running, by two variants the model already supports that target the phase A finding (the generator averages the text Conv scores over all 50 positions, mostly padding for a bare link): **C1** `mask_padding=True` and **C2** `use_average=False` (the generator reads the link numbers only).

The control (the unchanged model, retrained here) reproduces the group's recorded validation balanced accuracy exactly (largest difference 0.000000).

| Variant | Validation balanced accuracy (mean of 15) | F1 | Better than control in | False alarms / fold | Misses / fold | Bare-link false alarms | Other legitimate false alarms |
|---|---|---|---|---|---|---|---|
| Control (unchanged) | .9412 | .9344 | n/a | 27.7 | 17.9 | 80 / 81 | 5.32% |
| C1 mask padding | .9336 | .9262 | 1 / 15 | 30.1 | 21.2 | 81 / 81 | 5.88% |
| C2 link numbers only | .9431 | .9366 | 10 / 15 | 24.7 | 19.0 | 72 / 81 | 4.75% |
| (Concatenation, recorded) | .9541 | | | | | | |

Keep rule (better on average **and** in at least 11 of 15 pairs): **neither variant is kept.** C1 is worse. C2 is +0.002 on average and better in 10 of 15, which is within noise and still 0.011 below Concatenation; it does not fix the bare-link false alarms (72 of 81 remain).

Conclusion of the validation-only study: the cut-off (B1) does not help, the Link RF is not the cause (B2 dropped), and the two cheap FiLM variants (B3) do not close the gap. We stop here, as the stop rule says, and do not add more candidates after seeing these results.

---

## Phase C, FiLM upgrade study (run 2026-11-22; validation rows only; `scripts/film_upgrade_validation.py`, `experiments/film/lejet_final_dataset/film_upgrade/`)

Goal set by Franz: the Proposed model significantly better than every baseline in all nine Step 10 McNemar tests. On the 6,714-row data this
needs about b − c ≥ +31 to +39 per pairing (accuracy about .955 / .957 / .961 for seeds 42 / 123 / 2024).

**New finding that motivated it:** modality dropout hides the text 10% of the time by zeroing the 64 pooled numbers. In Concatenation and
CNN-only the link numbers still reach the head on those rows. In FiLM the link only acts through the text numbers, so on those rows the head sees
zeros: the link is hidden too, and the baselines get "decide from the link" practice that FiLM never gets.

Candidates (fixed before running; new `FiLMTextCNN` options, defaults unchanged, control replay gap 0.000000):
D1 `text_dropout="input"` (hide the words before the Conv, so gamma and beta still act); D1F2 D1 + generator reads the 37 RF contributions;
D1F3 D1 + two-layer generator; D1F23 both; D1F1 D1 + link numbers also into the head (not pure FiLM).

| Variant | Val bal-acc | Better than control in | False alarms / fold | Misses / fold | Lab-legit false alarms (15 folds) | Projected b − c vs CNN-only (42 / 123 / 2024) | vs Concat |
|---|---|---|---|---|---|---|---|
| control | .9410 | n/a | 24.2 | 20.9 | 181 | −56 / −18 / −24 | −19 / +4 / −41 |
| **D1** | **.9495** | **12 / 15** | 20.0 | 18.5 | 119 | −36 / +5 / +33 | +1 / +27 / +16 |
| D1F2 | .9379 | 7 | 20.8 | 25.7 | 86 | −87 / −26 / −6 | −50 / −4 / −23 |
| D1F3 | .9454 | 10 | 18.5 | 22.5 | 117 | −54 / +3 / +15 | −17 / +25 / −2 |
| D1F23 | .9373 | 5 | 19.9 | 26.9 | 119 | −84 / −44 / +5 | −47 / −22 / −12 |
| D1F1 | .9473 | 12 | 19.1 | 20.7 | 110 | −43 / +17 / +8 | −6 / +39 / −9 |
| (CNN-only / Concat, saved models) | .9482 / .9464 | | | | | | |

Projected b − c = validation rows summed over a seed's 5 folds (folds' validation rows can overlap, so a proxy, not a test).

Keep rule (fixed before running: better than control on average and in ≥ 11 of 15, AND projected b − c > 0 against both baselines in every
seed): **no candidate passes.** D1 passes the improvement part (+0.0085, 12 of 15; first version above both baselines on validation) and is ahead
of Concatenation in all three seeds and of CNN-only in two, but is behind CNN-only in seed 42 (−36). The extra generator inputs (37
contributions) make it worse; the two-layer generator and the head shortcut do not beat D1. As the rule says, the study stops here; the
shared-recipe check R1 was not run. Adopting D1 (a fix of a training inconsistency) is a decision for the group and the adviser and would need a
disclosed 4th test read or a fresh holdout. Test folds were not read.
