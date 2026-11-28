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

## Phase D, shared-recipe check R1 (written 2026-11-24 BEFORE the run; validation rows only; `scripts/recipe_r1_validation.py`, `experiments/film/lejet_final_dataset/recipe_r1/`)

**Why.** Every neural model stops after 3 epochs without a better validation balanced accuracy. On ~767 validation messages that score
jumps, so the kept epoch is partly luck (recorded best epochs: Proposed 2-16, Concatenation 3-12, CNN-only 2-12). FiLM's generator starts
at "change nothing" and learns slowly, so the Proposed model may lose the most from early stops.

**Arms (each applied to Proposed, Concatenation and CNN-only alike; KNN-RF-ETC has no epochs):** current = the saved models;
R1 = patience 6 + keep the lowest validation log loss; R1p = patience 6 + keep the best validation balanced accuracy. Nothing else
changes (`fit_model.fit(monitor=...)`, default unchanged). Retrained runs must repeat the recorded per-epoch logs exactly up to where
both stopped (replay check).

**Keep rule (fixed now):** within each arm, projected McNemar b − c (Proposed vs each baseline of the same arm, validation rows summed
over a seed's 5 folds). An arm is kept if its worst b − c over the 6 pairings AND its total b − c are both larger than the current
recipe's. If both arms pass, the larger worst b − c wins. Not part of the rule but reported: each model's own change, best epochs, and
how many pairings reach proxy p < .05. Adopting a kept arm still needs the group and the adviser, and a disclosed test read.

Caveat known in advance: the validation rows both choose the epoch and judge the result, for every model alike; R1p can only match or
beat each model's recorded validation score (it sees every epoch the recorded run saw), R1 can fall below it because it does not watch
balanced accuracy.

**Phase D result (2026-11-24):** replay exact (gap 1.1e-16). Mean validation balanced accuracy, current / R1 / R1p: Proposed .9495 /
.9479 / .9518, CNN-only .9482 / .9483 / .9514, Concatenation .9464 / .9465 / .9500. Worst projected b − c: current −36, R1 −40, R1p −28;
total: current +46, R1 +10, R1p +20. **No arm kept** (R1p improves the worst case but lowers the total). Waiting longer helps every model
about equally, so it does not widen the Proposed model's lead; the guess that FiLM loses most from early stops is not supported (its
kept epoch moved 6.2 → 7.1 on average, the baselines' more). Default recipe unchanged.

## Phase E, equal-budget tuning (written 2026-11-24 BEFORE the runs; validation rows only; `scripts/tune_validation.py`, `experiments/film/lejet_final_dataset/tuning/`)

Franz (2026-11-24): keep searching for the best fine-tune overnight; adopt and test only what passes, otherwise revert and try again.

**Configurations (12 per model kind, current included, so no model gets more tries):**
- shared recipe knobs, all three kinds: lr 2e-3, lr 5e-4, text hidden 0.20, page hidden 0.40, page hidden 0.10, label smoothing 0,
  batch 32, patience 6 (= R1p, imported);
- kind-specific (3 each): Proposed: unbounded gamma (`gamma_mode="free"`, Perez et al. 2018 found bounding gamma hurt), generator learning
  rate ×3, ×10 (the generator starts at zero and learns slowly, D22); CNN-only and Concatenation: lr 3e-3, text hidden 0.05, batch 128.

**Rules (fixed now):** each kind's tuned configuration = highest mean validation balanced accuracy over the 15 cells. The Proposed
model is judged against the TUNED baselines. A Proposed configuration earns a test read only if its projected b − c is positive against
both tuned baselines in all 3 seeds. Combinations of the best knobs may follow in later rounds; the baselines then get the same number
of combination tries. Every round and its result is appended here.

**Phase E round 1 result (2026-11-24):** 12 configurations per kind finished. Best per kind (mean validation balanced accuracy):
CNN-only lr 3e-3 **.9553**, Proposed patience 6 .9518, Concatenation patience 6 .9500 (current: .9482 / .9495 / .9464). Projected b − c
of the best Proposed against the tuned baselines (seeds 42 / 123 / 2024): vs CNN-only −33 / −5 / −9, vs Concatenation −5 / +16 / +16.
**No Proposed configuration passes.** Under equal tuning CNN-only gains the most. Faster learning hurts the Proposed model (lr 2e-3,
generator ×3 and ×10) but helps CNN-only. Label smoothing 0 hurts Proposed and Concatenation. Diagnosis (validation): in seed 42, 24 of the
96 messages the Proposed model loses to CNN-only are casinoplus.com.ph legitimate promos in one validation fold (domain-grouped split).

**Round 2 (written before running):** 4 combination tries per kind. Proposed: patience 6 + unbounded gamma; + text hidden 0.20; patience 6
+ text hidden 0.20; lr 3e-3 with the generator kept at 1e-3 (scale 1/3; tests whether lr 2e-3 hurt through the generator). CNN-only:
lr 3e-3 + patience 6; + label smoothing 0; both; lr 3e-3 + text hidden 0.20. Concatenation: patience 6 + lr 2e-3; + text hidden 0.20;
both; patience 6 + page hidden 0.10. Same rules as round 1.

**Round 2 result (2026-11-24):** 16 configurations per kind. Tuned: CNN-only lr 3e-3 + patience 6 **.9568**, Proposed patience 6 +
unbounded gamma **.9533** (best Proposed so far), Concatenation patience 6 + text hidden 0.20 .9527. Proposed vs tuned CNN-only −33 / −6 /
−6, vs tuned Concatenation −12 / +6 / +10. **No pass.** lr 3e-3 with the generator kept at 1e-3 scored .9438, so the Proposed model's
dislike of faster learning is in the text part, not the generator.

**Round 3 (written before running):** Proposed (all with patience 6 + unbounded gamma): generator reads the link only (`use_average=False`);
averages after ReLU; gamma only (`use_beta=False`); lr 7e-4. CNN-only: lr 4e-3 + patience 6; lr 3e-3 + patience 8; + page hidden 0.10;
+ batch 32. Concatenation: patience 6 + text hidden 0.30; + text hidden 0.20 + page hidden 0.10; + text hidden 0.20 + lr 1.5e-3;
patience 8 + text hidden 0.20. Same rules.

**Round 3 result (2026-11-25):** 20 configurations per kind. Tuned: CNN-only lr 3e-3 + patience 8 **.9578**, Proposed (unchanged
best, patience 6 + unbounded gamma) .9533, Concatenation patience 8 + text hidden 0.20 .9531. The FiLM structure options (link-only
generator, averages after ReLU, gamma only, lr 7e-4) did not beat the best Proposed. Worst projected b − c vs tuned baselines −31 at best.
**No pass.** Note: two Proposed configurations with nearly equal means (.9533 vs .9530) differ by up to 25 in per-seed b − c, so a
per-seed margin of ±25 is training noise.

**Round 4 (written before running), new reason:** under equal tuning the strongest model is CNN-only, the only one that reads the
37 raw link features. So the Proposed generator now ALSO reads those 37 features, prepared exactly as CNN-only prepares them (checked:
difference 0.0), plus its 5 missing flags: link = [P, u, page_missing, 37 prepared, 5 flags], link_dim 45. This is not the failed D1F2,
which read the forest's contributions. Proposed: patience 6; patience 6 + unbounded gamma. CNN-only: lr 3e-3 + patience 8 + page hidden
0.10; lr 2.5e-3 + patience 8. Concatenation: patience 8 + text hidden 0.25; patience 10 + text hidden 0.20. Same rules.

**Round 4 result (2026-11-25):** the generator reading the 37 raw features did NOT help (patience 6: .9513, + unbounded gamma .9511,
vs .9518 / .9533 without them). 22 configurations per kind: tuned CNN-only (lr 3e-3, patience 8, page hidden 0.10) **.9588**, Concatenation
(patience 10, text hidden 0.20) .9535, Proposed (patience 6, unbounded gamma) .9533. **No pass.**

**Round 5 (written before running), items of Franz's D:/Downloads/LeJet_Tuning_Roadmap.pdf that need no new data,** each on top of the kind's
best so far, 7 tries per kind: E9 sequence length 32; M7 128 filters; E3 frozen embedding; E3 weight decay 1e-3; M6 focal loss (gamma 2,
same class weights); E5 without u and Platt-calibrated P (fitted on the train part's out-of-fold P) for Proposed and Concatenation;
CNN-only instead gets sequence length 40 and a 128-unit URL branch. Already covered earlier: E1 (B1), E2, E4, E5 gamma / gamma-only /
use_average, M7 averages / hidden generator layer / link in the head, label smoothing. Not runnable without new data or lab work:
M2-M5, M9 (diagnostic), H1-H8, P1-P8.

**Round 5 result (2026-11-25):** all 7 Proposed roadmap variants scored below its best (.9533): Platt P .9522, without u .9518, sequence 32
.9500, 128 filters .9499, weight decay 1e-3 .9497, focal .9433, frozen embedding .9395; all behind tuned CNN-only in seed 42 by 50 to 70.
Because they already fail against the existing tuned baselines, the baselines' round-5 tries were not run (more baseline tries can only
raise the bar). E1 (threshold) on the tuned models, chosen leaving one seed out: cut-offs stay at about 0.5; Proposed .9533 -> .9523.

**Round 6 (written before running):** E6, the Link Random Forest with class_weight="balanced" on top of each fold's tuned settings (the
rebuild without the change reproduces the saved P and u to 1e-16). Proposed and Concatenation, each on its best configuration.

**Round 6 result (2026-11-25):** balanced forest hurts both kinds that read it (Proposed .9502 vs .9533, Concatenation .9494 vs .9535).
**No pass.** With this, every roadmap item that runs without new data or lab work has been tried.

**Round 7 (written before running), roadmap M9 learning curve (diagnostic, not a candidate):** each kind's tuned configuration trained
on a label-stratified 50% and 75% of every train part (same rows for all kinds; the forest is rebuilt out-of-fold on the kept rows only).
Question: does the Proposed model gain more per extra message than the baselines? If so, more data is the honest lever.

**Round 7 result (2026-11-25), learning curve of the tuned configurations (mean validation balanced accuracy):**

| Train data | Proposed | CNN-only | Concatenation | Proposed minus CNN-only |
|---|---|---|---|---|
| 50% | .9340 | .9490 | .9381 | −.0150 |
| 75% | .9456 | .9561 | .9463 | −.0105 |
| 100% | .9533 | .9588 | .9535 | −.0055 |

CNN-only is flattening (+.0028 from 75% to 100%); the Proposed model still climbs (+.0077) and gains the most per extra message (as
D22 found for the generator). The gap to CNN-only halves with each step. Naive extrapolation (not a measurement): the Proposed model would
need roughly twice the current training data to draw level with CNN-only.

**Phase D/E conclusion (2026-11-25):** 7 rounds, 30 Proposed configurations, 22-24 per baseline, every item of the tuning roadmap that
needs no new data. No configuration makes the Proposed model better than the equally tuned baselines on validation, so **no test read
was made and nothing was adopted** (all new options default to the recorded recipe). Under equal tuning, CNN-only (text + 37 raw link
features) is the strongest model on this data. The honest lever for the Proposed model is more real data (roadmap M6, M9; collection
of real legitimate messages with links), not more tuning: further tries on the same validation rows would start fitting their noise
(per-seed b − c moves by ±25 between near-identical configurations).

**Round 8 (written 2026-11-25 BEFORE running), FiLM-specific settings never run on the real data,** each on top of the tuned Proposed
(unbounded gamma, patience 6): **P8_warm** the embedding and Conv start from the fold's trained plain text CNN (D7b, plan step 5.12,
only ever run on the synthetic pilot); **P8_gwd0** and **P8_gwd1e3** the generator's weight decay 0 and 1e-3 instead of 1e-4 (D13);
**P8_pooled** FiLM after the max-pool (gamma x pooled + beta) instead of on every Conv position before ReLU (still gamma = 1, beta = 0
at the start). Rule: a try passes only if its mean validation balanced accuracy beats the tuned Proposed (.9533) AND its projected
b − c is positive against both tuned baselines in all 3 seeds. Because these are Proposed-only tries, a pass does NOT go straight to a
test read: each baseline first gets the same number of extra tries (4), and the test read happens only if the try still beats the
re-tuned baselines in all 3 seeds. No pass: nothing adopted, no test read.

**Round 8 result (2026-11-25):** all four FiLM-specific tries are below the tuned Proposed (.9533): FiLM after the max-pool .9511,
generator weight decay 1e-3 .9505, generator weight decay 0 .9500, warm start from the plain text CNN .9485 (worst against tuned
CNN-only: −54 / −27 / −55 per seed). **No pass**; no baseline tries were needed, nothing adopted, no test read. With round 8, every FiLM
option the model has, plus warm start, generator weight decay and FiLM placement, has been tried on validation; the best FiLM setting
stays unbounded gamma with patience 6.

## Phase F, multi-width redesign (written 2026-11-25 BEFORE running; validation rows only; Franz: "do this ... Redesign the FiLM part")

**Design.** The text part of ALL three neural models gets several Conv widths side by side instead of one width-3 Conv (as in Kim 2014),
so a gain cannot come from the bigger text part alone. Each width's input is right-padded with (width − smallest) zero vectors (the
padding token's embedding row is 0 and messages are padded at the end, so this equals extra padding), every width then gives the same
number of positions, and the filters stack into one layer. In the Proposed model the generator reads the link numbers plus the average
of every filter of every width and gives a gamma and a beta to every filter of every width (FiLM on each width); the baselines'
max-pool, head and URL branch are unchanged. Not an approved architecture: the group and the adviser decide on adoption.

**Round F1 (architecture, 3 tries per kind, each on the kind's tuned recipe from Phase E):** widths 2,3,4 with 64 filters each (192);
widths 3,4,5 with 64 each (192); widths 2,3,4 with 32 each (96, nearer today's size). **Round F2** (written after F1, before running):
the same number of further tries per kind around each kind's best F1 encoder. **Pass rule (as Phase E):** the Proposed model's best
configuration must have a higher mean validation balanced accuracy than every tuned baseline (best over Phases E and F) AND a positive
projected b − c against both tuned baselines in all 3 seeds. Only then: retrain all three kinds with their tuned configurations through
the normal pipeline and make one logged test read (Step 10 with tuned baselines, not the old ones). Otherwise nothing is adopted.

**Round F1 result (2026-11-25), mean validation balanced accuracy over 15 cells:**

| Encoder | Proposed | CNN-only | Concatenation |
|---|---|---|---|
| one width 3 x 64 (Phase E tuned) | .9533 | .9588 | .9535 |
| widths 2,3,4 x 64 | .9506 | .9640 | .9562 |
| widths 3,4,5 x 64 | .9530 | **.9647** | **.9563** |
| widths 2,3,4 x 32 | **.9543** | .9594 | .9541 |

Multi-width helps CNN-only most (+.0059), Concatenation +.0028, the Proposed model +.0010. Best Proposed (2,3,4 x 32) vs tuned CNN-only
(3,4,5 x 64): projected b − c −32 / −36 / −39; vs tuned Concatenation +4 / −38 / +20. No pass. With more filters the Proposed model's
generator grows fast (2,3,4 x 64: 195 inputs x 384 outputs, about 75,000 weights) while the train part stays about 4,600 messages.

**Round F2 (written after F1, BEFORE running), 3 tries per kind:** Proposed (unbounded gamma, patience 6): **per-width generator**
(each width's gammas and betas come only from the link numbers and that width's own filter averages; a fixed mask on the one generator
layer, still zero at the start; about a third of the weights) on 2,3,4 x 32 and on 3,4,5 x 64; and 2,3,4 x 16. CNN-only and
Concatenation, around their best encoder (3,4,5 x 64): 3,4,5 x 96; 2,3,4,5 x 64; patience +2. Same pass rule. As in round 8, the
baselines' F2 tries are run only if a Proposed F2 try passes against the current tuned baselines (they can only raise the bar).

**Round F2 result (2026-11-25), Proposed tries:** per-width generator on 2,3,4 x 32 .9542 (ties the F1 best .9543; projected b − c vs
tuned CNN-only −42 / −20 / −46); per-width on 3,4,5 x 64 .9515; 2,3,4 x 16 .9492. **No pass**, so the baselines' F2 tries were not run.

**Round F3 (written 2026-11-25 BEFORE running), "FiLM + URL branch":** the Proposed model gets CNN-only's URL branch (the same 44
numbers: 37 prepared features, page_missing, 5 missing flags, has_url -> Dense 64 -> ReLU -> dropout 0.2, joined before Dense 64) while
the generator still reads only P, u, page_missing and the filter averages and FiLMs the text filters. It is then CNN-only plus FiLM by the
forest's verdict, so Proposed vs CNN-only becomes a clean test of what FiLM adds. Caveat for the group: the Proposed model then reads
more than the Concatenation control (which keeps P, u, page_missing only), so Concatenation is no longer input-matched. 3 tries:
**PF3_br** one width 3 x 64 with the Proposed recipe (unbounded gamma, patience 6); **PF3_br_krec** the same with CNN-only's tuned
recipe (lr 3e-3, patience 8, page hidden 0.10, unbounded gamma); **PF3_br_w345x64_krec** CNN-only's best encoder (3,4,5 x 64) and recipe
plus FiLM = tuned CNN-only + FiLM. Same pass rule; baselines get 3 more tries each only if one passes.

**Round F3 result (2026-11-25):** FiLM + URL branch, 3,4,5 x 64 with CNN-only's recipe **.9631** (projected b − c vs tuned CNN-only
−2 / 0 / −14, vs tuned Concatenation +34 / −2 / +45); one width with CNN-only's recipe .9604 (vs tuned CNN-only −24 / −3 / −16); one
width with the Proposed recipe .9554. The Proposed model is now level with CNN-only (FiLM on top of CNN-only adds about +.0016 at one
width and −.0016 at 3,4,5 x 64, i.e. nothing measurable). **No pass** (needs every seed > 0 against both).

**Round F4 (written 2026-11-25 BEFORE running), FiLM refinements on the F3 best (FiLM + URL branch, 3,4,5 x 64, CNN-only's recipe,
unbounded gamma):** **PF4_pw** per-width generator; **PF4_cb** the generator also reads the URL branch's output (64 numbers, before
dropout), so FiLM is driven by the learned summary of the 37 features plus P, u, page_missing; **PF4_noavg** the generator reads the link
numbers only (no filter averages; 1,536 generator weights instead of 74,880). Same pass rule; baselines get 3 more tries each only if one
passes.

**Round F4 result (2026-11-25):** generator also reads the URL branch .9615 (vs tuned CNN-only −21 / −7 / −4); per-width generator
.9614 (−11 / −16 / −12); link numbers only .9609 (−19 / −12 / −11). All below the F3 best. **No pass.**

**Phase F conclusion (2026-11-25):** the best Proposed configuration ever found on validation is **FiLM + URL branch, widths 3,4,5 x 64,
CNN-only's recipe, unbounded gamma: .9631** (the recorded Proposed model was .9495; Phase E tuned .9533). It is level with tuned CNN-only
(.9647; projected b − c −2 / 0 / −14) and ahead of tuned Concatenation (.9563; +34 / −2 / +45), so no configuration passed and **no
test read was made; nothing was adopted.** Why tuning cannot go further: that model is CNN-only's network plus FiLM, and across every
variant FiLM adds between −.002 and +.002 on top of it (about the size of the noise between near-identical runs). Whatever improves
CNN-only (more filters, more widths) improves the Proposed model by the same amount, so the two stay tied by construction; a significant
win in every seed (b − c about +31 to +39 per seed on the test part) is not reachable by tuning on this data. Adopting "FiLM + URL
branch" would change the approved architecture and make Concatenation no longer input-matched (group and adviser decision); if adopted,
Step 10 must retrain all three neural models with their tuned configurations (tuned Proposed vs untuned baselines would not be fair).

## Phase G, research-based FiLM changes (written 2026-11-25 BEFORE running; validation rows only; Franz: "read researches about film ...")

**Reading.** FiLM (Perez et al. 2018): normalization before FiLM matters (CLEVR 97.4 with batch norm vs 93.7 without); a single
FiLM-ed block is the weakest depth (93.5 vs 97.1 with 2 blocks); gamma = 1 + delta and an unrestricted gamma are best (we already use
both). DAFT (Pölsterl et al. 2021): FiLM in the LAST block, batch norm's own affine switched off before it, a bias-free bottleneck
generator (squeeze to 4 units), and both scale and shift needed. HyperFusion (2024) reports concat .638 vs FiLM-like .635 vs DAFT-like
.658, i.e. conditioning gains are small and setting-dependent.

**Inside analysis (validation, before this phase).** Best Proposed (FiLM + URL branch, 3,4,5 x 64) vs tuned CNN-only: they disagree on
about 82-98 of 3,840 rows per seed and split them about 1:1 (b − c −2 / 0 / −14); both are wrong on 78-119 rows per seed (gambling,
banking and e-wallet domains such as 789bingo.com whose labels follow the message wording). The Proposed model loses mostly on links
whose page was not fetched (b − c −23) and wins on messages without a link (+7). For a significant test McNemar (about 1,343 rows per
seed, about 30 disagreements) the Proposed model would have to win about 2 of every 3 disagreements.

**Round G1 (3 tries, each on the F3 best: FiLM + URL branch, 3,4,5 x 64, CNN-only's recipe, unbounded gamma):** **PG1_bn** batch norm
without affine on the Conv output before FiLM (FiLM and DAFT design); **PG1_bn_deep** a second Conv layer (192 -> 192, width 3, same
length) after ReLU, then batch norm and FiLM on that last layer (DAFT: FiLM in the last block); **PG1_bn_daft** batch norm plus DAFT's
bias-free bottleneck generator (inputs -> 4 -> ReLU -> gammas and betas, last layer at zero). Every Proposed cell also stores the FiLM
paper's test-time ablations on validation: FiLM off (gamma = 1, beta = 0) and gamma or beta replaced by its mean. Same pass rule; if one
passes, CNN-only and Concatenation get the same normalization / depth change (with batch norm's affine on, since they have no FiLM) and
3 tries each before any test read.

**Round G1 result (2026-11-25):** batch norm before FiLM **.9508** (vs F3 best .9631; vs tuned CNN-only −23 / −70 / −65). Batch norm
statistics over (messages x positions) are dominated by padding windows here (most of the 48 positions of a short SMS are padding), so
the FiLM paper's image recipe does not carry over. Test-time ablations inside this model (validation, 11,513 rows): FiLM off .9171 (674
verdicts change), gamma replaced by its mean .9467 (230 change), beta replaced by its mean .9510 (139 change, no loss): the network relies
on FiLM, almost entirely on gamma, as in the FiLM paper. PG1_bn_deep and PG1_bn_daft were **cancelled before running** (both build on the
batch norm that just failed); no pass, nothing adopted.

**Round G2 (written 2026-11-25 BEFORE running), the same research ideas without batch norm, on the F3 best:** **PG2_ln** layer norm
across the filters at each position (no affine) before FiLM, the text-model normalization (each position normalized alone, so padding does
not distort real words); **PG2_deep** a second Conv layer (192 -> 192, width 3, same length) after ReLU, FiLM on it (FiLM in the last
of two blocks); **PG2_daft** DAFT's bias-free bottleneck generator (4 units). Same pass rule and baseline follow-up as G1.

**Round G2 result (2026-11-25):** layer norm per position **.9633** (vs tuned CNN-only +12 / −18 / −10); second Conv layer .9627
(+1 / −2 / −22); DAFT bottleneck generator .9615 (+2 / −14 / −20). **No pass.** Test-time ablations on validation: FiLM off costs
.9633 -> .8983 (948 verdicts change) with layer norm, .9627 -> .9337 (502) with two layers, .9615 -> .9516 (232) with the DAFT
bottleneck. So each design routes a different share of the link evidence through FiLM, yet all end within .002 of CNN-only: the accuracy
is capped by the information the models share, not by the route it takes.

**Phase G conclusion (2026-11-25):** the research-based changes (normalization, depth, DAFT generator) leave the Proposed model tied with
CNN-only. Best Proposed configuration on validation: FiLM + URL branch, 3,4,5 x 64, layer norm before FiLM, CNN-only's recipe, .9633.
No test read; nothing adopted.

## Phase H, inside FiLM (written 2026-11-25 BEFORE running; validation rows only; Franz: "look each angle of the film")

Question: the trained best Proposed model collapses when FiLM is switched off (.9633 -> .8983), yet it does not beat CNN-only. Two runs:
**PH_static** = the best configuration (PG2_ln) with the generator's WEIGHTS frozen at zero and only its bias trained, so gamma and
beta are learned per-filter constants that ignore the link and the text (FiLM without conditioning; the same network otherwise). PG2_ln
minus PH_static is what conditioning FiLM on the link adds. **PH_ln_keep** = PG2_ln retrained for seed 42 (5 folds) with the
checkpoints kept, to inspect gamma and beta: their size, how they depend on P and on the text, which messages FiLM's conditioning
changes and whether those changes are right, and how the link evidence is split between FiLM and the URL branch. Diagnostic only:
nothing is adopted from it and no test row is read.

**Phase H result (2026-11-25):** FiLM without conditioning (`PH_static`, generator weights frozen at zero, bias trained) **.9613** vs the
conditioned best .9633 (projected b − c +29 / −3 / −7): the link-dependent part of FiLM adds about .002. Inside the kept models (seed 42,
validation): switching FiLM off costs .9591 -> .8866, but most of that is FiLM's fixed per-filter scaling (mean gamma .29, 34% of filters
flipped, |beta| about 1); replacing gamma and beta by their averages costs only .9591 -> .9518, and those 74 verdict changes are 50 right
vs 24 wrong. The per-message variation of gamma comes mostly from the text averages, not from P and u; where gamma follows P, it turns up
the filters that push toward smishing (correlation +.445). The URL branch carries most of the link evidence (fixing it: .9591 -> .9115).

## Adoption and the test read (2026-11-25, Franz: "do all steps from 2 to 4")

Franz adopted the best configuration per model (Proposed `PG2_ln`, CNN-only `KF1_w345x64`, Concatenation `CF1_w345x64`); the code was
moved into `src/` (`--preset tuned`), all three models were retrained (they repeat the validated runs), and Steps 4-10 were rebuilt with
**one** test read (the 6th; `compare/test_reads.log` 2026-11-25 23:38 / 23:40). Test results in `docs/fine_tune_test.md` section 15b:
accuracy Concatenation .9588, Proposed .9568, CNN-only .9533, KNN-RF-ETC .8131; McNemar uncorrected: Proposed significantly better in 5 of
9 (KNN-RF-ETC x3, CNN-only seeds 42 and 2024), significantly worse in 1 (Concatenation seed 123). No further tuning is done on the test
results.

**Reverted (2026-11-25, Franz):** "its still not enough though revert to the last training without fine tuning". Models and step results
are back to D1 (restored byte for byte; archive of the tuned run in `D:/LeJetData/model_backups/tuned_adoption_20261125_reverted/`).
The 6th test read stays logged and disclosed.

## Phase I, FiLM-only tuning on the original recipe (written 2026-11-25 BEFORE running; validation rows only)

Franz: "make sure it will fine tune only about film or anything that is new in our proposed model". So the shared recipe stays exactly as
in D1 for every model (one Conv width 3 x 64, Adam 1e-3, patience 3, batch 64, label smoothing 0.1, page hidden 0.25, text hidden 0.10),
the baselines are the recorded models (`current_CNN-only` .9482, `current_Concat` .9464), and only the Proposed model's own parts change:
how FiLM is generated, what it reads and how strongly it acts. The Link Random Forest is not changed (Concatenation reads the same forest).
Disclosure: the baselines' own parts (CNN-only's URL branch, Concatenation's head) are not tuned either.

**Round I1 (7 tries, each on D1 + unbounded gamma, `P_gfree` .9510):** **ugate** FiLM's change scaled by (1 − 2u), so an unsure forest
modulates less; **pmgate** two learned strengths, page reached / not reached; **logitP** the generator reads logit(P)/4 instead of P
(spreads the many P near 0 and 1); **linkx** the generator also reads P·(1 − page_missing), u·(1 − page_missing) and 4(P − 0.5)²;
**ginit** the generator starts small and random (std 0.01) instead of at zero, so it learns from the first step; **daft4** DAFT's
bias-free bottleneck generator (4 units); **embfilm** a second FiLM on the word vectors before the Conv (FiLM at two places, as the FiLM
paper does in every block), generated from P, u, page_missing, at zero at the start.
**Rule:** the best try (highest mean validation balanced accuracy) is a candidate for a test read only if its projected b − c is at least
+20 against BOTH recorded baselines in EVERY seed (about what a significant McNemar needs on the 6,714-row test runs). Otherwise round I2
combines the tries that beat `P_gfree`, under the same rule, and the study stops there.

**Round I1 result (2026-11-26), mean validation balanced accuracy; projected b − c per seed (42 / 123 / 2024) vs the RECORDED
CNN-only and Concatenation:**

| Try | Val bal-acc | vs CNN-only | vs Concatenation |
|---|---|---|---|
| `P_gfree` (base, unbounded gamma) | .9510 | −42 / +20 / +41 | −5 / +42 / +24 |
| pmgate | .9507 | −32 / +20 / +32 | +5 / +42 / +15 |
| daft4 | .9500 | −9 / +3 / +9 | +28 / +25 / −8 |
| ginit | .9500 | −33 / +14 / +29 | +4 / +36 / +12 |
| linkx | .9474 | −46 / +10 / +16 | −9 / +32 / −1 |
| logitP | .9437 | −43 / −18 / +1 | −6 / +4 / −16 |
| ugate | .9412 | −50 / −32 / −19 | −13 / −10 / −36 |
| embfilm | .9377 | −54 / −17 / −45 | −17 / +5 / −62 |
| (recorded D1 Proposed) | .9495 | −36 / +5 / +33 | +1 / +27 / +16 |

No try reaches +20 against both baselines in every seed, and none beats `P_gfree` on average, so round I2 has nothing to combine: **the
study stops here, as written. No test read; the models stay D1.** With the shared recipe fixed, FiLM-only changes move the Proposed model
by at most +.0015 (unbounded gamma), while its gap to the recorded CNN-only in seed 42 is about 40 validation rows (about .01).

## Phase J, how the FiLM fusion is TRAINED (written 2026-11-26 BEFORE running; validation rows only; Franz: "research more ... what can
else change or fine tune ... without changing the architecture too much")

**Reading.** Multimodal networks tend to lean on the easier input and under-train the other one (modality imbalance): Wang, Tran and
Feiszli 2020 (CVPR, "What makes training multi-modal classification networks hard?": the best single-input network can beat the joint one;
remedy Gradient Blending, i.e. extra per-input losses); Peng et al. 2022 (OGM-GE); Du et al. 2023 (ICML, "On uni-modal feature learning in
supervised multi-modal learning": joint training saturates and ignores single-input features that are hard to learn; remedy Uni-Modal
Teacher, distilling a single-input model's features into the joint model's encoder); Neverova et al. 2016 (ModDrop: randomly drop a
modality while training). Our own evidence fits: H3 (the Proposed model's text-style shortcut on short messages, 37.6% false alarms on
legitimate messages of 3 words or fewer vs 0.9% for CNN-only) and the learning curve (Round 7: the Proposed model gains the most per extra
message, i.e. it is data-hungry). Every change below acts ONLY during training; the trained network, its inputs and its size are exactly
D1's (no architecture change), and the shared recipe and the baselines stay the recorded ones (the Phase I rule).

**Round J1 (5 tries, each on D1 + unbounded gamma = `P_gfree` .9510):**
- **aux03 / aux10** FiLM-off auxiliary loss (Gradient Blending idea): each training batch also goes through the same network with FiLM
  switched off (gamma = 1, beta = 0, the "no link" path) and that loss is added with weight 0.3 / 1.0, so the text part must work on its own
  and FiLM is pushed to add to it rather than replace it.
- **cdrop20** conditioning dropout (ModDrop idea, on the FiLM condition only): 20% of the training messages get FiLM switched off.
- **pjit10** noise on the forest's P while training (P + N(0, 0.10), clipped to 0..1), so FiLM does not trust P blindly (the RF's
  confident mistakes, e.g. legitimate gambling promos, are the Proposed model's worst losses).
- **umt** Uni-Modal Teacher: the fold's recorded text-only CNN (trained on the same train part; never sees validation or test labels)
  is the teacher; the Proposed model's pooled text features before FiLM are pulled toward the teacher's (relative squared error, weight
  1.0; messages whose text is hidden by text dropout are left out).

**Rule (as Phase I):** the best try (highest mean validation balanced accuracy) is a candidate only if its projected b − c is at least
+20 against BOTH recorded baselines in EVERY seed. Otherwise round J2 combines the tries that beat `P_gfree` (same rule) and the study
stops. A candidate is NOT adopted and NOT test-read automatically: it goes to Franz and the group first (any test read is logged and
disclosed).

**Round J1 result (2026-11-26), mean validation balanced accuracy; projected b − c per seed (42 / 123 / 2024) vs the RECORDED
CNN-only and Concatenation:**

| Try | Val bal-acc | vs CNN-only | vs Concatenation |
|---|---|---|---|
| `P_gfree` (base, unbounded gamma) | .9510 | −42 / +20 / +41 | −5 / +42 / +24 |
| aux10 (FiLM-off loss, weight 1.0) | .9502 | −3 / −9 / +13 | +34 / +13 / −4 |
| pjit10 (noise on P) | .9483 | −29 / −4 / +21 | +8 / +18 / +4 |
| aux03 (FiLM-off loss, weight 0.3) | .9482 | −19 / −1 / −3 | +18 / +21 / −20 |
| umt (text-only CNN as teacher) | .9479 | −21 / −7 / −4 | +16 / +15 / −21 |
| cdrop20 (FiLM off for 20% of messages) | .9476 | −36 / −5 / +15 | +1 / +17 / −2 |

No try beats `P_gfree` on average and none reaches +20 against both baselines in every seed, so round J2 has nothing to combine: **the
study stops here, as written. No test read; the models stay D1.** Observation: aux10 moves seed 42 against CNN-only from −42 to −3 but
loses 29 and 28 in the other two seeds, with the same mean: the per-seed margins move by ±25-40 between configurations of equal accuracy,
i.e. they are training noise, not a property of the configuration. The training-only hook (`training_penalty` in fit_model.fit) is off
for every recorded model.

## Phase K, FiLM at the decision layers (written 2026-11-26 BEFORE running; validation rows only; Franz: "find more ... you can also
experiment yourself, im desperate")

Phase J stopped by its rule; Phase K is a NEW study opened at Franz's request. Disclosure: ~55 Proposed configurations have now been
scored on the same validation rows, so the best validation score is optimistic (selection on noise); this is why any candidate must also
survive a re-run with other random starts (below) and why a fresh holdout remains the clean confirmation.

**Diagnosis (validation, `P_gfree` vs recorded CNN-only, 11,513 rows).** The Proposed model wins on smishing (b − c +70: accessible
smishing +55, inaccessible smishing +20) and loses on legitimate messages (−51: accessible −32, inaccessible −24). The two biggest
losses: casinoplus.com.ph legitimate promos (−38; page not fetched, forest P about 0.3) and netflix.com legitimate payment / password
notices (−21; page_rank 9, forest P about 0.13, yet the Proposed model gives 0.54-0.81). The forest already says "probably legitimate",
but FiLM can only re-weight the text filters, so smishing-looking wording wins; CNN-only, whose link evidence enters the decision layers
directly, gets them right. FiLM papers condition every block, not only the first (Perez et al. 2018; DAFT conditions the last block).

**Round K1 (4 tries, each on `P_gfree`; FiLM stays the only route of the link, generators start at zero so each try starts as `P_gfree`):**
- **outfilm** FiLM shift on the output layer: the two output scores get a shift generated from [P, u, page_missing] (no link: no shift).
- **densefilm** FiLM (scale and shift) on Dense 64 before its ReLU, generated from [P, u, page_missing].
- **allfilm** both, plus the existing FiLM on the Conv (FiLM at every layer).
- **contrib3** the Conv FiLM generator also reads WHY the forest decided: its contributions summed in three groups (link text 20, page
  13, outside services 4), so P = bias + the three sums.
New generator weights get the same weight decay as the existing generator.

**Rule.** The best try is a candidate only if its projected b − c is at least +20 against BOTH recorded baselines in EVERY seed. A
candidate is then retrained with two other random starts (same folds); it passes only if the mean margin over the three starts stays
≥ +20 in every seed. Round K2 (≤ 4 tries) may combine K1 tries that beat `P_gfree` with each other or with aux10; then the study stops.
Nothing is adopted or test-read without Franz and the group.

**Round K1 result (2026-11-26):** FiLM at every layer (`allfilm`) .9500 (vs CNN-only −32 / +18 / +30, vs Concatenation +5 / +40 / +13);
output-layer FiLM .9489 (−45 / +11 / +26); Dense-layer FiLM .9481 (−35 / +14 / +12); generator reads the forest's grouped reasons
(`contrib3`) .9451 (−44 / +9 / −9). None beats `P_gfree` (.9510), so round K2 has nothing to combine: **the study stops, as written. No test
read; the models stay D1.**

**Why the per-seed margins cannot be tuned (evidence collected in Phases I-K).** The validation margins and the test margins of the SAME
D1 models rank the seeds in opposite order: vs CNN-only, validation −36 / +5 / +33 but test +37 / +12 / +3 (seeds 42 / 123 / 2024); vs
Concatenation, validation +1 / +27 / +16 but test +52 / +18 / −20. Folds are grouped by domain, so each large domain (casinoplus.com.ph 60
validation rows per seed, netflix.com 132, moneymax.ph 140) falls into one fold as a block and moves that seed's margin by 20-40 messages in
whichever direction; configurations with equal mean accuracy differ by ±25-40 per seed (aux10). On the test runs the Proposed model's
average lead over CNN-only is about +17 disagreements per seed, about half of what one significant seed needs and far from what all
three need.

## Phase L, averaging several training runs, the same for every model (written 2026-11-26 BEFORE running; validation rows only; Franz: "find more more ... fine tune more more")

**Why.** Phases I-K showed that configurations with equal mean accuracy differ by ±25-40 per seed: a large part of each per-seed margin is
the random start of one training run (initial weights, dropout draws, batch order), not the configuration. A quick look at saved
validation predictions (no training): averaging the probabilities of `P_gfree` and the recorded CNN-only gives .9593 balanced accuracy vs
.9514 / .9497 alone, so single runs make many errors that another run does not make. The standard remedy is to average several runs
(an ensemble of random starts). Done for EVERY neural model alike, it is not tuning one model; it lowers the training noise of all of them.
Whether it widens the Proposed model's lead is unknown: it does if the FiLM model's single runs are noisier than the baselines' (the
learning curve, Round 7, and the per-seed swings suggest so).

**Round L1.** Every model gets two more random starts of its RECORDED recipe (start 0 = the recorded/saved run; starts 1 and 2 change only
the seed of the initial weights, dropout and batch order to seed + 10007 x start; folds, data, vocabulary and recipe unchanged):
Proposed D1 (`PL_d1_s1/s2`, start 0 = `current_Proposed`), Proposed `P_gfree` (`PL_gfree_s1/s2`, start 0 = `P_gfree`), CNN-only
(`KL_s1/s2`), Concatenation (`CL_s1/s2`). Ensemble = the mean probability of the 3 starts, cut at 0.5. Also reported: the per-seed margins
of each single start (how much a margin moves when only the random start changes).

**Rule (fixed now).** The Proposed ensemble (D1 or `P_gfree`, whichever has the higher mean validation balanced accuracy) is a candidate
only if its projected b − c is at least +20 against BOTH baseline ensembles in EVERY seed. If it is not a candidate but its worst seed is
≥ 0 against both, round L2 adds starts 3 and 4 to every model (5-run ensembles) under the same rule; otherwise the study stops. A
candidate is not adopted or test-read automatically: averaging runs changes the protocol for all models and needs Franz, the group and the
adviser; if adopted, Steps 4-10 are rebuilt with every neural model as its own N-run ensemble, and the test read is logged and disclosed.

**Round L1 result (2026-11-26), 120 new cells (2 extra starts x 4 configurations x 15).** (The first 27 cells' `.json` files carry a wrong
`seconds` value, a variable-name bug fixed during the run; their predictions are unaffected.)

| Model | Start 0 / 1 / 2 (single runs) | Average of the 3 runs |
|---|---|---|
| Proposed `P_gfree` | .9510 / .9516 / .9495 | **.9546** |
| Proposed D1 | .9495 / .9505 / .9495 | .9526 |
| CNN-only | .9482 / .9517 / .9497 | **.9546** |
| Concatenation | .9464 / .9454 / .9493 | .9485 |

Projected b − c per seed (42 / 123 / 2024), averaged runs vs averaged runs: `P_gfree` vs CNN-only −16 / +24 / −15, vs Concatenation
+15 / +34 / +31; D1 vs CNN-only −26 / +17 / −20, vs Concatenation +5 / +27 / +26. Single runs, Proposed start i vs baseline start j (9
pairings): D1 vs CNN-only seed 42 ranges −62..+11, seed 123 −12..+53, seed 2024 −32..+33 (means −22 / +15 / +2).

**Rule outcome:** the best Proposed ensemble (`P_gfree`) is behind CNN-only in two seeds, so it is not a candidate and round L2 is not run.
**The study stops; no test read; the models stay D1.** What it shows: (1) changing only the random start moves a per-seed margin by up to
~70 messages, so single-run per-seed margins are mostly chance; (2) once that chance is averaged out, the Proposed model and CNN-only are
tied (.9546 each) and the Proposed model is ahead of the input-matched Concatenation in every seed (+15 / +34 / +31). Averaging runs helps
every model about equally (+.003 to +.005), so it does not create a lead over CNN-only.
