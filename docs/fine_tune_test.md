# What changed and what was tried (2026-11-22): summary for the thesis group

Written by Franz's Claude Code session. Everything here is in the working tree of local `main` and is **not committed or pushed**.
Plain language on purpose. Numbers are from the files named in each section.

---

## 1. The short version

1. The Step 9 error analysis showed the data lead's labels contradicted themselves for some gambling operators. We fixed this with one written rule (section 2).
2. A few dataset rows were removed or rewritten (section 3). The dataset is now **6,714 messages** (3,723 legitimate, 2,991 smishing; 5,826 with a link, 888 without).
3. Your group mate's pipeline (Steps 3-9) was rerun on the corrected data, and **Step 10 (McNemar) was written and run** for the first time.
4. **Result: the Proposed (FiLM) model clearly beats KNN-RF-ETC, but it does not beat CNN-only or Concatenation.** It is about 0.5 F1 points behind them, and the difference is statistically significant in only 1 of 3 seeds for each.
5. We then looked for the reason and tried several fixes using validation data only. None improved the model. The diagnosis is in section 6, the failed attempts in section 7, and ideas still open in section 8.

---

## 2. Label decisions (115 messages)

File: `data/final/label_decisions.csv` (columns `sms_text, final_label, reason`; git-ignored, so use `git add -f` to track it). Applied by `scripts/build_final_from_master.py --decisions`.

**Rule we adopted:** a gambling promotion is **legitimate** when it links to a real, working gambling site (checked live) **and** has no deceptive bait (no fake payout, fake receipt or "you won"). If we cannot verify the site, it stays smishing. A site saying "PAGCOR licensed" is **not** proof, because scam sites say it too. We did **not** check the official PAGCOR licensee list.

| Domain | Messages | Label now |
|---|---|---|
| casinoplus.com.ph | 56 (+1 whose link was mis-read as `hintayan.play`) | legitimate |
| bingoplus.com | 18 | legitimate |
| arenaplus.ph | 12 | legitimate |
| ptgaming.ph | 12 | legitimate |
| 6bet.com / gbet.com | 2 / 1 | legitimate |
| 789bingo.com plain promos | 5 | legitimate (its 12 "PHP 789.00 Successful Released" fake-payout texts stay smishing) |
| applus.ph | 8 | **smishing** (Franz's decision, was legitimate) |

Other gambling domains (jili169, pin77, filbet, patokbet, bybet, panalobet, unibetnow, kbet and others) were **not judged and stay smishing**. bybet.com and filbet.com are borderline: the group should decide.

---

## 3. Rows removed or rewritten

| What | File | Why |
|---|---|---|
| 1 row removed: the group's own Azure lab URL (`https://lejet-x7k2q.japaneast.cloudapp.azure.com/`) | `data/final/excluded_messages.csv` | not an SMS |
| 1 row removed: `bit.ly.l.` | same file | a broken capture; its link is the bare domain `bit.ly`, which opens Bitly's own homepage, so its page features do not describe a smishing page |
| **16 messages rewritten** | `data/final/text_edits.csv` (`sms_text` = old, `new_text`, `reason`; git-ignored) | see below |

**The 16 rewrites.** The dataset had only 17 messages that were just a link (9 legitimate: Instagram, YouTube, TikTok, Rappler; 8 smishing: bit.ly and similar). Real smishing is not a bare link, and these had all landed in one fold of the cross-validation (they shared one template group), so the models were tested on a message type they had barely seen. Each of the 16 now has a realistic sentence around the **same link**. Labels and page features are unchanged. **The wording is ours, not the original senders'.** Franz reports the adviser agreed to this; please say so in the thesis.

Code changes: `build_final_from_master.py` got `--exclude` and `--text-edits` options; a test for the edits file was added in `tests/test_build_final_from_master.py`.

Message ids (`lj_NNNNNN`) and `dup_group` names are **renumbered on every rebuild**. Do not rely on an old id.

---

## 4. Results on the current dataset (6,714 messages)

Mean ± SD over 3 seeds, 5-fold grouped cross-validation (`experiments/film/lejet_final_dataset/step7/`).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| CNN-only | .951 | .946 | .944 | **.945** |
| Concatenation Fusion | .951 | .931 | .962 | **.946** |
| **Proposed (FiLM)** | .945 | .922 | .958 | **.940** |
| KNN-RF-ETC | .813 | .720 | .951 | **.819** |

F1 over the three dataset versions (so nobody mixes them up):

| Version | CNN-only | Concat | Proposed | KNN-RF-ETC |
|---|---|---|---|---|
| Group mate's first run (6,716 rows, original labels) | .946 | .950 | .946 | .813 |
| After the gambling relabel (6,715 rows) | .946 | .949 | .941 | .817 |
| **Current** (6,714 rows, 16 texts rewritten) | .945 | .946 | .940 | .819 |

**Step 10, McNemar** (`step10/McNemar_Test_Results.xlsx`, script `scripts/build_step10_files.py`; b = Proposed right and baseline wrong, c = the reverse):

| Comparison | Seed 42 | Seed 123 | Seed 2024 |
|---|---|---|---|
| vs CNN-only | p .157 | p .140 | p .001 (CNN-only better) |
| vs Concatenation | p .485 | p .135 | p < .001 (Concatenation better) |
| vs KNN-RF-ETC | p < .001 (Proposed better) | p < .001 (Proposed better) | p < .001 (Proposed better) |

The workbook also has Bonferroni and Holm columns. The adviser still has to say which correction to use (guide item 39). The two significant results against the baselines survive both corrections.

**Component checks** (Step 6, F1): reference .940, text-only CNN .910, FiLM neutralised .773, shuffled RF outputs .865, webpage hidden .937. Every part matters, so the design is not pointless.

**By message condition** (Step 8, Proposed F1 vs best baseline): no URL .696 vs .768, inaccessible webpage .936 vs .941, accessible webpage .951 vs .960. Only 75 of the 888 no-URL messages are smishing, so that row is noisy.

---

## 5. Honest reading of the results

- The Proposed model **is** much better than the KNN-RF-ETC baseline and each component contributes.
- It is **not** better than the two simple baselines (CNN-only, or one network that sees text plus the same 37 link features). Chapter 5 should say so and discuss why (section 6).
- Do not describe the narrowing of the McNemar gap after the rewrite as an improvement of the Proposed model. Its F1 did not rise (.941 to .940); the baselines dipped slightly.

---

## 6. Why does the Proposed model lose? (diagnosis, validation and saved predictions only)

Scripts: `scripts/validation_predictions.py`, `scripts/validation_study_analysis.py`. Outputs in `experiments/film/lejet_final_dataset/validation_study/`. **These files use the previous version's message ids and are stale for the current dataset.** Rerun them if you want current numbers.

| Finding | Evidence |
|---|---|
| It ranks messages as well as the baselines, so it is not a weaker detector | AUC .9878 vs .9896 (Concat) and .9882 (CNN-only) |
| It loses through **false alarms**, not misses | per seed: false alarms 237 vs 190 and 148, misses 123 vs 120 and 172 (previous version) |
| Most lost messages are close calls | 77% had a smishing probability between 0.3 and 0.7 |
| Its cut-off is too low (H1, supported) | best validation cut-off above 0.5 in 13 of 15 runs (mean .577) |
| The Link RF is not the cause (H2, **not** supported) | the RF flags only 0.9% of legitimate page-rank-10 messages, yet the Proposed model false-alarms on 25.8% of them |
| Bare-link messages (H3, supported) | the Proposed model called 80 of 81 legitimate bare-link validation rows smishing; the baselines called about none. This single type was 83 of its 170 extra false alarms on validation. The other 87 came from longer messages |

Probable reason for the bare-link behaviour (not proven): in FiLM the link reaches the verdict **only** through the scale and shift applied to the text features, so when the text is empty there is little to adjust. The baselines feed the link numbers straight into the head. This is a design property of FiLM, not a bug.

---

## 7. What we tried and why it failed (do not repeat without a new reason)

| # | Attempt | Result | Why it did not work |
|---|---|---|---|
| 1 | Fix the gambling labels | Proposed F1 .946 to .941, baselines about equal | the contradiction was real and is fixed, but it was not what separated the models |
| 2 | Choose the cut-off on validation data (B1) | honest split-half estimate: Proposed **-0.003** balanced accuracy, Concat -0.001, CNN-only +0.006, KNN-RF-ETC +0.038 | with 767 validation messages per fold the best cut-off swings between .47 and .66. It helps the baselines, not the Proposed model. (An F1 of .943-.950 at a better cut-off that I quoted earlier used test labels and was a best case, not an expected gain.) |
| 3 | Calibrate the Link RF score (B2) | **dropped before running** | phase A showed the RF is not over-scoring big-brand pages |
| 4 | "Start FiLM at gamma = 1, beta = 0" (B3) | **already how the model is built** (the generator starts at zero) | a mistake in my plan, not a result |
| 5 | C1: generator averages only over real words (`mask_padding=True`) | validation balanced accuracy .9336 vs control .9412 (better in 1 of 15 runs) | worse |
| 6 | C2: generator reads only the link numbers (`use_average=False`) | .9431 vs .9412 (better in 10 of 15; the rule needed 11) | within noise, still below Concatenation (.9541), bare-link false alarms stayed 72 of 81 |
| 7 | Rewrite the 16 bare-link messages | bare-link weakness gone (47 of 48 test predictions right); overall F1 unchanged | fixes the data gap, but the Proposed model is still a little behind on precision |

`scripts/b3_variants_validation.py` reproduced the group mate's recorded validation scores exactly (difference 0.000000), so the retraining loop is trustworthy.

---

## 8. Ideas not yet tried (for future reference)

Rules for all of them: choose on **validation data only**; give the baselines the same effort; log every candidate including the failures; do the final comparison once on a fresh holdout; tell the adviser, because any change after seeing test results has to be disclosed.

| Idea | Why it might help | Cost and risk |
|---|---|---|
| **Fresh holdout set** (new messages nobody has used) | the only way to make a clean claim now that the test folds were read in three separate runs. Aim for 2,000 or more messages, many domains, real (not templated) text, including short and bare-link legitimate messages | needs collecting; freeze and hash it before training |
| **More realistic legitimate messages with links** | `web_legit_lab` (2,466 messages) is 100% legitimate and `web_synthetic` (1,308) is 100% smishing, so models may learn the source style, not the threat | collection; also ask the adviser whether to report results with and without these two sources |
| **Rerun the diagnosis on the current data** | tells us whether the bare-link weakness is really gone and where the remaining false alarms on longer messages come from (big-brand links such as Instagram, YouTube, Spotify, Netflix) | cheap: rerun the two validation scripts |
| **Let the head also see the link numbers** (FiLM plus a direct path) | removes the "nothing to modulate" problem | the model becomes close to Concatenation, which weakens the thesis claim. Adviser decision |
| **Text-less fallback rule**: when a message has almost no words, let the Link RF decide | the text carries no information then | a post-hoc design choice, needs its own validation study and adviser approval |
| **Per-condition cut-offs** (no URL, inaccessible, accessible) | the gaps differ by condition | many more choices on small folds; B1 already showed the noise problem |
| **Tune the modality-dropout rates** (page hidden 0.25, text hidden 0.10) | one of the few untouched knobs | every value tried is another chance to win by luck; keep the list short |
| **Better Link RF** (features, tuning) | the Proposed model depends on it | RF itself did not look wrong in the diagnosis, so lower priority |
| **Average the models over seeds (ensemble)** | usually reduces variance | changes what "the model" is; baselines need it too |
| **Another measure of success** (unseen domains, calibration, behaviour when the page is missing) | FiLM may help there even if accuracy is equal | SOP 2 was removed and "accuracy only" decided, so this is an adviser conversation |

---

## 9. Still open for the group or the adviser

- McNemar correction: none, Holm or Bonferroni (guide item 39).
- Is a second model version after a diagnostic acceptable if reported as such?
- Remaining label doubts, where the lead's labels were left alone: Netflix receipt (smishing), HSBC promo, `contis.ph` cake promo, `taguigcom.cam` "FREE SP1NS 120% extra b0nus" (legitimate), `fbmplay.com` (one legitimate, one smishing); bybet.com and filbet.com; `applus.ph` confirmation; the official PAGCOR list.
- Nobody has done the second-person label check yet (`label_verified_by` is empty).
- `kbet.com` has a domain-extraction problem to fix at the next rebuild.
- Downloads copies of the dataset and Step 2 files were **not** refreshed.

## 10. Test-set reads (be honest about this in Chapter 3)

The test folds were read in **three separate runs**, each a Step 5 plus Step 6 pair (see `experiments/film/lejet_final_dataset/compare/test_reads.log`, six lines): the group mate's original run, the rerun after the gambling relabel, and the rerun after the 16 text rewrites. Step 10 reads no test fold itself; it only uses the Step 5 predictions. All the validation-only work above used validation rows only, and the scripts assert that no test row is written.

## 11. How to reproduce

```
# dataset (needs the lead's list, site CSV and ISDA CSV in Downloads / LeJetData)
.venv\Scripts\python.exe scripts/build_final_from_master.py --master "D:/Downloads/final_combined_dataset (1).csv" --site D:/LeJetData/web_import/site_final.csv --isda D:/Downloads/isda_features_labeled.csv --out data/final/lejet_final_dataset.csv
.venv\Scripts\python.exe -m src.data.split_crowd
.venv\Scripts\python.exe -m src.features.build_crowd_embeddings
.venv\Scripts\python.exe scripts/build_step2_files.py
# training and Steps 3-10 (about 35 minutes of training)
bash logs/final_training/rerun_textfix.sh
```

## 12. File index

| What | Where |
|---|---|
| Label decisions, exclusions, text edits | `data/final/label_decisions.csv`, `excluded_messages.csv`, `text_edits.csv` (all git-ignored) |
| Final dataset | `data/final/lejet_final_dataset.csv` (+ `.summary.json`) |
| Step 10 script and result | `scripts/build_step10_files.py`, `experiments/film/lejet_final_dataset/step10/McNemar_Test_Results.xlsx` |
| Validation-only study | `scripts/validation_predictions.py`, `validation_study_analysis.py`, `b3_variants_validation.py` |
| Plan written before the study (with its results appended) | `docs/validation_only_improvement_plan.md` |
| Rerun script and log | `logs/final_training/rerun_textfix.sh`, `rerun_textfix.log` |
| Group mate's open question and our answer | `docs/ARCHITECTURE.md` section 8.1 |
| The 17 original bare-link messages | `experiments/film/lejet_final_dataset/validation_study/bare_link_messages.csv` (old texts) |
