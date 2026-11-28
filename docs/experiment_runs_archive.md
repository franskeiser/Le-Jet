# Experiment runs removed from the branch (archive record)

**Date:** 2026-11-28 · **Branch:** `fine-tune/v5`

These run outputs, logs and run scripts were produced while tuning the models. They are no longer needed in the repository, so they were **moved out of the repo** (not pushed). This file records what each one was, so the work stays documented.

**Where they are now:** `D:/LeJetData/branch_cleanup_2026-11-28/`, with the same folder structure as in the repo. To restore one, copy it back to the same path.

**Nothing here changes the reported results.** The reported models and their steps (Steps 4–10) are unaffected; the removed files were exploratory runs whose conclusions are written in `docs/validation_only_improvement_plan.md` and `docs/fine_tune_test.md`.

## 1. What was removed

| Path | Size | Phase | What it was |
|---|---|---|---|
| `experiments/film/lejet_final_dataset/tuning/` | 52.5 MB | Phases E–L | Results of the validation-only tuning runs: one folder per run (per-cell validation predictions and training logs), plus `leaderboard.csv` and `report.txt`. See Appendix A for every run. |
| `experiments/film/lejet_final_dataset/recipe_r1/` | 4.3 MB | Phase D | Results of the R1 recipe check: `cells.csv`, `mcnemar_projection.csv`, `summary.txt`, `val_predictions.csv` (Appendix B). |
| `scripts/tune_validation.py` | 53 KB | Phases E–L | The run harness that defined and trained every tuning run (`VARIANTS`) and wrote the `tuning/` folder and its leaderboard. |
| `scripts/recipe_r1_validation.py` | 15 KB | Phase D | The run harness for the R1 recipe check (three training arms for all three neural models). |
| `logs/tune_validation.log` | 151 KB | Phases E–L | Console log of the tuning runs. |
| `logs/recipe_r1_validation.log` | 12 KB | Phase D | Console log of the R1 recipe check (stopped after 14 of 15 cells when the laptop ran low on memory). |
| `logs/recipe_r1_validation_resume.log` | 15 KB | Phase D | Console log of the resumed R1 run (the last cell). |
| `logs/phaseF_w1.chunk` | 580 B | Phase F | Worker 1 job list of the multi-width / URL-branch redesign runs. |
| `logs/phaseF_w2.chunk` | 646 B | Phase F | Worker 2 job list. |
| `logs/phaseF_w3.chunk` | 514 B | Phase F | Worker 3 job list. |
| `logs/final_training/rerun_tuned.sh` | 3 KB | Tuned adoption | Script that retrained all 45 models with the tuned presets (`--preset tuned`). |
| `logs/final_training/rerun_tuned_train.log` | 5 KB | Tuned adoption | Log of that retraining. |
| `logs/final_training/tuned_film.chunk` | 279 B | Tuned adoption | Job list for the Proposed model (tuned preset). |
| `logs/final_training/tuned_cnn_raw.chunk` | 49 B | Tuned adoption | Job list for CNN-only (tuned preset). |
| `logs/final_training/tuned_concat.chunk` | 356 B | Tuned adoption | Job list for Concatenation Fusion (tuned preset). |
| **Total** | **57.0 MB** | | |

## 2. What each phase did

All phases below were judged on **validation data** unless stated; a run was adopted only if it passed a keep rule written before it ran.

| Phase | Date | What was tried | Outcome |
|---|---|---|---|
| D · R1 recipe check | 2026-11-24 | Three training arms for all three neural models: current recipe; **R1** = patience 6 + log-loss early stopping; **R1p** = patience 6 + balanced-accuracy early stopping. | Waiting longer helped every model about equally (baselines slightly more); log-loss stopping helped none. Keep rule not met; nothing adopted. |
| E · equal-budget tuning | 2026-11-24/25 | Learning rate, patience, batch size, label smoothing, page/text modality dropout, Random Forest class weight, and learning curves (50% / 75% of the training data) for CNN-only, Concatenation Fusion and the Proposed model alike. | Best tuned Proposed 0.9533 vs tuned CNN-only 0.9588 vs Concatenation 0.9535 (validation balanced accuracy). Nothing passed. The learning curve showed the Proposed model gains the most from more data. |
| Round 8 · FiLM-only tries | 2026-11-25 | Warm start from the plain CNN, generator weight decay (0 / 0.001), FiLM after max-pooling. | All below the tuned Proposed model; nothing adopted. |
| F · redesign | 2026-11-25 | Several Conv widths side by side (3, 4, 5), a generator per width, and FiLM plus CNN-only's URL branch (`BranchFiLM`). | FiLM + URL branch (F3) reached 0.9631, tied with tuned CNN-only. Not adopted in this phase; the URL-branch idea was carried into later work. |
| G · research-based FiLM | 2026-11-25 | Batch norm before FiLM, layer norm per position, two-layer generator, DAFT-style bottleneck. | Layer norm 0.9633 (best), batch norm 0.9508; none beat tuned CNN-only on every seed. |
| Tuned adoption | 2026-11-25 | The tuned presets were put into `src` (`--preset tuned`) and all 45 models retrained (`rerun_tuned.sh`). | Test check: 5 of 9 McNemar tests significant. **Reverted** the same day to the earlier models; the presets stay in the code as an opt-in option (default off). |
| I · FiLM-only changes | 2026-11-26 | u-gate, page gates, logit P, link interactions, random generator start, DAFT-4, word-vector FiLM. | Best single change: unbounded FiLM scale (γ = 1 + Δγ), +0.0015. Nothing else passed. |
| J · training tricks | 2026-11-26 | FiLM-off auxiliary loss (0.3 / 1.0), conditioning dropout 20%, noise on P, Uni-Modal Teacher. | None better than the unbounded-scale model. |
| K · error analysis | 2026-11-26 | FiLM at the output layer, the Dense layer, every layer; generator reading grouped RF contributions. | None better. Error analysis: per-seed margins are driven by large domains landing in one fold. |
| L · run averaging | 2026-11-26 | Three random starts for every model (`PL_*`, `KL_*`, `CL_*` runs), predictions averaged. | Averaged, the Proposed model ties CNN-only and beats Concatenation Fusion consistently; led to averaging several runs per model. |

**Later work (not in this repository).** The phases after L, including the final Proposed model and the 9-run results with every neural model trained at patience 6, ran in separate working copies outside the repo (`D:/LeJetTestRun_6714_pf3br` and `D:/LeJetTestRun_7569_tune`). Their code and results are not part of this branch.

## 3. What was kept, and why

| Path | Why it stays |
|---|---|
| `src/features/model/conv_blocks.py` | Imported by `film_cnn.py`, `raw_cnn.py`, `concat_cnn.py`, `concat_control.py` and `raw_cnn_control.py` (multi-width Conv blocks of the opt-in tuned preset). Removing it breaks those imports. |
| `src/unified/cell_runs.py` | Imported by `film.py`, `concat_control.py` and `raw_cnn_control.py` (resumable per-cell training). |
| `tests/test_tuned_models.py` | Tests the code above. |
| `experiments/film/lejet_final_dataset/compare/test_reads.log` | The record of every time the test folds were read. Keep it: it is the evidence for the thesis's disclosure of test-fold use. |
| Modified files in `src/`, `scripts/`, `tests/`, `docs/` | Code options added during tuning (all default to the recorded recipe) and the documentation of the phases above. |

Comments in `src/unified/fit_model.py` and `src/features/model/conv_blocks.py` still mention `scripts/tune_validation.py` and `scripts/recipe_r1_validation.py`; those scripts are in the archive folder above.

## Appendix A. Every tuning run (`tuning/` and `scripts/tune_validation.py`)

Kind = which model was trained. Settings = what the run changed from that model's recorded recipe (`build` = model options, `fit` = training options; empty = the recorded recipe). Validation balanced accuracy is the mean over the 15 seed × fold cells from `leaderboard.csv`; runs without a value were averaged-run starts or were scored in a later report.

| Run | Kind | Settings | Val. balanced accuracy | Mean best epoch | Result folder |
|---|---|---|---|---|---|
| `K5_br128` | CNN-only | build: branch_units=128; fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | — |
| `K5_focal` | CNN-only | fit: lr=0.003, patience=8, p_page_hidden=0.1, loss=focal | — | — | — |
| `K5_frozen` | CNN-only | build: freeze_embedding=True; fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | — |
| `K5_nf128` | CNN-only | build: num_filters=128; fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | — |
| `K5_seq32` | CNN-only | build: _seq_len=32; fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | — |
| `K5_seq40` | CNN-only | build: _seq_len=40; fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | — |
| `K5_wd1e3` | CNN-only | fit: lr=0.003, patience=8, p_page_hidden=0.1, weight_decay=0.001 | — | — | — |
| `K7_frac50` | CNN-only | build: _frac=0.5; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9490 | 6.7 | yes |
| `K7_frac75` | CNN-only | build: _frac=0.75; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9561 | 7.1 | yes |
| `KF1_w234x32` | CNN-only | build: _widths=((2, 3, 4), 32); fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9594 | 7.0 | yes |
| `KF1_w234x64` | CNN-only | build: _widths=((2, 3, 4), 64); fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9640 | 6.5 | yes |
| `KF1_w345x64` | CNN-only | build: _widths=((3, 4, 5), 64); fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9647 | 8.6 | yes |
| `KF2_w2345x64` | CNN-only | build: _widths=((2, 3, 4, 5), 64); fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | — |
| `KF2_w345x64_pat+2` | CNN-only | build: _widths=((3, 4, 5), 64); fit: lr=0.003, patience=10, p_page_hidden=0.1 | — | — | — |
| `KF2_w345x96` | CNN-only | build: _widths=((3, 4, 5), 96); fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | — |
| `KL_s1` | CNN-only | build: _start=1 | — | — | yes |
| `KL_s2` | CNN-only | build: _start=2 | — | — | yes |
| `K_bs128` | CNN-only | fit: batch_size=128 | 0.9459 | 5.1 | yes |
| `K_bs32` | CNN-only | fit: batch_size=32 | 0.9496 | 3.9 | yes |
| `K_lr25e3_pat8` | CNN-only | fit: lr=0.0025, patience=8 | 0.9536 | 7.3 | yes |
| `K_lr2e3` | CNN-only | fit: lr=0.002 | 0.9517 | 3.2 | yes |
| `K_lr3e3` | CNN-only | fit: lr=0.003 | 0.9553 | 3.4 | yes |
| `K_lr3e3_ls0` | CNN-only | fit: lr=0.003, label_smoothing=0.0 | 0.9465 | 1.9 | yes |
| `K_lr3e3_pat6` | CNN-only | fit: lr=0.003, patience=6 | 0.9568 | 5.2 | yes |
| `K_lr3e3_pat6_bs32` | CNN-only | fit: lr=0.003, patience=6, batch_size=32 | 0.9503 | 5.5 | yes |
| `K_lr3e3_pat6_ls0` | CNN-only | fit: lr=0.003, patience=6, label_smoothing=0.0 | 0.9498 | 4.1 | yes |
| `K_lr3e3_pat6_pg10` | CNN-only | fit: lr=0.003, patience=6, p_page_hidden=0.1 | 0.9572 | 5.4 | yes |
| `K_lr3e3_pat8` | CNN-only | fit: lr=0.003, patience=8 | 0.9578 | 6.2 | yes |
| `K_lr3e3_pat8_pg10` | CNN-only | fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9588 | 6.5 | yes |
| `K_lr3e3_txt20` | CNN-only | fit: lr=0.003, p_text_hidden=0.2 | 0.9509 | 3.7 | yes |
| `K_lr4e3_pat6` | CNN-only | fit: lr=0.004, patience=6 | 0.9532 | 4.7 | yes |
| `K_lr5e4` | CNN-only | fit: lr=0.0005 | 0.9461 | 6.3 | yes |
| `K_ls0` | CNN-only | fit: label_smoothing=0.0 | 0.9506 | 5.3 | yes |
| `K_pat6` | CNN-only | fit: patience=6 | 0.9514 | 7.3 | yes |
| `K_pg10` | CNN-only | fit: p_page_hidden=0.1 | 0.9480 | 4.3 | yes |
| `K_pg40` | CNN-only | fit: p_page_hidden=0.4 | 0.9475 | 3.5 | yes |
| `K_txt05` | CNN-only | fit: p_text_hidden=0.05 | 0.9491 | 4.3 | yes |
| `K_txt20` | CNN-only | fit: p_text_hidden=0.2 | 0.9498 | 4.7 | yes |
| `current_CNN-only` | CNN-only | recorded recipe | 0.9482 | 5.0 | yes |
| `C5_focal` | Concat | fit: patience=10, p_text_hidden=0.2, loss=focal | — | — | — |
| `C5_frozen` | Concat | build: freeze_embedding=True; fit: patience=10, p_text_hidden=0.2 | — | — | — |
| `C5_nf128` | Concat | build: num_filters=128; fit: patience=10, p_text_hidden=0.2 | — | — | — |
| `C5_nou` | Concat | build: _no_u=True; fit: patience=10, p_text_hidden=0.2 | — | — | — |
| `C5_platt` | Concat | build: _platt=True; fit: patience=10, p_text_hidden=0.2 | — | — | — |
| `C5_seq32` | Concat | build: _seq_len=32; fit: patience=10, p_text_hidden=0.2 | — | — | — |
| `C5_wd1e3` | Concat | fit: patience=10, p_text_hidden=0.2, weight_decay=0.001 | — | — | — |
| `C6_rfbal` | Concat | build: _rf=('rfbal', {'class_weight': 'balanced'}); fit: patience=10, p_text_hidden=0.2 | 0.9494 | 8.1 | yes |
| `C7_frac50` | Concat | build: _frac=0.5; fit: patience=10, p_text_hidden=0.2 | 0.9381 | 14.9 | yes |
| `C7_frac75` | Concat | build: _frac=0.75; fit: patience=10, p_text_hidden=0.2 | 0.9463 | 12.0 | yes |
| `CF1_w234x32` | Concat | build: _widths=((2, 3, 4), 32); fit: patience=10, p_text_hidden=0.2 | 0.9541 | 12.7 | yes |
| `CF1_w234x64` | Concat | build: _widths=((2, 3, 4), 64); fit: patience=10, p_text_hidden=0.2 | 0.9562 | 14.2 | yes |
| `CF1_w345x64` | Concat | build: _widths=((3, 4, 5), 64); fit: patience=10, p_text_hidden=0.2 | 0.9563 | 10.2 | yes |
| `CF2_w2345x64` | Concat | build: _widths=((2, 3, 4, 5), 64); fit: patience=10, p_text_hidden=0.2 | — | — | — |
| `CF2_w345x64_pat+2` | Concat | build: _widths=((3, 4, 5), 64); fit: patience=12, p_text_hidden=0.2 | — | — | — |
| `CF2_w345x96` | Concat | build: _widths=((3, 4, 5), 96); fit: patience=10, p_text_hidden=0.2 | — | — | — |
| `CL_s1` | Concat | build: _start=1 | — | — | yes |
| `CL_s2` | Concat | build: _start=2 | — | — | yes |
| `C_bs128` | Concat | fit: batch_size=128 | 0.9444 | 8.9 | yes |
| `C_bs32` | Concat | fit: batch_size=32 | 0.9471 | 5.8 | yes |
| `C_lr2e3` | Concat | fit: lr=0.002 | 0.9498 | 4.9 | yes |
| `C_lr3e3` | Concat | fit: lr=0.003 | 0.9458 | 3.7 | yes |
| `C_lr5e4` | Concat | fit: lr=0.0005 | 0.9450 | 9.0 | yes |
| `C_ls0` | Concat | fit: label_smoothing=0.0 | 0.9309 | 3.7 | yes |
| `C_pat10_txt20` | Concat | fit: patience=10, p_text_hidden=0.2 | 0.9535 | 9.4 | yes |
| `C_pat6` | Concat | fit: patience=6 | 0.9500 | 8.7 | yes |
| `C_pat6_lr2e3` | Concat | fit: patience=6, lr=0.002 | 0.9505 | 6.4 | yes |
| `C_pat6_lr2e3_txt20` | Concat | fit: patience=6, lr=0.002, p_text_hidden=0.2 | 0.9489 | 5.2 | yes |
| `C_pat6_pg10` | Concat | fit: patience=6, p_page_hidden=0.1 | 0.9501 | 9.1 | yes |
| `C_pat6_txt20` | Concat | fit: patience=6, p_text_hidden=0.2 | 0.9527 | 8.2 | yes |
| `C_pat6_txt20_lr15e3` | Concat | fit: patience=6, p_text_hidden=0.2, lr=0.0015 | 0.9523 | 5.8 | yes |
| `C_pat6_txt20_pg10` | Concat | fit: patience=6, p_text_hidden=0.2, p_page_hidden=0.1 | 0.9508 | 8.4 | yes |
| `C_pat6_txt30` | Concat | fit: patience=6, p_text_hidden=0.3 | 0.9508 | 6.9 | yes |
| `C_pat8_txt20` | Concat | fit: patience=8, p_text_hidden=0.2 | 0.9531 | 8.7 | yes |
| `C_pat8_txt25` | Concat | fit: patience=8, p_text_hidden=0.25 | 0.9517 | 9.5 | yes |
| `C_pg10` | Concat | fit: p_page_hidden=0.1 | 0.9487 | 7.6 | yes |
| `C_pg40` | Concat | fit: p_page_hidden=0.4 | 0.9446 | 6.2 | yes |
| `C_txt05` | Concat | fit: p_text_hidden=0.05 | 0.9411 | 4.6 | yes |
| `C_txt20` | Concat | fit: p_text_hidden=0.2 | 0.9497 | 6.3 | yes |
| `current_Concat` | Concat | recorded recipe | 0.9464 | 5.9 | yes |
| `P5_focal` | Proposed | build: gamma_mode=free; fit: patience=6, loss=focal | 0.9433 | 5.9 | yes |
| `P5_frozen` | Proposed | build: gamma_mode=free, freeze_embedding=True; fit: patience=6 | 0.9395 | 7.9 | yes |
| `P5_nf128` | Proposed | build: gamma_mode=free, num_filters=128; fit: patience=6 | 0.9499 | 5.1 | yes |
| `P5_nou` | Proposed | build: gamma_mode=free, _no_u=True; fit: patience=6 | 0.9518 | 6.0 | yes |
| `P5_platt` | Proposed | build: gamma_mode=free, _platt=True; fit: patience=6 | 0.9522 | 5.9 | yes |
| `P5_seq32` | Proposed | build: gamma_mode=free, _seq_len=32; fit: patience=6 | 0.9500 | 5.6 | yes |
| `P5_wd1e3` | Proposed | build: gamma_mode=free; fit: patience=6, weight_decay=0.001 | 0.9497 | 6.3 | yes |
| `P6_rfbal` | Proposed | build: gamma_mode=free, _rf=('rfbal', {'class_weight': 'balanced'}); fit: patience=6 | 0.9502 | 6.7 | yes |
| `P7_frac50` | Proposed | build: gamma_mode=free, _frac=0.5; fit: patience=6 | 0.9340 | 9.9 | yes |
| `P7_frac75` | Proposed | build: gamma_mode=free, _frac=0.75; fit: patience=6 | 0.9456 | 7.5 | yes |
| `P8_gwd0` | Proposed | build: gamma_mode=free; fit: patience=6, generator_weight_decay=0.0 | 0.9500 | 7.5 | yes |
| `P8_gwd1e3` | Proposed | build: gamma_mode=free; fit: patience=6, generator_weight_decay=0.001 | 0.9505 | 6.1 | yes |
| `P8_pooled` | Proposed | build: gamma_mode=free, _pooled=True; fit: patience=6 | 0.9511 | 6.4 | yes |
| `P8_warm` | Proposed | build: gamma_mode=free, _warm=True; fit: patience=6 | 0.9485 | 9.1 | yes |
| `PF1_w234x32` | Proposed | build: gamma_mode=free, _widths=((2, 3, 4), 32); fit: patience=6 | 0.9543 | 6.3 | yes |
| `PF1_w234x64` | Proposed | build: gamma_mode=free, _widths=((2, 3, 4), 64); fit: patience=6 | 0.9506 | 5.5 | yes |
| `PF1_w345x64` | Proposed | build: gamma_mode=free, _widths=((3, 4, 5), 64); fit: patience=6 | 0.9530 | 6.7 | yes |
| `PF2_pw_w234x32` | Proposed | build: gamma_mode=free, _widths=((2, 3, 4), 32), _perwidth=True; fit: patience=6 | 0.9542 | 7.0 | yes |
| `PF2_pw_w345x64` | Proposed | build: gamma_mode=free, _widths=((3, 4, 5), 64), _perwidth=True; fit: patience=6 | 0.9515 | 4.9 | yes |
| `PF2_w234x16` | Proposed | build: gamma_mode=free, _widths=((2, 3, 4), 16); fit: patience=6 | 0.9492 | 4.8 | yes |
| `PF3_br` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True; fit: patience=6 | 0.9554 | 6.0 | yes |
| `PF3_br_krec` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9604 | 6.0 | yes |
| `PF3_br_w345x64_krec` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64); fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9631 | 8.9 | yes |
| `PF4_cb` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _cond_branch=True, link_dim=67; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9615 | 5.9 | yes |
| `PF4_noavg` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), use_average=False; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9609 | 7.7 | yes |
| `PF4_pw` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _perwidth=True; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9614 | 5.4 | yes |
| `PG1_bn` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _norm=True; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9508 | 4.9 | yes |
| `PG1_bn_daft` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _norm=True, _daft=4; fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | — |
| `PG1_bn_deep` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _norm=True, _deep=True; fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | — |
| `PG2_daft` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _daft=4; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9615 | 5.3 | yes |
| `PG2_deep` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _deep=True; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9627 | 8.5 | yes |
| `PG2_ln` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _ln=True; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9633 | 7.7 | yes |
| `PH_ln_keep` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _ln=True, _keep=True; fit: lr=0.003, patience=8, p_page_hidden=0.1 | — | — | yes |
| `PH_static` | Proposed | build: gamma_mode=free, _raw_link=True, _branch=True, _widths=((3, 4, 5), 64), _ln=True, _static_film=True; fit: lr=0.003, patience=8, p_page_hidden=0.1 | 0.9613 | 7.2 | yes |
| `PI1_daft4` | Proposed | build: gamma_mode=free, _daft=4 | 0.9500 | 4.7 | yes |
| `PI1_embfilm` | Proposed | build: gamma_mode=free, _embfilm=True | 0.9377 | 3.0 | yes |
| `PI1_ginit` | Proposed | build: gamma_mode=free, _ginit=0.01 | 0.9500 | 5.5 | yes |
| `PI1_linkx` | Proposed | build: gamma_mode=free, _link_tf=x | 0.9474 | 5.4 | yes |
| `PI1_logitP` | Proposed | build: gamma_mode=free, _link_tf=logit | 0.9437 | 3.7 | yes |
| `PI1_pmgate` | Proposed | build: gamma_mode=free, _gate=pm | 0.9507 | 5.2 | yes |
| `PI1_ugate` | Proposed | build: gamma_mode=free, _gate=u | 0.9412 | 5.2 | yes |
| `PJ1_aux03` | Proposed | build: gamma_mode=free, _aux=0.3 | 0.9482 | 5.2 | yes |
| `PJ1_aux10` | Proposed | build: gamma_mode=free, _aux=1.0 | 0.9502 | 4.4 | yes |
| `PJ1_cdrop20` | Proposed | build: gamma_mode=free, _cdrop=0.2 | 0.9476 | 5.8 | yes |
| `PJ1_pjit10` | Proposed | build: gamma_mode=free, _pjit=0.1 | 0.9483 | 5.1 | yes |
| `PJ1_umt` | Proposed | build: gamma_mode=free, _umt=1.0 | 0.9479 | 6.5 | yes |
| `PK1_allfilm` | Proposed | build: gamma_mode=free, _outfilm=True, _densefilm=True | 0.9500 | 4.7 | yes |
| `PK1_contrib3` | Proposed | build: gamma_mode=free, _contrib3=True | 0.9451 | 3.6 | yes |
| `PK1_densefilm` | Proposed | build: gamma_mode=free, _densefilm=True | 0.9481 | 5.3 | yes |
| `PK1_outfilm` | Proposed | build: gamma_mode=free, _outfilm=True | 0.9489 | 4.6 | yes |
| `PL_d1_s1` | Proposed | build: _start=1 | — | — | yes |
| `PL_d1_s2` | Proposed | build: _start=2 | — | — | yes |
| `PL_gfree_s1` | Proposed | build: gamma_mode=free, _start=1 | — | — | yes |
| `PL_gfree_s2` | Proposed | build: gamma_mode=free, _start=2 | — | — | yes |
| `P_bs32` | Proposed | fit: batch_size=32 | 0.9469 | 3.3 | yes |
| `P_gfree` | Proposed | build: gamma_mode=free | 0.9510 | 6.5 | yes |
| `P_glr10` | Proposed | fit: generator_lr_scale=10.0 | 0.9433 | 3.5 | yes |
| `P_glr3` | Proposed | fit: generator_lr_scale=3.0 | 0.9464 | 3.1 | yes |
| `P_lr2e3` | Proposed | fit: lr=0.002 | 0.9465 | 3.5 | yes |
| `P_lr3e3_g033` | Proposed | fit: lr=0.003, generator_lr_scale=0.3333333333333333 | 0.9438 | 3.9 | yes |
| `P_lr5e4` | Proposed | fit: lr=0.0005 | 0.9490 | 6.3 | yes |
| `P_ls0` | Proposed | fit: label_smoothing=0.0 | 0.9372 | 4.5 | yes |
| `P_pat6` | Proposed | fit: patience=6 | 0.9518 | 7.1 | yes |
| `P_pat6_gfree` | Proposed | build: gamma_mode=free; fit: patience=6 | 0.9533 | 7.7 | yes |
| `P_pat6_gfree_lr7e4` | Proposed | build: gamma_mode=free; fit: patience=6, lr=0.0007 | 0.9530 | 6.5 | yes |
| `P_pat6_gfree_noavg` | Proposed | build: gamma_mode=free, use_average=False; fit: patience=6 | 0.9503 | 5.7 | yes |
| `P_pat6_gfree_nobeta` | Proposed | build: gamma_mode=free, use_beta=False; fit: patience=6 | 0.9468 | 11.0 | yes |
| `P_pat6_gfree_relu` | Proposed | build: gamma_mode=free, average_after_relu=True; fit: patience=6 | 0.9497 | 6.1 | yes |
| `P_pat6_gfree_txt20` | Proposed | build: gamma_mode=free; fit: patience=6, p_text_hidden=0.2 | 0.9517 | 6.4 | yes |
| `P_pat6_txt20` | Proposed | fit: patience=6, p_text_hidden=0.2 | 0.9511 | 5.7 | yes |
| `P_pg10` | Proposed | fit: p_page_hidden=0.1 | 0.9496 | 5.8 | yes |
| `P_pg40` | Proposed | fit: p_page_hidden=0.4 | 0.9489 | 4.5 | yes |
| `P_raw_pat6` | Proposed | build: link_dim=45, _raw_link=True; fit: patience=6 | 0.9513 | 6.1 | yes |
| `P_raw_pat6_gfree` | Proposed | build: link_dim=45, gamma_mode=free, _raw_link=True; fit: patience=6 | 0.9511 | 6.1 | yes |
| `P_txt20` | Proposed | fit: p_text_hidden=0.2 | 0.9501 | 4.6 | yes |
| `current_Proposed` | Proposed | recorded recipe | 0.9495 | 6.2 | yes |

Helper folders in `tuning/` (not runs): `_lanes_frac0.5`, `_lanes_frac0.75` = Random Forest link lane rebuilt on 50% / 75% of the training data (learning curve); `_lanes_rfbal` = link lane with a class-balanced Random Forest.

## Appendix B. R1 recipe check (`recipe_r1/`)

```
largest replay gap (current models vs recorded score; retrained runs vs recorded train_log on every shared epoch): 1.11e-16 (must be <= 1e-12)

arm                                    current               R1              R1p
worst_b_minus_c                            -36              -40              -28
total_b_minus_c                             46               10               20
pairings_p_below_05_and_ahead                2                0                1
Proposed bal-acc                        0.9495           0.9479           0.9518
Proposed best epochs           2-16 (mean 6.2)  2-14 (mean 6.9)  2-16 (mean 7.1)
CNN-only bal-acc                        0.9482           0.9483           0.9514
CNN-only best epochs           2-12 (mean 5.0)  2-19 (mean 6.4)  2-21 (mean 7.3)
Concat bal-acc                          0.9464           0.9465             0.95
Concat best epochs             3-12 (mean 5.9)  3-17 (mean 9.3)  3-19 (mean 8.7)
kept                                     False            False            False

projected McNemar b - c per seed (Proposed vs baseline, same arm; validation rows, proxy):
seed              42    123   2024
arm     baseline                  
R1      CNN-only -40.0  13.0  14.0
        Concat    -5.0  15.0  13.0
R1p     CNN-only -28.0 -11.0  32.0
        Concat    -5.0  16.0  16.0
current CNN-only -36.0   5.0  33.0
        Concat     1.0  27.0  16.0

proxy p per seed:
seed                42      123     2024
arm     baseline                        
R1      CNN-only  0.0015  0.3053  0.3187
        Concat    0.7511  0.1994  0.3122
R1p     CNN-only  0.0254  0.3786  0.0188
        Concat    0.7464  0.1297  0.2049
current CNN-only  0.0051  0.7380  0.0138
        Concat    1.0000  0.0181  0.2049

KEPT: none (no arm beats the current recipe on the keep rule; stop and report)
```

## Appendix C. Tuning leaderboard (`tuning/report.txt`, first part)

```
LEADERBOARD (mean validation balanced accuracy over 15 cells)
            variant     kind  bal_acc  best_epoch  seconds
        KF1_w345x64 CNN-only 0.964692    8.600000   3382.9
        KF1_w234x64 CNN-only 0.964030    6.466667   2152.5
        KF1_w234x32 CNN-only 0.959439    7.000000   1566.3
  K_lr3e3_pat8_pg10 CNN-only 0.958815    6.466667    479.2
       K_lr3e3_pat8 CNN-only 0.957836    6.200000    484.9
  K_lr3e3_pat6_pg10 CNN-only 0.957198    5.400000    411.0
       K_lr3e3_pat6 CNN-only 0.956764    5.200000    472.0
          K7_frac75 CNN-only 0.956060    7.066667    464.7
            K_lr3e3 CNN-only 0.955260    3.400000    280.9
      K_lr25e3_pat8 CNN-only 0.953635    7.333333    598.6
       K_lr4e3_pat6 CNN-only 0.953221    4.666667    378.1
            K_lr2e3 CNN-only 0.951685    3.200000    236.1
             K_pat6 CNN-only 0.951382    7.333333      0.0
      K_lr3e3_txt20 CNN-only 0.950900    3.666667    579.8
              K_ls0 CNN-only 0.950554    5.266667    307.8
  K_lr3e3_pat6_bs32 CNN-only 0.950339    5.466667    827.9
   K_lr3e3_pat6_ls0 CNN-only 0.949827    4.066667    617.9
            K_txt20 CNN-only 0.949819    4.733333    275.3
             K_bs32 CNN-only 0.949641    3.866667    413.4
            K_txt05 CNN-only 0.949074    4.266667    267.1
          K7_frac50 CNN-only 0.949049    6.733333    364.5
   current_CNN-only CNN-only 0.948160    5.000000     14.7
             K_pg10 CNN-only 0.947956    4.266667    287.1
             K_pg40 CNN-only 0.947493    3.533333    247.5
        K_lr3e3_ls0 CNN-only 0.946512    1.933333    472.1
            K_lr5e4 CNN-only 0.946066    6.333333    320.0
            K_bs128 CNN-only 0.945883    5.066667    219.3
        CF1_w345x64   Concat 0.956283   10.200000   3015.6
        CF1_w234x64   Concat 0.956208   14.200000   3500.9
        CF1_w234x32   Concat 0.954088   12.733333   2492.4
      C_pat10_txt20   Concat 0.953459    9.400000    676.4
       C_pat8_txt20   Concat 0.953107    8.666667    525.9
       C_pat6_txt20   Concat 0.952734    8.200000    761.9
C_pat6_txt20_lr15e3   Concat 0.952278    5.800000    398.8
       C_pat8_txt25   Concat 0.951684    9.533333    654.2
       C_pat6_txt30   Concat 0.950808    6.866667    466.7
  C_pat6_txt20_pg10   Concat 0.950766    8.400000    549.3
       C_pat6_lr2e3   Concat 0.950498    6.400000    727.2
        C_pat6_pg10   Concat 0.950083    9.066667    859.5
             C_pat6   Concat 0.950040    8.733333      0.0
            C_lr2e3   Concat 0.949837    4.933333    266.4
            C_txt20   Concat 0.949680    6.266667    296.7
           C6_rfbal   Concat 0.949406    8.133333    664.7
 C_pat6_lr2e3_txt20   Concat 0.948856    5.200000    526.2
             C_pg10   Concat 0.948699    7.600000    364.6
             C_bs32   Concat 0.947099    5.800000    482.5
     current_Concat   Concat 0.946400    5.866667     11.6
          C7_frac75   Concat 0.946250   12.000000    575.1
            C_lr3e3   Concat 0.945791    3.733333    256.9
            C_lr5e4   Concat 0.945001    9.000000    373.1
             C_pg40   Concat 0.944644    6.200000    303.4
            C_bs128   Concat 0.944372    8.933333    281.3
            C_txt05   Concat 0.941060    4.600000    253.9
          C7_frac50   Concat 0.938103   14.933333    513.2
              C_ls0   Concat 0.930892    3.733333    244.6
             PG2_ln Proposed 0.963268    7.733333   3113.9
PF3_br_w345x64_krec Proposed 0.963057    8.866667   3195.3
           PG2_deep Proposed 0.962661    8.466667   3783.5
```

