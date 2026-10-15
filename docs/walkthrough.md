# Le-Jet walkthrough: where we are, what the RF is for, how to train it, what data is still missing

**Written 2026-10-14. Living document: update the status column as steps are done.**
Companion files: `rf_link_expert_architecture.md` (design and datasets), `unified_architecture.md` (the dataset columns, splits and the current trainer), `fresh_set_spec.md` (the fresh-set CSV format).

**Group constraints, never to be dropped:** page inspection and the web-derived page/domain features stay in the model. The CNN is the **text path only**.

---

## 1. Where we are

| Item | Status |
|---|---|
| Panel challenge: SMS text alone is not credible evidence | Accepted. Link and page evidence is the focus. |
| RF vs 1D CNN on the 31 link/page features, no text | **Done** on real pages (`web_phishing_clean`). RF wins. |
| One forest (complete + blanked pages) vs two dedicated forests | **Done.** One forest is enough. |
| Fusion (text CNN + RF) beats text alone | **Not shown.** Placeholder data cannot show it. Needs the fresh set. |
| Calibration, reliability gate, small head | **Not built.** Designed only. |
| Fresh inspected SMS set (D4) | **Not collected.** Needs the adviser's decision (section 7). |
| Serving path / demo | Still on the old two-extractor design. Move last. |

Everything run on `unified_every_sms*` and `unified_dataset` uses placeholder page features (an unrelated page paired with each SMS). Those scores are **not study results**.

---

## 2. The novelty of the RF and its role

### What the RF is
One Random Forest that reads **the link and the page behind it**: 14 URL features and 17 page/domain features (31 inputs). The page columns may be NaN when the page was not reached. It outputs:
- `P`: probability the link is smishing (calibrated to `P_cal`),
- `u`: its uncertainty,
- per-feature contributions `c_1..c_31` (explanations only).

### Why it is part of the architecture
1. **The evidence has two shapes.** Text is a word sequence; link/page evidence is a table of unordered, mixed-scale numbers (counts, ratios, ages in days, 0/1 flags). A CNN is built for the first and has no structure to exploit in the second.
2. **Measured, not assumed.** On 11,430 real pages (3 seeds, domain-grouped splits):

| Features | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 31 | RF | 0.911 ± 0.009 | 0.930 ± 0.014 | 0.875 ± 0.007 | 0.902 ± 0.009 |
| 31 | 1D CNN | 0.868 ± 0.020 | 0.866 ± 0.027 | 0.849 ± 0.022 | 0.858 ± 0.021 |
| 14 (URL only) | RF | 0.813 ± 0.010 | 0.816 ± 0.027 | 0.774 ± 0.014 | 0.795 ± 0.007 |
| 14 (URL only) | 1D CNN | 0.791 ± 0.009 | 0.813 ± 0.014 | 0.718 ± 0.007 | 0.763 ± 0.009 |

   The gap is clear on 31 features (4.3 points accuracy) and small on URL only. **Do not claim RF superiority on URL-only.** A panelist argued a CNN can do URL-only; our numbers do not rule that out.
3. **Trees handle missing evidence natively.** 82% of old smishing domains are dead. One forest trained on complete and blanked pages keeps F1 0.909 on complete pages and 0.781 on blanked ones (dedicated URL-only forest: 0.795). A forest never trained on blanks drops to recall 0.51 when the page is missing.
4. **Cheap and explainable.** Milliseconds on CPU, and its contributions show why a link was flagged. Page inspection (the network fetch) dominates latency in every design.

### What is new in our design (state it carefully)
- The RF is a **dedicated link/page expert** whose verdict (`P_cal`, `u`) is passed to the text model's head, instead of letting a neural net re-learn the table.
- **Missing-evidence handling** in one model: blank-augmented training plus a reliability gate that lowers trust in the RF when the page was not reached.
- **Live page inspection** feeds the features, so the verdict reflects the page as it is now.

> Not yet checked: how much of this is already published (RF/CNN stacking is common). Do a related-work pass before the thesis words it as a novelty. Use Grinsztajn et al. 2022 (arXiv 2207.08815, trees beat deep nets on ~10K-row tabular data) as support, and Zabergja et al. 2024 (arXiv 2402.03970, deep nets now win on 68 datasets) as the counterpoint. The claim to word: "a plain neural head on raw features vs a forest, at this data size." The honest test is RF vs MLP; **the MLP-only run on the real pages is still open**.

---

## 3. The architecture

```
SMS text ──► CNN text encoder (fastText, 64 filters) ──► 64 text features ─────────────┐
             (text path ONLY: the link is masked as <URL>)                             │
                                                                                       │
Link: 14 URL features ─┐                                                               │
Page: 17 features ─────┴─► ONE RF (page columns NaN if page not reached)               │
                              │                                                        │
            out-of-fold P (training rows) → calibrate → P_cal, u                       │
                              │                                                        │
   u, page_missing, text confidence ──► reliability gate g (0..1)                      │
                              │                                                        │
                    g × [P_cal, u] + has_url + page_missing ───────────────────────────┤
                                                                                       ▼
                                                 Dense 64 → ReLU → Dropout → 1 logit → smishing / legitimate
Side output, never an input to the head: contributions c_1..c_31 → explanation
```

### Keeping the CNN from overpowering the RF
The text vector is 64 numbers; the RF branch is about 4 numbers. Left alone, the head can learn to follow the text and ignore the RF (this is what happened on the placeholder data, where text alone scored ~0.95). Guards to build and test:

| Guard | What it does | Status |
|---|---|---|
| CNN sees text only | No raw link or page features reach the CNN. | Design rule |
| Freeze the text encoder in Phase C | The trainer currently optimizes all model parameters, so the encoder keeps adapting. Freezing keeps it a fixed text reader. | **To build and compare** |
| Modality dropout | During head training, randomly zero the text vector (e.g. 20-30% of batches) so the head must also learn from the RF alone. | To build |
| Narrow the text branch | Project the 64 text features down before the head so the branches are closer in width. | To test |
| Reliability gate | Scales the RF branch by how much it can be trusted, instead of letting text drown it. | To build |
| Verify, do not assume | Report the head's score with the text vector zeroed and with the RF branch zeroed. If zeroing the RF changes nothing, the CNN is overpowering. | To build |

---

## 4. Baselines (only these two)

| Baseline | What it is | Question it answers |
|---|---|---|
| **CNN** (`cnn_text`) | Text CNN, link masked, no RF. | Does link/page evidence add anything to text? |
| **UNGATED** (`unified_nogates`) | The same text CNN plus the RF outputs, every gate fixed at 1.0. | Does the gate add anything beyond just feeding the RF in? |

The proposed model is the **gated** model. Comparison order: CNN → UNGATED → gated. All three use the same splits, seeds, validation slice, class weights and training recipe.

`cnn_raw31` (CNN with raw features, no forest) still exists in the repo (`run_baseline.py`, `compare_models.py`) but is **not a reported baseline**. The RF-vs-neural question is answered by the features-only comparison (section 2), not by this model. The RF-only and MLP-only fusion-set baselines were removed.

Metrics, always: **accuracy, precision, recall, F1**, mean ± std over seeds 42, 123, 2024. McNemar's test per seed for paired comparisons.

---

## 5. Training steps, mapped to the architecture

| # | Step | Architecture block | Command / status |
|---|---|---|---|
| 0 | Build the dataset (one row per SMS; text, link, 14 URL + 17 page features, inspection status) | Inputs | `build_unified_dataset` (placeholder data now). Real data: D4. |
| 1 | Grouped split by domain/template, per seed (big scam domains to Training) | All | `python -m src.unified.make_splits` |
| 2 | **Train the RF.** Tune by grouped CV on Training link rows. Train on every row twice: complete and with the 17 page columns blanked. | RF | Forest tuning exists (`unified_rf.tune`). Blank-augmented training proven in `run_single_rf.py`; **to move into the trainer**. |
| 3 | **Out-of-fold `P`, `u`** for Training rows (5-fold, grouped; a row is never scored by a forest that saw it). Final forest scores `rf_fit`, test and live links. | RF → head | `unified_rf.oof_outputs` exists. |
| 4 | **Calibrate `P`** (isotonic or Platt) on the out-of-fold predictions. | RF → head | **To build** |
| 5 | **Phase A: train the text CNN** on text only. | CNN | `run_unified` (Phase A). Checkpoint per seed. |
| 6 | **Gates.** Current: static gates from permutation importance on `rf_fit`. Proposed: one reliability gate from `u`, `page_missing`, text confidence. | Gate | Static exists; **reliability gate to build** |
| 7 | **Phase C: train the head** with the CNN guards from section 3. Class weights, early stopping on the validation slice (balanced accuracy). | Head | Trainer exists; **guards to build** |
| 8 | **Baselines with the same recipe:** `cnn_text`, then UNGATED (`run_nogates`). | Baselines | `run_nogates`, `compare_models` exist; recheck after step 7 changes |
| 9 | **Test once** on the sealed test part. No tuning after this. | All | |
| 10 | Report accuracy, precision, recall, F1 for CNN, UNGATED, gated; plus the zeroing checks and results split by no link / dead page / live page. | | |

Ablation runs once D4 exists (D5 only to check the code runs): head inputs `{P_cal}` vs `{P_cal, u}` vs `{P_cal, u, c_1..c_31}`; gate on vs off; calibration on vs off; frozen vs trainable encoder; modality dropout on vs off.

Rules that always hold (`training_rules.md`): rows with a redacted `<URL>` are dropped from training and evaluation; the test set is read once; no row, domain or template crosses split boundaries.

---

## 6. What the data runs so far taught us

| Run | Data | Result | Meaning |
|---|---|---|---|
| Fusion vs text | Placeholder pairings | Text CNN ≈ 0.94-0.95 balanced accuracy; fusion never better | Text nearly solves this corpus; unrelated page features add nothing. Cannot show a fusion gain. |
| RF vs 1D CNN | `web_phishing_clean` (real) | RF F1 0.902 vs 0.858 on 31 features | RF is the better link/page reader. |
| One RF vs two | `web_phishing_clean` | Single 0.909 complete, 0.781 blanked | One forest is enough if trained on blanked pages. |

The text-only result is also why the panel's remark matters: this SMS corpus is probably easy from text alone (template and domain regularities), so a text score says little about real-world credibility. The study has to be argued on link/page evidence first.

---

## 7. Future work: the dataset

### What we need (D4: fresh inspected SMS set)
One row per SMS, text, link and live page features collected together.

| Column group | Content | Rule |
|---|---|---|
| Message | `sms_text`, `label`, `source` | Label is ground truth, never changed because a page was dead or looked clean. Personal details removed. |
| Link | first link as written, `link_status`, `has_url` | **Both classes must have links**, or "has a link" becomes a shortcut. |
| URL features (14) | from the raw link via `link_text_features` | Do not reimplement. |
| Page features (17) | page fetched within hours of receipt | Not reached: NaN, `page_fetched = 0`. Dead and blocked links stay in. |
| Record | `received_at`, `visited_at`, `hours_to_visit`, `fetch_status` | To report how fresh the inspection was. |

### How the sets divide the work

| Set | Rows | Used for |
|---|---|---|
| D4 fresh inspected SMS | to collect | `fresh_dev`: trains head, gate, calibrator. `fresh_test`: sealed, read once. |
| D1 `web_phishing_clean` | 11,430 | Pretrains the forest (real URL + page features). |
| D2 `sms_v2_balanced` | 21,910 | Pretrains the text CNN. |
| D3 `sms_link_features` | 7,869 with a real link (of 21,910) | URL-only statistics and the D1-to-SMS shift check. No page features. |
| D5 `unified_every_sms*`, `unified_dataset` | 11,430 / 11,430 / 15,000 | Code smoke tests only. Never reported. |

### The decision we are blocked on
`fresh_set_spec.md` says the fresh set is **test only**. This architecture needs paired text + `P_cal` + real page evidence to train the head, gate and calibrator, and only D4 provides that.
- **Option A (recommended):** split D4 into `fresh_dev` and a sealed `fresh_test`, grouped by domain. Update `fresh_set_spec.md` and `training_rules.md`.
- **Option B:** keep D4 test-only; train the head on D2 + D3 with URL-only `P`. The page branch is then judged only by the forest on D1 plus a fresh test, and the gate for `page_missing` cannot be learned from real pairs.

Size (my estimate, no power calculation): about 1,000 messages with links in `fresh_dev` and about 500 in `fresh_test`, both classes with links. If collection runs short, shrink the head (`{P_cal, u, has_url, page_missing}`) before shrinking the test set.

### Collection rules
- Record `hours_to_visit`; keep dead and blocked links; the share of live links is itself a result.
- No row from the old SMS data, and none of its domains without a flag. No domain on both sides of the split.
- No `hybrid_temp` rows and no re-inspected old SMS links in the final training set (82% of those domains are dead).
- If the forest trains on D1 + D4, a source flag must not be an input feature. Check D4 rows score no worse than D1 rows.
- Report in three groups: no link, link with dead/blocked page, link with live page. Page-inspection claims only on the live group.

### Other open items
- Adviser decision on option A or B.
- MLP-only vs RF on the real pages, so the "neural vs RF" claim does not rest on a 1D CNN.
- Check the D1 → SMS-link domain shift: the forest read with page columns blanked, applied to D3 links vs D1.
- Related-work pass for the novelty wording.
- Build: blank-augmented training in the trainer, calibration, reliability gate, the CNN guards.
- Decide the head's final inputs from the ablation, then move the serving path and `final_model.md` to this design.

---

## 8. Files to know

| File | Purpose |
|---|---|
| `src/unified/run_features_only.py` | RF vs 1D CNN, no text |
| `src/unified/run_single_rf.py` | one forest vs two, complete vs blanked pages |
| `src/unified/prepare_web_phishing.py` | prepares D1 and its domain-grouped splits |
| `src/unified/run_unified.py` | current trainer (Phase A, gates, Phase C) |
| `src/unified/run_nogates.py` | the UNGATED baseline |
| `src/unified/compare_models.py` | compares models on the same predictions |
| `src/unified/analyze_by_pair_type.py` | scores matched vs mismatched pairs (placeholder data only) |
| `experiments/web_phishing/` | real-page results |
| `experiments/every_sms*/` | placeholder results, not study results |
