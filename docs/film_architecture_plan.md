# Le-Jet final architecture: E2 (DAFT-style FiLM). Plan, study basis, decisions, phases

**Written 2026-10-19. Status: the group agreed to adopt E2 as the architecture (reported by Franz). Adviser approval is not confirmed yet (decision D1).**
Source design: https://claude.ai/artifact/F1mS94xC9WJn4Zrytyyibh ("Le-Jet Final Architecture").
Options page it came from: https://claude.ai/artifact/Gi7y316w18U1MavVFbitz1 ("Le-Jet FiLM Options").

**Where the rest of the project lives.** The C-prune plan this replaces is `docs/c_prune_architecture.md` **on the branch `feat/prune-architecture`** (not on `main`). That branch already holds the crowd-data pipeline, the link lane, the Phase A text CNN, the fusion trainer and two baselines, all run on a **synthetic pilot**. E2 reuses that code (Part C). Everything E2 does not need from C-prune (the Gate RF, ranking, pruning) simply stays unused.

**Nothing in this plan has been trained on Le-Jet data.** Sizes are arithmetic. Every accuracy claim below comes from a published study on a different task, and says so. The first evidence will come from Phase 4 and 5, and it will be pilot (synthetic) evidence until the crowd dataset exists.

How to read this file: Part A explains the architecture in plain words. Part B is the study basis (only papers read in full). Part C says what is reused and what changes. Part D is the decisions the group has to make. Part E is the menu of possible changes. Part F is the phased plan with steps. Part G is risks and fallbacks.

---

# Part A. The architecture

## A1. The idea in plain words

A smishing message has two parts: the **text** and the **link**. Two different models read them:

- a **text CNN** reads the words;
- a **Random Forest (the Link RF)** reads the link and the page behind it.

C-prune and the older designs glued the two together at the end. E2 does something different: the link evidence does not get its own seat at the decision table. Instead it **adjusts how the text model reads the words**. A risky link makes the text filters that look for threats "louder" for that message. A clean link leaves them alone. The final decision (Dense 64 + Output) only looks at text numbers. The forest never decides.

The adjustment is called **FiLM** (Feature-wise Linear Modulation): each of the 64 text filters gets its own multiplier γ ("gamma", how loud) and its own offset β ("beta", a push up or down), computed fresh for every message. So the gate is still **per filter** (as "features" in the research questions means filter features), but it is **graded and per message**, not fixed.

## A2. Picture

```
 SMS text (link removed)                              Link + the page behind it
        │                                                       │
        ▼                                                       ▼
 fastText embedding (300-d, frozen)                    Web inspector (37 features)
        │                                                       │
        ▼                                                       ▼
 Conv1D, 64 filters, width 3                           Link RF, 200 trees (frozen)
   output x: 64 filters × 48 positions                   → P, u, page_missing  (3 numbers)
        │                                                       │
        ├──► Conv average: mean of each filter                  │
        │    over its 48 positions → 64 numbers ───┐            │
        │                                          ▼            ▼
        │                              GENERATOR: one Dense layer
        │                              input 3 + 64 = 67 → output 128
        │                              Δγ (64) and Δβ (64)
        │                              γ = 1 + Δγ  (D2),  β = Δβ
        │                                          │
        ▼                                          │
 FiLM step:  x' = γ · x + β   (per filter, every position) ◄──── γ, β
        │
        ▼
 ReLU → max-pool over positions → 64 numbers
        │
        ▼
 Dense 64 → ReLU → dropout 0.5 → Output 2 → softmax → verdict
 (sees ONLY these 64 text numbers)
```

**Correction to the artifact's diagram.** In the artifact, the "Conv average" arrow is drawn on the same row as the Link RF, so it looks as if the average passes through the forest. It does not. The forest reads only the 37 link/page features. The Conv average goes to the **generator**, one row lower.

## A3. Glossary

| Term | Plain meaning |
|---|---|
| Filter | A small detector of 3 neighbouring words that learned one pattern (for example "account will be locked"). There are 64. |
| Position | One spot where a filter looks. 50 tokens and width 3 give 48 positions. |
| x | The grid of filter scores before the adjustment: 64 filters × 48 positions. |
| Conv average | For each filter, the mean of its 48 scores. 64 numbers. A summary of "what the filters found in this message". |
| P | The Link RF's probability that the link is phishing (mean over the 200 trees). |
| u | How much the 200 trees disagree (spread of the trees' probabilities). |
| page_missing | 1 if the page could not be fetched, so the page features were blank; 0 otherwise. |
| Generator | One Dense layer that turns (P, u, page_missing, Conv average) into γ and β. Trained together with the CNN. |
| γ (gamma) | Per-filter multiplier. 1 means "unchanged", above 1 louder, below 1 quieter. |
| β (beta) | Per-filter offset. It can raise a filter even when the words looked mild. |
| FiLM | Applying `γ · x + β` to every filter score. |

## A4. Stage by stage (what each step does and why)

| # | Stage | What it does | Shape | Trained? |
|---|---|---|---|---|
| 1 | Split | Cuts the link out of the SMS (the link becomes the `<URL>` token, as in the live app). Text goes to the CNN, link to the inspector. Stops the text path learning link spelling, which the forest handles. | text, link | no |
| 2 | Embedding | Each word becomes 300 numbers (fastText), message cut/padded to 50 words. | 50 × 300 | **no, frozen** (decided 2026-10-18 for every model) |
| 3 | Conv1D | 64 filters slide over every 3-word window. | 64 × 48 | yes |
| 4 | Conv average | Mean of each filter's scores. Feeds the generator only. | 64 | no (no weights) |
| 5 | Web inspector | Reads the link and fetches the page: 37 features (20 URL + 17 page/domain). | 37 | no |
| 6 | Link RF | Turns 37 features into P, u, page_missing. | 3 | **no, frozen** |
| 7 | Generator | One Dense layer, 67 → 128. Its weights and bias start at zero, so at the start γ = 1 and β = 0 and training begins as a plain text CNN. | 128 | yes |
| 8 | FiLM step | `x' = γ · x + β`, before ReLU. | 64 × 48 | no (no weights) |
| 9 | ReLU + max-pool | ReLU zeroes negatives; max-pool keeps each filter's strongest value. | 64 | no |
| 10 | Dense 64 → Output | Head. Decides the verdict. Sees text numbers only. | 64 → 64 → 2 | yes |

Worked example (the numbers are **invented to show behaviour, not results**). Message A: `BDO: your account will be locked. Verify now bdo-secure-login.xyz/verify`. Message B: `BDO: your parcel is on its way. Track it at bdo-secure-login.xyz/verify`. Same risky link. The "urgent threat" filter averages 0.8 on A and 0.1 on B. The generator sees the same P, u, page_missing but different averages, so A gets a bigger boost than B (for example γ 1.6 and β +0.3 against γ 1.1 and β +0.1). This is why the generator reads the text summary: without it, every message with the same link would get the same γ and β, and the generator could be replaced by a lookup table.

## A5. Size and compute (exact arithmetic, verified)

| Part | Trainable | Working |
|---|---|---|
| Conv1D, 64 filters | 57,664 | 300 × 3 × 64 + 64 |
| Generator | 8,704 | (3 + 64) × 128 + 128 |
| Dense 64 | 4,160 | 64 × 64 + 64 |
| Output | 130 | 64 × 2 + 2 |
| **Total** | **70,658** | embedding frozen (counted separately as frozen) |

- With γ only, the generator has 4,352 parameters and the total is 66,306.
- Per message: Conv 2,764,800 multiply-adds (48 × 300 × 3 × 64), FiLM 3,072 multiplies + 3,072 additions, Conv average 3,072 additions.

**Honest comparison with the models already built on the C-prune branch** (trainable parameters, from `c_prune_architecture.md` section 5.3):

| Model | Trainable | vs E2 |
|---|---|---|
| C-prune k = 16 | 18,194 | E2 is 3.9× larger |
| C-prune k = 32 | 33,634 | E2 is 2.1× larger |
| C-prune k = 48 | 49,074 | E2 is 1.4× larger |
| Plain text CNN (64 filters) | 61,954 | E2 is +14.1% |
| CNN_CONCAT / UNGATED (64 filters + 40 link numbers) | 64,514 | E2 is +9.5% |
| **E2** | **70,658** | |
| E2, γ only | 66,306 | |

So E2 is **not** a smaller model than the base or than C-prune. Its possible efficiency gains are elsewhere, and none of them is measured yet:
- **One forest, 3 outputs.** E2 reads only P and u from the Link RF. C-prune's link lane also computes 37 per-feature contributions (Saabas path decomposition). On the synthetic pilot the C-prune plan notes the Link RF takes about **213 ms per message** for that step, about 600× the text model (0.35 ms). E2 does not need the contributions, so this cost should drop. **Measure it in Phase 4; do not claim it before.**
- The real latency is still dominated by fetching the page (seconds). The generator does not change that.

---

# Part B. Study basis

Rule used: **a paper is cited only if its full text was read.** Each paper below was opened and read in full (PDF text extracted and checked against the HTML version). Where I could not confirm something, it says "not verified". Papers listed in B6 were **not** read, so this plan does not rely on them.

## B1. FiLM: Perez, Strub, de Vries, Dumoulin, Courville (AAAI 2018), arXiv 1709.07871

What it is: the original FiLM layer. A question (text) conditions a CNN that answers questions about images (CLEVR).

What I read and can state:
- **Definition (Eq. 1 and 2).** γ and β are functions of the conditioning input; `FiLM(F | γ, β) = γ · F + β` per feature map. This is what the generator + FiLM step do in E2.
- **Placement.** They turn off the batch-norm affine parameters that immediately precede a FiLM layer, and the ReLU comes after FiLM. They write: "since our model uses a ReLU after FiLM, γ < 0 can cause a significantly different set of activations to pass the ReLU". In their trained model, 36% of γ values were negative.
- **γ and β matter differently.**
  - Training-time ablation (Section 4.3): constant γ = 1 gave a 1.5% accuracy drop, β = 0 a 0.5% drop. "γ is more important than β."
  - Test-time ablation: replacing β by its training mean dropped accuracy 1.0%; replacing γ by its mean dropped it **65.4%**. "FiLM largely conditions through γ rather than β."
  - *Correction to the options page:* the 65.4% / 1.0% numbers are test-time mean replacement on a trained model, not "removing γ's conditioning at training time". The training-time gap is the smaller 1.5% vs 0.5%.
- **Restricting γ hurts.** Restricting γ with a sigmoid, tanh or exp lowered accuracy compared with leaving γ free (95.9, 96.3, 96.3 against 97.4 ± 0.4, Table 2, read from the HTML version). The text says the sigmoid and tanh restrictions "hurt performance, roughly as much as" fixing γ = 1.
- **Parametrization (Appendix 7.2, Eq. 3).** They output Δγ and use γ = 1 + Δγ, "since initially zero-centered γ can zero out CNN feature map activations and thus gradients". They say this "does not seem to affect" CLEVR performance significantly.
- **Placement inside the block (Table 2).** Variants ranged from 95.0 to 97.7 depending on where FiLM sits; the text says moving FiLM within the block gave "no substantial performance drop" for most positions. The exact row labels are garbled in the PDF text, so only the range is cited.

What it does **not** support: any text-classification use. All of its evaluation is visual question answering. The authors name RNNs and reinforcement learning as future directions.

Why it matters for us: it is the origin of `γ · x + β`, the "1 + Δγ" start-at-identity trick, and the warning about restricting γ (see D2).

## B2. DAFT: Pölsterl, Wolf, Wachinger (MICCAI 2021), arXiv 2107.05990

What it is: FiLM where the generator reads **both** the tabular data and a summary of the image feature map. This is the "E2" idea.

What I read and can state:
- **Mechanism (Section 3, Eq. 1).** `F' = α · F + β`, with α and β computed from the feature map *and* the tabular data. "DAFT first creates a bottleneck by global average pooling of the image feature map, concatenating the tabular data, and squeezing the combined vector by a factor r via an FC layer", then an MLP outputs the scale and shift. FC layers have no bias terms. An activation is applied to the scale only, not the shift; linear, sigmoid and tanh were tried, and the proposed configuration is described as using a linear activation (σ(x) = x), i.e. **γ was not squeezed into a range**.
- **This supports our "Conv average" idea** (the global average of the feature map goes into the generator with the side data). It does **not** support our exact generator: DAFT squeezes through a small bottleneck (4 units for 15 tabular features), uses an MLP, no biases, and sits in the last residual block of a 3D ResNet. Our generator is one Dense layer with bias. E2's generator is "DAFT-inspired", not DAFT.
- **Results (Table 2, balanced accuracy, ADNI, 5 folds, test column).** DAFT 0.622 ± 0.044; FiLM 0.601 ± 0.036; concatenation variants 0.587, 0.576, 0.591. Time-to-dementia c-index: DAFT 0.748, FiLM 0.712, concatenation 0.723–0.729. **Caveats:** on the *validation* column FiLM (0.652) is above DAFT (0.642); the gaps on test are within one standard deviation; the paper runs no significance tests.
- **Scale vs shift (Table 4).** Proposed 0.622; scale removed (α = 1) 0.581; shift removed (β = 0) 0.609. The paper: "scaling is more essential than shifting, but removing any of them comes with a decisive performance drop of at least 0.013". Both help; scale helps more.
- Data: ADNI, 3D brain MRI + 15 tabular features, 1,341 subjects (diagnosis task). Not text.
- The exact position of DAFT in the network is described inconsistently between text and table, so no position is cited.

What it does **not** support: any text task, any link/URL task, datasets of a few hundred to a few thousand rows of a different kind.

## B3. HyperFusion: Duenias et al. (2024), arXiv 2403.13319 (v3)

What it is: a hypernetwork that turns tabular data into weights of the image network. Used here only as a second comparison of conditioning against concatenation.

Table 3, three-class Alzheimer's classification, ADNI, balanced accuracy (mean ± std):

| Model | Balanced accuracy |
|---|---|
| Concatenation | 0.638 ± 0.008 |
| FiLM-like | 0.635 ± 0.014 |
| DAFT-like | 0.658 ± 0.014 |
| HyperFusion (their method) | 0.673 ± 0.012 |
| Reported DAFT (from the DAFT paper) | 0.622 ± 0.044 |

- "FiLM-like" and "DAFT-like" are the HyperFusion authors' own adapted re-implementations, not the original code.
- The FiLM-like row ties plain concatenation (0.635 vs 0.638). The DAFT-like row is above both.
- **Not independent of B2:** both use ADNI. They are two analyses of related data, so they are not two separate confirmations.

## B4. Text use of FiLM: Zayed, McCrae, Buitelaar (Findings of EMNLP 2020), arXiv 2010.05633

The only study read that applies FiLM to text (metaphor identification in sentences).
- **How FiLM is used.** A feed-forward "contextual modulator" turns the embedding of the targeted verb–noun expression into γ and β (`γ = W_γ c + b_γ`, `β = W_β c + b_β`), which modulate **every word's LSTM hidden state** (`γ ⊙ h + β`). The conditioning input comes from the same sentence, not from another data source.
- **Results (Table 1, F1, without vs with the FiLM layer, no attention):** ZayTw tweets 0.685 → 0.786; TSV 0.630 → 0.834; TroFi 0.725 → 0.879; VUAMC 0.407 → 0.673; MOH 0.759 → 0.771. The `*` in their table marks significance against the SSN baseline, not against the plain LSTM. The MOH gain is small and unmarked. Averages over five random seeds. Dataset sizes were not verified, so none are quoted.

What it does **not** support: conditioning a text CNN on a different modality (a link), or anything beyond metaphor tasks.

## B5. Concatenation as a baseline: Kim et al. (TACL 2019), "Categorical Metadata Representation for Customized Text Classification"

- Task: text classification with categorical metadata (user, product). Methods that use the metadata "as is" inside the model (word embeddings, attention) are "outperformed even by simple concatenation of categorical features in the final layer of the sentence encoder". Their own proposal (basis vectors) beats concatenation.
- **This is not about FiLM.** It supports one narrow point: **concatenation is a strong baseline that clever fusion has to beat**, which is why CNN_CONCAT and UNGATED stay in the comparison. The options page described this as "simple concatenation is hard to beat"; that phrase is not in the paper, and the paper itself reports beating concatenation with its own method.

## B6. Read nothing, cite nothing

The options and architecture pages, and the C-prune plan, cite other studies. **I did not read them**, so this plan does not rely on them: Sui et al. 2021 (CHIP), Li et al. 2017 (Pruning Filters), the VoiceFilter-Lite paper, the keyword-spotting paper, Temporal FiLM, and the other pruning, dataset-size and sample-size references in `c_prune_architecture.md` section 8. `progress.md` item 9 also names gated multimodal units (Arevalo 2017) and ModDrop (Neverova 2016); not read here either.

No study was found for using **tree disagreement (u)** as a trust signal (`progress.md`, task 9). Feeding `u` to the generator is our own choice; test it (E5).

## B7. Claim ledger: what the artifact says and where it stands

| Claim in the artifact / options page | Status |
|---|---|
| FiLM definition `γ · x + β` | Verified (Perez Eq. 2). |
| "Beat plain FiLM and concatenation in two studies" | Numbers verified (DAFT Table 2; HyperFusion Table 3). Weakened: same ADNI data, medical images, DAFT gaps within one standard deviation, no significance tests, validation column favours FiLM in the DAFT paper. |
| "Scaling is more essential than shifting" | Verified wording (DAFT). Both matter: dropping β cost 0.013, dropping γ 0.041. |
| γ-only: "removing γ cost 65.4%, β 1.0%" | Numbers exist, but they are test-time mean replacement, not training-time removal (Perez). Training-time drop: 1.5% vs 0.5%. |
| "γ = 1 + tanh(Δγ), between 0 and 2" | **Not what either paper did.** Perez used γ = 1 + Δγ (unbounded) and found restricting γ hurt; DAFT's proposed config was linear. See D2. |
| "FiLM beat concatenation in text (F1 0.685 → 0.786)" | The 0.685 is the plain LSTM (no FiLM), not concatenation. It shows FiLM > no modulation, in one metaphor task, with conditioning from the same sentence. |
| "[F9] warns simple concatenation is hard to beat" | The paper says metadata-"as-is" methods lose to concatenation and its own method beats it. Not a FiLM result. |
| "Safe start: γ = 1, β = 0 gives a plain text CNN" | Verified idea (Perez Appendix 7.2); implement and unit-test (Phase 2). |
| "FiLM works for a text classifier with side data from another source" | **No study read supports this directly.** It is the thesis's own test. |

## B8. What our own results already say

From `docs/progress.md` on `main` (synthetic-text lean set, so **not study results**): giving the head the link features (CNN_CONCAT) beat text-only by about 8.3 F1 points, and the forest added about 2 more points over raw concatenation, significant in 1 of 3 seeds ("suggestive, not shown"). Static gates showed no gain over shuffled gates (sections 2e and 2g). From the C-prune branch pilot rehearsal (synthetic, pipeline test only): UNGATED and C-prune k = 16 were not significantly different. So the starting position for E2 is: **concatenation is already strong, and the thesis must show FiLM adds something on top of it, or at a better cost.**

---

# Part C. What is reused, what changes

## C1. Already decided in the C-prune plan, and kept for E2

These came from `c_prune_architecture.md` Phase 0 (decided 2026-10-18) and apply to **every model**, E2 included. E2 does not reopen them.

| Item | Decision |
|---|---|
| Data | Crowdsourced SMS with a link, one dataset (`data/processed/crowd_smishing_v1.csv`, 57 columns). Minimum 2,000 messages, recommended 4,000. Every message has a link, so `has_url` is dropped. No synthetic text in the real study. |
| Evaluation | 5-fold cross-validation grouped by domain below 4,000 messages; seeds 42, 123, 2024; 15% of each training part held out as validation. Test folds read **once**, last. |
| Embedding | fastText 300-d, **frozen** for every model. |
| Recipe | Adam 1e-3, up to 30 epochs, batch 64, patience 3 on validation balanced accuracy, label smoothing 0.1, class weights. Weight decay 1e-4 on Dense 64's weights (`DENSE_WEIGHT_DECAY`). |
| Modality dropout | While training only: page hidden in 25% of messages (link numbers swapped for the blank-page version, `page_missing = 1`), text hidden in 10%. Same for every model. |
| Research questions | RQ1 accuracy, precision, recall, F1 (smishing = positive). RQ2 trainable parameters, latency per message (total wall-clock including the web inspection, one shared inspection per test link), training time. RQ3 exact McNemar per seed, Holm-corrected. RQ4 was removed on 2026-10-18 and is not re-added. |
| Thesis title | Unchanged so far: *A Random Forest-Guided Static Gating Framework for Financial-Related Smishing Classification Using SMS Text*. E2's gates are per message, so see D1 and D15. |

## C2. Code on `feat/prune-architecture` that E2 reuses (read headers and function lists only)

| Need | Existing code | Use for E2 |
|---|---|---|
| Crowd data, splits, per-fold vocabulary | `src/data/build_crowd_dataset.py`, `src/data/split_crowd.py`, `src/features/build_crowd_embeddings.py`, `data/splits_crowd/` | As is. |
| Link lane | `src/unified/link_lane.py`: `load_lane(seed, fold)`; per row P, u, 37 contributions `c_*`, `page_missing`; train rows out-of-fold, val/test from the final forest; every row also scored with the page hidden | E2 takes only P, u and `page_missing` from it. Skipping the contributions is the point of the single-forest, 3-output design. A cheaper "P, u only" scoring path may be needed (Phase 1). |
| Text CNN (Phase A) | `src/unified/text_cnn.py`: `load_phase_a`, per-fold vocabulary and `SEQ_LEN_L`, frozen embedding, sidecar JSON with hashes; `TextEncoder` in `src/features/model/cnn_encoder.py` | Warm start for the Conv (D7). `TextEncoder.encode` returns only the pooled vector, so E2 needs a new class that exposes the pre-pool Conv output. |
| Fusion training | `src/unified/fusion.py`: `fusion_inputs`, `fine_tune` (recipe, early stopping, modality-dropout draws), sidecar save, `load_c_prune`, `predict_messages` | Pattern to copy for `film.py`. |
| Baselines and comparison | `src/unified/baselines.py`: `plain_cnn` and `ungated` built and trained; `compare` scores every model on the test folds (RQ1), counts parameters and times stages (RQ2), runs McNemar with Holm (RQ3); refuses real-data test reads without `--read-test` and logs them | Add E2 to the same `compare`. CNN_CONCAT and static-gated baselines are still unchecked in the C-prune Phase 7 list. |
| McNemar | `src/features/model/mcnemar.py` | As is. |
| Tests | `tests/test_prune.py` | Model for `tests/test_film.py`. |

Not needed by E2 and left alone: `gate_rf.py` (Gate RF, ranking, masks), `prune_encoder`, `select_top_k`, `PrunedFusionClassifier` masks. They remain the C-prune comparison model (D19).

## C3. What actually changes for the model

| Area | C-prune | E2 |
|---|---|---|
| Link evidence reaches the verdict | Concatenated into the head (40 numbers) | Only through γ and β on the Conv output (3 numbers + Conv average) |
| Head input | k text numbers + 40 link numbers | 64 text numbers |
| Gating | Fixed 0/1 mask on Conv filters (static) | γ, β per filter per message (dynamic) |
| Link lane | P, u, 37 contributions, `page_missing` | P, u, `page_missing` |
| Pruning, Gate RF | Yes | None |
| Text CNN | Phase A then pruned and fine-tuned | Phase A warm start, then Conv + generator + head trained together |
| Trainable parameters | 18,194 / 33,634 / 49,074 | 70,658 |
| New code | n/a | `FiLMTextCNN`, `film.py`, `tests/test_film.py`, a line in `baselines.py` |

---

# Part D. Decisions the group has to make

Each decision has a recommended default so work can start; change it if the group disagrees. "Blocks" says which phase needs the answer first. ✅ means already decided in the C-prune plan or by the earlier pages.

| ID | Decision | Options | Recommendation | Blocks |
|---|---|---|---|---|
| **D1** | **Adviser sign-off.** C-prune was chosen because it met three requirements. E2 meets two. | See table below. | Ask the adviser now with the diagram, size table and a clear "these three rules": (1) extra layer at prediction time, (2) size above the 64-filter base (SOP 2), (3) title and RQ1 say "static gating". If refused, fall back (Part G). | Phase 0 |
| **D2** | **γ range.** | (a) `1 + Δγ`, unbounded (Perez). (b) `1 + tanh(Δγ)`, between 0 and 2 (artifact). (c) linear without the "1 +" (DAFT). | (a) as the default, (b) as an ablation. Perez found restricting γ hurt and 36% of γ were negative in their model; DAFT left γ free. The artifact's bound has no paper behind it. | Phase 2 |
| **D3** | **Keep β?** | γ and β / γ only. | Keep both (DAFT: dropping either hurts). β is also the only way the link can act when the text filters score near zero. Test γ-only in Phase 5. | Phase 5 |
| **D4** | **Generator input.** | 3 RF numbers + Conv average (artifact) / the 37 contributions instead of the 3 / no Conv average (design E). | The artifact's choice. The other two are ablations (Part E). | Phase 5 |
| **D5** | **How the Conv average is taken.** Padding positions (the median message is about 19 tokens of 50) are included in a plain average, so the average depends on message length; and it is taken before ReLU, so negatives can cancel positives. | Plain average over 48 / average over real-word positions only (mask) / average after ReLU. | Start with the plain average (matches the artifact); run the masked and post-ReLU versions as small ablations. | Phase 2, 5 |
| **D6** | **Embedding.** | ✅ Frozen for every model (decided 2026-10-18). | The 70,658 figure holds. | — |
| **D7** | **Training schedule.** | (a) Everything from scratch, together. (b) Warm-start the Conv from `phase_a_seed<S>_fold<F>.pt`, generator zero-start, train Conv + generator + head together. (c) Freeze the Conv, train generator + head only. | (b), with (a) as a check that warm-starting is not hiding anything. (c) is the cheap fallback if data is tiny. | Phase 3 |
| **D8** | **Training data.** | ✅ Crowd dataset (minimum 2,000, recommended 4,000) and the synthetic pilot until it exists. | Build and debug on the synthetic pilot (`crowd_pilot_synthetic.csv`); no result from it goes in the thesis. | — |
| **D9** | **Link RF numbers and calibration.** | Raw P, u from `link_lane` / Platt-calibrated P. | Raw P and u, exactly as UNGATED and C-prune get them, so only the architecture differs. Train rows read out-of-fold values; val and test read the final forest's. Calibrated P as a small ablation. **Never feed the final forest's P to training rows** (it has seen them, so P would be overconfident). | Phase 1 |
| **D10** | **Modality dropout in E2.** | ✅ Page hidden 0.25, text hidden 0.10, same for every model. | Page hidden = swap in the blank-page P, u and `page_missing = 1`. Text hidden = set the 64 pooled text numbers to 0 (as C-prune sets its k filter scores to 0). | Phase 3 |
| **D11** | **Messages with no link.** | ✅ None exist (URL messages only). | `has_url` stays out. | — |
| **D12** | **Models in the comparison.** | Plain CNN ✅, UNGATED ✅, CNN_CONCAT, current GATED, C-prune (k = 48, 32, 16), E2. | All of them on the same folds. Plain CNN and UNGATED are already built; the other baselines are still open in the C-prune checklist. | Phase 4 |
| **D13** | **Weight decay for the generator.** Dense 64 has 1e-4 in the shared recipe; the generator is new and the data is small. | Same 1e-4 / tuned on validation / none. | Start at 1e-4 on the generator too, tune on validation only. | Phase 3 |
| **D14** | **Statistics.** | ✅ 5-fold grouped CV, 3 seeds, exact McNemar, Holm-corrected. | — | — |
| **D15** | **Thesis wording.** | Re-title / keep the title and describe FiLM as dynamic gating / keep static gating as the baseline. | Decide with the adviser (D1). Keep the static-gated and C-prune models as baselines either way. RQ1's phrase "proposed RF-guided static gating framework" then needs to name E2. | Phase 8 |
| **D16** | **Serving and demo.** C-prune's own Phase 8 is also unstarted. | Migrate now / after trained on real data. | After real data, so a demo is never served from placeholder-trained weights. | Phase 7 |
| **D17** | **Can the inspector produce all 37 features at runtime?** | Yes / drop or replace features. | Confirm before the real run. The forest needs all 37; the generator only gets 3 numbers. | Phase 1 |
| **D18** | **Where the code lives.** `main` does not have the crowd pipeline; the branch does. | New branch from `feat/prune-architecture` / work on that branch / merge it first. | New branch off `feat/prune-architecture` (name for example `feat/film-architecture`), so C-prune stays intact as a baseline. Nothing is created or committed until the group says so. | Phase 1 |
| **D19** | **What happens to C-prune.** | Keep as a comparison model / drop. | Keep. The scripts exist and re-run on real data cheaply; "E2 vs C-prune at equal accuracy and cost" is the question the adviser will ask. | Phase 4 |

**D1 detail: the three requirements C-prune met.**

| Who | Requirement (from the C-prune plan) | C-prune | E2 |
|---|---|---|---|
| Panel | RF inside or replacing a CNN layer | Gate RF decides which filters exist | The Link RF sets γ and β on the Conv output: **met** |
| Group leader | RF must not be the classifier | Dense + Output decide | Dense 64 + Output decide: **met** |
| Adviser | No extra layer when an existing one can carry it | No new layer | **Not met**: the generator is one extra Dense layer run for every message |
| Size (SOP 2) | Smaller than the 64-filter base | 18,194–49,074 | **Not met**: 70,658 |
| Title | "Static gating" | static 0/1 mask | **Not met**: γ and β change per message |

---

# Part E. Possible changes (the menu)

Each is a variant to test, not a plan to adopt. "Evidence" says whether a paper I read backs it.

| ID | Change | What it does | Cost | Evidence |
|---|---|---|---|---|
| **E1** | γ only (no β) | Generator outputs 64 instead of 128. | 4,352 params, total 66,306 | Mixed: Perez 1.5% vs 0.5%; DAFT dropping β costs 0.013. |
| **E3** | Generator reads the 37 contributions instead of P, u, page_missing | More link detail reaches the generator. | +34 inputs × 128 = +4,352 params, and the slow contribution step returns | None. Our own ablation. |
| **E4** | Link-only generator (no Conv average) | The "E" design. Same link gives the same γ and β for all messages. | −8,192 params (64 inputs × 128) | DAFT vs FiLM comparison is the closest evidence. |
| **E5** | Drop u (tree disagreement) | Check whether u helps. | −128 params | No study found. |
| **E6** | **Hybrid: FiLM plus concatenating (P, u, page_missing) into the head** | Safety net: if FiLM alone cannot use the link when the words are neutral, the head still can. | +3 × 64 params | Not studied directly. Practical hedge, because concatenation is already strong. |
| **E7** | Masked or post-ReLU Conv average | Removes the length and negative-value effects (D5). | none | None. |
| **E8** | Hidden layer in the generator | 67 → 32 → 128: more flexible path, but γ and β pass through a bottleneck. | 6,400 params instead of 8,704 (−2,304) | DAFT uses a bottleneck MLP. |
| **E9** | FiLM after max-pool, γ only, γ ≥ 0 | Exactly the same result with 64 instead of 3,072 multiplies (math identity). | saves compute | Mathematical fact, no study needed; breaks with β or negative γ, so it conflicts with D2(a). |
| **E10** | Prune first (C-prune), then FiLM on the k kept filters | Smaller model. | extra training steps | Pruning papers not read, not cited. Only if size becomes the priority. |
| **E11** | Rule early exit (blocklists, known bank domains, raw IP, punycode) | Skips the page fetch for easy messages. | rules decide some verdicts | Untested idea. |
| **E12** | Lookup-table generator (design "E-lite") | Generator only in training. | possible accuracy loss | Untested idea. Only if D1 is refused. |
| **E13** | Bounded γ `1 + tanh` | The artifact's choice. Stable but cannot flip a filter. | none | Opposed by Perez (restricting γ hurt). |

---

# Part F. Phases and steps

Conventions: run Python with `.venv/Scripts/python.exe`. Seeds 42, 123, 2024. Report accuracy, precision, recall, F1 as mean ± std. Never use test folds to choose anything. Proposed commands follow the style of the existing scripts and **do not exist yet**. Commits only when the group asks.

## Phase 0. Lock the spec (about 1 day)

| Step | What it does | Done when |
|---|---|---|
| 0.1 | Answer D1, D2, D7, D18, D19 in this file (edit the table; add the date and who agreed). | Every row has a chosen option. |
| 0.2 | Send the adviser the diagram (with the corrected arrow), the size table (A5) and the D1 table. | Reply recorded here, or a decision to build and test first. |
| 0.3 | Write the E2 decisions into the C-prune plan's Phase 0 checklist so the two documents do not disagree. | Both files point to each other. |

## Phase 1. Shared base: data, link lane, Phase A (mostly done on the pilot)

E2 adds almost nothing here. The real work is waiting for crowd data (C-prune Phase 1 in its plan, unchanged).

| Step | What it does | Done when |
|---|---|---|
| 1.1 | Create the working branch (D18). Check the pilot files load: `data/processed/crowd_pilot_synthetic.csv`, `data/splits_crowd/`, per-fold vocabularies, `models/c_prune/crowd_pilot_synthetic/phase_a_seed<S>_fold<F>.pt` (15 files). | A short script lists seeds × folds present. |
| 1.2 | Read the E2 inputs from `link_lane.load_lane(seed, fold)`: P, u, `page_missing` for the full row and the blank-page version, train rows out-of-fold. Assert the ids line up with the text rows (as `fusion_inputs` does). | Arrays of shape (n,) for each, ids asserted equal. |
| 1.3 | Time the Link RF twice: with contributions (as now) and with only P and u (`tree_probabilities`). This gives the measured size of the single-forest, 3-output saving (A5). | Two timings per message, batch 1, 1 thread, after 50 warm-ups. |
| 1.4 | Resolve D17 (inspector produces all 37 features live). | Answer recorded. |

## Phase 2. Model code and unit tests (about 2-3 days)

Goal: a `FiLMTextCNN` that is correct before it is trained.

| Step | What it does | Done when |
|---|---|---|
| 2.1 | New `src/features/model/film_cnn.py`: frozen embedding, `Conv1d(300, 64, 3)`, generator `Linear(67, 128)` with **weight and bias zero-initialised**, FiLM step, ReLU, max-pool, dropout 0.2, then `Dense 64 → ReLU → dropout 0.5 → Output 2`. Keep a method that returns x and the Conv average so they can be inspected. Include a `text_keep` switch for modality dropout (zero the 64 pooled numbers), as `PrunedFusionClassifier.forward(…, text_keep)` does. | Class imports; forward returns logits (batch, 2). |
| 2.2 | γ rule from D2 with an option for the bounded version. Optional masked average (D5). Provide `parameter_groups(weight_decay)` like `PrunedFusionClassifier` (decay on Dense 64 and, per D13, the generator). | Shapes verified. |
| 2.3 | `tests/test_film.py`: (a) at initialisation the output equals the same network with the generator removed (γ = 1, β = 0); (b) trainable parameters = exactly 70,658 with the frozen embedding, and 66,306 for γ only; (c) after one optimiser step the generator weights have a non-zero gradient; (d) changing only the link numbers changes γ and β once the generator is non-zero; (e) with the bounded option γ is between 0 and 2; (f) if E9 is built, "γ after max-pool" equals "γ before ReLU" for γ ≥ 0 and β = 0; (g) `text_keep = 0` zeroes the head input. | `pytest tests/test_film.py` passes. |
| 2.4 | Winning-window explanations: a method like `encode_with_positions` so each filter's strongest 3-word window can be named, plus γ and β per filter ("the link raised these filters"). | Test: same pooled values as `forward`. |

## Phase 3. Training (about 2 days)

| Step | What it does | Done when |
|---|---|---|
| 3.1 | New `src/unified/film.py`, modelled on `fusion.py`: `build`, `train`, `check`, `save`, `load_film`, `predict_messages` (raw rows → inspector numbers → Link RF → model, as the app will), with `--pilot` and `models/film/<dataset>/` outputs plus sidecars with hashes. Proposed: `python -m src.unified.film --pilot build`. | Builds all 15 seed-fold models; parameter count equals A5. |
| 3.2 | Warm start (D7b): load the Conv from the fold's Phase A checkpoint through `text_cnn.load_phase_a` (the old temporary head is dropped). Generator zero-start. Dense 64 and Output new, seeded. | Check: at epoch 0 the pooled text features equal Phase A's pooled features (generator is the identity). |
| 3.3 | Train with the shared recipe (`fusion.fine_tune` pattern): Adam 1e-3, up to 30 epochs, patience 3 on val balanced accuracy, label smoothing 0.1, class weights, weight decay on Dense 64 (+ generator, D13). Modality dropout as D10, same random draws as the other fused models. Train rows read out-of-fold link numbers; val reads the final forest's. Proposed: `python -m src.unified.film --pilot train`. | `train_summary.csv` with val metrics, epochs, training seconds (RQ2). |
| 3.4 | Log how far γ and β move from 1 and 0 per epoch, and for each filter. | Statistics in the log and summary. |
| 3.5 | `python -m src.unified.film --pilot check`: rebuild every model from its sidecar alone and repeat its val metrics exactly. | Gap 0 on all 15. |

## Phase 4. Compare with the baselines (about 2 days)

| Step | What it does | Done when |
|---|---|---|
| 4.1 | Add E2 to `baselines.py` so `python -m src.unified.baselines --pilot compare` scores it with plain CNN, UNGATED and C-prune on the same test folds (ids asserted identical). | One predictions file with every model. |
| 4.2 | RQ1: accuracy, precision, recall, F1 per seed, mean ± std (smishing positive). | `test_summary.csv`. |
| 4.3 | RQ2: trainable parameters counted in code (compare with A5); latency per stage (stage 3 Link RF with P and u only for E2; stage 4 text model + generator); training time per step. Median, 95th percentile, mean; batch 1, CPU, 1 thread, 50 warm-ups. The web inspection is shared and not timed on the pilot. | `test_parameters.csv`, `test_latency.csv`. |
| 4.4 | RQ3: exact McNemar per seed, E2 against each baseline, Holm-corrected: E2 vs plain CNN, vs UNGATED, vs C-prune, vs CNN_CONCAT (once built). | `test_mcnemar.csv`. |
| 4.5 | Build the missing baselines (CNN_CONCAT, current GATED) once for both architectures, if the group wants them (D12). | Models and rows present. |

On the synthetic pilot these numbers prove the pipeline only. They are a rehearsal, as the C-prune plan already says for its own pilot.

## Phase 5. Ablations and controls specific to E2

Each is one more model on the same folds.

| Step | Ablation | Question |
|---|---|---|
| 5.1 | E2 vs plain text CNN | Does the link help at all through γ and β? |
| 5.2 | E2 vs UNGATED / CNN_CONCAT | Does FiLM add anything over concatenation? (the key comparison) |
| 5.3 | E2 vs C-prune (k = 48, 32, 16) | Is E2 better at its extra size and extra layer? |
| 5.4 | E2 vs E4 (no Conv average) | Does reading the text summary help? |
| 5.5 | γ and β vs γ only (E1) | Is β needed? |
| 5.6 | γ rule: `1 + Δγ` vs bounded (D2) | Does bounding γ hurt? |
| 5.7 | Generator input: 3 numbers vs 37 contributions; with and without u | Which link numbers matter? |
| 5.8 | Hybrid E6 (FiLM + concat of the 3 numbers) | Does the head need direct link access? |
| 5.9 | Conv average variants (D5, E7) | Is the padding / sign effect real? |
| 5.10 | **Link-shuffle control:** at test time, permute the link numbers across messages. If accuracy barely changes, the model is not using the link. | Does E2 really use the link? |
| 5.11 | **Neutral-wording stress test** (rows whose wording is shared by both classes; the synthetic pilot has these; the real data needs a flag). Concatenation can use the link directly; E2 must do it through β. | The case where FiLM is most likely to lose. |
| 5.12 | Page hidden: `page_missing = 1` versus 0 on the same messages. | Behaviour when the page is dead. |
| 5.13 | From-scratch (D7a) vs warm start (D7b). | Is the warm start hiding anything? |
| 5.14 | γ and β diagnostics: how many filters move, mean \|γ − 1\| and \|β\| by class, correlation of the Conv average with P, which filters change most. | Table or figure for the thesis. |

## Phase 6. Real-data run (when the crowd dataset exists)

| Step | What it does | Done when |
|---|---|---|
| 6.1 | Rebuild the dataset, splits, embeddings, link lane and Phase A with the existing scripts on `crowd_smishing_v1.csv`. | Same checks as the pilot pass. |
| 6.2 | Fix every choice on **validation only**: D2, D3, D5, D13, learning rates, any generator input choice. Write them into this file with the date and numbers, once. | Choices frozen before any test read. |
| 6.3 | Learning curve: 25 / 50 / 75 / 100% of the training part. | Plot or table. |
| 6.4 | Read the test folds **once** (`--read-test`), logged in `test_reads.log`. Fill the RQ1-RQ3 tables. | Tables complete. |
| 6.5 | Write the verdict by the stop rules in Part G. | Section added to the thesis notes. |

## Phase 7. Serving and demo (after Phase 6)

Not started on purpose (D16). I have not read `src/serving/*` or `src/demo/*` closely; read them first.

| Step | What it does | Done when |
|---|---|---|
| 7.1 | `film.load_film(seed, fold)` rebuilds the model and Link RF from a sidecar. | Round-trip test. |
| 7.2 | `lejet_service.py` and `predict.py`: inspect link, Link RF (P, u), FiLM CNN, verdict. Keep the C-prune path behind a flag until E2 is checked. | Serving tests pass. |
| 7.3 | `explain_text.py`: winning 3-word windows plus "link evidence raised these filters". | Response includes reasons. |
| 7.4 | Remove `has_url` from the live vector if still present. | Tests pass. |

## Phase 8. Documents and thesis

| Step | What it does | Done when |
|---|---|---|
| 8.1 | Add this file to `docs/README.md` as the current design; add "Superseded by `film_architecture_plan.md`" lines to the C-prune plan, `unified_architecture.md` and `final_model.md`. | Index and banners updated. |
| 8.2 | Update `docs/progress.md` section 1 ("the design in five lines") to E2. | Section rewritten. |
| 8.3 | Thesis wording (D15), chapter 3 text for γ, β and the Conv average, the corrected diagram. | Adviser-approved wording. |
| 8.4 | Limitations: crowd data size, scarcity of legitimate links, evidence only from other tasks, extra layer and size over the base. | Section written. |

**Claims you can make once built, and claims you cannot:**
- Can: the architecture trains end to end; its size and compute (arithmetic); the identity start (unit test); any result from Phase 4-6, labelled pilot or real accordingly.
- Cannot (yet): that FiLM beats concatenation; that E2 helps on text (no read study shows this); that conditioning on a link works like conditioning on tabular data in medical imaging; that E2 is faster than C-prune (not measured).

---

# Part G. Risks, stop rules and fallbacks

| Risk | Why it matters | What to do |
|---|---|---|
| Extra layer at prediction time | Adviser's rule. | D1. Fallback E12 (lookup table, generator only in training) or return to C-prune. |
| Larger than the base and than C-prune | SOP 2 and RQ2 on size. | Argue on accuracy per size and on latency; show γ-only (66,306); count parameters from code. |
| Title says static gating | E2 is per message. | D15. Keep C-prune / static-gated as baselines. |
| Evidence is from other tasks | Perez: images. DAFT, HyperFusion: brain MRI + tabular. Zayed: metaphors, conditioning from the same sentence. | Phase 5 is the real test. Do not quote the studies as results about Le-Jet. |
| FiLM cannot use the link when wording is neutral | The head never sees the link; only β can raise a filter that scores near zero. | Step 5.11; fallback E6 (hybrid). |
| Static gates showed no gain over shuffled gates | Warning that cleverness on top of a strong baseline may add nothing. | Steps 5.1 and 5.2 decide. Report a tie honestly. |
| Small data: 2,000 messages vs about 70K parameters | Overfitting, noisy comparisons. | Grouped CV, weight decay, warm start, early stopping, E1 (γ only), learning curve. |
| Synthetic pilot | Its text is invented, it is very redundant (the C-prune pilot noted any 16 filters do the job). | Never report pilot scores as results. |
| Training leak | The final forest has seen train rows; its P on them is overconfident. | Out-of-fold P and u for train rows (D9). |
| Conv average depends on padding and sign | Skews the generator input (D5). | E7 ablation. |
| γ range | A bounded γ may block what FiLM learned (36% negative in Perez). | D2, step 5.6. |
| The plan sits on a different branch from `main` | Easy to lose track of which code is current. | D18. |

**Stop rules (decide in advance):**
- If E2 does not beat the plain text CNN: report the plain text CNN as the model and E2 as a tested negative result.
- If E2 ties UNGATED / CNN_CONCAT: say so; recommend the simpler model, and describe FiLM as an explored alternative.
- If E2 ties or loses to C-prune while being larger and slower: C-prune is the proposed model, E2 a tested alternative.
- If E2 beats the baselines and C-prune with a clear margin on complete **and** page-hidden messages and the link-shuffle control (5.10) lowers accuracy: adopt E2 and write the claims allowed above.

---

# Appendix A. Illustrative code sketch (not the implementation)

```python
class FiLMTextCNN(nn.Module):
    def __init__(self, embedding, n_filters=64, kernel=3, link_dim=3):
        super().__init__()
        self.embedding = embedding                       # frozen (decided 2026-10-18)
        self.conv = nn.Conv1d(300, n_filters, kernel)
        self.gen = nn.Linear(link_dim + n_filters, 2 * n_filters)
        nn.init.zeros_(self.gen.weight); nn.init.zeros_(self.gen.bias)   # gamma = 1, beta = 0 at the start
        self.drop_pool = nn.Dropout(0.2)
        self.head = nn.Sequential(nn.Linear(n_filters, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2))

    def forward(self, token_ids, link, text_keep=None):          # link: (B, 3) = P, u, page_missing
        x = self.conv(self.embedding(token_ids).transpose(1, 2))  # (B, 64, 48)
        avg = x.mean(dim=2)                                       # Conv average (B, 64)
        d_gamma, beta = self.gen(torch.cat([link, avg], 1)).chunk(2, dim=1)
        gamma = 1 + d_gamma                                       # D2 default; 1 + tanh(d_gamma) is the bounded option
        x = gamma.unsqueeze(2) * x + beta.unsqueeze(2)
        pooled = torch.relu(x).max(dim=2).values
        if text_keep is not None:                                 # modality dropout: hide the text numbers
            pooled = pooled * text_keep.unsqueeze(1)
        return self.head(self.drop_pool(pooled))
```

# Appendix B. Papers read, and how

| Paper | Read via | Checked |
|---|---|---|
| Perez et al. 2018, arXiv 1709.07871 | HTML (ar5iv) and PDF text | Eq. 1-2, Sections 4.2-4.3, Table 2, Appendix 7.2 |
| Pölsterl et al. 2021, arXiv 2107.05990 (v1) | HTML and PDF text | Section 3, Tables 1, 2 and 4 |
| Duenias et al. 2024, arXiv 2403.13319 (v3) | HTML (start) and PDF text | Table 3; ADNI use |
| Zayed et al. 2020, arXiv 2010.05633 | HTML and PDF text | Section 3 method, Table 1 |
| Kim et al. 2019, TACL Q19-1013 | ACL page and PDF text | Abstract, introduction, results discussion |

I did not compare the arXiv versions with the conference proceedings versions.

# Appendix C. Files read for this plan

On `main`: `docs/final_model.md`, `docs/unified_architecture.md`, `docs/progress.md`, `src/unified/run_fused_lean.py`, `src/unified/run_single_rf.py`, `src/features/model/cnn_encoder.py`, `src/config.py`.
On `feat/prune-architecture` (read-only through git, no checkout): `docs/c_prune_architecture.md` in full, and the module docstrings and function lists of `src/unified/text_cnn.py`, `link_lane.py`, `fusion.py`, `baselines.py`. The bodies of those functions were **not** read, so Phase 1-4 steps that name them (for example the exact shape returned by `load_lane`) must be checked when coding starts.
