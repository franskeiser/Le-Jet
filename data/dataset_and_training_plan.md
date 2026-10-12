# Dataset and Training Plan
**Thesis:** A Random Forest-Guided Static Gating Framework for Financial-Related Smishing Classification Using SMS Text
**Topic of this plan:** which data to use, what each dataset is for, and how to train, with website inspection added
**Written:** 2026-10-10

Items marked **[verified]** were read in a source during this work. Items marked **[not verified]** come from search-result titles or my own knowledge. Check those before you cite them.

---

## 1. The short version

1. Your 18,885-message SMS set trains the CNN and the head. Keep the text-only messages as well as the ones with links.
2. A separate public set of 11,430 web pages (already converted to features) trains the RF extractor. The extractor never sees your SMS data.
3. The extractor is then frozen. For every SMS link, it turns page and URL features into two numbers: P (how phishing-like the page looks) and u (how much the trees disagree). Messages with no link get -1 for both.
4. Most links in older SMS data are probably dead. Test whether visiting pages helps on live links only, using a small fresh set you collect and snapshot yourselves.
5. Do all page visits from an isolated machine, observe only, with hard limits.

```
Public web set (11,430 pages) ──> train 2 RF extractors (url_only, full) ──> freeze

SMS link ──┬─ old SMS data ──────────────────────────> url_only ──┐
           └─ fresh link ──> inspector ─┬─ page loaded ──> full ──┤
                                        └─ page failed ──> url_only ┤
                                                                  v
                          P, u + 31 per-feature contributions (0 where a feature was not read;
                          P = u = -1 when there is no link)
                                                                  │
SMS text ──> fastText 300-d ──> CNN ──────────────┐               │
has_url ──────────────────────────────────────────┼─> gates <─────┘
                                                  v
                                          dense ──> softmax
CNN gates (0.1 to 3.0) come from a 200-tree forest fit on the RF-Fitting Set, then discarded.
The 31 RF-output slots are gated by the full forest's own permutation importances (public web set).
has_url, P and u get a fixed gate of 1.0.
```

---

## 2. Datasets and their jobs

| # | Dataset | Size | Job | Status |
|---|---|---|---|---|
| A | Combined SMS set (your Table 1) | 18,885 (10,955 smishing, 7,930 legitimate) | Train the CNN and head; the RF-Fitting Set sets the gates; the Test Set measures everything | [verified] from your thesis |
| B | Public web phishing set (Hannousse & Yahiouche) | 11,430 pages, half phishing, half legitimate, 87 features | Train and test the RF extractor only | [verified] |
| C | Fresh collected set | You decide (aim for a few hundred links) | Test inspection on live pages | To be collected |
| D | ~~Tranco Top 1M snapshot~~ | — | **Removed (2026-10-11).** Domain popularity is no longer a model input | — |
| E | Optional: Smishtank | 1,062 smishing messages | More smishing with link fields (final URL, VirusTotal, WHOIS) | [verified], license not stated |
| F | Optional: legitimate messages with links | to be collected | Fix the shortage of legitimate messages with links | Needs ethics sign-off |

### Notes on each

**A. Combined SMS set.** Sources and final counts from your Table 1:
NCSU 2,902 smishing; Fishing for Smishing 6,320 + 433; Mishra and Soni 66 + 4,820; Philippines Spam and Marketing 423 + 863; Philippine Spam/Scam 504 + 23; SMS Spam Collection 5 + 384; Filipino Spam 127 + 292; ISDA 608 + 1,115.
- Keep the text-only messages. Real phones get both kinds, and the -1 coding is part of your design.
- The split into Training Set, RF-Fitting Set (10 to 15%) and Test Set stays as in your thesis.
- When you split the messages that have links, group by domain so the same scam domain is never in both Training and Test.
- Unknown today: how many of the 18,885 contain a link, and how many are legitimate. Phase 0 answers this.

**B. Public web set.** The Hugging Face copy `husseinnashed/phishing-url` duplicates the original Mendeley release (doi:10.17632/c2gw7fy2j4.2) and is CC-BY-4.0. Cite Hannousse and Yahiouche (2021). Phishing URLs came from PhishTank and OpenPhish, legitimate ones from Alexa and Yandex. [verified]
- It has no SMS text, so it can never be mixed into set A.
- It is already split randomly 2/3 and 1/3 on Hugging Face. Use the combined file and make your own split.

**C. Fresh set.** Recent smishing messages with links, plus recent legitimate messages with links. See section 8 for how to collect and snapshot it.

**E. Smishtank.** The paper (Timko and Rahman, CODASPY 2024) lists the sender, message body, brands, a message category, the final URL destination, and for the first link the VirusTotal results and WHOIS dates. The paper states no license, so check smishtank.com/dataset before use. [verified]

**F. Legitimate messages with links.** I found no public dataset of these. The SMS Spam Collection has almost none (one paper's table implies about 13 of 4,807, my estimate). The IMC 2025 smishing dataset (33,869 messages) did not collect legitimate messages. [verified] I am assuming it is the same data as the "Fishing for Smishing" row in your Table 1, because the message count (33,869) matches; confirm that. Options: volunteers sharing real bank, e-wallet, delivery and telco messages with personal details removed; or a small set of templated messages built from real notification formats (label it synthetic).

---

## 3. Phase 0: Audit your SMS data (do this first)

Nothing else should start until you know these numbers.

1. Run `extract_links.py` on the combined SMS file. It makes no network calls. It finds links with `http://`, with `www.`, with no scheme (like `qinpwhe.com/xA9dXLMD`), and disguised ones (`hxxps://site[.]top`).
2. Read `link_summary.txt`. Record, per class: messages, messages with a link, unique domains, links through shorteners.
3. **Shortcut check.** If most legitimate messages have no link and most smishing messages do, the model can learn "has a link means smishing" and ignore the link itself. If the gap is large, do step 5 below before trusting any with-link result.
4. **Liveness sample.** Resolve about 200 unique domains per class using DNS only. Record the share that still resolves. This is your expected dead rate, measured, not guessed.
5. If legitimate messages with links are too few, add some from sources F before training the with-link experiments.

**Done when:** you have a one-page table of link counts and the DNS liveness rate per class.

---

## 4. Phase 1: Prepare the public web set

1. Convert the Parquet files to one CSV (`pandas`, `pyarrow`, `huggingface_hub`) and check the shape is about 11,430 rows and 89 columns.
2. Map the label: `status` "phishing" to 1, "legitimate" to 0.
3. Keep `url` in the CSV for traceability, but do not feed it to the model (an identifier, not a feature).
4. Keep 32 features: 26 from the source and 6 new ones computed from the `url` text. This is the column set in `data/processed/web_phishing_clean.csv`, produced by `src/data/clean_web_phishing.py`. See `docs/web_phishing_cleaning.md` for each definition and the source-column mapping.

| Group | Features | Reason |
|---|---|---|
| URL structural (16) | `url_length`, `https_presence`, `number_of_hyphens`, `number_of_slashes`, `subdomain_count`, `special_char_count`, `suspicious_url_token_count`, `shortened_url_indicator`, `has_ip_address`, `suspicious_keyword_count`, `has_url`, `page_rank`, `path_length`, `random_domain`, `length_hostname`, `tld_in_path` | Computed from the link string, free and safe. `https_presence`, `special_char_count`, `suspicious_url_token_count`, `suspicious_keyword_count` and `path_length` are new and recomputed from `url`, so the same code works on SMS links. `has_url` is always 1 here; it matches the SMS feature set. `page_rank` comes from Open PageRank. `path_length` has no literature reference yet. `tld_in_path` is the dataset's spelling (the feature list's `tld_in_pth` is a typo) |
| Webpage (12) | `nb_hyperlinks`, `ratio_intHyperlinks`, `nb_extCSS`, `login_form`, `external_favicon`, `links_in_tags`, `ratio_intMedia`, `ratio_extMedia`, `iframe`, `empty_title`, `domain_in_title`, `domain_with_copyright` | From one page's HTML |
| External (4) | `whois_registered_domain`, `domain_registration_length`, `domain_age`, `dns_record` | Cheap lookups. Negative `domain_age` and `domain_registration_length` (WHOIS failures) are set to -1 |
| Drop (61) | All other source columns. These include `ratio_nullHyperlinks` and `sfh` (0 in every row), the separate character counts (`nb_dots`, `nb_at`, `nb_qm`, …), the word-length stats, `ratio_extHyperlinks`, `submit_email`, `popup_window`, `safe_anchor`, `onmouseover`, `right_clic`, `web_traffic`, `google_index`, `statistical_report`, `ratio_intRedirection`, `ratio_extRedirection`, `ratio_intErrors`, `ratio_extErrors`, `nb_redirection`, `nb_external_redirection`, and `phish_hints` and `https_token` (replaced by `suspicious_url_token_count` and `https_presence`) | Not in our feature list. Some feed the new features |

5. **Check the definitions against the paper's code.** If the content features were built from static HTML, your inspector can fetch plain HTML with no JavaScript, which is safer than a full browser. If not, you need a browser.
6. **Scheme normalization.** Features like `https_token` and `nb_www` assume `http(s)://`. Add a scheme to every SMS link before computing features.
7. **Overlap check.** List domains that appear in both the public set and your SMS links. Report the overlap. Remove or flag shared domains in your evaluation.

**Done when:** you have a clean CSV with 32 features, a label column, and a short note on each feature's definition. Done: `data/processed/web_phishing_clean.csv` (11,430 rows, 5,715 per class) and `docs/web_phishing_cleaning.md`.

---

## 5. Phase 2: Train the RF extractor

1. **Split** the public set 70/15/15 (train, validation, test). Group by domain if one domain appears more than once.
2. **Model:** random forest. Start with 200 to 500 trees and tune on the validation part only. Suggested starting points, not thesis facts.
3. **Two extractors instead of masking (decided 2026-10-11, Option C in `docs/decision_web_inspection.md`).** Phase 0 found that 82% of old smishing domains no longer resolve, so anything looked up today mostly says "the domain died", not "this is a scam". Train two extractors on the public set:
   - `url_only`: the 14 link-text features (no `page_rank`). Used for every old SMS link, and for fresh links whose page failed.
   - `full`: all features, trained on complete pages. Used only for fresh links whose page was fetched within hours.
   No masking is needed, because each extractor only sees the kind of data it was trained on.
4. **Outputs per link:**
   - P: the average of the trees' phishing probabilities.
   - u: the standard deviation of the individual trees' probabilities (tree disagreement).
   - Both are -1 for messages with no link.
5. **Shortcut ablations.** Train four versions of `full`: all features; without the path and length features (`url_length`, `path_length`, `number_of_slashes`, and similar); without domain age; without `page_rank`. If scores fall sharply, the model was leaning on a dataset artifact (for example, legitimate links being mostly site homepages; this is a suspicion, not confirmed).
6. **Report:** accuracy, precision, recall, F1 and a calibration check on the held-out public test part.
7. **Trim the feature list.** Compute permutation importance on the validation part and keep the top 25 or so. Retrain with only those. Your inspector then has fewer features to compute and less contact with live pages.
8. **Freeze:** save with `joblib`. From here on the extractor is never retrained on SMS data.

**Done when:** you have a saved extractor, its held-out scores, the ablation table, and a final feature list.

---

## 6. Phase 3: Build the inspector (safe)

Two tiers. Go to the second only for links that still need it.

**Tier 1: no page contact.** URL structure, DNS, RDAP (domain age, registrar), certificate-log lookup. Run for every link.

**Tier 2: isolated visit.**
1. Use a throwaway container or VM, not your own computer. Non-root, no personal logins, no shared folders, CPU, memory and time limits. Destroy it after each batch.
2. Decide the network with your adviser. Cloud addresses are sometimes shown a blank page (log as "blocked"). A home or campus address shows your location to the attacker.
3. Block internal addresses (10.x, 192.168.x, 127.x, 169.254.x). Resolve the name yourself, check the address is public, and re-check at every redirect. Add a firewall rule too. [from general practice, not verified against OWASP or Playwright docs]
4. Observe only: no typing, no clicking, no downloads, deny camera, location and notifications, close popups.
5. Limits: 15 to 30 second timeout, at most 10 redirects, page size cap, at most one retry, one link per container, rate limit per domain.
6. Record: page status (live, redirected, dead, blocked, timeout), redirect count, final domain, then the chosen content features.
7. Save each page's HTML as inert text with a hash. Never open saved pages in a browser on your own computer.
8. Pilot with 100 links, then scale. Example only: 5,000 links at 10 seconds each across 5 containers is about 2.8 hours.

**Confirming a link is really phishing without visiting.** Ladder: no-contact checks; threat feeds (URLhaus, PhishTank, OpenPhish, VirusTotal); a private scan on a third-party service such as urlscan.io; your own isolated visit last. Count a link as confirmed only with two independent signals. Reviewers look at screenshots, never at live pages. Keep the dataset label as the ground truth; do not relabel a message because its page was dead or looked clean.

**Done when:** the pilot of 100 links runs without incident and the saved features match the public set's definitions.

---

## 7. Phase 4 and 5: Build features and train the main model

### 7.1 Features per SMS
- Text: fastText 300-d fine-tuned embeddings into the single-kernel CNN: one conv branch, kernel size 3, 64 filters (decided 2026-10-12; `KERNEL_SIZE` and `NUM_FILTERS` in `src/config.py`), giving a 64-dim text vector.
- Link information reaches the head only through the RF extractor: the link-text features are read by the forest (together with the page, domain and external features when a page was reached), so the head gets `has_url`, P, u and the 31 per-feature contributions c_<feature> (0 where a feature was not read; P = u = -1 without a link). The head never sees the raw link features (decided 2026-10-12, `docs/final_model.md`).
- From the frozen extractors: P and u (-1 without a link) and the per-feature breakdown. Old SMS data always uses `url_only`; fresh links use `full` when the page loaded and `url_only` when it failed (`route_vector`). Page status is only used to pick the extractor, never as a model input.

### 7.2 Gates
1. CNN features: fit a 200-tree forest on the RF-Fitting Set only, compute permutation importance, scale to 0.1 to 3.0.
2. RF-output slots: gate each of the 31 contributions by that feature's permutation importance in the `full` extractor (validation part of the public web set), scaled to 0.1 to 3.0. The old SMS data has no page features to fit these on.
3. P, u and `has_url` get a fixed gate of 1.0.
4. Discard the gating forest. The extractor stays.

### 7.3 Training
1. Train the CNN, gates and head on the Training Set only. Hold out part of the Training Set for early stopping. Never look at the Test Set while tuning.
2. Run 3 random seeds (42, 123, 2024, `SEEDS` in `src/config.py`; the splits and extractors exist for these). Report mean and spread.
3. Report training time, trainable parameters and inference latency.

### 7.4 Experiment rows

| Row | Model | Question it answers |
|---|---|---|
| 1 | CNN only | Baseline |
| 2 | CNN + URL features, no gating | Baseline |
| 3 | KNN-RF-ETC voting classifier + URL phishing classifier | Baseline from your thesis |
| 4 | Proposed model (gating, no inspection) | Your thesis result |
| 5 | Row 4 + inspection features directly in the head | Do page features help on their own? |
| 6 | Row 4 + extractor P and u (extractor sees URL features only) | Does the extractor help without the page? |
| 7 | Row 4 + extractor P and u (extractor sees URL + page features) | The full proposal |
| 8 | Row 7 with a small neural network in place of the RF extractor (P only) | Is the RF specifically needed? |
| 9 | Row 7 with the extractor trained on your own SMS-link pages instead of the public set | Does training on the public set cost accuracy? (needs Phase 3 visits) |

Compare each row with row 4 using McNemar's test at 0.05.

---

## 8. Fresh set protocol

1. Collect recent smishing messages (your own phones, volunteers, forums). Remove personal details. Get consent and ethics sign-off.
2. Also collect recent legitimate messages that contain links.
3. Within hours of receipt, run Tier 1 and Tier 2 and write every feature to a timestamped CSV. After that the link may die and the row remains complete.
4. Keep this set out of all training. It is for testing only.
5. Record the time between receipt and visit for each link.

---

## 9. Evaluation

- Report results in three groups: messages with no link, links with a dead or blocked page, links with a live page.
- Report the share of links that were alive. This is a finding, not a failure.
- **Page status caution.** Scam links die faster than legitimate ones. In old data, "dead" can act as a label shortcut. Report every result with and without the page status feature.
- Metrics: accuracy, precision, recall, F1, trainable parameters, training time, and latency in two parts: model time (ms) and inspection time (s).
- Claim an inspection effect only for the live-link group.
- Report the extractor's own scores on the public test part, then on your SMS links, to show how much it drops.

---

## 10. Safety and ethics checklist

- [ ] No link opened on a personal device
- [ ] Inspector runs only in an isolated, disposable environment
- [ ] Internal addresses blocked at the browser and the firewall
- [ ] Observe only: no typing, clicking, downloads or permissions
- [ ] Time, size, redirect and rate limits set
- [ ] Saved pages stored as inert text
- [ ] Volunteer messages: consent, personal details removed, storage plan
- [ ] Private (not public) scans if a third-party scanner is used
- [ ] Adviser or ethics reviewer approved the visiting steps
- [ ] Terms of any third-party service (VirusTotal, urlscan.io) read before use

---

## 11. Risks and open questions

| Risk or question | What to do |
|---|---|
| Most SMS links are dead | Measure it (Phase 0); test on the fresh set; report coverage |
| "Has a link" or "page is dead" acts as a label shortcut | Check in Phase 0; report with and without |
| Public set looks different from smishing pages (PhishTank pages vs smishing, Alexa sites as legitimate) | Treat P as one input; run the row 9 comparison; report the drop |
| Public set may have path and homepage artifacts | Ablation in Phase 2 |
| Features recomputed by you may not match the paper's definitions | Read the paper's feature table and code before building the inspector |
| Too few legitimate messages with links | Add from sources F; label synthetic data |
| Smishtank license is not stated | Check the download page before using |
| Why Google Safe Browsing was removed | Your group still needs to write the reason in the doc |
| Kernel size and filter count for the single-kernel CNN | Decided 2026-10-12: kernel 3, 64 filters |
| Time between message and visit | Record it; shorter is better |

---

## 12. Thesis text that needs to change

- The statement that the RF is not used at inference (the RF extractor now runs at inference).
- The scope statement excluding HTML and JavaScript inspection.
- The multi-kernel CNN literature argument and any "192 features" mentions.
- Every mention of Google Safe Browsing (Phase 1, the Scope hard-bypass sentence, Ethics, the DSmishSMS comparison, Theme 8).
- Table 1 and the data chapter: add the public web set as a separate, non-SMS source.

---

## 13. Sources

**Verified (read in this work):**
- Hannousse and Yahiouche, web phishing dataset paper: https://arxiv.org/pdf/2010.12847
- Hugging Face copy of the dataset (CC-BY-4.0): https://huggingface.co/datasets/husseinnashed/phishing-url
- Smishtank dataset paper (Timko and Rahman, CODASPY 2024): https://arxiv.org/html/2402.18430v2
- IMC 2025 smishing dataset paper (Agarwal et al.): https://discovery.ucl.ac.uk/10214522/1/Agarwal_IMC_2025_Smishing.pdf and https://github.com/reportsmishing/Smishing-Dataset-IMC25
- PhiUSIIL phishing URL dataset on UCI (235,795 URLs, 54 features, CC BY 4.0): https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Machine Learning Driven Smishing Detection Framework (SMS Spam Collection use): https://arxiv.org/pdf/2412.09641
- Look Before You Leap, URL and HTML phishing detection: https://arxiv.org/pdf/2011.04412

**Not verified (found in search results or from my knowledge):**
- Playwright Docker documentation: https://playwright.dev/docs/docker
- OWASP Proactive Controls C10, SSRF: https://owasp.org/www-project-proactive-controls/v4/en/c10-stop-server-side-request-forgery
- PhishFarm (cloaking): https://sefcom.asu.edu/publications/phishfarm-oakland2019.pdf
- RFC 9083, RDAP JSON responses: https://www.rfc-editor.org/rfc/rfc9083.html
- urlscan.io API and FAQ: https://urlscan.io/docs/api/ and https://urlscan.io/docs/faq/
- State of the Art: Content-based and Hybrid Phishing Detection: https://arxiv.org/pdf/2101.12723
- SMS Phishing Attacks and Defenses: A Systematic Review (dataset section not read): https://arxiv.org/pdf/2604.11429
- Smishing Data Hub: https://github.com/MarazMia/SMISH_DT
- Grinsztajn et al., 2022, tree-based models on tabular data (NeurIPS) and Le Pochat et al., 2019, Tranco (NDSS): cited from knowledge, not searched
