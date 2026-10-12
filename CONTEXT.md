# Le-Jet — Context: what this repo is, why it exists, what changed from SmishGuard

> **Read this first.** It explains where Le-Jet came from, how it relates to the **SmishGuard** repo, and what is different. It is written so a new teammate or a fresh AI session can get oriented without having seen the earlier conversations.
>
> **How to read the claims in this file:** statements marked **[stated]** were said by the project owner; **[observed]** were read from the SmishGuard repo or this machine; **[inferred]** are my reasoning and should be confirmed or corrected. Nothing here has been executed yet.
>
> Last updated: 2026-10-11. The feature scope, the repo state and the role of the inspector were updated; the SmishGuard description in §2 is as of 2026-10-10.

---

## 1. One-paragraph summary

**SmishGuard** (existing repo) classifies a pasted SMS as *smishing* or *legitimate* using a CNN over the message text plus a small set of **lexical URL features** and a **Google Safe Browsing** lookup. It never actually visits the link. **Le-Jet** (this repo, started from scratch) keeps the same overall goal but moves the model's evidence to **web inspection**: safely visiting the real URL from an isolated lab and extracting features from what is really there (the page's HTML, the domain's registration and DNS data) — **[stated]**: "it's still the same, but our features should be on the web inspection." The first goal is simply to **prove web inspection works** on a few URLs before scaling to 10,000+.

---

## 2. What SmishGuard is (the repo this one grew out of)

Location on this machine **[observed]**: `C:\Users\Franz\OneDrive\Desktop\Projects\irs-db\SmishGuard` (GitHub: `franskeiser/SmishGuard`).

| Aspect | SmishGuard today **[observed]** |
|---|---|
| Task | Binary classification of SMS text: smishing vs legitimate; bilingual context (English/Filipino); peso amounts and Philippine-style scams appear in the scope logic |
| Model | CNN text encoder; a Random Forest is fitted once offline only to rank feature importance, and the resulting **frozen "gate" multipliers** re-weight the URL/structural features; fused into one neural classifier. Multi-seed training (42, 123, 2024) |
| URL features | Roughly a dozen **string-only** features computed from the link text (`src/features/url_features.py`: length, hyphens, slashes, suspicious keywords, shortener, IP host, subdomain count, special characters, suspicious tokens/TLD…). Git history shows a 16-feature experiment that was later reverted to an 11-feature model on `main` — check the current file before relying on a number |
| Use of the web | **Only** a Google Safe Browsing API lookup (`src/serving/safe_browsing_client.py`). A confirmed match is a **hard bypass**: the system skips the model and reports smishing. The page itself is never fetched |
| Serving | Flask `/predict` (`app/app.py`), explanation modules, a **scope assessment** that warns when a verdict rests on weak ground (`src/serving/scope.py`: unreadable text, non-financial smishing, peso-amount detection) |
| Frontend | React app in `frontend/` (a git submodule). **Its codename is "LE-JET"** (see §3 about the name) |
| Baselines/experiments | Jain et al. (2022) ensemble baseline replication, ungated vs gated models, multi-seed runs, McNemar test, calibration, efficiency measurements (`experiments/`) |
| Data | Versioned labeled CSVs in `data/labeled/` (e.g., `v2_2026-09-15.csv`, `v2withSynthetic_2026-10-01.csv`), never overwritten; disjoint train / RF-fit / test ID lists per seed in `data/splits/` |
| Docs | `README.md`, `docs/PROJECT_EXPLAINER.md`, `docs/PIPELINE_WALKTHROUGH.md`, `docs/DEV_TEAM_BUILD_PLAN.md`, `docs/THESIS_VS_IMPLEMENTATION_COMPARISON.md`, plus `docs/decisions/` and `docs/guides/` |

Design ideas in SmishGuard worth remembering **[observed]**:
- *Graceful fallback:* the Safe Browsing client never raises; every failure becomes a structured `{"status": "error", …}` and the app continues.
- *Sentinel values:* "no URL present" is encoded differently from "a URL with a low score" (`0` for counts, `-1` for measurements).
- *Serving-layer warnings never change the label* — they only tell the user when the verdict is weakly grounded.
- Strict disjoint splits, multi-seed reporting, versioned data with a changelog.

---

## 3. A note on the name "Le-Jet" **[observed / needs confirmation]**

In SmishGuard, **"LE-JET" is the codename of the React frontend** (`docs/PROJECT_EXPLAINER.md` §1–2). This new repo is also called **Le-Jet**. The relationship is not recorded anywhere I could see. Possibilities **[inferred]**: (a) the project is being renamed/re-branded around Le-Jet and this repo is the new home of the whole system; (b) it is a deliberate reuse of the brand for the new web-inspection backend. **Please confirm which**, and write the answer here:

> Decision on naming / relationship to the SmishGuard frontend: _TBD_

---

## 4. Why a new repo

**[stated]** by the project owner: this is "the new repo that we are going to build from scratch"; it is "basically the same" project, but "the features should be on the web inspection," and the first goal is to **try web inspection**.

Reasons the split makes sense **[inferred — confirm or edit]**:
1. **Different evidence.** SmishGuard's URL evidence is the link *string* plus a reputation lookup. Le-Jet's evidence comes from *visiting* the link. That changes the data, the features, the model inputs and the evaluation, so a clean start avoids dragging the old feature pipeline along.
2. **Different risk profile.** Le-Jet touches live phishing infrastructure. It needs an isolated lab (two VirtualBox VMs, VPN, firewalls), hostile-data handling, and secrets (VPN key, SSH key). None of that should live in, or be able to leak through, the thesis repo that already has a frontend, models and API keys.
3. **Different dependencies and tooling.** VirtualBox/WireGuard/nftables scripts, `httpx`/`lxml`/RDAP code, a host-side output validator — a very different stack from the PyTorch/Flask/React stack.
4. **Different dataset and feature definition.** Le-Jet builds on the public web-page phishing dataset (87 features, of which 32 are kept; see §6), whose features are defined on URL + HTML + domain data, rather than SmishGuard's SMS-centric feature set.
5. **A clean history and a safe-by-default repo.** Starting from scratch lets the repo be built with `.gitignore`, hashed requirements, run manifests and tests from day one, and keeps experimental lab work from destabilizing the working SmishGuard system.
6. **Reversibility.** SmishGuard keeps working as-is; Le-Jet can be merged back, wrapped as a service SmishGuard calls, or discarded.

---

## 5. What is different — side by side

| | SmishGuard (existing) | Le-Jet (this repo) |
|---|---|---|
| Core idea | Message text + lexical URL features + Safe Browsing | **Web inspection**: visit the real URL safely and measure it |
| URL evidence | String features only (~a dozen) | **31 model-input features** (the `full` extractor): 14 URL-text + `page_rank` + 12 static-HTML + 4 RDAP/DNS; only the 14 for old links (`url_only`) |
| Network contact | One API call to Google | Live fetches of (possibly malicious) sites, DNS, RDAP — **from an isolated VM lab** |
| Where code runs | Developer machine / Flask server | Analysis VM behind a gateway VM (firewall, VPN kill switch, DNS) |
| Data returned to the host | Model outputs | **Only sanitized, schema-validated feature rows** (JSONL) — never raw HTML or screenshots |
| JavaScript executed? | n/a | **No** in v1 (HTTP-only static parse); browser stage is optional later |
| Missing data | Sentinels `0` / `-1` | `-1` plus indicator columns (`fetch_ok`, `rdap_ok`, `html_available`, …) |
| Reputation services | Safe Browsing is part of the decision (hard bypass) | **Never a training feature** (label leak); at most a separate baseline/fallback |
| Labels/data | SMS corpora (public + Philippine collections), versioned CSVs | URL sets: phishing from public feeds, benign from a **matched sampling plan** |
| Biggest risk | Model quality | **Safety of the lab** and **dataset validity** (dead links, source bias) |
| Status | Working system with trained models | Data cleaning, RF extractors, SMS link features, splits and embeddings done; **inspector (lab + fetcher) still plan only** |

---

## 6. The feature scope that was decided **[stated by owner]**

Final v1 scope (changed 2026-10-11; the earlier 79-feature scope is superseded) = the **31 inputs of the `full` Random Forest extractor**, taken from the cleaned public set (`data/processed/web_phishing_clean.csv`, 32 features including the constant `has_url`):
- **14 URL-text**, from the link string alone: `url_length`, `https_presence`, `number_of_hyphens`, `number_of_slashes`, `subdomain_count`, `special_char_count`, `suspicious_url_token_count`, `shortened_url_indicator`, `has_ip_address`, `suspicious_keyword_count`, `path_length`, `random_domain`, `length_hostname`, `tld_in_path`. Implemented in `src/features/link_features.py`; also the whole input of the `url_only` extractor.
- **1** `page_rank` (Open PageRank, looked up on the host).
- **12 webpage** (static HTML): `nb_hyperlinks`, `ratio_intHyperlinks`, `nb_extCSS`, `login_form`, `external_favicon`, `links_in_tags`, `ratio_intMedia`, `ratio_extMedia`, `iframe`, `empty_title`, `domain_in_title`, `domain_with_copyright`.
- **4 external**: `whois_registered_domain`, `domain_registration_length`, `domain_age` (RDAP) and `dns_record`; missing = `-1`; `1` is the suspicious value for `whois_registered_domain` and `dns_record`.
- Plus the routing flag `page_fetched` (not a model input) for `route_p_u()`.
- Dropped: the rest of the old URL-string features, `sfh` and `ratio_nullHyperlinks` (constant 0), `ratio_extHyperlinks`, `submit_email`, `popup_window`, `safe_anchor`, `onmouseover`, `right_clic`, `web_traffic`, `google_index`, `statistical_report`, and the four `ratio_*Redirection/Errors`.
- Column names follow the cleaned set; definitions in `docs/web_phishing_cleaning.md`. The dataset is Hannousse & Yahiouche (2021), verified in `data/dataset_and_training_plan.md`.
- **Caveat:** no RDAP coverage check is recorded. Treat RDAP (especially `.ph`) as unverified until the Level 1 pilot measures per-TLD success.

**Who is inspected (Option C, `docs/decision_web_inspection.md`):** old SMS links are not visited (82% of old smishing domains are dead); they use `url_only`. Fresh links are inspected within hours: page loaded → `full`, page failed → `url_only`.

Other decisions already made (details in `WEB_INSPECTION_PLAN.md`):
| Decision | Where |
|---|---|
| Local VirtualBox lab, **two VMs** (gateway + analysis), no cloud | Plan §3, §9–§12 |
| **Final-URL rule:** the 14 URL-text features use the **original SMS link** after `normalize_link` (training parity); HTML, RDAP and DNS features describe the **final URL** after redirects/shorteners; `orig_*` columns kept | Plan §13.2b |
| **Free-hosting hosts** (`pages.dev`, `github.io`, …): private-suffix registrable domain, flagged, RDAP skipped (features `-1`) | Plan §13.2c |
| **Benign sampling plan** (five groups, hard negatives, matching on coarse bins, domain-grouped splits, nuisance-only classifier as the bias test) | Plan §29 |
| Data policy default: **features only** leave the VM | Plan §6 |
| Upstream DNS must be **non-filtering** (a filtering resolver makes live phishing look dead) | Plan §10.6, §27 V1 |

---

## 7. Current state of this repo **[observed 2026-10-11]**

- A git repo with history (merged PRs for the URL features and the embeddings).
- **Done:** web phishing set cleaned (`src/data/clean_web_phishing.py`); two RF extractors (`url_only`, `full`) trained over seeds 42/123/2024 (`src/features/model/rf_extractor.py`); synthetic legitimate-with-link SMS; link features for every SMS (`src/features/link_features.py`); grouped splits; English + Tagalog embeddings; Stage 0 inspector code (`src/lejet/urlnorm.py`, `src/lejet/safety.py`); tests for these.
- **Not done:** the lab VMs and the fetcher stages (DNS, RDAP, HTTP, HTML features); adapting the CNN, gating and fusion classifier (carried over from SmishGuard); collecting the fresh set.
- Read `README.md` for how to run each step and `docs/training_rules.md` before writing training code.
- **Warnings:**
  1. `pretrained/` holds the fastText `.bin` files (~19 GB with the `.gz`). It is git-ignored and not needed by the inspector, but the repo sits under **OneDrive**, which may sync it. Consider moving it out.
  2. The laptop is short on disk (~41 GB free on `C:` when last measured); the plan's §7.6 lists what could be freed.

---

## 8. Where the plan stands

`WEB_INSPECTION_PLAN.md` (≈170 KB, 29 sections) covers: threat model, architecture, step-by-step VM/gateway/firewall/VPN setup, the pipeline, the 31-feature catalogue, data model, a test ladder, batch procedure, monitoring/abort criteria, cleanup, thesis data-quality issues, master checklists (§21), a loophole review (§27), suggested extra features (§28) and the benign sampling plan (§29).

**Open items** (tracked in plan §27): which challenge/parked-page detectors to build (V4), canary URLs (V14), duplicate/label-conflict reports (V15), the exact definitions of the 12 HTML features and of `domain_registration_length` and `nb_redirection`/`nb_external_redirection` (from the paper), RDAP coverage, scoped `sudo`, extra service sandboxing, virtio NIC, snapshot refresh routine, and — before any user-facing demo — an ephemeral-worker design for the live inspector (§24).

**Suggested first coding steps** (nothing started):
1. Done already: README, `.gitignore`, URL-text features with their parity test (`link_features.py`, `tests/test_link_features.py`), Stage 0 (`urlnorm.py`, `safety.py`).
2. Implement the registrable-domain / free-hosting logic and its unit tests (plan §13.2c).
3. Implement the 12 HTML features against fixture pages (plan §16 Level 0), plus the `route_p_u` hand-off test.
4. Implement the DNS, RDAP and HTTP stages against local fixtures.
5. Then build the VMs and run the isolation tests (plan §16 Level 2) before touching a single live URL.

---

## 9. Machine and environment facts **[observed]**

- Windows 11 Home, AMD Ryzen 7 7435HS (8C/16T), ~19.7 GB RAM, virtualization enabled in firmware.
- VirtualBox is installed; WSL2 (Ubuntu) and Docker Desktop are also installed — they enable the Windows hypervisor, which makes VirtualBox run in a slower mode. Stop them while the lab runs.
- Shell: PowerShell (primary), Git Bash available. No `D:` drive.
- Git user: Franz Kester P. Calumpit.
- Python venvs exist in SmishGuard (`venv/`); Le-Jet has none yet.

---

## 10. Working agreements

- **Thesis sources must be published 2020 or later.**
- **Never open scanned URLs on the host.** Keep URL lists **defanged** (`hxxps://evil[.]example`) in documents and chats; only the machine-read file is refanged and it stays inside the lab.
- **Treat everything that comes back from the VM as hostile data**: validate against a strict schema, cap strings, escape CSV formulas.
- **No secrets in git or OneDrive:** VPN config (`wg0.conf`), SSH keys, API keys.
- **Don't change SmishGuard from here** unless asked; reuse ideas, not files.
- **Confirm before** anything outward-facing or irreversible (buying a VPN, deleting files, contacting registries/providers, running fetches against live sites).
- Features beyond the confirmed 31 are optional extras and must not replace them.

---

## 11. Glossary

- **Smishing** — phishing delivered by SMS.
- **Web inspection** — safely fetching a URL's page and domain data to measure features, instead of judging the link text alone.
- **RDAP** — the modern, JSON replacement for WHOIS; here used for domain registration and age.
- **Final URL** — the page you end up on after redirects and shorteners.
- **Defanged URL** — a URL written so it cannot be clicked (`hxxps://example[.]com`).
- **Gateway / analysis VM** — the two lab machines: the gateway enforces the firewall, VPN and DNS; the analysis VM runs the fetcher and has no direct internet.
- **Parity test** — recomputing the 14 URL-text features from its `url` column and checking they match the published values.
- **Hard bypass** (SmishGuard) — the rule that a confirmed Safe Browsing match skips the model.

---

## 12. Pointers

| What | Where |
|---|---|
| Full plan | `WEB_INSPECTION_PLAN.md` (this folder) |
| SmishGuard overview | `…\Projects\irs-db\SmishGuard\docs\PROJECT_EXPLAINER.md` |
| SmishGuard setup | `…\SmishGuard\README.md` |
| SmishGuard URL features | `…\SmishGuard\src\features\url_features.py` |
| SmishGuard Safe Browsing client | `…\SmishGuard\src\serving\safe_browsing_client.py` |
| SmishGuard scope assessment | `…\SmishGuard\src\serving\scope.py` |
| SmishGuard labeled data and changelog | `…\SmishGuard\data\labeled\` |
