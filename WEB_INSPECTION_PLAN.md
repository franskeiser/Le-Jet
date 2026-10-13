# Le-Jet — Web Inspection Lab Plan (Local VirtualBox, Maximum-Safety Edition)

> **Goal of this document:** a complete, step-by-step plan to build an isolated local lab, visit *live* (possibly malicious) URLs, extract web-inspection features from them, and bring only sanitized feature data back to the host.
>
> **First milestone:** *prove that web inspection works* on a handful of URLs. Scaling to 10,000+ URLs comes only after the isolation tests and a small pilot pass.

**Status of this plan (updated 2026-10-11):** the **lab and fetcher are still unbuilt** (no VMs, no `fetcher`/stages code, nothing run against live sites). What *does* exist in the repo and is reused by this plan: the Stage 0 code (`src/lejet/urlnorm.py`, `src/lejet/safety.py`), the 14 link-text features (`src/features/link_features.py`), the trained RF extractors (`src/features/model/rf_extractor.py`) and their tests. Commands and flag names for the lab were written from knowledge of VirtualBox 7.x, Ubuntu 24.04, nftables, Unbound, WireGuard and Playwright. **Verify each command against the versions you actually install** (see §22 Troubleshooting). Items marked **[VERIFY]** are the ones most likely to differ.

**Thesis reminder:** any reference you cite in the thesis for these methods must be published **2020 or later**.

> **SCOPE UPDATE (2026-10-11) — v1 feature set. This supersedes the earlier 79-feature scope and anything below that still says otherwise.**
>
> **What web inspection still is:** the inspector **visits the website** (DNS, HTTP fetch with redirects, one static HTML parse, a domain-registration lookup) and **extracts features from the link and from what it finds there**. That has not changed. What changed is *which* features, because the model now consumes the inputs of the **`full` Random Forest extractor** (`src/features/model/rf_extractor.py`, `VARIANTS["full"]`), not the old 79.
>
> **v1 collects 31 features** per link (names exactly as in `data/processed/web_phishing_clean.csv`; definitions in `docs/web_phishing_cleaning.md`):
> - **14 URL-text** features, computed from the link string alone (no network): `url_length`, `https_presence`, `number_of_hyphens`, `number_of_slashes`, `subdomain_count`, `special_char_count`, `suspicious_url_token_count`, `shortened_url_indicator`, `has_ip_address`, `suspicious_keyword_count`, `path_length`, `random_domain`, `length_hostname`, `tld_in_path`. **Already implemented** in `src/features/link_features.py`; the inspector must call that code, not re-implement it.
> - **1 domain-popularity** feature, `page_rank` (Open PageRank). A lookup *about the domain*, done on the **host** (the API key never enters the VM).
> - **12 webpage** features (static HTML parse): `nb_hyperlinks`, `ratio_intHyperlinks`, `nb_extCSS`, `login_form`, `external_favicon`, `links_in_tags`, `ratio_intMedia`, `ratio_extMedia`, `iframe`, `empty_title`, `domain_in_title`, `domain_with_copyright`.
> - **4 external** features (RDAP + DNS): `whois_registered_domain`, `domain_registration_length`, `domain_age`, `dns_record`.
> - plus the routing flag **`page_fetched`** (1 when the page was fetched and parsed), which `route_p_u()` needs to choose between the `full` and `url_only` extractors. It is *not* a model input.
>
> **Dropped from the old scope:** the rest of the old 55 URL-string features (separate character counts, word-length statistics, brand lists, `nb_redirection`…), replaced by the 14 above; `ratio_extHyperlinks`, `ratio_nullHyperlinks`, `submit_email`, `sfh`, `popup_window`, `safe_anchor`, `onmouseover`, `right_clic` (the last three needed script-pattern matching; `sfh` and `ratio_nullHyperlinks` are 0 in every row of the public set); `web_traffic`, `google_index`, `statistical_report`; the four `ratio_*Redirection/Errors` features (one request per link on the page).
>
> **Where the inspector sits in the pipeline (decision `docs/decision_web_inspection.md`, Option C):** old SMS links are never inspected (82% of old smishing domains are dead; today's lookups would only measure "the domain died"). They use the `url_only` extractor (14 URL-text features). The inspector is used for **fresh links, visited within hours of arrival**: page loaded → `full` extractor; page failed → `url_only`. Time between receipt and visit must be recorded for every link.
>
> Consequences:
> 1. The fetcher is **HTTP-only** in v1: fetch the HTML once, parse it statically, **execute no JavaScript**. The browser stage (§13.6) is **optional** (v1.1/v2). With only 12 static HTML features and no script-pattern features, the parser is smaller and safer, and a URL takes about 0.5–3 s.
> 2. **RDAP is in v1** (3 features) plus the DNS-record check (§13.4b).
> 3. TLS, HTTP-header, redirect-chain, behavior and rendered-DOM features are **v2 extras** (§14.2–14.7).
> 4. The data policy is "features only" with options A–D (§6 item 2); screenshots are out of v1 (§14.9).
> 5. New sections: §7.6 (freeing disk space), §24 (integration with the Le-Jet pipeline), §25 (latency budget), §26 (role of Safe Browsing).

---

## Table of contents

1. [Scope, goals and non-goals](#1-scope-goals-and-non-goals)
2. [Threat model](#2-threat-model)
3. [Architecture overview](#3-architecture-overview)
4. [Decision log (what we chose and why)](#4-decision-log)
5. [Bill of materials (everything you need)](#5-bill-of-materials)
6. [Phase 0 — Governance and approvals](#6-phase-0--governance-and-approvals)
7. [Phase 1 — Host hardening and prep](#7-phase-1--host-hardening-and-prep)
8. [Phase 2 — Repo skeleton (Le-Jet)](#8-phase-2--repo-skeleton-le-jet)
9. [Phase 3 — Create the two VMs](#9-phase-3--create-the-two-vms)
10. [Phase 4 — Gateway VM configuration](#10-phase-4--gateway-vm-configuration)
11. [Phase 5 — Analysis VM configuration](#11-phase-5--analysis-vm-configuration)
12. [Phase 6 — Management access and data transfer](#12-phase-6--management-access-and-data-transfer)
13. [Phase 7 — The inspection pipeline (design)](#13-phase-7--the-inspection-pipeline-design)
14. [Feature catalogue](#14-feature-catalogue)
15. [Data model and output format](#15-data-model-and-output-format)
16. [Phase 8 — Test ladder (start here to "try web inspection")](#16-phase-8--test-ladder)
17. [Phase 9 — Batch operating procedure for 10,000+ URLs](#17-phase-9--batch-operating-procedure)
18. [Monitoring, abort criteria, incident response](#18-monitoring-abort-criteria-incident-response)
19. [Phase 10 — Decommissioning and cleanup](#19-phase-10--decommissioning-and-cleanup)
20. [Data-quality issues to document in the thesis](#20-data-quality-issues-to-document-in-the-thesis)
21. [Master checklists](#21-master-checklists)
22. [Troubleshooting](#22-troubleshooting)
23. [Appendices (config files and code)](#23-appendices)
24. [Integration with the Le-Jet pipeline](#24-integration-with-the-le-jet-pipeline)
25. [Latency budget and measurements](#25-latency-budget-and-measurements)
26. [Role of Google Safe Browsing](#26-role-of-google-safe-browsing)
27. [Review: loopholes and gaps found](#27-review-loopholes-and-gaps-found)
28. [Suggested additional features](#28-suggested-additional-features)
29. [Benign sampling plan](#29-benign-sampling-plan)

---

## 1. Scope, goals and non-goals

### 1.1 Goals
- G1. Safely **visit the real URL** (DNS, RDAP, HTTP redirects, page source) from an isolated environment. (Rendered page / TLS / behavior: v2.)
- G2. Extract the **31 v1 features** per URL (14 URL-text, 1 `page_rank`, 12 static-HTML, 4 RDAP/DNS; see the scope update above and §14.0) plus the `page_fetched` routing flag.
- G3. Store features in a reproducible, auditable format with provenance (when, from where, with what tool versions).
- G4. Return **only sanitized, schema-validated feature data** to the host.
- G5. Make every safety control **testable** (negative tests that must fail) and **repeatable** (preflight before every batch).

### 1.2 Non-goals (v1)
- No form submission, no credential entry, no clicking inside pages, no file downloads/execution.
- No reputation-API calls from inside the VM (keeps API keys out of the VM). Labels/reputation, and the one allowed lookup `page_rank` (Open PageRank), are done on the host (§5.7).
- No ASN/geo enrichment, no TLS/HTTP-header/redirect-chain/behavior features, and no rendered-DOM (JavaScript) features in v1 (these are v2 extras, §14.2–14.7). **RDAP *is* in v1** (3 features, §13.4b).
- No crawling of the links found on a page (this is why the four `ratio_*Redirection/Errors` features were dropped).
- No inspection of **old SMS links** (they use the `url_only` extractor, which needs no visit; decision `docs/decision_web_inspection.md`). The inspector exists for the fresh set and a live demo.
- The inspector's output plugs into the model only through `route_p_u()` (§24). This plan does not change the CNN, gates or fusion classifier.

### 1.3 Definition of "web inspection works" (milestone M1 exit criteria)
1. The isolation test suite (§16, Level 2) passes 100%.
2. The harness fixtures test (§16, Level 0) passes 100%.
3. At least 20 safe external URLs produce complete feature rows with no crashes (§16, Level 1).
4. One row per input URL is produced, including failures (with a reason code).
5. The results file passes host-side validation.
6. The validated rows (14 URL-text + `page_rank` + 12 webpage + 4 external + `has_url` + `page_fetched`) run through `route_p_u()` with the frozen extractors without errors, and `page_fetched` picks the right extractor (§16 Level 0).

---

## 2. Threat model

### 2.1 Assets to protect (in priority order)
1. Your laptop and its files (thesis, Le-Jet and SmishGuard repos, OneDrive contents, saved browser sessions, credentials).
2. Your home network (router, other devices).
3. Your identity and home IP address.
4. Integrity of your dataset (poisoned/forged feature data).
5. Your VirtualBox install and VM images.

### 2.2 Adversaries and what they can do
| Adversary | Capability | Example |
|---|---|---|
| Malicious web page | Runs JavaScript in the browser, serves exploits, redirects, forces downloads, popups | Browser 0-day, drive-by download |
| Malicious server | Sends hostile HTTP/TLS responses, huge bodies, slowloris-style stalls | Resource exhaustion, parser bugs |
| Malicious data in features | Page-controlled strings flow into your output | CSV formula injection, JSON/SQL injection, terminal escape sequences, log injection |
| Hostile DNS answers | Domain resolves to `127.0.0.1`, `192.168.x.x`, or `10.0.2.2` (DNS rebinding) | Page tries to reach your router or host via the VM |
| Compromised analysis VM | Attacker has code execution in the guest | Tries to reach host/LAN, scans, mines crypto, spams |
| Attacker-run monitoring | Sees your IP when you connect | Targeted spam/scans against your home IP |
| Yourself (human error) | Copy/paste, clicking links, wrong network mode | Opening a live URL on the host, leaving shared folders on |

### 2.3 Attack paths and the control for each
| # | Attack path | Control |
|---|---|---|
| A1 | Browser exploit → code exec in browser | **v1 runs no browser at all (HTTP-only static parse).** If the optional browser stage is enabled: Chromium sandbox **explicitly enabled**; unprivileged user; fresh context per URL; patched browser. (HTTP/HTML parser bugs in `httpx`/`lxml` remain possible: keep packages pinned and updated, cap sizes, run as unprivileged user.) |
| A2 | Code exec in guest → VM escape to host | Minimal VM hardware (no USB/audio/3D/shared folders/clipboard/Guest Additions); updated VirtualBox; **gateway VM** so there is no direct guest↔host network path |
| A3 | Guest → host via NAT gateway `10.0.2.2` | Analysis VM has **no NAT adapter**; gateway firewall drops all private ranges; host reachable only via SSH tunnel host→VM (never VM→host) |
| A4 | Guest → LAN (router, printers, other PCs) | Gateway forward policy default drop; RFC1918/link-local/CGNAT blocked; no bridged adapters anywhere |
| A5 | DNS rebinding (public name → private IP) | Resolver on gateway with `private-address` filtering; fetcher refuses private/loopback IPs (defense in depth) |
| A6 | Data exfil/C2 on odd ports | Egress only TCP 80/443 via the gateway; QUIC/UDP blocked; connection metadata logged |
| A7 | Home IP exposure | WireGuard VPN on gateway with kill switch (variant A) |
| A8 | Drive-by download / auto-run | Downloads cancelled and never saved; no execution of anything fetched; no auto-open |
| A9 | Resource exhaustion (fork bomb, memory, crypto-mining, huge bodies) | `systemd-run` cgroup limits; per-URL hard timeouts; response size caps; browser recycling; idle-CPU monitoring |
| A10 | Poisoned output into host tools | JSONL only (no SQLite/HTML/screenshots to host); schema validation; string caps; control-char stripping; CSV formula escaping; parse in isolated Python (`-I`) |
| A11 | Malicious file from `scp`/`sftp` server (client bug) | Pull exactly one named file into an empty quarantine dir; modern OpenSSH; verify sha256; never run anything from it |
| A12 | Credential theft from the VM | No personal credentials, no API keys, no agent forwarding in the VMs; throwaway SSH key pair for the lab only |
| A13 | Persistence across batches | Revert to clean snapshot after every batch |
| A14 | Accidental click on a live URL on host | URL lists stored **defanged**; never open URLs in the host browser |
| A15 | Time skew breaks TLS validation | chrony on gateway syncing via VPN; analysis VM syncs from gateway |

### 2.4 Residual risk (be honest in the thesis)
- A browser 0-day **plus** a VirtualBox escape could still reach the host. This is rare and expensive, but not impossible. Keep everything patched and keep the network design so a guest compromise has nowhere useful to go.
- VPN exit IPs are sometimes blocked/cloaked by phishing kits, which biases which pages you see.
- You cannot prove a page is benign; you can only reduce what it can do.

---

## 3. Architecture overview

```
                           Internet
                              ▲
                              │  (only via WireGuard, only TCP 80/443 + DoT 853 from gateway's resolver)
                       ┌──────┴────────────────────────────────────┐
  Windows host         │ VM1: lejet-gw  (Ubuntu Server, 1 vCPU, 1GB)│
  (VirtualBox)         │  • NIC1: VirtualBox NAT  (out to internet) │
        │              │  • NIC2: internal net "lejet-net"          │
        │ SSH tunnel   │  • nftables firewall (default DROP)        │
        │ 127.0.0.1:   │  • WireGuard VPN + kill switch             │
        │ 2201 → :22   │  • Unbound DNS resolver (logs queries)     │
        └─────────────▶│  • chrony time server, tcpdump/nflog logs  │
                       └──────────────┬─────────────────────────────┘
                                      │ internal network "lejet-net" 10.77.0.0/24
                                      │ gateway=10.77.0.1   analysis=10.77.0.2
                       ┌──────────────┴─────────────────────────────┐
                       │ VM2: lejet-an  (Ubuntu Server, 4 vCPU, 6GB)│
                       │  • NIC1: internal net ONLY (no NAT)        │
                       │  • Runs fetcher as unprivileged user       │
                       │  • httpx, lxml, dnspython (browser: v2)    │
                       │  • Local nftables (second firewall layer)  │
                       │  • No secrets, no Guest Additions          │
                       └────────────────────────────────────────────┘
```

**Why two VMs:** a single NAT VM enforces its firewall *inside the machine being attacked*; a compromise can switch it off. With a gateway VM, the filter lives on a different machine that the analysis VM can only reach on DNS/NTP.

**Data flow (always host-initiated, one direction at a time):**
```
Host (defanged url list) ──push──▶ VM2 ──run──▶ VM2 (SQLite + logs) ──export JSONL.gz──▶ Host quarantine
                                                                               │
                                                      validate (schema, caps, sanitize) ─▶ clean data
```

---

## 4. Decision log

| Decision | Choice | Reason | Alternative |
|---|---|---|---|
| Hypervisor | VirtualBox (already installed) | Free, works on Windows 11 Home | VMware Workstation |
| Topology | Gateway VM + Analysis VM | Filtering outside the compromised machine | Single NAT VM + guest firewall (weaker) |
| Guest OS | Ubuntu Server 24.04 LTS | Long support, Playwright support, nftables/apparmor | Debian 12 |
| Browser driver | Playwright (Chromium) | Per-context isolation, request interception, download/dialog hooks | Selenium |
| VPN | WireGuard on gateway, kill switch | Hides home IP; fail-closed | Variant B: no VPN (home IP exposed) |
| Output to host | JSONL.gz only | Avoids parsing untrusted SQLite/HTML/images on host | SQLite export (riskier) |
| Reputation APIs | Not in VM | No secrets in VM; keeps label join on host | In-VM with scoped keys |
| Screenshots | Not exported in v1 | Image parsers on host are attack surface | Downscaled PNG later |
| JS execution | **No in v1** (HTTP-only static parse); optional sandboxed Chromium later | The 12 HTML features come from page source; no JS execution removes the largest risk and is 3–5× faster | Playwright render (v1.1/v2, needed for JS-built pages) |
| Feature set | 31 features of the `full` RF extractor: 14 URL-text, `page_rank`, 12 HTML, 4 RDAP/DNS (2026-10-11; replaces the 79-feature scope) | Matches what the trained extractor consumes (`rf_extractor.VARIANTS["full"]`) and the cleaned public set; drops features that were constant, script-pattern based, or needed per-link requests | Add TLS/behavior extras in v2 |
| Who is inspected | Fresh links only, visited within hours; old SMS links are not visited | 82% of old smishing domains are dead, so today's lookups would measure "the domain died" (`docs/decision_web_inspection.md`, Option C) | Inspect old links too (rejected: fake signal) |
| Two extractors | `full` when `page_fetched = 1`, otherwise `url_only` (`route_p_u`) | Each extractor only sees the kind of data it was trained on; no masking | One extractor trained with masking (rejected) |
| URL-text features | Call `src/features/link_features.py`; never re-implement them in the inspector | One definition for training, SMS data and live inspection (train/serve consistency) | Separate inspector copy (drifts) |
| `page_rank` | Looked up on the **host** (Open PageRank, free key) for the final registrable domain | It is a third-party service; keeping the key out of the VM keeps the "no secrets in the VM" rule | Query from the VM (key exposed) |
| Safe Browsing | Not a training feature; separate baseline/fallback (§26) | It would leak the label; it only knows already-reported URLs | Use as a feature (not recommended) |
| Missing values | Sentinel `-1` **plus** indicator columns `fetch_ok`, `rdap_ok`, `html_available` | Tree models accept −1; neural nets need the indicators; lets you measure liveness bias | Imputation |
| Storage location | `D:\VMs\LeJet` (outside OneDrive) | Prevent OneDrive syncing VM disks; the project and lab now live on the D: SSD (about 870 GB free), so the 41 GB limit on `C:` no longer applies | Back to `C:` (not enough space) |

---

## 5. Bill of materials

### 5.1 Hardware (your laptop)
- AMD Ryzen 7 7435HS (8C/16T), 19.7 GB RAM, ~40 GB free on `C:`, virtualization enabled in firmware (confirmed).
- **Disk budget:** measured on this laptop: `C:` ≈ 475 GB total, 433 GB used, **~41 GB free**. Lab needs: gateway 6 GB + analysis 16–20 GB (dynamic) + ISO 3 GB (delete after install) + snapshot growth 2–5 GB + logs/data 1–3 GB ≈ **18–22 GB realistic, 29–37 GB worst case**. Windows also needs 10–15 GB headroom. **Decision (2026-10-11): the lab lives on the `D:` SSD (about 870 GB free), with `$Base = D:\VMs\LeJet` and data in `D:\LeJetData`**, so the `C:` shortage no longer limits it. Use a 16 GB analysis disk (`-DiskMB 16384`). Never place VMs under OneDrive.

### 5.2 Host software
| Item | Notes |
|---|---|
| VirtualBox 7.x + (optional) Extension Pack | Already installed. Update to latest 7.x. Extension Pack is **not needed**; skip it (less attack surface) |
| Windows OpenSSH client | Built into Windows 11 (`ssh`, `scp`, `sftp`, `ssh-keygen`) |
| Python 3.11+ | Use a **separate** venv for Le-Jet host tooling (validation only) |
| Git | Already in use |
| 7-Zip or built-in tar | Optional |
| PowerShell 5.1/7 | For VM scripts |

### 5.3 Downloads (verify checksums, §7.4)
- Ubuntu Server 24.04 LTS ISO + `SHA256SUMS` + `SHA256SUMS.gpg` from the official Ubuntu releases site.
- VirtualBox installer + SHA256 from virtualbox.org (only if updating).
- A WireGuard configuration file from a VPN provider that permits WireGuard config export and allows security research (read their terms). Avoid free VPNs.

### 5.4 Guest software — gateway (apt)
`nftables`, `wireguard`, `wireguard-tools`, `unbound`, `chrony`, `tcpdump`, `openssh-server`, `unattended-upgrades`, `ufw` is **not** used (nftables only).

### 5.5 Guest software — analysis VM
- apt: `python3`, `python3-venv`, `python3-pip`, `sqlite3`, `nftables`, `chrony`, `openssh-server`, `dnsutils` (dig), `curl`, `ca-certificates`, `jq`, `stress-ng` (for limit tests).
- Python (pinned with hashes), **v1 core:** `httpx[http2]`, `beautifulsoup4`, `lxml`, `tldextract`, `dnspython`, `pydantic`, `tenacity`, `tqdm`, `pytest`, `pytest-asyncio`, plus whatever `src/features/link_features.py` imports (`nostril` for `random_domain`). Keep versions in step with the repo's `requirements.txt`; the extractor `.joblib` files are tied to its scikit-learn version, so load them only where that version is installed.
- Python, **optional (only if the browser stage or TLS extras are enabled):** `playwright`, `cryptography`.
- Browser (**optional, not needed for v1**): Playwright-managed Chromium (installed once by root into `/opt/lejet/browsers`). If you skip it, also skip §11.8–11.9 and the sandbox tests.
- RDAP needs no library: `httpx` plus the IANA bootstrap file (§13.4b).

### 5.6 External data sources (outside the VM, used on the host)
- Phishing/malware URL feeds: OpenPhish community feed, PhishTank, URLhaus, PhishStats.
- Benign URLs: Tranco top list (research-oriented ranking).
- The Le-Jet SMS links (`data/processed/sms_link_features.csv`; expect most to be dead, which is why they are *not* inspected, see the scope update) and the **fresh set** (dataset C in `data/dataset_and_training_plan.md`), the only set the inspector is for.
- *Reputation APIs (Safe Browsing, VirusTotal)*: host only, v2.

### 5.7 APIs
| API | In v1? | Where it runs | Why |
|---|---|---|---|
| Open PageRank | **v1** (feature `page_rank`) | **Host**, after the VM returns the final registrable domain | Free API key; one lookup per final registrable domain, cached. Failure → `-1` and `page_rank_ok = 0`. Not queried from the VM (key stays out). Same source as the training column, so the scale matches |
| Google Safe Browsing | No | Host (v2 labeling) | Still in the carried-over SmishGuard serving code; see §26 |
| VirusTotal | No | Host (v2) | Rate-limited free tier |
| URLhaus / OpenPhish / PhishTank feeds | Source lists only | Host | Feed download, not per-URL API calls from VM |
| RDAP | **v1** | Analysis VM (HTTPS via gateway) | `whois_registered_domain`, `domain_registration_length`, `domain_age`; IANA bootstrap picks the registry server; cache per registrable domain; failures → `-1` (§13.4b). I have not verified RDAP coverage for `.ph` and other ccTLDs — test early |
| IPinfo / MaxMind GeoLite2 | v2 | Offline DB copied in | ASN/hosting-provider feature |

---

## 6. Phase 0 — Governance and approvals

1. **Adviser approval (written):** one email stating you will contact live phishing infrastructure from an isolated VM for thesis research. Save the reply.
2. **Define data policy** and write it in the thesis ethics section (what you collect, why, where it is stored, how long, when destroyed). Default: **features only**; raw HTML/screenshots stay in the VM and are destroyed with the snapshot revert.

   *Why page text is not copied to the laptop:*
   - Page content is attacker-controlled: spreadsheet formula injection (`=cmd|...`), terminal/ANSI escapes, Unicode bidi tricks, oversized or malformed content, and lookalike links that get clicked by accident. Numbers and booleans cannot do any of that and can be validated strictly.
   - Privacy: phishing pages often show real people's data (names, numbers, leaked credentials). Storing that on your laptop (or in OneDrive) creates a personal-data problem under the Data Privacy Act that you would have to justify.
   - HTML and images are the riskiest to parse on the host; size is also a factor (several GB vs a few MB of features).
   - Cost of the rule: you cannot compute a **new** feature later without refetching, and many pages will be dead by then. So decide the feature list carefully now.

   *Options (pick one, record it in the run manifest):*

   | Option | What leaves the VM | Risk | Use for |
   |---|---|---|---|
   | **A. Features only** (default, v1) | Counts, ratios, booleans, hashes | Lowest | The 31-feature scope |
   | **B. Compute more features inside the VM** | Add numeric text features (keyword counts, FastText embeddings using `cc.en.300.bin`/`cc.tl.300.bin`, language ID) computed in the VM; only vectors leave | Low | Using page text in a model without exporting text |
   | **C. Export sanitized text sample** | Visible text only, truncated (5–10 KB), control chars/bidi stripped, PII patterns masked (phones, emails, card-like numbers), inert JSON strings; small sample (a few hundred pages) with adviser approval; never in OneDrive | Medium | Error analysis, thesis examples |
   | **D. Encrypted offline evidence archive** | Raw HTML/screenshots on an encrypted external disk | Medium–high | Only if the adviser requires it |

   **Recommendation: A, plus B if you add text-based features. Use C only for a small approved sample.**
3. **Define ethical rules for the fetcher:** load pages only; never submit forms; never enter credentials; never click; never download; never attempt exploitation; identify research traffic only if the adviser requires it.
4. **Choose a freeze window:** decide the date range during which you fetch (so features are temporally consistent). Record it in the manifest.
5. **URL provenance:** every URL must come from your dataset or a public feed. No manual crawling beyond the URL itself (only redirects it triggers).
6. **Check university policy** on contacting malicious infrastructure from personal equipment.
7. **Decide stop conditions** (§18.2) *before* starting.

---

## 7. Phase 1 — Host hardening and prep

### 7.1 Windows
- [ ] Install all Windows updates; reboot.
- [ ] Windows Security: Real-time protection **on**; Tamper Protection **on**.
- [ ] Turn on **Controlled Folder Access** and add your thesis/project folders.
- [ ] Use a standard (non-admin) account for daily work; admin only for installs.
- [ ] Make an **offline backup** (USB drive) of thesis + code. OneDrive is not ransomware-proof.
- [ ] Note your **home public IP** now (for the VPN test later): visit an "what is my IP" site from the host.
- [ ] Close Docker Desktop and stop WSL before running VMs:
  ```powershell
  wsl --shutdown
  ```
  (Docker/WSL2 enable the Windows hypervisor; VirtualBox then runs in a slower compatibility mode. It still works.)

### 7.2 VirtualBox
- [ ] Update to the latest 7.x release (security fixes matter for escapes).
- [ ] Do **not** install the Extension Pack.
- [ ] Do not install Guest Additions in either VM (ever).
- [ ] Set the machine folder to `D:\VMs\LeJet` (already the SSD): *File → Preferences → General*.
- [ ] Confirm you have enough free disk (≥ 35 GB).

### 7.3 Folders on the host
```powershell
New-Item -ItemType Directory -Force D:\VMs\LeJet, D:\LeJetData\iso, D:\LeJetData\input, D:\LeJetData\quarantine, D:\LeJetData\clean, D:\LeJetData\manifests | Out-Null
```
`D:\LeJetData` is **outside OneDrive** on purpose: attacker-influenced strings land here.

Optional: add `D:\LeJetData` and `D:\VMs` to Defender exclusions **only if** performance requires it (not recommended; scanning is useful).

### 7.4 Verify downloads
```powershell
# Ubuntu: compare to the SHA256SUMS file from the same official page
Get-FileHash D:\LeJetData\iso\ubuntu-24.04*-live-server-amd64.iso -Algorithm SHA256
```
For stronger verification also validate the GPG signature of `SHA256SUMS` (Gpg4win), following the instructions on the Ubuntu download page.

### 7.5 SSH keys (lab-only)
```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\lejet_ed25519 -C "lejet-lab-only"
```
Use a passphrase. This key is used **only** for the lab VMs. Never load it into an agent that will be forwarded anywhere.

### 7.6 Freeing disk space on `C:` (optional now: the lab lives on `D:`; keep `C:` healthy, about 15 GB free minimum)

A read-only scan of this laptop (nothing was deleted) found these candidates. **Review each item yourself before deleting.** Sizes are as measured; the scan of the rest of the user profile (OneDrive Desktop/Documents, videos, games, old projects) had not finished, so the larger remainder is still unlocated — re-run a folder-size scan to find it (e.g., WinDirStat/TreeSize Free).

**Safe to delete (caches, duplicates, installers)**
| Item | Size | How |
|---|---|---|
| `%LOCALAPPDATA%\Temp` | 8.7 GB | Close apps; delete contents; skip locked files |
| Duplicate `Neurotec_Biometric_2025_2_SDK_… (1).zip` in Downloads | 4.4 GB | Delete the `(1)` copy |
| `Unconfirmed *.crdownload` in Downloads | 2.3 GB | Abandoned partial downloads |
| `Docker Desktop Installer.exe`, `capcut-1-5-0.exe` in Downloads | ~1 GB | Already installed |
| `%USERPROFILE%\.nuget` | 3.6 GB | Cache; re-downloads on demand |
| `%USERPROFILE%\.gradle` | 1.1 GB | Cache |
| `%USERPROFILE%\.cache\huggingface` | 5.6 GB | Cache; check SmishGuard doesn't rely on a cached model first |
| Windows Disk Cleanup / Storage Sense | a few GB | Update leftovers, thumbnails, Recycle Bin |

**Your decision**
| Item | Size | Question |
|---|---|---|
| Ollama models (`ollama list`, `ollama rm <name>`) | ≤ 24 GB | Six models (qwen3.5, qwen3, qwen3-vl, moondream). Keep only what you use |
| `anaconda3` | 13 GB | Do you use conda or only project venvs? |
| Neurotec SDK / MegaMatcher zips in Downloads | 4.4 + 1.2 GB | Delete if already installed/extracted |
| Clipchamp app data | 6.7 GB | Uninstall if unused |
| Docker Desktop data (`docker system prune -a`) | ~5 GB | Needed only if you use containers |
| Media in Downloads (videos, photo zips, `memory-lane` folder) | ~3 GB | Move to external drive/Drive instead of deleting |
| Hibernation file `hiberfil.sys` | 7.9 GB | `powercfg -h off` frees it; you lose hibernate/Fast Startup |

**Leave alone:** `pagefile.sys` (19.7 GB, Windows-managed), the Claude desktop app's data (~8.9 GB), WSL Ubuntu (3.1 GB) unless unused, project `venv` folders.

Total identified above ≈ 70–85 GB at most, so reaching 150 GB free means finding another ~65+ GB elsewhere (or moving large personal media to an external drive). The lab itself only needs ~50 GB free; an external SSD is the simplest way to avoid deleting anything you want to keep.

---

## 8. Phase 2 — Repo skeleton (Le-Jet)

Project folder: `D:\Le-Jet` (moved from OneDrive on 2026-10-11; a git repo with history; see `README.md` for the full layout and `CONTEXT.md` for background).

### 8.1 Layout: what exists today and what this plan adds
The inspector is **added to the existing structure**, not built as a separate project. `[exists]` = already in the repo; `[new]` = to be written for web inspection.
```
Le-Jet/
├─ README.md, CONTEXT.md, WEB_INSPECTION_PLAN.md (this file)      [exists]
├─ requirements.txt                                                [exists; add hashed vm.txt for the VM, §8.3]
├─ data/
│  ├─ processed/web_phishing_clean.csv                             [exists: 11,430 rows, 32 features; the training set of the extractors]
│  ├─ processed/sms_link_features.csv                              [exists: link features of the SMS data]
│  ├─ splits/seed_{42,123,2024}/                                   [exists]
│  └─ fresh/                                                       [exists, empty (.gitkeep): dataset C, the inspector's output lands here after validation]
├─ docs/
│  ├─ decision_web_inspection.md (Option C), web_phishing_cleaning.md,
│  │  training_rules.md, sms_synthetic_legit_links.md               [exists]
│  └─ run_log_template.md, threat_model.md                         [new]
├─ models/rf_extractor/{url_only,full}_seed{N}.joblib              [exists locally, git-ignored; regenerate with python -m src.features.model.rf_extractor]
├─ src/
│  ├─ config.py                                                    [exists: seeds, paths]
│  ├─ lejet/                                                       (the inspector package; tests import it as `lejet`)
│  │  ├─ safety.py          (private-IP gate, size caps, sanitizers)   [exists]
│  │  ├─ urlnorm.py         (parse, normalize, refuse unsafe targets)  [exists]
│  │  ├─ schema.py          (pydantic models = single source of truth) [new]
│  │  ├─ storage.py         (SQLite in VM, JSONL export)               [new]
│  │  ├─ pipeline.py        (orchestration, concurrency, resume)       [new]
│  │  ├─ cli.py             (run / export / preflight)                 [new]
│  │  ├─ stages/
│  │  │  ├─ dns_stage.py    (v1: dns_record)                           [new]
│  │  │  ├─ rdap_stage.py   (v1: IANA bootstrap, cache, -1 on failure) [new]
│  │  │  ├─ http_stage.py   (v1: manual redirects, byte caps, HTML <= 2 MB) [new]
│  │  │  ├─ tls_stage.py    (v2 optional)                              [new, later]
│  │  │  └─ browser_stage.py (v2 optional: Playwright)                 [new, later]
│  │  └─ features/
│  │     ├─ html_features.py        (v1: the 12 static-HTML features)  [new]
│  │     ├─ external_features.py    (v1: the 4 RDAP/DNS features)      [new]
│  │     ├─ free_hosting_suffixes.txt (§13.2c)                         [new]
│  │     └─ extras/                 (v2: tls.py, network.py, dom.py)   [new, later]
│  ├─ features/
│  │  ├─ link_features.py   (the 14 URL-text features + link finder + normalize_link) [exists; the inspector imports it]
│  │  ├─ url_features.py, text_preprocessing.py                        [exists; SmishGuard-era text/URL code, not the inspector's source of truth]
│  │  └─ model/rf_extractor.py (extract_p_u, route_p_u)                [exists; consumes the inspector's output]
│  ├─ data/                 (cleaning, splitting, SMS link features)   [exists; clean_web_phishing.py defines the 12 webpage + 4 external columns]
│  └─ serving/              (predict.py, scope.py, safe_browsing_client)  [exists, carried over from SmishGuard; §24 adds an inspector client]
├─ tests/
│  ├─ test_urlnorm.py, test_safety.py                              [exists]
│  ├─ test_link_features.py (parity of the 14 URL-text features against the public set) [exists; this is the old "Level 0a" test]
│  ├─ test_rf_extractor.py, test_split_dataset.py, test_text_preprocessing.py, test_embedding.py [exists]
│  ├─ fixtures/             (mock HTML pages, see §16 Level 0)         [new]
│  ├─ test_html_features.py (fixture pages with known expected values) [new]
│  ├─ test_external_features.py (recorded RDAP JSON samples incl. missing/404/no-server; dns_record direction) [new]
│  ├─ test_free_hosting.py  (§13.2c cases)                           [new]
│  └─ test_pipeline_local.py                                         [new]
├─ infra/                                                          [new]
│  ├─ vm/create_vms.ps1
│  ├─ gateway/{nftables.conf,unbound.conf,wg0.conf.example,sysctl.conf,sshd_lejet.conf}
│  └─ analysis/{nftables.conf,preflight.sh,lejet-run.sh,apparmor-lejet-chromium}
└─ scripts/                                                        [new]
   ├─ push_urls.ps1
   ├─ pull_results.ps1
   ├─ lookup_page_rank.py     (host side: Open PageRank for the final registrable domains)
   └─ validate_output.py      (host-side validator; run with python -I)
```
Notes:
- **Import style:** the existing `lejet` modules are imported as `from lejet import safety` (the tests put `src` on `sys.path`), while the rest of the repo uses `from src...`. Keep the inspector's own modules under `lejet`, and import the shared URL-text code as `from src.features.link_features import ...`. Make both work with one small `conftest.py` or an installed package rather than more path tweaks.
- **Naming:** the model and the cleaned data use *our* column names (`url_length`, `number_of_hyphens`, …), **not** the original dataset's (`length_url`, `nb_hyphens`, …). The inspector output must use ours so it can go straight into `route_p_u()`. The 12 webpage and 4 external columns keep the dataset's own names.
- `pretrained/` (about 11 GB of fastText vectors) is not needed by the inspector and must stay out of git and, ideally, out of OneDrive.

### 8.2 `.gitignore` (already in the repo)
The repo's `.gitignore` already excludes what this plan needs: `pretrained/` and `*.bin`, trained models (`models/*`, except the committed `models/embeddings/`), `/data/`, VM disks and ISOs (`*.vdi`, `*.vmdk`, `*.ova`, `*.iso`), `*.jsonl`, `*.jsonl.gz`, `*.sqlite`, `*.db`, `*.pcap`, secrets (`*.key`, `*.pem`, `wg0.conf`, `.env`) and `known_hosts_lejet`. Before the first inspector commit, check that `git status` shows none of these. Note that some files under `data/` are tracked even though `/data/` is ignored (for example `data/interim/…csv` and the split ID lists): do not force-add inspector output there with `-f`; keep batch results in `D:\LeJetData` (§7.3) and only the validated fresh-set CSV under `data/fresh/` locally.
`wg0.conf` contains your VPN private key: **never commit it**; only `wg0.conf.example`.

### 8.3 Dependency hygiene
- Use `pip-compile --generate-hashes` (pip-tools) to make hashed lock files. Install in the VM with `pip install --require-hashes -r vm.txt`.
- Record `pip freeze`, `playwright --version`, Chromium build, kernel version, VirtualBox version in each run manifest.

---

## 9. Phase 3 — Create the two VMs

Run in **PowerShell** on the host. Adjust paths. **[VERIFY]** flag spellings with `VBoxManage modifyvm` for your VirtualBox version (6.x vs 7.x differ in hyphenation; 7.x still accepts most legacy spellings).

### 9.1 Script `infra/vm/create_vms.ps1`
```powershell
$VBox = "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
$Base = "D:\VMs\LeJet"
$ISO  = "D:\LeJetData\iso\ubuntu-24.04-live-server-amd64.iso"   # adjust exact filename
$Net  = "lejet-net"

function New-LeJetVm($Name, $Mem, $Cpus, $DiskMB) {
  & $VBox createvm --name $Name --ostype Ubuntu_64 --basefolder $Base --register
  & $VBox modifyvm $Name `
      --memory $Mem --cpus $Cpus --vram 16 `
      --graphicscontroller vmsvga --accelerate3d off `
      --audio-driver none `
      --usbohci off --usbehci off --usbxhci off `
      --clipboard-mode disabled --draganddrop disabled `
      --vrde off `
      --rtcuseutc on `
      --boot1 dvd --boot2 disk --boot3 none --boot4 none `
      --paravirtprovider kvm `
      --nested-hw-virt off
  & $VBox createmedium disk --filename "$Base\$Name\$Name.vdi" --size $DiskMB --format VDI
  & $VBox storagectl $Name --name SATA --add sata --controller IntelAhci
  & $VBox storageattach $Name --storagectl SATA --port 0 --device 0 --type hdd --medium "$Base\$Name\$Name.vdi"
  & $VBox storageattach $Name --storagectl SATA --port 1 --device 0 --type dvddrive --medium $ISO
}

# --- Gateway: NAT (internet) + internal net
New-LeJetVm -Name lejet-gw -Mem 1024 -Cpus 1 -DiskMB 6144
& $VBox modifyvm lejet-gw `
    --nic1 nat --natpf1 "ssh,tcp,127.0.0.1,2201,,22" `
    --nic2 intnet --intnet2 $Net --nicpromisc2 deny

# --- Analysis: internal net ONLY (no NAT, no bridged, no host-only)
New-LeJetVm -Name lejet-an -Mem 6144 -Cpus 4 -DiskMB 20480
& $VBox modifyvm lejet-an `
    --nic1 intnet --intnet1 $Net --nicpromisc1 deny `
    --nic2 none --nic3 none --nic4 none
```

### 9.2 Settings you must double-check in the GUI afterwards
For **each** VM → *Settings*:
- **General → Advanced:** Shared Clipboard *Disabled*; Drag'n'Drop *Disabled*.
- **System:** Hardware Clock in UTC *on*; boot order DVD then Disk.
- **Display:** Remote Display (VRDE) *off*; 3D acceleration *off*.
- **Storage:** no extra controllers; shared folders none.
- **Audio:** *disabled*. **USB:** *disabled* (no USB controller).
- **Serial Ports / Parallel:** disabled.
- **Network:** Gateway = Adapter 1 NAT (+ port forward `127.0.0.1:2201 → 22`), Adapter 2 Internal `lejet-net`. Analysis = Adapter 1 Internal `lejet-net`, Adapters 2–4 *Not attached*. **No Bridged. No Host-Only.**
- Promiscuous Mode: *Deny* everywhere.

### 9.3 Install Ubuntu Server 24.04 (both VMs)
Installer choices (same on both):
1. Language/keyboard of your choice.
2. Install type: **Ubuntu Server (minimized)**.
3. Network: gateway will get DHCP on NAT; analysis will show no network — that's expected. Continue without network/mirror.
4. Storage: guided, entire disk, **no LVM** needed; encryption optional (not needed).
5. Profile: hostnames `lejet-gw` / `lejet-an`; user `ops` with a strong unique password (used only for console/sudo; SSH will be key-only).
6. **Install OpenSSH server** and paste the content of `lejet_ed25519.pub` as an authorized key (installer option "Import SSH key" → paste, or add it after).
7. **Do not select any featured snaps.**
8. Reboot, remove the ISO:
   ```powershell
   & $VBox storageattach lejet-gw --storagectl SATA --port 1 --device 0 --type dvddrive --medium emptydrive
   & $VBox storageattach lejet-an --storagectl SATA --port 1 --device 0 --type dvddrive --medium emptydrive
   ```
   (If the analysis VM's OpenSSH install fails offline, install it later through the gateway in §11.)

### 9.4 Static network config
**Gateway** `/etc/netplan/50-lejet.yaml` (`sudo netplan apply`):
```yaml
network:
  version: 2
  ethernets:
    enp0s3:               # NAT -> internet
      dhcp4: true
      dhcp6: false
      accept-ra: false
    enp0s8:               # internal net
      dhcp4: false
      addresses: [10.77.0.1/24]
```
**Analysis** `/etc/netplan/50-lejet.yaml`:
```yaml
network:
  version: 2
  ethernets:
    enp0s3:
      dhcp4: false
      dhcp6: false
      accept-ra: false
      addresses: [10.77.0.2/24]
      routes:
        - to: default
          via: 10.77.0.1
      nameservers:
        addresses: [10.77.0.1]
```
Interface names may differ (`ip link`). **[VERIFY]**

**Disable IPv6 on both** (prevents leaks):
```bash
echo 'net.ipv6.conf.all.disable_ipv6=1
net.ipv6.conf.default.disable_ipv6=1' | sudo tee /etc/sysctl.d/60-lejet-ipv6.conf
sudo sysctl --system
```

### 9.5 Baseline snapshot after OS install
Power off, then:
```powershell
& $VBox snapshot lejet-gw take "gw-os-installed" --description "Fresh Ubuntu, before config"
& $VBox snapshot lejet-an take "an-os-installed" --description "Fresh Ubuntu, before config"
```

---

## 10. Phase 4 — Gateway VM configuration

Order matters: network first, then updates, then firewall, then VPN, then DNS.

### 10.1 Updates and packages (while NAT is open, before locking down)
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y nftables wireguard wireguard-tools unbound chrony tcpdump unattended-upgrades jq
sudo systemctl enable --now unattended-upgrades
```

### 10.2 SSH hardening — `/etc/ssh/sshd_config.d/99-lejet.conf`
```
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
PubkeyAuthentication yes
AllowUsers ops
X11Forwarding no
AllowAgentForwarding no
# The gateway is a jump host to the analysis VM, nothing else:
AllowTcpForwarding yes
PermitOpen 10.77.0.2:22
ClientAliveInterval 60
ClientAliveCountMax 3
LoginGraceTime 20
MaxAuthTries 3
```
```bash
sudo sshd -t && sudo systemctl restart ssh
```
Test key login **from the host** (via the forwarded port) before disabling anything else:
```powershell
ssh -i $env:USERPROFILE\.ssh\lejet_ed25519 -p 2201 ops@127.0.0.1
```

### 10.3 Kernel settings — `/etc/sysctl.d/70-lejet-gw.conf`
```
net.ipv4.ip_forward=1
net.ipv4.conf.all.rp_filter=1
net.ipv4.conf.default.rp_filter=1
net.ipv4.conf.all.accept_redirects=0
net.ipv4.conf.all.send_redirects=0
net.ipv4.conf.all.accept_source_route=0
net.ipv4.conf.all.log_martians=1
net.ipv6.conf.all.disable_ipv6=1
```
```bash
sudo sysctl --system
```

### 10.4 WireGuard VPN (Variant A — recommended)

#### 10.4.0 Getting a WireGuard config from a provider (do this first)
You need a **paid VPN that lets you export a plain WireGuard `.conf` file** (private key, server public key, endpoint IP and port). The gateway VM runs it with `wg-quick`, so a provider's own app is not enough.

**Step 1 — Pick a provider.** Two requirements:
- **WireGuard config export.** Some providers have offered this (for example Mullvad, ProtonVPN, IVPN), but verify it is still true and that your plan includes it.
- **Terms that do not forbid your use.** Read the acceptable-use policy for phrases like "malicious sites", "security testing" or "scanning". Most prohibit attacks and spam, not visiting URLs. If the wording is unclear, email support, and keep the reply with your approvals (§6):
  > "I'm an undergraduate researcher. I will load individual public phishing URLs (about 10,000 over several days, at most 2 requests/second) from an isolated VM for a thesis on phishing detection. No scanning, no attacks, no downloads. Is this allowed under your terms?"

Prefer: a Philippines or nearby exit (Singapore, Hong Kong), since SMS scams often target local users; no free VPNs (they rarely export WireGuard configs, often log traffic, and their IP ranges are often blocked by the sites you study). Budget is typically about 5 USD/month; you need it for roughly a month.

**Step 2 — Generate the config.**
1. Sign up and pay (or use a trial if it includes WireGuard).
2. Open the provider's WireGuard config generator (usually under account → "WireGuard configuration").
3. Choose **Linux** as the platform and pick one server.
4. Generate and download the `.conf` (some providers make you generate or upload a keypair first).

**Step 3 — Edit the file for this lab.** It looks roughly like:
```ini
[Interface]
PrivateKey = <your key>
Address = 10.x.x.x/32
DNS = 10.x.x.x          # DELETE this line (our Unbound resolver handles DNS)

[Peer]
PublicKey = <server key>
AllowedIPs = 0.0.0.0/0
Endpoint = vpn.example.com:51820
```
Change these:
- **Delete the `DNS =` line.** Otherwise `wg-quick` rewrites the gateway's resolver settings.
- **Replace the hostname in `Endpoint` with its IP address** (run `nslookup vpn.example.com` on the host). The gateway firewall only allows UDP to one IP and port. Put that IP and port into `VPN_ENDPOINT_IP` and `VPN_ENDPOINT_PORT` in `/etc/nftables.conf` (§10.5).
- Remove any `PostUp`/`PostDown` lines.
- Remove `::/0` from `AllowedIPs` if present (IPv6 is disabled).

**Step 4 — Install it on the gateway.** Do this *before* loading the strict firewall (or load a temporary open ruleset first), because the strict ruleset only allows traffic to the endpoint you configured.
```bash
sudo nano /etc/wireguard/wg0.conf      # paste the edited contents
sudo chmod 600 /etc/wireguard/wg0.conf
sudo systemctl enable --now wg-quick@wg0
sudo wg show                           # look for "latest handshake"
curl -s https://api.ipify.org          # must show the VPN's IP, not your home IP
```

**Step 5 — Handle the keys.**
- `wg0.conf` contains your private key. Never commit it, never put it in OneDrive, never share it (it is already excluded in `.gitignore`, §8.2).
- A config uses one device slot on your plan; one gateway VM uses one slot.
- When the project ends (§19), delete the config and revoke or regenerate the key in the provider's account.
- Record the provider name and exit country in each run manifest (§15.3).

**If you cannot get a VPN:** use Variant B below (no VPN). Your home IP is then visible to every site you load. The risk is mostly spam or scans against your IP rather than access to your laptop; tell your adviser if you go this way. A middle option is tethering through a phone hotspot so the exposed IP is not your home line.

#### 10.4.1 Configure WireGuard on the gateway
1. Put your provider's config at `/etc/wireguard/wg0.conf`, permissions `600`. Make sure it contains **`Table = auto`** semantics (default) and **no PostUp/PostDown** that open holes. Typical shape:
   ```ini
   [Interface]
   PrivateKey = <YOUR_KEY>
   Address = 10.x.x.x/32
   DNS = <ignore, we use our own resolver>   # remove this line; wg-quick would rewrite resolv.conf

   [Peer]
   PublicKey = <PROVIDER_KEY>
   Endpoint = <VPN_ENDPOINT_IP>:51820
   AllowedIPs = 0.0.0.0/0
   PersistentKeepalive = 25
   ```
   Use the endpoint **IP address**, not a hostname (the firewall only permits UDP to that one IP/port).
2. ```bash
   sudo chmod 600 /etc/wireguard/wg0.conf
   sudo systemctl enable --now wg-quick@wg0
   sudo wg show
   ```
3. Confirm the exit IP is **not** your home IP:
   ```bash
   curl -s https://api.ipify.org ; echo
   ```

**Variant B (no VPN):** skip WireGuard; in the nftables file replace `$VPN_IF` with `$WAN_IF` and remove the endpoint rule. Your home IP will be visible to every site you visit. Not recommended.

### 10.5 Firewall — `/etc/nftables.conf` (gateway)
Default policy is DROP everywhere. Logging is applied to new egress connections.
```nft
#!/usr/sbin/nft -f
flush ruleset

define LAN_IF   = "enp0s8"
define WAN_IF   = "enp0s3"
define VPN_IF   = "wg0"
define ANALYSIS = 10.77.0.2
define VPN_ENDPOINT_IP   = 203.0.113.10     # <-- replace
define VPN_ENDPOINT_PORT = 51820            # <-- replace

table inet lejet {

  set private4 {
    type ipv4_addr; flags interval
    elements = { 0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10, 127.0.0.0/8,
                 169.254.0.0/16, 172.16.0.0/12, 192.0.0.0/24, 192.0.2.0/24,
                 192.168.0.0/16, 198.18.0.0/15, 198.51.100.0/24,
                 203.0.113.0/24, 224.0.0.0/3 }
  }

  chain input {
    type filter hook input priority 0; policy drop;
    ct state established,related accept
    iifname "lo" accept
    # SSH only from the host (arrives via VirtualBox NAT as 10.0.2.2)
    iifname $WAN_IF ip saddr 10.0.2.2 tcp dport 22 accept
    # Services for the analysis VM, nothing else:
    iifname $LAN_IF ip saddr $ANALYSIS udp dport { 53, 123 } accept
    iifname $LAN_IF ip saddr $ANALYSIS tcp dport 53 accept
  }

  chain forward {
    type filter hook forward priority 0; policy drop;
    ct state established,related accept
    # Attempts to reach private/reserved space: LOG (rate-limited) then drop, whatever the output interface.
    # (Abort criterion §18.2 #1 depends on these log lines existing.)
    iifname $LAN_IF ip saddr $ANALYSIS ip daddr @private4 limit rate 10/second \
        log prefix "lejet-blocked-private " level warn counter drop
    iifname $LAN_IF ip saddr $ANALYSIS ip daddr @private4 counter drop
    # Cap the rate of NEW connections (a compromised VM could otherwise use your VPN IP to attack others)
    iifname $LAN_IF ip saddr $ANALYSIS ct state new limit rate over 30/second burst 60 packets \
        log prefix "lejet-ratelimit " level warn counter drop
    # Clamp MSS so big pages do not stall on the WireGuard MTU (classic symptom: small pages load, big ones hang)
    oifname $VPN_IF tcp flags syn tcp option maxseg size set rt mtu
    # Analysis VM -> internet via VPN, web ports only
    iifname $LAN_IF oifname $VPN_IF ip saddr $ANALYSIS tcp dport { 80, 443 } ct state new \
        log prefix "lejet-egress " level info counter accept
    # Everything else (UDP/QUIC, ICMP, other ports) is logged (rate-limited) and dropped
    iifname $LAN_IF limit rate 10/second log prefix "lejet-blocked-other " level warn counter drop
  }

  chain output {
    type filter hook output priority 0; policy drop;
    ct state established,related accept
    oifname "lo" accept
    # DHCP on WAN
    oifname $WAN_IF udp sport 68 udp dport 67 accept
    # WireGuard handshake/transport to the VPN endpoint only
    oifname $WAN_IF ip daddr $VPN_ENDPOINT_IP udp dport $VPN_ENDPOINT_PORT accept
    # Anything the gateway itself needs goes through the VPN (DoT, NTP, apt)
    oifname $VPN_IF accept
    # Replies to host SSH
    oifname $WAN_IF ip daddr 10.0.2.2 tcp sport 22 accept
    # Gateway -> analysis (SSH jump only)
    oifname $LAN_IF ip daddr $ANALYSIS tcp dport 22 accept
  }
}

table ip lejet_nat {
  chain postrouting {
    type nat hook postrouting priority 100;
    oifname "wg0" masquerade
  }
}
```
```bash
sudo nft -c -f /etc/nftables.conf    # syntax check
sudo systemctl enable --now nftables
sudo nft list ruleset
```
Notes:
- **Kill switch:** the analysis VM can only exit through `wg0`. If `wg0` is down, forwarded traffic has nowhere to go and the gateway itself cannot reach the internet except the VPN handshake.
- **apt updates:** the gateway's updates go through the VPN (allowed). If the VPN is down, temporarily load a relaxed ruleset deliberately; do not edit the production file ad hoc.
- QUIC/HTTP3 is blocked by design. Also pass `--disable-quic` to Chromium.

### 10.6 DNS resolver — `/etc/unbound/unbound.conf.d/lejet.conf`
```
server:
  interface: 10.77.0.1
  interface: 127.0.0.1                 # the gateway itself resolves through this (see note below)
  access-control: 0.0.0.0/0 refuse
  access-control: 127.0.0.0/8 allow
  access-control: 10.77.0.2/32 allow
  do-ip6: no
  do-udp: yes
  do-tcp: yes
  hide-identity: yes
  hide-version: yes
  qname-minimisation: yes
  aggressive-nsec: yes
  tls-cert-bundle: /etc/ssl/certs/ca-certificates.crt
  # DNS-rebinding protection: drop answers pointing to private space
  private-address: 10.0.0.0/8
  private-address: 100.64.0.0/10
  private-address: 127.0.0.0/8
  private-address: 169.254.0.0/16
  private-address: 172.16.0.0/12
  private-address: 192.168.0.0/16
  # Log every query for auditing/liveness stats
  use-syslog: yes
  log-queries: yes
  log-replies: yes
  log-tag-queryreply: yes
  verbosity: 1

forward-zone:
  name: "."
  forward-tls-upstream: yes
  # NON-FILTERING upstreams on purpose (see the warning below)
  forward-addr: 1.1.1.1@853#cloudflare-dns.com
  forward-addr: 1.0.0.1@853#cloudflare-dns.com
```
```bash
sudo unbound-checkconf
sudo systemctl enable --now unbound
dig @10.77.0.1 example.com +short          # from the gateway itself
dig @10.77.0.1 localtest.me +short         # expect NO answer (rebinding protection)
```
If `systemd-resolved` conflicts on port 53 on the gateway's `10.77.0.1` address, either bind `DNSStubListener=no` in `/etc/systemd/resolved.conf` or disable resolved.

**Warning — use a non-filtering upstream resolver.** Security-filtering resolvers (for example Quad9's default `9.9.9.9`, many VPN "malware blocking" DNS features, family-safe resolvers) answer `NXDOMAIN` for domains on threat lists. For this research that is a silent data bug: live phishing domains would look **dead**, corrupting `dns_record`, liveness and the labels. Use a plain resolver (Cloudflare `1.1.1.1` as above, or Quad9's unfiltered `9.9.9.10` / `dns10.quad9.net`), and switch off any DNS/malware blocking feature of your VPN provider. Add a check to the pilot: a domain currently listed in a phishing feed must resolve via the gateway exactly as it does via a known-unfiltered resolver.

**The gateway itself also needs working DNS** (for `apt`, `chrony`, the IANA bootstrap file) and the firewall only lets it out through the VPN. Point it at the local Unbound:
```bash
sudo mkdir -p /etc/systemd/resolved.conf.d
printf '[Resolve]\nDNS=127.0.0.1\nDNSStubListener=no\n' | sudo tee /etc/systemd/resolved.conf.d/lejet.conf
sudo ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf
sudo systemctl restart systemd-resolved unbound
```
Without this, the gateway's default DNS (the VirtualBox NAT resolver `10.0.2.3`) is blocked by the output chain and name lookups on the gateway fail.

### 10.7 Time — chrony
`/etc/chrony/chrony.conf` additions:
```
allow 10.77.0.2
bindaddress 10.77.0.1
# keep your default pool/NTS servers (they go out via the VPN)
```
```bash
sudo systemctl restart chrony
chronyc tracking
```

### 10.8 Traffic logging
Per-batch pcap of metadata (not for sharing):
```bash
sudo tcpdump -i enp0s8 -n -s 96 -w /var/log/lejet/batch-$(date +%Y%m%d-%H%M).pcap &
```
nftables `log` lines go to the journal: `sudo journalctl -k | grep lejet-egress`.
Create `/var/log/lejet` first. Rotate/delete logs after each batch.

### 10.9 Gateway snapshot
After everything works and §16 Level 2 tests pass:
```powershell
& $VBox snapshot lejet-gw take "gw-configured-clean"
```

---

## 11. Phase 5 — Analysis VM configuration

### 11.1 First boot and updates
With the gateway up, from the analysis console:
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3 python3-venv python3-pip sqlite3 nftables chrony \
     dnsutils curl ca-certificates jq stress-ng openssh-server
```
(apt works through the gateway because ports 80/443 are allowed and DNS is the gateway.)

### 11.2 Time
`/etc/chrony/chrony.conf`: replace pools with
```
server 10.77.0.1 iburst
```
```bash
sudo systemctl restart chrony && chronyc tracking
```

### 11.3 SSH hardening (same as gateway, but only the gateway may connect)
`/etc/ssh/sshd_config.d/99-lejet.conf`:
```
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
AllowUsers ops
X11Forwarding no
AllowAgentForwarding no
AllowTcpForwarding no
```
Local firewall (below) additionally restricts SSH to `10.77.0.1`.

### 11.4 Local firewall — `/etc/nftables.conf` (analysis)
Second layer of defense. Does not replace the gateway filter.
```nft
#!/usr/sbin/nft -f
flush ruleset

table inet lejet_an {
  set private4 {
    type ipv4_addr; flags interval
    elements = { 0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10, 127.0.0.0/8,
                 169.254.0.0/16, 172.16.0.0/12, 192.0.0.0/24, 192.168.0.0/16,
                 198.18.0.0/15, 224.0.0.0/3 }
  }

  chain input {
    type filter hook input priority 0; policy drop;
    ct state established,related accept
    iifname "lo" accept
    ip saddr 10.77.0.1 tcp dport 22 accept
  }

  chain output {
    type filter hook output priority 0; policy drop;
    ct state established,related accept
    oifname "lo" accept
    ip daddr 10.77.0.1 udp dport { 53, 123 } accept
    ip daddr 10.77.0.1 tcp dport 53 accept
    ip daddr @private4 counter drop
    tcp dport { 80, 443 } accept
  }
}
```
```bash
sudo nft -c -f /etc/nftables.conf && sudo systemctl enable --now nftables
```

### 11.5 Remove or disable what you don't need
```bash
sudo systemctl disable --now snapd.service snapd.socket 2>/dev/null
sudo systemctl disable --now avahi-daemon cups 2>/dev/null
sudo apt purge -y snapd 2>/dev/null
sudo systemctl mask ctrl-alt-del.target
ss -tulpn          # expect only sshd on :22 (and chrony/loopback stuff)
```
Do **not** install `virtualbox-guest-utils`. Verify:
```bash
lsmod | grep -E 'vbox|vboxsf'   # expect nothing
```

### 11.6 Unprivileged service user and directories
```bash
sudo useradd --system --create-home --home-dir /var/lib/lejet --shell /usr/sbin/nologin fetcher
sudo install -d -o fetcher -g fetcher -m 750 /var/lib/lejet /var/lib/lejet/in /var/lib/lejet/out /var/lib/lejet/work
sudo install -d -o root -g root -m 755 /opt/lejet /opt/lejet/browsers
```
`fetcher` has **no sudo, no shell, no password**. You run the job from `ops` via `systemd-run --uid=fetcher`.

### 11.7 Python environment (pinned, hashed)
```bash
sudo python3 -m venv /opt/lejet/venv
sudo /opt/lejet/venv/bin/pip install --upgrade pip
sudo /opt/lejet/venv/bin/pip install --require-hashes -r /opt/lejet/requirements/vm.txt
```
(Copy `requirements/vm.txt` and the `src/lejet` package into `/opt/lejet` via `scp` from the host; code is yours, so it is trusted.)

### 11.8 Install Chromium for Playwright (root, once)
```bash
sudo PLAYWRIGHT_BROWSERS_PATH=/opt/lejet/browsers /opt/lejet/venv/bin/playwright install --with-deps chromium
sudo chmod -R a+rX /opt/lejet/browsers
```

### 11.9 Enable the Chromium sandbox on Ubuntu 24.04 (important)
Ubuntu 24.04 restricts unprivileged user namespaces with AppArmor, which makes Chromium's sandbox fail ("No usable sandbox"). The tempting wrong fix is `--no-sandbox`. **Do not do that.**

Preferred fix: allow user namespaces only for the Playwright Chromium binary. Create `/etc/apparmor.d/lejet-chromium` **[VERIFY exact path; run `ls /opt/lejet/browsers`]**:
```
abi <abi/4.0>,
include <tunables/global>

profile lejet-chromium /opt/lejet/browsers/chromium*/chrome-linux*/{chrome,headless_shell} flags=(unconfined) {
  userns,
  include if exists <local/lejet-chromium>
}
```
```bash
sudo apparmor_parser -r /etc/apparmor.d/lejet-chromium
```
Fallback (weaker, system-wide, acceptable only inside this isolated VM):
```bash
echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/61-lejet-userns.conf
sudo sysctl --system
```
**Playwright's `chromium_sandbox` option defaults to `false`**, which launches Chromium with `--no-sandbox`. Your code must set `chromium_sandbox=True` explicitly (Appendix B). The preflight test (§16 L2) confirms no running Chromium has `--no-sandbox` in its command line.

### 11.10 Run command with resource limits (`infra/analysis/lejet-run.sh`)
```bash
#!/usr/bin/env bash
set -euo pipefail
BATCH_ID="${1:?batch id}"
sudo systemd-run --quiet --wait --collect --pipe \
  --uid=fetcher --gid=fetcher \
  --unit="lejet-${BATCH_ID}" \
  -p MemoryMax=4G -p MemorySwapMax=0 \
  -p CPUQuota=300% \
  -p TasksMax=1024 \
  -p LimitNOFILE=4096 \
  -p PrivateTmp=yes -p NoNewPrivileges=yes \
  -p ProtectSystem=strict -p ProtectHome=yes \
  -p ReadWritePaths=/var/lib/lejet \
  -p TimeoutStopSec=30 \
  -p RuntimeMaxSec=14400 \
  -E PLAYWRIGHT_BROWSERS_PATH=/opt/lejet/browsers \
  -E TMPDIR=/var/lib/lejet/work \
  /opt/lejet/venv/bin/python -I -m lejet.cli run \
    --input /var/lib/lejet/in/urls.txt \
    --db /var/lib/lejet/out/${BATCH_ID}.sqlite \
    --batch-id "${BATCH_ID}"
```
- `RuntimeMaxSec` is a hard ceiling (4 h) so a hung run dies on its own.
- `TMPDIR` points to `/var/lib/lejet/work`; clear it after each batch. (Optionally mount a 512 MB tmpfs there.)
- `python -I` (isolated mode) ignores env vars and the current directory when importing.

### 11.11 Analysis snapshot
After §16 Level 2 passes:
```powershell
& $VBox snapshot lejet-an take "an-configured-clean" --description "Hardened, preflight green"
```
**All batches start from this snapshot.**

---

## 12. Phase 6 — Management access and data transfer

### 12.1 Host SSH config (`%USERPROFILE%\.ssh\config`)
```
Host lejet-gw
  HostName 127.0.0.1
  Port 2201
  User ops
  IdentityFile ~/.ssh/lejet_ed25519
  IdentitiesOnly yes
  ForwardAgent no
  ForwardX11 no
  StrictHostKeyChecking yes
  UserKnownHostsFile ~/.ssh/known_hosts_lejet

Host lejet-an
  HostName 10.77.0.2
  User ops
  ProxyJump lejet-gw
  IdentityFile ~/.ssh/lejet_ed25519
  IdentitiesOnly yes
  ForwardAgent no
  ForwardX11 no
  StrictHostKeyChecking yes
  UserKnownHostsFile ~/.ssh/known_hosts_lejet
```
First connection: `ssh -o StrictHostKeyChecking=accept-new lejet-gw`, then check the fingerprint against what the VM console shows (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`).

### 12.2 Direction rules (memorize)
- **Host → VM only.** The VM can never initiate connections to the host (firewalls drop it, and the host is not on the internal network).
- Only **two kinds of data** cross: `urls.txt` (host → VM) and `<batch>.jsonl.gz` (+ manifest) (VM → host, pulled by the host).

### 12.3 Push the URL list (`scripts/push_urls.ps1`)
```powershell
param([string]$File = "D:\LeJetData\input\urls.txt")
# urls.txt holds ONE URL PER LINE, refanged (https://...), no other text
scp $File lejet-an:/tmp/urls.txt
ssh lejet-an "sudo install -o fetcher -g fetcher -m 640 /tmp/urls.txt /var/lib/lejet/in/urls.txt && rm /tmp/urls.txt"
```
Keep a **defanged** copy (`hxxps://evil[.]example/login`) as the one you view, e-mail or paste anywhere. Only the machine-consumed file is refanged and it never gets opened in a browser.

### 12.4 Export and pull results
On the VM (inside the batch script, after the run), export from SQLite to JSONL.gz:
```bash
/opt/lejet/venv/bin/python -I -m lejet.cli export --db /var/lib/lejet/out/${BATCH_ID}.sqlite \
   --out /var/lib/lejet/out/${BATCH_ID}.jsonl.gz --manifest /var/lib/lejet/out/${BATCH_ID}.manifest.json
sha256sum /var/lib/lejet/out/${BATCH_ID}.jsonl.gz | tee /var/lib/lejet/out/${BATCH_ID}.sha256
```
On the host (`scripts/pull_results.ps1`):
```powershell
param([Parameter(Mandatory)][string]$BatchId)
$Q = "D:\LeJetData\quarantine\$BatchId"
if (Test-Path $Q) { throw "quarantine dir exists, refusing" }
New-Item -ItemType Directory $Q | Out-Null
foreach ($ext in "jsonl.gz","manifest.json","sha256") {
  scp "lejet-an:/var/lib/lejet/out/$BatchId.$ext" "$Q\"
}
$expected = (Get-Content "$Q\$BatchId.sha256").Split(" ")[0]
$actual   = (Get-FileHash "$Q\$BatchId.jsonl.gz" -Algorithm SHA256).Hash.ToLower()
if ($expected -ne $actual) { throw "hash mismatch" }
```
- Use an up-to-date OpenSSH (modern `scp` uses the SFTP protocol). Pull into an **empty, new** directory. Do not use wildcards.
- The hash only proves the file wasn't corrupted in transit; **it does not prove the content is safe** (the VM could be compromised). The validator is what makes it safe.

### 12.5 Host-side validator (`scripts/validate_output.py`, run with `python -I`)
Rules it must enforce:
1. Decompress as a **stream** with a hard cap (e.g., 500 MB decompressed) — zip-bomb protection.
2. Each line ≤ 64 KB, valid UTF-8, valid JSON object.
3. **Strict schema** (pydantic with `extra="forbid"`): known field names only, exact types, numeric ranges, enum for `fail_reason`.
4. Strings: max length per field (e.g., title ≤ 120), strip control chars (`\x00-\x1f`, `\x7f`, bidi controls U+202A–U+202E, U+2066–U+2069), no ANSI escapes.
5. Row count equals number of input URLs (or reconcile with explicit missing list).
6. URL fields stored **defanged** in clean output (`hxxp`, `[.]`) so nobody clicks them from a spreadsheet.
7. Write cleaned data to `D:\LeJetData\clean\<batch>.parquet` (or CSV with formula-escaping below).
8. CSV formula-injection guard: any cell starting with `=`, `+`, `-`, `@`, tab or CR gets a leading `'`.
9. Never `eval`, `pickle`, `yaml.load`, or `shell=True` on any field. Use parametrized SQL if you load into a DB.
10. Never open the quarantine files in Excel, a browser, or a text editor with preview.

---

## 13. Phase 7 — The inspection pipeline (design)

### 13.1 Stages (each stage writes its own row group; a failure in a stage never aborts later independent stages)
```
Stage 0  Normalize + safety gate   (pure string ops, no network)   [v1] + the 14 URL-text features (link_features.py)
Stage 1  DNS                       (via gateway resolver)           [v1] -> dns_record
Stage 1b RDAP                      (registry lookup over HTTPS)     [v1] -> 3 domain features (whois_registered_domain, domain_registration_length, domain_age)
Stage 2  TLS handshake             (https only; read cert)          [v2 optional]
Stage 3  HTTP fetch                (manual redirects, byte caps)    [v1] -> HTML bytes <= 2 MB
Stage 4  Browser render            (Playwright)                     [v2 optional, OFF by default]
Stage 5  Feature extraction        (pure functions)                 [v1] -> 12 static-HTML features + page_fetched
Stage 6  Persist                   (SQLite in VM)                   [v1]
Host     page_rank lookup          (Open PageRank, after export)    [v1] -> page_rank (host side, key never in the VM)

v1 path = 0 -> 1 (original domain, liveness) -> 3 (HTTP fetch, follow chain) -> 1/1b again on the FINAL domain (cached if unchanged) -> 5 -> 6.
No JavaScript is executed in v1. The 14 URL-text features come from the original SMS link (as `normalize_link` shapes it); DNS/RDAP/HTML features and `page_rank` describe the final URL (§13.2b).
```

### 13.2 Stage 0 — Normalization and safety gate
**Two representations of every URL (important):**
- **The link as the SMS contained it, shaped only by `normalize_link()`** (`src/features/link_features.py`: strip trailing punctuation, un-defang `hxxp`/`[.]`, add `http://` if there is no scheme) → used to compute the **14 URL-text features**. The extractor was trained on the public set's `url` column, whose URLs all start with `http://` or `https://`, so this is the only shaping allowed. Lower-casing, punycode conversion, trailing-slash fixes or adding `https://` *before* feature computation would silently change `url_length`, `https_presence`, `number_of_slashes`, etc. and break the parity test (`tests/test_link_features.py`). A schemeless SMS link therefore gets `https_presence = 0` (HTTPS was not stated), even if the fetcher later succeeds over HTTPS.
- **Normalized form** → used **only** to fetch (and for `url_id`).

**Schemeless URLs** (very common in SMS, e.g., `gcash-verify.xyz/login`): add a scheme for fetching only. Try `https://` first, fall back to `http://` once if the HTTPS connect fails (many phishing sites are HTTP-only). Record which scheme worked (`scheme_used`). Strip trailing punctuation that SMS extractors attach (`.`, `,`, `)`) before normalizing.

Reject (record `fail_reason = "rejected_url"`) if:
- Scheme is present and not `http`/`https`.
- Contains credentials (`user:pass@host`), or IDNA/punycode conversion fails.
- Length > 2048 chars; contains whitespace/control chars.
- Host is an IP in a private/loopback/link-local/CGNAT/multicast/reserved range. **Parse numeric hosts with the same rules the OS resolver uses**, not just dotted-quad text: decimal (`2130706433`), hex (`0x7f.0.0.1`), octal (`0177.0.0.1`), short forms (`127.1`), IPv4-mapped IPv6 (`[::ffff:127.0.0.1]`), IPv6 literals, trailing dot (`localhost.`), and names like `localhost`, `*.localhost`, `*.local`, `*.internal`. Reject after canonicalizing with `ipaddress`/`socket.inet_aton`-style parsing. *(Test mode only: `--allow-loopback` for fixtures.)*
- Explicit port other than 80/443: the gateway only allows 80/443, so such URLs can never succeed. Do **not** try them; record `fail_reason = "port_blocked"` (not "dead") so they are not mistaken for takedowns, and still compute the 14 URL-text features from the string (the row stays usable by the `url_only` extractor).
- Host equals the VM's own names/internal IPs.

Also compute and store: normalized URL, registered domain (`tldextract`), punycode flag, `scheme_used`, a stable `url_id`.
- **`tldextract` gotcha:** by default it downloads the Public Suffix List at runtime (an unexpected outbound request, and it fails offline) and writes a cache under the user's home. Construct it with the bundled snapshot and no network: `tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)`. **Private suffixes are ON** so that shared-hosting hosts such as `evil.pages.dev` are their own registrable domain (decision in §13.2c). This registrable-domain function is for fetch-side logic only; the 14 URL-text features are still computed by `link_features.py`, exactly as in training (note that file uses its own private-suffix-**off** extractor for the features and a private-suffix-**on** one only for grouping; do not "fix" that).
- **`url_id` should be generated on the host**, not in the VM: the host computes `url_id = sha256(raw_url)[:16]`, pushes `url_id<TAB>raw_url` lines, and the VM only echoes the id back. This keeps the label join lossless and stops the VM from relabeling rows.
- Deduplicate by `url_id` (keep a host-side record of duplicates and of any URL that appears with two different labels). Shuffle order with a fixed seed (spreads same-host load; keeps reproducibility).

### 13.2b DECISION — which URL the features describe (settles §27 V3)
**Features are taken from the FINAL URL** (the page you actually land on after redirects and shorteners).

| Feature group | Computed on | Why |
|---|---|---|
| The 14 URL-text features | The **original SMS link** (after `normalize_link`) | Parity with training (`tests/test_link_features.py`); this is what the SMS contained and what `url_only` was trained on |
| The 12 HTML features | The **final page**; "internal/external" is relative to the **final URL's registrable domain** | That is the page the victim sees |
| `whois_registered_domain`, `domain_registration_length`, `domain_age`, `dns_record` | The **final URL's registrable domain** | A shortener's own age says nothing about the destination |
| `page_rank` | The **final URL's registrable domain** (host-side lookup) | Same reason; it describes the destination, not the shortener |

Extra columns stored for every row (so the choice can be audited or reversed in analysis):
`final_url` (defanged), `final_reg_domain` (defanged), `final_differs` (0/1), `hop_count`, `chain_complete` (0/1, false if the chain broke before a final page), `scheme_used`, and the **original-domain** versions `orig_whois_registered_domain`, `orig_domain_registration_length`, `orig_domain_age`, `orig_dns_record`.
Optional (v1.1): the 14 URL-text features recomputed on the final URL, prefixed `f_`, to see whether they add signal (never fed to the frozen extractors).

Edge cases:
- **No redirect:** final = original; `final_differs = 0`; the `orig_*` columns equal the main ones (no second lookup).
- **Chain breaks** (dead shortener, loop, blocked hop, timeout): final = the last hop reached, `chain_complete = 0`. If not even the first hop answers, the final domain is the original domain.
- **Redirect to a big benign site** (a cloaking kit sending bots to google.com): the features then describe the benign destination. Flag it with `final_reg_domain` in your list of popular domains (e.g., Tranco top 10k) as `final_is_popular`, and treat those rows as "cloaked/unknown" in analysis rather than as clean benign samples.
- **Ordering changes:** the final domain is only known after the HTTP stage, so the DNS-record and RDAP lookups for the final domain run **after** Stage 3 (the original domain's DNS is still resolved first for the liveness check). When the domain did not change, reuse the cached result, so the cost is only paid for redirected URLs (about +0.2–1.5 s for those).
- **Per-domain politeness** uses the final registrable domain as well.

### 13.2c DECISION — free-hosting and shared platforms (settles §27 V5)
**Problem:** on a free-hosting subdomain (`evil.pages.dev`, `x.github.io`, `x.blogspot.com`) the "registrable domain" by the ordinary public-suffix rules is the *platform* (`pages.dev`), which is old and legitimate. RDAP would then report an old domain and make exactly the sites attackers like look benign.

**Rules:**
1. **Registrable domain = with private suffixes** (§13.2). `evil.pages.dev` → `evil.pages.dev`. Use it for: internal/external link classification, per-domain concurrency and politeness, `final_differs`, deduplication of hosts, and the RDAP decision. (Never for the 14 URL-text features.)
2. **Flag `free_hosting_subdomain = 1`** when the host's suffix is in the Public Suffix List's private section **or** in the curated list `free_hosting_suffixes.txt`. Store the matched suffix in `hosting_platform` (e.g., `pages.dev`, otherwise empty).
3. **RDAP is skipped for flagged hosts:** `rdap_skipped = 1`, the three RDAP features = `-1`, `rdap_ok = 0`. The platform's age says nothing about the site. `dns_record` is still computed on the full hostname. The substitute age signal (Certificate-Transparency first-seen date) is a v1.1 extra (§28 item 6).
4. **Path-hosted platforms** (one hostname shared by all users, the content is in the path, e.g., Google Forms/Sites, Facebook pages, link-in-bio pages): set `platform_path_hosting = 1`, skip RDAP the same way, and **do not** set `final_is_popular` for them (otherwise a phishing form on a Google domain would be mistaken for a "cloaking redirect to a big benign site"). Detect by host + path prefix from a curated list.
5. **Starter list for `free_hosting_suffixes.txt`** (illustrative, **not verified as current — review it and extend it from your own data**):
   - static/serverless hosting: `pages.dev`, `workers.dev`, `vercel.app`, `netlify.app`, `github.io`, `gitlab.io`, `web.app`, `firebaseapp.com`, `onrender.com`, `herokuapp.com`, `glitch.me`, `surge.sh`
   - site builders / blogs: `wixsite.com`, `weebly.com`, `blogspot.com`, `wordpress.com`, `webflow.io`, `carrd.co`, `notion.site`, `canva.site`
   - tunnels / dynamic DNS: `ngrok-free.app`, `ngrok.io`, `trycloudflare.com`, `loca.lt`, `duckdns.org`, `ddns.net`
   - object storage: `s3.amazonaws.com` and regional S3 hosts, `storage.googleapis.com`, `blob.core.windows.net`, `r2.dev`
   - IPFS gateways: `ipfs.io`, `dweb.link`, `cloudflare-ipfs.com`
   - free web hosts: `000webhostapp.com`, `infinityfreeapp.com`, `epizy.com`, `rf.gd`, `byethost*`
   - path-hosted (rule 4): Google Forms/Sites (`docs.google.com/forms`, `forms.gle`, `sites.google.com`), `linktr.ee`, Facebook/`fb.com` pages
6. **New columns:** `free_hosting_subdomain`, `hosting_platform`, `platform_path_hosting`, `rdap_skipped` (listed in §14.0.6).
7. **Analysis rule:** report results with and without these rows, and state in the thesis that the RDAP features are *structurally missing* for them (not randomly missing).
8. **Unit tests** (no network): `evil.pages.dev` → registrable `evil.pages.dev`, flagged; `a.b.github.io` → `b.github.io`, flagged; `x.blogspot.com` → `x.blogspot.com`, flagged; `www.example.co.uk` → `example.co.uk`, not flagged; `login.gcash-verify.xyz` → `gcash-verify.xyz`, not flagged; for every flagged host assert that **no RDAP request is made**.

### 13.3 Stage 1 — DNS
Use `dnspython` with the resolver set to `10.77.0.1`, 3 s timeout, 1 retry. Query A, AAAA, CNAME chain, NS, MX, TXT-count. Record `resolves`, answer counts, min TTL, `nxdomain`, `timeout`, `servfail`, `private_answer_blocked` (the resolver returns nothing for rebinding answers — log as `no_answer`). If no A/AAAA after CNAME resolution → mark `fail_reason = "dns_nxdomain"` (or timeout/servfail) and **skip Stages 2–4** (still emit a row).

### 13.4 Stage 2 — TLS
For `https` only. Connect to `host:443` with `ssl` module, `check_hostname=False`, `verify_mode=CERT_NONE` **solely to read the certificate**, then separately attempt a verified handshake using system CAs to record `cert_valid`. Capture: protocol version, cipher, subject CN, SAN list, issuer org/CN, not-before/not-after, self-signed (issuer==subject), chain length. Timeout 6 s. Never send any application data.

### 13.4b Stage 1b — RDAP (v1: 3 features)
Purpose: `whois_registered_domain`, `domain_registration_length`, `domain_age`. This is a lookup **about the domain at its registry**, not a visit to the site. It runs in the analysis VM over HTTPS through the gateway (port 443 is already allowed). I have not verified RDAP coverage for `.ph` and other country-code TLDs — test it in Level 1.

Procedure:
1. Reduce the host to its **registrable domain** with `tldextract` (e.g., `login.gcash-verify.xyz` → `gcash-verify.xyz`), with private suffixes on (§13.2c). **If the host is a free-hosting subdomain or a path-hosted platform, skip RDAP entirely** (`rdap_skipped = 1`, features `-1`, `rdap_ok = 0`). Run RDAP on the **final** URL's registrable domain (§13.2b) and cache it.
2. Load the IANA RDAP bootstrap file for DNS (`https://data.iana.org/rdap/dns.json`). Cache it in the VM (refresh weekly; fetch it during setup while the network is open). Pick the RDAP base URL for the TLD. If the TLD has no RDAP server → `rdap_ok = 0`, features `-1`.
3. `GET {base}domain/{registrable_domain}` with `httpx`: timeout 4 s, max response 256 KB, `follow_redirects` limited to 2 and only to `https`, accept `application/rdap+json`.
4. Parse `events` (`registration`, `expiration`, `last changed`) and `status`.
5. Compute (all in days; store `-1` when unknown):
   - `whois_registered_domain` = `0` if a record is returned (HTTP 200), `1` if HTTP 404 (no such registration), `-1` if the lookup failed. **1 is the suspicious value**, matching the dataset: in the public set it is `0` for 94.5% of legitimate and 91.0% of phishing rows. (Corrected 2026-10-11; an earlier version of this plan had it reversed.)
   - `domain_age` = now − registration date.
   - `domain_registration_length` = **verify against the paper**: either (expiration − registration) or (expiration − now). Implement the paper's definition, document it, and also store the helper column `days_to_expiry` so you can switch.
6. `dns_record` (from Stage 1) = `0` if the registrable domain or host returns A/AAAA/NS records, `1` on NXDOMAIN/no records, `-1` on resolver error/timeout. **1 is the suspicious value ("no DNS record")**, matching the dataset: in the public set it is `0` for 99.7% of legitimate and 96.3% of phishing rows. (Corrected 2026-10-11; an earlier version of this plan had it reversed, which would have fed every SMS link to the extractor with this feature flipped.)
7. Set `rdap_ok` = `1` only if the lookup completed (200 or 404).

Rules: cache per registrable domain (many URLs share a domain); ≤ 1 request/second per RDAP server; retry once with backoff on 429/5xx; the RDAP JSON is **third-party, untrusted data** — parse with a size cap, never `eval`, store only the derived numbers.

### 13.5 Stage 3 — HTTP probe
`httpx.AsyncClient(follow_redirects=False, http2=False, verify=False, timeout=Timeout(connect=6, read=10, write=6, pool=6), headers={"User-Agent": <mobile UA>, "Accept-Encoding": "gzip", "Accept-Language": "en-PH,en;q=0.8,fil;q=0.6"}, cookies=<fresh in-memory jar per URL, discarded afterwards>)`:
- `http2=False`: HTTP/1.1 only, to avoid the extra `h2` parser attack surface. `Accept-Encoding: gzip` only, so brotli/zstd decoders are never exercised.
- A **per-URL in-memory cookie jar** (not "no cookies"): many sites set a cookie on the first redirect and loop forever without it, which would be misreported as `too_many_redirects`. Nothing persists between URLs.
- Follow redirects **manually**, max 8 hops, re-running Stage 0 safety gate on every hop target (so a redirect can't point to a private IP/odd scheme).
- Stream the response; read **at most 2 MB of *decoded* bytes** (count after gzip decoding, not the compressed size, so a decompression bomb cannot expand past the cap); abort if `Content-Length` > 10 MB.
- Treat as redirects, in addition to HTTP 3xx: `<meta http-equiv="refresh">` and a **trivial** script redirect found by pattern (`location.href = "..."`, `location.replace("...")`, `window.location = "..."` with a string literal). Count each as a hop (shared 8-hop limit) and re-run the safety gate on the target. This catches the very common one-line redirect shells **without running JavaScript**.
- Parsing happens **outside the event loop**: run HTML parsing and regex work in a process pool with a per-task time limit (e.g., 5 s) and kill-on-timeout. Parsing 2 MB of hostile HTML in the event loop can freeze every other URL, and a single catastrophic-backtracking regex can pin a core.
- Record per hop: status, `Location`, response ms, host.
- Final: content-type, content-length, selected security headers (HSTS, CSP, X-Frame-Options, Server, X-Powered-By), `Set-Cookie` count/flags (names are not stored).
- Detect HTML via content-type or sniff first bytes.
- Detect meta-refresh and `window.location` redirects (parsed from first 2 MB; not executed here).

### 13.5b Static HTML feature extraction (v1: 12 features)
Input: the HTML bytes captured by Stage 3 (≤ 2 MB, decoded with the declared charset, falling back to UTF-8 with replacement). Parse with `lxml`/BeautifulSoup. **Nothing is executed and no linked resource is fetched.** None of the 12 features needs script text: they come from tags and attributes (anchors, stylesheets, favicon, media, iframes, forms, title, copyright text), never from running or regex-scanning scripts.

Rules:
- "Internal" means same registrable domain as the final URL (or a relative URL); "external" means a different registrable domain; "null" means empty, `#`, `javascript:void(0)`, `about:blank`.
- Count only what is in the document; never resolve or request the links.
- Cap work: parse at most 2 MB, count at most 20,000 elements, wrap each extractor in try/except returning `-1` and logging `extractor_error`.
- Because many phishing pages are JavaScript-built, a static parse can miss content. Record `html_bytes` and `script_count` helpers so you can measure how often pages are near-empty (candidates for the optional browser stage).
- Concurrency can be much higher than with a browser: 20–30 simultaneous fetches, still 1 per registrable domain at a time.

### 13.6 Stage 4 — Browser render (OPTIONAL — OFF by default in v1)
> Enable only after v1 works end to end, and only if the near-empty-page rate justifies it. All browser safeguards below then apply in full.
Per URL: **new browser context** (incognito-equivalent), closed at the end. Browser process recycled every 50 URLs or on any crash/timeout.

Required context options (see Appendix B for code):
- Device: Playwright's mobile device descriptor (e.g., Pixel 7) → consistent UA/viewport/touch; `locale="en-PH"`, `timezone_id="Asia/Manila"`.
- `accept_downloads=False`, `permissions=[]`, `geolocation=None`, `service_workers="block"`, `java_script_enabled=True`, `ignore_https_errors=True` (cert problems are *recorded*, not fatal).
- Launch args: `--disable-quic`, `--disable-background-networking`, `--disable-extensions`, `--disable-sync`, `--no-first-run`, `--mute-audio`, `--force-webrtc-ip-handling-policy=disable_non_proxied_udp`, `--disable-features=Translate,MediaRouter`. **Not** `--no-sandbox`. `chromium_sandbox=True`.

Event hooks:
- `page.on("dialog")` → dismiss, count.
- `page.on("popup")` → close, count (`popup_attempts`).
- `page.on("download")` → `await download.cancel()`, count (`download_attempts`).
- `page.on("console")` → count errors only (do not store messages).
- `page.on("request")` / `"response"` → count total requests, third-party domains, bytes; **abort the page** if > 300 requests or > 15 MB.
- `page.route("**/*")` → abort resource types `media`, `font`, `websocket`, `eventsource`, `manifest`, `other` if desired (reduces bytes/risk; record that you did so).

Timing:
- `page.goto(url, wait_until="domcontentloaded", timeout=20000)`.
- Then wait for settle: up to 4 s extra for network idle (`wait_for_load_state("networkidle", timeout=4000)` with try/except).
- **Hard per-URL wall clock:** `asyncio.wait_for(..., 45)`; on timeout kill the context (and the browser if it hangs).

Interaction policy: **none**. No clicks, no typing, no scrolling-triggered flows. (A single read-only `scrollTo(0,0)` is not needed.)

Capture after load (in-page, read-only):
- `document.documentElement.outerHTML.length` (cap reading at 3 MB for parsing).
- Run extraction JS that returns **counts and booleans, not raw page strings** (title truncated to 120 chars).
- Final URL after JS redirects.
- Optional (v1.1): a small screenshot kept **inside the VM** only.

### 13.7 Stage 5 — Feature extraction
Pure, deterministic functions over the captured structures. Extraction must never `eval`/execute page content. Use BeautifulSoup/lxml for static HTML features and in-page JS for rendered-DOM features; compare both (static vs rendered) to detect JS-injected forms.

### 13.8 Stage 6 — Persist
SQLite with WAL mode; parametrized statements only; one transaction per URL; resume support (skip `url_id`s already present with `status='done'`). Every URL ends in exactly one terminal state: `done` or `failed:<reason>`.

### 13.9 Concurrency and politeness
| Setting | v1 value |
|---|---|
| Browser contexts in parallel | 3 (max 4) |
| HTTP/DNS/TLS concurrency | 20 |
| Global start rate | ≤ 2 URLs/second |
| Per-registered-domain concurrency | 1 |
| Retries | 1 retry for connect timeout only; none for HTTP errors |
| Random jitter between starts | 0–500 ms |
| Order | seeded shuffle |

### 13.10 Failure taxonomy (`fail_reason` enum)
`rejected_url`, `port_blocked`, `dns_nxdomain`, `dns_timeout`, `dns_servfail`, `dns_no_answer`, `connect_refused`, `connect_timeout`, `read_timeout` (added: connected, but the server stalled or trickled the body past the wall-clock deadline), `tls_error`, `http_4xx`, `http_5xx`, `too_many_redirects`, `redirect_to_blocked`, `non_html`, `too_large`, `challenge_page` (bot wall/CAPTCHA/"Just a moment"), `parked_or_default_page`, `render_timeout`, `render_crash`, `blocked_or_empty`, `internal_error`, `none`.

**Semantics (defined here to avoid ambiguity):**
- `fetch_ok = 1` only when the HTTP stage returned a usable HTML document (`fail_reason = none`). It says nothing about RDAP.
- `rdap_ok` and `dns_ok` are independent flags. A page can be `fetch_ok = 1` with `rdap_ok = 0`, and vice-versa.
- `html_available = 1` iff the 12 HTML features were computed from real HTML (so `fetch_ok = 1` and not a challenge/parked page).
- `page_fetched = html_available`. It is the routing flag for `route_p_u()` (full extractor vs `url_only`) and is never a model input. A challenge/parked page is **not** a fetched page: it must route to `url_only`, otherwise the `full` extractor would score the bot wall.
- `page_rank_ok` is independent of the others (host-side lookup).
- Every URL still gets exactly one row, with every feature present (use `-1`, never omit a key).

---

## 14. Feature catalogue

Naming: `snake_case`. Types: `i` int, `f` float, `b` bool (0/1), `s` short string (capped), `e` enum.

### 14.0 v1 feature set — 31 features (this is what v1 collects)

> **Updated 2026-10-11.** This replaces the earlier 79-feature list (55 URL-string, 20 HTML, 4 RDAP/DNS). The current list is the input of the `full` extractor: `rf_extractor.VARIANTS["full"]` = `URL_TEXT_FEATURES + [page_rank] + WEBPAGE_FEATURES + EXTERNAL_FEATURES`. Sources of truth, in order: `src/features/model/rf_extractor.py` (the exact column names and order), `docs/web_phishing_cleaning.md` (definitions and cleaning decisions), `data/dataset_and_training_plan.md` §4. If this section and those disagree, they win.
>
> **Note on RDAP:** the project notes say RDAP coverage was "verified earlier", but no verification is recorded. Treat RDAP coverage (especially `.ph`) as **unverified** until Level 1 measures the per-TLD success rate.

Column names are the **cleaned set's** names (our renames for the URL group, the dataset's own names for the webpage and external groups). The frozen extractor selects columns by name, so a misspelled or renamed column fails loudly or, worse, silently becomes missing. Anything beyond this list (§28) is an optional extra and must not replace it.

#### 14.0.1 URL-text features — 14 (no network; computed from the link string)
**Already implemented** in `src/features/link_features.py` (`link_text_features(normalize_link(raw))`) and checked against all 11,430 public-set rows by `tests/test_link_features.py` (13 features at 99.95–100%; `random_domain` at about 92%, because the original's exact input is unknown). The inspector calls this code; it does not re-implement it.

| Group | Features |
|---|---|
| Length | `url_length`, `length_hostname`, `path_length` |
| Character counts | `number_of_hyphens`, `number_of_slashes`, `special_char_count` |
| Host structure | `subdomain_count` (the dataset's quirk: dots in the whole URL, capped at 3), `has_ip_address`, `shortened_url_indicator`, `tld_in_path`, `random_domain` |
| Scheme | `https_presence` |
| Keyword counts | `suspicious_url_token_count` (phishing-kit path tokens), `suspicious_keyword_count` (lure words) |

Rules: these are the **same 14 columns the `url_only` extractor uses**, so they are always available, even when the page fails. Sentinel for "no link" is `-1` for all 14 (`no_link_features()`), but the inspector is only ever called with a link.

#### 14.0.2 `page_rank` — 1 (Open PageRank, host-side lookup)
The domain's Open PageRank value for the **final** registrable domain, as in the public set, where it is an integer from 0 to 10 (checked in `web_phishing_clean.csv`: 0 for 2,666 rows, mean 4.48 for legitimate vs 1.89 for phishing pages). It describes the domain *at lookup time*, which is why it belongs to the `full` extractor and not to `url_only`. Looked up on the host (the key stays out of the VM), cached per domain.
- A domain the service answers for but has no rank for → `0` (that is what the training column does for unrated domains; it is a real value, not missing).
- The lookup itself failed (network, quota, bad key) → `-1` and `page_rank_ok = 0`. Never fill failures with `0`, or outages would read as "unrated, suspicious".
- Free-hosting platforms (§13.2c) get the **platform's** rank, which is high and misleading; flag these rows and report results with and without them.
- Check that the API's current scale is still 0–10 integers (round the way the dataset did); a different scale would shift the feature the extractor was trained on.

#### 14.0.3 Webpage features — 12 (from the fetched page source; static parse)
Meanings below are the **intended** meanings; implement each according to the paper's feature table (Hannousse & Yahiouche, 2021) and write down any ambiguity you resolved. They cannot be validated against the dataset (its pages are mostly dead), so test them with fixtures (§16 Level 0). All are unchanged from the source dataset.

| Feature | Meaning (intended) |
|---|---|
| `nb_hyperlinks` | Total hyperlinks in the page (anchors and linked resources per the paper's tag list) |
| `ratio_intHyperlinks` | Share of internal links (same registrable domain / relative) |
| `nb_extCSS` | Number of stylesheets loaded from an external domain |
| `login_form` | A login-style form whose action is suspicious (empty, `#`, `about:blank`, or another domain) |
| `external_favicon` | Favicon loaded from a different domain |
| `links_in_tags` | Share of internal links among links in `<meta>`, `<script>`, `<link>` tags |
| `ratio_intMedia` | Share of media (img/audio/video/embed) loaded from the same domain |
| `ratio_extMedia` | Share of media loaded from external domains |
| `iframe` | An invisible/hidden iframe or frame is present |
| `empty_title` | `<title>` missing or empty |
| `domain_in_title` | The domain name appears in the page title |
| `domain_with_copyright` | The domain name appears next to a copyright mark/text |

**No script-pattern matching is needed any more.** `popup_window`, `onmouseover` and `right_clic` were dropped, so v1 never regex-scans attacker-controlled script text (this also removes the ReDoS concern for those patterns, §27 S4). `sfh`, `ratio_nullHyperlinks` (constant 0 in the public set), `ratio_extHyperlinks`, `submit_email` and `safe_anchor` are also gone. "Null" links are no longer a feature but still matter for the internal/external counting rule in §13.5b.

#### 14.0.4 External features — 4 (RDAP + DNS; see §13.4b)
`whois_registered_domain`, `domain_registration_length`, `domain_age` (RDAP), `dns_record` (DNS). Unknown / lookup failed = `-1`. Negative durations are never stored; they are `-1` (the cleaning step did the same to the public set: 1,837 negative `domain_age` and 46 negative `domain_registration_length` values became `-1`).

**Direction:** `whois_registered_domain` and `dns_record` use `1` for the *abnormal* case (not in WHOIS/RDAP; no DNS record) and `0` for the normal case, as in the dataset. Never "1 = record exists". Add a unit test: a domain that resolves gives `dns_record = 0`; an NXDOMAIN gives `1`. The inspector must follow this direction (`README.md` repeats the warning).

#### 14.0.5 Dropped (do **not** collect)
- Old URL-string features not in the 14 (character counts such as `nb_dots`/`nb_at`/`nb_qm`, word-length statistics, `nb_www`, `nb_com`, `punycode`, `port`, brand and suspicious-TLD features, `nb_redirection`, `nb_external_redirection`, …).
- Page features: `ratio_extHyperlinks`, `ratio_nullHyperlinks`, `submit_email`, `sfh`, `popup_window`, `safe_anchor`, `onmouseover`, `right_clic`.
- Third-party services: `web_traffic`, `google_index`, `statistical_report`. (`page_rank` is the one third-party lookup that stays.)
- Per-link requests: `ratio_intRedirection`, `ratio_extRedirection`, `ratio_intErrors`, `ratio_extErrors`.

#### 14.0.6 Label, routing and bookkeeping columns (not model inputs)
| Column | Meaning |
|---|---|
| `url` | Defanged in exported files |
| `status` | Label (`legitimate`/`phishing`) from the **host-side** label source, never decided in the VM. The cleaned public set uses the same column name |
| `has_url` | Always `1` for an inspected link; the extractors need it as a routing input |
| **`page_fetched`** | **1 iff `html_available = 1`** (a real page was fetched and parsed). `route_p_u()` reads it to choose `full` (1) or `url_only` (0). It is never an input to either forest |
| `label_source`, `label_ts` | Provenance of the label (kept on the host) |
| `message_id` / `url_id`, `batch_id`, `fetch_ts` | Identity and time. For the fresh set also store `received_ts` and `hours_since_received` (the plan requires the delay between message arrival and visit to be recorded) |
| `fetch_ok`, `fail_reason`, `rdap_ok`, `page_rank_ok`, `html_available` | **Indicator columns** — separate "unreachable" from "benign" (liveness bias, §20) |
| `html_bytes`, `script_count` | Helpers to estimate how many pages are near-empty (JS-built) |
| `stage_ms_dns`, `stage_ms_rdap`, `stage_ms_http`, `stage_ms_html`, `duration_ms` | Per-stage timing for the latency analysis (§25) |
| `days_to_expiry` | Helper for the `domain_registration_length` ambiguity |
| `free_hosting_subdomain`, `hosting_platform`, `platform_path_hosting`, `rdap_skipped` | Free/shared-hosting handling (§13.2c); explain why external features are `-1` for those rows |
| `final_url`, `final_reg_domain`, `final_differs`, `hop_count`, `chain_complete`, `scheme_used`, `final_is_popular`, `orig_whois_registered_domain`, `orig_domain_registration_length`, `orig_domain_age`, `orig_dns_record` | Final-URL decision (§13.2b). `final_is_popular` is a bookkeeping flag for spotting cloaking redirects only; the Tranco snapshot was removed as a model input (`data/dataset_and_training_plan.md`, dataset D), so use a frozen, recorded list and never feed the flag to a model |

**Missing-value rule:** `-1` for numeric features whose source was unavailable (all 12 webpage features when `html_available = 0`, RDAP features when `rdap_ok = 0`, `page_rank` when `page_rank_ok = 0`). This is the same `-1` the extractors expect (`extract_p_u` also treats a row of pure `-1` as "nothing known"). Tree models can use `-1` directly. When the page was not fetched, the row is routed to `url_only`, which never reads the webpage, external or `page_rank` columns, so a failed page costs only those columns (the decision doc notes a small loss: a fresh link whose page fails also loses RDAP/DNS even if they worked).

> Sections 14.1–14.8 below are the earlier, broader catalogue. **Only 14.1 (meta fields) and 14.8 (what is not collected) apply to v1.** 14.2–14.7 are v2 extras.

### 14.1 Provenance / meta
| Feature | Type | Meaning |
|---|---|---|
| `url_id` | s | stable hash of normalized URL |
| `batch_id` | s | run identifier |
| `fetch_ts` | s | UTC ISO timestamp of first stage |
| `fetch_ok` | b | complete success flag |
| `fail_reason` | e | see §13.10 |
| `stage_reached` | i | 0–6, last stage that executed |
| `duration_ms` | i | total wall time |
| `vpn_exit_ip_hash` | s | hash of exit IP at batch start (for audit) |
| `ua_profile` | s | e.g., `pixel7` |

### 14.2 Extra lexical ideas (v2 — superseded in v1 by the 14 URL-text features in 14.0.1)
`lex_url_length`, `lex_hyphens`, `lex_slashes`, `lex_subdomain_count`, `lex_special_chars`, `lex_keyword_count`, `lex_shortener`, `lex_suspicious_token_count`, `lex_has_ip_host`, `lex_has_punycode`, `lex_tld`, `lex_digits_ratio`, `lex_path_depth`, `lex_query_params`, `lex_at_symbol`, `lex_double_slash_in_path`, `lex_https_in_hostname` (e.g., `https-bank.example`).

### 14.3 Extra DNS (v2)
`dns_resolves` (b), `dns_a_count` (i), `dns_aaaa_count` (i), `dns_cname_depth` (i), `dns_ns_count` (i), `dns_mx_count` (i), `dns_txt_count` (i), `dns_min_ttl` (i), `dns_nxdomain` (b), `dns_timeout` (b).

### 14.4 TLS (v2)
`tls_present` (b), `tls_version` (e), `tls_cert_valid` (b, verified), `tls_self_signed` (b), `tls_hostname_match` (b), `tls_issuer_org` (s), `tls_issuer_free_ca` (b: Let's Encrypt/ZeroSSL etc.), `tls_cert_age_days` (i), `tls_validity_days` (i), `tls_days_to_expiry` (i), `tls_san_count` (i), `tls_wildcard` (b), `tls_chain_len` (i).

### 14.5 HTTP / redirects (v2)
`http_final_status` (i), `http_hop_count` (i), `http_cross_domain_hops` (i), `http_final_domain_differs` (b), `http_shortener_in_chain` (b), `http_http_to_https` (b), `http_total_ms` (i), `http_content_type` (e), `http_content_length` (i), `http_has_hsts` (b), `http_has_csp` (b), `http_has_xfo` (b), `http_server_present` (b), `http_x_powered_by_present` (b), `http_set_cookie_count` (i), `http_meta_refresh` (b), `http_js_redirect_suspected` (b), `http_body_truncated` (b).

### 14.6 DOM (rendered) (v2 — needs the optional browser stage)
| Feature | Type | Notes |
|---|---|---|
| `dom_title_len` | i | title length only |
| `dom_title_has_brand_kw` | b | title includes PH brand/bank keyword |
| `dom_domain_brand_mismatch` | b | title/brand keyword not in registered domain |
| `dom_html_bytes` | i | |
| `dom_text_len` | i | |
| `dom_text_to_html_ratio` | f | |
| `dom_num_forms` | i | |
| `dom_has_password_field` | b | |
| `dom_num_inputs` | i | |
| `dom_num_hidden_inputs` | i | |
| `dom_form_action_external` | b | any form posts to a different registered domain |
| `dom_form_action_empty_or_hash` | b | `action=""`, `#`, `javascript:` |
| `dom_form_method_get_with_password` | b | |
| `dom_num_links` | i | |
| `dom_ext_link_ratio` | f | |
| `dom_null_link_ratio` | f | `#`, `javascript:void(0)` |
| `dom_num_scripts` | i | |
| `dom_ext_script_ratio` | f | |
| `dom_inline_script_bytes` | i | |
| `dom_obfuscation_hits` | i | counts of `eval(`, `atob(`, `unescape(`, `fromCharCode`, long hex/base64 strings |
| `dom_num_iframes` | i | |
| `dom_hidden_iframe` | b | 0-size or `display:none` |
| `dom_num_images` | i | |
| `dom_ext_image_ratio` | f | |
| `dom_favicon_external` | b | |
| `dom_right_click_blocked` | b | `contextmenu` handlers preventing default |
| `dom_status_bar_spoof` | b | `onmouseover` window.status tricks |
| `dom_urgency_kw_count` | i | EN/Tagalog urgency terms |
| `dom_ph_brand_kw_count` | i | GCash, Maya, BDO, BPI, UnionBank, Landbank, Metrobank, Shopee, Lazada, Globe, Smart, BIR, SSS, PhilHealth, Pag-IBIG, LTO, LTFRB, etc. |
| `dom_login_kw_count` | i | login, sign in, verify, OTP, PIN, password |
| `dom_lang` | e | `<html lang>` |
| `dom_lang_text_mismatch` | b | optional (via simple heuristic) |
| `dom_meta_generator` | s | capped (CMS hint) |
| `dom_title_sha1_prefix` | s | for clustering clones without storing text |
| `dom_html_sha256_prefix` | s | for clone/duplicate detection |

### 14.7 Behavior (observed during render) (v2 — needs the optional browser stage)
`beh_popup_attempts`, `beh_dialog_count`, `beh_download_attempts`, `beh_total_requests`, `beh_third_party_domains`, `beh_third_party_ratio`, `beh_total_bytes`, `beh_console_errors`, `beh_request_cap_hit` (b), `beh_js_redirect_final_differs` (b), `beh_cookies_set_count`, `beh_load_ms`, `beh_websocket_attempts`.

### 14.8 Deliberately NOT collected
Raw HTML, full text, cookies' values, form field values (you never type), screenshots (v1), IP of the VPN exit in clear (hash only), any personal data that appears on pages. See §6 item 2 for the reasoning and the A–D options.

### 14.9 Why there are no screenshots in v1
Screenshots are **not needed** for the 31 features. People take them for: visual similarity to a brand (logo/layout), evidence for a panel, manual error analysis, pages whose content lives in images/canvas, and image-based models. We skip them in v1 because they add risk and cost (image parsing on the host, personal data visible on pages, several GB for 10,000 images), need the browser stage, and a visual model is a separate project.
If added later, the safe version is: take a small screenshot (about 360×640) **inside the VM**, compute a perceptual hash or an embedding from a pretrained image model **inside the VM**, and export only that number/vector — never the image. For a handful of thesis examples (about 10 pages), get adviser approval and view them only in a safe image viewer.

---

## 15. Data model and output format

### 15.1 SQLite inside the VM (`/var/lib/lejet/out/<batch>.sqlite`)
Tables: `runs(batch_id, started_at, finished_at, tool_versions_json, input_sha256, snapshot_name)`, `urls(url_id PK, normalized_url, reg_domain, status, fail_reason, stage_reached, duration_ms)`, `dns(...)`, `tls(...)`, `http(...)`, `hops(url_id, idx, status, host, location_host, ms)`, `dom(...)`, `beh(...)`, `errors(url_id, stage, exc_type, msg_trunc)`.

### 15.2 Export schema (JSONL, one line per URL)
The `features` object holds exactly the 31 model-input columns of §14.0 (cleaned-set names, `-1` for unknown), so it can be loaded straight into the frame that `route_p_u()` takes. `page_rank` is absent in the VM's export and added on the host (§13.1). The remaining fields are bookkeeping.
```json
{"url_id":"a1b2c3d4e5f60718","batch_id":"b001","fetch_ts":"2026-10-12T03:14:15Z",
 "url":"hxxps://example[.]test/login","has_url":1,"page_fetched":1,
 "fetch_ok":true,"fail_reason":"none","rdap_ok":true,"html_available":true,
 "html_bytes":48211,"script_count":7,
 "stage_ms":{"dns":41,"rdap":612,"http":930,"html":38},"duration_ms":1650,
 "features":{"url_length":27,"https_presence":1,"number_of_hyphens":0,"number_of_slashes":4,
             "...":"... the other 10 URL-text, the 12 webpage and the 4 external features ...",
             "nb_hyperlinks":31,"domain_age":-1,"dns_record":0}}
```
The **URL itself** is exported defanged (`hxxps://...`) plus `reg_domain` (defanged).

### 15.3 Run manifest (`<batch>.manifest.json`)
`batch_id`, `started/finished`, `input_sha256`, `n_input`, `n_done`, `n_failed_by_reason`, `git_commit`, `vbox_version`, `kernel`, `python`, `playwright`, `chromium_build`, `snapshot_name`, `vpn_provider`, `vpn_exit_country`, `preflight_pass: true`, `config_hash`.

---

## 16. Phase 8 — Test ladder

> This is the answer to "I want to try web inspection": start at Level 0 and go up. **Do not skip levels.** A level must be green before the next.

### Level 0a — URL-text feature parity test (host or VM, zero network) — mostly done
The 14 URL-text features already have their parity test: `tests/test_link_features.py` recomputes them with `src/features/link_features.py` on the public set's `url` column and compares with `data/processed/web_phishing_clean.csv` (13 features at 99.95–100%, `random_domain` at about 92%, a known and documented gap). Remaining work for the inspector:
1. Run `python -m pytest tests/test_link_features.py tests/test_urlnorm.py tests/test_safety.py` **inside the analysis VM** (same Python and package versions as the host) and confirm the same numbers. `random_domain` depends on `nostril`, so a different version there would silently change a feature.
2. Add a test that the inspector's Stage 0 passes `normalize_link(raw)` (not the fetch-normalized URL) to `link_text_features`, using a schemeless SMS link: `https_presence` must be `0`.
3. Add the registrable-domain / free-hosting unit tests from §13.2c (no network).
4. Keep the dataset's own `url` list as a test fixture only; do **not** open any URL from it.
This test needs no network and no fetching, so run it first.

### Level 0 — Harness tests with local fixtures (zero external contact)
Run **inside the analysis VM** against a local web server on `127.0.0.1` (test mode `--allow-loopback`). Create these mock pages in `tests/fixtures/`:

| Fixture | Purpose | Expected |
|---|---|---|
| `plain.html` | Baseline page | `dom_num_forms=0` |
| `login_form.html` | Form with password input, posts to same origin | `has_password_field=1`, `form_action_external=0` |
| `login_external_post.html` | Password form posting to another domain | `form_action_external=1` |
| `hidden_iframe.html` | 0×0 iframe | `hidden_iframe=1` |
| `meta_refresh.html` | `<meta http-equiv=refresh>` | `meta_refresh=1` |
| `js_redirect.html` | `location.href=...` after 100 ms | `beh_js_redirect_final_differs=1` |
| `redirect_chain` (server) | 302 ×3 | `hop_count=3` |
| `redirect_loop` (server) | infinite 302 | `fail_reason=too_many_redirects` |
| `popup.html` | `window.open` | `popup_attempts≥1` and popup closed |
| `alert.html` | `alert()/confirm()` | `dialog_count≥1`, no hang |
| `download.html` | `Content-Disposition: attachment` | `download_attempts≥1`, **no file on disk** |
| `huge_body` (server) | 50 MB response | capped; `too_large` or `body_truncated=1` |
| `slow_body` (server) | trickles bytes | killed by timeout |
| `obfuscated.html` | `eval(atob(...))` | `obfuscation_hits≥1` |
| `rightclick_block.html` | contextmenu preventDefault | `right_click_blocked=1` |
| `many_requests.html` | 500 tiny image requests | `request_cap_hit=1` |
| `evil_strings.html` | title with `=cmd|...`, ANSI escapes, bidi chars, 10k-char title | output sanitized and capped; validator passes |
| `cpu_loop.html` | `while(true){}` | killed by wall clock; others unaffected |

**Fixtures for the 12 HTML features (v1, static parse — these are the ones that matter first):**

| Fixture | Expected feature values |
|---|---|
| `html_all_internal.html` (10 links, same domain) | `nb_hyperlinks=10`, `ratio_intHyperlinks=1.0` |
| `html_mixed_links.html` (5 internal, 3 external, 2 `#`/`javascript:void(0)`) | `ratio_intHyperlinks` matches the paper's counting rule (settle whether null links stay in the denominator and write it down) |
| `html_ext_css.html` (2 external stylesheets) | `nb_extCSS=2` |
| `html_login_ext_action.html` (form action to another domain) | `login_form=1` |
| `html_login_empty_action.html` (`action=""` / `#` / `about:blank`) | `login_form=1` |
| `html_ext_favicon.html` | `external_favicon=1` |
| `html_media_mix.html` (img/video, internal + external) | `ratio_intMedia`, `ratio_extMedia` correct |
| `html_hidden_iframe.html` (0×0 / `display:none`) | `iframe=1` |
| `html_no_title.html` / `html_empty_title.html` | `empty_title=1` |
| `html_domain_in_title.html` | `domain_in_title=1` |
| `html_copyright_domain.html` | `domain_with_copyright=1` |
| `html_meta_script_link_links.html` | `links_in_tags` correct |
| `html_malformed.html` (broken tags, 5 MB, deep nesting) | no crash; capped; extractor errors logged, `-1` where failed |
| `html_evil_strings.html` | no raw strings in output; validator passes |
| `html_empty.html` (JS-only shell, tiny body) | `html_bytes` small, `script_count>0`, flagged near-empty; `page_fetched` decision documented (§13.10) |
| `parked_or_challenge.html` (bot wall / parked page) | `html_available=0`, `page_fetched=0` (routes to `url_only`) |

Also test the **hand-off to the model**: feed a fixture row (14 URL-text + `page_rank` + 12 + 4 + `has_url` + `page_fetched`) to `route_p_u(url_only, full, frame)` using the frozen extractors, and check that a `page_fetched=1` row reports `extractor == "full"`, a `page_fetched=0` row reports `url_only`, and neither raises on `-1` values. A missing or misspelled column must fail the test, not be skipped.

Pass criteria: all assertions green; zero files created outside `/var/lib/lejet`; no browser/zombie processes remain (`pgrep -f chrome` returns nothing in v1).

**Note:** the browser-oriented fixtures in the first table (popup/alert/download/request-cap/CPU-loop) apply **only if** the optional browser stage is enabled. In v1 they are replaced by the HTML fixtures above plus these server-side ones: `redirect_chain`, `redirect_loop`, `huge_body`, `slow_body`, `evil_strings` (all HTTP-level, no JS).

### Level 1 — Safe external pilot (≈ 20 URLs)
Run only after Level 2. Use well-known benign and TLS-test targets (verify they still exist):
- `https://example.com`, `https://www.wikipedia.org`
- `https://httpbin.org/redirect/3` (redirect chain)
- `https://expired.badssl.com`, `https://self-signed.badssl.com`, `https://wrong.host.badssl.com` (TLS errors)
- `http://neverssl.com` (plain HTTP)
- a couple of Tranco top sites, a URL shortener you control, and a domain that surely does not exist (`nxdomain` test)
- Google's Safe Browsing test pages (designed to trigger warnings; harmless)

Expected: each URL yields exactly one row with plausible features and the right `fail_reason`. Manually compare 5 rows against what a normal browser sees **on the host** for the benign sites only (never for phishing URLs).

### Level 2 — Isolation verification (negative tests that MUST fail)
Run from the analysis VM as `ops` (`infra/analysis/preflight.sh` automates these; the fetcher refuses to start unless the preflight writes a fresh pass token).

| # | Test | Command (examples) | Expected |
|---|---|---|---|
| 1 | Only internal IPv4 present; no IPv6 | `ip -4 a; ip -6 a` | `10.77.0.2` only; no global v6 |
| 2 | Default route only via gateway | `ip route` | `default via 10.77.0.1` |
| 3 | Host via VirtualBox NAT address unreachable | `timeout 3 bash -c '</dev/tcp/10.0.2.2/445'` ; `…/22` | fail |
| 4 | Your home router/LAN unreachable | `timeout 3 bash -c '</dev/tcp/<router-ip>/80'` | fail |
| 5 | Direct DNS to public resolver blocked | `dig @8.8.8.8 example.com +time=2 +tries=1` | timeout |
| 6 | DNS via gateway works | `dig example.com +short` | answer |
| 7 | Rebinding blocked | `dig localtest.me +short` | no answer |
| 8 | Web ports work | `curl -m8 -sI https://example.com` | 200/301 |
| 9 | Non-web ports blocked | `timeout 4 bash -c '</dev/tcp/1.1.1.1/22'` ; `…/25` ; `…/3389` ; `…/8080` | fail |
| 10 | UDP/QUIC blocked | send a UDP packet to port 443, e.g. `python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.settimeout(3);s.sendto(b'x',('1.1.1.1',443));s.recv(10)"` (Ubuntu's `curl` usually has no HTTP/3, so a `curl --http3` test would "fail" for the wrong reason) | timeout/error |
| 11 | ICMP blocked | `ping -c1 -W2 1.1.1.1` | fail |
| 12 | Exit IP is VPN, not home | `curl -s https://api.ipify.org` | ≠ your home IP |
| 13 | **Kill switch**: `sudo wg-quick down wg0` on gateway | repeat #8 | fail; restore with `wg-quick up wg0` |
| 14 | No VirtualBox integration | `lsmod \| grep vbox` ; `ls /media /mnt` | none |
| 15 | Listening sockets | `ss -tulpn` | only sshd on 22 (+ local chrony) |
| 16 | `fetcher` can't escalate | `sudo -l -U fetcher` ; `sudo -u fetcher sudo -n true` | not allowed |
| 17 | `fetcher` can't write outside work dir | `sudo -u fetcher touch /etc/x /opt/x` | denied |
| 18 | Chromium sandbox on | launch via Playwright `chromium_sandbox=True`; `ps -ww -o args -C chrome` | no `--no-sandbox` anywhere |
| 19 | Limits enforced | `systemd-run … -p MemoryMax=512M stress-ng --vm 1 --vm-bytes 2G --timeout 20` | OOM-killed |
| 20 | Fork-bomb contained | run capped `TasksMax=64` with `stress-ng --fork 200` | contained, VM stays responsive |
| 21 | Gateway cannot be reached from analysis except DNS/NTP | `timeout 3 bash -c '</dev/tcp/10.77.0.1/22'` | fail |
| 22 | Clock sane | `chronyc tracking` ; `date -u` | offset < 1 s |
| 23 | Firewall ruleset unchanged | `sudo nft -s list ruleset \| sha256sum` equals recorded hash (use `-s`, stateless: without it the packet counters change the hash every time) | match |
| 24 | Log capture works | gateway `journalctl -k \| grep lejet-egress \| tail` after test #8; also confirm a blocked attempt (#3) appears as `lejet-blocked-private` | entries present |
| 25 | DNS is not threat-filtered | resolve a domain currently listed in a phishing feed, via the gateway and via a known-unfiltered resolver | same answer (no NXDOMAIN) |
| 26 | Stale-clock recovery | after a snapshot restore, run the §17.1 clock step; `wg show` | recent handshake, offset < 1 s |

Record the output of each test in `docs/run_log_template.md`. A single failure = **stop and fix**.

### Level 3 — Small phishing pilot (50 phishing + 50 benign)
- Source: freshest public feed entries (pick URLs listed **within the last 24 h** to maximize liveness) + Tranco benign.
- Run one batch of 100. Watch live: gateway egress log, `htop` on the VM, gateway `tcpdump` byte counts.
- Review outputs: liveness rate, failure-reason distribution, feature sanity (e.g., phishing pages with `login_form = 1`, `page_rank` near 0 for new phishing domains), the share of rows with `page_fetched = 1`, crash rate, mean duration. Run `route_p_u` on the rows and compare the `full` and `url_only` P values; a large systematic gap on the benign half is a warning (class-source bias, §29).
- Check the abort criteria (§18.2) were never tripped.
- Revert snapshot; confirm the VM is clean.

### Level 4 — Scale-up ladder
500 → 2,000 → full 10,000+ in batches (§17). After each step: review failure/crash rates and gateway logs; tune concurrency only if the gateway and VM look healthy.

---

## 17. Phase 9 — Batch operating procedure

Use batches of **≤ 1,000 URLs** (≈ 15–40 min each) so a revert loses little and anomalies are easy to attribute.

### 17.1 Pre-flight (every batch)
- [ ] Host: Windows Update status ok, Defender running, offline backup recent.
- [ ] Docker/WSL stopped.
- [ ] Both VMs powered **off**, then revert to clean snapshots:
  ```powershell
  & $VBox snapshot lejet-gw restore "gw-configured-clean"
  & $VBox snapshot lejet-an restore "an-configured-clean"
  ```
- [ ] Start gateway, then analysis (headless):
  ```powershell
  & $VBox startvm lejet-gw --type headless
  Start-Sleep 40
  & $VBox startvm lejet-an --type headless
  Start-Sleep 40
  ```
- [ ] **Fix both VMs' clocks from the host before anything else.** Restoring a snapshot puts the guest clock back to the snapshot time (there are no Guest Additions to resync it). Besides breaking TLS and `domain_age`, a stale clock can make **WireGuard handshakes fail**: WireGuard uses a timestamp for replay protection and a server can reject an initiation whose time is older than the previous one from the same key. Do this right after both VMs boot:
  ```powershell
  $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  ssh lejet-gw "sudo date -u -s @$now && sudo systemctl restart wg-quick@wg0 chrony"
  ssh lejet-an "sudo date -u -s @$now && sudo systemctl restart chrony"
  ```
  (`sudo` over a non-interactive SSH session needs a scoped passwordless rule, see §27 S14.)
- [ ] `ssh lejet-gw "sudo wg show; chronyc tracking | head -3"` → handshake recent, clock ok.
- [ ] Run `preflight.sh` on the analysis VM: must pass all Level 2 tests.
- [ ] Push `urls.txt` (`scripts/push_urls.ps1`). Record `sha256` and line count.
- [ ] Start gateway packet capture (§10.8).

### 17.2 Run
```bash
ssh lejet-an
./lejet-run.sh b0007
```
Watch from a second terminal:
- Gateway: `sudo journalctl -kf | grep lejet-egress` (rate + destinations)
- Gateway: `sudo conntrack -L | wc -l` (if installed) / `nstat`
- Analysis: `htop`, `df -h /var/lib/lejet`, `pgrep -fc chrome`

### 17.3 Post-run
- [ ] Wait for the process to finish (or stop it; results are resumable).
- [ ] Export JSONL.gz + manifest + sha256 (§12.4).
- [ ] Stop capture on the gateway; copy gateway egress summary: top destination ports, count of connections per minute, any non-80/443 attempts (should be zero by construction).
- [ ] Pull results to quarantine (§12.4), verify hash.
- [ ] Run `python -I scripts/validate_output.py <batch>` → clean data.
- [ ] Update the run log: batch id, counts, failure distribution, anomalies, abort triggers.
- [ ] Power off VMs (`controlvm poweroff`) and **revert to clean snapshots** (so nothing persists to the next batch).

### 17.4 Batch composition rules
- Mix phishing and benign **in every batch** (do not run all phishing first and all benign later; time-of-day effects would confound labels).
- Record the label and label source **on the host**, not in the VM.
- Never include the same registered domain more than N=20 times per batch (for benign URLs the stricter cap of 5 per registrable domain in total, §29, applies).
- Batches are composed from the frozen, matched sample list produced in §29 — do not hand-pick URLs per batch.

---

## 18. Monitoring, abort criteria, incident response

### 18.1 What "normal" looks like
- Gateway egress: only TCP 80/443 new connections; steady rate ≤ ~10/s.
- VM CPU: bursts during render, mostly < 300% (the cgroup cap).
- Disk: slow growth; `/var/lib/lejet/work` stays small.
- Processes: only `python`, `chrome`/`headless_shell`, system daemons.

### 18.2 Abort immediately (hard power-off) if any occurs
1. Any gateway log of a **blocked** attempt to a private IP, to 10.0.2.2, or to a non-80/443 port *originating from the analysis VM* in volumes beyond a few (indicates probing).
2. `wg` handshake lost or exit IP changes unexpectedly mid-run.
3. VM CPU pegged ≥ 95% for > 5 minutes while no URLs are being processed (possible miner).
4. New listening sockets, new users, new systemd units, or modified firewall rules in the VM.
5. Disk usage in `/var/lib/lejet/work` > 1 GB, or any file appearing outside the allowed paths.
6. Outbound traffic volume far above expectation (e.g., > 5× the pilot's per-URL average).
7. Any Defender alert on the host.
8. Any abuse notice/e-mail.
9. VirtualBox host crash/BSOD or unexplained VM behavior.

### 18.3 Response procedure
1. **Hard power off** both VMs: `VBoxManage controlvm lejet-an poweroff`, then gateway.
2. **Do not** restore the snapshot yet. Copy the VM disk files (or take a snapshot named `incident-<date>`) for evidence.
3. Disconnect the laptop from the network if you suspect host impact.
4. Run Windows Defender full scan (and Microsoft Defender Offline scan if suspicious).
5. Rotate: change passwords and review sessions for accounts you used on the host in that period (e-mail, GitHub, OneDrive). Rotate the VPN key and generate a new lab SSH key.
6. Write down: time, URL batch, symptoms, logs. Tell your adviser.
7. Only then restore clean snapshots and, if necessary, rebuild the VMs from scratch.

---

## 19. Phase 10 — Decommissioning and cleanup

After data collection is complete:
- [ ] Power off VMs; delete VM disks: `VBoxManage unregistervm lejet-an --delete`, same for gateway.
- [ ] Securely delete `D:\LeJetData\quarantine\*` (keep only validated `clean/` data and manifests).
- [ ] Delete pcaps and gateway logs.
- [ ] Revoke/rotate the VPN config; delete `wg0.conf` copies.
- [ ] Remove `known_hosts_lejet` entries and the lab SSH key.
- [ ] Keep: scripts, configs, manifests, validation logs, run logs (they support your methodology chapter).
- [ ] Archive an offline copy of `clean/` data + manifests.

---

## 20. Data-quality issues to document in the thesis

1. **Liveness bias:** phishing sites die fast. If `fetch_ok=false` correlates with phishing, a model can learn "dead = phishing". Report results with and without failure-derived features and on a live-only subset.
2. **Temporal consistency:** fetch all URLs in a tight window; record `fetch_ts`; avoid mixing fetch days between classes.
3. **Cloaking:** kits may show benign content to datacenter/VPN IPs, non-mobile UAs, or non-PH locales. You used a mobile UA, `en-PH`, `Asia/Manila`, but your VPN exit country may still affect what you see. Record the exit country; consider repeating a sample without VPN or with a PH exit if the provider offers one.
4. **Headless detection:** some kits detect automation; capture a `blocked_or_empty` rate.
5. **No-interaction limitation:** multi-step flows (OTP pages after submit) are invisible without submitting forms; this is by design for safety.
6. **Label noise:** feeds mislabel; keep `label_source` and `label_ts`; consider a second-source confirmation.
7. **Class balance and leakage:** split by **registered domain** (and ideally by time) to avoid near-duplicate kits across train/test. Clone-detection hashes help.
8. **Reproducibility:** pinned versions, manifests, seeds, config hash.
9. **Ethics/safety statement:** include the architecture, isolation tests and abort criteria (§2, §16, §18).
10. **Sources:** only cite **2020+** references.

---

## 21. Master checklists

### 21.0 How to use these checklists
- Work **top to bottom**; each part (A–H) is a gate for the next one. Do not start fetching (21.2) until every box in 21.1 A–G is ticked.
- Tick a box only when you have the evidence (command output, test result, file hash) saved in `docs/run_log`.
- Items marked **(open)** correspond to unresolved loopholes in §27; decide or build them before the step they block.
- Items marked *(optional)* apply only if you enable the optional browser stage (§13.6) or the v2 extras.

### 21.1 One-time setup

**A. Governance and data policy (§6)**
- [ ] Adviser/university approval for contacting live phishing infrastructure saved
- [ ] Data policy written (option A/B/C/D chosen, §6 item 2) and added to the thesis ethics section
- [ ] Fetcher ethics rules written down (load pages only, no forms, no credentials, no clicks, no downloads)
- [ ] Fetch time window ("freeze window") chosen and recorded
- [ ] Abort criteria and response procedure agreed (§18) before the first run
- [ ] VPN provider's terms checked; support reply saved (§10.4.0)
- [ ] Cited sources for methods are **2020 or later**

**B. Host (§7)**
- [ ] Windows updated; Defender and Tamper Protection on; Controlled Folder Access on for thesis/project folders
- [ ] Offline backup of thesis + code done (not just OneDrive)
- [ ] Home public IP noted (for the VPN test)
- [ ] Lab on the `D:` SSD (≥ 50 GB free there); `D:\VMs\LeJet` and `D:\LeJetData` are outside OneDrive. Windows itself still needs 10–15 GB free on `C:`
- [ ] VirtualBox updated to latest 7.x; no Extension Pack; no Guest Additions in either VM
- [ ] Ubuntu ISO checksum (and signature) verified
- [ ] Lab-only SSH key created with a passphrase; host `known_hosts_lejet` configured
- [ ] Docker Desktop and WSL stopped while the lab runs (`wsl --shutdown`)

**C. Repo and code (§8, §13, §14, §16)**
- [ ] `.gitignore` in place; `wg0.conf`, data, VM images, ISO excluded; only `wg0.conf.example` committed
- [ ] Hashed requirements (`host.txt`, `vm.txt`) generated and installed
- [ ] The 14 URL-text features come from `src/features/link_features.py` (not re-implemented), computed on `normalize_link(raw)` (§13.2); output uses the cleaned set's column names
- [ ] **Level 0a parity test green** (`tests/test_link_features.py`: 13 features ≥ 99.9%, `random_domain` ≈ 92%, also green inside the VM)
- [ ] The 12 HTML feature definitions written down from the paper's feature table, with every ambiguity noted; fixtures pass (§16 Level 0); `page_fetched` set only for real pages
- [ ] RDAP stage implemented: IANA bootstrap cached, `-1` on failure, `rdap_ok`/`rdap_skipped` set; `domain_registration_length` definition settled **(open, V17)**
- [ ] Final-URL rule implemented (§13.2b): `final_*` and `orig_*` columns, `chain_complete`, `final_is_popular`
- [ ] Free-hosting handling implemented (§13.2c): `free_hosting_suffixes.txt` reviewed, private-suffix registrable domain, RDAP skipped for flagged hosts, unit tests pass
- [ ] Stage 0 safety gate tests pass: numeric/obfuscated IP hosts, `localhost` aliases, schemeless URLs, explicit ports (`port_blocked`), credentials in URL, trailing punctuation
- [ ] HTTP stage tests pass: redirect chain/loop, meta-refresh and JS-literal redirects, decoded-size cap (decompression bomb), per-URL cookie jar, `http2=False`, gzip-only
- [ ] HTML parsing runs in a process pool with per-task kill timeout; regexes are bounded/linear-time (ReDoS) (§13.5)
- [ ] `fetch_ok` / `rdap_ok` / `html_available` semantics implemented exactly as §13.10; every URL yields exactly one row with every feature key present
- [ ] Challenge-page / parked / default-page detectors written and their rates reported **(open, V4)**
- [ ] Canary URLs defined (known-answer URLs added to every batch) **(open, V14)**
- [ ] `url_id` generated on the host; push file is `url_id<TAB>raw_url` (§13.2)
- [ ] Per-stage timing (`stage_ms_*`) recorded for latency analysis (§25)
- [ ] Host validator written and tested with malicious fixtures (formula injection, ANSI/bidi strings, oversized lines, zip bomb, extra/missing fields)

**D. Gateway VM (§9, §10)**
- [ ] Built from the ISO; static IPs; IPv6 disabled; SSH key-only; `AllowUsers ops`; `PermitOpen 10.77.0.2:22`
- [ ] WireGuard up with kill switch; `wg0.conf` has no `DNS=`/`PostUp`/`PostDown`; endpoint is an IP
- [ ] Exit IP differs from the home IP; VPN provider's DNS/malware blocking is **off**
- [ ] nftables loaded: default drop, private ranges dropped **and logged**, new-connection rate limit, MSS clamp, egress logging
- [ ] Unbound uses a **non-filtering** upstream; the gateway's own resolver points to local Unbound; rebinding protection verified
- [ ] chrony serves the analysis VM; offset < 1 s
- [ ] Packet-capture and log directories created; log-pull procedure written
- [ ] NIC type set to virtio on both VMs and connectivity re-tested **(open, S11)**

**E. Analysis VM (§9, §11)**
- [ ] Built; internal network only (no NAT/bridged/host-only); IPv6 disabled; no `vbox` kernel modules
- [ ] Local nftables loaded; SSH only from `10.77.0.1`; only expected listeners (`ss -tulpn`)
- [ ] `fetcher` user has no shell/sudo; run script uses `systemd-run` limits; extra sandboxing properties added **(open, S12)**
- [x] Scoped passwordless `sudo` (exact commands only) so scripts work over non-interactive SSH **(S14 done 2026-10-12: root-owned helpers plus one rule, §21.8 item 5.12)**
- [ ] Pinned, hashed Python packages installed; code copied in; `python -I` used
- [ ] `tldextract` configured offline with private suffixes (no runtime download, no cache dir)
- [x] *(optional browser stage)* AppArmor profile or sysctl fallback; Playwright/Chromium installed; `chromium_sandbox=True`; no `--no-sandbox` anywhere (AppArmor profile + self-test, 2026-10-12)

**F. Verification (§16)**
- [ ] Level 0a (URL parity) green
- [ ] Level 0 (local fixtures, incl. all HTML fixtures and server-side HTTP fixtures) green
- [ ] Level 2 isolation tests all pass **including 25 (DNS not threat-filtered) and 26 (stale-clock recovery)**; kill-switch test done manually
- [ ] `preflight.sh` passes and writes a fresh token; the fetcher refuses to start without it
- [x] Level 1 safe external pilot (22 URLs) green (see the Level 1 line in §21.8 for the numbers); RDAP success rate per TLD: .com/.org/.net/.edu/.gov/.uk ok, `.ph` not available (2 hosts), small sample
- [ ] Firewall ruleset hash recorded for the preflight check

**G. Snapshots**
- [ ] Clean snapshots taken after F passes: `gw-configured-clean`, `an-configured-clean`
- [ ] Snapshot refresh schedule set (monthly, 21.4) **(open, S13)**

**H. Dataset preparation (§20, §29)**
- [ ] Phishing URL set frozen: feed names, date range, label source recorded
- [ ] Benign sampling plan executed (§29.12): inputs frozen, brand login list curated, pools built, filters applied, seed fixed
- [ ] Frozen `urls.txt` + host-side `labels.csv` (url_id, label, group A–E, label_source, bin ids) produced
- [ ] Duplicate and label-conflict report produced **(open, V15)**
- [ ] Bias diagnostics run **before** any real model training: nuisance-only AUC, SMD table, feature-importance sanity check (§29.8)
- [ ] Host-side RDAP cache file created **(open, V19)**
- [ ] Level 3 phishing pilot (50 phishing + 50 benign) reviewed; abort criteria never tripped

### 21.2 Before every batch
- [ ] Batch built from the frozen, matched sample list (not hand-picked); ≤ 1,000 URLs; phishing and benign interleaved; ≤ 20 URLs per domain (benign ≤ 5 overall)
- [ ] Canary URLs added to the batch
- [ ] Host: Windows Update/Defender ok; offline backup recent; Docker/WSL stopped
- [ ] Both VMs powered off, then **snapshots reverted**; gateway started first, then analysis
- [ ] **Clock fixed from the host on both VMs** (§17.1), WireGuard restarted
- [ ] VPN handshake recent; **exit IP ≠ home IP**; exit country recorded
- [ ] Upstream DNS still non-filtering (spot check with a listed phishing domain)
- [ ] `preflight.sh` green (fresh token); firewall hash unchanged
- [ ] `urls.txt` pushed; `sha256` and line count recorded; viewing only the **defanged** copy
- [ ] RDAP cache pushed in; packet capture running on the gateway
- [ ] Run log opened: batch id, snapshot names, versions

### 21.3 After every batch
- [ ] Process finished or stopped cleanly; results exported (JSONL.gz + manifest + sha256)
- [ ] Gateway logs and capture summary pulled **before** the revert; non-80/443 or private-range attempts counted (expected ≈ 0)
- [ ] Results pulled into a fresh quarantine directory; hash verified
- [ ] Validator passed; clean data written; row count = input count (or missing list explained)
- [ ] **Canary rows correct**; otherwise discard the batch
- [ ] Batch statistics recorded: dead rate, failure reasons, `challenge_page`/`parked_or_default_page` rate, near-empty-page rate, free-hosting share, RDAP missing rate by TLD, `port_blocked` count, median and p95 latency
- [ ] Updated RDAP cache pulled back to the host
- [ ] Abort criteria reviewed (§18.2); anything unusual written down
- [ ] VMs powered off and **reverted** to clean snapshots
- [ ] Run log completed

### 21.4 Periodic / maintenance
- [ ] **Monthly:** boot the VMs, apply updates, re-run Level 2 + preflight, replace the clean snapshots; record package versions
- [ ] **Monthly:** update VirtualBox; review `free_hosting_suffixes.txt` and the brand lists
- [ ] **Per fetch campaign:** re-verify the VPN terms and rotate the VPN key if it was exposed; record exit country
- [ ] **After any incident:** follow §18.3 before reusing the lab

### 21.5 Gates between milestones
- [ ] **M1 "web inspection works"** (§1.3): isolation suite 100%, fixtures 100%, ≥ 20 safe external URLs complete, one row per URL, validator passed
- [ ] **Before scaling past 500 URLs:** Level 3 pilot reviewed; failure/crash rates acceptable; gateway logs clean
- [ ] **Before the full 10,000+ run:** bias diagnostics passed (§29.8) or the remaining bias is documented; canaries working; RDAP cache working
- [ ] **Before any demo that accepts user URLs (§24):** per-request ephemeral worker, authentication, rate limits and a separate VM from the research lab **(open, S5)**

### 21.6 Rules of the road
**DO**
- Keep everything patched; use the clean snapshot every time.
- Treat *all* output, including RDAP responses and page strings, as hostile data.
- Keep the lab SSH key and VPN config out of git and out of OneDrive.
- Keep URL lists defanged in documents and chats.
- Compute the 14 URL-text features on the **original SMS link** and the web features (HTML, RDAP/DNS, `page_rank`) on the **final** URL.
- Record the label source and the sampling inputs for every row.
- Report results with and without dead/blocked/free-hosting rows.

**DON'T**
- Open any scanned URL in the host browser.
- Run any file that came from the VM.
- Add shared folders, clipboard, bridged/host-only adapters, USB, or Guest Additions.
- Use `--no-sandbox` (if the optional browser stage is enabled).
- Install personal accounts, tokens or API keys in the VMs.
- Use a threat-filtering DNS resolver (it makes live phishing look dead).
- Edit firewall rules ad hoc during a run.
- Use Safe Browsing or feed membership as a model feature (label leak).
- Skip a level of the test ladder.
- Skip the clock step after a snapshot restore.

### 21.7 End of project
- [ ] Final batch accepted; data frozen; manifests archived
- [ ] Decommissioning done (§19)
- [ ] Thesis sections written: architecture and safeguards (§2, §16, §18), data policy (§6), feature definitions and parity result (§14, §16), limitations (§20, §27), sampling and bias diagnostics (§29)

### 21.8 Phase-by-phase progress tracker
Legend: `[x]` done and verified, `[ ]` not done, `[-]` skipped on purpose. Last updated 2026-10-11. Tick a box only with evidence. This groups the work by phase and subphase; 21.1–21.7 above remain the gate checklists.

#### Phase 0: Governance and approvals (§6)
- [x] 0.1 Written adviser approval to contact live phishing infrastructure from an isolated VM (save the reply)
- [x] 0.2 Data policy chosen (option A/B/C/D, default A) and written in the thesis ethics section
- [x] 0.3 Fetcher ethics rules written down (load pages only; no forms, credentials, clicks, downloads, exploitation)
- [x] 0.4 Freeze window (fetch date range) chosen and recorded
- [x] 0.5 URL provenance rule agreed (only dataset or public-feed URLs)
- [x] 0.6 University policy on contacting malicious infrastructure checked
- [x] 0.7 Stop conditions (§18.2) agreed before the first run

#### Phase 1: Host hardening and prep (§7)
**1.1 Windows (§7.1)**
- [x] Windows Update fully applied (no reboot pending; re-check for newer updates)
- [x] Real-time protection on; Tamper Protection on
- [x] Controlled Folder Access on, `D:\Le-Jet` protected (add the thesis documents folder if it is elsewhere)
- [-] Standard (non-admin) daily account (skipped on purpose; UAC is on)
- [x] Offline backup of thesis + code on a USB drive, then unplugged
- [x] Home public IP noted (from the host browser)
- [x] Docker Desktop closed; `wsl --shutdown` run

**1.2 VirtualBox (§7.2)**
- [x] VirtualBox 7.2.20 installed (confirm it is still the latest 7.x before the first run)
- [x] Extension Pack removed
- [x] Guest Additions confirmed absent in the lab VMs (existing VMs show none attached; confirm inside the lab VMs once built)
- [x] Machine folder set to `D:\VMs\LeJet`
- [x] Free disk checked (D: 860 GB, C: 98 GB)

**1.3 Host folders (§7.3)**
- [x] `D:\VMs\LeJet`
- [x] `D:\LeJetData\iso`, `input`, `quarantine`, `clean`, `manifests` created (outside OneDrive)

**1.4 Verify downloads (§7.4)**
- [x] Ubuntu 24.04 server ISO downloaded to `D:\LeJetData\iso` (24.04.5)
- [x] SHA256 matches `SHA256SUMS` (97f3d7ff…fe01cafae0fd8)
- [ ] GPG signature of `SHA256SUMS` checked (skipped for now: HTTPS download from releases.ubuntu.com, hash verified; optional, do later with Gpg4win)

**1.5 Lab SSH key (§7.5)**
- [x] `lejet_ed25519` created with a passphrase; used only for the lab VMs

**1.6 Free disk space on C: (optional, §7.6)**
- [ ] `C:` has about 15 GB free or more (currently 98 GB, fine)

#### Phase 2: Repo skeleton (§8)
- [x] 2.1 New folders and docs created (`docs/run_log_template.md`, `docs/threat_model.md`, `infra/{vm,gateway,analysis}/`, `scripts/`, `tests/fixtures/`)
- [x] 2.1 New `src/lejet/` modules written (`schema.py`, `domains.py`, `settings.py`, `storage.py`, `pipeline.py`, `cli.py`, `parse_pool.py`, `stages/{dns,rdap,http}_stage.py`, `features/{html,external}_features.py`); I used plain dataclasses/dicts instead of pydantic (fewer dependencies in the VM)
- [x] 2.1 Import style settled: `tests/conftest.py` puts `src` and the repo root on `sys.path`; on the VM a `.pth` file in the venv does the same (works under `python -I`)
- [x] 2.2 `git status` shows none of: `wg0.conf`, data, VM disks, ISOs, `.env`, `known_hosts_lejet`
- [x] 2.3a VM lock file with hashes: `requirements/vm.txt` (snapshot) and the authoritative copies inside each bundle, generated by `scripts/build_vm_bundle.py` (offline wheels for Python 3.12 / linux, `nostril` built from its pinned commit) and checked by `scripts/verify_vm_bundle.py` (manifest, hashes, complete dependency closure)
- [ ] 2.3b Host lock file (`host.txt`): not generated; the host still uses `requirements.txt` (only the VM needed the hashed install)
- [ ] 2.3 Versions to record per run noted (pip freeze, kernel, VirtualBox, Chromium if used)

#### Phase 3: Create the two VMs (§9)
- [x] 3.1 `infra/vm/create_vms.ps1` written and run (`lejet-gw`: NAT + `lejet-net`; `lejet-an`: `lejet-net` only)
- [x] 3.2 Settings checked on both VMs (via `VBoxManage showvminfo`): clipboard, drag and drop, VRDE, 3D, audio, USB, serial all off; no shared folders; no Bridged or Host-Only. (Optionally re-check in the GUI.)
- [x] 3.3a `lejet-gw`: Ubuntu Server 24.04 (minimized) installed; user `ops`; OpenSSH; no snaps; ISO ejected
- [x] 3.3b `lejet-an`: Ubuntu Server 24.04 (minimized) installed; user `ops`; no snaps; ISO ejected. OpenSSH: **to confirm via the gateway** (install later through the gateway if missing)
- [x] 3.4a `lejet-gw`: static netplan applied (`10.77.0.1/24`); IPv6 disabled; SSH password and key login tested from the host. Note: after a reboot Ubuntu re-enables the per-interface IPv6 switch and a link-local `fe80::` address appears, so both VMs' netplan files carry `link-local: []` on every interface (verified after reboot on both)
- [x] 3.4b `lejet-an`: static netplan applied (`10.77.0.2/24`, default route via `10.77.0.1`); IPv6 disabled (`/etc/sysctl.d/60-lejet-ipv6.conf`)
- [x] 3.5 Baseline snapshots `gw-os-installed`, `an-os-installed` (taken 2026-10-11, both VMs powered off, ISO ejected)

#### Phase 4: Gateway VM configuration (§10)
- [x] 4.1 `apt full-upgrade` and packages installed (nftables, wireguard, unbound, chrony, tcpdump, unattended-upgrades, jq)
- [x] 4.2 SSH hardened (key-only, `AllowUsers ops`, `PermitOpen 10.77.0.2:22`); key login tested from the host; password login refused. Note: the file is `/etc/ssh/sshd_config.d/00-lejet.conf` (not `99-`), because the installer's `50-cloud-init.conf` sets `PasswordAuthentication yes` and SSH uses the first value it reads
- [x] 4.3 Kernel settings in `/etc/sysctl.d/70-lejet-gw.conf` applied (verified: ip_forward=1, rp_filter=1, redirects/source-route off, log_martians=1, IPv6 off)
- [x] 4.4.0 WireGuard config obtained from provider (Mullvad, Manila `ph-mnl-wg-001`, saved as `D:\LeJetData\wg0.conf`); provider terms checked 2026-10-11; support email sent, **reply pending** (add it to `docs/` when it arrives)
- [x] 4.4.1 WireGuard up with kill switch; no `DNS=`/`PostUp`/`PostDown`; endpoint is an IP; exit IP differs from home IP (exit `129.227.118.172`, Manila, `mullvad_exit_ip: true`; kill switch tested: with `wg0` down the gateway reaches nothing). **MTU note:** big downloads from the analysis VM stalled at the default `wg0` MTU 1420; lowering it to 1280 fixed it. Make it permanent by adding `MTU = 1280` under `[Interface]` in `/etc/wireguard/wg0.conf` (status: done, `MTU = 1280` is now in `wg0.conf` and takes effect at the next tunnel start)
- [x] 4.5 nftables loaded (default drop, private ranges dropped and logged, rate limit, MSS clamp, egress logging); file kept in `infra/gateway/nftables.conf`, enabled at boot. The `lejet-*` log lines still have to be seen in Level 2 tests 24 once the analysis VM generates traffic
- [x] 4.6 Unbound uses a non-filtering upstream (Cloudflare 1.1.1.1/1.0.0.1 over DoT); rebinding protection verified (`localtest.me` returns no answer); config kept in `infra/gateway/unbound.conf`; gateway resolves through 127.0.0.1; queries logged in the journal. Note: Unbound still returns AAAA (IPv6) records to clients even with `do-ip6: no`, so the pipeline must connect over IPv4 only
- [x] 4.7 chrony serves the analysis VM; offset under 1 s (offset 0.0001 s, 6 sources via the VPN; listening on `10.77.0.1:123` only; settings in `/etc/chrony/conf.d/lejet.conf`; the analysis VM's sync from the gateway is checked in Phase 5)
- [x] 4.8a Packet-capture and log directory created (`/var/log/lejet`, mode 750); `tcpdump` tested; journal is persistent. Grep firewall logs with the specific prefixes (`lejet-egress`, `lejet-blocked`, `lejet-ratelimit`), because the hostname `lejet-gw` appears in every journal line
- [ ] 4.8b Log-pull procedure written (script to pull the gateway journal and pcaps before each snapshot revert)
- [x] 4.9 Snapshot `gw-configured-clean` (taken 2026-10-11, VM powered off; automated Level 2 tests were green; reboot of the gateway verified first)

#### Phase 5: Analysis VM configuration (§11)
- [x] 5.1 First boot and updates (via the gateway). Note: the first `apt update` hung on big files until the gateway's `wg0` MTU was lowered to 1280 (see the MTU note under 4.4.1)
- [x] 5.2 Time synced from the gateway (`^* _gateway`, offset about 0.00001 s, also stable with the local firewall on)
- [x] 5.3 SSH hardened; only the gateway may connect (`/etc/ssh/sshd_config.d/00-lejet.conf`: key-only, no root, `AllowUsers ops`, no forwarding; key login works, password login refused with `Permission denied (publickey)`; the local firewall limits SSH to `10.77.0.1`)
- [x] 5.4a Local nftables loaded and enabled (`infra/analysis/nftables.conf`, table `inet lejet_an`); tested: DNS, web via VPN work; `1.1.1.1:22`, `10.0.2.2:445`, `10.77.0.1:22` blocked
- [x] 5.4b Only expected listeners (`ss -tulpn`) checked: sshd :22 (firewall limits it to `10.77.0.1`), chrony control port on loopback :323, systemd-resolved stub on loopback :53; nothing else
- [x] 5.5 Unneeded services removed or disabled (snapd purged, avahi/cups off, `ctrl-alt-del` masked); `vboxguest` (built into the Ubuntu kernel) blacklisted in `/etc/modprobe.d/lejet-no-vbox.conf`, verified after reboot: no `vbox` modules, firewall and chrony came back by themselves. Note: do not run `apt autoremove` blindly (`ubuntu-server-minimal` was removed with snapd)
- [x] 5.6 Unprivileged `fetcher` user (no shell, no sudo) and directories created (verified: `nologin`, locked password, not allowed to run sudo, cannot write to `/etc` or `/opt`, can write to `/var/lib/lejet/work`)
- [x] 5.7a Offline bundle built and verified on the host (19 runtime wheels + 6 test wheels, hashes, closure complete); its test subset passes (373 passed, 1 skipped) with pandas/numpy/scikit-learn blocked, which simulates the VM
- [x] 5.7b Bundle installed on the VM (`scripts/deploy_to_vm.ps1`: pinned wheels, code in `/opt/lejet`, `.pth`, RDAP bootstrap) and the test suite run INSIDE the VM with `python -I`: **375 passed, 1 skipped** (2026-10-12). The first in-VM runs found 3 real problems that the host could not show: (1) the Linux lxml turns garbage such as `<` or NUL bytes into an empty page while the Windows build raises, so `parse_document` now returns `None` when the text has no tag-like start; (2) two pool timeout tests assumed a 2-3 MB page is slow to parse, which is false on a fast VM, so the worker has a test-only `LEJET_TEST_PARSE_DELAY` and `_Worker.call` starts its timer before the page is written; (3) `lejet-export.sh` was missing from the bundle (the packer now fails loudly via the verifier)
- [x] 5.8 *(optional browser stage)* Chromium for Playwright installed (2026-10-12): `playwright==1.63.0` is hash-pinned in the offline bundle; the browser itself is fetched once by `infra/analysis/install_browser.sh` from Playwright's CDN over HTTPS into `/opt/lejet/browsers` (root-owned) and every file is hashed into `/opt/lejet/browsers/SHA256SUMS` (authenticity rests on TLS to the CDN: Playwright publishes no signed hash list)
- [x] 5.9 *(optional browser stage)* Chromium sandbox enabled on Ubuntu 24.04; no `--no-sandbox`: an AppArmor profile (`/etc/apparmor.d/lejet-chromium`, written by the installer) grants `userns` to exactly the Playwright Chromium binaries; `chromium_sandbox=True` and a code guard (`chromium_args_ok`) refuse any sandbox-off switch; `python -I -m lejet.browser_selftest` proves from `/proc` that no Chromium process runs without its sandbox (**BROWSER SELFTEST OK**)
- [x] 5.12 Root-owned helpers and one passwordless sudo rule (`/etc/sudoers.d/91-lejet-helpers`, checked by `visudo` before install): `lejet-sudo-preflight`, `-install-urls` (validates every line of the push file), `-batch`, `-export`, `-shots`, plus `systemctl poweroff`. Replaces the open S14 item: nothing `ops` can edit ever runs as root without a password
- [x] 5.13 `scripts/run_batch.ps1`: the whole run in one command (build push file, push, preflight, run, export, pull, validate, report, optional `-Screenshots`, optional `-Reset`), with the SSH key in the Windows ssh-agent and BatchMode; sets BOTH VM clocks from the host at the start of every run (see the clock finding below)
- [x] 5.14 Finding, VM stability: the analysis VM hung at boot ("Loading essential drivers") with 4 vCPUs every time and about half the time with 2 under VirtualBox's Windows (Hyper-V/NEM) backend; **1 vCPU booted 7 of 7 times**, as the gateway always has. The vCPU count is part of the snapshot, so `an-configured-clean` was retaken at 1 vCPU
- [x] 5.15 Finding, clock: after the host slept, the gateway clock was 6977 s (about 2 h) behind and chrony only slewed. Upstream encrypted DNS then failed intermittently (certificates look "not yet valid"): 14 of 22 pages became `dns_timeout`. Fixed by stepping both clocks from the host; run_batch.ps1 does it every run. Test 26 (stale-clock recovery after a snapshot restore) is covered the same way
- [x] 5.10a `infra/analysis/lejet-run.sh` (refuses to start without a fresh preflight token; `fetcher` user; systemd memory/CPU/task/file limits; read-only file system except `/var/lib/lejet`; 4 h ceiling) and `lejet-export.sh` written and syntax-checked
- [x] 5.10b First real run on the VM (`ssh -t lejet-an ~/lejet-run.sh b0001`, `ops` password for `systemd-run`; 2026-10-12): 22 benign URLs, 18 done, 3 `http_4xx`, 1 `too_many_redirects`. Fixed on the way: `lejet-run.sh`/`lejet-export.sh` tested `-f` on files `ops` cannot see (now `sudo test -f`); `push_urls.ps1` counted lines with a redirect that ran as `ops`
- [x] 5.11a Snapshot `an-configured-clean` (first taken 2026-10-11 before the pipeline; **retaken 2026-10-12** with the pipeline v1 installed, tests green, pilot data wiped, VM powered off). Preflight 18 PASS / 0 FAIL / 8 SKIP, ruleset hash recorded
- [ ] 5.11b Monthly snapshot refresh schedule set (see §21.4)

#### Phase 6: Management access and data transfer (§12)
- [x] 6.1 Host SSH config (`lejet-gw`, `lejet-an` via jump) with `known_hosts_lejet` (`~/.ssh/config`; fingerprints recorded; analysis VM fingerprint checked against the VM console)
- [ ] 6.2 Direction rules understood (host pushes in, host pulls out; the VM never connects to the host)
- [x] 6.3 `scripts/push_urls.ps1` works (file is `url_id<TAB>raw_url`; sha256 and line count recorded). Real push done 2026-10-12: 22 lines, sha256 on the VM equal to the host's
- [x] 6.4 Export and pull procedure works (JSONL.gz + manifest + sha256, pulled into quarantine). Real pull done 2026-10-12: sha256 verified, `VALIDATION OK: 22 rows`, `D:\LeJetData\clean\b0001.clean.jsonl` written
- [x] 6.5 `scripts/validate_output.py` written and tested with malicious fixtures (48 cases: formula injection, ANSI/bidi, oversized lines, zip bomb, extra/missing fields, NaN/Infinity, duplicates, wrong types and ranges, bad sha256, row-count mismatch) and chained with the real exporter (`tests/test_chain_validator.py`)

#### Phase 7: The inspection pipeline (§13, §14, §15)
- [x] 7.1 Stage 0: normalization and safety gate (private/obfuscated IPs, ports, credentials in URL); the gate runs again on every redirect hop. Test-only `allow_loopback` added to `urlnorm.normalize`
- [x] 7.1b Final-URL rule (§13.2b) and free-hosting handling (§13.2c): `final_*` and `orig_*` columns, `final_is_popular` (flag from a frozen list; none supplied yet), RDAP never requested for flagged hosts (tested)
- [x] 7.2 Stage 1: DNS (`dns_record`; direction 0 = has records, 1 = none, -1 = failure; rebinding backstop with `safety.is_ip_allowed`)
- [x] 7.3 Stage 1b: RDAP (IANA bootstrap file, cache, 1 request/s per server, one retry, `-1` on failure, `rdap_ok`/`rdap_skipped`; `domain_registration_length` = days until expiry, settled by the public set's medians). **Live coverage, first measurement (Level 1 pilot b0001, 2026-10-12): RDAP worked for .com/.org/.net/.edu/.gov/.uk hosts (20 of 22 rows) and failed for both .ph hosts tried (`up.edu.ph`, `gov.ph`): the .ph registry has no usable RDAP, so registration features for .ph domains will be -1. Report this as a limitation; more TLDs are still to be measured on a bigger sample**
- [x] 7.4 Stage 3: HTTP probe (manual redirects, decoded 2 MB cap, 10 MB announcement refusal, wall-clock deadlines, per-URL cookie jar, HTTP/1.1, gzip only), tested against a local server
- [x] 7.5 Static HTML features (12): definitions and every ambiguity in `docs/html_feature_definitions.md`; process pool with kill-on-timeout (`parse_pool.py`); linear-time regexes; **to be compared with the paper's feature table by you**
- [x] 7.6 Stage 5: feature extraction; the 14 URL-text features come from `src/features/link_features.py` (not re-implemented), computed on the original link
- [x] 7.7 Stage 6: persist (SQLite WAL, one transaction per URL, resume), JSONL.gz export, manifest, sha256 sidecar
- [x] 7.8 Concurrency and politeness limits set (20 concurrent, 2 starts/s, seeded jitter and shuffle, one request per registrable domain)
- [x] 7.9 Failure taxonomy implemented (plus `read_timeout`); exactly one row per URL with every feature key (tested)
- [ ] 7.10a Detectors: challenge/parked/default pages written and tested on fixtures; **their rates on real data are still to be reported (V4)**
- [ ] 7.10b Canary URLs (known-answer URLs added to every batch) not defined yet (V14)
- [x] 7.11 `page_rank` host-side lookup (`scripts/lookup_page_rank.py`; key from a file outside the repo, never printed; failures are -1; tested against a fake API; **not run against the real service: needs your key**)
- [x] 7.12 *(optional)* Browser render stage = real screenshots (off by default; `run_batch.ps1 -Screenshots`): `src/lejet/stages/shot_stage.py`, `pngcheck.py`, `cli screenshots`, helper `lejet-sudo-shots`, host `validate_shots.py`, `make_gallery.py`, `pull_shots.ps1`. Safeguards: unprivileged `fetcher`, own systemd unit (3 GB, 2048 tasks, 2 h), sandbox on, a fresh context per page, **a local guard proxy (`guard_proxy.py`) is the SSRF guard**: every connection Chromium makes (page, redirects, sub-resources, WebSockets) goes through one proxy per page that resolves each name itself and refuses, at connect time, anything that is not a public address on port 80/443 (any non-public answer refuses the whole name: DNS rebinding; numeric host tricks such as `2130706433` or `0x7f.1` refused; 80 adversarial tests). Design history, both found by real-browser runs: (1) an in-browser request filter never saw redirect targets (Chromium follows redirects natively; the connection to a private address was stopped only by the VM firewall); (2) the fix of re-fetching every request through Playwright's Node client (`route.fetch`) broke bot-protected phishing pages (`ECONNRESET`, different TLS fingerprint): only 36 of 88 attempts captured on the smishing batch, 24 of the failures were pages the plain fetch loaded. The proxy keeps Chromium's native network stack and puts the check where it cannot be bypassed. The 10 links that failed under (2) all captured under the proxy. Downloads/popups/dialogs/permissions blocked, strict PNG structure check in the VM and again on the host, archive validated member by member, inert image-only gallery. 701 host tests + 69 real-browser/stage tests in the VM. First real run: b0006, 22 benign URLs, 17 fetched, **14 of 17 screenshots in the first run, 17 of 17 after a fix to the request handler** (checked by hand in one browser session); python.org and bbc.co.uk looked exactly like the live pages. **Capture-all rule (b0007): every URL that is safe to open gets an attempt, not only those the plain fetch loaded** (`capture_target`: skipped only for refused-by-gate and non-existent DNS names; failed TLS tolerated; one retry): **21 of 22 captured**; the one miss, up.edu.ph, is a true redirect loop. The block pages are evidence: gov.ph shows Cloudflare "Error 1005: the owner has banned the ASN (21859)", i.e. some sites refuse the VPN provider's whole address range (a limitation to report; plain-fetch 403s on gov.ph, w3.org, inquirer.net came from this kind of blocking). `scripts/sync_results.ps1` copies validated reports, data and screenshots into `D:\Le-Jet\results\<batch>\` (git-ignored except the .md reports)
- [x] 7.15 Team smishing links (two spreadsheets, 113 links, 2 duplicates, 111 unique, 36 shorteners) cleaned by `scripts/clean_smishing_links.py` (invisible characters, newlines, defanged text, the classic `trusted.com@evil.com` form kept and flagged, duplicates merged, reviewer names not copied) and run through the lab (batch smish01: 47 loaded, 32 reachable but blocked, 32 not live). **The team's labels vs the lab:** of 29 "live", 18 load; of 48 "dead", 10 load; of 11 "unsure", 7 load. `scripts/build_smishing_workbook.py` writes one workbook (Summary, Cleaned, Live, Reachable-blocked, Not live, Unsure) with every URL defanged and a caveat column: 7 of the 47 loaded pages are on free hosting platforms, end on the shortener's own page, or end on a very popular site (page rank 9-10), so they are probably not usable phishing pages. `edgeone.dev`, `edgeone.app`, `eu.cc` were added to the free-hosting list from this data. **Second run (smish02, guard-proxy screenshots): 49 loaded, 33 reachable-blocked, 29 not live; 80 of 88 screenshot attempts captured** (23 links not attempted: 19 names that do not exist in DNS, 4 redirects into forbidden space; the 8 failures: 5 silent servers, 1 TLS error, 2 other). **Reproducibility: the plain-fetch result was identical for 109 of the 111 links across the two runs** (the 2 differences were sites that erred in run 1 and loaded in run 2). Page rank answered for 42 domains. Screenshots: `D:\LeJetData\clean\shots\smish02\index.html`; workbook: `smishing_links_cleaned.xlsx`
- [x] 7.16 Embedded-link defang bug (2026-10-13): `safety.defang_url` only defanged a leading scheme, so a sign-in page whose address carries `?continue=https://...` left a clickable link and the strict validator rejected the whole batch (`benign01`, 1 of 226 rows). Fixed in `src/lejet/safety.py` (every `http(s)://` is defanged; `defang_embedded_urls`, idempotent) with tests; **the VM still runs the older build until `scripts\deploy_to_vm.ps1` is run again (needs the ops password)**. The one rejected export was repaired on the host with `scripts/repair_embedded_links.py` (rewrites only `http(s)://` inside text fields, writes a NEW file, validator unchanged and still has to pass), then validated normally.
- [x] 7.17 Live set + head training + serving (2026-10-13), see `docs/live_set_and_serving.md`: `src/data/build_live_set.py` (domain-grouped `live_train` / `fresh_test`, overlap and leak checks), `src/features/model/live_training.py` + `run_experiment --live-train ... --tag _live` (Phase C only; CNN, gates, extractors untouched; real-SMS scores unchanged within noise), `src/data/eval_live_set.py` (fresh_test only, Wilson intervals), `src/serving/api.py` (frontend `/predict` contract, inspected-links store, screenshot endpoint, never opens a link). Batches used: `smish03` (111 team links, screenshots) + `benign01` (226 generated legitimate links).
- [x] 7.18 Live inspection from the demo (2026-10-13): `src/serving/live_inspect.py` + `POST /inspect`, `GET /inspect/<job>`. A link that is not in the store is handed to the unattended lab pipeline (`scripts/run_batch.ps1 -Screenshots`: isolation self-test, fetch in the analysis VM, screenshot, pull, strict validator, host-side repair of an embedded clickable link, page rank) and never opened by the API or the browser. One job at a time; measured 157 s and 165 s per link (about 75 s is the isolation self-test, kept). The result goes to `data/live/on_demand_links.csv` (never read by training/evaluation). The frontend Checker shows the job's real steps and times; with the lab off it says the link was not opened. Also fixed: `dns_record` / `whois_registered_domain` were labelled the wrong way round (0 = normal). Tests: `tests/test_live_inspect.py`, `frontend/src/components/Checker.lab.test.jsx`. Needs both VMs running and the SSH key loaded.
- [x] 7.14 `page_rank` (host side, Open PageRank): `scripts/lookup_page_rank.py` now reads the key from a `.env` file outside the repo (`--env-file D:\LeJetData\secrets\.env`; the variable `OPENPAGERANK_API_KEY`, then `--key-file`; the value is never printed or logged, bad files are refused without echoing it; 32 tests). **Run for real on 2026-10-12 against batch b0007: 22 of 22 domains ranked, 0 failed batches**, range 7 to 10 for benign sites; the output `b0007.model_input.csv` has all 31 model columns. Free-hosting platform rank caveat (§14.0.2) still applies
- [x] 7.13 Certificate-age columns for TLDs without RDAP (`ct_first_cert_days`, `ct_ok`, `registration_unavailable_tld`; crt.sh with a start-of-run health probe, one request at a time, no retry after a timeout, circuit breaker). `.ph` has neither RDAP nor port-43 WHOIS (IANA's WHOIS field for `.ph` is empty); the 13 no-RDAP rows of the b0002-b0004 pilots (11 `.ph`, 2 `.jp`) all got `registration_unavailable_tld = 1`. **crt.sh itself was down (502 / timeouts, also from the host) on 2026-10-12, so the certificate age has not yet been seen on real data**; Cert Spotter's free index only goes back to about September 2025, so it is not a substitute
- [x] 7.13 Per-stage timing (`stage_ms_*`) recorded

#### Phase 8: Test ladder (§16). Do not skip levels.
- [x] Level 0a: URL-text parity test green on the host (24 tests + 41 golden-value tests, `tests/test_link_features_golden.py`) **and inside the VM** (the golden test is the pandas-free version that runs there; it passed inside the VM on 2026-10-12 as part of the 375 tests)
- [ ] Level 0: local-fixture tests green. Host: the complete suite passes (HTML fixtures, server-side HTTP fixtures, DNS/RDAP fakes, storage, CLI, validator, `route_p_u` hand-off with tiny forests). **Inside the VM: still to do** (`deploy_to_vm.ps1` runs it)
- [ ] Level 2: all 26 isolation tests pass (including 25 DNS not filtered, 26 stale-clock recovery); kill switch tested by hand. Status 2026-10-11: automated tests 18 PASS / 0 FAIL / 8 SKIP; 24 done by hand (with the analysis firewall flushed for a few seconds, the gateway logged `lejet-blocked-private 10.77.0.2 -> 10.0.2.2:445`, `lejet-blocked-other ... -> 1.1.1.1:22` and `lejet-egress ... :443`); still to do by hand: 25 (DNS not filtered), 26 (stale clock after a snapshot restore); 13 (kill switch) was shown on the gateway; 4, 11, 12 optional; 18 N/A in v1
- [ ] `preflight.sh` passes and writes a fresh token (done: `infra/analysis/preflight.sh`, token in `/run/lejet-preflight-ok`); the fetcher refuses to start without it (to do: check the token in `lejet-run.sh`, §11.10)
- [x] Firewall ruleset hash recorded (analysis VM: `/etc/lejet-ruleset.sha256` from `nft -s list ruleset`; gateway: `/etc/lejet-ruleset.sha256` from `nft -s` of its own tables `inet lejet` and `ip lejet_nat` only, because `wg-quick` adds its own table, which would change the hash after a reboot)
- [x] Level 1: safe external pilot (22 benign URLs, batch b0001, 2026-10-12): pushed, run, exported, pulled and validated end to end; 18 done, 3 `http_4xx` (w3.org, inquirer.net, gov.ph answered 403 to the bot), 1 `too_many_redirects` (up.edu.ph, 9 hops: to be checked whether a real loop). RDAP: 20/22 ok, both .ph hosts failed (no RDAP for .ph). Only 2 `.ph` hosts and 1 run: small sample
- [ ] Level 3: phishing pilot (50 phishing + 50 benign) reviewed; abort criteria never tripped
- [ ] Level 4: scale-up 500, then 2,000, then the full 10,000+ run

#### Phase 9: Batch operating procedure (§17, §18)
**9.1 Before every batch**
- [ ] Batch built from the frozen matched list (≤ 1,000 URLs, interleaved, ≤ 20 per domain); canaries added
- [ ] Host checks: Windows Update/Defender ok, offline backup recent, Docker/WSL stopped
- [ ] VMs off, snapshots reverted, gateway started first
- [ ] Clocks fixed from the host; WireGuard restarted
- [ ] VPN handshake recent; exit IP differs from home IP
- [ ] Upstream DNS still non-filtering; `preflight.sh` green; firewall hash unchanged
- [ ] `urls.txt` and RDAP cache pushed; packet capture running; run log opened

**9.2 Run and monitor (§17.2, §18.1)**
- [ ] Gateway egress only TCP 80/443, about 10/s or less; VM CPU/disk/processes as expected

**9.3 After every batch**
- [ ] Results exported, gateway logs pulled before the revert, results in quarantine with hash verified
- [ ] Validator passed; row count = input count; canary rows correct
- [ ] Batch statistics recorded; RDAP cache pulled back
- [ ] Abort criteria reviewed (§18.2); VMs powered off and reverted; run log completed

**9.4 Incident response (§18.3), only if an abort trigger fires**
- [ ] Hard power off, preserve evidence, disconnect if host impact is suspected, full Defender scan, rotate credentials and keys, tell the adviser, then restore

#### Phase 10: Dataset preparation (§20, §29)
- [ ] Phishing URL set frozen (feeds, date range, label source)
- [ ] Benign sampling plan executed (§29.12); seed fixed
- [ ] Frozen `urls.txt` and host-side `labels.csv` produced
- [ ] Duplicate and label-conflict report
- [ ] Bias diagnostics run before any model training (nuisance-only AUC, SMD table, importance check)
- [ ] Data-quality issues for the thesis documented (§20)

#### Phase 11: Decommissioning and cleanup (§19)
- [ ] VMs powered off and deleted (`unregistervm --delete`)
- [ ] `D:\LeJetData\quarantine\*` securely deleted (keep `clean/` and manifests)
- [ ] pcaps and gateway logs deleted
- [ ] VPN config revoked/rotated; `wg0.conf` copies deleted
- [ ] `known_hosts_lejet` entries and lab SSH key removed
- [ ] Offline copy of `clean/` data + manifests archived

---

## 22. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| VirtualBox: `VERR_NEM_VM_CREATE_FAILED` or very slow VM | Windows hypervisor on (WSL2/Docker/Memory Integrity) | `wsl --shutdown`, quit Docker Desktop; if needed disable Hyper-V features or `bcdedit /set hypervisorlaunchtype off` (reboot; re-enable later for WSL/Docker) |
| VM can't see the other VM | Internal network names differ | Both must use the exact name `lejet-net`; check `VBoxManage showvminfo` |
| Analysis has no DNS | Unbound not bound to `10.77.0.1`, or resolved occupies :53 | `ss -ulpn \| grep :53`; disable the stub listener |
| Gateway itself cannot resolve names (apt/chrony fail) | Its resolver still points at the blocked NAT DNS `10.0.2.3` | Apply the resolved/Unbound change in §10.6 |
| Small pages load, big ones hang | MTU/MSS problem over WireGuard | Make sure the MSS-clamp rule from §10.5 is loaded |
| WireGuard handshake fails only after a snapshot revert | Stale guest clock (replay-protection timestamp) | Run the clock step in §17.1, then restart `wg-quick@wg0` |
| Almost every phishing domain looks dead | Threat-filtering DNS upstream | Switch Unbound to a non-filtering resolver (§10.6) |
| Analysis gets NXDOMAIN for valid sites | Gateway VPN down / DoT blocked | `wg show`; check nft `output` allows `oifname wg0` |
| `wg` handshake never completes | Wrong endpoint/port, firewall, time skew | Check nft endpoint rule, `chronyc tracking`, provider config |
| TLS errors everywhere | Clock skew | Fix chrony on both VMs |
| Playwright: "No usable sandbox" / `Operation not permitted` on userns | Ubuntu 24.04 AppArmor userns restriction | Apply AppArmor profile (§11.9); don't use `--no-sandbox` |
| Chromium crashes in the VM | Low `/dev/shm` or RAM | Increase VM RAM; set `--disable-dev-shm-usage` (this flag is safe) |
| `systemd-run` fails with property errors | Unit property names/ systemd version | Check `man systemd.resource-control`; remove unsupported property |
| Can't SSH to analysis | ProxyJump / `PermitOpen` / firewall | Gateway `PermitOpen 10.77.0.2:22`; analysis input rule from `10.77.0.1` |
| Downloads leave files on disk | Context not created with `accept_downloads=False` or handler missing | Fix code; re-run Level 0 `download.html` |
| Very low liveness | Old URLs / cloaking | Use feeds ≤ 24 h old; record liveness stats |
| Windows host low on disk | VDIs grow dynamically | Compact: `VBoxManage modifymedium disk <vdi> --compact` after zero-fill; move to external SSD |

---

## 23. Appendices

### Appendix A — Preflight script skeleton (`infra/analysis/preflight.sh`)
```bash
#!/usr/bin/env bash
# Writes /var/lib/lejet/preflight.ok (with timestamp) only if ALL checks pass.
set -u
fail=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }
expect_fail() { # description, command...
  desc="$1"; shift
  if "$@" >/dev/null 2>&1; then bad "$desc (should have FAILED)"; else ok "$desc"; fi; }
expect_ok() {
  desc="$1"; shift
  if "$@" >/dev/null 2>&1; then ok "$desc"; else bad "$desc (should have WORKED)"; fi; }

expect_fail "host NAT addr 10.0.2.2:445"   timeout 3 bash -c '</dev/tcp/10.0.2.2/445'
expect_fail "host NAT addr 10.0.2.2:22"    timeout 3 bash -c '</dev/tcp/10.0.2.2/22'
expect_fail "gateway ssh from analysis"    timeout 3 bash -c '</dev/tcp/10.77.0.1/22'
expect_fail "direct DNS 8.8.8.8"           dig @8.8.8.8 example.com +time=2 +tries=1 +short
expect_fail "ssh outbound 1.1.1.1:22"      timeout 4 bash -c '</dev/tcp/1.1.1.1/22'
expect_fail "smtp outbound 1.1.1.1:25"     timeout 4 bash -c '</dev/tcp/1.1.1.1/25'
expect_fail "ping 1.1.1.1"                 ping -c1 -W2 1.1.1.1
expect_ok   "https example.com"            curl -m8 -sI https://example.com
expect_ok   "dns via gateway"              bash -c 'dig example.com +short | grep -q .'
# rebinding: must return NO answer
if [ -n "$(dig localtest.me +short +time=3 +tries=1)" ]; then bad "rebinding protection"; else ok "rebinding protection"; fi
# exit IP must differ from home IP (set HOME_IP in env or a file the host provides)
EXIT_IP="$(curl -m8 -s https://api.ipify.org || true)"
if [ -z "${HOME_IP:-}" ]; then bad "HOME_IP not provided"; elif [ "$EXIT_IP" = "$HOME_IP" ] || [ -z "$EXIT_IP" ]; then bad "exit IP is home/unknown"; else ok "exit IP differs ($EXIT_IP)"; fi
# no vbox modules
if lsmod | grep -qi vbox; then bad "vbox modules loaded"; else ok "no vbox modules"; fi
# listening sockets: only sshd(22) + loopback/chrony
if ss -tulnH | awk '{print $5}' | grep -vE '^(127\.|\[::1\]|10\.77\.0\.2:22$|\*:123$|.*:323$)' | grep -q .; then bad "unexpected listeners"; else ok "listeners"; fi
# firewall unchanged
if [ -f /opt/lejet/nft.sha256 ]; then
  [ "$(sudo nft -s list ruleset | sha256sum | cut -d' ' -f1)" = "$(cut -d' ' -f1 /opt/lejet/nft.sha256)" ] && ok "nft hash" || bad "nft hash differs"
else bad "no recorded nft hash"; fi
# clock
off=$(chronyc tracking | awk '/System time/ {print $4}'); awk -v o="$off" 'BEGIN{exit !(o<1.0)}' && ok "clock offset ${off}s" || bad "clock offset"

if [ $fail -eq 0 ]; then date -u +%FT%TZ | sudo tee /var/lib/lejet/preflight.ok >/dev/null; echo "ALL PASS"; else sudo rm -f /var/lib/lejet/preflight.ok; echo "PREFLIGHT FAILED"; exit 1; fi
```
Notes: the fetcher's CLI must check that `/var/lib/lejet/preflight.ok` exists and is **< 30 minutes old**, otherwise refuse to run. Also add the kill-switch test (Level 2 #13) as a manual step that you perform from the host (it requires toggling the gateway).

### Appendix B — Playwright configuration sketch (`stages/browser_stage.py`) — OPTIONAL, not used in v1
```python
import asyncio
from playwright.async_api import async_playwright

LAUNCH_ARGS = [
    "--disable-quic",
    "--disable-background-networking",
    "--disable-extensions",
    "--disable-sync",
    "--no-first-run",
    "--mute-audio",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--disable-features=Translate,MediaRouter",
    "--disable-dev-shm-usage",
]
BLOCK_TYPES = {"media", "font", "websocket", "eventsource", "manifest"}
MAX_REQUESTS = 300
MAX_BYTES = 15 * 1024 * 1024

async def inspect(pw, browser, url, device):
    ctx = await browser.new_context(
        **device,                       # e.g. pw.devices["Pixel 7"]
        locale="en-PH",
        timezone_id="Asia/Manila",
        accept_downloads=False,
        permissions=[],
        service_workers="block",
        ignore_https_errors=True,
        java_script_enabled=True,
    )
    stats = dict(requests=0, bytes=0, dialogs=0, popups=0, downloads=0, console_err=0, cap_hit=False)
    page = await ctx.new_page()

    async def route(r):
        if r.request.resource_type in BLOCK_TYPES:
            await r.abort()
        else:
            await r.continue_()
    await page.route("**/*", route)

    page.on("dialog", lambda d: (stats.__setitem__("dialogs", stats["dialogs"] + 1), asyncio.create_task(d.dismiss())))
    page.on("popup", lambda p: (stats.__setitem__("popups", stats["popups"] + 1), asyncio.create_task(p.close())))
    page.on("download", lambda dl: (stats.__setitem__("downloads", stats["downloads"] + 1), asyncio.create_task(dl.cancel())))
    page.on("console", lambda m: stats.__setitem__("console_err", stats["console_err"] + (m.type == "error")))
    # request/response counters: on "request" increment; on "response" add content-length; set cap_hit and close ctx when exceeded

    try:
        await asyncio.wait_for(_load_and_extract(page, url, stats), timeout=45)
    finally:
        await ctx.close()
    return stats

async def main(urls, device_name="Pixel 7"):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            chromium_sandbox=True,      # IMPORTANT: default is False (= --no-sandbox)
            args=LAUNCH_ARGS,
        )
        # ... process URLs with a semaphore of 3, recycle browser every 50 URLs ...
```
Key points: `chromium_sandbox=True`; never `--no-sandbox`; fresh context per URL; the extraction JS returns counts/booleans only; the whole thing wrapped in a wall-clock timeout; a separate watchdog kills a hung browser.

### Appendix C — In-page extraction JS (read-only, returns counts only)
Sketch of what `page.evaluate` should compute (no network, no mutation, no event dispatch):
```js
() => {
  const q = (s) => document.querySelectorAll(s);
  const host = location.hostname;
  const regDomain = (h) => h.split('.').slice(-2).join('.');      // replace with tldextract result passed in
  const forms = [...q('form')];
  const ext = (u) => { try { return new URL(u, location.href).hostname !== host } catch { return false } };
  return {
    title_len: (document.title || '').length,
    num_forms: forms.length,
    has_password: q('input[type=password]').length > 0,
    num_inputs: q('input,textarea,select').length,
    num_hidden: q('input[type=hidden]').length,
    form_action_external: forms.some(f => f.getAttribute('action') && ext(f.getAttribute('action'))),
    form_action_empty_or_hash: forms.some(f => ['','#'].includes((f.getAttribute('action')||'').trim()) || /^javascript:/i.test(f.getAttribute('action')||'')),
    num_links: q('a[href]').length,
    num_scripts: q('script').length,
    num_iframes: q('iframe').length,
    hidden_iframe: [...q('iframe')].some(i => { const r = i.getBoundingClientRect(); return (r.width<2 && r.height<2) || getComputedStyle(i).display==='none' }),
    num_images: q('img').length,
    text_len: (document.body?.innerText || '').length,
    html_bytes: document.documentElement.outerHTML.length,
    lang: (document.documentElement.lang || '').slice(0,10),
  };
}
```
Keyword counting (urgency/brand/login) runs in Python on `innerText` truncated to 200 KB; the text itself is never stored.

### Appendix D — Per-URL SQLite write (parametrized)
```python
cur.execute(
  "INSERT OR REPLACE INTO urls(url_id, normalized_url, reg_domain, status, fail_reason, stage_reached, duration_ms) VALUES (?,?,?,?,?,?,?)",
  (url_id, norm_url, reg_domain, status, fail_reason, stage, dur_ms),
)
```
Never f-string SQL. Never build shell commands from URL contents.

### Appendix E — Command cheat sheet (host)
```powershell
# start / stop
& $VBox startvm lejet-gw --type headless
& $VBox startvm lejet-an --type headless
& $VBox controlvm lejet-an acpipowerbutton        # graceful
& $VBox controlvm lejet-an poweroff               # hard stop
# snapshots
& $VBox snapshot lejet-an list
& $VBox snapshot lejet-an restore "an-configured-clean"
# inspect VM config
& $VBox showvminfo lejet-an --details | Select-String -Pattern "NIC|Shared|Clipboard|USB|Audio|VRDE"
# connect
ssh lejet-gw
ssh lejet-an
```

### Appendix F — Milestones and suggested order of work
| Milestone | Deliverable | Exit test |
|---|---|---|
| M0 | Approvals, folders, ISO verified | §21.1 first items done |
| M1 | Gateway VM up with firewall, VPN, DNS, chrony | `dig` via gateway works; kill switch works |
| M2 | Analysis VM up with hardening | Level 2 tests 1–24 pass |
| M2a | 14 URL-text features (already in `link_features.py`) wired into Stage 0 | Level 0a parity green on host and in the VM |
| M3 | Fetcher v1: Stage 0, 1, 1b (RDAP), 3 (HTTP) | Level 0 server-side fixtures green; RDAP parse tests green |
| M4 | 12 static-HTML features + `page_fetched` + `route_p_u` hand-off test | Level 0 HTML fixtures green |
| M4b (optional) | Browser stage for JS-built pages | Only if the near-empty-page rate justifies it; browser fixtures green |
| M5 | Safe external pilot | Level 1 green |
| M6 | Host validator + export/import flow | malicious-fixture round trip passes |
| M7 | Phishing pilot (100 URLs) | Level 3 reviewed, no abort triggers |
| M8 | Scale to 10k+ in batches | all batches validated; manifests complete |
| M9 | Decommission + thesis methods chapter | §19 and §20 done |

**Suggested first working session ("just try web inspection"):** M0 → M1 → M2 → M3/M4 against local fixtures (Level 0), then Level 2 and Level 1. You will have seen web inspection work end-to-end, safely, without touching a single malicious URL. Only after that move to Level 3.

---
## 24. Integration with the Le-Jet pipeline

The lab VMs exist to **collect the fresh set** (dataset C, `data/dataset_and_training_plan.md` §8), which is used only to test the website-inspection claim and never to train. The finished product needs a separate path for a user who pastes a message.

### 24.1 Request flow
```
SMS ──▶ extract_links / normalize_link          (src/features/link_features.py)
    ──▶ 14 URL-text features + has_url          (instant, no network)
    ──▶ ask the inspector for page evidence     ──▶ isolated VM: dns_record, 3 RDAP features, 12 HTML features
    ──▶ host: page_rank lookup for the final registrable domain
    ──▶ route_p_u(url_only, full, frame)        (src/features/model/rf_extractor.py; page_fetched picks the extractor)
    ──▶ P, u + CNN + URL features ──▶ gates ──▶ classifier ──▶ verdict + reasons
```
Only the inspector step touches the web. If the page fails, the row still gets P and u from the `url_only` extractor, so the user always gets an answer. The serving code carried over from SmishGuard (`src/serving/predict.py`) already uses the pattern "call out with a timeout, never raise, fall back on error" for Safe Browsing (`src/serving/safe_browsing_client.py`); the inspector client should do the same. Wiring the inspector into `predict.py` is future work: today `predict.py` still calls the old SmishGuard model path and has not been adapted to the Le-Jet fusion classifier (README status table).

### 24.2 Option 1 — thesis demo on your laptop (build this)
- A small **inspector service** runs inside the analysis VM and listens on `127.0.0.1:8080` of the VM only.
- The Flask app on the host calls it **host-initiated** through an SSH tunnel; the VM never initiates a connection to the host:
  ```powershell
  ssh -N -L 8081:127.0.0.1:8080 lejet-an      # host listens on 127.0.0.1:8081
  ```
  This needs two small config changes: in the analysis `sshd` use `AllowTcpForwarding local` with `PermitOpen 127.0.0.1:8080` for the `ops` user (instead of `AllowTcpForwarding no`), and keep the gateway's `PermitOpen 10.77.0.2:22`.
- **Contract:**
  - Request: `POST /inspect` with `{"url": "<normalized url>"}` (the service re-runs the Stage 0 safety gate; it never trusts the caller).
  - Response (strict schema, nothing but numbers/booleans/enums):
    ```json
    {"status":"ok|partial|failed","fail_reason":"none|dns_nxdomain|connect_timeout|http_4xx|...|rejected_url",
     "flags":{"fetch_ok":true,"rdap_ok":false,"html_available":true},
     "features":{"nb_hyperlinks":31,"...":"the 12 HTML + 4 external features","domain_age":-1},
     "final_reg_domain":"example[.]test",
     "stage_ms":{"dns":41,"rdap":612,"http":930,"html":38},"duration_ms":1650}
    ```
  - Error codes: `400` bad URL, `429` rate-limited, `503` busy/queue full, `504` inspector timeout.
- Limits: 1–3 concurrent inspections, bounded queue, per-user rate limit, results cached by normalized URL for a few hours.
- Because the VM is now long-running, revert to the clean snapshot on a schedule (e.g., daily) and re-run `preflight.sh` each time.
- Limitations: needs your laptop, the VMs and the VPN to be on; one-at-a-time throughput.

### 24.3 Option 2 — real deployment (describe as future work)
A hosted inspector service on a cloud server with throwaway workers (containers with gVisor or microVMs), a job queue, rate limiting and caching; the main app calls it over an authenticated internal API. Scales, but needs money and the provider's abuse-policy approval.

### 24.4 What the finished product must handle
- **Fallback:** if the inspector times out, the URL is dead, or the VM is down, answer with the 14 URL-text features and the `url_only` extractor (this is exactly the `route_p_u` fallback). The two extractors already are the "two model variants"; no extra flags are needed beyond `page_fetched`.
- **Train/serve consistency:** same fetch settings, same user agent, same timeouts, same RDAP logic, same missing-value rule as in the training run.
- **Abuse protection:** users can submit anything: refuse private IPs, localhost, odd ports and schemes; rate-limit; never follow a link into your own network.
- **Privacy:** only the URL goes to the inspector, never the message text; tell users that links are checked.
- **Freshness:** a site that dies or cloaks a minute later returns "dead". Show it as "couldn't verify", never as "safe".
- **UX:** return the text/URL-feature verdict immediately, then upgrade it when the web check finishes.

---

## 25. Latency budget and measurements

Estimates, **not measurements** — replace them with your own from Levels 1 and 3.

| Stage (v1) | Typical | Slow case |
|---|---|---|
| Stage 0: normalize + 14 URL-text features | < 5 ms | same |
| Stage 1: DNS (via gateway + VPN) | 30–200 ms | up to 3 s (timeout) |
| Stage 1b: RDAP (runs in parallel with the fetch; cache hits ≈ 0) | 200–1500 ms | 4 s (timeout) |
| Stage 3: HTTP fetch + 1–3 redirects | 300 ms – 3 s | 10+ s (shorteners, slow hosts) |
| Stage 5: static HTML features | 20–150 ms | up to 500 ms on large pages |
| **Total, live page (v1)** | **≈ 0.5 – 3 s** | p95 ≈ 6 s; hard cap 12 s (interactive) / 30 s (batch) |
| **Total, dead domain (DNS fails)** | **< 0.5 s** | 3 s on a DNS timeout |
| *Optional browser render (v2)* | *+2–6 s* | *up to 20 s* |

So the full web check will usually take **more than one second**. The VPN adds roughly 50–150 ms per round trip.

How to keep it feeling fast: answer from the URL/text model first, then upgrade when the web check completes; short-circuit on DNS failure or a Safe Browsing match; cache by URL; keep connections and the RDAP bootstrap warm; use a 10–15 s cap for interactive use; offer a "passive-only" mode (DNS + RDAP + headers, ≈ 0.5–2 s) trained on its own feature subset.

**Required instrumentation:** record `stage_ms_*` and `duration_ms` for every URL (columns in §14.0.6). For the thesis, report median and 95th percentile per stage and overall, split by outcome (live, dead, error), and discuss the accuracy-versus-latency trade-off.

---

## 26. Role of Google Safe Browsing

**What it is:** a free Google service (v4 Lookup API, `threatMatches:find`) that says whether a URL is on Google's list of known dangerous sites (malware, social engineering/phishing, unwanted software, harmful apps) — the same list behind Chrome's red warning page.

**Status in Le-Jet (2026-10-11):** the dataset plan lists Google Safe Browsing as *removed* from the thesis design (`data/dataset_and_training_plan.md` §11–12; the group still has to write down why). The code is still present in the carried-over serving files, so this section only records what it was and how it relates to inspection. Do not build anything on it for the inspector.

**How SmishGuard used it** (`src/serving/safe_browsing_client.py`, `src/serving/predict.py`, still in the repo): `matched` → immediate "malicious" hard bypass without running the CNN; `clean` or `error` → the model decides and the verdict is shown alongside. Research Mode and baseline models skip it so scores stay comparable.

**Strengths:** fast (≈ 100–300 ms), no browser or VM, high precision on a match, a useful fallback when web inspection fails.

**Limits:** poor coverage of fresh threats (smishing sites often live for hours before being listed — a new phishing URL usually comes back `clean`); yes/no only, so it gives no learnable features; third-party dependency that needs an API key and sends each URL to Google.

**How it fits with web inspection (if the group ever re-adds it):** Safe Browsing answers "is this *already known* to be bad?" (fast, high precision, low recall). The model plus web features answers "does it *look and behave* like phishing even if nobody reported it yet?" The two are complementary.

**For the thesis:** if it is mentioned at all, do **not** use Safe Browsing as a training feature (it would leak the label); report it as a separate baseline/fallback, and measure how often your model flags URLs that Safe Browsing marked `clean`.

## 27. Review: loopholes and gaps found

A line-by-line review of this plan. **Status** says whether the plan body was already corrected (**Fixed**) or the item still needs a decision or work (**Open**). Severity: **Critical** = would silently corrupt the dataset or break the safety model; **High** = likely to bite you; **Medium** = worth doing; **Low** = note it.

### 27.1 Data-validity loopholes (these can ruin the thesis results without any error message)

| ID | Sev. | Loophole | Why it matters | Status / fix |
|---|---|---|---|---|
| V1 | Critical | The gateway's upstream DNS was Quad9 `9.9.9.9`, which **blocks known-malicious domains** | Live phishing domains returned NXDOMAIN, so they looked dead; `dns_record`, liveness and labels would all be wrong | **Fixed** (§10.6: Cloudflare, plus a warning and test 25). Also turn off your VPN's DNS/malware blocking |
| V2 | Critical | URL-string features were to be computed on the **normalized** URL | Dataset values come from the raw string; normalizing first changes `length_url`, `nb_dots`, `nb_www`, `punycode`… and breaks the parity test | **Fixed** (§13.2: raw string for features, normalized only for fetching) |
| V3 | High | Which domain do the HTML/RDAP features describe when the URL **redirects or is a shortener**? (e.g., `bit.ly/x` → phishing page) | RDAP on `bit.ly` says "old, legitimate domain" and hides the signal; HTML features describe the destination | **DECIDED (§13.2b): features come from the final URL.** URL-string features stay on the original URL (dataset parity); HTML features, "internal/external", RDAP and DNS describe the final URL's registrable domain; the original-domain values are stored as `orig_*` columns. Remaining risk: redirects to a big benign site (cloaking) make the features describe the benign destination, so flag them with `final_is_popular` |
| V4 | High | **Bot walls, parked, suspended and default server pages** are parsed as if they were the site (Cloudflare "Just a moment", CAPTCHA, "Account suspended", Apache/nginx default page, registrar parking) | Features describe the wall, not the phishing page; the label becomes noise | **Partly fixed** (new `fail_reason` values `challenge_page`, `parked_or_default_page`). **Open:** write the detectors (title/body markers, `cf-mitigated`/`cf-chl` headers, 403/429/503 with tiny bodies) and report their rate. They now also decide `page_fetched`: a wall mistaken for a page would be scored by the `full` extractor |
| V5 | High | **Shared hosting and free subdomains** (`*.pages.dev`, `*.workers.dev`, `*.vercel.app`, `*.netlify.app`, `*.github.io`, `*.web.app`, `*.blogspot.com`, `*.weebly.com`, `*.000webhostapp.com`, IPFS gateways, tunnelling hosts) | The registrable domain is an old, legitimate platform, so `domain_age`/`whois_registered_domain` look benign for exactly the sites attackers like | **DECIDED (§13.2c):** private-suffix registrable domains, `free_hosting_subdomain` / `hosting_platform` / `platform_path_hosting` flags from a curated list, RDAP skipped for those hosts (features `-1`, `rdap_skipped = 1`). **Still open:** keep the list current, and add a Certificate-Transparency first-seen date as a substitute age (§28 item 6, v1.1) |
| V6 | High | **Class-source bias**: benign URLs from Tranco are homepages of old, popular domains; phishing URLs from feeds are deep links on new domains | The model learns "homepage vs deep link" and "old vs new domain", not phishing | **DECIDED (§29 benign sampling plan):** benign URLs are drawn from five groups with realistic structure, hard negatives are required, classes are matched on coarse nuisance bins (common support, not identical distributions), and a nuisance-only classifier is the bias test. Results are reported on the full set and on a matched subset |
| V7 | High | **Concept drift**: the public set's features were designed in 2020–21 and the extractors are trained on it; 2025–26 kits use JavaScript frameworks, Cloudflare Turnstile, serverless hosting, QR-code and short-lived pages | Static features can degrade badly on modern kits; a model validated on the old dataset may fail on new data | **Open.** Report the near-empty-page rate and the per-feature missing rate; keep a held-out *recent* test set; discuss in the thesis |
| V8 | High | **Static parse is blind to JavaScript-built pages** (an empty `<div id=root>` shell has no forms or links) | Many phishing forms exist only after JS runs, so `login_form`, `nb_hyperlinks`, etc. read as 0 | **Open.** Measure with `html_bytes`/`script_count`; if high, enable the optional browser stage for the near-empty subset only |
| V9 | Medium | Liveness bias (dead = phishing) | Model may learn dead-ness | Documented (§20) and indicator columns exist. Also keep dead-URL rows out of any headline accuracy number |
| V10 | Medium | `fetch_ok` semantics were ambiguous (did an RDAP failure make `fetch_ok` false?) | Inconsistent rows | **Fixed** (§13.10 defines `fetch_ok`, `rdap_ok`, `html_available`) |
| V11 | Medium | **Schemeless SMS URLs** (`gcash-verify.xyz/login`) and trailing punctuation were rejected or mangled | Many real SMS URLs would be dropped | **Fixed** (§13.2: https-then-http fallback, scheme added for fetching only) |
| V12 | Medium | URLs with an **explicit port** (`:8080`, `:8443`) can never succeed (gateway allows only 80/443) and would look dead | Counted as takedowns | **Fixed** (`port_blocked`); remember to exclude them from "dead" statistics |
| V13 | Medium | Cookies were disabled | Sites that set a cookie on a redirect loop forever → false `too_many_redirects` | **Fixed** (§13.5: per-URL in-memory cookie jar) |
| V14 | Medium | The VM could relabel or fabricate rows (it computed `url_id`; the host trusted its output) | "Integrity of the dataset" was listed as an asset with no control beyond schema checks | **Partly fixed** (host generates `url_id`). **Open:** add **canary URLs** to every batch (a few URLs whose results you know: `example.com`, an NXDOMAIN name, a local fixture) and fail the batch if canary rows are wrong |
| V15 | Medium | The same URL may appear with **two labels** or twice in different batches | Dedup by `url_id` silently drops one | **Open.** Host-side duplicate/conflict report before pushing |
| V16 | Low | ~~The 4 brand features use the authors' lists~~ | **Obsolete (2026-10-11):** the brand features are not in the 31-feature scope | Philippine-brand features remain an optional extra (§28.1 item 4), outside the frozen extractors |
| V17 | Medium | `domain_registration_length` definition is unclear (total term vs. remaining) | A wrong definition inverts the feature | **Open:** settle from the paper; helper `days_to_expiry` is stored |
| V18 | Low | RDAP has no data for many ccTLDs (`.ph` unverified) and registries rate-limit by IP | Many `-1` values, or `429` bursts through the shared VPN IP | **Open:** measure in Level 1; cache; back off; report the missing rate per TLD |
| V19 | Low | The RDAP cache and IANA bootstrap file are **lost on every snapshot revert** | Repeated lookups → more rate limiting | **Open:** keep a host-side RDAP cache (domain → result, timestamp) pushed with each batch and pulled back |
| V20 | Low | VPN exit country/IP reputation changes what cloaking kits show | Systematic bias toward "benign-looking" pages | Documented (§20). Record exit country per batch; if possible re-fetch a sample from a PH exit |
| V21 | High | **`page_rank` train/serve mismatch**: the training column is Open PageRank values from the 2021 dataset; the live lookup is made today, on the host, with a possibly different scale or coverage | A shifted feature silently shifts P and u for every inspected link | **Open.** Before the pilot, look up a sample of public-set domains today and compare with the stored values; record the API date in the manifest; lookup failures are `-1`, unrated domains `0` (§14.0.2); report results with and without `page_rank` (it is also an ablation in `rf_extractor.ABLATIONS`) |
| V22 | High | **The 12 HTML features and the RDAP/DNS features are re-implemented by us**, but the extractors were trained on the dataset authors' values, and the public set's pages are mostly dead, so there is no parity test for them | A different counting rule changes the inputs the forest sees, with no error message | **Open.** Write each definition from the paper (§14.0.3), test with fixtures, then compare the **distribution** of each feature on live benign and phishing pages with the public set's (class means, `-1` rate); investigate any feature whose distribution differs sharply |

### 27.2 Safety / infrastructure loopholes

| ID | Sev. | Loophole | Why it matters | Status / fix |
|---|---|---|---|---|
| S1 | High | The firewall **logged accepted connections but not dropped ones**, yet abort criterion #1 depends on seeing blocked attempts | A probing compromised VM would leave no log trail | **Fixed** (§10.5: rate-limited `lejet-blocked-private` / `lejet-blocked-other` logs) |
| S2 | High | The **gateway itself could not resolve names** (its resolver pointed at the blocked NAT DNS) | `apt`, `chrony`, the IANA bootstrap would silently fail | **Fixed** (§10.6: local Unbound + resolved config) |
| S3 | High | **WireGuard + snapshot revert**: stale clock → handshake rejected (replay-protection timestamp), plus bad TLS validation and wrong `domain_age` | Batch starts "randomly" without a VPN, or with wrong ages | **Fixed** (§17.1 clock step, test 26) |
| S4 | High | **CPU-bound parsing in the asyncio loop**: one hostile page freezes every other URL; a catastrophic regex pins a core | Hung batches, timeouts mistaken for dead sites | **Fixed in design** (§13.5: process pool with per-task kill timeout). The three script-pattern features (`popup_window`, `onmouseover`, `right_clic`) were dropped on 2026-10-11, which removes the main ReDoS exposure; keep any remaining regex bounded or linear-time |
| S5 | High | Live-service mode (§24) exposes a **long-running** VM to URLs chosen by arbitrary users | Persistent compromise across users, abuse of your VPN IP to hit third parties, SSRF probing, no revert between requests | **Open.** For a deployed inspector use one **ephemeral worker per request** (fresh process/container or VM revert), authentication, per-user and global rate limits, an allow-list of schemes, and daily VM reverts. Keep the research lab and the demo service on separate VMs |
| S6 | Medium | Decompression bombs and exotic encodings | Memory/CPU exhaustion | **Fixed** (§13.5: count decoded bytes, gzip only) |
| S7 | Medium | HTTP/2 parser surface | More code exposed to hostile servers | **Fixed** (`http2=False`) |
| S8 | Medium | Obfuscated/numeric IP hosts (decimal, octal, hex, IPv4-mapped IPv6, trailing dot, `localhost` aliases) could slip past the private-IP check | SSRF to loopback/LAN | **Fixed** (§13.2). **Open:** DNS-rebinding TOCTOU (name resolves differently at check vs connect). The gateway resolver's `private-address` filter and the firewall are the backstop; optionally pin the resolved IP and connect to it |
| S9 | Medium | `tldextract` fetches the Public Suffix List at runtime and writes a cache | Unexpected egress; failure offline; cache path under `ProtectHome` | **Fixed** (§13.2: bundled snapshot, no network, no cache dir) |
| S10 | Medium | A compromised VM can still talk to **any host on 80/443** and tunnel data through **DNS** | Exfiltration and C2 are possible by design (the VM must reach arbitrary sites) | **Residual risk.** Mitigations: no secrets in the VM, the new connection-rate limit, egress logging, short batches with a revert. State this in the thesis |
| S11 | Medium | Emulated NIC code is a historical VM-escape surface (VirtualBox's default Intel NIC) | Smaller attack surface is cheap | **Open:** use `--nic-type1 virtio` (VirtualBox "Paravirtualized Network") on both VMs; Ubuntu has the driver. Re-check connectivity |
| S12 | Medium | `systemd-run` sandboxing is thinner than it could be | Defense in depth | **Open:** add `PrivateDevices=yes`, `RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX`, `CapabilityBoundingSet=`, `SystemCallFilter=@system-service`, `ProtectKernelTunables=yes`, `ProtectControlGroups=yes`, `LockPersonality=yes`, and `IPAddressDeny=` for private ranges |
| S13 | Medium | The "clean" snapshots **go stale**: unpatched Ubuntu/Python packages | A year-old base image is easier to exploit | **Open:** monthly routine: boot, update, re-run preflight, replace the clean snapshot; record versions in the manifest |
| S14 | Medium | `sudo` over non-interactive SSH (`ssh lejet-an "sudo install …"`, `preflight.sh`, the clock step) **will fail** (no TTY/password) | Scripts break, tempting you to loosen sudo | **Open:** add a narrowly scoped `/etc/sudoers.d/lejet` for `ops` allowing only the exact commands needed with `NOPASSWD`, or push files into a directory `ops` owns and let `fetcher` read it via group |
| S15 | Low | `Level 2 #10` (`curl --http3-only`) may "fail" simply because curl lacks HTTP/3 | False pass | **Fixed** (replaced by a raw UDP/443 test) |
| S16 | Low | Boot-order window: before `nftables` loads, the default policy is accept | Brief exposure on the gateway | Start the gateway first, keep `nftables` enabled (it loads before the network), verify in the preflight. Low risk because the analysis VM has its own firewall |
| S17 | Low | The VPN provider sees all traffic; RDAP registries and the resolver see the domains you investigate | Privacy/trust | Document in the ethics section; choose a provider with a no-logs policy you can cite (2020+ sources) |
| S18 | Low | MTU/MSS stalls over WireGuard | Large pages hang → counted as timeouts | **Fixed** (§10.5 MSS clamp) |
| S19 | Low | Markdown table cells containing `|` broke rendering of several tests | Misreading the test commands | **Fixed** (escaped) |
| S20 | Low | Legal/ethical status of contacting phishing infrastructure is not something this plan can settle | Thesis risk | Needs your adviser's/university's written position (§6); I cannot verify the Philippine legal specifics |

### 27.3 Things I could not verify (treat as assumptions)
- VirtualBox 7.x flag spellings, interface names (`enp0s3`/`enp0s8`), the AppArmor path of the Chromium binary, and the exact `systemd-run` properties on your Ubuntu build.
- That RDAP serves the TLDs you need (especially `.ph`), and that a given VPN provider's terms allow this use.
- The exact definitions of the 12 HTML features in the original paper (and of `domain_registration_length`), and whether `page_rank` still uses the same 0–10 scale as the training column.
- Nothing in this plan has been executed; the first run will find more issues.

### 27.4 Recommended order for fixing the **Open** items
1. V3 (which domain the features describe), V5 (free hosting), V6 (class-source bias): decide these before collecting anything.
2. V4 (challenge/parked detectors) and V14 (canaries): build them with the first fetcher.
3. S14 (sudo) and S12 (service sandboxing): during VM setup.
4. S11 (virtio NIC), S13 (patch routine), V19 (RDAP cache): before the large run.
5. S5 (live-service design) before any demo that accepts user URLs.

---

## 28. Suggested additional features

All of these are chosen to stay compatible with the v1 safety model (no JavaScript execution, one fetch per URL) unless marked **[browser]** or **[reference data]**. Add them as **extra columns after** the 31 model-input ones, never replacing them, so the parity tests still work and the frozen extractors still receive exactly the columns they were trained on (they are never retrained on SMS data).

### 28.1 Top 10 to add first (best value for effort)
1. **Data-exfiltration endpoints in the page:** form `action` or script strings pointing to `api.telegram.org/bot…`, Discord webhooks, `formspree.io`, `emailjs`, `getform`, `web3forms`, Google Forms/Apps Script, `mailto:`; plus counts of `fetch(`/`XMLHttpRequest`/`sendBeacon` to external hosts. Very common in phishing kits and fully static.
2. **Redirect-chain features:** hop count, cross-domain hops, final domain ≠ original, shortener in chain, meta-refresh/JS-literal redirects, http→https upgrade.
3. **Page-type flags:** `challenge_page`, `parked_or_default_page`, `suspended_page`, `soft_404`, `near_empty_js_shell` (also needed to repair V4/V8).
4. **Brand impersonation with typosquat distance:** edit distance/homoglyph (confusable characters, IDN) between the registrable domain and a list of target brands; Philippine brands (GCash, Maya, BDO, BPI, UnionBank, Metrobank, Landbank, PNB, Shopee, Lazada, Globe, Smart, DITO, SSS, PhilHealth, Pag-IBIG, BIR, LTO, LTFRB, PSA, NBI); brand name in title/alt text/path but not in the domain.
5. **TLS certificate features:** certificate age, validity length, free-CA flag (Let's Encrypt/ZeroSSL), self-signed, hostname mismatch, SAN count, wildcard.
6. **Free-hosting / tunnelling flag + Certificate-Transparency first-seen date** (a substitute for domain age when RDAP is meaningless). Source: CT log search (e.g., crt.sh) — rate-limited and slow; cache it.
7. **Hosting network:** ASN, hosting-provider class (cloud/CDN/bulletproof), country, "behind Cloudflare" flag, number of A records, NS provider, TTL. **[reference data]** offline MaxMind GeoLite2 ASN/country copied into the VM (no live calls).
8. **DOM structure fingerprint:** hash of the tag-sequence skeleton and a SimHash of visible text tokens, for clustering cloned kits (also protects against train/test leakage by splitting on cluster).
9. **Form field semantics:** counts of `type=password`, `tel`, `email`, `number`, fields named/labeled OTP, PIN, CVV, card number, account number, MPIN, birthday, mother's maiden name; `autocomplete=off`; hidden-input count; form method.
10. **Obfuscation indicators:** counts of `eval`, `atob`, `unescape`, `String.fromCharCode`, `document.write`, very long base64/hex strings, entropy of inline scripts, ratio of script bytes to total bytes.

### 28.2 URL-level extras (no network)
- Path/query entropy, digit↔letter transitions, uppercase ratio, longest repeated token, count of brand/keyword tokens (EN + Tagalog: *verify, kumpirmahin, claim, libre, premyo, panalo, suspendido, na-block, i-update*), keyword density.
- TLD risk score (e.g., `.xyz .top .icu .cfd .sbs .click .shop` — derive from your own data, not a fixed list), presence of `@`, double-encoded characters, base64 in the path, open-redirect parameters (`url=`, `redirect=`, `next=`).
- Character n-gram TF-IDF or a character-level CNN over the raw URL string as a separate sub-model (this pairs naturally with your existing CNN).

### 28.3 HTTP / header extras
Server header family, `X-Powered-By` present, HSTS/CSP/X-Frame-Options present, `Set-Cookie` count and flags (Secure/HttpOnly/SameSite — flags only), content-type, content-length, time-to-first-byte, total response time, gzip used, page size, number of resources referenced, `robots` meta (`noindex`), `generator` meta (CMS family), presence of footer/privacy-policy/contact links, count of `<noscript>`.

### 28.4 Domain / DNS extras
RDAP registrar name (hashed or bucketed), registrar-status flags (`clientHold`, `serverHold`, `pendingDelete`), days since last update, nameserver count/provider, DNSSEC flag, MX present, SPF/DMARC present, number of A/AAAA/CNAME records, TTL, whether the host resolves to a shared IP seen with many other URLs in your dataset (careful: this can leak labels between train and test).

### 28.5 Cross-URL and SMS-context features (Le-Jet-specific, probably your strongest contribution)
- **Message ↔ page consistency:** the SMS claims to be from brand X (from text), but the URL's domain/title/forms belong to brand Y or to none; the message is urgent/"verify now" while the page asks for an OTP/PIN.
- Message features combined with link features: shortener used, link position, link density, sender type, language (Tagalog/English/Taglish mix).
- Dataset-level graph features (shared IP/ASN/NS/certificate/DOM-fingerprint cluster size) with strict cluster-wise train/test splits.

### 28.6 Browser-only features (optional stage) **[browser]**
Rendered form count vs static form count (JS-injected forms), popup/dialog/download counts, `beforeunload` traps, notification/clipboard/fullscreen permission requests, number and domains of third-party requests, WebSocket use, canvas/audio fingerprinting calls, auto-redirect after load, screenshot perceptual hash and logo/brand similarity to reference pages **[reference data]** (fetch the real brand pages once to build the references).

### 28.7 Cautions when adding features
- Keep the 31 untouched and add new ones as a separate group so you can report an ablation (31 vs 31 + extras). Extras cannot go into the frozen extractors; they would need their own model or a new extractor trained on a public set that has those columns.
- Anything derived from a reputation service (Safe Browsing, VirusTotal, feed membership) is a **label leak** if the same service produced your labels. Keep those out of training features.
- Features that depend on the date of collection (CT first-seen, domain age, certificate age) must be computed at the **same fetch time** for both classes.
- Every new feature needs: a definition, a fixture test, a missing-value rule (`-1` + indicator), and a note on what fraction of rows have it.

## 29. Benign sampling plan

### 29.1 Why this exists
If benign URLs come from a popular-sites list (mostly homepages of old, popular domains) and phishing URLs come from feeds (mostly deep links on new, cheap domains), a model can score well by learning **how the data was collected**: "short homepage on an old domain = benign, long deep link on a new domain = phishing". Those shortcuts fail on real SMS links. This section defines how benign URLs are chosen so the classes **overlap** on collection artifacts and the model has to learn phishing behavior instead. (It settles §27 V6.)

**Important trade-off:** some differences between the classes are *real signal* (phishing URLs really are longer and deeper on average). Matching should therefore aim for **common support** — benign examples must exist across the same ranges as phishing ones — not for identical distributions, otherwise you erase features that legitimately work. Always report results both ways (§29.8).

### 29.2 Inputs that must be frozen and recorded
| Input | Record in the manifest |
|---|---|
| Popular-domain ranking (Tranco) | list ID/date, how many ranks used |
| Public URL index (e.g., Common Crawl) | snapshot/crawl ID, query used |
| Legitimate SMS links (from your SMS dataset's non-spam messages) | dataset name/version, filter used |
| Curated brand login/payment/account URLs (incl. Philippine brands) | file hash, date, who curated it |
| The phishing set you are matching against | feed names, date range, `n` |
| Sampling seed and the script's git commit | seed, commit |

### 29.3 Benign groups and candidate pools
Build a candidate pool about **3× the target** per group (dead links, filters and matching will discard many).

| Group | What it is | Source / method | Starting share |
|---|---|---|---|
| **A. Deep paths on popular domains** | Article, product, help, account pages on well-known sites | Query a public URL index for URLs under Tranco domains (this finds paths **without visiting** the sites) | 40% |
| **B. Brand login / payment / account pages** | Real login and payment pages of banks, e-wallets, telcos, government agencies, shopping sites — including Philippine ones (GCash, Maya, BDO, BPI, UnionBank, Metrobank, Landbank, Globe, Smart, DITO, SSS, PhilHealth, Pag-IBIG, BIR, LTO, Shopee, Lazada) | Hand-curated list of **official** URLs (verify each manually against the organization's own site) | 20% |
| **C. Benign short links** | Legitimate shortened links (bit.ly-style, brand shorteners, telco/bank links) | Links found in legitimate (ham) messages of your SMS dataset | 15% |
| **D. Small and free-hosted legitimate sites** | Personal sites, blogs, small businesses, GitHub/Pages/Blogspot-style sites, new but legitimate startups | Lower part of the Tranco list, plus URL-index hits on free-hosting suffixes (§13.2c) | 15% |
| **E. Plain homepages** | Root pages | Tranco domains (only a minority, to avoid the homepage shortcut) | 10% |

Example for a 10,000-row training set (5,000 phishing + 5,000 benign): A 2,000, B 1,000, C 750, D 750, E 500. Treat these shares as a **starting proposal** and adjust them to match the phishing set (§29.5).

**Group B is mandatory:** login forms are the most phishing-looking thing a legitimate site has, so without them `login_form` and `domain_in_title` would simply become "is it phishing" detectors.

### 29.4 Cleaning filters (apply to every candidate)
1. **Not in any phishing/malware feed** you hold (OpenPhish, PhishTank, URLhaus, PhishStats), and not flagged by Safe Browsing at the time of fetch. Use these only as a **filter**, never as a model feature (label leakage, §26).
2. Valid, well-formed URL; passes the Stage 0 safety gate (§13.2).
3. De-duplicate; remove URLs that also appear in the phishing set.
4. **Cap 5 URLs per registrable domain** across the whole benign set (groups A and B especially), so a few large sites cannot dominate.
5. Drop URLs that require login to reach (they will return a wall, not the page) unless they are in group B on purpose.
6. Re-check liveness shortly before the fetch window with the DNS-only check (§13.3). Benign pages usually stay up, but index entries are often stale, so sample extra.

### 29.5 Matching on nuisance variables
Done **on the host before any fetching**, using only string-level variables (no network needed):

1. **Nuisance vector** for every URL (phishing and benign): path depth, URL length bucket, TLD group (e.g., `.com`, `.ph`, other country codes, cheap new gTLDs, other), scheme (http/https), has query string, subdomain count bucket, shortener (yes/no), free-hosting (yes/no), has port.
2. **Coarsen** each variable into a few bins.
3. **Stratified sampling:** for each joint bin (or, when bins are sparse, for the marginals) draw benign URLs so that the benign set reaches **common support** with the phishing set: every bin that has phishing examples must also have benign examples (target at least 0.5× to 1× the phishing count per bin where the candidate pool allows it).
4. Keep a small fraction of "unmatched" benign URLs that show typical benign structure, so the model still sees the natural benign distribution.
5. Use a fixed seed; save the sampled list and the bin counts.
6. **Top-up after fetching:** dead links change the distribution, so after the fetch recompute the bin counts on the live subset and sample replacements for under-filled bins (still using the same frozen candidate pool and seed rules).

### 29.6 Hard negatives (required quotas)
Within the benign set, make sure these exist in meaningful numbers (suggested at least 5% each, counted after fetching):
- legitimate pages **with a login or password form**;
- legitimate sites on **new domains** (so `domain_age` is not a perfect label proxy);
- legitimate sites on **free hosting** (so `free_hosting_subdomain` does not mean "phishing");
- legitimate pages that **redirect** (shorteners, `http`→`https`, `www` redirects);
- legitimate pages with **many external links, iframes or popups** (ads, embeds).

### 29.7 Splits and evaluation sets
- **Split by registrable domain** (group k-fold), never by random URL, and keep all URLs of one domain in the same fold. Also split near-duplicate phishing kits together (use the DOM/text fingerprint when you add it, §28).
- **Time-aware test set:** hold out the most recent fetch days as a final test.
- **Realistic-prior evaluation:** real smishing is rare. In addition to the balanced training set, evaluate on a set with about **1 phishing : 20 benign** (draw extra benign URLs under the same sampling rules) and report precision, recall and false-positive counts at that prior.

### 29.8 Bias diagnostics (the test that this plan worked)
1. **Nuisance-only classifier:** train a simple model (e.g., logistic regression or a small tree) using **only** the nuisance variables from §29.5 (no HTML, no RDAP) and measure AUC with domain-grouped cross-validation. If it separates the classes far better than chance, collection artifacts are still carrying the label — rebalance and re-sample. Treat "close to 0.5–0.6" as the target, not a guarantee; some real difference is expected.
2. **Standardized mean difference** per nuisance variable between classes; aim for absolute values below about 0.1–0.2 for the ones you chose to match.
3. **Feature-importance sanity check:** inspect the top features of the final model. If the top ones are URL length, path depth or TLD only, suspect a shortcut.
4. **Report two results in the thesis:** (a) full set, (b) matched subset (and the realistic-prior result). Discuss the gap.
5. **Missingness check:** compare the share of `-1` (dead/blocked/RDAP-missing) by class and by group; strong imbalance means the model can learn "missing = class".

### 29.9 Interaction with the fetching plan
- Sampling and matching are done on the host, then written to the frozen `urls.txt` plus a host-side `labels.csv` (url_id, label, group A–E, label_source, bin ids). The VM never sees labels or groups.
- Benign and phishing URLs are **interleaved in every batch** (§17.4), fetched in the same time window with the same settings.
- The benign fetches are ordinary single page loads at a polite rate (≤ 1 request per domain at a time, shared global rate limit); no crawling beyond the URL, no form submission.

### 29.10 Manifest fields (add to §15.3)
`tranco_list_id`, `tranco_rank_range`, `url_index_snapshot`, `url_index_query`, `ham_sms_source`, `brand_list_hash`, `sampling_seed`, `sampling_commit`, `candidate_pool_sizes_by_group`, `final_counts_by_group`, `bin_definition_hash`, `dead_rate_by_group`, `nuisance_auc`, `smd_table_path`, `hard_negative_counts`.

### 29.11 Caveats
- Index entries for benign pages may no longer exist, and benign short links from old SMS may have expired — hence the 3× pools and the post-fetch top-up.
- Group B is small and hand-curated; it may under-represent less-known local brands. Extend it from your own data.
- Public rankings favor large, English-language, well-known sites. Philippine local sites are under-represented unless you add them on purpose (groups B and D).
- Matching can only control the variables you list; unlisted differences (page language, CMS, hosting country) may still leak. Check the feature-importance sanity test.
- Even a good sample is not the real-world traffic mix; that is why the realistic-prior evaluation is required.

### 29.12 Tasks
- [ ] Freeze inputs and record them (§29.2)
- [ ] Hand-curate the brand login/payment URL list (group B) and have it reviewed
- [ ] Build candidate pools for groups A–E (3× target)
- [ ] Apply the cleaning filters (§29.4)
- [ ] Define the nuisance bins (§29.5) and write the sampling script with a fixed seed
- [ ] Produce the frozen `urls.txt` and host-side `labels.csv`
- [ ] After the pilot fetch: recompute bin counts on live rows and top up
- [ ] Run the nuisance-only classifier and SMD checks (§29.8) before training any real model
- [ ] Record all manifest fields (§29.10)

---
*End of plan.*
