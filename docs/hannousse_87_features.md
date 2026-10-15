# The public-set features in the lab (73 of the 87 columns)

| | |
|---|---|
| **Dataset** | Hannousse & Yahiouche (2021), *Towards benchmark datasets for machine learning based website phishing detection* (11,430 URLs, 87 features: 56 URL-based, 24 content-based, 7 external) |
| **Where they are measured** | the analysis VM (`lejet.features.hannousse_url`, `hannousse_html`, the existing external stages) and, for `page_rank`, the host (`scripts/add_full_features.py`) |
| **Where they appear** | `features_full` in every exported record, and `<batch>.full_features.csv` next to `<batch>.csv` in `D:\LeJetData\clean\` |
| **What did not change** | the 30 VM model inputs in `features`, their names, order and values, and therefore the trained models |

`features_full` holds **73 columns**: the public CSV's columns **in its order and under its names**, minus fourteen that
were left out on purpose. `-1` means *not measured* (unknown), as everywhere else. The full list of the dataset's 87 names is
`schema.DATASET_COLUMNS` (checked against `data/raw/web-phishing/train-1.csv` by `tests/test_schema.py`).

## The fourteen columns that are not exported

(`schema.DROPPED_FROM_FULL`; decided 2026-10-14. All of them are still *computed or computable*; the lab simply does not
export them. The reasons come from an analysis of the public set's training split, see "Why these were dropped" below.)

| Column | Why it is left out |
|---|---|
| `ratio_intRedirection`, `ratio_extRedirection`, `ratio_intErrors`, `ratio_extErrors` | **Cannot be captured.** The dataset's extractor sent a request to every hyperlink of the page. The lab never fetches a linked resource (threat model). |
| `google_index` | **Cannot be captured.** Needs one Google search per link (against Google's terms, and captchas). |
| `nb_or` | Not needed: 0 in every row of the public set. |
| `nb_hyphens` | Not needed: weak (correlation -0.10). It is **still a model input** under its cleaned name `number_of_hyphens`; only the extra table drops it. |
| `dns_record` | Not needed: weak (correlation 0.12). It is **still a model input** under the same name; only the extra table drops it. |
| `web_traffic` | Not needed: weak (correlation 0.06), and the Alexa rank it held no longer exists. |
| `nb_subdomains` | Exact duplicate: a function of `nb_dots` (1, 2 or 3). Still a model input as `subdomain_count`. |
| `ratio_extHyperlinks` | Exact duplicate: in the lab it is exactly `1 - ratio_intHyperlinks` (empty links count as internal). In the public set the two sum to 1 in only 87.6% of rows. |
| `ratio_nullHyperlinks`, `submit_email`, `sfh` | No information in the public set: 0 in every row (its 2020 extractor never measured them), so they cannot be evaluated. The lab could measure them; they were dropped to keep the table lean. |

If you also want `nb_hyphens` / `dns_record` / `nb_subdomains` out of the **model**, that is a different change (the trained
forests expect their 31 inputs); say so and it needs a retrain.

### Why these were dropped (and what that analysis is and is not)

A random forest (5-fold cross-validation) on the 7,658-row training split of the public set: 78 features 95.59%, 41
features (without constants, duplicates and weak columns) 95.42%, a difference smaller than the noise (+/- 0.5). It
only says which columns look redundant **on that dataset**, whose accuracy is driven by shortcuts (`nb_hyperlinks`,
`page_rank`, `nb_www`: big homepages versus deep links). It is not the accuracy of Le-Jet. 33 further columns are weak
there (rare flags such as `punycode`, `port`, `iframe`) and were **kept** on purpose: weak on that dataset does not mean
useless on SMS links. Choose the final features from the lab's own data.

## What the lab measures

| Group | Exported | How |
|---|---|---|
| URL-based | **53** of 56 | computed from the original link (dropped: `nb_or`, `nb_hyphens`, `nb_subdomains`); the two redirect counts come from the redirect chain the lab recorded |
| Content-based | **16** of 24 | the 12 original page features + 4 new ones (below) |
| External | **4** of 7 | `whois_registered_domain`, `domain_registration_length`, `domain_age` on the VM; `page_rank` on the host (Open PageRank, key never in the VM) |

So the VM fills 72 of the 73 columns; `page_rank` is added on the host.

## URL-based features: how well they reproduce the dataset

`tests/test_hannousse_url.py` recomputes all 56 URL features on the 11,430 URLs of the public set and compares with the
dataset's own value (53 of them are exported; the three dropped ones are still checked). Share of rows reproduced:

| Share | Columns |
|---|---|
| **100%** (32 columns) | `length_hostname`, `ip`, `nb_hyphens`, `nb_at`, `nb_qm`, `nb_and`, `nb_or`, `nb_eq`, `nb_underscore`, `nb_percent`, `nb_slash`, `nb_star`, `nb_colon`, `nb_comma`, `nb_semicolumn`, `nb_dollar`, `nb_dslash`, `https_token`, `ratio_digits_host`, `port`, `prefix_suffix`, `shortening_service`, `shortest_word_host`, `longest_word_host`, `avg_word_host`, `suspecious_tld`, `tld_in_path`, `tld_in_subdomain`, `http_in_path`, `nb_www`, `nb_com`, `phish_hints` |
| 99.9% – 99.99% | `nb_dots`, `nb_subdomains`, `shortest_words_raw`, `domain_in_brand` (99.99); `nb_tilde`, `nb_space`, `path_extension`, `statistical_report`, `longest_words_raw`, `char_repeat` (99.97–99.98); `shortest_word_path`, `length_words_raw`, `longest_word_path`, `length_url`, `ratio_digits_url` (99.95–99.96); `punycode`, `avg_words_raw`, `avg_word_path` (99.93–99.94) |
| 99.4% – 99.7% | `brand_in_path` 99.69, `abnormal_subdomain` 99.61, `brand_in_subdomain` 99.48 |
| **92.4%** | `random_domain` (the `nostril` "is this word nonsense" test; the original's exact input and library version are unknown, the 14-feature extractor already had this limit) |
| not comparable | `nb_redirection`, `nb_external_redirection` (the dataset counted `requests`' redirect history; here they come from the redirect chain the lab recorded, so they exist only when the page was fetched) |

The few 0.01–0.07% differences are links the dataset itself stored in a slightly different form (truncated, `&amp;`).

### Features that depend on a list the authors did not publish

| Column | What is used here |
|---|---|
| `domain_in_brand`, `brand_in_subdomain`, `brand_in_path` | the authors used a brand list from 101domain.com. **Reconstructed** from the dataset: the 52 second-level names it flags (`google`, `paypal`... only those). Brands that are not in the 2020 set are unknown to it (for example Philippine banks). |
| `suspecious_tld` | **Reconstructed**: the 26 suffixes the dataset flags. Other abused TLDs are not in the list. |
| `statistical_report` | The authors resolved the domain's IP address in 2020 and compared it with a published list of top phishing IPs. Here: a table of the 230 hosts / IPs the dataset flags, looked up by registrable domain or raw IP. A 2020 snapshot; unknown hosts get 0. |
| `abnormal_subdomain` | The paper's pattern `w[w]?[0-9]*`; the authors' exact regular expression could not be recovered, a fitted rule reproduces 99.6% (a raw IP, or a first sub-domain label starting with a digit or like `www1`/`www-x`). |

These lists live in `src/lejet/features/hannousse_lists.py` (a header explains it). Because they are reconstructed from the
dataset, **do not report these columns as an independent test on the dataset's own rows**: they reproduce it by
construction. On new links they are a conservative reading ("not a known brand / not a known bad host").

## Content-based features

The 12 original page features are unchanged. The 4 new ones are collected in the same single pass over the parsed
page (`hannousse_html.collect_facts`), with plain substring tests on capped script text (no regular expressions on
attacker-controlled text, nothing executed or fetched). The public set contains no pages, so they cannot be compared
with it row by row; `tests/test_hannousse_html.py` pins each definition with a small page.

| Column | Definition used | Note |
|---|---|---|
| `popup_window` | an inline script calls `prompt(`, or calls `window.open(` while the page has a text field | the paper: "pop-up windows with text fields" |
| `safe_anchor` | percentage (0-100, like the dataset) of `<a href>` whose link contains `#`, `javascript` or `mailto` | `0` if the page has no anchors. Contains `#` anywhere, so `/page#top` counts as unsafe. Individually the strongest of the new columns on the public set, but it mostly repeats what the original 12 give. |
| `onmouseover` | any element has an `onmouseover` attribute | the paper's wording |
| `right_clic` | `event.button==2` (also `===`, spaces ignored) in an inline script or an `on...` attribute | the paper's wording; modern right-click blocking via `contextmenu` is not detected |

## Which link and which page the columns describe

Same rule as the 31 model inputs (`WEB_INSPECTION_PLAN.md` 13.2b): URL-based columns describe the **original link as the
SMS contained it** (after the same `http://` normalisation); content columns describe the **final page** the victim
lands on; external columns describe the **final registrable domain**. A page that is a challenge page, a parked page
or could not be read has all content columns at `-1`.

## Running it

The VM must run the new build (the bundle is rebuilt with `D:\LeJetData\tools\repack_bundle.py` and deployed with
`scripts\deploy_to_vm.ps1`, which asks for the ops password). After that nothing changes in the daily steps:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_batch.ps1 -UrlFile <links.txt> -Batch <name>
```

now also writes `D:\LeJetData\clean\<name>.full_features.csv` (73 columns; `page_rank` is `-1` until the host step). To add
`page_rank` on the host:

```powershell
python -I scripts\lookup_page_rank.py D:\LeJetData\clean\<name>.clean.jsonl --out D:\LeJetData\clean\<name>.model_input.csv --env-file D:\LeJetData\secrets\.env
python -I scripts\add_full_features.py D:\LeJetData\clean\<name>.clean.jsonl --out D:\LeJetData\clean\<name>.full_features.csv --page-rank-csv D:\LeJetData\clean\<name>.model_input.csv
```

Batches exported before the new build still validate (their `features_full` is simply missing); only rows from the new
build carry the extra columns.

## Our own labelled table (smishing master + legitimate links)

Host-side scripts turn the master sheet into a table in the same format as the trimmed public set
(`D:\LeJetData\datasets\train-1_73features.csv`): `url`, the 73 features, `status`.

1. **The input table grows.** `D:\LeJetData\datasets\feature_input.csv` has the columns
   `url, status, source, trust, url_id, date_added`. `python scripts\build_labeled_input.py` adds only what is new:
   the smishing master links not in the table yet and the **same number of new legitimate links**, each row stamped
   with the date it was added; old rows are never changed. The link hunter does this by itself after
   `hunt.py --import-lab` (live links only, `--only-live`). For the new rows it writes the lab input files
   `feature_input_<stamp>_partNofM.txt` (120 links each, both classes in every part). The first run turned the older
   per-day file `feature_input_20261014.csv` into the start of the table (all its rows dated 2026-10-14).
   Legitimate links come from (the `source` column says which): the real legitimate SMS of the project's SMS dataset,
   **only from an allowlist of recognisable senders** (that dataset's "legitimate" label is noisy: it holds lookalikes
   such as `gcash-10-year.lc`), and two curated lists of well-known sites. Nothing is crawled or made up. When these
   lists run low the script says so; add more with `--curated <file of defanged legitimate links>`.
2. **One command runs everything pending.** `powershell -ExecutionPolicy Bypass -File .\scripts\run_parts.ps1` runs
   every part that has no result yet (oldest first; `-DryRun` shows the plan; after a failure just run it again), then
   `python scripts\build_feature_dataset.py --balance --lean`, which joins ALL lab results run so far
   (`D:\LeJetData\clean\feat*.clean.jsonl`) to the labels and writes `feature_dataset.csv` (one stable file, rebuilt each time) and
   `feature_dataset_lean.csv`. Only links whose page was fetched are kept (`--all` keeps the rest with `-1` page columns);
   `--platform-cap` (default 20) keeps at most 20 phishing rows per hosting platform such as pages.dev, because a kit copied
   hundreds of times onto one free platform would teach the table the platform; `--balance` then drops random rows of the larger
   class so both are equal; `--page-rank-csv` adds `page_rank`; `url` is
   the REAL link (`--defanged` for hxxps://a[.]com).

Limits to state in the thesis: `status = phishing` for every master row (the master holds feed phishing that is not
proven to be SMS-delivered; `trust = smishing-verified` marks the confirmed ones); the legitimate class is mostly
well-known homepages plus a few real SMS links, so a model can learn "homepage versus deep link" instead of legitimacy
(the same shortcut as in the public set); liveness differs by class (legitimate pages load far more often than
phishing ones), which is why `--balance` is applied after the lab, not before.

### A large legitimate pool (Tranco)
`python scripts\build_legit_pool.py` samples 4,000 legitimate-site links from the Tranco ranking (the public list downloaded
into `D:\LeJetData\input\tranco\`, with its download date and sha256 in `tranco_<date>_INFO.txt`) and writes the defanged pool
`D:\LeJetData\input\legit_tranco_pool.txt` (+ `.csv` with each site's rank). It takes the top 100,000 sites in three rank bands
(15% from 1-5,000, 35% from 5,001-20,000, 50% from 20,001-100,000: the very top refuses the lab's plain fetch, the middle loads
far more often), and skips infrastructure names (CDNs, ad/analytics networks), free-hosting platform roots, clearly adult or
gambling names, and the TLDs the phishing feeds are full of (.top, .xyz, .icu...). The input builder uses it automatically as
its third source (`source = legit_tranco`) and `python scripts\build_labeled_input.py --add-legit N` adds N legitimate links on
top of the matched ones. **Why:** the final table is balanced on pages that LOADED, and phishing pages load far less often
than legitimate ones, so the legitimate side needs many more links in the lab than the phishing side.
Limits: a popularity ranking is a proxy for "legitimate", not a safe list (a few ranked sites can be shady, more so in the
long tail); the legitimate links are homepages (`https://domain/`), so the "homepage versus deep link" shortcut remains;
excluding abused TLDs makes `suspecious_tld` an easier signal than it would be in the wild; the Tranco paper (Le Pochat et al.)
is from 2019, so cite the list itself (its ID, from tranco-list.eu) if your thesis requires sources from 2020 on.

## The daily command (any phishing domain, 2,500 + 2,500)
The target is 5,000 rows: 2,500 phishing (any domain, no SMS-style filter) and 2,500 legitimate, counted on pages that
LOADED. One command does a round:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\daily_phishing.ps1        # add -DryRun to only see the counts
```
1. `hunt.py --min-score -100` collects every fresh link from PhishTank (last 3 days), OpenPhish, Phishunt, PhishStats,
   Phishing.Database and urlscan.io (tagged phishing scans) into the master;
2. `build_labeled_input.py --legit-ratio 0.6 --platform-cap 20` adds the links not yet in `feature_input.csv`, at most 20 per
   hosting platform, with 0.6 legitimate Tranco links per phishing link (legitimate pages load about 59% of the time, phishing
   ones about 33%, so this balances the pages that load without wasting lab time), each row dated today;
3. `run_parts.ps1` runs the lab, fills `page_rank`, and rebuilds the tables, ending with a PROGRESS line.
Do it while the links are fresh: same-day OpenPhish links loaded 33%, PhishTank rows older than a month about 0%. A round
brings roughly 1,000 new phishing links, so about 300-400 loaded pages; expect about five rounds, a few days apart.
Limits: one brand can dominate a feed (Allegro was 22% of the first 5,000 PhishTank rows; the hunter caps 200 per brand); the
links are phishing in general, not proven SMS-delivered.
