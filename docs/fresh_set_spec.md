# Fresh-set CSV: what the inspector must produce

The fresh set is the test for the web-inspection claim (`data/dataset_and_training_plan.md` §8): recent SMS links, visited within hours of receipt, scored by the frozen RF extractors. **It is for testing only; it never enters training, gating or tuning.**

One row per message. File: `data/fresh/fresh_<YYYY-MM-DD>.csv` (UTF-8, header row, comma-separated). Written once per batch and never edited afterwards.

## Column groups

### 1. Message (6)

| Column | Type | Notes |
|---|---|---|
| `message_id` | text | Unique. Prefix `fresh_`. |
| `sms_text` | text | Personal details removed (consent and ethics sign-off first). |
| `label` | text | `smishing` or `legitimate`. Ground truth, set when collected; **never relabel because a page was dead or looked clean** (plan §6). |
| `source` | text | Where it came from (own phone, volunteer, forum...). |
| `link` | text | The link exactly as it appears in the SMS (the first link, as in the old data). Empty if the message has none. |
| `link_status` | text | `link` or `none`. Never `redacted`: fresh messages keep their real link. |

### 2. Link-text features (14) + `has_url`

Computed from the **original raw link** in the SMS, not the final URL after redirects (`WEB_INSPECTION_PLAN.md` §13.2b). **Do not reimplement them.** Call `src.features.link_features.link_text_features(normalize_link(link))` for a message with a link and `no_link_features()` for one without, so the values match the training data exactly.

`url_length`, `https_presence`, `number_of_hyphens`, `number_of_slashes`, `subdomain_count`, `special_char_count`, `suspicious_url_token_count`, `shortened_url_indicator`, `has_ip_address`, `suspicious_keyword_count`, `path_length`, `random_domain`, `length_hostname`, `tld_in_path`, plus `has_url` (1 if the message has a link, else 0).

No link → all 14 are `-1` and `has_url = 0`.

### 3. Page, domain and external features (1 + 12 + 4)

Needed by the `full` extractor. Describe the **final URL after redirects** (§13.2b). Definitions are in `docs/web_phishing_cleaning.md` and `WEB_INSPECTION_PLAN.md` (§13; the 20-feature list there is superseded by these 32).

| Group | Columns |
|---|---|
| Domain rank (1) | `page_rank` (Open PageRank) |
| Webpage (12) | `nb_hyperlinks`, `ratio_intHyperlinks`, `nb_extCSS`, `login_form`, `external_favicon`, `links_in_tags`, `ratio_intMedia`, `ratio_extMedia`, `iframe`, `empty_title`, `domain_in_title`, `domain_with_copyright` |
| External (4) | `whois_registered_domain` (1 = not found, the suspicious value), `domain_registration_length`, `domain_age`, `dns_record` (1 = no DNS record, the suspicious value) |

Column names and spellings must match exactly. Same units as the public set (ratios in percent, ages and lengths in days).

**Missing values: always `-1`, never blank, NaN or 0.**
- Page not fetched (`page_fetched = 0`): the 12 webpage columns are `-1`.
- A lookup failed (WHOIS/RDAP, DNS, Open PageRank): that column is `-1`.
- No link: all 17 are `-1`.

### 4. Inspection record (8)

| Column | Type | Notes |
|---|---|---|
| `page_fetched` | 0/1 | **1 only if the page HTML was actually retrieved and parsed.** This alone decides which extractor runs: 1 → `full`, 0 → `url_only` (`route_p_u`). Never an input feature. |
| `fetch_status` | text | One of `live`, `redirected`, `dead`, `blocked`, `timeout`, `not_attempted` (plan §6). Reporting only. |
| `final_url` | text | URL after redirects. Empty if not fetched. |
| `final_domain` | text | Registrable domain of `final_url`. |
| `n_redirects` | int | `-1` if not fetched. |
| `received_at` | ISO 8601, UTC | When the SMS arrived. |
| `visited_at` | ISO 8601, UTC | When the inspector fetched the page. Empty if never attempted. |
| `hours_to_visit` | float | `visited_at − received_at`, in hours. Shorter is better; the plan requires it to be recorded (§8). |

### 5. Optional, recommended

| Column | Notes |
|---|---|
| `html_sha256` | Hash of the saved page text (the HTML itself is stored separately as inert text, never in this CSV). |
| `batch_id` | Ties the row to its run manifest. |

## Rules the file must satisfy

1. Exactly the column names above (the 31 extractor inputs, `has_url`, and the message and record columns). Extra columns are allowed; missing ones are not.
2. All feature columns numeric, no empty cells. Use `-1`.
3. `page_fetched = 1` ⇒ `fetch_status` ∈ {`live`, `redirected`} and `final_url` is non-empty.
4. `has_url = 0` ⇒ `link_status = none`, `page_fetched = 0`, and all 31 extractor inputs `-1`.
5. `message_id` unique; each label is `smishing` or `legitimate`.
6. Both classes must contain messages with links. A set where only the smishing messages have links reproduces the "has a link means smishing" shortcut.
7. Dead and blocked links stay in the file. The share of live links is itself a result (plan §9).
8. No row from the old SMS data, and none of its domains without being flagged (plan §4, overlap check).

## How the model side will use it

- `full` extractor (31 inputs: 14 link-text + `page_rank` + 12 webpage + 4 external) on rows with `page_fetched = 1`.
- `url_only` extractor (the 14 link-text columns) on every other row with a link.
- Reported in three groups: no link, link with a dead or blocked page, link with a live page; claims about web inspection only on the live group (plan §9).
