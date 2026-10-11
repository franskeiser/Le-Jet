# Web Phishing Dataset Cleaning

| | |
|---|---|
| **Script** | `src/data/clean_web_phishing.py` |
| **Input** | `data/interim/hussein-train-and-test-merged.csv` (11,430 rows × 89 columns) |
| **Output** | `data/processed/web_phishing_clean.csv` (11,430 rows × 34 columns) |
| **Labels** | `status`: 5,715 legitimate, 5,715 phishing |

Run from the project root:

```bash
python src/data/clean_web_phishing.py
```

Optional arguments: `python src/data/clean_web_phishing.py <input_csv> <output_csv>`.

## Output columns

The output keeps `url` (for traceability), 32 features in three groups, and the `status` label.

### URL structural features (16)

| Our column | Source | Notes |
|---|---|---|
| `url_length` | `length_url` | Renamed |
| `https_presence` | **New**, computed | 1 if the URL starts with `https://`, else 0 |
| `number_of_hyphens` | `nb_hyphens` | Renamed |
| `number_of_slashes` | `nb_slash` | Renamed |
| `subdomain_count` | `nb_subdomains` | Renamed |
| `special_char_count` | **New**, computed | See [Decisions](#decisions) |
| `suspicious_url_token_count` | **New**, computed | See [Decisions](#decisions) |
| `shortened_url_indicator` | `shortening_service` | Renamed |
| `has_ip_address` | `ip` | Renamed |
| `suspicious_keyword_count` | **New**, computed | See [Decisions](#decisions) |
| `has_url` | **New**, constant | Always 1 (see [Decisions](#decisions)) |
| `page_rank` | `page_rank` | Unchanged (Open PageRank) |
| `path_length` | **New**, computed | See [Decisions](#decisions) |
| `random_domain` | `random_domain` | Unchanged |
| `length_hostname` | `length_hostname` | Unchanged |
| `tld_in_path` | `tld_in_path` | Unchanged; the feature list's `tld_in_pth` was a typo |

### Webpage features (12), all unchanged

`nb_hyperlinks`, `ratio_intHyperlinks`, `nb_extCSS`, `login_form`, `external_favicon`, `links_in_tags`, `ratio_intMedia`, `ratio_extMedia`, `iframe`, `empty_title`, `domain_in_title`, `domain_with_copyright`

### External-based features (4)

| Column | Change |
|---|---|
| `whois_registered_domain` | Unchanged. `1` = not found in WHOIS (the suspicious value), `0` = registered |
| `domain_registration_length` | Negative values set to `-1` |
| `domain_age` | Negative values set to `-1` |
| `dns_record` | Unchanged. `1` = no DNS record (the suspicious value), `0` = the domain resolves |

### Dropped columns

The other 61 source columns are dropped, including `ratio_nullHyperlinks` and `sfh`, which are 0 in every row (see [Decisions](#decisions)). They include the separate character counts (`nb_at`, `nb_qm`, etc.), the word-length stats, and `web_traffic`, `google_index`, `phish_hints` and `https_token`. Some dropped columns feed into the new features (see below).

## Decisions

1. **New features are computed from the `url` text.** This lets the same code produce identical features for the SMS links later, without depending on the original dataset's feature extractor.

2. **`https_presence`.** It's 1 when the URL starts with `https://`. The source column `https_token` is the exact inverse (1 means plain http). We checked this on all 11,430 rows and dropped `https_token` to avoid a redundant column.

3. **`special_char_count`.** It counts these characters in the URL: `@ ? & % = _ ~ * : , ; $ ! # +  | ^` and spaces.
   - The scheme prefix (`http://`, `https://`) is stripped first so every URL doesn't score a colon.
   - Hyphens, slashes and dots are **not** counted because they already have their own features or appear in every URL.

4. **`suspicious_url_token_count`.** It counts occurrences of path or file tokens that are common in phishing kits: `wp, login, includes, admin, content, site, images, js, alibaba, css, myaccount, dropbox, themes, plugins, signin, view`.
   - This is the same list behind the source column `phish_hints`, and our version matches `phish_hints` on 100% of rows.
   - It's renamed and recomputed so it can be applied to new URLs.

5. **`suspicious_keyword_count`.** It counts distinct social-engineering words in the URL: `login, log-in, signin, sign-in, verify, verification, account, secure, security, update, bank, banking, confirm, password, webscr, ebayisapi, paypal, billing, unlock, suspend, wallet, authenticate, validation, recover, support, free, bonus, lucky`.
   - The difference from feature 4: feature 4 targets the *structure* of phishing-kit URLs, while this one targets *lure words* aimed at the user.

6. **`path_length`.** No source column existed. It's the length of the URL path, i.e. the part after the domain up to `?` or `#`. The query string is not included. If a URL can't be parsed, the fallback is everything after the first `/` that follows the host.

7. **`has_url`.** It's kept as a constant `1` because every row in this dataset is a URL. It exists so the web and SMS feature sets share the same columns. In the SMS data it will tell messages with links apart from messages without them.

8. **WHOIS failures → `-1`.** Negative `domain_age` values (1,837 rows, ranging from -12 to -1) and negative `domain_registration_length` values (46 rows) are lookup failures, not real durations. They are all normalized to a single `-1` sentinel and the rows are kept.

9. **Duplicates are kept.** One URL (`http://e710z0ear.du.r.appspot.com/c:/users/user/downlo`) appears twice, at rows 3811 and 4927.
   - The rows differ in `length_url` (54 vs 62), word stats, `ratio_extRedirection`, `domain_age` and `domain_registration_length`. This suggests two different original URLs that the source truncated to the same string.
   - Both rows are kept.

10. **`ratio_nullHyperlinks` and `sfh` are dropped.** Both are 0 in all 11,430 rows, so no model can learn anything from them, and the inspector would have to compute them for nothing.

11. **Column names follow our feature list.** Source names are kept only where our list already used them.

## Quick sanity check (class means)

| Feature | Legitimate | Phishing |
|---|---|---|
| `https_presence` | 0.44 | 0.33 |
| `special_char_count` | 0.51 | 1.86 |
| `suspicious_url_token_count` | 0.05 | 0.61 |
| `suspicious_keyword_count` | 0.05 | 0.48 |
| `path_length` | 20.16 | 26.43 |
