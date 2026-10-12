# The 12 webpage features: definitions, decisions, ambiguities

Implemented in `src/lejet/features/html_features.py`; tested in `tests/test_html_features.py`
(90+ cases, all on fixture pages with known expected values, no network).
Plan references: `WEB_INSPECTION_PLAN.md` §13.5b, §14.0.3, §16 Level 0.

> **Honest status.** The public set's pages are mostly dead, so these definitions **cannot be
> validated against the dataset's own pages**. They follow the extractor of the source dataset
> (Hannousse & Yahiouche, 2021) *as far as I could reconstruct it*, and each choice was checked
> against the **value ranges and group means of the cleaned public set**
> (`data/processed/web_phishing_clean.csv`). Where the paper's code is not available to me, the
> choice below is a documented decision, not a verified fact. Please compare it with the paper's
> feature table before the thesis states these definitions as "the paper's".

## Scales (checked against the public set)

| Feature | Scale | Public set |
|---|---|---|
| `nb_hyperlinks` | integer count | 0 to 4659, median 34 |
| `ratio_intHyperlinks` | **0 to 1** | median 0.74 |
| `links_in_tags`, `ratio_intMedia`, `ratio_extMedia` | **0 to 100** (percent) | max 100 |
| `nb_extCSS` | integer count | 0 to 124 |
| `login_form`, `external_favicon`, `iframe`, `empty_title`, `domain_in_title`, `domain_with_copyright` | 0 / 1 | |

Missing values are `-1` (all 12 when `html_available = 0`). Ratios with an empty denominator are `0`.

## Link groups

The page is walked once (at most 20,000 elements). Each URL-bearing attribute goes into a group:

| Group | Source |
|---|---|
| href | `<a href>`, `<area href>` |
| link | `<link href>`, `<script src>`, `<meta content="http(s)://...">` |
| media | `<img src>`, `<audio src>`, `<video src>`, `<embed src>`, `<iframe src>` |
| form | `<form action>` (only forms that HAVE an action attribute) |
| css | `<link rel~=stylesheet href>` |
| favicon | `<link rel~=icon href>` inside `<head>` |

Each link is **internal**, **external** or **null**:

* **null**: empty, `#`, `about:blank`, or a `javascript:` URL.
* **internal**: relative, a non-http(s) scheme (`mailto:`, `tel:`, `data:`), or the **same registrable
  domain** as the page (public suffix list with private suffixes ON, so `a.pages.dev` and
  `b.pages.dev` are different domains).
* **external**: an http(s) URL on another registrable domain. For an IP host, the host itself is compared.

## The features

| Feature | Definition |
|---|---|
| `nb_hyperlinks` | Total entries over the six groups. |
| `ratio_intHyperlinks` | (internal + null) / total, 0 to 1. |
| `nb_extCSS` | External entries in the css group. |
| `login_form` | 1 if any form (with an action attribute) has a null or external action, **or** an internal action that is a bare `name.php`. |
| `external_favicon` | 1 if any `<head>` icon link is external. |
| `links_in_tags` | 100 x (internal + null) / total in the link group. |
| `ratio_intMedia`, `ratio_extMedia` | 100 x internal(+null) / total, and 100 x external / total, in the media group. |
| `iframe` | 1 if any iframe is hidden: `display:none`, `visibility:hidden`, the `hidden` attribute, or width AND height zero. |
| `empty_title` | 1 if there is no `<title>` or its text is empty. |
| `domain_in_title` | **0 if the domain label appears in the title, else 1** (direction as in the public set: phishing mean 0.92 vs legitimate 0.63 means 1 = "not in the title"). |
| `domain_with_copyright` | 1 if a (c), (tm) or (r) mark exists and the domain label is NOT within 50 characters of the first mark; 0 if there is no mark or the label is next to it. |

"Domain label" = the label left of the public suffix with private suffixes OFF (the dataset's own
extractor, the same split `src/features/link_features.py` documents): `login.gcash-verify.xyz` gives
`gcash-verify`; `evil.pages.dev` gives `pages`.

## Decisions on ambiguities (all of them)

1. **Null links count as internal** in `nb_hyperlinks`' ratio. In the public set `ratio_nullHyperlinks`
   is 0 in every row, so its extractor never separated them. (The plan asked to settle this and write it down.)
2. **Stylesheets and icons are counted in more than one group** (a stylesheet is a `<link>` and a css
   entry; an icon is a `<link>` and a favicon entry). This reproduces the structure of the source extractor
   so `nb_hyperlinks` keeps its meaning; it is a deliberate duplication, not a bug.
3. **Internal/external is judged by registrable domain**, not by substring tests on the URL text; relative
   links are resolved against the page URL first. Close to the original, but not identical on odd URLs.
4. **`<meta content>` counts only when it starts with `http://` or `https://`** (Open Graph images, etc.).
5. **Media includes `video` and `iframe src`**; the plan's prose lists img/audio/video/embed.
6. **Forms without an `action` attribute are ignored** (they submit to the page itself).
7. **Copyright window is clamped at the start of the text** (`max(0, i - 50)`); a naive negative slice would
   silently search the wrong part of the page.
8. **The text searched for the copyright mark includes the `<title>`** (visible text of the whole document
   minus script/style/noscript). On very short pages the title can therefore sit within 50 characters of the mark.
9. **`domain_registration_length`** (RDAP, not HTML) is days **until expiry**. The public set's median
   registration length is 242 days while its median domain age is 3,993 days, so it cannot be
   "expiration minus registration". Stored twice: `domain_registration_length` and `days_to_expiry`.

## Redirects found without running JavaScript (Stage 3)

* `<meta http-equiv="refresh" content="N; url=...">` with `N <= 10` seconds.
* A **trivial** script redirect with a string literal target (`location = "..."`, `location.href = "..."`,
  `location.replace("...")`, `location.assign("...")`), **only on pages of 32 KB or less**. On a large page the
  same text is usually an event handler (a click, not a redirect), so it is ignored.
* Each redirect counts as a hop (shared limit of 8) and the target goes through the Stage 0 gate again.

## Page verdicts (open item V4 of the plan: "challenge/parked detectors written and their rates reported")

A page that is not the real page gets `fail_reason` and **no features**, so it routes to the `url_only`
extractor instead of being scored as if it were the site:

* `challenge_page`: a bot-wall title ("Just a moment", "Attention Required", "Checking your browser",
  "Verify you are human", ...) or, on a small page, known challenge markers in the HTML
  (`cf-browser-verification`, `challenge-platform`, `g-recaptcha`, `h-captcha`, ...).
* `parked_or_default_page`: default server pages ("Welcome to nginx", "Apache2 ... Default Page",
  "Index of /", "It works!"), parking and for-sale pages ("This domain is for sale"), hosting
  suspension pages ("Account suspended").

These are string lists, not a classifier: they will miss some pages and may flag a few real ones.
**The rates must be reported per batch** (`challenge_page` and `parked_or_default_page` counts are in
every manifest) and checked by hand on a sample in the pilot.

## Things the static parse cannot see

JavaScript-built pages look nearly empty. Every row carries `html_bytes`, `script_count` and
`near_empty` (less than 50 characters of visible text) so the share of such pages can be measured and
reported (plan §20). The optional browser stage is not implemented in v1.
