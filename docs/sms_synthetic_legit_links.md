# Synthetic legitimate messages with links

| | |
|---|---|
| **Script** | `src/data/add_synthetic_legit_links.py` |
| **Input** | `data/raw/sms/v2_2026-09-15.csv` (18,885 rows: 10,955 smishing, 7,930 legitimate) |
| **Output** | `data/processed/sms_v2_balanced.csv` (21,910 rows: 10,955 per class) |
| **Added** | 3,025 synthetic legitimate messages, every one with a link |

Run from the project root:

```bash
python -m src.data.add_synthetic_legit_links
```

## Why

The CNN turns every link into the same `<URL>` token. In the real data, a link (or the `<URL>` placeholder) appears in 6.1% of legitimate messages and 67.7% of smishing messages, so the model can learn "has a link => smishing".

## How many and why that number

3,025 rows is exactly the gap between the classes (10,955 − 7,930), so both classes end up the same size.

| | Before | After |
|---|---|---|
| Legitimate messages | 7,930 | 10,955 |
| Legitimate with a link | 483 (6.1%) | 3,500 (31.9%) |
| Smishing with a link | 7,232 (66.0%) | unchanged |

Closing the link gap completely would need about 15,000 synthetic rows, more than the real legitimate messages. The CNN would then mostly learn the template style, so the gap is narrowed, not closed. Report test results separately for messages with and without links.

(The link counts here use the CNN's URL pattern from `src/features/text_preprocessing.py`, so they differ slightly from the Phase 0 counts.)

## Where the texts come from

The 46 legitimate templates written for SmishGuard (`src/data/build_v2_with_synthetic.py`): social shares, work and school links, delivery, bill and advisory notices, in English, Tagalog and Taglish. Rows are spread evenly over the templates (12 to 69 each; a few templates have small fill pools and run out of unique texts).

Four SmishGuard templates (`bank_statement_info`, `telco_bill`, `netflix_receipt`, `flight_confirmed`) chose the brand and the website independently, producing texts like "UnionBank: ... Learn more at securitybank.com". A mismatched brand and domain is a smishing sign, so this script replaces them with versions that keep each brand with its own domain.

## Columns

| Column | Meaning |
|---|---|
| `sms` | Message text |
| `label` | `smishing` or `legitimate` |
| `source` | `v2_2026-09-15` for real rows, `synthetic_legit_url` for added rows |
| `template_id` | Template that produced a synthetic row; empty for real rows |

## Rules for training

- **Split synthetic rows by `template_id`.** Synthetic test rows must come from templates the model never trained on, or the score only measures template memorization.
- **Report results on real rows only** as the main number; synthetic rows are a training aid, not evidence.
- Synthetic rows are 27.6% of the legitimate class. Say so in the thesis and label them as synthetic.
- No lookalike smishing was added. If the model starts treating any brand domain as legitimate (SmishGuard saw this), add the lookalike smishing templates from `build_v2_with_synthetic.py` with the same number of extra legitimate rows to keep the classes balanced.
