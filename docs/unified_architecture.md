# Le-Jet: unified architecture (one dataset, one Random Forest extractor)

**Written 2026-10-14. Status: proposed, implemented, to be confirmed with the group and the adviser.**
It replaces the two-extractor design of `decision_web_inspection.md` (Option C). That file and `final_model.md` describe the previous design and stay as history.

**Constraint from the group:** page inspection and the page/domain features that come from the web must stay in the model.

## Why change
The old design trained two forests (`url_only`, `full`) on a separate public web set (11,430 pages), froze them, and routed each SMS link to one of them. That gave two models, a domain shift from web pages to SMS links, and routing logic. Giving one forest the page features for old SMS links is not an option either: 82% of old smishing domains are dead, so a lookup made today would teach "dead domain = scam" (`decision_web_inspection.md`).

## The design

```
ONE dataset, one row per SMS: text + link + 14 link-text features + 17 page/domain features
   │
   one grouped split (data/splits: train / rf_fit / test), by domain or template
   │
   RF extractor (one forest, 31 inputs, NaN allowed)
   ├─ train rows: out-of-fold P, u, c_*   (5-fold, grouped; a row is never scored by a forest that saw it)
   └─ final forest on all train link rows -> rf_fit, test, live links
   │
SMS text ─ fastText ─ CNN (64 filters, kernel 3) ───────────────┐
has_url, P, u, c_* (34 values) ─────────────────────────────────┼─ static gates ─ Dense 64 ─ softmax
```

- **No link, or a redacted `<URL>` placeholder:** the forest is not asked; P = u = -1, c_* = 0.
- **Link without page evidence:** the 17 page columns are NaN and the forest reads the 14 link-text features.
- **Gates:** CNN gates from a forest on the frozen CNN's features of the `rf_fit` rows; RF-output gates from the final forest's permutation importance on the same rows (held out from both models); `has_url`, P, u fixed at 1.0. The `rf_fit` split is kept for this and for nothing else, so the split files are unchanged.

## The dataset (`data/processed/unified_dataset.csv`)
Built by `python -m src.unified.build_unified_dataset`.

| Group | Columns |
|---|---|
| Identity | `message_id`, `label`, `source`, `template_id` |
| Text | `sms_text` |
| Link | `link_status`, `link`, `link_domain`, `n_links`, `has_url` |
| Link-text features (14) | `url_length` … `tld_in_path` |
| Page evidence | `inspect_status` (`loaded` / `failed` / `not_attempted`), `page_origin` (`hybrid_temp` / `inspector` / `none`) |
| Page/domain features (17) | `page_rank`, 12 webpage, 4 external; NaN unless `loaded` |

## Today: placeholder page features
No fresh inspected set exists yet. Until it does, the 7,869 old SMS rows with a real link borrow the 17 page/domain values of a randomly chosen web-set page of the same class (`src/data/build_hybrid_temp_links.py`; smishing gets phishing, legitimate gets legitimate; each page used once). They carry `page_origin = hybrid_temp`.

**Scores trained on this file are not study results.** The borrowed page values are unrelated to the SMS link they sit next to, so the forest learns the page signal and the link signal independently and reports optimistic numbers.

## When the fresh set arrives
1. Collect SMS with links, both classes (legitimate with links included), each link inspected within hours; record `hours_since_received`.
2. Write their rows with `inspect_status = loaded` (or `failed`, with NaN page columns) and `page_origin = inspector`.
3. Drop every `hybrid_temp` row and rebuild. Nothing else changes.
4. Run page evidence on versus off on the fresh rows: `--rf-inputs full` against `--rf-inputs url_only`. That comparison is the page-inspection result.

Two things to watch with NaN page columns: the forest must not learn that missing page features mean a class (check the NaN rate per class), and old rows without inspection should not be mixed into the RF's training set once the fresh rows exist.

## Training
```
python -m src.unified.build_hybrid_temp_links
python -m src.unified.build_unified_dataset
python -m src.unified.run_unified                       # full RF
python -m src.unified.run_unified --rf-inputs url_only  # page-inspection ablation
```
Per seed (42, 123, 2024): tune the forest on the train link rows (grouped CV) → out-of-fold P/u/c_* → final forest → Phase A (CNN) → gates → Phase C (head) → test, read once. Results go to `experiments/unified/<rf-inputs>/`, models to `models/unified/<rf-inputs>/`.

## Not yet moved to the new design
The serving path (`src/serving/`, live inspection, the defense demo) and `docs/final_model.md` still use the two web-set extractors. Move them after the unified model is trained on real fresh data, so a demo is never served from placeholder-trained weights.
