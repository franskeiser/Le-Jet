# Decision: how to use website inspection when old SMS links are dead

| | |
|---|---|
| **Superseded** | 2026-10-14 (proposed): one extractor on one dataset, see `unified_architecture.md`. Kept as history. |
| **Status** | Option C implemented 2026-10-11. To be confirmed with the group and the adviser. |
| **Evidence** | `experiments/phase0/summary.txt` (from `src/data/phase0_domain_lookups.py`) |
| **Code** | `src/features/model/rf_extractor.py` (`extract_p_u`, `route_p_u`) |

## The facts

- In our SMS data, **82% of smishing domains no longer exist**, while **90% of legitimate ones still do** (Phase 0 DNS check on 200 smishing and 62 legitimate domains).
- Anything looked up today (the page, WHOIS, DNS, `page_rank`) mostly tells the model "the domain died". That correlates with smishing only because time passed, so it is a fake signal. When a real person receives a smishing SMS, the scam site is alive, so this signal never exists in real use.
- Our SMS data has no dates (`sms`, `label` only), so we cannot recover what a site looked like when the message was sent. Archives (Wayback Machine), historical WHOIS and certificate logs cannot fix this without message dates.

## Options

**A. Drop website inspection completely.** Model = CNN + URL text features + URL-only extractor.
- ✅ Simplest; no inspector, no VM, no fresh data collection.
- ❌ Drops the core idea of Le-Jet. The thesis loses its main contribution.

**B. One extractor for everything, trained with masking** (30% of training pages hidden).
- ✅ One extractor for all links.
- ❌ It would still receive today's lookups for old links, so extra rules are needed to block the fake signal.
- ❌ The masking rate is a guess that has to be justified.

**C. Two extractors, chosen by what is available, no masking (chosen).**
- Old SMS data → `url_only` extractor (14 link-text features, no lookups at all).
- Fresh links, inspected within hours → `full` extractor when the page loaded, `url_only` when it failed.
- ✅ No masking and no guessed rates; each extractor only sees the kind of data it was trained on.
- ✅ No fake "dead domain" signal anywhere.
- ✅ Easy to explain: "use page evidence only when it is real and current."
- ❌ Website inspection is proven only on the fresh set, which still has to be collected.
- ❌ A fresh link whose page fails also loses its WHOIS/DNS data, even if that lookup worked. A small loss.

**D. Remove SMS messages whose links are dead; inspect only the live ones.**
- ❌ Of the smishing messages whose domain we checked, only 28.7% still have a live domain, against 98.6% of legitimate ones. This removes most smishing links and almost no legitimate ones.
- ❌ The surviving "live" smishing domains are mostly link shorteners (`tinyurl.com`, `bit.ly`, `v.ht`, …) and real brands mentioned in scam texts (`maya.ph`). The actual scam domains are nearly all dead, so there are almost no real scam pages left to inspect.
- ❌ Messages with links would become mostly legitimate, creating the opposite shortcut ("has a link = legitimate").
- ❌ The surviving smishing messages are unusual cases, so results would not apply to typical smishing.

## Why C

C keeps website inspection as the thesis contribution, removes the fake signal, needs no masking and no removed data, and is the simplest to defend.

## Extractor results (public web set, test part, mean over seeds 42 / 123 / 2024)

| Extractor | Inputs | F1 | ROC-AUC |
|---|---|---|---|
| `full` | 31 features, complete pages | 0.930 | 0.978 |
| `url_only` | 14 link-text features | 0.835 | 0.903 |

The gap between the two is what page and domain evidence can add. It has to be confirmed on the fresh SMS set.

## What this means for the thesis

1. **Main results (old SMS data):** CNN + URL text features, with the `url_only` extractor. Large, but makes no web-inspection claim.
2. **Web-inspection results (fresh set, dataset C):** links collected and inspected within hours. This is the only evidence for the website-inspection claim, so collection should start early and include legitimate messages with links.

## Questions for the group and the adviser

1. Do we agree that results on old SMS data use only the URL text features?
2. Is the fresh set enough to support the website-inspection claim? How many links can we realistically collect, and by when?
3. Who handles the fresh collection, including legitimate messages with links?
