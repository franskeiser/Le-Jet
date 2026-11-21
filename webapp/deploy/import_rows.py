"""
import_rows.py -- append rows that ALREADY have their features to the website's shared CSV (no inspection).

Run only by webapp/deploy/import-rows.sh, as the website's own user, while the site is stopped:

    /opt/lejet-web/venv/bin/python -I import_rows.py <rows.csv>

<rows.csv> must have EXACTLY the website's 50 columns (webapp/dataset.COLUMNS), in that order; the message_id cell is ignored
(new ids web_NNNNNN are given in order, continuing the numbering of the file). A row is skipped, not added, when
  * the same message text with the same link is already in the dataset (the site's own duplicate rule), or twice in the import;
  * the label or language is not one of the site's allowed values, or the message is empty.
Nothing is changed unless the whole input passes the header check. Missing values are -1, never blank (as the site writes them).

UPDATE MODE (`--update`): the rows are NOT added. Each one is matched to an existing row by (message text, link) and only that
row's inspection result is replaced: final_url, link_domain, page_fetched, fetch_status and the 37 features (used when a better
inspection of the same link exists, e.g. a page the first fetch could not read). Message, label, language, date, id and
label_verified_by are never touched. Rows that match nothing are counted and ignored. The file is rewritten as a whole (to a
temporary file, then renamed), so a crash never leaves a half-written CSV.

It uses the site's own Store (webapp/dataset.py), so rows are written exactly as a normal submission would be written.
"""
from __future__ import annotations

import csv
import os
import sys

ROOT = os.environ.get("LEJET_WEB_ROOT", "/opt/lejet-web")          # (tests point this at the repo)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from webapp import dataset                                   # noqa: E402

DATASET = os.environ.get("LEJET_DATASET", "/var/lib/lejetweb/web_submissions.csv")


NL = chr(10)
UPDATE_COLUMNS = ["final_url", "link_domain", "page_fetched", "fetch_status"] + dataset.FEATURES_37


def update(path: str, rows) -> int:
    with open(DATASET, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != dataset.COLUMNS:
            print("the website's CSV does not have the expected columns; nothing was changed", file=sys.stderr)
            return 3
        existing = list(reader)
    index = {(r["sms_text"], r["link"]): r for r in existing}
    changed = unmatched = same = 0
    for r in rows:
        hit = index.get((dataset._cell("sms_text", (r["sms_text"] or "").strip()), dataset._cell("link", r["link"])))
        if hit is None:
            unmatched += 1
            continue
        new = {c: str(dataset._cell(c, r[c])) for c in UPDATE_COLUMNS}
        if all(hit[c] == new[c] for c in UPDATE_COLUMNS):
            same += 1
            continue
        hit.update(new)
        changed += 1
    tmp = DATASET + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=dataset.COLUMNS, lineterminator=NL)
        writer.writeheader()
        writer.writerows(existing)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, DATASET)
    print(f"update mode: read {len(rows)}; changed {changed}; already identical {same}; matched nothing {unmatched}; rows in the file: {len(existing)}")
    return 0


def main(argv) -> int:
    do_update = "--update" in argv
    argv = [a for a in argv if a != "--update"]
    if len(argv) != 2:
        print("usage: import_rows.py [--update] <rows.csv>", file=sys.stderr)
        return 2
    with open(argv[1], "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != dataset.COLUMNS:
            print("the input does not have the website's 50 columns in the right order; nothing was changed", file=sys.stderr)
            return 3
        rows = list(reader)
    if do_update:
        return update(argv[1], rows)
    store = dataset.Store(DATASET)
    before = len(store)
    added = {"smishing": 0, "legitimate": 0}
    skipped = {"duplicate": 0, "bad_value": 0}
    for r in rows:
        text = (r["sms_text"] or "").strip()
        if not text or r["label"] not in dataset.LABELS or r["language"] not in dataset.LANGUAGES:
            skipped["bad_value"] += 1
            continue
        r["sms_text"] = text
        if store.existing(text, r["link"]):
            skipped["duplicate"] += 1
            continue
        r["message_id"] = store.next_id()
        store.append(r)
        added[r["label"]] += 1
    print(f"rows in the file before: {before}; read {len(rows)}; added {added}; skipped {skipped}; rows now: {len(store)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
