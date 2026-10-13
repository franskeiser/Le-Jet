# Live set, head training on it, and the serving API

This page records what was added on 2026-10-13: a dataset of links the lab really inspected, how it is kept apart
from the thesis training, and how the frontend is connected to the final model.

## 1. The live set (`data/live/live_links.csv`)

Built by `python -m src.data.build_live_set --batch smish03:smishing:team --batch benign01:legitimate:generated`
from validated lab batches (`D:\LeJetData\clean\<batch>.clean.jsonl` + the Open PageRank file).

* **Smishing** links: the team's collection, opened in the isolated VMs. **Legitimate** links: generated well-known sites
  (`D:\LeJetData\input\benign_gen.txt`), opened the same way. Labels are the ground truth given by us, never changed by
  whether a page was reachable (a dead smishing link stays smishing).
* One row per link: the 14 link-spelling features, page_rank, 12 webpage features, 4 external features (`-1` when the
  page was not reached), where the link led (defanged), fetch status, a `key` = sha256 of the normalised link.
  No raw clickable URL is stored anywhere in the file.
* **Two parts, split by domain group:** `live_train` (may train the fusion head) and `fresh_test` (never used to train,
  gate or tune). Shorteners are grouped by where they lead. Domains already seen by the extractors (web set) or the CNN
  (SMS data) are forced into `live_train` (`overlap_moved`), so `fresh_test` is domains nobody has seen.
  `check_leaks` fails the build if a group is in both parts, a blocked domain is in the test part, a link repeats, or
  a class is missing from a part. `data/live/live_report.json` has the counts.

## 2. What may and may not change the thesis training

| Part | Uses live data? |
|---|---|
| RF extractors (`url_only`, `full`) | No. Trained on the public web set only, frozen (training rules). |
| CNN (Phase A) and the gates | No. Real SMS data only. |
| Fusion head (Phase C) | **Optional**: `run_experiment --live-train data/live/live_links.csv --tag _live`. Adds `live_train` links wrapped in synthetic SMS wording (half class-specific, half ambiguous), each read with or without its page evidence (`MixedAuxDataset`). |
| Validation / test scores in the thesis tables | Real messages only, unchanged. |
| `fresh_test` | Only `src.data.eval_live_set`, after training. |

The default `run_experiment` command is unchanged; the live head is a separate tagged run (`models/fusion/rf_vec_live`,
`experiments/.../rf_vec_live`), so the committed thesis results cannot be overwritten by accident.
`--reuse-from models/fusion/rf_vec` copies Phase A and the gates of the finished run, so only Phase C differs.

## 3. Evaluation (`python -m src.data.eval_live_set`)

* Link level: extractors alone, `url_only` vs `full` on the same links, with and without shorteners, 3 seeds.
* Message level: ambiguous text ("See details here <link>") wrapped around each fresh-test link; the final head with the
  page shown vs hidden vs the CNN alone.
* Everything has Wilson 95% intervals. The sample is small and the two classes come from different sources, so this is a
  sanity check, not a thesis headline.

## 4. Serving: `src/serving/lejet_service.py`, `src/serving/api.py`

`python -m src.serving.api` → `http://127.0.0.1:5000` (the frontend's default `VITE_API_BASE_URL`).

* `POST /predict {message, model_id?}` → the contract of `frontend/src/data/backendClient.js`, plus `inspection`.
  Models: `fusion_gated` (final `rf_vec` head, seed 42) and `cnn_only_baseline`; other ids answer `unknown_model`.
* **The server never opens a link itself.** Page evidence comes only from the store of links the lab inspected
  (`live_links.csv` plus the on-demand file, looked up by sha256). A link in the store goes to the `full` extractor
  (`page_fetched = 1`); a link the lab could not load, or never saw, goes to `url_only`, from its spelling alone.
* **Live inspection (`POST /inspect`, `GET /inspect/<job>`, `src/serving/live_inspect.py`).** For a link that is not in the
  store the demo hands it to the same unattended pipeline the batches use (`scripts/run_batch.ps1 -Screenshots`): push
  file, 26-check isolation self-test, fetch in the analysis VM, screenshot, pull, strict validator. The page only ever
  sees a job with real steps and timings; when the job is done it calls `/predict` and the new evidence is used.
  One job at a time, at most 3 waiting. A new link takes about 2.5 to 3.5 minutes (measured: 157 s, 165 s; about 75 s of
  that is the isolation self-test, which is a safety gate and is not skipped). A link already in the store answers at
  once only with `--reuse-stored`; by default EVERY link, stored or not, is opened in the lab again. The result goes to `data/live/on_demand_links.csv`, which training and evaluation never read.
  If the lab VMs are off or the SSH key is not loaded, `/health` says `live_lab.ready = false`, `/inspect` answers 503
  and the page says plainly that the link was not opened. A failed or invalid lab result adds nothing; a lab result with
  a clickable embedded link is repaired on the host (`scripts/repair_embedded_links.py`) and must then pass the same
  validator. `--no-lab` switches live inspection off.
* `GET /evidence/<16 hex>.png`: the lab's screenshot of that page (only from `D:\LeJetData\clean\shots`, by id).
* Listens on 127.0.0.1 only; CORS only for the local Vite ports.
* Tests: `tests/test_lejet_service.py`, `tests/test_live_training.py`, `tests/test_live_set.py`.

## 5. Frontend (`frontend/`, a git submodule)

`ResultDetails.jsx` and `EvidenceDetailsPage.jsx` were hard-coded mock screens (a made-up BDO story: "3 days old
domain", fake CNN filters, fake timings). They now draw only what `/predict` returns: highlighted phrases, the link's
final destination (defanged), the page screenshot, page facts, the RF extractors' P and u, the CNN filters, the head's
34 inputs with gates and contributions. A part the backend did not send is not drawn.
The submodule has its own git history: the changes there must be committed and pushed in the frontend repository
(`kateivpf/ToolDef-Smishing-Detection`), then the pointer updated in this repo.
The Checker no longer plays a canned animation: it shows the lab job's real steps and times (or, without the lab, four
steps that only say what really happened). New frontend tests: `ResultView.live.test.jsx`, `Evidence.live.test.jsx`,
`Checker.lab.test.jsx`. The older tests in the submodule
(`ResultDetails.test.jsx`, `ScopeWarning.test.jsx`, `Checker.test.jsx`, ...) target components that the redesign
removed and were already failing before this change.

Run it:

```
python -m src.serving.api          # terminal 1
cd frontend; npm run dev           # terminal 2 → http://localhost:5050
```

## 6. Results (2026-10-13, `fresh_test`: 46 smishing + 90 legitimate links, 75 with a page reached; 3 seeds)

Raw files: `experiments/live_eval/<head>_seed<S>/live_eval.json`. Read every number as small-sample (intervals are in the JSON).

**Real-SMS headline (unchanged protocol).** Retrained `rf_vec`: real_all balanced accuracy 0.961 ± 0.006.
With live rows in the head (`rf_vec_live`): 0.962 ± 0.007. With a link, 0.932 ± 0.024 → 0.910 ± 0.016 (a small drop, inside
about one standard deviation; reported, not hidden). The retrained numbers differ slightly from the committed tables
(different machine/threads), so they are kept apart in `experiments/retrain_2026-10-13/` and the committed tables were restored.

**Link level (frozen extractors, never trained on this data).** On the 75 links whose page the lab reached, the `full`
extractor reaches balanced accuracy 0.88 to 0.94 (AUC about 0.985); the `url_only` extractor on the same links 0.64 to 0.73
(AUC 0.74 to 0.89). Page evidence is what separates them. Across all 136 links, routed (full where a page was reached, else
url_only) is 0.83 to 0.87.

**Message level (text that gives nothing away, e.g. "See details here <link>").**

| Model | balanced accuracy, 3 seeds | AUC |
|---|---|---|
| CNN alone | 0.48 to 0.54 | 0.52 to 0.56 |
| Final head, real SMS only (`rf_vec`), page shown | 0.52 to 0.54 | 0.56 to 0.59 |
| Head + live rows (`rf_vec_live`), page hidden (url_only) | 0.62 to 0.69 | 0.73 to 0.86 |
| Head + live rows (`rf_vec_live`), page shown | **0.71 to 0.79** | **0.83 to 0.97** |

Reading: the head trained only on real SMS ignores page evidence (page shown = page hidden), because the SMS data never
had any. Showing it lab-inspected pages in training makes it use them; with a page its verdict becomes useful even when the
text is ambiguous. Caveats: the legitimate links are generated well-known sites and the smishing links come from the team,
so the two classes differ in more than spelling; shorteners are 35% of the smishing links and 4% of the legitimate ones;
the text around each link is synthetic. This is a sanity check, not a headline result.

## 7. Known limits

* Live inspection needs both lab VMs running and the SSH key loaded in the Windows ssh-agent; it is one link at a time and
  takes minutes. Without the lab, a link that is not in the store is scored from its spelling only, and the page says so.
* The label of `dns_record` / `whois_registered_domain` follows the public dataset: 0 = normal, 1 = missing (fixed in the
  API's page facts and in the evidence page on 2026-10-13; they had been worded the other way round).
* `data/` and `models/` are git-ignored: rebuild with the commands in section 1, section 2 and `python -m src.features.model.rf_extractor`.
* The VM still runs the older build (embedded-link defang fix not deployed; needs the ops password).

## 8. Making live inspection faster (without dropping a check)

Measured per new link (three live runs, 157 to 165 s): link check and clock sync 15 s, send 8 s, isolation self-test 75 s,
open 19 s, screenshot 34 s, bring back and validate 11 s, page rank 3 s. The self-test is the largest part, and most of it is
waiting: every "must be blocked" probe waits its full 3 to 4 s timeout, one after another.

`infra/analysis/preflight.sh` is now the parallel version (the previous sequential one is kept as `preflight_sequential.sh`, the
reference and rollback): the same 26 checks with the same commands, timeouts, messages and token rule, with the independent
network probes run at the same time and the two resource-limit tests (19, 20) run alone afterwards. Expected: about 75 s down to
about 25 s, so a link in about 110 s instead of 165 s. `tests/test_preflight_parallel.py` compares it with
`preflight_sequential.sh` (same commands and messages, nothing dropped); the framework was run with stub checks (pass, a failing
check, a crashing check: no token on any failure). DEPLOYED to the VM on 2026-10-13 (VM test suite: 647 passed, 14 skipped). Measured on the first live link afterwards (wikipedia.org, whole job 138 s): self-test 47 s (was 75 s; I had expected about 25 s), still `PASS 18  FAIL 0  SKIP 8`, the same result as before. Per-link total 165 s -> 138 s. The rest is the VM job (open 18 s, screenshot 35 s), SSH round trips (about 1 s each, about 15 of them) and file transfer. The new bundle (same wheels as the working one,
current code and infra files) is `D:\LeJetData\bundle_new\lejet-bundle.tar.gz`, built with `D:\LeJetData\tools\repack_bundle.py`.
To deploy: `scripts\deploy_to_vm.ps1 -Bundle D:\LeJetData\bundle_new\lejet-bundle.tar.gz` (needs the `ops` password; this also
deploys the embedded-link defang fix), then run one batch and check `PASS 18 FAIL 0 SKIP 8`.
Reusing an old passing self-test for later links was considered and NOT done: every link gets a full self-test.

## 9. Fast path: background isolation gate + one-call live helper (2026-10-13)

Decision (by the person running the study): a live link does not repeat the 26-check self-test itself; the self-test runs in
the background and a link may only start while a passing result is recent. Everything else per link is unchanged (one link, fetched
and screenshotted inside the VM, host pulls, strict validators).

* `infra/analysis/systemd/lejet-gate.{service,timer}`: runs `preflight.sh` as root every 4 minutes after the previous run finished
  (first run 5 s after the timer starts), under an exclusive lock (`/run/lejet-gate.lock`). A pass writes `/run/lejet-preflight-ok`;
  a failure deletes it (so it blocks the old helpers too).
* `infra/analysis/helpers/lejet-sudo-live.sh` (`/usr/local/sbin/lejet-sudo-live <batch_id>`, the sixth passwordless helper): takes the
  lock shared (waits up to 3 minutes for a running self-test, and keeps a new one from starting), refuses unless the token is at most
  10 minutes old (exit 3), then calls the existing `install-urls`, `batch`, `shots`, `export` helpers (same limits as before), exactly
  one link per call.
* `scripts/run_live.ps1`: the host side. Clock sync only when a clock is more than 3 s off; one `scp` for the result files; same
  `validate_output.py` / `validate_shots.py`. Exit 4 = helper not installed (the API then uses `run_batch.ps1`); exit 3 from the helper
  (token stale, e.g. right after a VM restore) = the full self-test is run once and the call repeated.
* The page starts the lab job when a link has sat in the box for 1.5 s, and "Check" attaches to that job.
* DEPLOYED on 2026-10-13. Measured first live link on the fast path (mozilla.org): 70.6 s in total (was 138 s, before that 165 s): check 2.9 s, send 2.6 s, gate check 4.9 s (self-test had passed 49 s earlier), VM fetch 16.6 s, screenshot 34.0 s, bring back and validate 6.9 s, page rank 2.7 s. Result validated (data and screenshot), full-extractor verdict, screenshot shown. Not yet measured: a link that arrives while the background self-test is running (it waits for it, up to about a minute).
* Expected per link about 70 to 75 s (was 138 s); the real floor of this design is the VM job (fetch 6 to 9 s, screenshot with 11 s of
  fixed waits and browser launch). A resident worker with a warm browser and a concurrent fetch/screenshot is the next step (about 20 to 25 s).
* Deploy: `scripts\deploy_to_vm.ps1 -Bundle D:\LeJetData\bundle_new\lejet-bundle.tar.gz` (needs the `ops` password). Tests:
  `tests/test_live_helper_wiring.py`.

## 10. The resident live worker (built 2026-10-13)

Why: for one demo link the VM spent most of its time starting programs, not inspecting. Measured on the fast path (70 s per link): the
fetch job took 17 s for about 2 s of real stages (the rest is starting Python, libraries, the database and worker processes), and the
screenshot 24 to 35 s (a whole Chromium started and closed for one page, plus 3 s settle + 8 s extra wait on purpose + two pictures).
The analysis VM has ONE CPU and 5.9 GB RAM, which makes every start-up slower.

What: `src/lejet/live_worker.py`, run by `infra/analysis/systemd/lejet-live.service` as the unprivileged user `fetcher` with the limits
of the per-run units (private /tmp, read-only file system except /var/lib/lejet, 3.5 GB RAM without swap, 1 CPU, Chromium sandbox on,
recycled every hour, restarted if it dies). It keeps the Pipeline (parse workers, DNS/RDAP/CT clients and caches) and ONE shared
Chromium (fresh context per page, same code as the batch screenshots) loaded between links. Per request it does the same work as
`run` + `screenshots` + `export`, with the same functions (`Pipeline.inspect`, `ShotRunner.run`, `pack_tar`, `export_jsonl`).

How a link flows: host `run_live.ps1` -> `lejet-sudo-live` (root; checks the gate token, installs the one-line link) -> request file
`/var/lib/lejet/live/req/<batch>.json` (only the batch id) -> worker -> `done/<batch>.json` (`ok` or a short error code only) -> the helper
copies the result files to `ops` (plain files only, never symbolic links) -> host pulls and validates exactly as before.
If the worker is not ready (heartbeat older than 20 s, unit inactive: starting, recycled, crashed) the helper uses the per-process
path (install-urls, batch, shots, export), slower but with the same result. A worker that does not answer in 180 s is restarted.

Safety properties kept: the worker checks the isolation token itself (at most 10 minutes old) before every link; one link per request;
a batch id is never reused; requests are tiny id-checked JSON files (nothing chooses a path or an option); errors are short codes, never
page text; no new passwordless sudo command (the same six helpers). Checked on the VM before building: while Chromium runs, nothing
listens on a non-loopback address except sshd, so the self-test (check 15) is not disturbed by a resident browser.

Tuning (a one-line change, not made): the screenshot waits 3 s for the page to settle plus `LEJET_SHOT_EXTRA_WAIT_MS` (default 8000,
the same as `lejet-sudo-shots`) for late scripts; lowering the second makes links faster but can miss content of pages that load late.
The VM's single CPU is another lever (VirtualBox `modifyvm lejet-an --cpus 2`, VM powered off; a snapshot restore resets it).

Tests: `tests/test_live_worker.py` (real pipeline against the local fixture server, real storage/export/tar; the browser is a fake that
writes a valid PNG; also runs inside the VM) and the static checks in `tests/test_live_helper_wiring.py`.

Measured after the deploy (2026-10-13, openstreetmap.org/copyright, whole job 90 s, resident worker used): check 5.3 s, send 34.3 s
(of which about 30 s was the job WAITING for the background self-test that was running when it arrived: the exclusive/shared lock
doing its job), gate check and install 7.0 s, fetch by the worker 8.7 s (was 17 to 22 s), screenshot + export 25.3 s (was 34 s),
bring back and validate 6.8 s, page rank 2.9 s. Without a self-test in the way that is about 56 s (was 70 s). The screenshot is now the
largest part: its fixed waits (3 s settle + 8 s extra + early pictures) and Chromium rendering on the VM's single CPU.
A link that arrives while the self-test runs waits for it (observed about 30 s; the self-test takes about 47 s and runs about every 5 min,
so roughly one link in six). The first worker start after the deploy took 83 s (the install, the gate and the VM's own tests were using
the single CPU at the same time).
VM test suite after this deploy: 655 passed, 1 failed (`test_gzip_bomb_cannot_expand_past_the_cap`, a timing-sensitive test, while the
single CPU was busy starting the worker and running the self-test); it passed 3 of 3 times when re-run on the idle VM.

### VM CPU experiment and the early-start fix (2026-10-13)

* The analysis VM was set to 4 vCPUs (it had 1; the host has 16) to speed up Chromium, the parse workers and the fetch. One live link on 4 vCPUs
  with the resident worker: check 2.5 s, send 2.2 s, gate 2.4 s, fetch 1.2 s, screenshot 13.5 s, bring back 6.1 s, rank 2.4 s, about 30 s of
  work. About 6 minutes after that boot the guest FROZE (its journal ends abruptly at 01:26:06 UTC, the console showed only the self-test's
  normal memory-limit kill, the host saw one core spinning, ssh timed out during banner exchange and then "No route to host"). A hard reset
  brought it back with its disk and deployed software intact; the self-test passed again afterwards. The cause was not found (the log simply
  stops). It never froze at 1 vCPU (a dozen self-tests and many links that day), so the VM was put back to 1 vCPU, the configuration known to be
  stable. A snapshot restore also resets the CPU count (the snapshot says 1 vCPU). A 2-vCPU trial with a soak of several self-test cycles is possible
  but was not done.
* With the resident worker on 1 vCPU and no queue or self-test in the way, one link needs about 38 s of work (check 3.3, send 2.1, gate 3.3, fetch 6.9,
  screenshot 13.9, bring back 5.7, rank 3.2); it was 70 s before the worker and 165 s at the start of the day.
* Fix in the page: the early start (lab job while the person reads) fired again after a page reload with a saved message, and once more after
  each Check, which queued stray lab jobs (one a minute from a tab that reloaded itself). It now starts only for text typed or pasted in the current
  session, and never again for a link that a Check has already used (tests added).
