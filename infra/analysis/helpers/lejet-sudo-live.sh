#!/usr/bin/env bash
# Passwordless-sudo helper 6: ONE call for a live demo link: check the isolation gate, install the pushed link, have it fetched
# and screenshotted, and hand the result files to `ops`.
# Installed as /usr/local/sbin/lejet-sudo-live (root-owned, mode 755).
#
#   sudo lejet-sudo-live <batch_id>
#
# How the link is processed:
#   * FAST path: the resident worker (lejet-live.service, user `fetcher`) is running, so its pipeline and browser are already loaded.
#     This helper drops a tiny request file for it, waits for its answer, and copies the result files to `ops` itself.
#   * FALLBACK path: the worker is not ready (starting, recycled, crashed) -> the existing per-process helpers do the same
#     work (install-urls, batch, shots, export), slower, with their own limits.
# Either way:
#   * the 26-check isolation self-test is not repeated in this call. It runs in the BACKGROUND every ~5 minutes
#     (lejet-gate.timer, as root) and this helper refuses to start unless a passing result is at most 10 minutes old. A failed
#     background run deletes the token (preflight.sh does that), so a failure blocks every live run until the next pass;
#   * the self-test and a live run never overlap: the self-test holds an exclusive lock, this helper a shared one, so a run
#     that arrives during a self-test waits for it (at most about a minute) and the self-test waits for a running link;
#   * exactly ONE link per call (the installed file must have one line); only the batch id is taken from the caller;
#   * files that the worker (user `fetcher`) produced are copied only if they are plain files, never symbolic links.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
[ $# -eq 1 ] || { echo "usage: lejet-sudo-live <batch_id>" >&2; exit 2; }
BATCH_ID="$1"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9_-]{1,40}$ ]] || { echo "bad batch id (letters, digits, - and _ only)" >&2; exit 2; }

PY=/opt/lejet/venv/bin/python
TOKEN=/run/lejet-preflight-ok
LOCK=/run/lejet-gate.lock
IN=/var/lib/lejet/in/urls.txt
SPOOL=/var/lib/lejet/live
EXPORT=/var/lib/lejet/out/export
DEST=/home/ops/export
MAX_AGE=600
WAIT_S=180

# the gate lock (shared): waits while the background self-test runs, and keeps it from starting while we work
exec 9>>"$LOCK"
flock -s -w 180 9 || { echo "REFUSING TO START: the isolation self-test is still running; try again in a minute." >&2; exit 3; }

if ! "$PY" -I -m lejet.cli check-token --preflight-token "$TOKEN" --preflight-max-age "$MAX_AGE" >/dev/null; then
  echo "REFUSING TO START: no passing isolation self-test from the last $((MAX_AGE / 60)) minutes (the background gate renews it every ~5 minutes)." >&2
  exit 3
fi
age=$(( $(date -u +%s) - $(cat "$TOKEN") ))
echo "== 3 isolation gate: the 26-check self-test passed ${age} s ago"

/usr/local/sbin/lejet-sudo-install-urls
[ "$(wc -l < "$IN")" -eq 1 ] || { echo "live mode takes exactly one link per call" >&2; exit 2; }

# is the resident worker alive? (a heartbeat file younger than 20 s, and the unit active)
worker_ready() {
  [ -f "$SPOOL/ready" ] && [ ! -L "$SPOOL/ready" ] \
    && [ $(( $(date +%s) - $(stat -c %Y "$SPOOL/ready") )) -le 20 ] \
    && systemctl is-active --quiet lejet-live.service
}

# copy the result files to ops; only plain files, never symbolic links; the three result files are required
hand_over() {
  install -d -o ops -g ops -m 755 "$DEST"
  local ext f
  for ext in .jsonl.gz .jsonl.gz.sha256 .manifest.json .shots.tar .shots.tar.sha256; do
    f="$EXPORT/${BATCH_ID}${ext}"
    if [ -L "$f" ]; then echo "refusing $f: it is a symbolic link" >&2; exit 5; fi
    if [ -f "$f" ]; then
      install -o ops -g ops -m 644 "$f" "$DEST/"
    else
      case "$ext" in .shots.tar|.shots.tar.sha256) ;; *) echo "missing result file ${BATCH_ID}${ext}" >&2; exit 5 ;; esac
    fi
  done
}

if worker_ready; then
  echo "== 4 fetching the link (resident worker)"
  rm -f "$SPOOL/done/$BATCH_ID.json" "$SPOOL/state/$BATCH_ID"
  tmp="$(mktemp)"
  printf '{"batch_id": "%s"}\n' "$BATCH_ID" > "$tmp"
  install -o fetcher -g fetcher -m 640 "$tmp" "$SPOOL/req/$BATCH_ID.json"
  rm -f "$tmp"
  shown=""
  waited=0
  while [ ! -f "$SPOOL/done/$BATCH_ID.json" ]; do
    if [ "$waited" -ge $((WAIT_S * 5)) ]; then
      echo "the live worker did not answer in ${WAIT_S} s; restarting it" >&2
      systemctl restart lejet-live.service || true
      exit 5
    fi
    stage="$(head -c 16 "$SPOOL/state/$BATCH_ID" 2>/dev/null | tr -d '\n' || true)"
    if [ "$stage" != "$shown" ]; then
      case "$stage" in
        shot)   echo "== 4b screenshot" ;;
        export) echo "== 5 export" ;;
      esac
      shown="$stage"
    fi
    sleep 0.2
    waited=$((waited + 1))
  done
  answer="$(head -c 512 "$SPOOL/done/$BATCH_ID.json")"
  if ! printf '%s' "$answer" | grep -q '"ok": true'; then
    code="$(printf '%s' "$answer" | grep -o '"error": "[a-z_]*"' | head -n1 || true)"
    echo "the live worker could not process the link: ${code:-unknown error}" >&2
    exit 5
  fi
  hand_over
else
  echo "== 4 fetching the link (per-process path: the resident worker is not ready)"
  /usr/local/sbin/lejet-sudo-batch "$BATCH_ID"
  echo "== 4b screenshot"
  /usr/local/sbin/lejet-sudo-shots "$BATCH_ID"
  echo "== 5 export"
  /usr/local/sbin/lejet-sudo-export "$BATCH_ID"
fi
echo "== done"
