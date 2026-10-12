#!/usr/bin/env bash
# Passwordless-sudo helper (screenshots): take real screenshots of the pages of a finished batch.
# Installed as /usr/local/sbin/lejet-sudo-shots (root-owned, mode 755).
#
#   sudo lejet-sudo-shots <batch_id> [--limit N] [--desktop]
#
# By default the browser looks like an Android phone (smishing links are opened on phones and many pages
# hide from desktop browsers) and runs as the full Chromium in its newer headless mode without the automation
# switch (many pages cloak against the stripped-down headless shell and show nothing); --desktop switches to a
# desktop profile.
#
# This runs headless Chromium on pages the research is about (some are phishing pages), so the unit is
# locked down harder than the plain fetcher run:
#   * as the unprivileged user `fetcher`; own systemd unit; memory 3 GB (no swap), tasks 2048, 2 h ceiling;
#   * a private /tmp, a read-only file system except /var/lib/lejet, no access to /home;
#   * the Chromium SANDBOX stays on. NoNewPrivileges is deliberately NOT set for this unit: it would
#     stop the AppArmor profile that lets Chromium create its sandbox namespaces from applying;
#   * it refuses to start without a fresh preflight pass and without the installed browser.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
usage() { echo "usage: lejet-sudo-shots <batch_id> [--limit N] [--desktop]" >&2; exit 2; }

[ $# -ge 1 ] || usage
BATCH_ID="$1"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9_-]{1,40}$ ]] || { echo "bad batch id (letters, digits, - and _ only)" >&2; exit 2; }
shift
LIMIT_ARGS=()
MODE_ARGS=(--mobile --realistic --extra-wait-ms 8000)   # the default: an Android phone in the full browser, with time for late scripts
while [ $# -gt 0 ]; do
  case "$1" in
    --limit)   [[ "${2:-}" =~ ^[0-9]{1,6}$ ]] || usage; LIMIT_ARGS=(--limit "$2"); shift 2 ;;
    --desktop) MODE_ARGS=(); shift ;;
    *)         usage ;;
  esac
done

PY=/opt/lejet/venv/bin/python
TOKEN=/run/lejet-preflight-ok
OUT=/var/lib/lejet/out
BROWSERS=/opt/lejet/browsers

[ -f "$OUT/${BATCH_ID}.sqlite" ] || { echo "no database for batch $BATCH_ID: run the batch first" >&2; exit 2; }
[ -s "$BROWSERS/SHA256SUMS" ] || { echo "the browser is not installed: run install_browser.sh first" >&2; exit 2; }

if ! "$PY" -I -m lejet.cli check-token --preflight-token "$TOKEN" --preflight-max-age 21600; then
  echo "REFUSING TO START: run the preflight first (it must finish with 0 failures)." >&2
  exit 3
fi

install -d -o fetcher -g fetcher -m 700 /var/lib/lejet/work/home
install -d -o fetcher -g fetcher -m 750 "$OUT/export"

systemd-run --quiet --wait --collect --pipe \
  --uid=fetcher --gid=fetcher \
  --unit="lejet-shots-${BATCH_ID}" \
  -p MemoryMax=3G -p MemorySwapMax=0 \
  -p CPUQuota=100% \
  -p TasksMax=2048 \
  -p LimitNOFILE=8192 \
  -p PrivateTmp=yes \
  -p ProtectSystem=strict -p ProtectHome=yes \
  -p ReadWritePaths=/var/lib/lejet \
  -p TimeoutStopSec=30 \
  -p RuntimeMaxSec=7200 \
  -E HOME=/var/lib/lejet/work/home -E TMPDIR=/var/lib/lejet/work \
  -E PLAYWRIGHT_BROWSERS_PATH="$BROWSERS" \
  "$PY" -I -m lejet.cli screenshots \
    --db "$OUT/${BATCH_ID}.sqlite" \
    --out-dir "$OUT/export" \
    --batch-id "$BATCH_ID" \
    "${MODE_ARGS[@]}" "${LIMIT_ARGS[@]}"
