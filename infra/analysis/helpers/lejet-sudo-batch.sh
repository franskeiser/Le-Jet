#!/usr/bin/env bash
# Passwordless-sudo helper 3/4: run ONE batch inside hard resource limits, as the unprivileged user
# `fetcher` (WEB_INSPECTION_PLAN.md sections 11.10, 17.2).
# Installed as /usr/local/sbin/lejet-sudo-batch (root-owned, mode 755).
#
#   sudo lejet-sudo-batch <batch_id> [--seed N]
#
# Only those arguments are accepted; everything else is fixed here. What protects you:
#   * it refuses to start unless the preflight passed recently (fresh token, 6 hours);
#   * the job runs as `fetcher` (no shell, no sudo) in a systemd unit with memory, CPU, task and file
#     limits, a private /tmp, a read-only file system except /var/lib/lejet, and a hard 4-hour
#     ceiling, so a hung or hostile run dies on its own;
#   * `python -I` ignores environment variables and the current directory when importing.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
usage() { echo "usage: lejet-sudo-batch <batch_id> [--seed N]" >&2; exit 2; }

[ $# -ge 1 ] || usage
BATCH_ID="$1"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9_-]{1,40}$ ]] || { echo "bad batch id (letters, digits, - and _ only)" >&2; exit 2; }
SEED_ARGS=()
if [ $# -eq 3 ] && [ "$2" = "--seed" ] && [[ "$3" =~ ^[0-9]{1,9}$ ]]; then
  SEED_ARGS=(--seed "$3")
elif [ $# -ne 1 ]; then
  usage
fi

PY=/opt/lejet/venv/bin/python
TOKEN=/run/lejet-preflight-ok
IN=/var/lib/lejet/in/urls.txt
OUT=/var/lib/lejet/out

[ -f "$IN" ] || { echo "no input file: push urls.txt first ($IN)" >&2; exit 2; }

# 1. refuse to start without a fresh preflight pass (plan 16, Level 2)
if ! "$PY" -I -m lejet.cli check-token --preflight-token "$TOKEN" --preflight-max-age 21600; then
  echo "REFUSING TO START: run the preflight first (it must finish with 0 failures)." >&2
  exit 3
fi

# 2. run the batch under systemd limits
RDAP_ARGS=()
[ -f /opt/lejet/rdap-dns.json ] && RDAP_ARGS+=(--rdap-bootstrap /opt/lejet/rdap-dns.json)

systemd-run --quiet --wait --collect --pipe \
  --uid=fetcher --gid=fetcher \
  --unit="lejet-${BATCH_ID}" \
  -p MemoryMax=4G -p MemorySwapMax=0 \
  -p CPUQuota=300% \
  -p TasksMax=1024 \
  -p LimitNOFILE=4096 \
  -p PrivateTmp=yes -p NoNewPrivileges=yes \
  -p ProtectSystem=strict -p ProtectHome=yes \
  -p ReadWritePaths=/var/lib/lejet \
  -p TimeoutStopSec=30 \
  -p RuntimeMaxSec=14400 \
  -E TMPDIR=/var/lib/lejet/work \
  "$PY" -I -m lejet.cli run \
    --input "$IN" \
    --db "$OUT/${BATCH_ID}.sqlite" \
    --batch-id "$BATCH_ID" \
    --snapshot-name "${LEJET_SNAPSHOT_NAME:-an-configured-clean}" \
    --rdap-cache "$OUT/rdap_cache.json" \
    --ct --ct-cache "$OUT/ct_cache.json" \
    "${RDAP_ARGS[@]}" \
    "${SEED_ARGS[@]}"
