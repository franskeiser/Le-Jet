#!/usr/bin/env bash
# The ONE command the PC runs on the Azure VM (through sudo, allowed for the SSH user by 92-lejet-azure.sudoers).
# Installed as /usr/local/sbin/lejet-azure-live (root-owned, mode 755).
#
#   sudo -n /usr/local/sbin/lejet-azure-live <batch_id>
#
# It: checks the safety gate token, installs the pushed one-line URL file for user `fetcher`, runs
# `python -m lejet.azure_run <batch>` as `fetcher` in a locked-down systemd unit (hard memory cap so the SMS website
# that shares this VM can never be starved), and hands the result files to the SSH user for the PC to pull.
# Prints "== n ..." progress lines (the tool's frontend shows them). Exactly ONE link per call.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
[ $# -eq 1 ] || { echo "usage: lejet-azure-live <batch_id>" >&2; exit 2; }
BATCH_ID="$1"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9_-]{1,40}$ ]] || { echo "bad batch id (letters, digits, - and _ only)" >&2; exit 2; }

SSH_USER="${SUDO_USER:-}"
[ -n "$SSH_USER" ] && [ "$SSH_USER" != root ] || { echo "run this through sudo from the SSH user" >&2; exit 2; }
SRC="/home/$SSH_USER/incoming/urls.txt"
DEST="/home/$SSH_USER/export"
PY=/opt/lejet/venv/bin/python
TOKEN=/run/lejet-preflight-ok
IN=/var/lib/lejet/in/urls.txt
OUT=/var/lib/lejet/out
EXPORT="$OUT/export"
BROWSERS=/opt/lejet/browsers
LOCK=/run/lejet-azure-live.lock

exec 9>>"$LOCK"
flock -n 9 || { echo "another inspection is running on this VM; try again in a minute" >&2; exit 4; }

echo "== 3 isolation gate"
if ! "$PY" -I -m lejet.cli check-token --preflight-token "$TOKEN" --preflight-max-age 900 >/dev/null; then
  /usr/local/sbin/lejet-azure-gate >/dev/null 2>&1 || true            # try once to renew it right now
  "$PY" -I -m lejet.cli check-token --preflight-token "$TOKEN" --preflight-max-age 900 >/dev/null \
    || { echo "REFUSING TO START: the safety gate does not pass (run: sudo /usr/local/sbin/lejet-azure-gate)." >&2; exit 3; }
fi
[ -s "$BROWSERS/SHA256SUMS" ] || { echo "the browser is not installed: run setup_shot_vm.sh" >&2; exit 2; }

# the pushed file must be a plain file with exactly one non-empty line "url_id<TAB>url"
[ -f "$SRC" ] && [ ! -L "$SRC" ] || { echo "no pushed URL file" >&2; exit 2; }
[ "$(stat -c %s "$SRC")" -le 4096 ] || { echo "URL file too large" >&2; exit 2; }
[ "$(grep -c . "$SRC")" -eq 1 ] || { echo "live mode takes exactly one link per call" >&2; exit 2; }
grep -qP '^[0-9a-f]{16}\t\S+$' "$SRC" || { echo "URL file is not 'url_id<TAB>url'" >&2; exit 2; }
install -d -o fetcher -g fetcher -m 750 /var/lib/lejet/in "$OUT" "$EXPORT" /var/lib/lejet/work/home
install -o fetcher -g fetcher -m 640 "$SRC" "$IN"
sha256sum "$IN" | cut -d' ' -f1 | sed "s/\$/  urls.txt/"

VPN_ENV=()
if [ -f /etc/lejet/vpn.env ]; then      # the Mullvad tunnel is installed: tell the program (for the export manifest and DNS)
  safe='^[A-Za-z0-9_ .-]{1,40}$'
  while IFS='=' read -r k v; do
    case "$k" in
      LEJET_VPN_ACTIVE|LEJET_VPN_PROVIDER|LEJET_VPN_EXIT_COUNTRY) if [[ "$v" =~ $safe ]]; then VPN_ENV+=(-E "$k=$v"); fi ;;
    esac
  done < /etc/lejet/vpn.env
fi

echo "== 4 opening the link (fetch, then whole-page screenshots)"
systemd-run --quiet --wait --collect --pipe \
  --uid=fetcher --gid=fetcher \
  --unit="lejet-azure-${BATCH_ID}" \
  -p MemoryMax=1500M -p MemorySwapMax=0 -p MemoryHigh=1300M \
  -p CPUQuota=100% -p TasksMax=1024 -p LimitNOFILE=8192 \
  -p PrivateTmp=yes -p ProtectSystem=strict -p ProtectHome=yes -p InaccessiblePaths=/var/lib/lejetweb \
  -p ReadWritePaths=/var/lib/lejet \
  -p TimeoutStopSec=20 -p RuntimeMaxSec=900 \
  -E HOME=/var/lib/lejet/work/home -E TMPDIR=/var/lib/lejet/work \
  -E PLAYWRIGHT_BROWSERS_PATH="$BROWSERS" "${VPN_ENV[@]}" \
  "$PY" -I -m lejet.azure_run "$BATCH_ID"

echo "== 5 hand the result to $SSH_USER"
install -d -o "$SSH_USER" -g "$SSH_USER" -m 755 "$DEST"
for ext in .jsonl.gz .jsonl.gz.sha256 .manifest.json .shots.tar .shots.tar.sha256; do
  f="$EXPORT/${BATCH_ID}${ext}"
  if [ -L "$f" ]; then echo "refusing $f: it is a symbolic link" >&2; exit 5; fi
  if [ -f "$f" ]; then
    install -o "$SSH_USER" -g "$SSH_USER" -m 644 "$f" "$DEST/"
  else
    case "$ext" in .shots.tar|.shots.tar.sha256) ;; *) echo "missing result file ${BATCH_ID}${ext}" >&2; exit 5 ;; esac
  fi
done
# keep the VM's disk small: the working copies are not needed after the hand-over
rm -rf "$EXPORT/shots-${BATCH_ID}" "$OUT/${BATCH_ID}.sqlite"* 2>/dev/null || true
find "$EXPORT" -maxdepth 1 -type f -mmin +120 -delete 2>/dev/null || true
echo "== done"
