#!/usr/bin/env bash
# The command the PC runs on the Azure VM to get the FEATURES of MANY links at once (the dataset; no browser, no screenshots).
# Installed as /usr/local/sbin/lejet-azure-features (root-owned, mode 755); allowed for the SSH user by 92-lejet-azure.sudoers.
#
#   sudo -n /usr/local/sbin/lejet-azure-features <batch_id>
#
# Same barriers as lejet-azure-live (isolation gate token, locked-down unit for user `fetcher`, memory cap, no access to the
# website's data, one run at a time on the VM), but it takes 1 to 400 lines "url_id<TAB>url" and runs `lejet.azure_features`
# instead of the browser. The live tool's mode (one link, pictures) is untouched.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
[ $# -eq 1 ] || { echo "usage: lejet-azure-features <batch_id>" >&2; exit 2; }
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
LOCK=/run/lejet-azure-live.lock          # the SAME lock as the live mode: one inspection at a time on this VM
MAX_LINKS=400

exec 9>>"$LOCK"
flock -n 9 || { echo "another inspection is running on this VM; try again in a minute" >&2; exit 4; }

echo "== 3 isolation gate"
if ! "$PY" -I -m lejet.cli check-token --preflight-token "$TOKEN" --preflight-max-age 900 >/dev/null; then
  /usr/local/sbin/lejet-azure-gate >/dev/null 2>&1 || true            # try once to renew it right now
  "$PY" -I -m lejet.cli check-token --preflight-token "$TOKEN" --preflight-max-age 900 >/dev/null \
    || { echo "REFUSING TO START: the safety gate does not pass (run: sudo /usr/local/sbin/lejet-azure-gate)." >&2; exit 3; }
fi

[ -f "$SRC" ] && [ ! -L "$SRC" ] || { echo "no pushed URL file" >&2; exit 2; }
[ "$(stat -c %s "$SRC")" -le 400000 ] || { echo "URL file too large" >&2; exit 2; }
N="$(grep -c . "$SRC" || true)"
[ "$N" -ge 1 ] && [ "$N" -le "$MAX_LINKS" ] || { echo "this mode takes 1 to $MAX_LINKS links per call (found $N)" >&2; exit 2; }
# every non-empty line must be 'url_id<TAB>url'; the program re-checks that url_id = sha256(url)[:16]
[ "$(grep -cP '^[0-9a-f]{16}\t\S+$' "$SRC")" -eq "$N" ] || { echo "URL file is not 'url_id<TAB>url' on every line" >&2; exit 2; }
install -d -o fetcher -g fetcher -m 750 /var/lib/lejet/in "$OUT" "$EXPORT" /var/lib/lejet/work/home
install -o fetcher -g fetcher -m 640 "$SRC" "$IN"
sha256sum "$IN" | cut -d' ' -f1 | sed "s/\$/  urls.txt/"

VPN_ENV=()
if [ -f /etc/lejet/vpn.env ]; then      # the Mullvad tunnel is installed: tell the program (for the export manifest)
  safe='^[A-Za-z0-9_ .-]{1,40}$'
  while IFS='=' read -r k v; do
    case "$k" in
      LEJET_VPN_ACTIVE|LEJET_VPN_PROVIDER|LEJET_VPN_EXIT_COUNTRY) if [[ "$v" =~ $safe ]]; then VPN_ENV+=(-E "$k=$v"); fi ;;
    esac
  done < /etc/lejet/vpn.env
fi

echo "== 4 fetching $N links (features only)"
systemd-run --quiet --wait --collect --pipe \
  --uid=fetcher --gid=fetcher \
  --unit="lejet-azfeat-${BATCH_ID}" \
  -p MemoryMax=1500M -p MemorySwapMax=0 -p MemoryHigh=1300M \
  -p CPUQuota=100% -p TasksMax=1024 -p LimitNOFILE=8192 \
  -p PrivateTmp=yes -p ProtectSystem=strict -p ProtectHome=yes -p InaccessiblePaths=/var/lib/lejetweb \
  -p ReadWritePaths=/var/lib/lejet \
  -p TimeoutStopSec=20 -p RuntimeMaxSec=1800 \
  -E HOME=/var/lib/lejet/work/home -E TMPDIR=/var/lib/lejet/work "${VPN_ENV[@]}" \
  "$PY" -I -m lejet.azure_features "$BATCH_ID"

echo "== 5 hand the result to $SSH_USER"
install -d -o "$SSH_USER" -g "$SSH_USER" -m 755 "$DEST"
for ext in .jsonl.gz .jsonl.gz.sha256 .manifest.json; do
  f="$EXPORT/${BATCH_ID}${ext}"
  if [ -L "$f" ]; then echo "refusing $f: it is a symbolic link" >&2; exit 5; fi
  [ -f "$f" ] || { echo "missing result file ${BATCH_ID}${ext}" >&2; exit 5; }
  install -o "$SSH_USER" -g "$SSH_USER" -m 644 "$f" "$DEST/"
done
rm -f "$OUT/${BATCH_ID}.sqlite"* 2>/dev/null || true
find "$EXPORT" -maxdepth 1 -type f -mmin +120 -delete 2>/dev/null || true
echo "== done"
