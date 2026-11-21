#!/usr/bin/env bash
# The command the PC runs on the Azure VM to get `page_rank` (Open PageRank) for a list of DOMAINS.
# Installed as /usr/local/sbin/lejet-azure-pagerank (root-owned, mode 755); allowed for the SSH user by 92-lejet-azure.sudoers.
#
#   sudo -n /usr/local/sbin/lejet-azure-pagerank <batch_id>
#
# Why a separate helper: the API key must never be readable by `fetcher`, the account that opens hostile pages. Here the look-up
# runs as a throw-away user in its own locked-down unit, and it only ever reads a plain list of domain names that the PC took from
# a batch the strict validator had already accepted (so no byte produced by a hostile page is read here, not even by root).
#
# Input : /home/<ssh user>/incoming/domains.txt   one domain per line (the helper re-checks every line), at most 2000
# Key   : /etc/lejet/openpagerank.key  (written by infra/azure/set_rank_key.sh; root-only), or, if that does not exist, the website's
#         /etc/lejet-web/openpagerank.key. NO key = nothing is looked up, this prints "NO KEY" and exits 0 (page_rank stays -1).
# Output: /home/<ssh user>/export/<batch_id>.page_rank.json  {"ranks": {domain: 0..10}, "asked": N, "failed_batches": K}
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
[ $# -eq 1 ] || { echo "usage: lejet-azure-pagerank <batch_id>" >&2; exit 2; }
BATCH_ID="$1"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9_-]{1,40}$ ]] || { echo "bad batch id (letters, digits, - and _ only)" >&2; exit 2; }

SSH_USER="${SUDO_USER:-}"
[ -n "$SSH_USER" ] && [ "$SSH_USER" != root ] || { echo "run this through sudo from the SSH user" >&2; exit 2; }
SRC="/home/$SSH_USER/incoming/domains.txt"
DEST="/home/$SSH_USER/export"
PY=/opt/lejet/venv/bin/python
MAX_DOMAINS=2000

KEYFILE=""
for k in /etc/lejet/openpagerank.key /etc/lejet-web/openpagerank.key; do
  if [ -f "$k" ] && [ ! -L "$k" ] && [ -s "$k" ]; then KEYFILE="$k"; break; fi
done
if [ -z "$KEYFILE" ]; then
  echo "NO KEY: no Open PageRank key is installed on this VM (run: sudo bash ~/lejet-src/infra/azure/set_rank_key.sh). page_rank stays -1."
  exit 0
fi

[ -f "$SRC" ] && [ ! -L "$SRC" ] || { echo "no pushed domain file" >&2; exit 2; }
[ "$(stat -c %s "$SRC")" -le 200000 ] || { echo "domain file too large" >&2; exit 2; }
N="$(grep -c . "$SRC" || true)"
[ "$N" -ge 1 ] && [ "$N" -le "$MAX_DOMAINS" ] || { echo "this helper takes 1 to $MAX_DOMAINS domains per call (found $N)" >&2; exit 2; }
# every non-empty line must be a plain lower-case domain name (letters, digits, dots, hyphens)
[ "$(grep -cP '^[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?\.[a-z0-9-]{2,63}$' "$SRC")" -eq "$N" ] \
  || { echo "domain file has a line that is not a plain domain name" >&2; exit 2; }
sha256sum "$SRC" | cut -d' ' -f1 | sed "s/\$/  domains.txt/"

TMP="$(mktemp /run/lejet-pagerank.XXXXXX)"
trap 'rm -f "$TMP"' EXIT
echo "== looking up $N domains (own unit, no page content)"
set +e
systemd-run --quiet --wait --collect --pipe \
  --unit="lejet-azrank-${BATCH_ID}" \
  -p DynamicUser=yes -p NoNewPrivileges=yes \
  -p LoadCredential=key:"$KEYFILE" -p LoadCredential=domains:"$SRC" \
  -p MemoryMax=200M -p MemorySwapMax=0 -p TasksMax=64 -p RuntimeMaxSec=900 \
  -p PrivateTmp=yes -p ProtectSystem=strict -p ProtectHome=yes -p PrivateDevices=yes \
  -p InaccessiblePaths=/var/lib/lejetweb -p InaccessiblePaths=/var/lib/lejet -p InaccessiblePaths=/etc/lejet-web \
  -p RestrictAddressFamilies="AF_INET AF_INET6 AF_UNIX" \
  "$PY" -I -m lejet.azure_page_rank > "$TMP"
RC=$?
set -e
# rc 3 = some batches failed (the answer file then lacks those domains: they stay -1 on the PC); anything else non-zero = no answer
if [ "$RC" -ne 0 ] && [ "$RC" -ne 3 ]; then echo "the look-up unit failed (exit $RC); page_rank stays -1" >&2; exit 5; fi
[ "$(stat -c %s "$TMP")" -le 400000 ] || { echo "answer unexpectedly large" >&2; exit 5; }
"$PY" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d.get("ranks"),dict)' "$TMP" \
  || { echo "answer is not the expected JSON" >&2; exit 5; }
install -d -o "$SSH_USER" -g "$SSH_USER" -m 755 "$DEST"
install -o "$SSH_USER" -g "$SSH_USER" -m 644 "$TMP" "$DEST/${BATCH_ID}.page_rank.json"
echo "== done (exit $RC)"
exit "$RC"
