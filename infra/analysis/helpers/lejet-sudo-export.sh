#!/usr/bin/env bash
# Passwordless-sudo helper 4/4: export one finished batch and hand exactly three files to `ops`.
# Installed as /usr/local/sbin/lejet-sudo-export (root-owned, mode 755).
#
#   sudo lejet-sudo-export <batch_id>
#
# Writes <batch>.jsonl.gz, <batch>.jsonl.gz.sha256 and <batch>.manifest.json into /home/ops/export
# (owned by ops), which is where the host pulls them from. Optional provenance for the manifest
# (VPN provider, exit country, VirtualBox version) is read from the root-owned file
# /etc/lejet/provenance.env, never from the caller's environment.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
[ $# -eq 1 ] || { echo "usage: lejet-sudo-export <batch_id>" >&2; exit 2; }
BATCH_ID="$1"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9_-]{1,40}$ ]] || { echo "bad batch id" >&2; exit 2; }

OUT=/var/lib/lejet/out
DEST=/home/ops/export
[ -f "$OUT/${BATCH_ID}.sqlite" ] || { echo "no database for batch $BATCH_ID" >&2; exit 2; }

LEJET_VPN_PROVIDER=""; LEJET_VPN_EXIT_COUNTRY=""; LEJET_VBOX_VERSION=""; LEJET_GIT_COMMIT=""
# shellcheck disable=SC1091
[ -f /etc/lejet/provenance.env ] && . /etc/lejet/provenance.env

runuser -u fetcher -- env \
  LEJET_VPN_PROVIDER="$LEJET_VPN_PROVIDER" \
  LEJET_VPN_EXIT_COUNTRY="$LEJET_VPN_EXIT_COUNTRY" \
  LEJET_VBOX_VERSION="$LEJET_VBOX_VERSION" \
  LEJET_GIT_COMMIT="$LEJET_GIT_COMMIT" \
  /opt/lejet/venv/bin/python -I -m lejet.cli export \
    --db "$OUT/${BATCH_ID}.sqlite" --out-dir "$OUT/export" --batch-id "$BATCH_ID"

# /var/lib/lejet belongs to `fetcher` (mode 750), so `ops` cannot read it: hand over exactly the
# three files the host will pull.
install -d -o ops -g ops -m 755 "$DEST"
for ext in .jsonl.gz .jsonl.gz.sha256 .manifest.json; do
  install -o ops -g ops -m 644 "$OUT/export/${BATCH_ID}${ext}" "$DEST/"
done
# the screenshot archive exists only if lejet-sudo-shots was run for this batch
for ext in .shots.tar .shots.tar.sha256; do
  if [ -f "$OUT/export/${BATCH_ID}${ext}" ]; then
    install -o ops -g ops -m 644 "$OUT/export/${BATCH_ID}${ext}" "$DEST/"
  fi
done
echo
ls -l "$DEST/${BATCH_ID}".*
