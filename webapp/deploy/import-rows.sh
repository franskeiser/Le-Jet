#!/usr/bin/env bash
# Adds rows that already have their features to the website's shared CSV, without inspecting anything.
#   1. from the PC:  scp web_import.csv import_rows.py import-rows.sh lejet-azure:incoming/
#   2. on the VM:    sudo bash ~/incoming/import-rows.sh            (adds the rows)
#                    sudo bash ~/incoming/import-rows.sh update     (refreshes the inspection result of rows that exist)
# What it does: backs up the CSV, stops the site for a few seconds (so the running site and this tool never write at once),
# appends the rows with webapp/deploy/import_rows.py as the site's own user, starts the site again and prints its health check.
# If anything fails the backup is put back and the site is started again. The backup stays next to the CSV:
#   /var/lib/lejetweb/web_submissions.csv.before-import.<time>
set -euo pipefail

[ "$(id -u)" = 0 ] || { echo "Run this with sudo."; exit 1; }
SSH_USER="${SUDO_USER:-}"
[ -n "$SSH_USER" ] && [ "$SSH_USER" != root ] || { echo "run this through sudo from the SSH user"; exit 1; }
INC="/home/$SSH_USER/incoming"
CSV="$INC/web_import.csv"; PYF="$INC/import_rows.py"
DATA=/var/lib/lejetweb/web_submissions.csv
WORK=/var/lib/lejetweb/_import
PY=/opt/lejet-web/venv/bin/python
MODE_ARGS=()
[ "${1:-}" = "update" ] && MODE_ARGS=(--update)

for f in "$CSV" "$PYF"; do [ -f "$f" ] && [ ! -L "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$DATA" ] || { echo "the website's CSV was not found: $DATA"; exit 1; }
[ "$(head -c 10 "$CSV" | grep -c message_id)" -ge 1 ] || { echo "web_import.csv does not start with the message_id header"; exit 1; }

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$DATA.before-import.$STAMP"
ROWS_BEFORE="$(( $(wc -l < "$DATA") ))"
cp -p "$DATA" "$BACKUP"
echo "backup: $BACKUP"

install -d -o lejetweb -g lejetweb -m 700 "$WORK"
install -o lejetweb -g lejetweb -m 600 "$CSV" "$WORK/rows.csv"
install -o lejetweb -g lejetweb -m 600 "$PYF" "$WORK/import_rows.py"

restore_and_start() {
  echo "FAILED: putting the backup back"
  cp -p "$BACKUP" "$DATA" || true
  systemctl start lejet-web || true
}
systemctl stop lejet-web
if ! runuser -u lejetweb -- env LEJET_DATASET="$DATA" "$PY" -I "$WORK/import_rows.py" "${MODE_ARGS[@]}" "$WORK/rows.csv"; then
  restore_and_start
  rm -rf "$WORK"
  exit 1
fi
rm -rf "$WORK"
systemctl start lejet-web
sleep 6
echo "the website now says:"
curl -s http://127.0.0.1:8000/api/health || echo "(no answer yet; check: systemctl status lejet-web)"
echo
echo "done. To undo everything this run added:  sudo systemctl stop lejet-web && sudo cp -p $BACKUP $DATA && sudo systemctl start lejet-web"
