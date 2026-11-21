#!/usr/bin/env bash
# Installs the Open PageRank API key for the Azure LAB's page_rank look-up (the helper lejet-azure-pagerank).
#   sudo bash ~/lejet-src/infra/azure/set_rank_key.sh
# The key is typed at a hidden prompt (not shown, not in your shell history, not on a command line). It goes into
# /etc/lejet/openpagerank.key: owned by root, mode 600 -- NOT readable by `fetcher` (the account that opens hostile pages),
# by the website's account, or by the SSH user. Only the throw-away unit the helper starts gets a copy, through systemd.
# Remove it again:  sudo rm /etc/lejet/openpagerank.key
set -euo pipefail

[ "$(id -u)" = 0 ] || { echo "Run this with sudo."; exit 1; }

read -rsp "Open PageRank API key (typing is hidden): " KEY
echo
if ! [[ "$KEY" =~ ^[A-Za-z0-9_-]{8,128}$ ]]; then
  echo "That does not look like a key (letters, digits, - and _ only, 8 to 128 characters). Nothing was changed."
  exit 1
fi
[ -d /etc/lejet ] || install -d -o root -g root -m 755 /etc/lejet     # (an existing folder keeps its permissions)
( umask 077; printf '%s' "$KEY" > /etc/lejet/openpagerank.key.new )
unset KEY
chown root:root /etc/lejet/openpagerank.key.new
chmod 600 /etc/lejet/openpagerank.key.new
mv /etc/lejet/openpagerank.key.new /etc/lejet/openpagerank.key
echo "Key installed. The lab now looks up page_rank for every batch (scripts/azure_page_rank.py)."
