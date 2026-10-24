#!/usr/bin/env bash
# Installs the Open PageRank API key for the website, so page_rank is filled in automatically.
#   sudo bash /opt/lejet-web/webapp/deploy/set-rank-key.sh
# The key is typed at a hidden prompt (it is not shown, not kept in your shell history, not passed on a command line).
# It goes into /etc/lejet-web/openpagerank.key: owned by root, readable only by root and the website's own user.
# To remove it again:  sudo rm /etc/lejet-web/openpagerank.key && sudo systemctl restart lejet-web
set -euo pipefail

[ "$(id -u)" = 0 ] || { echo "Run this with sudo."; exit 1; }
id lejetweb >/dev/null 2>&1 || { echo "The website is not installed yet: run setup.sh first."; exit 1; }

read -rsp "Open PageRank API key (typing is hidden): " KEY
echo
if ! [[ "$KEY" =~ ^[A-Za-z0-9_-]{8,128}$ ]]; then
    echo "That does not look like a key (letters, digits, - and _ only, 8 to 128 characters). Nothing was changed."
    exit 1
fi

install -d -o root -g lejetweb -m 750 /etc/lejet-web
( umask 077; printf '%s' "$KEY" > /etc/lejet-web/openpagerank.key.new )
unset KEY
chown root:lejetweb /etc/lejet-web/openpagerank.key.new
chmod 640 /etc/lejet-web/openpagerank.key.new
mv /etc/lejet-web/openpagerank.key.new /etc/lejet-web/openpagerank.key

systemctl restart lejet-web
sleep 4
echo "Website restarted. Its health check now says:"
curl -s http://127.0.0.1:8000/api/health
echo
echo '"page_rank":true means the key was accepted by the site (the first message you add shows whether Open PageRank answers).'
