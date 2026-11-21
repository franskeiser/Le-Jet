#!/usr/bin/env bash
# Run ONCE on the Azure VM (Ubuntu 24.04) as root, from inside the unpacked bundle folder:
#     sudo bash setup.sh <your-dns-name>      e.g.  sudo bash setup.sh lejet-demo.southeastasia.cloudapp.azure.com
# Re-running it is safe (it updates the code and restarts the service).
set -euo pipefail

HOST="${1:?usage: sudo bash setup.sh <public dns name of this VM>}"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"          # the bundle root (contains src/ and webapp/)
APP=/opt/lejet-web

echo "== packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip git curl nftables debian-keyring debian-archive-keyring apt-transport-https gpg
apt-get install -y libgl1 libglib2.0-0t64          # libraries OpenCV (inside the screenshot reader) needs on a server
if ! command -v caddy >/dev/null; then
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -y && apt-get install -y caddy
fi

echo "== user and folders"
id lejetweb >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin lejetweb
install -d -o lejetweb -g lejetweb -m 750 /var/lib/lejetweb
install -d -m 755 "$APP"

echo "== code"
rm -rf "$APP/src" "$APP/webapp"
cp -r "$SRC/src" "$SRC/webapp" "$APP/"
chown -R root:root "$APP"
chmod -R a+rX "$APP"

echo "== python environment"
[ -d "$APP/venv" ] || python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install --quiet --upgrade pip
"$APP/venv/bin/pip" install --quiet -r "$APP/webapp/requirements-web.txt"

echo "== screenshot reader self-test"
if "$APP/venv/bin/python" -c "import cv2; from rapidocr_onnxruntime import RapidOCR; RapidOCR()" 2>/tmp/lejet-ocr-test.log; then
    echo "ok: the site can read screenshots"
else
    echo "WARNING: screenshot reading will be OFF (pasting text still works). Last lines of the error:"
    tail -5 /tmp/lejet-ocr-test.log
fi

echo "== RDAP bootstrap (which registry answers for which TLD)"
curl -fsS https://data.iana.org/rdap/dns.json -o "$APP/rdap-dns.json"

echo "== access code (none by default: anyone who has the address can use the site)"
if [ ! -f /etc/lejet-web.env ]; then
    cat > /etc/lejet-web.env <<'EOF'
# Optional settings of the website. Remove the # of the next line (and pick your own word) to make the site ask for a code,
# then run:  sudo systemctl restart lejet-web
#LEJET_ACCESS_CODE=choose-a-code
EOF
fi
chmod 600 /etc/lejet-web.env

echo "== page_rank key folder (the key itself is added with set-rank-key.sh, never by this script)"
install -d -o root -g lejetweb -m 750 /etc/lejet-web
if [ -s /etc/lejet-web/openpagerank.key ]; then
    echo "an Open PageRank key is already installed"
else
    echo "no Open PageRank key yet: page_rank stays -1 until you run  sudo bash $APP/webapp/deploy/set-rank-key.sh"
fi

echo "== daily backup copy of the dataset (kept 30 days, in /var/lib/lejetweb/backups)"
cat > /etc/cron.daily/lejet-backup <<'EOF'
#!/bin/sh
SRC=/var/lib/lejetweb/web_submissions.csv
DIR=/var/lib/lejetweb/backups
[ -s "$SRC" ] || exit 0
install -d -m 750 "$DIR"
cp "$SRC" "$DIR/web_submissions_$(date +%F).csv"
find "$DIR" -name 'web_submissions_*.csv' -mtime +30 -delete
EOF
chmod 755 /etc/cron.daily/lejet-backup

echo "== firewall (outer barrier)"
install -m 644 "$APP/webapp/deploy/egress.nft" /etc/lejet-egress.nft
grep -q 'lejet-egress.nft' /etc/nftables.conf || echo 'include "/etc/lejet-egress.nft"' >> /etc/nftables.conf
systemctl enable --now nftables
nft -f /etc/lejet-egress.nft 2>/dev/null || { nft delete table inet lejet_egress 2>/dev/null; nft -f /etc/lejet-egress.nft; }

echo "== web server (HTTPS) and the app"
sed "s/__HOST__/$HOST/" "$APP/webapp/deploy/Caddyfile" > /etc/caddy/Caddyfile
install -m 644 "$APP/webapp/deploy/lejet-web.service" /etc/systemd/system/lejet-web.service
systemctl daemon-reload
systemctl enable --now lejet-web
systemctl restart lejet-web
systemctl restart caddy

echo
echo "== done. Check:  curl -s http://127.0.0.1:8000/api/health   and open  https://$HOST"
