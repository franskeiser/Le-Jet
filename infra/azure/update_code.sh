#!/usr/bin/env bash
# Install / refresh the Le-Jet CODE on the VM (after a code change; setup_shot_vm.sh calls this too).
# Run with sudo from the unpacked ~/lejet-src. Same layout as the VirtualBox lab's bundle (infra/analysis/install_bundle.sh):
#   * the `lejet` package            -> the venv's site-packages (read-only for everyone but root)
#   * the two repo files it imports  -> /opt/lejet/src/features/link_features.py, /opt/lejet/src/data/clean_web_phishing.py
#   * /opt/lejet/lejet.pth           -> so `python -I` finds `src.features` (the venv reads .pth files even with -I)
#   * /opt/lejet/rdap-dns.json       -> IANA domain-registry bootstrap (RDAP), fetched once
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run this with sudo" >&2; exit 1; }
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/opt/lejet/venv/bin/python
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"

find "$SRC/src" -type f -name '*.py' -exec sed -i 's/\r$//' {} +          # a Windows checkout may have CRLF

rm -rf "$SITE/lejet" && cp -r "$SRC/src/lejet" "$SITE/lejet"
find "$SITE/lejet" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

rm -rf /opt/lejet/src && install -d -m 755 /opt/lejet/src/features /opt/lejet/src/data
install -m 644 "$SRC/src/features/link_features.py" /opt/lejet/src/features/link_features.py
install -m 644 "$SRC/src/data/clean_web_phishing.py" /opt/lejet/src/data/clean_web_phishing.py
printf '/opt/lejet\n' > "$SITE/lejet.pth"
chown -R root:root "$SITE/lejet" /opt/lejet/src "$SITE/lejet.pth"
chmod -R go-w "$SITE/lejet" /opt/lejet/src

if [ ! -s /opt/lejet/rdap-dns.json ]; then
  if curl -fsS --max-time 30 --max-filesize 5000000 -o /opt/lejet/rdap-dns.json.tmp https://data.iana.org/rdap/dns.json \
     && "$PY" -c 'import json; assert json.load(open("/opt/lejet/rdap-dns.json.tmp"))["services"]'; then
    mv /opt/lejet/rdap-dns.json.tmp /opt/lejet/rdap-dns.json
    chmod 644 /opt/lejet/rdap-dns.json
  else
    rm -f /opt/lejet/rdap-dns.json.tmp
    echo "WARNING: could not fetch the RDAP bootstrap; domain-age lookups will be empty until /opt/lejet/rdap-dns.json exists" >&2
  fi
fi

# the import check, as the unprivileged user and with -I (exactly how the lab will run)
runuser -u fetcher -- "$PY" -I - <<'PY'
import sys
import lejet.cli, lejet.pipeline, lejet.live_worker, lejet.azure_run, lejet.azure_features
from src.features import link_features as lf
import httpx, dns.version, lxml.etree as et, tldextract
print("   imports as user fetcher with python -I: OK | python", sys.version.split()[0], "| httpx", httpx.__version__)
PY
# the features-only helper (dataset mode) and the sudoers rule that lets the SSH user run it: refreshed with every code update, so
# `deploy_to_azure.ps1 -CodeOnly` is enough (it needs the SSH user's sudo password once; nothing else on the VM is touched)
if [ -f /usr/local/sbin/lejet-azure-live ] && [ -f "$SRC/infra/azure/lejet-azure-features.sh" ]; then
  SSH_USER="${SUDO_USER:-azureuser}"
  install -o root -g root -m 755 "$SRC/infra/azure/lejet-azure-features.sh" /usr/local/sbin/lejet-azure-features
  sed "s/__SSH_USER__/$SSH_USER/g" "$SRC/infra/azure/92-lejet-azure.sudoers" > /tmp/92-lejet-azure
  visudo -cf /tmp/92-lejet-azure >/dev/null && install -o root -g root -m 440 /tmp/92-lejet-azure /etc/sudoers.d/92-lejet-azure
  rm -f /tmp/92-lejet-azure
  echo "   features-only helper installed"
fi
echo "code updated"
