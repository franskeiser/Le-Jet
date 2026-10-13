#!/usr/bin/env bash
# Install the Le-Jet inspector from an OFFLINE, hash-pinned bundle.
# Run on the ANALYSIS VM, as root:   sudo bash install_bundle.sh /tmp/lejet-bundle [test]
#
# What it does (WEB_INSPECTION_PLAN.md sections 11.6, 11.7):
#   1. verifies every file of the bundle against MANIFEST.sha256;
#   2. creates /opt/lejet/venv and installs ONLY the pinned wheels (no network, hashes required);
#   3. copies the code to /opt/lejet/src (and the tests to /opt/lejet/tests);
#   4. adds a .pth file so `python -I -m lejet.cli` finds `lejet` and `src.features` (-I ignores
#      PYTHONPATH and the current directory, but still reads .pth files in site-packages);
#   5. fetches the IANA RDAP bootstrap file while the network is open;
#   6. locks permissions (root owns the code; `fetcher` can only read it) and self-checks;
#   7. installs six root-owned helpers in /usr/local/sbin plus a sudoers rule that lets `ops` run
#      exactly those without a password (see infra/analysis/helpers/), and the background isolation gate
#      (a systemd timer that repeats the 26-check self-test every few minutes).
set -euo pipefail

BUNDLE="${1:-/tmp/lejet-bundle}"
MODE="${2:-run}"                       # "test" also installs pytest from the same wheel folder
REQ="vm.txt"
[ "$MODE" = "test" ] && REQ="vm-test.txt"

[ "$(id -u)" -eq 0 ] || { echo "run this with sudo"; exit 1; }
[ -f "$BUNDLE/requirements/$REQ" ] && [ -d "$BUNDLE/wheels" ] && [ -d "$BUNDLE/code" ] \
  || { echo "bundle not found or incomplete at $BUNDLE"; exit 1; }

echo "== 1/6 verifying the bundle"
( cd "$BUNDLE" && sha256sum --quiet -c MANIFEST.sha256 ) && echo "   all files match MANIFEST.sha256"

echo "== 2/6 virtual environment and pinned wheels"
install -d -o root -g root -m 755 /opt/lejet
rm -rf /opt/lejet/venv
python3 -m venv /opt/lejet/venv
# --no-deps: the requirements file lists the WHOLE closure with hashes (checked by
# scripts/verify_vm_bundle.py), and nostril's packaging wrongly lists pytest as a dependency.
/opt/lejet/venv/bin/python -m pip install --quiet --no-index --no-deps --find-links "$BUNDLE/wheels" \
  --require-hashes -r "$BUNDLE/requirements/$REQ"
/opt/lejet/venv/bin/python -m pip list --format=freeze | sed 's/^/   /'

echo "== 3/6 code"
rm -rf /opt/lejet/src /opt/lejet/tests
cp -a "$BUNDLE/code/." /opt/lejet/
install -d -o root -g root -m 755 /opt/lejet/browsers

echo "== 4/6 import path (.pth)"
SITE="$(/opt/lejet/venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '/opt/lejet/src\n/opt/lejet\n' > "$SITE/lejet.pth"

echo "== 5/6 IANA RDAP bootstrap"
if curl -fsS --max-time 30 --max-filesize 5000000 -o /opt/lejet/rdap-dns.json.tmp https://data.iana.org/rdap/dns.json \
   && /opt/lejet/venv/bin/python -c 'import json,sys; d=json.load(open("/opt/lejet/rdap-dns.json.tmp")); assert d["services"]'; then
  mv /opt/lejet/rdap-dns.json.tmp /opt/lejet/rdap-dns.json
  echo "   saved /opt/lejet/rdap-dns.json ($(/opt/lejet/venv/bin/python -c 'import json; print(len(json.load(open("/opt/lejet/rdap-dns.json"))["services"]))') services)"
else
  rm -f /opt/lejet/rdap-dns.json.tmp
  echo "   WARNING: could not fetch the bootstrap file; RDAP will report rdap_ok=0 until it exists"
fi

echo "== 6/6 permissions and self-check"
chown -R root:root /opt/lejet
chmod -R u=rwX,go=rX /opt/lejet
sudo -u fetcher /opt/lejet/venv/bin/python -I - <<'PY'
import sys
import lejet.cli, lejet.pipeline, lejet.parse_pool
from src.features import link_features as lf
import httpx, dns.version, lxml.etree as et, tldextract
print("   python", sys.version.split()[0], "| httpx", httpx.__version__, "| dnspython", dns.version.version,
      "| lxml", et.LXML_VERSION, "| imports as user fetcher with python -I: OK")
print("   sample features:", lf.link_text_features(lf.normalize_link("gcash-verify.xyz/login"))["url_length"], "(url_length of the sample link)")
for bad in ("pandas", "numpy"):
    assert bad not in sys.modules, bad + " should not be needed"
PY
echo "== 7/7 root-owned helpers (passwordless sudo for exactly six fixed commands) and the background gate"
HELPERS="$BUNDLE/infra/analysis/helpers"
install -d -o root -g root -m 755 /usr/local/lib/lejet /etc/lejet
install -o root -g root -m 755 "$BUNDLE/infra/analysis/preflight.sh" /usr/local/lib/lejet/preflight.sh
for h in preflight install-urls batch export shots live; do
  install -o root -g root -m 755 "$HELPERS/lejet-sudo-$h.sh" "/usr/local/sbin/lejet-sudo-$h"
done
# provenance for the export manifest: created once, then yours to edit (root-owned)
if [ ! -f /etc/lejet/provenance.env ]; then
  printf 'LEJET_VPN_PROVIDER=Mullvad\nLEJET_VPN_EXIT_COUNTRY=PH\nLEJET_VBOX_VERSION=7.2.20\n' > /etc/lejet/provenance.env
  chmod 644 /etc/lejet/provenance.env
fi
install -d -o ops -g ops -m 755 /home/ops/incoming
# the background isolation gate: the self-test every few minutes, under the lock that lejet-sudo-live shares
install -o root -g root -m 644 "$BUNDLE/infra/analysis/systemd/lejet-gate.service" /etc/systemd/system/lejet-gate.service
install -o root -g root -m 644 "$BUNDLE/infra/analysis/systemd/lejet-gate.timer" /etc/systemd/system/lejet-gate.timer
systemctl daemon-reload
systemctl enable --now lejet-gate.timer >/dev/null 2>&1 && echo "   background isolation gate: lejet-gate.timer enabled"
# the resident live worker (pipeline + shared Chromium stay loaded between links); restarted so it runs the code just installed.
# If it cannot start (e.g. the browser is not installed yet) nothing breaks: lejet-sudo-live falls back to the per-process path.
install -o root -g root -m 644 "$BUNDLE/infra/analysis/systemd/lejet-live.service" /etc/systemd/system/lejet-live.service
systemctl daemon-reload
systemctl enable lejet-live.service >/dev/null 2>&1 || true
if systemctl restart lejet-live.service; then echo "   resident live worker: lejet-live.service (re)started"; else echo "   WARNING: lejet-live.service did not start; live runs use the per-process path"; fi
tmp_sudoers="$(mktemp)"
install -m 440 "$HELPERS/91-lejet-helpers.sudoers" "$tmp_sudoers"
if visudo -cf "$tmp_sudoers" >/dev/null; then
  install -o root -g root -m 440 "$tmp_sudoers" /etc/sudoers.d/91-lejet-helpers
  echo "   helpers installed; sudoers rule accepted by visudo"
else
  echo "   ERROR: visudo rejected the sudoers rule; NOT installed" >&2
  rm -f "$tmp_sudoers"
  exit 1
fi
rm -f "$tmp_sudoers"
echo "done."
