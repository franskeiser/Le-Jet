#!/usr/bin/env bash
# One-time setup of the screenshot lab on the Azure VM that ALSO runs the SMS collection website.
# Run on the VM as root, from the folder where deploy_to_azure.ps1 unpacked the code (~/lejet-src):
#
#     sudo bash ~/lejet-src/infra/azure/setup_shot_vm.sh
#
# What it does, and what it deliberately does NOT do:
#   * creates the unprivileged user `fetcher` (no login, no sudo) that will open the links, /opt/lejet and /var/lib/lejet;
#   * installs the Python packages (venv /opt/lejet/venv) and Chromium (infra/analysis/install_browser.sh, reused as is);
#   * loads its OWN nftables table `lejet_shot_egress` for user `fetcher` (the website's table `lejet_egress` is not touched);
#   * installs the root helper `lejet-azure-live`, the gate `lejet-azure-gate` + 4-minute timer, and a sudoers rule that
#     lets ONLY the SSH user run those two commands;
#   * adds 2 GiB of swap if the VM has none (Chromium + the website's OCR on 4 GiB RAM);
#   * never restarts, reconfigures or reads the website (Caddy, lejet-web.service, /var/lib/lejetweb); the last step
#     checks that it is still running.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run this with sudo" >&2; exit 1; }
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_USER="${SUDO_USER:-azureuser}"
export DEBIAN_FRONTEND=noninteractive
# files copied from a Windows checkout may have CRLF line endings, which Linux shells reject ("set: pipefail\r"): normalise
find "$SRC/infra" "$SRC/requirements" -type f \( -name '*.sh' -o -name '*.nft' -o -name '*.sudoers' -o -name '*.service' \
  -o -name '*.timer' -o -name '*.in' \) -exec sed -i 's/\r$//' {} +

web_state() { systemctl is-active lejet-web.service 2>/dev/null || echo "not-installed"; }
WEB_BEFORE="$(web_state)"
echo "website state before: $WEB_BEFORE"

echo "== 1/8 packages"
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip nftables ca-certificates curl \
  fonts-noto-core fonts-noto-cjk fonts-noto-color-emoji fonts-liberation >/dev/null

echo "== 2/8 user and folders"
id fetcher >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/lejet/work/home --shell /usr/sbin/nologin fetcher
install -d -o root -g root -m 755 /opt/lejet
install -d -o fetcher -g fetcher -m 750 /var/lib/lejet /var/lib/lejet/in /var/lib/lejet/out /var/lib/lejet/out/export \
  /var/lib/lejet/work /var/lib/lejet/work/home
# the website's data folder is NOT touched here: it is already closed to other users (drwxr-x--- lejetweb), and
# gate.sh proves every few minutes that `fetcher` cannot read it

echo "== 3/8 swap safety net"
if [ "$(swapon --show --noheadings | wc -l)" -eq 0 ]; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== 4/8 Python environment"
python3 -m venv /opt/lejet/venv
/opt/lejet/venv/bin/pip install -q --upgrade pip
# same pins as requirements/vm.in, installed online (no offline bundle on this VM); nostril comes from its git commit
grep -vE '^\s*#|^\s*$|^nostril' "$SRC/requirements/vm.in" > /tmp/vm-online.in
/opt/lejet/venv/bin/pip install -q -r /tmp/vm-online.in "git+https://github.com/casics/nostril.git@fbc0c91249283a9fbc9036206391ce1138826fd3" \
  || /opt/lejet/venv/bin/pip install -q -r /tmp/vm-online.in
rm -f /tmp/vm-online.in
# the lejet code (package, the two repo files it imports, .pth, RDAP bootstrap, import check): update_code.sh
bash "$SRC/infra/azure/update_code.sh"

echo "== 5/8 Chromium (reusing infra/analysis/install_browser.sh)"
bash "$SRC/infra/analysis/install_browser.sh"

echo "== 6/8 firewall rule for user fetcher (own table; the website's table is untouched)"
install -m 644 "$SRC/infra/azure/egress.nft" /etc/lejet-shot-egress.nft
grep -q 'lejet-shot-egress.nft' /etc/nftables.conf 2>/dev/null || echo 'include "/etc/lejet-shot-egress.nft"' >> /etc/nftables.conf
nft delete table inet lejet_shot_egress 2>/dev/null || true
nft -f /etc/lejet-shot-egress.nft
systemctl enable nftables >/dev/null 2>&1 || true

echo "== 7/8 helper, gate, timer, sudoers"
install -o root -g root -m 755 "$SRC/infra/azure/lejet-azure-live.sh" /usr/local/sbin/lejet-azure-live
install -o root -g root -m 755 "$SRC/infra/azure/gate.sh" /usr/local/sbin/lejet-azure-gate
install -m 644 "$SRC/infra/azure/systemd/lejet-azure-gate.service" /etc/systemd/system/
install -m 644 "$SRC/infra/azure/systemd/lejet-azure-gate.timer" /etc/systemd/system/
sed "s/__SSH_USER__/$SSH_USER/g" "$SRC/infra/azure/92-lejet-azure.sudoers" > /tmp/92-lejet-azure
visudo -cf /tmp/92-lejet-azure >/dev/null && install -o root -g root -m 440 /tmp/92-lejet-azure /etc/sudoers.d/92-lejet-azure
rm -f /tmp/92-lejet-azure
install -d -o "$SSH_USER" -g "$SSH_USER" -m 755 "/home/$SSH_USER/incoming" "/home/$SSH_USER/export"
systemctl daemon-reload
systemctl enable --now lejet-azure-gate.timer >/dev/null
/usr/local/sbin/lejet-azure-gate

echo "== 8/8 the website must be exactly as before"
WEB_AFTER="$(web_state)"
echo "website state after: $WEB_AFTER (before: $WEB_BEFORE)"
[ "$WEB_BEFORE" = "$WEB_AFTER" ] || { echo "WARNING: the website state changed; look at: systemctl status lejet-web caddy" >&2; exit 1; }
echo "done. From your PC:  ssh lejet-azure 'sudo -n /usr/local/sbin/lejet-azure-gate'"
