#!/usr/bin/env bash
# Install headless Chromium for the OPTIONAL screenshot stage.
# Run on the ANALYSIS VM, as root, AFTER install_bundle.sh (the Playwright package must be there):
#
#     sudo bash ~/install_browser.sh
#
# What it does (WEB_INSPECTION_PLAN.md phases 5.8 and 5.9):
#   1. installs the system libraries Chromium needs (apt, through the gateway: ports 80/443);
#   2. downloads Chromium once from Playwright's CDN (HTTPS) into /opt/lejet/browsers;
#   3. makes the files root-owned and read-only for everyone else, and records the SHA-256 of every
#      file in /opt/lejet/browsers/SHA256SUMS (so a later change is detectable);
#   4. adds a narrow AppArmor profile: Ubuntu 24.04 blocks the unprivileged user namespaces that the
#      Chromium SANDBOX needs; this profile allows exactly that for exactly these browser binaries,
#      so the sandbox can stay ON (never --no-sandbox);
#   5. runs the browser self-test as the unprivileged user `fetcher` (renders a page, checks the PNG,
#      and proves from /proc that no Chromium process runs without its sandbox).
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run this with sudo" >&2; exit 1; }
VENV=/opt/lejet/venv
PW="$VENV/bin/playwright"
export PLAYWRIGHT_BROWSERS_PATH=/opt/lejet/browsers
[ -x "$PW" ] || { echo "Playwright is not installed: run deploy_to_vm.ps1 first" >&2; exit 1; }

echo "== 1/5 system libraries for Chromium (apt)"
export DEBIAN_FRONTEND=noninteractive
"$PW" install-deps chromium

echo "== 2/5 Chromium download (Playwright CDN, HTTPS)"
install -d -o root -g root -m 755 "$PLAYWRIGHT_BROWSERS_PATH"
"$PW" install chromium

echo "== 3/5 permissions and file hashes"
chown -R root:root "$PLAYWRIGHT_BROWSERS_PATH"
chmod -R u=rwX,go=rX "$PLAYWRIGHT_BROWSERS_PATH"
rm -f "$PLAYWRIGHT_BROWSERS_PATH/SHA256SUMS"
( cd "$PLAYWRIGHT_BROWSERS_PATH" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /tmp/lejet-browser-sums )
install -o root -g root -m 644 /tmp/lejet-browser-sums "$PLAYWRIGHT_BROWSERS_PATH/SHA256SUMS"
rm -f /tmp/lejet-browser-sums
echo "   $(wc -l < "$PLAYWRIGHT_BROWSERS_PATH/SHA256SUMS") files, listing hash: $(sha256sum "$PLAYWRIGHT_BROWSERS_PATH/SHA256SUMS" | cut -d' ' -f1)"

echo "== 4/5 AppArmor: allow user namespaces (the Chromium sandbox) for these binaries only"
mapfile -t BINS < <(find "$PLAYWRIGHT_BROWSERS_PATH" -type f -perm -u+x \( -name chrome -o -name chrome-headless-shell \) | sort)
[ "${#BINS[@]}" -ge 1 ] || { echo "no Chromium binary found under $PLAYWRIGHT_BROWSERS_PATH" >&2; exit 1; }
PROFILE=/etc/apparmor.d/lejet-chromium
{
  echo "# Written by infra/analysis/install_browser.sh. Allows user namespaces (needed by the Chromium"
  echo "# sandbox on Ubuntu 24.04) for the Playwright Chromium binaries ONLY. Nothing else is confined or"
  echo "# unconfined differently because of this file."
  echo "abi <abi/4.0>,"
  echo "include <tunables/global>"
  echo
  for b in "${BINS[@]}"; do
    name="lejet-chromium-$(echo "$b" | sha256sum | cut -c1-8)"
    echo "profile $name $b flags=(unconfined) {"
    echo "  userns,"
    echo "}"
    echo
  done
} > "$PROFILE"
chmod 644 "$PROFILE"
if command -v apparmor_parser >/dev/null; then
  apparmor_parser -r "$PROFILE"
  echo "   profile loaded for ${#BINS[@]} binar$( [ ${#BINS[@]} = 1 ] && echo y || echo ies)"
else
  echo "   apparmor_parser not found: skipping (the self-test below shows whether the sandbox works)"
fi

echo "== 5/5 self-test as the unprivileged user 'fetcher'"
install -d -o fetcher -g fetcher -m 700 /var/lib/lejet/work/home
runuser -u fetcher -- env HOME=/var/lib/lejet/work/home TMPDIR=/var/lib/lejet/work \
  PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" \
  "$VENV/bin/python" -I -m lejet.browser_selftest
echo "done."
