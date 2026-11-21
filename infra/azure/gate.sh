#!/usr/bin/env bash
# The Azure lab's safety gate (the small cousin of infra/analysis/preflight.sh, which belongs to the VirtualBox lab).
# Run by root every few minutes (systemd timer lejet-azure-gate.timer). It writes the token /run/lejet-preflight-ok
# (a UNIX time) ONLY while every check below passes; any failure deletes the token, and lejet.azure_run refuses
# to start without a fresh token. So a broken firewall rule, or a wrong file permission, stops all page opening.
#
# Checks:
#   1. user `fetcher` exists, is not root, has no sudo rights;
#   2. the nftables table lejet_shot_egress is loaded;
#   3. AS fetcher: a connection to the Azure metadata service (169.254.169.254:80) is refused;
#   4. AS fetcher: a connection to this machine's SSH and website ports on loopback is refused;
#   5. AS fetcher: the website's data folder (/var/lib/lejetweb) cannot be read or listed;
#   6. the website is still active (the lab must never take it down).
set -uo pipefail
TOKEN=/run/lejet-preflight-ok
fail=0
say() { echo "gate: $*"; }
bad() { say "FAIL $*"; fail=1; }

id fetcher >/dev/null 2>&1 || bad "user fetcher does not exist"
[ "$(id -u fetcher 2>/dev/null || echo 0)" -ne 0 ] || bad "fetcher is root"
if runuser -u fetcher -- sudo -n true >/dev/null 2>&1; then bad "fetcher has sudo rights"; fi
nft list table inet lejet_shot_egress >/dev/null 2>&1 || bad "nftables table lejet_shot_egress is not loaded"

blocked() {   # blocked <host> <port>: must NOT connect, as fetcher
  if runuser -u fetcher -- timeout 4 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null; then
    bad "fetcher can connect to $1:$2"
  fi
}
blocked 169.254.169.254 80
blocked 168.63.129.16 80
for port in 22 80 443 8000; do blocked 127.0.0.1 "$port"; done

if [ -d /var/lib/lejetweb ]; then
  if runuser -u fetcher -- ls /var/lib/lejetweb >/dev/null 2>&1; then bad "fetcher can list /var/lib/lejetweb"; fi
  if runuser -u fetcher -- sh -c 'cat /var/lib/lejetweb/web_submissions.csv >/dev/null 2>&1'; then
    bad "fetcher can read the website's dataset"
  fi
fi
if systemctl list-unit-files lejet-web.service >/dev/null 2>&1 && systemctl cat lejet-web.service >/dev/null 2>&1; then
  systemctl is-active --quiet lejet-web.service || bad "the website service lejet-web is not active"
fi

# 7. only when the Mullvad tunnel is installed (vpn_setup.sh): the lab MUST leave through it
if [ -f /etc/lejet/vpn.env ] && grep -q '^LEJET_VPN_ACTIVE=1' /etc/lejet/vpn.env; then
  nft list table inet lejet_shot_egress 2>/dev/null | grep -q 'oifname != "lejet-vpn"' || bad "the VPN kill-switch rule is not loaded"
  hs="$(wg show lejet-vpn latest-handshakes 2>/dev/null | awk '{print $2}' | head -n1)"
  now="$(date +%s)"
  if [ -z "${hs:-}" ] || [ "$hs" = 0 ] || [ $(( now - hs )) -gt 240 ]; then bad "the VPN tunnel has no recent handshake"; fi
  # as fetcher, ask Mullvad where we appear to be: must be a Mullvad exit, and the expected country if one was set
  info="$(runuser -u fetcher -- curl -fsS --max-time 12 https://am.i.mullvad.net/json 2>/dev/null || true)"
  exit_ok="$(printf '%s' "$info" | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: print("unreadable"); sys.exit()
print("yes" if d.get("mullvad_exit_ip") is True else "no")
print(d.get("country",""))' 2>/dev/null || echo unreadable)"
  case "$(printf '%s' "$exit_ok" | head -n1)" in
    yes) ;;
    *) bad "fetcher does not leave through a Mullvad exit ($(printf '%s' "$exit_ok" | head -n1))" ;;
  esac
  if [ -s /etc/lejet/vpn.expect ]; then
    want="$(head -n1 /etc/lejet/vpn.expect)"; got="$(printf '%s' "$exit_ok" | sed -n 2p)"
    [ "$got" = "$want" ] || bad "exit country is '$got', expected '$want'"
  fi
fi

if [ "$fail" -eq 0 ]; then
  tmp="$(mktemp /run/lejet-token.XXXXXX)"
  date -u +%s > "$tmp"
  chmod 644 "$tmp"
  mv -f "$tmp" "$TOKEN"
  say "all checks passed: token written"
else
  rm -f "$TOKEN"
  say "token deleted: nothing will open links until this passes"
fi
exit "$fail"
