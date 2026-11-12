#!/usr/bin/env bash
# Philippine exit for the Azure lab: send ONLY user `fetcher`'s traffic through a Mullvad WireGuard tunnel.
# Run on the VM as root, from the unpacked ~/lejet-src (scripts/deploy_to_azure.ps1 -VpnConfig does this for you):
#
#     sudo bash ~/lejet-src/infra/azure/vpn_setup.sh install [expected-country]     # config staged in ~/vpn/lejet-vpn.conf
#     sudo bash ~/lejet-src/infra/azure/vpn_setup.sh remove
#
# How it keeps everything else safe:
#   * policy routing by USER: `ip rule uidrange <fetcher> lookup 51821`, and table 51821 has a default route via the tunnel.
#     SSH, Caddy, the website and apt use the normal route; the tunnel's own packets (kernel, uid 0) use the normal route too;
#   * KILL-SWITCH in firewall: egress-vpn.nft only lets `fetcher` out through interface lejet-vpn (or loopback). Tunnel down =
#     the lab has no network at all, never the VM's real address;
#   * the config is REBUILT from validated fields only (keys, one IPv4 address, one IPv4:port endpoint): any PostUp/PreUp/Table/
#     DNS line or script in the file you give is ignored, so a config file can never run commands as root;
#   * the key lives in /etc/wireguard/lejet-vpn.conf, root-only (mode 600); `fetcher` cannot read it; the staged copy is removed.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run this with sudo" >&2; exit 1; }
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_USER="${SUDO_USER:-azureuser}"
IFACE=lejet-vpn
CONF=/etc/wireguard/$IFACE.conf
STAGED="/home/$SSH_USER/vpn/lejet-vpn.conf"
TABLE=51821
LISTEN=51921          # the tunnel's fixed local UDP port (the firewall exception for the tunnel's own packets matches it)
ACTION="${1:-}"
EXPECT="${2:-}"

find "$SRC/infra" -type f \( -name '*.sh' -o -name '*.nft' \) -exec sed -i 's/\r$//' {} +

install_pieces() {   # the current gate, helper and firewall variant (the gate and helper know about the VPN)
  install -o root -g root -m 755 "$SRC/infra/azure/gate.sh" /usr/local/sbin/lejet-azure-gate
  install -o root -g root -m 755 "$SRC/infra/azure/lejet-azure-live.sh" /usr/local/sbin/lejet-azure-live
}
load_nft() {         # $1 = file in infra/azure; $2 = VPN endpoint "IP:port" (fills the placeholders in egress-vpn.nft)
  local ep_ip="${2%:*}" ep_port="${2##*:}"
  if [ -n "${2:-}" ]; then
    sed -e "s/__WG_ENDPOINT__/$ep_ip/" -e "s/__WG_PORT__/$ep_port/" -e "s/__WG_LISTEN__/$LISTEN/"       "$SRC/infra/azure/$1" > /etc/lejet-shot-egress.nft
    chmod 644 /etc/lejet-shot-egress.nft
  else
    install -m 644 "$SRC/infra/azure/$1" /etc/lejet-shot-egress.nft
  fi
  nft -c -f /etc/lejet-shot-egress.nft
  nft delete table inet lejet_shot_egress 2>/dev/null || true
  nft -f /etc/lejet-shot-egress.nft
}
web_state() { systemctl is-active lejet-web.service 2>/dev/null || echo "not-installed"; }

case "$ACTION" in
remove)
  systemctl disable --now "wg-quick@$IFACE" 2>/dev/null || true
  ip link del "$IFACE" 2>/dev/null || true
  rm -f "$CONF" /etc/lejet/vpn.env /etc/lejet/vpn.expect
  install_pieces
  load_nft egress.nft
  /usr/local/sbin/lejet-azure-gate || true
  echo "VPN removed: the lab leaves through the VM's own address again."
  exit 0 ;;
refresh)   # re-apply the firewall and the fixed port to an ALREADY installed tunnel (no key needed)
  [ -f "$CONF" ] || { echo "no tunnel installed: run install first" >&2; exit 2; }
  ENDP="$(awk -F= '/^Endpoint/ {gsub(/[ ]/, "", $2); print $2}' "$CONF")"
  [[ "$ENDP" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}:[0-9]{1,5}$ ]] || { echo "cannot read the endpoint from $CONF" >&2; exit 2; }
  grep -q '^ListenPort' "$CONF" || sed -i "/^MTU = /a ListenPort = $LISTEN" "$CONF"
  sed -i "s/^ListenPort = .*/ListenPort = $LISTEN/" "$CONF"
  if [ -n "$EXPECT" ]; then       # lock the exit country: the gate then refuses to open links from any other country
    printf '%s
' "$EXPECT" > /etc/lejet/vpn.expect && chmod 644 /etc/lejet/vpn.expect
    printf 'LEJET_VPN_ACTIVE=1
LEJET_VPN_PROVIDER=Mullvad
LEJET_VPN_EXIT_COUNTRY=%s
' "$EXPECT" > /etc/lejet/vpn.env
    chmod 644 /etc/lejet/vpn.env
  fi
  install_pieces
  load_nft egress-vpn.nft "$ENDP"
  systemctl restart "wg-quick@$IFACE"
  sleep 2
  runuser -u fetcher -- curl -s -m 12 -o /dev/null http://1.1.1.1/ || true
  /usr/local/sbin/lejet-azure-gate
  exit 0 ;;
install) ;;
*) echo "usage: vpn_setup.sh install [expected-country] | refresh [expected-country] | remove" >&2; exit 2 ;;
esac

WEB_BEFORE="$(web_state)"
[ -f "$STAGED" ] && [ ! -L "$STAGED" ] || { echo "no staged config at $STAGED" >&2; exit 2; }
id fetcher >/dev/null 2>&1 || { echo "run setup_shot_vm.sh first (user fetcher is missing)" >&2; exit 2; }

echo "== 1/6 tools"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wireguard-tools iproute2 curl >/dev/null

echo "== 2/6 read the staged config (validated fields only)"
field() { awk -F= -v k="$1" 'BEGIN{IGNORECASE=1} $1 ~ "^[[:space:]]*"k"[[:space:]]*$" {sub(/^[^=]*=[[:space:]]*/, ""); gsub(/\r/, ""); print; exit}' "$STAGED"; }
PRIV="$(field PrivateKey)"; PUB="$(field PublicKey)"; ADDR_RAW="$(field Address)"; ENDP="$(field Endpoint)"
ADDR="$(printf '%s' "$ADDR_RAW" | tr ',' '\n' | tr -d ' ' | grep -E '^[0-9]{1,3}(\.[0-9]{1,3}){3}/[0-9]{1,2}$' | head -n1 || true)"
[[ "$PRIV" =~ ^[A-Za-z0-9+/]{43}=$ ]] || { echo "PrivateKey is missing or not a WireGuard key" >&2; exit 2; }
[[ "$PUB"  =~ ^[A-Za-z0-9+/]{43}=$ ]] || { echo "the peer PublicKey is missing or not a WireGuard key" >&2; exit 2; }
[[ "$ENDP" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}:[0-9]{1,5}$ ]] || { echo "Endpoint must be IPv4:port" >&2; exit 2; }
[ -n "$ADDR" ] || { echo "no IPv4 Address found in the config" >&2; exit 2; }
echo "   endpoint ${ENDP}, tunnel address ${ADDR} (keys not shown)"

echo "== 3/6 write /etc/wireguard/$IFACE.conf (root only)"
FUID="$(id -u fetcher)"
install -d -m 700 /etc/wireguard /etc/lejet
umask 077
cat > "$CONF" <<EOF
# Written by infra/azure/vpn_setup.sh. Do not edit by hand: run vpn_setup.sh again.
[Interface]
PrivateKey = $PRIV
Address = $ADDR
MTU = 1380
ListenPort = $LISTEN
Table = off
PostUp = ip -4 route add default dev %i table $TABLE
PostUp = ip -4 rule add uidrange $FUID-$FUID lookup $TABLE priority 11100
PostDown = ip -4 rule del uidrange $FUID-$FUID lookup $TABLE priority 11100
PostDown = ip -4 route del default dev %i table $TABLE

[Peer]
PublicKey = $PUB
AllowedIPs = 0.0.0.0/0
Endpoint = $ENDP
PersistentKeepalive = 25
EOF
chmod 600 "$CONF"
shred -u "$STAGED" 2>/dev/null || rm -f "$STAGED"
rmdir "/home/$SSH_USER/vpn" 2>/dev/null || true

echo "== 4/6 firewall with the kill-switch"
install_pieces
load_nft egress-vpn.nft "$ENDP"

echo "== 5/6 bring the tunnel up (and at every boot)"
systemctl disable --now "wg-quick@$IFACE" 2>/dev/null || true
ip link del "$IFACE" 2>/dev/null || true
systemctl enable --now "wg-quick@$IFACE" >/dev/null
for i in $(seq 1 20); do
  hs="$(wg show "$IFACE" latest-handshakes 2>/dev/null | awk '{print $2}' | head -n1 || true)"
  [ -n "${hs:-}" ] && [ "$hs" != 0 ] && break
  sleep 1
done
[ -n "${hs:-}" ] && [ "$hs" != 0 ] || { echo "the tunnel did not complete a handshake in 20 s (check the endpoint and key; run: vpn_setup.sh remove)" >&2; exit 3; }

echo "== 6/6 where does the lab appear to be?"
printf 'LEJET_VPN_ACTIVE=1\nLEJET_VPN_PROVIDER=Mullvad\nLEJET_VPN_EXIT_COUNTRY=%s\n' "${EXPECT:-unchecked}" > /etc/lejet/vpn.env
chmod 644 /etc/lejet/vpn.env
INFO="$(runuser -u fetcher -- curl -fsS --max-time 12 https://am.i.mullvad.net/json 2>&1 || true)"
echo "$INFO" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    print("   could not read the exit check:", raw[:200]); sys.exit(0)
print("   exit IP:", d.get("ip"), "| country:", d.get("country"), "| city:", d.get("city"),
      "| Mullvad exit:", d.get("mullvad_exit_ip"), "| server:", d.get("mullvad_exit_ip_hostname"))
' || true
if [ -n "$EXPECT" ]; then
  printf '%s\n' "$EXPECT" > /etc/lejet/vpn.expect
  chmod 644 /etc/lejet/vpn.expect
  echo "   the gate will now insist on exit country: $EXPECT"
else
  rm -f /etc/lejet/vpn.expect
  echo "   no expected country given: the gate checks only that traffic leaves through Mullvad"
fi
/usr/local/sbin/lejet-azure-gate
[ "$(web_state)" = "$WEB_BEFORE" ] || { echo "WARNING: the website state changed" >&2; exit 1; }
echo "done. The lab now leaves through Mullvad; SSH and the website are unchanged."
