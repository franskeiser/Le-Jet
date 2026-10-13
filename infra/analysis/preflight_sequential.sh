#!/usr/bin/env bash
# REFERENCE / ROLLBACK COPY of the sequential preflight (the version before 2026-10-13). preflight.sh is now the
# parallel version; tests/test_preflight_parallel.py compares the two. To roll back: copy this over preflight.sh and redeploy.
# Le-Jet Level 2 isolation preflight (WEB_INSPECTION_PLAN.md section 16, Level 2)
# Runs the automatable tests on the ANALYSIS VM. Tests that need the gateway or the host are listed as MANUAL.
#
# Usage:  ./preflight.sh [--router-ip IP] [--home-ip IP] [--record]
#   --router-ip  your home router address (test 4); skipped if not given
#   --home-ip    your real public IP (test 12); the exit IP is only printed if not given
#   --record     store the firewall ruleset hash (test 23) as the reference
set -u

ROUTER_IP=""; HOME_IP=""; RECORD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --router-ip) ROUTER_IP="$2"; shift 2 ;;
    --home-ip)   HOME_IP="$2"; shift 2 ;;
    --record)    RECORD=1; shift ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

PASS=0; FAIL=0; SKIP=0
ok()   { printf 'PASS  %-3s %s\n' "$1" "$2"; PASS=$((PASS+1)); }
bad()  { printf 'FAIL  %-3s %s\n' "$1" "$2"; FAIL=$((FAIL+1)); }
skip() { printf 'SKIP  %-3s %s\n' "$1" "$2"; SKIP=$((SKIP+1)); }
blocked() { ! timeout 4 bash -c "</dev/tcp/$1/$2" 2>/dev/null; }   # true when the connection fails
# true when the text on stdin contains at least one IP address line (dig +short answer)
has_answer() { grep -Eq '^([0-9]{1,3}\.){3}[0-9]{1,3}$|^[0-9a-fA-F]*:[0-9a-fA-F:]+$'; }

sudo -v || { echo "sudo is needed"; exit 2; }
echo "== Le-Jet Level 2 preflight on $(hostname) at $(date -u +%FT%TZ) =="

# 1 Only the internal IPv4 address; no IPv6
v4=$(ip -4 -o addr show scope global | awk '{print $4}' | tr '\n' ' ')
v6=$(ip -6 -o addr show | grep -vc ' lo ')
if [ "$v4" = "10.77.0.2/24 " ] && [ "$v6" = "0" ]; then ok 1 "only 10.77.0.2/24, no IPv6"; else bad 1 "addresses: '$v4' ipv6 lines: $v6"; fi

# 2 Default route only via the gateway
dr=$(ip route | grep -c '^default')
if ip route | grep -q '^default via 10.77.0.1 ' && [ "$dr" = "1" ]; then ok 2 "default route via 10.77.0.1 only"; else bad 2 "routes: $(ip route | grep '^default')"; fi

# 3 VirtualBox host (NAT address) unreachable
if blocked 10.0.2.2 445 && blocked 10.0.2.2 22; then ok 3 "10.0.2.2 ports 445 and 22 blocked"; else bad 3 "10.0.2.2 reachable"; fi

# 4 Home router unreachable
if [ -n "$ROUTER_IP" ]; then
  if blocked "$ROUTER_IP" 80; then ok 4 "router $ROUTER_IP:80 blocked"; else bad 4 "router reachable"; fi
else skip 4 "no --router-ip given"; fi

# 5 Direct DNS to a public resolver blocked
if ! dig @8.8.8.8 example.com +short +time=2 +tries=1 2>/dev/null | has_answer; then ok 5 "direct DNS to 8.8.8.8 blocked"; else bad 5 "direct DNS works"; fi

# 6 DNS through the gateway works
if dig example.com +short +time=3 +tries=1 2>/dev/null | has_answer; then ok 6 "DNS via gateway works"; else bad 6 "no DNS answer"; fi

# 7 Rebinding protection
if ! dig localtest.me +short +time=3 +tries=1 2>/dev/null | has_answer; then ok 7 "localtest.me returns no answer"; else bad 7 "rebinding answer got through"; fi

# 8 Web ports work
code=$(curl -4 -m 10 -s -o /dev/null -w '%{http_code}' https://example.com 2>/dev/null)
case "$code" in 200|301|302) ok 8 "HTTPS to example.com works ($code)";; *) bad 8 "HTTPS failed ($code)";; esac

# 9 Non-web ports blocked
all=1; for p in 22 25 3389 8080; do blocked 1.1.1.1 $p || all=0; done
if [ $all = 1 ]; then ok 9 "1.1.1.1 ports 22, 25, 3389, 8080 blocked"; else bad 9 "a non-web port is reachable"; fi

# 10 UDP/QUIC blocked
r=$(python3 - <<'PY'
import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(3)
try:
    s.sendto(b'x',('1.1.1.1',443)); s.recv(10); print('reply')
except Exception:
    print('blocked')
PY
)
if [ "$r" = "blocked" ]; then ok 10 "UDP 443 blocked"; else bad 10 "UDP 443 got a reply"; fi

# 11 ICMP blocked (needs the ping tool; minimized Ubuntu has none)
if command -v ping >/dev/null; then
  if ! ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then ok 11 "ICMP blocked"; else bad 11 "ping works"; fi
else skip 11 "ping not installed (ICMP is dropped by policy anyway)"; fi

# 12 Exit IP is the VPN, not home
exitip=$(curl -4 -m 10 -s https://1.1.1.1/cdn-cgi/trace 2>/dev/null | sed -n 's/^ip=//p')
if [ -n "$HOME_IP" ]; then
  if [ -n "$exitip" ] && [ "$exitip" != "$HOME_IP" ]; then ok 12 "exit IP $exitip differs from home IP"; else bad 12 "exit IP '$exitip'"; fi
else skip 12 "exit IP is $exitip (compare with your home IP yourself, or pass --home-ip)"; fi

# 13 Kill switch: MANUAL (gateway: wg-quick down wg0, repeat test 8)
skip 13 "MANUAL: kill switch (already shown on the gateway)"

# 14 No VirtualBox integration
if ! lsmod | grep -q vbox && [ -z "$(ls -A /media 2>/dev/null)" ] && [ -z "$(ls -A /mnt 2>/dev/null)" ]; then ok 14 "no vbox modules, /media and /mnt empty"; else bad 14 "vbox module or mounted media found"; fi

# 15 Listening sockets: nothing on a non-loopback address except sshd :22
extra=$(ss -H -tulnp | awk '{print $5}' | grep -vE '^(127\.|\[::1\]|\*%lo|127\.0\.0\.[0-9]+%lo)' | grep -vE ':22$' | grep -v '^$')
if [ -z "$extra" ]; then ok 15 "only sshd listens outside loopback"; else bad 15 "unexpected listeners: $extra"; fi

# 16 fetcher cannot escalate
if sudo -l -U fetcher 2>&1 | grep -q 'not allowed' && ! sudo -u fetcher sudo -n true >/dev/null 2>&1; then ok 16 "fetcher has no sudo"; else bad 16 "fetcher can escalate"; fi

# 17 fetcher cannot write outside its work dir
if ! sudo -u fetcher touch /etc/lejet-x /opt/lejet-x 2>/dev/null; then ok 17 "fetcher cannot write to /etc or /opt"; else bad 17 "fetcher wrote outside"; sudo rm -f /etc/lejet-x /opt/lejet-x; fi

# 18 Chromium sandbox: not applicable in v1 (no browser)
skip 18 "N/A in v1 (no browser stage)"

# 19 Memory limit enforced
sudo systemd-run --quiet --wait --collect --pipe -p MemoryMax=256M -p MemorySwapMax=0 python3 -c 'x = bytearray(1024*1024*1024)' >/dev/null 2>&1
if [ $? -ne 0 ]; then ok 19 "job asking for 1 GB was killed by the 256 MB limit"; else bad 19 "memory limit not enforced"; fi

# 20 Fork bomb contained (the VM must stay responsive; the job itself may fail)
timeout 40 sudo systemd-run --quiet --wait --collect --pipe -p TasksMax=64 stress-ng --fork 200 --timeout 8 >/dev/null 2>&1
if timeout 10 uptime >/dev/null 2>&1; then ok 20 "fork test finished and the VM is still responsive"; else bad 20 "VM unresponsive after fork test"; fi

# 21 Gateway reachable only for DNS/NTP
if blocked 10.77.0.1 22; then ok 21 "gateway SSH port blocked from here"; else bad 21 "gateway SSH reachable"; fi

# 22 Clock sane
off=$(chronyc tracking 2>/dev/null | awk '/System time/ {print $4}')
if [ -n "$off" ] && awk -v o="$off" 'BEGIN{exit !(o<1.0)}'; then ok 22 "clock offset ${off}s"; else bad 22 "clock offset '$off'"; fi

# 23 Firewall ruleset unchanged
# -s (stateless) leaves out packet counters, which change all the time and would break the hash
cur=$(sudo nft -s list ruleset | sha256sum | awk '{print $1}')
ref=/etc/lejet-ruleset.sha256
if [ $RECORD = 1 ]; then echo "$cur" | sudo tee $ref >/dev/null; ok 23 "recorded ruleset hash ${cur:0:16}..."
elif [ -f $ref ]; then
  if [ "$cur" = "$(cat $ref)" ]; then ok 23 "ruleset hash matches"; else bad 23 "ruleset changed"; fi
else skip 23 "no reference hash yet (run once with --record)"; fi

# 24, 25, 26: MANUAL (gateway logs, unfiltered-DNS comparison, stale-clock recovery)
skip 24 "MANUAL: gateway log lines lejet-egress / lejet-blocked-private"
skip 25 "MANUAL: DNS not threat-filtered (compare a listed domain with an unfiltered resolver)"
skip 26 "MANUAL: stale-clock recovery after a snapshot restore"

echo "== PASS $PASS  FAIL $FAIL  SKIP $SKIP =="
if [ $FAIL -eq 0 ]; then
  date -u +%s | sudo tee /run/lejet-preflight-ok >/dev/null
  echo "Preflight token written (/run/lejet-preflight-ok)."
  exit 0
else
  sudo rm -f /run/lejet-preflight-ok
  echo "Preflight FAILED: fix the FAIL lines before any fetch."
  exit 1
fi
