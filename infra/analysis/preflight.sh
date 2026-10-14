#!/usr/bin/env bash
# Le-Jet Level 2 isolation preflight, PARALLEL version (same 26 checks as preflight.sh, finishes in about a third of the time).
#
# STATUS: this is the installed preflight (install_bundle.sh installs preflight.sh). The previous sequential
# version is kept as preflight_sequential.sh (reference and rollback).
#
# Why it is faster: the checks of preflight.sh run one after another, and every "this must be BLOCKED" probe waits its whole
# 3-4 s timeout (test 9 waits for four ports in a row). Those probes do not depend on each other, so they run at the same time.
# What does NOT change: every command, every timeout, every pass rule, the token logic, and the output format. The two
# resource-limit tests (19 memory, 20 fork) run alone, after the network probes, so they cannot disturb them.
# Test 15 (listening sockets) also runs alone, after the probes: test 10 opens a temporary UDP socket on a random port
# (0.0.0.0:<high port>) for up to 3 s, and `ss -tuln` run at the same moment listed it as an "unexpected listener"
# (a false FAIL seen on 2026-10-14). Two preflights must not overlap either: lejet-sudo-preflight takes the gate lock.
# A bug in this file can only make the preflight FAIL (no token is written, so nothing is fetched): it fails closed.
#
# Usage:  ./preflight.sh [--router-ip IP] [--home-ip IP] [--record]
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

T=$(mktemp -d) || { echo "no temp dir"; exit 2; }
trap 'rm -rf "$T"' EXIT
# every test writes exactly one result file $T/<n> holding "STATUS<TAB>message"; the report is printed in order at the end
ok()   { printf 'PASS\t%s\n' "$2" > "$T/$1"; }
bad()  { printf 'FAIL\t%s\n' "$2" > "$T/$1"; }
skip() { printf 'SKIP\t%s\n' "$2" > "$T/$1"; }
blocked() { ! timeout 4 bash -c "</dev/tcp/$1/$2" 2>/dev/null; }   # true when the connection fails
has_answer() { grep -Eq '^([0-9]{1,3}\.){3}[0-9]{1,3}$|^[0-9a-fA-F]*:[0-9a-fA-F:]+$'; }

# blocked_all HOST PORT...: all ports probed at the same time; true only if EVERY one is blocked
blocked_all() {
  local host="$1"; shift
  local d; d=$(mktemp -d "$T/b.XXXXXX")
  local p
  for p in "$@"; do ( blocked "$host" "$p" && : > "$d/$p" ) & done
  wait
  for p in "$@"; do [ -f "$d/$p" ] || return 1; done
  return 0
}

sudo -v || { echo "sudo is needed"; exit 2; }
echo "== Le-Jet Level 2 preflight on $(hostname) at $(date -u +%FT%TZ) =="

t1() {  # 1 Only the internal IPv4 address; no IPv6
  v4=$(ip -4 -o addr show scope global | awk '{print $4}' | tr '\n' ' ')
  v6=$(ip -6 -o addr show | grep -vc ' lo ')
  if [ "$v4" = "10.77.0.2/24 " ] && [ "$v6" = "0" ]; then ok 1 "only 10.77.0.2/24, no IPv6"; else bad 1 "addresses: '$v4' ipv6 lines: $v6"; fi
}
t2() {  # 2 Default route only via the gateway
  dr=$(ip route | grep -c '^default')
  if ip route | grep -q '^default via 10.77.0.1 ' && [ "$dr" = "1" ]; then ok 2 "default route via 10.77.0.1 only"; else bad 2 "routes: $(ip route | grep '^default')"; fi
}
t3() {  # 3 VirtualBox host (NAT address) unreachable
  if blocked_all 10.0.2.2 445 22; then ok 3 "10.0.2.2 ports 445 and 22 blocked"; else bad 3 "10.0.2.2 reachable"; fi
}
t4() {  # 4 Home router unreachable
  if [ -n "$ROUTER_IP" ]; then
    if blocked "$ROUTER_IP" 80; then ok 4 "router $ROUTER_IP:80 blocked"; else bad 4 "router reachable"; fi
  else skip 4 "no --router-ip given"; fi
}
t5() {  # 5 Direct DNS to a public resolver blocked
  if ! dig @8.8.8.8 example.com +short +time=2 +tries=1 2>/dev/null | has_answer; then ok 5 "direct DNS to 8.8.8.8 blocked"; else bad 5 "direct DNS works"; fi
}
t6() {  # 6 DNS through the gateway works
  if dig example.com +short +time=3 +tries=1 2>/dev/null | has_answer; then ok 6 "DNS via gateway works"; else bad 6 "no DNS answer"; fi
}
t7() {  # 7 Rebinding protection
  if ! dig localtest.me +short +time=3 +tries=1 2>/dev/null | has_answer; then ok 7 "localtest.me returns no answer"; else bad 7 "rebinding answer got through"; fi
}
t8() {  # 8 Web ports work
  code=$(curl -4 -m 10 -s -o /dev/null -w '%{http_code}' https://example.com 2>/dev/null)
  case "$code" in 200|301|302) ok 8 "HTTPS to example.com works ($code)";; *) bad 8 "HTTPS failed ($code)";; esac
}
t9() {  # 9 Non-web ports blocked
  if blocked_all 1.1.1.1 22 25 3389 8080; then ok 9 "1.1.1.1 ports 22, 25, 3389, 8080 blocked"; else bad 9 "a non-web port is reachable"; fi
}
t10() {  # 10 UDP/QUIC blocked
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
}
t11() {  # 11 ICMP blocked (needs the ping tool; minimized Ubuntu has none)
  if command -v ping >/dev/null; then
    if ! ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then ok 11 "ICMP blocked"; else bad 11 "ping works"; fi
  else skip 11 "ping not installed (ICMP is dropped by policy anyway)"; fi
}
t12() {  # 12 Exit IP is the VPN, not home
  exitip=$(curl -4 -m 10 -s https://1.1.1.1/cdn-cgi/trace 2>/dev/null | sed -n 's/^ip=//p')
  if [ -n "$HOME_IP" ]; then
    if [ -n "$exitip" ] && [ "$exitip" != "$HOME_IP" ]; then ok 12 "exit IP $exitip differs from home IP"; else bad 12 "exit IP '$exitip'"; fi
  else skip 12 "exit IP is $exitip (compare with your home IP yourself, or pass --home-ip)"; fi
}
t14() {  # 14 No VirtualBox integration
  if ! lsmod | grep -q vbox && [ -z "$(ls -A /media 2>/dev/null)" ] && [ -z "$(ls -A /mnt 2>/dev/null)" ]; then ok 14 "no vbox modules, /media and /mnt empty"; else bad 14 "vbox module or mounted media found"; fi
}
t15() {  # 15 Listening sockets: nothing on a non-loopback address except sshd :22
  extra=$(ss -H -tulnp | awk '{print $5}' | grep -vE '^(127\.|\[::1\]|\*%lo|127\.0\.0\.[0-9]+%lo)' | grep -vE ':22$' | grep -v '^$')
  if [ -z "$extra" ]; then ok 15 "only sshd listens outside loopback"; else bad 15 "unexpected listeners: $extra"; fi
}
t16() {  # 16 fetcher cannot escalate
  if sudo -l -U fetcher 2>&1 | grep -q 'not allowed' && ! sudo -u fetcher sudo -n true >/dev/null 2>&1; then ok 16 "fetcher has no sudo"; else bad 16 "fetcher can escalate"; fi
}
t17() {  # 17 fetcher cannot write outside its work dir
  if ! sudo -u fetcher touch /etc/lejet-x /opt/lejet-x 2>/dev/null; then ok 17 "fetcher cannot write to /etc or /opt"; else bad 17 "fetcher wrote outside"; sudo rm -f /etc/lejet-x /opt/lejet-x; fi
}
t19() {  # 19 Memory limit enforced
  sudo systemd-run --quiet --wait --collect --pipe -p MemoryMax=256M -p MemorySwapMax=0 python3 -c 'x = bytearray(1024*1024*1024)' >/dev/null 2>&1
  if [ $? -ne 0 ]; then ok 19 "job asking for 1 GB was killed by the 256 MB limit"; else bad 19 "memory limit not enforced"; fi
}
t20() {  # 20 Fork bomb contained (the VM must stay responsive; the job itself may fail)
  timeout 40 sudo systemd-run --quiet --wait --collect --pipe -p TasksMax=64 stress-ng --fork 200 --timeout 8 >/dev/null 2>&1
  if timeout 10 uptime >/dev/null 2>&1; then ok 20 "fork test finished and the VM is still responsive"; else bad 20 "VM unresponsive after fork test"; fi
}
t21() {  # 21 Gateway reachable only for DNS/NTP
  if blocked 10.77.0.1 22; then ok 21 "gateway SSH port blocked from here"; else bad 21 "gateway SSH reachable"; fi
}
t22() {  # 22 Clock sane
  off=$(chronyc tracking 2>/dev/null | awk '/System time/ {print $4}')
  if [ -n "$off" ] && awk -v o="$off" 'BEGIN{exit !(o<1.0)}'; then ok 22 "clock offset ${off}s"; else bad 22 "clock offset '$off'"; fi
}
t23() {  # 23 Firewall ruleset unchanged
  # -s (stateless) leaves out packet counters, which change all the time and would break the hash
  cur=$(sudo nft -s list ruleset | sha256sum | awk '{print $1}')
  ref=/etc/lejet-ruleset.sha256
  if [ $RECORD = 1 ]; then echo "$cur" | sudo tee $ref >/dev/null; ok 23 "recorded ruleset hash ${cur:0:16}..."
  elif [ -f $ref ]; then
    if [ "$cur" = "$(cat $ref)" ]; then ok 23 "ruleset hash matches"; else bad 23 "ruleset changed"; fi
  else skip 23 "no reference hash yet (run once with --record)"; fi
}

# phase 1: everything that only probes the network or reads the system, all at once (NOT t15: see phase 2)
for n in 1 2 3 4 5 6 7 8 9 10 11 12 14 16 17 21; do ( "t$n" ) & done
wait
# phase 2: tests that must run alone. t15 lists the listening sockets, so it must not run while a probe of phase 1
# (test 10's UDP socket) is open. t19 and t20 are the resource-limit tests: they must not disturb the probes, and the
# probes must not disturb them.
t15
t19
t20
# phase 3: clock and firewall hash, after the load of phase 2 is gone
t22
t23

skip 13 "MANUAL: kill switch (already shown on the gateway)"
skip 18 "N/A in v1 (no browser stage)"
skip 24 "MANUAL: gateway log lines lejet-egress / lejet-blocked-private"
skip 25 "MANUAL: DNS not threat-filtered (compare a listed domain with an unfiltered resolver)"
skip 26 "MANUAL: stale-clock recovery after a snapshot restore"

PASS=0; FAIL=0; SKIP=0
for n in $(seq 1 26); do
  if [ -f "$T/$n" ]; then
    IFS=$'\t' read -r status msg < "$T/$n"
  else
    status=FAIL; msg="test produced no result"      # a test that crashed or never reported counts as a failure
  fi
  printf '%s  %-3s %s\n' "$status" "$n" "$msg"
  case "$status" in PASS) PASS=$((PASS+1));; SKIP) SKIP=$((SKIP+1));; *) FAIL=$((FAIL+1));; esac
done

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
