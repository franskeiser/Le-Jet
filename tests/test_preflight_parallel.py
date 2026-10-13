"""infra/analysis/preflight.sh (parallel) must run EXACTLY the checks of preflight_sequential.sh, only at the same time.

Nothing here runs a probe (the script needs the analysis VM). The tests compare the two scripts: same commands,
same timeouts, same pass messages, nothing dropped, nothing added, and no check weakened on the way.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[1] / "infra" / "analysis"
ORIGINAL = (INFRA / "preflight_sequential.sh").read_text(encoding="utf-8")   # the version before 2026-10-13
PARALLEL = (INFRA / "preflight.sh").read_text(encoding="utf-8")                # the installed one
RESULT_CALL = re.compile(r'\b(ok|bad|skip)\s+(\d+)\s+"((?:[^"\\]|\\.)*)"')

# every probe command of the original, character for character
COMMANDS = [
    "ip -4 -o addr show scope global", "ip -6 -o addr show", "grep -q '^default via 10.77.0.1 '",
    "dig @8.8.8.8 example.com +short +time=2 +tries=1", "dig example.com +short +time=3 +tries=1",
    "dig localtest.me +short +time=3 +tries=1", "curl -4 -m 10 -s -o /dev/null -w '%{http_code}' https://example.com",
    "curl -4 -m 10 -s https://1.1.1.1/cdn-cgi/trace", "s.settimeout(3)", "s.sendto(b'x',('1.1.1.1',443))",
    "ping -c1 -W2 1.1.1.1", "lsmod | grep -q vbox", "ss -H -tulnp", "sudo -l -U fetcher", "sudo -u fetcher sudo -n true",
    "sudo -u fetcher touch /etc/lejet-x /opt/lejet-x",
    "-p MemoryMax=256M -p MemorySwapMax=0 python3 -c 'x = bytearray(1024*1024*1024)'",
    "timeout 40 sudo systemd-run --quiet --wait --collect --pipe -p TasksMax=64 stress-ng --fork 200 --timeout 8",
    "timeout 10 uptime", "blocked 10.77.0.1 22", "chronyc tracking", "sudo nft -s list ruleset | sha256sum",
    "timeout 4 bash -c", "date -u +%s | sudo tee /run/lejet-preflight-ok", "sudo rm -f /run/lejet-preflight-ok",
]


def results(text: str) -> set[tuple[str, int, str]]:
    return {(kind, int(n), msg) for kind, n, msg in RESULT_CALL.findall(text)}


def test_same_pass_fail_and_skip_messages_for_every_test():
    assert results(PARALLEL) == results(ORIGINAL)


def test_every_probe_command_is_still_there_unchanged():
    for command in COMMANDS:
        assert command in ORIGINAL, f"test list is out of date: {command}"
        assert command in PARALLEL, f"parallel version lost or changed: {command}"


def test_every_check_from_1_to_26_reports_and_a_missing_result_counts_as_a_failure():
    assert {n for _, n, _ in results(PARALLEL)} == set(range(1, 27))
    assert 'msg="test produced no result"' in PARALLEL and "status=FAIL" in PARALLEL


def test_blocked_probes_keep_their_timeouts_and_ports():
    assert "blocked_all 10.0.2.2 445 22" in PARALLEL
    assert "blocked_all 1.1.1.1 22 25 3389 8080" in PARALLEL
    assert "blocked_all() {" in PARALLEL and "[ -f \"$d/$p\" ] || return 1" in PARALLEL   # ALL ports must be blocked


def test_resource_limit_tests_run_alone_after_the_probes():
    phase1 = PARALLEL.index("# phase 1")
    phase2 = PARALLEL.index("# phase 2")
    assert phase1 < PARALLEL.index("\nwait\n", phase1) < phase2
    after = PARALLEL[phase2:]
    assert after.index("\nt19\n") < after.index("\nt20\n") < after.index("\nt22\n") < after.index("\nt23\n")
    assert "t19" not in PARALLEL[phase1:phase2].split("\nwait\n")[0].replace("t19()", "")


def test_the_token_is_written_only_when_nothing_failed():
    tail = PARALLEL[PARALLEL.index('echo "== PASS $PASS'):]
    assert tail.index("if [ $FAIL -eq 0 ]") < tail.index("/run/lejet-preflight-ok") < tail.index("else")
    assert tail.count("sudo tee /run/lejet-preflight-ok") == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_script_is_valid_bash():
    done = subprocess.run(["bash", "-n", "preflight.sh"], capture_output=True, text=True, cwd=INFRA)
    assert done.returncode == 0, done.stderr
