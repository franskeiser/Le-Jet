"""The live helper and the background gate: what the sudo rule allows, what the helper refuses, and that the installer wires them.

Static checks of the files (the VM is not available here). They guard the properties that make the shortcut acceptable:
exactly five fixed helpers can be run without a password, the live helper never runs without a recent passing self-test,
it handles one link only, it never overlaps the self-test, and a failed self-test blocks it.
"""
from __future__ import annotations

import re
from pathlib import Path

INFRA = Path(__file__).resolve().parents[1] / "infra" / "analysis"
ROOT = INFRA.parents[1]


def read(rel: str) -> str:
    return (INFRA / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


SUDOERS = read("helpers/91-lejet-helpers.sudoers")
HELPER = read("helpers/lejet-sudo-live.sh")
INSTALL = read("install_bundle.sh")
SERVICE = read("systemd/lejet-gate.service")
TIMER = read("systemd/lejet-gate.timer")
PREFLIGHT = read("preflight.sh")
BUNDLE_PY = (ROOT / "scripts" / "build_vm_bundle.py").read_text(encoding="utf-8")


def rule() -> list[str]:
    line = next(l for l in SUDOERS.splitlines() if l.startswith("ops "))
    return [c.strip() for c in line.split("NOPASSWD:")[1].split(",")]


def test_exactly_six_helpers_and_a_clean_shutdown_run_without_a_password():
    commands = rule()
    helpers = sorted(c.split()[0].rsplit("/", 1)[1] for c in commands if "lejet-sudo-" in c)
    assert helpers == ["lejet-sudo-batch", "lejet-sudo-export", "lejet-sudo-install-urls", "lejet-sudo-live",
                       "lejet-sudo-preflight", "lejet-sudo-shots"]
    assert len(commands) == 7 and commands[-1] == "/usr/bin/systemctl poweroff"
    assert all(c.startswith("/usr/local/sbin/lejet-sudo-") or c == "/usr/bin/systemctl poweroff" for c in commands)
    assert "ALL" not in SUDOERS.split("NOPASSWD:")[1] and "systemd-run" not in SUDOERS.split("NOPASSWD:")[1]


def test_the_live_helper_is_root_only_takes_one_checked_argument():
    assert 'id -u' in HELPER and '[ $# -eq 1 ]' in HELPER
    assert '^[A-Za-z0-9_-]{1,40}$' in HELPER


def test_it_refuses_without_a_recent_passing_self_test():
    assert "check-token" in HELPER and "--preflight-max-age" in HELPER
    assert re.search(r"^MAX_AGE=600$", HELPER, re.M)                           # 10 minutes
    assert HELPER.index("check-token") < HELPER.index("lejet-sudo-install-urls") < HELPER.index("lejet-sudo-batch")
    assert "exit 3" in HELPER                                                  # host reacts by running the full self-test


def test_it_never_overlaps_the_self_test_and_handles_one_link():
    assert "flock -s" in HELPER and "/run/lejet-gate.lock" in HELPER
    assert "flock -x /run/lejet-gate.lock" in SERVICE and "/usr/local/lib/lejet/preflight.sh" in SERVICE
    assert '-eq 1 ]' in HELPER and "one link per call" in HELPER


def test_every_step_is_done_by_the_existing_limited_helpers():
    for sub in ("lejet-sudo-install-urls", "lejet-sudo-batch", "lejet-sudo-shots", "lejet-sudo-export"):
        assert f"/usr/local/sbin/{sub}" in HELPER
    code = "\n".join(l for l in HELPER.splitlines() if not l.lstrip().startswith("#"))
    assert "systemd-run" not in code and "sudo " not in code.replace("through sudo", "").replace("/usr/local/sbin/lejet-sudo-", "")


def test_a_failed_self_test_deletes_the_token_so_it_blocks_live_runs():
    assert 'sudo rm -f /run/lejet-preflight-ok' in PREFLIGHT
    assert "if [ $FAIL -eq 0 ]" in PREFLIGHT


def test_the_gate_repeats_often_enough_for_the_ten_minute_limit():
    assert "OnUnitInactiveSec=4min" in TIMER and "OnActiveSec=" in TIMER
    assert "Type=oneshot" in SERVICE and "TimeoutStartSec=" in SERVICE
    assert "timers.target" in TIMER


def test_the_installer_installs_the_helper_the_units_and_enables_the_timer():
    assert "for h in preflight install-urls batch export shots live; do" in INSTALL
    assert "systemd/lejet-gate.service" in INSTALL and "systemd/lejet-gate.timer" in INSTALL
    assert "enable --now lejet-gate.timer" in INSTALL and "daemon-reload" in INSTALL
    assert "visudo -cf" in INSTALL                                             # the sudo rule is still validated first


def test_the_bundle_carries_the_new_files():
    for rel in ("infra/analysis/helpers/lejet-sudo-live.sh", "infra/analysis/systemd/lejet-gate.service",
                "infra/analysis/systemd/lejet-gate.timer"):
        assert rel in BUNDLE_PY and (ROOT / rel).exists()


# ---------------------------------------------------------------- the resident worker
LIVE_UNIT = read("systemd/lejet-live.service")
UNIT_CODE = "\n".join(l for l in LIVE_UNIT.splitlines() if l.strip() and not l.lstrip().startswith("#"))


def test_the_worker_runs_unprivileged_with_the_limits_of_the_per_run_units():
    for line in ("User=fetcher", "Group=fetcher", "PrivateTmp=yes", "ProtectSystem=strict", "ProtectHome=yes",
                 "ReadWritePaths=/var/lib/lejet", "MemorySwapMax=0", "MemoryMax=3500M", "TasksMax=2048",
                 "RuntimeMaxSec=3600", "Restart=always"):
        assert line in UNIT_CODE, line
    assert "NoNewPrivileges" not in UNIT_CODE            # as for lejet-sudo-shots: it would stop Chromium's AppArmor profile
    assert "/opt/lejet/venv/bin/python -I -m lejet.live_worker" in UNIT_CODE
    assert "no-sandbox" not in LIVE_UNIT.lower()
    roots = [l for l in UNIT_CODE.splitlines() if l.startswith("ExecStartPre=+")]
    assert roots and all("/usr/bin/install -d" in l for l in roots)                    # the only root step: create folders


def test_the_helper_hands_the_worker_only_the_batch_id():
    assert """printf '{"batch_id": "%s"}""" in HELPER and '"$BATCH_ID" > "$tmp"' in HELPER
    assert HELPER.count("$SPOOL/req/") == 1


def test_the_helper_falls_back_to_the_per_process_path_when_the_worker_is_not_ready():
    assert "worker_ready()" in HELPER and "systemctl is-active --quiet lejet-live.service" in HELPER
    assert "-le 20 ]" in HELPER                                                          # heartbeat at most 20 s old
    fast, slow = HELPER.split("else\n  echo \"== 4 fetching the link (per-process path")
    assert "lejet-sudo-batch" not in fast and "lejet-sudo-batch" in slow


def test_a_silent_worker_is_restarted_and_the_call_fails():
    assert "systemctl restart lejet-live.service" in HELPER and "exit 5" in HELPER


def test_results_made_by_the_worker_are_copied_only_if_they_are_plain_files():
    assert 'if [ -L "$f" ]; then echo "refusing $f: it is a symbolic link" >&2; exit 5; fi' in HELPER
    assert "missing result file" in HELPER
    assert "grep -q '\"ok\": true'" in HELPER
    assert HELPER.index("grep -q '\"ok\": true'") < HELPER.rindex("\n  hand_over\n")      # only after the worker said ok


def test_the_installer_installs_and_restarts_the_worker_without_failing_the_install():
    assert "systemd/lejet-live.service" in INSTALL
    assert "systemctl restart lejet-live.service" in INSTALL and "WARNING: lejet-live.service did not start" in INSTALL


def test_the_bundle_carries_the_worker_unit_and_its_tests():
    assert "infra/analysis/systemd/lejet-live.service" in BUNDLE_PY and "tests/test_live_worker.py" in BUNDLE_PY
    assert (ROOT / "src" / "lejet" / "live_worker.py").exists()
