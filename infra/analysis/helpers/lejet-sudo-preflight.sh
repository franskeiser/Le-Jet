#!/usr/bin/env bash
# Passwordless-sudo helper 1/4: run the Level 2 isolation preflight as root.
# Installed by install_bundle.sh as /usr/local/sbin/lejet-sudo-preflight (root-owned, mode 755).
# It runs the ROOT-OWNED copy /usr/local/lib/lejet/preflight.sh, never a file that `ops` can edit,
# and it takes NO arguments (so `--record`, which changes the saved baseline, still needs the password).
set -u

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
[ $# -eq 0 ] || { echo "this helper takes no arguments" >&2; exit 2; }
[ -x /usr/local/lib/lejet/preflight.sh ] || { echo "preflight.sh is not installed" >&2; exit 2; }
# Take the same exclusive lock as the background gate (lejet-gate.service), so a manual preflight and the gate's own
# self-test never run at the same time: one run's temporary probe sockets would show up as "listeners" in the other's
# test 15 (a false FAIL seen on 2026-10-14). Waits up to 5 minutes (a live run holds the lock shared); on timeout it
# exits non-zero and the caller refuses to fetch: it fails closed.
exec flock -x -w 300 /run/lejet-gate.lock /usr/local/lib/lejet/preflight.sh
