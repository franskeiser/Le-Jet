#!/usr/bin/env bash
# Passwordless-sudo helper 1/4: run the Level 2 isolation preflight as root.
# Installed by install_bundle.sh as /usr/local/sbin/lejet-sudo-preflight (root-owned, mode 755).
# It runs the ROOT-OWNED copy /usr/local/lib/lejet/preflight.sh, never a file that `ops` can edit,
# and it takes NO arguments (so `--record`, which changes the saved baseline, still needs the password).
set -u

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
[ $# -eq 0 ] || { echo "this helper takes no arguments" >&2; exit 2; }
[ -x /usr/local/lib/lejet/preflight.sh ] || { echo "preflight.sh is not installed" >&2; exit 2; }
exec /usr/local/lib/lejet/preflight.sh
