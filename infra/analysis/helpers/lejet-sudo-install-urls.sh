#!/usr/bin/env bash
# Passwordless-sudo helper 2/4: move the pushed URL file into the work area, after checking it.
# Installed as /usr/local/sbin/lejet-sudo-install-urls (root-owned, mode 755); takes NO arguments.
#
#   source      /home/ops/incoming/urls.txt   (the host copies the file there with scp)
#   destination /var/lib/lejet/in/urls.txt    (owner fetcher, mode 640)
#
# The file is copied once into a root-owned temporary file and checked THERE, so nothing can change
# between the check and the install. Every line must be exactly: url_id (16 hex) TAB url, where the
# url has no whitespace or control characters and is at most 2048 characters long.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run this through sudo" >&2; exit 2; }
[ $# -eq 0 ] || { echo "this helper takes no arguments" >&2; exit 2; }

SRC=/home/ops/incoming/urls.txt
DST=/var/lib/lejet/in/urls.txt
MAX_BYTES=5242880
MAX_LINES=200000

[ -f "$SRC" ] || { echo "no file at $SRC (copy the push file there first)" >&2; exit 2; }
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
head -c $((MAX_BYTES + 1)) "$SRC" > "$tmp"

[ "$(stat -c %s "$tmp")" -le "$MAX_BYTES" ] || { echo "file is larger than $MAX_BYTES bytes" >&2; exit 2; }
[ -s "$tmp" ] || { echo "file is empty" >&2; exit 2; }
[ -z "$(tail -c1 "$tmp")" ] || { echo "the last line has no newline" >&2; exit 2; }
n="$(wc -l < "$tmp")"
[ "$n" -ge 1 ] && [ "$n" -le "$MAX_LINES" ] || { echo "line count $n is out of range" >&2; exit 2; }
if [ -n "$(tr -d -c '\r' < "$tmp" | head -c1)" ]; then
  echo "the file contains CR characters (it must use LF line endings)" >&2
  exit 2
fi
if LC_ALL=C grep -qvE '^[0-9a-f]{16}'$'\t''[^[:space:][:cntrl:]]{1,2048}$' "$tmp"; then
  echo "bad line found (expected: url_id<TAB>url, no spaces or control characters)" >&2
  exit 2
fi
if [ -n "$(cut -f1 "$tmp" | sort | uniq -d | head -n1)" ]; then
  echo "duplicate url_id found" >&2
  exit 2
fi

install -o fetcher -g fetcher -m 640 "$tmp" "$DST"
echo "on VM: $n lines"
sha256sum "$DST"
