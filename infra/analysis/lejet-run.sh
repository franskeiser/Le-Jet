#!/usr/bin/env bash
# Run ONE batch inside hard resource limits, as the unprivileged user `fetcher`
# (WEB_INSPECTION_PLAN.md section 11.10, 17.2).
#
#   ssh lejet-an '~/lejet-run.sh b0007'            # run the batch in /var/lib/lejet/in/urls.txt
#   ssh lejet-an '~/lejet-run.sh b0007 --seed 5'   # the only extra option is --seed N
#
# The real work is done by the root-owned helper /usr/local/sbin/lejet-sudo-batch (installed by
# install_bundle.sh; `ops` may run it without a password). It refuses to start unless the preflight
# passed recently, and runs the job as `fetcher` under systemd limits.
exec sudo /usr/local/sbin/lejet-sudo-batch "$@"
