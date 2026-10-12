#!/usr/bin/env bash
# Export one finished batch: <batch>.jsonl.gz + <batch>.manifest.json + <batch>.jsonl.gz.sha256
# (WEB_INSPECTION_PLAN.md sections 12.4, 15.2, 15.3). Run on the ANALYSIS VM:
#
#   ssh lejet-an '~/lejet-export.sh b0007'
#
# The work is done by the root-owned helper /usr/local/sbin/lejet-sudo-export (installed by
# install_bundle.sh; `ops` may run it without a password). Provenance for the manifest comes from
# /etc/lejet/provenance.env.
exec sudo /usr/local/sbin/lejet-sudo-export "$@"
