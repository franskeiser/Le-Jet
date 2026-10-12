# Run log: batch <batch_id>

Copy this file for every batch (for example `run_log_b0007.md`) and fill it in.
Tick a box only with evidence. Source: `WEB_INSPECTION_PLAN.md` §17, §21.2, §21.3.

## Batch info
- Batch id:
- Date / time (UTC):
- Input file sha256:
- Number of input URLs:
- Git commit:
- VirtualBox version / guest kernel / Python:
- Snapshot names used (gateway / analysis):
- VPN provider / exit country:

## Before the batch (§21.2)
- [ ] Batch from the frozen sample list, at most 1,000 URLs, canaries added
- [ ] Host: Windows Update / Defender ok, offline backup recent, Docker/WSL stopped
- [ ] VMs reverted to clean snapshots, gateway started first
- [ ] Clocks fixed from the host, WireGuard restarted
- [ ] VPN handshake recent, exit IP differs from home IP (exit IP: )
- [ ] Upstream DNS non-filtering
- [ ] `preflight.sh` green, firewall hash unchanged
- [ ] `urls.txt` pushed, packet capture running

## Isolation tests (Level 2 results, paste output or note pass/fail)
| # | Test | Result |
|---|---|---|
| 1 to 26 | see plan §16 Level 2 | |

## After the batch (§21.3)
- [ ] Results exported (JSONL.gz + manifest + sha256), pulled into quarantine, hash verified
- [ ] Gateway logs pulled before the revert; non-80/443 or private attempts counted (expected 0):
- [ ] Validator passed; row count = input count
- [ ] Canary rows correct
- [ ] VMs powered off and reverted

## Statistics
- Dead rate:
- Failure reasons:
- Challenge/parked page rate:
- Free-hosting share:
- RDAP missing rate by TLD:
- `port_blocked` count:
- Median / p95 latency:

## Anomalies and abort triggers (§18.2)
- Anything unusual:
- Abort criteria tripped? yes / no
