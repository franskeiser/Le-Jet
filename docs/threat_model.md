# Threat model: Le-Jet web inspection

Short version of `WEB_INSPECTION_PLAN.md` §2, for the thesis ethics and safeguards section.
The plan is the source of truth; update this file if the plan changes.

## What we protect (priority order)
1. The laptop and its files (thesis, repos, OneDrive, saved sessions, credentials)
2. The home network (router, other devices)
3. Identity and home IP address
4. Integrity of the dataset (no poisoned feature data)
5. The VirtualBox install and VM images

## Who or what can hurt us
| Adversary | What it can do |
|---|---|
| Malicious web page | Run scripts, serve exploits, redirect, force downloads |
| Malicious server | Send hostile HTTP/TLS, huge bodies, stalls |
| Malicious strings in features | CSV formula injection, terminal escapes, log injection |
| Hostile DNS answers | Point a domain at 127.0.0.1 or the LAN (DNS rebinding) |
| Compromised analysis VM | Scan, mine crypto, try to reach the host or LAN |
| Anyone monitoring the site | Sees our IP |
| Ourselves (human error) | Opening a live URL on the host, wrong network mode |

## Main controls (plan §2.3, attack paths A1 to A15)
- Two VMs: a gateway (internet via VPN) and an analysis VM (internal network only, no direct internet).
- Gateway firewall: default drop, only TCP 80/443 out, all private ranges blocked and logged, kill switch.
- WireGuard VPN hides the home IP.
- v1 runs no browser: plain HTTP fetch and static HTML parse. Sizes capped, unprivileged user, resource limits.
- Only validated numbers and booleans leave the VM (features only, no raw pages).
- Snapshot revert after every batch; no shared folders, clipboard, USB, or Guest Additions.
- URLs stored defanged; never opened in the host browser.

## Residual risk (state honestly in the thesis)
- A browser 0-day plus a VirtualBox escape could reach the host. Rare, but not impossible.
- VPN exit IPs may be cloaked by phishing kits, which biases which pages we see.
- We can only reduce what a page can do; we cannot prove a page is benign.
