# Inspector runbook: from a URL list to validated data

Everything here is a command you run on **your laptop** (PowerShell) unless it says "on the VM".
`ssh lejet-gw` / `ssh lejet-an` are the shortcuts from `~/.ssh/config`; each asks for your key
passphrase (twice for `lejet-an`, once per hop).

## 0. Once: build and check the offline bundle (host, no VM needed)
```
python scripts\build_vm_bundle.py        # downloads wheels, builds nostril, writes the hashes
python scripts\verify_vm_bundle.py       # must end with "BUNDLE OK"
```

## 1. Start the lab (every session)
1. Restore both snapshots, start the gateway, wait 40 s, start the analysis VM, wait 40 s
   (plan §17.1; `VBoxManage snapshot ... restore`, `startvm ... --type headless`).
2. Fix both clocks from the host (plan §17.1, needs the scoped sudo rules, plan item S14):
   ```
   $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
   ssh lejet-gw "sudo date -u -s @$now"; ssh lejet-gw "sudo systemctl restart wg-quick@wg0"; ssh lejet-gw "sudo systemctl restart chrony"
   ssh lejet-an "sudo date -u -s @$now"; ssh lejet-an "sudo systemctl restart chrony"
   ```
3. `ssh lejet-gw "sudo wg show | grep handshake"` must show a recent handshake.

## 2. Once per snapshot: install the inspector on the analysis VM
```
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_to_vm.ps1
```
It verifies the bundle, copies it, installs the pinned wheels offline, copies the code to `/opt/lejet`,
fetches the RDAP bootstrap file, and runs the test suite **inside the VM** with `python -I`. Take the
`an-configured-clean` snapshot again afterwards (power off first) so every batch starts with it installed.

## 3. Preflight (on the VM, every batch)
```
ssh -t lejet-an "~/preflight.sh --record"    # first time only: records the firewall hash
ssh -t lejet-an "~/preflight.sh"             # must end with 0 FAIL; writes the token
```

## 4. Build the push file and push it
```
python scripts\make_push_file.py D:\LeJetData\input\urls.txt --out D:\LeJetData\input\urls_push.txt
powershell -ExecutionPolicy Bypass -File .\scripts\push_urls.ps1 -File D:\LeJetData\input\urls_push.txt
```
Only ever look at `urls_push.txt.defanged.txt`. Never open `urls_push.txt` in a browser.

## 5. Run, export (VM)
```
ssh -t lejet-an "./lejet-run.sh b0001"
ssh -t lejet-an "LEJET_VPN_PROVIDER=Mullvad LEJET_VPN_EXIT_COUNTRY=PH ./lejet-export.sh b0001"
```
`lejet-run.sh` refuses to start without a fresh preflight token. A run can be stopped and started
again: finished URLs are skipped.

## 6. Pull and validate (host)
```
powershell -ExecutionPolicy Bypass -File .\scripts\pull_results.ps1 -Batch b0001 -PushFile D:\LeJetData\input\urls_push.txt
```
It copies three named files into a new quarantine folder, checks the sha256, runs the strict validator and
writes `D:\LeJetData\clean\b0001.clean.jsonl` (+ CSV) only if everything passes.

## 7. Afterwards
Pull the gateway logs and captures **before** reverting (plan §21.3), then power off and restore the snapshots.
