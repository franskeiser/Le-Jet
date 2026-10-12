<#
.SYNOPSIS
  Install or update the inspector on the analysis VM from the OFFLINE, hash-pinned bundle
  (WEB_INSPECTION_PLAN.md sections 11.6, 11.7).

.DESCRIPTION
  1. verifies the bundle on the host (scripts\verify_vm_bundle.py);
  2. copies it to the VM with ONE scp (you type the key passphrase for each hop);
  3. on the VM: checks the sha256, unpacks, runs infra\analysis\install_bundle.sh as root
     (venv with the pinned wheels, code in /opt/lejet, IANA RDAP bootstrap), copies
     preflight.sh / lejet-run.sh / lejet-export.sh to the home folder;
  4. runs the test suite INSIDE the VM (python -I) unless -SkipTests.

  Run it with the gateway AND the analysis VM started (gateway first), clocks fixed and the VPN up:
  the install fetches the RDAP bootstrap file from data.iana.org through the gateway.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\deploy_to_vm.ps1
  powershell -ExecutionPolicy Bypass -File .\scripts\deploy_to_vm.ps1 -SkipTests
  powershell -ExecutionPolicy Bypass -File .\scripts\deploy_to_vm.ps1 -Browser     # + Chromium for screenshots
#>
[CmdletBinding()]
param(
  [string]$Bundle = "D:\LeJetData\bundle\lejet-bundle.tar.gz",
  [switch]$SkipTests,
  [switch]$Browser          # also install headless Chromium for the screenshot stage (a few minutes, ~1 GB)
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $Bundle)) {
  throw "Bundle not found: $Bundle. Build it first:  python scripts\build_vm_bundle.py"
}

$py = Join-Path $repo ".venv\Scripts\python.exe"
if (Test-Path $py) {
  Write-Host "== verifying the bundle on the host"
  & $py -B (Join-Path $PSScriptRoot "verify_vm_bundle.py") --bundle $Bundle
  if ($LASTEXITCODE -ne 0) { throw "Bundle verification failed; not deploying." }
} else {
  Write-Warning "No .venv found; skipping the host-side bundle verification."
}

$hash = (Get-FileHash $Bundle -Algorithm SHA256).Hash.ToLower()
Write-Host "== bundle sha256: $hash"

Write-Host "== copying the bundle to lejet-an (key passphrase: gateway hop, then analysis hop)"
scp $Bundle lejet-an:/tmp/lejet-bundle.tar.gz
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

Write-Host "== installing on lejet-an (key passphrase twice, then the ops password for sudo)"
$remote = "set -e; cd /tmp; echo '$hash  lejet-bundle.tar.gz' | sha256sum -c -; " +
          "rm -rf lejet-bundle; tar -xzf lejet-bundle.tar.gz; " +
          "sudo bash lejet-bundle/install_bundle.sh /tmp/lejet-bundle test; " +
          "cp lejet-bundle/infra/analysis/preflight.sh lejet-bundle/infra/analysis/lejet-run.sh lejet-bundle/infra/analysis/lejet-export.sh lejet-bundle/infra/analysis/install_browser.sh ~/; " +
          "chmod +x ~/preflight.sh ~/lejet-run.sh ~/lejet-export.sh ~/install_browser.sh; ls -l ~/*.sh"
ssh -t lejet-an $remote
if ($LASTEXITCODE -ne 0) { throw "installation on the VM failed" }

if ($Browser) {
  Write-Host "== installing headless Chromium on lejet-an (apt libraries + browser download + AppArmor + self-test)"
  Write-Host "   (key passphrase twice via the agent, then the ops password for sudo; this takes a few minutes)"
  ssh -t lejet-an "sudo bash ~/install_browser.sh"
  if ($LASTEXITCODE -ne 0) { throw "browser installation failed (see the message above)" }
}

if (-not $SkipTests) {
  Write-Host "== running the test suite inside the VM (python -I)"
  # Exit code 255 = ssh itself could not connect (seen right after the install session closes):
  # wait a few seconds and try again. Any other non-zero code is a real test failure.
  for ($try = 1; $try -le 3; $try++) {
    ssh -t lejet-an "cd /opt/lejet && /opt/lejet/venv/bin/python -I -m pytest -p no:cacheprovider -q tests"
    if ($LASTEXITCODE -ne 255) { break }
    Write-Warning "ssh connection failed (attempt $try of 3); waiting 10 seconds"
    Start-Sleep -Seconds 10
  }
  if ($LASTEXITCODE -ne 0) { throw "tests failed inside the VM (exit code $LASTEXITCODE)" }
}

Write-Host ""
Write-Host "Done. Next on the VM: ~/preflight.sh --record (once), then push URLs and ~/lejet-run.sh <batch>."
