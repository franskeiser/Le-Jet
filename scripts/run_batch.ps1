<#
.SYNOPSIS
  Run one batch of URLs through the whole Le-Jet lab, start to finish, with no prompts:
  build the push file, push it, preflight, run, export, pull, validate, summarize.

.DESCRIPTION
  Needs, before you start (nothing in this script ever sees or stores a secret):
    * both VMs running (lejet-gw first, then lejet-an);
    * your SSH key loaded in the Windows ssh-agent (ssh-add, once per session): every ssh/scp call
      here runs with BatchMode, so it FAILS instead of asking for a passphrase;
    * the bundle deployed once with scripts\deploy_to_vm.ps1 (this installs the root-owned helpers and
      the sudoers rule that let `ops` run exactly four fixed commands without a password).

  Steps (each one stops the script if it fails):
    1  build  the push file on the host (host-computed url_id, refang, dedupe)
    2  push   copy it to the VM; a root-owned helper checks every line before installing it
    3  preflight  the 26-test isolation check; the batch refuses to start without a fresh pass
    4  run    the batch, as the unprivileged user `fetcher`, under systemd limits
    5  export + pull  three named files into a NEW quarantine folder, sha256 check, strict validator
    6  summary  a plain-language report (per-TLD registration / certificate coverage)
    7  -Reset (optional)  power the analysis VM off, restore the clean snapshot, start it again and fix its clock

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\run_batch.ps1 -UrlFile D:\LeJetData\input\pilot_L1b.txt -Batch b0002
  powershell -ExecutionPolicy Bypass -File .\scripts\run_batch.ps1 -UrlFile D:\LeJetData\input\list.txt -Batch b0003 -Reset
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$UrlFile,      # one URL per line (may be defanged)
  [Parameter(Mandatory = $true)][string]$Batch,        # letters, digits, - and _
  [int]$Seed = 0,                                      # 0 = the default seed
  [switch]$Screenshots,                                # also take real screenshots (needs the browser installed)
  [int]$ShotLimit = 0,                                 # at most this many screenshots (0 = all fetched pages)
  [switch]$Reset,                                      # restore the clean snapshot afterwards
  [string]$Snapshot = "an-configured-clean",
  [string]$InputDir = "D:\LeJetData\input",
  [string]$ReportDir = "D:\LeJetData\reports"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$vb = "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$sshOpts = @("-o", "BatchMode=yes")

function Step($n, $text) { Write-Host ""; Write-Host "== $n  $text" -ForegroundColor Cyan }
function Fail($text) { throw $text }

function Native([scriptblock]$call) {
  # runs a native command with stderr merged into the output, without PowerShell 5.1 turning
  # stderr text into a terminating error; returns the output as strings (exit code: $LASTEXITCODE)
  $old = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try { return @(& $call 2>&1 | ForEach-Object { "$_" }) }
  finally { $ErrorActionPreference = $old }
}

function Vm([string]$command) {
  # runs one command on lejet-an; returns the output lines, throws on a non-zero exit code
  $out = Native { ssh @sshOpts lejet-an $command }
  $code = $LASTEXITCODE
  $out | ForEach-Object { Write-Host "   $_" }
  if ($code -ne 0) { Fail "command on the VM failed (exit $code): $command" }
  return $out
}

# ---------------------------------------------------------------------------------------------
if ($Batch -notmatch '^[A-Za-z0-9_-]{1,40}$') { Fail "Bad batch id (letters, digits, - and _ only)." }
if (-not (Test-Path $UrlFile)) { Fail "URL file not found: $UrlFile" }

Step 0 "checks"
$running = & $vb list runningvms
if (-not ($running -match '"lejet-gw"')) { Fail "lejet-gw is not running. Start it first:  VBoxManage startvm lejet-gw --type headless" }
if (-not ($running -match '"lejet-an"')) { Fail "lejet-an is not running. Start it first:  VBoxManage startvm lejet-an --type headless (then wait a minute)" }
$null = Native { ssh-add -l }
if ($LASTEXITCODE -ne 0) { Fail "No SSH key is loaded in ssh-agent. Run once:  ssh-add `$HOME\.ssh\lejet_ed25519" }
$reachable = $false
for ($i = 0; $i -lt 36; $i++) {                # a freshly started VM needs about a minute
  $null = Native { ssh @sshOpts -o ConnectTimeout=4 lejet-an "true" }
  if ($LASTEXITCODE -eq 0) { $reachable = $true; break }
  if ($i -eq 0) { Write-Host "   waiting for lejet-an to answer on SSH (up to 3 minutes)" }
  Start-Sleep -Seconds 5
}
if (-not $reachable) { Fail "lejet-an does not answer on SSH. Check its console: VBoxManage controlvm lejet-an screenshotpng D:\LeJetData\an.png" }
# A VM that was paused (laptop sleep, snapshot restore) wakes up with an OLD clock and has no Guest
# Additions to fix it. A stale gateway clock breaks the encrypted DNS upstream (certificates look "not
# yet valid"), which showed up as dns_timeout on most URLs. So: set both clocks from this PC, gateway
# first (the analysis VM takes its time from the gateway), at the start of EVERY run.
Write-Host "   syncing both VM clocks from this PC"
foreach ($name in @("lejet-gw", "lejet-an")) {
  $epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  $out = Native { ssh @sshOpts $name "sudo -n /usr/bin/date -u -s @$epoch >/dev/null && sudo -n /usr/bin/systemctl restart chrony && date -u" }
  if ($LASTEXITCODE -ne 0) { Fail "could not set the clock on ${name}: $($out -join ' ')" }
  Write-Host "   $name clock: $($out | Select-Object -Last 1)"
}
Start-Sleep -Seconds 8                          # let chrony on the analysis VM pick up the gateway again
$null = Vm "test -x /usr/local/sbin/lejet-sudo-batch && sudo -n -l /usr/local/sbin/lejet-sudo-preflight >/dev/null && echo 'helpers installed, passwordless rule active'"

Step 1 "build the push file"
New-Item -ItemType Directory -Force -Path $InputDir | Out-Null
$push = Join-Path $InputDir "${Batch}_push.txt"
& $py -B (Join-Path $PSScriptRoot "make_push_file.py") $UrlFile --out $push
if ($LASTEXITCODE -ne 0) { Fail "make_push_file.py failed" }
$localHash = (Get-FileHash $push -Algorithm SHA256).Hash.ToLower()
$localLines = ([IO.File]::ReadAllLines($push) | Where-Object { $_.Trim() -ne "" }).Count
Write-Host "   local: $localLines URLs, sha256 $localHash"
if ($localLines -lt 1) { Fail "no usable URLs in $UrlFile" }

Step 2 "push to the VM"
$null = Vm "mkdir -p ~/incoming"
$null = Native { scp @sshOpts $push "lejet-an:incoming/urls.txt" }
if ($LASTEXITCODE -ne 0) { Fail "scp failed" }
$res = Vm "sudo -n /usr/local/sbin/lejet-sudo-install-urls"
$remoteHash = ((($res | Where-Object { $_ -match '^[0-9a-f]{64}\s' }) | Select-Object -First 1) -split '\s+')[0]
if ($remoteHash -ne $localHash) { Fail "sha256 on the VM ($remoteHash) differs from the local file ($localHash)" }
Write-Host "   sha256 matches on both sides" -ForegroundColor Green

Step 3 "preflight (isolation tests)"
$pf = Vm "sudo -n /usr/local/sbin/lejet-sudo-preflight"
if (-not ($pf -match 'FAIL 0')) { Fail "preflight did not finish with FAIL 0: nothing was fetched" }

Step 4 "run the batch $Batch"
$cmd = "sudo -n /usr/local/sbin/lejet-sudo-batch $Batch"
if ($Seed -gt 0) { $cmd += " --seed $Seed" }
$run = Vm $cmd
$summary = $run | Where-Object { $_ -match '^\{.*"status_counts"' } | Select-Object -Last 1
if (-not $summary) { Fail "the run did not print its summary line" }

if ($Screenshots) {
  Step "4b" "real screenshots of the fetched pages (headless Chromium, sandbox on)"
  $null = Vm "test -s /opt/lejet/browsers/SHA256SUMS && echo 'browser installed'"
  $shotCmd = "sudo -n /usr/local/sbin/lejet-sudo-shots $Batch"
  if ($ShotLimit -gt 0) { $shotCmd += " --limit $ShotLimit" }
  $shotRun = Vm $shotCmd
}

Step 5 "export, pull and validate"
$null = Vm "sudo -n /usr/local/sbin/lejet-sudo-export $Batch"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "pull_results.ps1") -Batch $Batch -PushFile $push
if ($LASTEXITCODE -ne 0) { Fail "pull / validation failed (see the message above)" }

if ($Screenshots) {
  Step "5b" "pull, validate and gallery for the screenshots"
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "pull_shots.ps1") -Batch $Batch
  if ($LASTEXITCODE -ne 0) { Fail "screenshot pull / validation failed (see the message above)" }
}

Step 6 "summary"
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$clean = "D:\LeJetData\clean\$Batch.clean.jsonl"
& $py -I -B (Join-Path $PSScriptRoot "summarize_batch.py") $clean --out (Join-Path $ReportDir "$Batch.md")

if ($Reset) {
  Step 7 "reset the analysis VM to the clean snapshot '$Snapshot'"
  $null = Native { ssh @sshOpts lejet-an "sudo -n /usr/bin/systemctl poweroff" }    # clean shutdown
  function VmState { ((& $vb showvminfo lejet-an --machinereadable) | Where-Object { $_ -match '^VMState=' }) -replace '^VMState="?([^"]*)"?$', '$1' }
  for ($i = 0; $i -lt 45; $i++) {
    Start-Sleep -Seconds 2
    if ((VmState) -eq "poweroff") { break }          # fully off, not merely "stopping"
  }
  if ((VmState) -ne "poweroff") {
    & $vb controlvm lejet-an poweroff
    for ($i = 0; $i -lt 20 -and (VmState) -ne "poweroff"; $i++) { Start-Sleep -Seconds 2 }
  }
  if ((VmState) -ne "poweroff") { Fail "the analysis VM did not power off (state: $(VmState))" }
  & $vb snapshot lejet-an restore $Snapshot
  if ($LASTEXITCODE -ne 0) { Fail "snapshot restore failed" }
  & $vb startvm lejet-an --type headless | Out-Null
  Write-Host "   waiting for the VM to answer on SSH (up to 3 minutes)"
  $up = $false
  for ($i = 0; $i -lt 36; $i++) {
    Start-Sleep -Seconds 5
    $null = Native { ssh @sshOpts -o ConnectTimeout=4 lejet-an "true" }
    if ($LASTEXITCODE -eq 0) { $up = $true; break }
  }
  if (-not $up) { Fail "the VM did not come back; check its console (VBoxManage controlvm lejet-an screenshotpng)" }
  # a restored snapshot has a stale clock: set it from this PC, then let chrony take over
  $epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  $null = Vm "sudo -n /usr/bin/date -u -s @$epoch && sudo -n /usr/bin/systemctl restart chrony && date -u"
  Write-Host "   VM is clean again and its clock is set" -ForegroundColor Green
}

Write-Host ""
Write-Host "DONE. Batch ${Batch}: $summary" -ForegroundColor Green
Write-Host "Validated data : $clean"
Write-Host "Report         : $(Join-Path $ReportDir "$Batch.md")"
if ($Screenshots) { Write-Host "Screenshots    : D:\LeJetData\clean\shots\$Batch\index.html" }
