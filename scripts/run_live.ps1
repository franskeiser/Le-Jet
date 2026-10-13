<#
.SYNOPSIS
  Inspect ONE link in the isolated lab as fast as the design allows: the same fetch, screenshot, pull and strict validation as
  run_batch.ps1, with about a dozen ssh round trips folded into one call to the VM.

.DESCRIPTION
  What is different from run_batch.ps1 (and why it is faster):
    * the VM runs ONE helper (lejet-sudo-live) that does install, fetch, screenshot and export in a single ssh call;
    * the 26-check isolation self-test is not repeated per link: it runs in the background on the VM every ~5 minutes
      (lejet-gate.timer) and the helper refuses to start unless a passing result is at most 10 minutes old;
    * the clocks are only re-synchronised when they have drifted by more than 3 seconds;
    * both result files are pulled with one scp and validated with the same validators as before.
  Unchanged: the host pulls (the VM never connects back), only named files are copied into a NEW quarantine folder, the sha256
  is checked, and validate_output.py / validate_shots.py decide what is accepted.

  Exit codes: 0 done; 4 the live helper is not installed on the VM (run_batch.ps1 is the fallback); anything else = failed.
  If the background gate is not fresh (exit 3 from the helper, e.g. right after a VM restore) the full self-test is run once
  and the call is repeated, so a fresh VM still works, only slower.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$UrlFile,
  [Parameter(Mandatory = $true)][string]$Batch,
  [string]$InputDir = "D:\LeJetData\input",
  [string]$Quarantine = "D:\LeJetData\quarantine",
  [string]$CleanDir = "D:\LeJetData\clean"
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
  $old = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try { return @(& $call 2>&1 | ForEach-Object { "$_" }) }
  finally { $ErrorActionPreference = $old }
}

function VmEpoch([string]$name) {
  $o = Native { ssh @sshOpts $name "date +%s" }
  if ($LASTEXITCODE -ne 0) { return $null }
  $last = ($o | Select-Object -Last 1)
  if ($last -match '^\d{9,11}$') { return [int64]$last } else { return $null }
}

if ($Batch -notmatch '^[A-Za-z0-9_-]{1,40}$') { Fail "Bad batch id (letters, digits, - and _ only)." }
if (-not (Test-Path $UrlFile)) { Fail "URL file not found: $UrlFile" }

Step 0 "checks"
$running = & $vb list runningvms
if (-not ($running -match '"lejet-gw"')) { Fail "lejet-gw is not running. Start it first:  VBoxManage startvm lejet-gw --type headless" }
if (-not ($running -match '"lejet-an"')) { Fail "lejet-an is not running. Start it first:  VBoxManage startvm lejet-an --type headless (then wait a minute)" }
$null = Native { ssh-add -l }
if ($LASTEXITCODE -ne 0) { Fail "No SSH key is loaded in ssh-agent. Run once:  ssh-add `$HOME\.ssh\lejet_ed25519" }
# one call: the analysis VM answers, tells its clock, and says whether the live helper is installed
$probe = Native { ssh @sshOpts -o ConnectTimeout=10 lejet-an "date +%s; test -x /usr/local/sbin/lejet-sudo-live && echo live-helper-ok" }
if ($LASTEXITCODE -ne 0) { Fail "lejet-an does not answer on SSH." }
if (-not ($probe -contains "live-helper-ok")) { Write-Host "the live helper is not installed on the VM"; exit 4 }
$anEpoch = [int64]($probe | Select-Object -First 1)
$gwEpoch = VmEpoch "lejet-gw"
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$drift = [math]::Max([math]::Abs($anEpoch - $now), $(if ($null -eq $gwEpoch) { 999 } else { [math]::Abs($gwEpoch - $now) }))
if ($drift -le 3) {
  Write-Host "   both VM clocks agree with this PC (within 3 s): no sync needed"
} else {
  # a VM that was paused (laptop sleep, snapshot restore) wakes up with an OLD clock: set both, gateway first
  Write-Host "   a VM clock is off by $drift s: syncing both clocks from this PC"
  foreach ($name in @("lejet-gw", "lejet-an")) {
    $epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $out = Native { ssh @sshOpts $name "sudo -n /usr/bin/date -u -s @$epoch >/dev/null && sudo -n /usr/bin/systemctl restart chrony && date -u" }
    if ($LASTEXITCODE -ne 0) { Fail "could not set the clock on ${name}: $($out -join ' ')" }
  }
  Start-Sleep -Seconds 8
}

Step 1 "build the push file"
New-Item -ItemType Directory -Force -Path $InputDir | Out-Null
$push = Join-Path $InputDir "${Batch}_push.txt"
& $py -B (Join-Path $PSScriptRoot "make_push_file.py") $UrlFile --out $push
if ($LASTEXITCODE -ne 0) { Fail "make_push_file.py failed" }
$localHash = (Get-FileHash $push -Algorithm SHA256).Hash.ToLower()
$localLines = ([IO.File]::ReadAllLines($push) | Where-Object { $_.Trim() -ne "" }).Count
if ($localLines -ne 1) { Fail "run_live.ps1 takes exactly one link (found $localLines usable)" }

Step 2 "push to the VM"
$null = Native { scp @sshOpts $push "lejet-an:incoming/urls.txt" }
if ($LASTEXITCODE -ne 0) { Fail "scp failed" }

function RunLive {
  # streams the helper's output line by line (its "== n ..." lines drive the progress shown to the person); returns exit code + lines
  $lines = New-Object System.Collections.Generic.List[string]
  $old = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & ssh @sshOpts lejet-an "sudo -n /usr/local/sbin/lejet-sudo-live $Batch" 2>&1 | ForEach-Object {
      $t = "$_"
      $lines.Add($t)
      Write-Host "   $t"
    }
  } finally { $ErrorActionPreference = $old }
  return @{ Code = $LASTEXITCODE; Lines = $lines }
}

$res = RunLive
if ($res.Code -eq 3) {
  Step 3 "isolation self-test (the background gate was not fresh: running the full self-test once)"
  $pf = Native { ssh @sshOpts lejet-an "sudo -n /usr/local/sbin/lejet-sudo-preflight" }
  if (-not ($pf -match 'FAIL 0')) { Fail "preflight did not finish with FAIL 0: nothing was fetched" }
  $res = RunLive
}
if ($res.Code -ne 0) { Fail "the live run on the VM failed (exit $($res.Code))" }
$remoteHash = (($res.Lines | Where-Object { $_ -match '^[0-9a-f]{64}\s' }) | Select-Object -First 1) -split '\s+' | Select-Object -First 1
if ($remoteHash -ne $localHash) { Fail "sha256 on the VM ($remoteHash) differs from the local file ($localHash)" }

Step 5 "export, pull and validate"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dir = Join-Path $Quarantine "$Batch-$stamp"
New-Item -ItemType Directory -Path $dir | Out-Null
$names = @(".jsonl.gz", ".jsonl.gz.sha256", ".manifest.json", ".shots.tar", ".shots.tar.sha256")
$sources = $names | ForEach-Object { "lejet-an:export/$Batch$_" }
$null = Native { scp @sshOpts @sources $dir }          # the shots files are optional; the three result files are checked below
foreach ($ext in @(".jsonl.gz", ".jsonl.gz.sha256", ".manifest.json")) {
  if (-not (Test-Path (Join-Path $dir "$Batch$ext"))) { Fail "scp of $Batch$ext failed" }
}
$jsonl = Join-Path $dir "$Batch.jsonl.gz"
$expected = ([IO.File]::ReadAllText("$jsonl.sha256").Trim() -split '\s+')[0].ToLower()
if ($expected -ne (Get-FileHash $jsonl -Algorithm SHA256).Hash.ToLower()) { Fail "sha256 mismatch: the file was damaged or altered in transit." }
& $py -I (Join-Path $PSScriptRoot "validate_output.py") $jsonl --clean-dir $CleanDir --csv (Join-Path $CleanDir "$Batch.csv") --input $push
if ($LASTEXITCODE -ne 0) { Fail "VALIDATION FAILED: nothing was written to $CleanDir. Keep the quarantine folder as evidence." }
$tar = Join-Path $dir "$Batch.shots.tar"
if ((Test-Path $tar) -and (Test-Path "$tar.sha256")) {
  & $py -I (Join-Path $PSScriptRoot "validate_shots.py") $tar --clean-dir $CleanDir --clean-jsonl (Join-Path $CleanDir "$Batch.clean.jsonl") --batch $Batch
  if ($LASTEXITCODE -ne 0) { Write-Host "SHOTS VALIDATION FAILED: no screenshot was accepted (the verdict does not need it)" }
} else {
  Write-Host "no screenshot archive came back (the page may not have loaded)"
}
Write-Host ""
Write-Host "DONE. Batch ${Batch}: validated data in $CleanDir\$Batch.clean.jsonl"
