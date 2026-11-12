<#
.SYNOPSIS
  Inspect ONE link on the Azure screenshot lab (whole-page screenshots + a few pages of the same site) and bring the
  validated evidence back. Same trust rules as run_live.ps1; only the machine is different.

.DESCRIPTION
  PC -> VM:   the push file (one line, url_id<TAB>url) is copied with scp.
  On the VM:  ONE ssh call runs `sudo -n /usr/local/sbin/lejet-azure-live <batch>`: safety gate, fetch, screenshots, export.
  VM -> PC:   the result files are PULLED with scp (the VM never connects back), their sha256 is checked, and the SAME strict
              validators as the VirtualBox lab decide what is accepted (validate_output.py, validate_shots.py).
  Output lines "== n  text" drive the progress steps shown by the tool (src/serving/live_inspect.py).

  Settings (environment, all optional):
    LEJET_LAB_HOST   ssh host alias from ~/.ssh/config   (default: lejet-azure)

  Exit codes: 0 done; 3 the VM's safety gate refused; 4 the VM is busy; anything else = failed.
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
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$labHost = if ($env:LEJET_LAB_HOST) { $env:LEJET_LAB_HOST } else { "lejet-azure" }
if ($labHost -notmatch '^[A-Za-z0-9._-]{1,80}$') { throw "Bad LEJET_LAB_HOST" }
$sshOpts = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")

function Step($n, $text) { Write-Host ""; Write-Host "== $n  $text" -ForegroundColor Cyan }
function Fail($text) { throw $text }
function Native([scriptblock]$call) {
  $old = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try { return @(& $call 2>&1 | ForEach-Object { "$_" }) }
  finally { $ErrorActionPreference = $old }
}

if ($Batch -notmatch '^[A-Za-z0-9_-]{1,40}$') { Fail "Bad batch id (letters, digits, - and _ only)." }
if (-not (Test-Path $UrlFile)) { Fail "URL file not found: $UrlFile" }

Step 0 "checks"
$null = Native { ssh @sshOpts $labHost "true" }
if ($LASTEXITCODE -ne 0) { Fail "The Azure lab ($labHost) does not answer on SSH. Is the VM running (Azure portal > Start) and your IP allowed on port 22?" }

Step 1 "build the push file"
New-Item -ItemType Directory -Force -Path $InputDir | Out-Null
$push = Join-Path $InputDir "${Batch}_push.txt"
& $py -B (Join-Path $PSScriptRoot "make_push_file.py") $UrlFile --out $push
if ($LASTEXITCODE -ne 0) { Fail "make_push_file.py failed" }
$localHash = (Get-FileHash $push -Algorithm SHA256).Hash.ToLower()
$localLines = ([IO.File]::ReadAllLines($push) | Where-Object { $_.Trim() -ne "" }).Count
if ($localLines -ne 1) { Fail "run_live_azure.ps1 takes exactly one link (found $localLines usable)" }

Step 2 "push to the Azure lab"
$null = Native { scp @sshOpts $push "${labHost}:incoming/urls.txt" }
if ($LASTEXITCODE -ne 0) { Fail "scp failed" }

$lines = New-Object System.Collections.Generic.List[string]
$old = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  & ssh @sshOpts $labHost "sudo -n /usr/local/sbin/lejet-azure-live $Batch" 2>&1 | ForEach-Object {
    $t = "$_"
    $lines.Add($t)
    Write-Host "   $t"
  }
} finally { $ErrorActionPreference = $old }
$code = $LASTEXITCODE
if ($code -ne 0) { Write-Host "the lab run failed (exit $code)"; exit $(if ($code -in 3, 4) { $code } else { 1 }) }
$remoteHash = (($lines | Where-Object { $_ -match '^[0-9a-f]{64}\s' }) | Select-Object -First 1) -split '\s+' | Select-Object -First 1
if ($remoteHash -ne $localHash) { Fail "sha256 on the VM ($remoteHash) differs from the local file ($localHash)" }

Step 5 "export, pull and validate"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dir = Join-Path $Quarantine "$Batch-$stamp"
New-Item -ItemType Directory -Path $dir | Out-Null
$names = @(".jsonl.gz", ".jsonl.gz.sha256", ".manifest.json", ".shots.tar", ".shots.tar.sha256")
$sources = $names | ForEach-Object { "${labHost}:export/$Batch$_" }
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
