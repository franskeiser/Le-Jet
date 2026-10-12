<#
.SYNOPSIS
  Pull the screenshot archive of one finished batch from the analysis VM into a FRESH quarantine
  folder, verify it with the strict validator, and build the image-only gallery page.

.DESCRIPTION
  Direction rule: the HOST pulls; the VM never connects to the host. Exactly two NAMED files are
  copied (<batch>.shots.tar and its .sha256) into a new, empty quarantine directory. The archive is
  never extracted by tar: scripts\validate_shots.py reads every member into memory, checks it (names,
  sizes, PNG structure, manifest) and only then writes the screenshots to
  D:\LeJetData\clean\shots\<batch>\ . Nothing from the VM is ever opened or run.

  Run `sudo lejet-sudo-shots <batch>` and `lejet-sudo-export <batch>` on the VM first (run_batch.ps1
  -Screenshots does all of it).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\pull_shots.ps1 -Batch b0005
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Batch,
  [string]$Quarantine = "D:\LeJetData\quarantine",
  [string]$CleanDir = "D:\LeJetData\clean"
)

$ErrorActionPreference = "Stop"
if ($Batch -notmatch '^[A-Za-z0-9_-]{1,40}$') { throw "Bad batch id (letters, digits, - and _ only)." }
$cleanJsonl = Join-Path $CleanDir "$Batch.clean.jsonl"
if (-not (Test-Path $cleanJsonl)) { throw "Pull and validate the batch first (pull_results.ps1): $cleanJsonl is missing." }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dir = Join-Path $Quarantine "$Batch-shots-$stamp"
New-Item -ItemType Directory -Path $dir | Out-Null
if ((Get-ChildItem $dir | Measure-Object).Count -ne 0) { throw "Quarantine folder is not empty: $dir" }
Write-Host "quarantine: $dir"

foreach ($ext in @(".shots.tar", ".shots.tar.sha256")) {
  & scp -o BatchMode=yes "lejet-an:export/$Batch$ext" $dir
  if ($LASTEXITCODE -ne 0) { throw "scp of $Batch$ext failed (was the screenshot step run for this batch?)" }
}

$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -I (Join-Path $PSScriptRoot "validate_shots.py") (Join-Path $dir "$Batch.shots.tar") `
  --clean-dir $CleanDir --clean-jsonl $cleanJsonl --batch $Batch
if ($LASTEXITCODE -ne 0) { throw "SHOTS VALIDATION FAILED: nothing was written to $CleanDir. Keep the quarantine folder as evidence." }

$shotsDir = Join-Path $CleanDir "shots\$Batch"
& $py -I (Join-Path $PSScriptRoot "make_gallery.py") $cleanJsonl --shots-dir $shotsDir
if ($LASTEXITCODE -ne 0) { throw "gallery build failed" }
Write-Host "Screenshots: $shotsDir"
Write-Host "Gallery    : $(Join-Path $shotsDir 'index.html')   (images only; safe to open)"
