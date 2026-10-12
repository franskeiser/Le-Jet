<#
.SYNOPSIS
  Pull one finished batch from the analysis VM into a FRESH quarantine folder, verify its hash and
  run the strict validator (WEB_INSPECTION_PLAN.md sections 12.4, 12.5).

.DESCRIPTION
  Direction rule: the HOST pulls; the VM never connects to the host. Exactly three NAMED files are
  copied (<batch>.jsonl.gz, its .sha256, the manifest) into an empty, new quarantine directory.
  Nothing from the VM is ever run or opened. Run `~/lejet-export.sh <batch>` on the VM first; it
  puts the three files into ~/export.

  After the copy: the sha256 is checked, then `python -I scripts\validate_output.py` is run. Only a
  file that passes is written to the clean folder.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\pull_results.ps1 -Batch b0007 -PushFile D:\LeJetData\input\urls_push.txt
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Batch,
  [string]$PushFile = "",      # the file pushed to the VM: every url_id must come back, nothing else
  [string]$Quarantine = "D:\LeJetData\quarantine",
  [string]$CleanDir = "D:\LeJetData\clean"
)

$ErrorActionPreference = "Stop"
if ($Batch -notmatch '^[A-Za-z0-9_-]{1,40}$') { throw "Bad batch id (letters, digits, - and _ only)." }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dir = Join-Path $Quarantine "$Batch-$stamp"
New-Item -ItemType Directory -Path $dir | Out-Null
if ((Get-ChildItem $dir | Measure-Object).Count -ne 0) { throw "Quarantine folder is not empty: $dir" }
Write-Host "quarantine: $dir"

foreach ($ext in @(".jsonl.gz", ".jsonl.gz.sha256", ".manifest.json")) {
  scp "lejet-an:export/$Batch$ext" $dir
  if ($LASTEXITCODE -ne 0) { throw "scp of $Batch$ext failed (did you run ~/lejet-export.sh $Batch on the VM?)" }
}

$jsonl = Join-Path $dir "$Batch.jsonl.gz"
$sidecar = Join-Path $dir "$Batch.jsonl.gz.sha256"
$expected = ([IO.File]::ReadAllText($sidecar).Trim() -split '\s+')[0].ToLower()
$actual = (Get-FileHash $jsonl -Algorithm SHA256).Hash.ToLower()
if ($expected -ne $actual) { throw "sha256 mismatch: the file was damaged or altered in transit." }
Write-Host "sha256 verified: $actual"

$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$argsList = @("-I", (Join-Path $PSScriptRoot "validate_output.py"), $jsonl, "--clean-dir", $CleanDir,
              "--csv", (Join-Path $CleanDir "$Batch.csv"))
if ($PushFile -ne "") { $argsList += @("--input", $PushFile) }
& $py @argsList
if ($LASTEXITCODE -ne 0) { throw "VALIDATION FAILED: nothing was written to $CleanDir. Keep the quarantine folder as evidence." }
Write-Host "Validated data: $CleanDir\$Batch.clean.jsonl"
