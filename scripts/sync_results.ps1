<#
.SYNOPSIS
  Copy the VALIDATED results of one or more batches from D:\LeJetData into the project folder
  (D:\Le-Jet\results\<batch>\) so you can browse them next to the code, including the screenshots.

.DESCRIPTION
  What is copied (an exact allow-list, nothing else, never anything from the VM or the quarantine):
    <batch>.md              the plain-language report (defanged)
    <batch>.clean.jsonl     the validated data (defanged URLs)
    <batch>.csv             the same as a table
    shots\<16 hex>.png      the validated screenshots
    shots\shots.jsonl       what happened for each screenshot attempt
    shots\index.html        the image-only gallery (no scripts, no links)

  Safety: .gitignore keeps the screenshots, .jsonl, .csv and .html under results\ OUT of git, so only
  the .md reports could ever be committed. Every PNG is checked again (lejet.pngcheck) before it is copied.
  The screenshots are pictures of real phishing pages: do not upload them anywhere without thinking.

  Windows Controlled Folder Access protects D:\Le-Jet. If this script says "access denied", either allow
  PowerShell in Windows Security > Ransomware protection > Allow an app through Controlled folder access,
  or copy the folders by hand with File Explorer (the same files, same names).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\sync_results.ps1 -Batch b0007
  powershell -ExecutionPolicy Bypass -File .\scripts\sync_results.ps1            # every validated batch
#>
[CmdletBinding()]
param(
  [string[]]$Batch = @(),
  [string]$Data = "D:\LeJetData",
  [string]$Dest = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ($Dest -eq "") { $Dest = Join-Path $repo "results" }
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$clean = Join-Path $Data "clean"
if ($Batch.Count -eq 0) {
  $Batch = @(Get-ChildItem $clean -Filter "*.clean.jsonl" | ForEach-Object { $_.Name -replace '\.clean\.jsonl$', '' })
}
if ($Batch.Count -eq 0) { throw "No validated batches found in $clean" }

function Copy-One($src, $dst) {
  Copy-Item -LiteralPath $src -Destination $dst -Force
  Write-Host "   copied $(Split-Path $src -Leaf)"
}

foreach ($b in $Batch) {
  if ($b -notmatch '^[A-Za-z0-9_-]{1,40}$') { throw "Bad batch id: $b" }
  $jsonl = Join-Path $clean "$b.clean.jsonl"
  if (-not (Test-Path $jsonl)) { Write-Warning "$b has no validated data; skipped"; continue }
  Write-Host "== $b"
  $out = Join-Path $Dest $b
  New-Item -ItemType Directory -Force -Path $out | Out-Null

  Copy-One $jsonl $out
  foreach ($extra in @((Join-Path $clean "$b.csv"), (Join-Path $Data "reports\$b.md"))) {
    if (Test-Path $extra) { Copy-One $extra $out }
  }

  $shots = Join-Path $clean "shots\$b"
  if (Test-Path $shots) {
    $outShots = Join-Path $out "shots"
    New-Item -ItemType Directory -Force -Path $outShots | Out-Null
    $n = 0
    foreach ($f in Get-ChildItem $shots -File) {
      if ($f.Name -match '^[0-9a-f]{16}\.png$') {
        # re-check the picture before it leaves the quarantine-checked folder
        $ok = & $py -I -c "import sys; sys.path.insert(0, sys.argv[1]); from lejet import pngcheck; sys.exit(0 if pngcheck.check_png(open(sys.argv[2],'rb').read())[0] else 1)" (Join-Path $repo "src") $f.FullName
        if ($LASTEXITCODE -ne 0) { Write-Warning "$($f.Name) failed the PNG check; NOT copied"; continue }
        Copy-Item -LiteralPath $f.FullName -Destination $outShots -Force
        $n++
      }
      elseif ($f.Name -in @("shots.jsonl", "index.html")) {
        Copy-Item -LiteralPath $f.FullName -Destination $outShots -Force
      }
      else { Write-Warning "unexpected file $($f.Name) in $shots; NOT copied" }
    }
    Write-Host "   copied $n screenshot(s) + manifest + gallery -> $outShots"
  }
  else { Write-Host "   (no screenshots for this batch)" }
}

Write-Host ""
Write-Host "Done. Open: $Dest\<batch>\shots\index.html for the gallery, <batch>.md for the report."
Write-Host "Git will NOT pick up the screenshots, .jsonl, .csv or .html (see .gitignore)."
