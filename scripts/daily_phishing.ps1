<#
.SYNOPSIS
  The daily command: collect fresh phishing links (any domain), add them to the feature input with matching legitimate
  links, run them through the lab, and rebuild the feature tables. One command, in the order that keeps the links fresh.

.DESCRIPTION
  1  collect   D:\LeJetLinkHunter\hunt.py with NO SMS-style filter (--min-score -100): PhishTank (verified in the last 3
               days), OpenPhish, Phishunt, PhishStats, Phishing.Database and urlscan.io (scans others tagged phishing, taken as they are: no verdict check, no
               report downloads). New links
               go to the master sheet; nothing is visited.
  2  add       scripts\build_labeled_input.py: every master link not yet in D:\LeJetData\datasets\feature_input.csv is added
               (at most 20 per hosting platform such as pages.dev) with legitimate links from the Tranco pool at the ratio
               -LegitRatio per phishing link, each row dated today; the lab input files are written.
  3  run      scripts\run_parts.ps1: the lab (needs both VMs running and your SSH key loaded), then page rank and the final
               tables D:\LeJetData\datasets\feature_dataset.csv / feature_dataset_lean.csv.

  Why the order and the haste: links die within hours. Your own data: same-day OpenPhish links loaded 33%, PhishTank rows
  older than a month about 0%. Run it when you can stay with it, not at the end of the day.

  Why 0.6 legitimate per phishing link: legitimate pages load more often (about 59%) than phishing ones (about 33%), so about
  0.6 gives similar numbers of pages that LOAD, which is what the final table is balanced on.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\daily_phishing.ps1
  powershell -ExecutionPolicy Bypass -File .\scripts\daily_phishing.ps1 -DryRun     # how many new links, nothing written
  powershell -ExecutionPolicy Bypass -File .\scripts\daily_phishing.ps1 -NoLab      # collect + add only, run the lab later
#>
[CmdletBinding()]
param(
  [double]$LegitRatio = 0.6,           # legitimate links per new phishing link
  [int]$PlatformCap = 60,              # phishing links per hosting platform in the whole table (0 = no cap)
  [int]$UrlscanHours = 72,             # urlscan.io: scans from the last N hours (older = more links, more dead ones)
  [int]$Chunk = 400,                   # links per lab part
  [int]$Max = 9000,                    # most links to collect in one go
  [switch]$NoUrlscan,                  # do not use the urlscan.io key
  [switch]$NoLab,                      # stop after adding the links
  [switch]$DryRun                      # show what would happen; write nothing, run nothing
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$hunter = "D:\LeJetLinkHunter\hunt.py"
if (-not (Test-Path $hunter)) { throw "Link hunter not found: $hunter" }

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }

# ---- 1 collect ---------------------------------------------------------------------------------------------
Say "`n== 1/3  collecting fresh phishing links (no SMS-style filter)" "Cyan"
$h = @($hunter, "--min-score", "-100", "--max", "$Max", "--per-source", "4000", "--per-brand", "200", "--max-age-hours", "72")
if ($NoUrlscan) { $h += "--no-urlscan" } else { $h += @("--urlscan-any", "--any-no-verify", "--any-hours", "$UrlscanHours", "--any-pages", "12", "--any-max", "3000", "--no-urlscan-sms") }   # the SMS-style brand queries are not needed
if ($DryRun) { $h += "--dry-run" }
Push-Location (Split-Path $hunter)
try { & $py @h } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "The link hunter failed (see above); nothing was added." }

# ---- 2 add to the feature input ----------------------------------------------------------------------------------
Say "`n== 2/3  adding the new links to the feature input (with legitimate partners)" "Cyan"
$b = @((Join-Path $PSScriptRoot "build_labeled_input.py"), "--legit-ratio", "$LegitRatio", "--platform-cap", "$PlatformCap", "--chunk", "$Chunk")
if ($DryRun) { $b += "--dry-run" }
& $py @b
if ($LASTEXITCODE -ne 0) { throw "build_labeled_input.py failed (see above)." }

if ($DryRun) { Say "`nDry run: nothing was written and nothing was run." "Yellow"; return }

# ---- 3 the lab ---------------------------------------------------------------------------------------------------
if ($NoLab) {
  Say "`nStopped before the lab (-NoLab). Run it soon, while the links are fresh:" "Yellow"
  Say "  powershell -ExecutionPolicy Bypass -File .\scripts\run_parts.ps1" "Yellow"
  return
}
Say "`n== 3/3  running the lab (both VMs must be running and your SSH key loaded)" "Cyan"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_parts.ps1")
exit $LASTEXITCODE
