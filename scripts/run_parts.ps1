<#
.SYNOPSIS
  Run ALL pending parts of the labelled feature input through the lab with ONE command, then build the feature table.

.DESCRIPTION
  scripts\build_labeled_input.py keeps the table D:\LeJetData\datasets\feature_input.csv (a `date_added` column says
  when each link was added) and, for the links added in each run, writes part files:
  feature_input_<stamp>_part1of3.txt ... This runs run_batch.ps1 for every part that has no result yet, oldest
  first, one after another, and at the end scripts\build_feature_dataset.py (url + the 73 features + status, over
  EVERYTHING run so far, classes balanced).

  Before you start (the same as for run_batch.ps1): both VMs running (lejet-gw first), your SSH key loaded
  in the ssh-agent. Nothing here sees a secret; every ssh call is BatchMode.

  It is safe to run again: a part that already has D:\LeJetData\clean\<batch>.clean.jsonl is skipped, so after a
  failure just run the same command and it carries on with the first unfinished part. A failed part is tried once
  more after a minute (the VM resumes a batch, it does not redo finished links).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\run_parts.ps1
  powershell -ExecutionPolicy Bypass -File .\scripts\run_parts.ps1 -DryRun          # show the plan, run nothing
  powershell -ExecutionPolicy Bypass -File .\scripts\run_parts.ps1 -Stamp 20261014 -To 1   # one group, first part only
#>
[CmdletBinding()]
param(
  [string]$DatasetDir = "D:\LeJetData\datasets",
  [string]$Stamp = "",                 # default: every group of part files that is not finished yet, oldest first
  [int]$From = 1,                      # only with -Stamp
  [int]$To = 0,                        # only with -Stamp; 0 = the last part
  [int]$Retries = 1,                   # extra tries for a part that failed
  [int]$RetryWaitSeconds = 60,
  [switch]$Force,                      # run parts again even if their clean file exists
  [switch]$NoBuild,                    # do not build the feature table at the end
  [switch]$NoBalance,                  # build it without dropping rows of the larger class
  [switch]$DryRun,                     # print the plan and stop
  [string]$CleanDir = "D:\LeJetData\clean"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }
function BatchName([string]$stamp, [int]$n) { "feat${stamp}p$n" }
function CleanFile([string]$stamp, [int]$n) { Join-Path $CleanDir ("{0}.clean.jsonl" -f (BatchName $stamp $n)) }

# ---- the groups of part files, by stamp ----------------------------------------------------------------------
$groups = @{}                          # stamp -> @{ n = path }
Get-ChildItem -Path $DatasetDir -Filter "feature_input_*_part*of*.txt" -ErrorAction SilentlyContinue | ForEach-Object {
  if ($_.BaseName -match '^feature_input_(.+)_part(\d+)of(\d+)$') {
    if (-not $groups.ContainsKey($Matches[1])) { $groups[$Matches[1]] = @{} }
    $groups[$Matches[1]][[int]$Matches[2]] = $_.FullName
  }
}
if ($groups.Count -eq 0) { throw "No part files in $DatasetDir. Run: python scripts\build_labeled_input.py" }

function Pending([string]$stamp) {
  @($groups[$stamp].Keys | Where-Object { $Force -or -not (Test-Path (CleanFile $stamp $_)) })
}

if ($Stamp -ne "") {
  if (-not $groups.ContainsKey($Stamp)) { throw "No part files for stamp $Stamp (have: $(($groups.Keys | Sort-Object) -join ', '))" }
  $stamps = @($Stamp)
} else {
  $stamps = @($groups.Keys | Sort-Object | Where-Object { (Pending $_).Count -gt 0 })
}

# ---- the plan --------------------------------------------------------------------------------------------------
$plan = @()
foreach ($st in $stamps) {
  $last = ($groups[$st].Keys | Measure-Object -Maximum).Maximum
  $lo = 1; $hi = $last
  if ($Stamp -ne "") { $lo = [math]::Max(1, $From); if ($To -gt 0 -and $To -le $last) { $hi = $To } }
  foreach ($n in $lo..$hi) {
    if (-not $groups[$st].ContainsKey($n)) { continue }
    $done = (Test-Path (CleanFile $st $n)) -and -not $Force
    $plan += [pscustomobject]@{ Stamp = $st; N = $n; Last = $last; Batch = (BatchName $st $n); File = $groups[$st][$n]; Done = $done }
  }
}
Say ""
if (@($plan | Where-Object { -not $_.Done }).Count -eq 0) {
  Say "Nothing pending: every part already has its result." "Green"
} else {
  Say "Parts to run:" "Cyan"
  foreach ($p in $plan) {
    $state = if ($p.Done) { "already done, will be skipped" } else { "to run" }
    Say ("  {0}  part {1} of {2}  [{3}]" -f $p.Batch, $p.N, $p.Last, $state)
  }
}
if ($DryRun) { Say "`nDry run: nothing was started." "Yellow"; return }

# ---- the lab must be reachable (same rule as run_batch.ps1: BatchMode, no passphrase prompt) --------------------
if (@($plan | Where-Object { -not $_.Done }).Count -gt 0) {
  $oldPref = $ErrorActionPreference
  $ErrorActionPreference = "Continue"               # Windows PowerShell 5.1 turns ssh's stderr text into an error
  $null = & ssh -o BatchMode=yes -o ConnectTimeout=10 lejet-an "true" 2>&1
  $sshCode = $LASTEXITCODE
  $ErrorActionPreference = $oldPref
  if ($sshCode -ne 0) {
    throw "Cannot reach the analysis VM with your SSH key. Start both VMs (lejet-gw first) and load the key: ssh-add <your lejet-lab-only key>"
  }
}

# ---- run the parts -----------------------------------------------------------------------------------------------
$started = Get-Date
$failed = ""
foreach ($p in $plan) {
  if ($p.Done) { Say "`n== $($p.Batch): already done, skipped" "DarkGray"; continue }
  $ok = $false
  for ($try = 0; $try -le $Retries -and -not $ok; $try++) {
    if ($try -gt 0) { Say "`n!! $($p.Batch) failed; trying again in $RetryWaitSeconds s ($try of $Retries)" "Yellow"; Start-Sleep -Seconds $RetryWaitSeconds }
    Say ("`n========== {0}  part {1} of {2}  {3:HH:mm:ss} ==========" -f $p.Batch, $p.N, $p.Last, (Get-Date)) "Cyan"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_batch.ps1") -UrlFile $p.File -Batch $p.Batch
    $ok = ($LASTEXITCODE -eq 0) -and (Test-Path (CleanFile $p.Stamp $p.N))
  }
  if (-not $ok) { $failed = $p.Batch; break }
}

$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
if ($failed -ne "") {
  Say "`nSTOPPED at $failed after $minutes min. Read the message above (a failed self-test, a lost SSH key or a VM that is off)," "Red"
  Say "fix it, then run the SAME command again: finished parts are skipped." "Red"
  exit 1
}
Say "`nAll pending parts finished in $minutes min." "Green"

# ---- the feature table, over everything run so far -------------------------------------------------------------
if ($NoBuild) { return }
$unfinished = @($groups.Keys | Where-Object { (Pending $_).Count -gt 0 -and -not $Force })
if ($unfinished.Count -gt 0) {
  Say ("Not building the feature table yet: groups with parts still to run: {0}. Run them, then build." -f (($unfinished | Sort-Object) -join ", ")) "Yellow"
  return
}
# page_rank for the pages that loaded (host side; skipped quietly if the key file is missing; a failure never stops the run)
Say "`n== page_rank for the pages that loaded" "Cyan"
& $py (Join-Path $PSScriptRoot "refresh_page_rank.py")
$argsList = @((Join-Path $PSScriptRoot "build_feature_dataset.py"), "--lean")
$prCsv = Join-Path $CleanDir "lookup_fetched.model_input.csv"
if (($LASTEXITCODE -eq 0) -and (Test-Path $prCsv)) { $argsList += @("--page-rank-csv", $prCsv) }
if (-not $NoBalance) { $argsList += "--balance" }
Say "`n== building the feature table (url + the features + status)" "Cyan"
& $py @argsList
if ($LASTEXITCODE -ne 0) { throw "build_feature_dataset.py failed (see above); the lab results are safe in $CleanDir" }
