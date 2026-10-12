<#
.SYNOPSIS
  Push the URL list to the analysis VM (WEB_INSPECTION_PLAN.md section 12.3).

.DESCRIPTION
  Direction rule: HOST -> VM only. The file must come from scripts\make_push_file.py: one
  `url_id<TAB>url` per line, refanged, machine-consumed only. NEVER open it in a browser; view the
  `.defanged.txt` copy instead.

  The file is installed as /var/lib/lejet/in/urls.txt owned by `fetcher`, and its sha256 and line
  count are printed on both sides so you can record them in the run log.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\push_urls.ps1 -File D:\LeJetData\input\urls_push.txt
#>
[CmdletBinding()]
param([string]$File = "D:\LeJetData\input\urls_push.txt")

$ErrorActionPreference = "Stop"
if (-not (Test-Path $File)) { throw "Push file not found: $File (build it with scripts\make_push_file.py)" }

$text = [IO.File]::ReadAllText($File)
if ($text.Contains("`r")) { throw "The push file has CR characters; rebuild it with scripts\make_push_file.py (LF only)." }
$lines = ($text -split "`n" | Where-Object { $_.Trim() -ne "" -and -not $_.StartsWith("#") })
if (-not $lines) { throw "The push file has no URLs." }
foreach ($l in $lines) {
  if ($l -notmatch '^[0-9a-f]{16}\t\S+$') { throw "Bad line (expected url_id<TAB>url): $($l.Substring(0, [Math]::Min(60, $l.Length)))" }
}
$hash = (Get-FileHash $File -Algorithm SHA256).Hash.ToLower()
Write-Host "local : $($lines.Count) URLs, sha256 $hash"

scp $File lejet-an:/tmp/urls.txt
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

$remote = "sudo install -o fetcher -g fetcher -m 640 /tmp/urls.txt /var/lib/lejet/in/urls.txt && rm -f /tmp/urls.txt && " +
          "echo -n 'on VM: ' && sudo wc -l /var/lib/lejet/in/urls.txt && sudo sha256sum /var/lib/lejet/in/urls.txt"
ssh -t lejet-an $remote
if ($LASTEXITCODE -ne 0) { throw "installing the file on the VM failed" }
Write-Host "The sha256 above must equal: $hash"
