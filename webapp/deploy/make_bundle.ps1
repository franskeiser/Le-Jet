# Packs ONLY what the website needs into lejet-web-bundle.tar.gz (no data, no models, no .git, no .venv).
#   powershell -ExecutionPolicy Bypass -File webapp\deploy\make_bundle.ps1
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$stage = Join-Path $env:TEMP "lejet-web-bundle"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force "$stage\src\features", "$stage\src\data" | Out-Null

Copy-Item -Recurse "$root\src\lejet" "$stage\src\lejet"
Copy-Item "$root\src\features\link_features.py" "$stage\src\features\"
Copy-Item "$root\src\data\clean_web_phishing.py" "$stage\src\data\"
Copy-Item -Recurse "$root\webapp" "$stage\webapp"
Get-ChildItem $stage -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Linux reads the Windows line ending (CR LF) as a stray character at the end of every line: `set -o pipefail`
# fails, systemd values break. Every file in the bundle is text, so convert them all to LF.
$utf8 = New-Object System.Text.UTF8Encoding($false)
Get-ChildItem $stage -Recurse -File | ForEach-Object {
    $text = [IO.File]::ReadAllText($_.FullName)
    [IO.File]::WriteAllText($_.FullName, $text.Replace("`r`n", "`n"), $utf8)
}

$out = Join-Path $root "lejet-web-bundle.tar.gz"
& "$env:SystemRoot\System32\tar.exe" -czf $out -C $stage .     # Windows' own tar (Git's GNU tar reads "D:" as a host name)
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
Write-Host "Wrote $out"
