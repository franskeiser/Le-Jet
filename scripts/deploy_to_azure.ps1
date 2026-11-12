<#
.SYNOPSIS
  Install (or update) the screenshot lab on the Azure VM from this PC. Safe to run again: it only touches the lab
  (user `fetcher`, /opt/lejet, /var/lib/lejet, its own firewall table); the SMS collection website is not changed.

.DESCRIPTION
  Needs `ssh lejet-azure` to work without a password first (see docs/azure_lab.md). Steps:
    1. pack the code the VM needs (src/lejet, infra/azure, the Chromium installer, requirements/vm.in) into one tar;
    2. scp it to the VM and unpack it in ~/lejet-src;
    3. run infra/azure/setup_shot_vm.sh with sudo (the first run takes several minutes: apt, pip, Chromium download).
  Use -CodeOnly after you change Python code: it only refreshes the lejet package on the VM (seconds).

  Philippine exit (Mullvad WireGuard, only for the lab user; SSH and the website stay on the normal connection):
    -VpnConfig <file>      a Mullvad WireGuard config made for THIS VM (its own device key). It is copied over ssh, rebuilt from
                           validated fields only, kept root-only on the VM, and the copy is deleted. Implies -CodeOnly first.
    -ExpectCountry <name>  optional, e.g. Philippines: the VM's safety gate then refuses to open links unless Mullvad reports
                           exactly that exit country.
    -RemoveVpn             go back to the VM's own (Japan) address.
#>
[CmdletBinding()]
param(
  [string]$LabHost = $(if ($env:LEJET_LAB_HOST) { $env:LEJET_LAB_HOST } else { "lejet-azure" }),
  [switch]$CodeOnly,
  [string]$VpnConfig = "",
  [string]$ExpectCountry = "",
  [switch]$RemoveVpn
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ($LabHost -notmatch '^[A-Za-z0-9._-]{1,80}$') { throw "Bad host alias" }
$sshOpts = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")

& ssh @sshOpts $LabHost "true"
if ($LASTEXITCODE -ne 0) { throw "ssh $LabHost does not work yet. Fix that first (docs/azure_lab.md, step 1)." }

$tmp = Join-Path ([IO.Path]::GetTempPath()) "lejet-src.tar"
Remove-Item $tmp -ErrorAction SilentlyContinue
Push-Location $repo
try {
  & "$env:SystemRoot\System32\tar.exe" -cf $tmp --exclude=__pycache__ --exclude=*.pyc src/lejet src/features/link_features.py src/data/clean_web_phishing.py infra/azure infra/analysis/install_browser.sh requirements/vm.in
  if ($LASTEXITCODE -ne 0) { throw "tar failed" }
} finally { Pop-Location }
Write-Host "packed $([math]::Round((Get-Item $tmp).Length / 1KB)) KB"

& scp @sshOpts $tmp "${LabHost}:lejet-src.tar"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }
& ssh @sshOpts $LabHost "rm -rf ~/lejet-src && mkdir ~/lejet-src && tar -xf ~/lejet-src.tar -C ~/lejet-src && find ~/lejet-src -type f \( -name '*.sh' -o -name '*.nft' -o -name '*.sudoers' -o -name '*.service' -o -name '*.timer' -o -name '*.py' -o -name '*.in' \) -exec sed -i 's/$//' {} +"
if ($LASTEXITCODE -ne 0) { throw "unpacking on the VM failed" }

if ($ExpectCountry -and $ExpectCountry -notmatch '^[A-Za-z .-]{2,40}$') { throw "Bad -ExpectCountry" }
if ($RemoveVpn) {
  & ssh -t @sshOpts $LabHost "sudo bash ~/lejet-src/infra/azure/vpn_setup.sh remove"
} elseif ($VpnConfig) {
  if (-not (Test-Path $VpnConfig)) { throw "VPN config not found: $VpnConfig" }
  & ssh -t @sshOpts $LabHost "sudo bash ~/lejet-src/infra/azure/update_code.sh"
  if ($LASTEXITCODE -ne 0) { throw "code update failed" }
  & ssh @sshOpts $LabHost "umask 077; mkdir -p ~/vpn && chmod 700 ~/vpn"
  & scp @sshOpts $VpnConfig "${LabHost}:vpn/lejet-vpn.conf"
  if ($LASTEXITCODE -ne 0) { throw "copying the VPN config failed" }
  & ssh @sshOpts $LabHost "chmod 600 ~/vpn/lejet-vpn.conf"
  $expect = if ($ExpectCountry) { " '$ExpectCountry'" } else { "" }
  & ssh -t @sshOpts $LabHost "sudo bash ~/lejet-src/infra/azure/vpn_setup.sh install$expect"
} elseif ($CodeOnly) {
  & ssh -t @sshOpts $LabHost "sudo bash ~/lejet-src/infra/azure/update_code.sh"
} else {
  & ssh -t @sshOpts $LabHost "sudo bash ~/lejet-src/infra/azure/setup_shot_vm.sh"
}
if ($LASTEXITCODE -ne 0) { throw "the setup on the VM did not finish (see the messages above)" }
Write-Host "OK. Test it:  ssh $LabHost 'sudo -n /usr/local/sbin/lejet-azure-gate'"
