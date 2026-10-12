<#
.SYNOPSIS
  Creates the two Le-Jet lab VMs (gateway + analysis) with locked-down hardware.

.DESCRIPTION
  WEB_INSPECTION_PLAN.md section 9.1. Option spellings checked against VirtualBox 7.2.
  - lejet-gw : NAT (internet) + internal network "lejet-net". Runs the VPN and firewall.
  - lejet-an : internal network "lejet-net" ONLY. No NAT, no bridged, no host-only.
  Both VMs have no USB, audio, clipboard, drag and drop, remote display, 3D or serial ports.

  The script does not start the VMs and does not install Ubuntu. After it runs:
  start each VM from the VirtualBox window and follow plan section 9.3.

.PARAMETER DryRun
  Print every VBoxManage command without running it.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\infra\vm\create_vms.ps1 -DryRun
  powershell -ExecutionPolicy Bypass -File .\infra\vm\create_vms.ps1
#>
[CmdletBinding()]
param(
  [string]$VBox = "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe",
  [string]$Base = "D:\VMs\LeJet",
  [string]$ISO  = "D:\LeJetData\iso\ubuntu-24.04.5-live-server-amd64.iso",
  [string]$Net  = "lejet-net",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-VBox {
  param([Parameter(ValueFromRemainingArguments)][string[]]$VBoxArgs)
  if ($DryRun) { Write-Host "[dry-run] VBoxManage $($VBoxArgs -join ' ')"; return }
  & $VBox @VBoxArgs
  if ($LASTEXITCODE -ne 0) { throw "VBoxManage failed (exit $LASTEXITCODE): $($VBoxArgs -join ' ')" }
}

# --- Safety checks before touching anything
if (-not (Test-Path $VBox)) { throw "VBoxManage not found at $VBox" }
if (-not (Test-Path $ISO))  { throw "Ubuntu ISO not found at $ISO (download and verify it first, plan 7.4)" }
if (-not (Test-Path $Base)) { New-Item -ItemType Directory -Force $Base | Out-Null }

$existing = & $VBox list vms
foreach ($n in "lejet-gw", "lejet-an") {
  if ($existing -match "`"$n`"") { throw "VM '$n' already exists. Remove it first or pick another name." }
}

function New-LeJetVm {
  param([string]$Name, [int]$Mem, [int]$Cpus, [int]$DiskMB)

  Invoke-VBox createvm --name $Name --ostype Ubuntu_64 --basefolder $Base --register

  # Hardware: only what Ubuntu Server needs. No USB, audio, 3D, remote display, clipboard, drag and drop.
  Invoke-VBox modifyvm $Name `
    --memory $Mem --cpus $Cpus --vram 16 `
    --graphicscontroller vmsvga --accelerate-3d off `
    --audio-driver none `
    --usb-ohci off --usb-ehci off --usb-xhci off `
    --clipboard-mode disabled --drag-and-drop disabled `
    --vrde off `
    --rtc-use-utc on `
    --boot1 dvd --boot2 disk --boot3 none --boot4 none `
    --paravirt-provider kvm `
    --nested-hw-virt off `
    --uart1 off --uart2 off --lpt1 off --lpt2 off

  # Disk and install ISO
  Invoke-VBox createmedium disk --filename "$Base\$Name\$Name.vdi" --size $DiskMB --format VDI
  Invoke-VBox storagectl $Name --name SATA --add sata --controller IntelAhci
  Invoke-VBox storageattach $Name --storagectl SATA --port 0 --device 0 --type hdd --medium "$Base\$Name\$Name.vdi"
  Invoke-VBox storageattach $Name --storagectl SATA --port 1 --device 0 --type dvddrive --medium $ISO
}

# --- Gateway: NAT (internet) + internal network. SSH from the host only via 127.0.0.1:2201.
New-LeJetVm -Name lejet-gw -Mem 1024 -Cpus 1 -DiskMB 6144
Invoke-VBox modifyvm lejet-gw `
  --nic1 nat --nat-pf1 "ssh,tcp,127.0.0.1,2201,,22" `
  --nic2 intnet --intnet2 $Net --nic-promisc2 deny `
  --nic3 none --nic4 none

# --- Analysis: internal network ONLY (no NAT, no bridged, no host-only)
New-LeJetVm -Name lejet-an -Mem 6144 -Cpus 4 -DiskMB 20480
Invoke-VBox modifyvm lejet-an `
  --nic1 intnet --intnet1 $Net --nic-promisc1 deny `
  --nic2 none --nic3 none --nic4 none

Write-Host ""
if ($DryRun) {
  Write-Host "Dry run finished. Nothing was created."
} else {
  Write-Host "Done. Created lejet-gw and lejet-an under $Base."
  Write-Host "Next: check the settings in the GUI (plan 9.2), then install Ubuntu on each VM (plan 9.3)."
}
