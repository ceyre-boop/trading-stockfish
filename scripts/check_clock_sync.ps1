Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Section {
    param([string]$Title)
    Write-Host "=== $Title ==="
}

Write-Section "Local System Time"
Write-Host "Local time: $(Get-Date -Format o)"
Write-Host "Time zone: $([System.TimeZoneInfo]::Local.Id)"

Write-Section "Windows Time Service Status"
try {
    $svc = Get-Service -Name W32Time -ErrorAction Stop
    Write-Host "Service status: $($svc.Status)"
    Write-Host "Start type: $($svc.StartType)"
} catch {
    Write-Warning "W32Time service not found: $_"
}

Write-Section "NTP Configuration"
try {
    w32tm /query /configuration
} catch {
    Write-Warning "Failed to query configuration: $_"
}

Write-Section "Time Source"
try {
    w32tm /query /source
} catch {
    Write-Warning "Failed to query source: $_"
}

Write-Section "Status"
try {
    w32tm /query /status
} catch {
    Write-Warning "Failed to query status: $_"
}

Write-Section "Drift Estimate (stripchart samples)"
try {
    # Collect a few samples to view observed offset; this may require network access.
    w32tm /stripchart /computer:time.windows.com /dataonly /samples:3 /period:2
} catch {
    Write-Warning "Stripchart failed: $_"
}
