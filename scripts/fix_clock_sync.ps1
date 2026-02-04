Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Log {
    param([string]$Message)
    $ts = (Get-Date -Format o)
    Write-Host "$ts $Message"
    $global:LogBuilder += "$ts $Message`n"
}

# Prepare log target
$logDir = Join-Path -Path (Get-Location) -ChildPath "logs/system"
if (-not (Test-Path $logDir)) { New-Item -Path $logDir -ItemType Directory | Out-Null }
$logPath = Join-Path -Path $logDir -ChildPath ("clock_sync_{0}.log" -f (Get-Date -Format yyyyMMdd))
$global:LogBuilder = ""

try {
    Write-Log "Starting clock sync remediation"

    # Set service to Automatic (Delayed Start)
    try {
        Set-Service -Name W32Time -StartupType Automatic
        Write-Log "Set W32Time startup type to Automatic"
    } catch {
        Write-Log "Failed to set startup type: $_"
    }

    # Restart time service
    try {
        Write-Log "Restarting W32Time"
        Restart-Service -Name W32Time -Force -ErrorAction Stop
        Write-Log "W32Time restarted"
    } catch {
        Write-Log "Failed to restart W32Time: $_"
    }

    # Force resync
    try {
        Write-Log "Forcing NTP resync"
        w32tm /resync /force
    } catch {
        Write-Log "Resync command failed: $_"
    }

    # Show status and source after resync
    try {
        Write-Log "Querying status"
        $status = w32tm /query /status
        $status | Out-String | ForEach-Object { Write-Log $_ }
    } catch {
        Write-Log "Status query failed: $_"
    }

    try {
        Write-Log "Querying source"
        $source = w32tm /query /source
        $source | Out-String | ForEach-Object { Write-Log $_ }
    } catch {
        Write-Log "Source query failed: $_"
    }

    Write-Log "Clock sync remediation complete"
}
finally {
    $global:LogBuilder | Set-Content -Path $logPath -Encoding UTF8
    Write-Host "Log written to: $logPath"
}
