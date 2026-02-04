$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $projectRoot "logs\tasks"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir "heartbeat_$(Get-Date -Format yyyyMMdd).log"
if (-not (Test-Path $python)) { "missing venv python at $python" | Tee-Object -FilePath $logFile -Append; exit 1 }
$cmd = "from engine.guardrails import kill_switch; from engine.modes import Mode; import json; r=kill_switch(Mode.SIMULATION); print(json.dumps(r))"
& $python -c $cmd 1>> $logFile 2>&1
exit $LASTEXITCODE
