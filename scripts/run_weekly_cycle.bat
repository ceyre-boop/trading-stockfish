@echo off
setlocal
REM Set project root to the directory containing this script's parent
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."
set "PROJECT_ROOT=%CD%"

REM Activate venv and ensure PYTHONPATH
set "VENV_PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo Missing venv python at %VENV_PY%
  popd
  exit /b 1
)
set "PYTHONPATH=%PROJECT_ROOT%"

REM Ensure logs directory exists
set "LOG_DIR=%PROJECT_ROOT%\logs\scheduled"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Compose log file with date
set "LOG_FILE=%LOG_DIR%\weekly_%DATE%.log"

REM Run weekly cycle in SIMULATION mode
"%VENV_PY%" -m engine.jobs.weekly_cycle --mode SIMULATION > "%LOG_FILE%" 2>&1
set "RC=%ERRORLEVEL%"

popd
exit /b %RC%
