# Clock Synchronization Guide

Clock sync is critical for Phase 10 because timestamps drive regimes, storage partitioning, drift windows, stats windows, gating, and connector behavior. An unsynced clock can corrupt partitions, misclassify sessions, and break safety gates. Fix clock drift before running unattended PAPER days.

## How to Check
Run from project root:

```powershell
./scripts/check_clock_sync.ps1
```

Review:
- Local time and timezone
- W32Time service status/start type
- NTP configuration and source
- Status (should report synchronized)
- Stripchart offsets (drift samples)

## How to Fix
Run from project root (PowerShell as Administrator recommended):

```powershell
./scripts/fix_clock_sync.ps1
```

What it does:
- Sets W32Time to Automatic (Delayed Start)
- Restarts W32Time
- Forces immediate NTP resync
- Logs output to `logs/system/clock_sync_YYYYMMDD.log`

## Expected Output
- Service status: Running
- Start type: Automatic (Delayed Start)
- Source: trusted NTP (e.g., time.windows.com)
- Status: synchronized; small offset
- Log file created under `logs/system/`

## Verification Steps
1) Run `./scripts/check_clock_sync.ps1` after the fix.
2) Confirm `w32tm /query /status` shows "synchronized" and offsets are minimal.
3) Confirm `w32tm /query /source` shows the NTP source (e.g., time.windows.com).
4) Proceed with PAPER runs only after synchronization is verified.
