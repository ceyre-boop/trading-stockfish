# Session Regimes (UTC)

## Overview
This document defines deterministic session_regime labels and their UTC time windows for FX/indices use. The rules are self-contained, do not reference evaluator logic, and resolve overlaps unambiguously. Weekend/holiday handling is explicitly defined.

## Regime Windows (UTC)

| Regime               | Window (UTC)                |
|----------------------|-----------------------------|
| PRE_SESSION          | 05:00 – 07:59               |
| ASIA                 | 00:00 – 06:59               |
| LONDON               | 08:00 – 11:59               |
| LONDON_NY_OVERLAP    | 12:00 – 16:59               |
| NY                   | 17:00 – 21:59               |
| POST_SESSION         | 22:00 – 23:59               |

## Overlap Resolution
1) Derive a candidate set from windows above for a given UTC timestamp.
2) If multiple regimes match, pick the one with the latest start time (i.e., most specific / latest in the day).
3) If still tied (should not occur with the defined windows), choose the regime with lexical order: LONDON_NY_OVERLAP > NY > LONDON > ASIA > PRE_SESSION > POST_SESSION.

## Weekend & Holiday Handling
- Saturday and Sunday (based on UTC) are treated as no-session; return None/"NONE".
- For exchange or regional holidays, treat as no-session (same as weekend) until explicit holiday calendars are added.

## Examples (all UTC)
- 2026-02-02 01:30 → ASIA
- 2026-02-02 07:30 → ASIA (PRE_SESSION ends at 07:59 but ASIA started earlier; latest-start rule keeps ASIA)
- 2026-02-02 08:30 → LONDON
- 2026-02-02 12:45 → LONDON_NY_OVERLAP
- 2026-02-02 17:15 → NY
- 2026-02-02 22:30 → POST_SESSION
- 2026-02-06 23:30 (Friday) → POST_SESSION
- 2026-02-07 10:00 (Saturday) → NONE (weekend)
