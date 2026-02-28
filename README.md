# rabit2.0

Clean workspace: generated outputs live in data/reports and data/snapshots. Remove those directories to reset artifacts.

## VSCode PowerShell Meta Runner

When running meta pipeline in VSCode PowerShell, use `scripts/run_meta_cycle.ps1` to avoid false crash popups for business outcomes.
The wrapper normalizes `10 (SUCCESS_WITH_REJECT)` and `20 (BUSINESS_FAIL)` to process exit `0`, and keeps other non-zero exits unchanged.
It also prints a concise line: `original_exit=<n> meaning=<...> normalized_exit=<m>`.

Example commands:
- `.\scripts\run_meta_cycle.ps1 --csv data/live/XAUUSD_M1_live.csv --reason "manual_cycle" --strict 1`
- `.\scripts\run_meta_cycle.ps1 --csv data/live/XAUUSD_M1_live.csv --reason "manual_cycle_skip_lock" --strict 1 --skip_global_lock 1`
