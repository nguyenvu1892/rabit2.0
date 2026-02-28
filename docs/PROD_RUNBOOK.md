# Production Runbook (Phase 4 Hardening)

This runbook covers operational use of:
- `scripts.meta_schedule` (live run entrypoint)
- `scripts.meta_cycle` (governance pipeline)
- `scripts.prod_smoke_test` (pre-flight smoke)
- Phase 4 safeguards: exit taxonomy, lock handling, atomic IO, structured JSONL logs, anomaly guardrail

## 1) DAILY START CHECK

1. Run the smoke test before any live cycle:

```bash
python -m scripts.prod_smoke_test --csv data/live/XAUUSD_M1_live.csv
```

Expected:
- `STATUS=PASS`
- Process exit code `0`

2. Check structured logs in `logs/meta_cycle.jsonl` for last smoke/live entries:

```powershell
Get-Content logs\meta_cycle.jsonl -Tail 50
```

Look for:
- `module=prod_smoke_test` summary
- `module=meta_cycle` `stage_end` records with expected `rc`
- no unresolved exceptions

3. Verify no active lock is stuck:

```powershell
$lock = "data\meta_states\.locks\meta_cycle.lock"
if (Test-Path $lock) {
  $payload = Get-Content $lock -Raw | ConvertFrom-Json
  $payload | Select-Object active,pid,host,start_ts_utc,reason,cmd
}
```

Expected:
- `active = 0` when idle
- if `active = 1`, confirm the owning process is expected before starting a new run

## 2) DAILY LIVE RUN

Run one scheduled cycle:

```bash
python -m scripts.meta_schedule --csv data/live/XAUUSD_M1_live.csv --run_once 1 --strict 0
```

Expected exit codes:
- `0` SUCCESS (includes acceptable business outcomes normalized by scheduler)
- `10` BUSINESS_REJECT (acceptable business outcome)
- `11` BUSINESS_SKIP (retryable lock-held/business skip)
- `20` DATA_INVALID
- `30` STATE_CORRUPT
- `40` LOCK_TIMEOUT
- `50` INTERNAL_ERROR
- `60` ANOMALY_HALT

Operational rule:
- Treat `0/10/11` as non-crash outcomes.
- Treat `20/30/40/50/60` as incidents requiring operator action.

## 3) WEEKLY MAINTENANCE

1. Check ledger growth and recency:

```powershell
Get-Item data\meta_states\ledger.jsonl | Select-Object FullName,Length,LastWriteTime
Get-Content data\meta_states\ledger.jsonl -Tail 20
```

2. Check incident inventory:

```powershell
Get-ChildItem data\reports\incidents | Sort-Object LastWriteTime -Descending | Select-Object -First 20 Name,Length,LastWriteTime
```

3. Backup `meta_states` (atomic snapshots and history):

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force backups | Out-Null
Compress-Archive -Path data\meta_states\* -DestinationPath ("backups\meta_states_" + $ts + ".zip")
```

## 4) EMERGENCY PROCEDURE

### A) ANOMALY HALT (`exit 60`)

1. Stop any repeating scheduler loop.
2. Inspect latest incident:

```powershell
Get-ChildItem data\reports\incidents | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Format-List FullName,LastWriteTime,Length
```

3. Review anomaly details:

```powershell
Get-Content <latest_incident_path>.json
```

4. Validate baseline with smoke test:

```bash
python -m scripts.prod_smoke_test --csv data/live/XAUUSD_M1_live.csv
```

5. If still failing, perform rollback (see rollback steps below), then rerun smoke.

### B) Lock stuck / lock held

1. Inspect lock payload (`active`, `pid`, `start_ts_utc`, `reason`, `cmd`).
2. If process is stale/dead, run one protected cycle with stale-lock recovery:

```bash
python -m scripts.meta_schedule --csv data/live/XAUUSD_M1_live.csv --run_once 1 --strict 0 --on_lock_held retryable --force_lock_break 1 --lock_ttl_sec 1800 --reason "lock_recovery"
```

3. If lock remains active unexpectedly, escalate and avoid manual state edits until ownership is clear.

### C) STATE_CORRUPT (`exit 30`)

1. Freeze writes; keep current artifacts for audit.
2. Run rollback dry-run:

```bash
python -m scripts.meta_rollback --steps 1 --reason "state_corrupt_recovery" --dry_run 1 --strict 1
```

3. Execute rollback:

```bash
python -m scripts.meta_rollback --steps 1 --reason "state_corrupt_recovery" --dry_run 0 --strict 1
```

4. Re-run smoke test and confirm `STATUS=PASS` before resuming live run.

### D) Rollback steps (general)

Use either hash-targeted rollback or step rollback:

```bash
python -m scripts.meta_rollback --to_hash <sha256> --reason "manual_rollback" --dry_run 1 --strict 1
python -m scripts.meta_rollback --to_hash <sha256> --reason "manual_rollback" --dry_run 0 --strict 1
```

or

```bash
python -m scripts.meta_rollback --steps 1 --reason "manual_rollback" --dry_run 1 --strict 1
python -m scripts.meta_rollback --steps 1 --reason "manual_rollback" --dry_run 0 --strict 1
```

Then validate:

```bash
python -m scripts.prod_smoke_test --csv data/live/XAUUSD_M1_live.csv
```

## 5) EXIT CODE TABLE

| Code | Name |
|---|---|
| 0 | SUCCESS |
| 10 | BUSINESS_REJECT |
| 11 | BUSINESS_SKIP |
| 20 | DATA_INVALID |
| 30 | STATE_CORRUPT |
| 40 | LOCK_TIMEOUT |
| 50 | INTERNAL_ERROR |
| 60 | ANOMALY_HALT |
