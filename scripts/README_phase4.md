# Phase 4 State Layout (TASK-4A)

Directory layout:
- `data/meta_states/current_approved/` (approved meta state mirror)
- `data/meta_states/candidate/`
- `data/meta_states/rejected/`
- `data/meta_states/history/`

Fallback behavior:
- If `data/meta_states/current_approved/meta_risk_state.json` exists, load it as the approved state.
- Otherwise fall back to the legacy path `data/reports/meta_risk_state.json`.

Saving behavior:
- Legacy path remains the primary save target in Phase 4A.
- Approved state is mirrored into `current_approved/` on save.

Healthcheck:
- `scripts.meta_healthcheck` continues to read the legacy path (`data/reports/meta_risk_state.json`) for now.

Rollback (TASK-4F):
- Entrypoint: `python -m scripts.meta_rollback`
- Required: `--reason <string>`
- Target selection: exactly one of `--to_hash <sha256>` or `--steps <int>` (unless `--dry_run 1` discovery-only mode)
- Defaults:
  - `--strict 1`
  - `--ledger_path data/meta_states/ledger.jsonl`
  - `--dry_run 0`
- Example dry-run:
  - `python -m scripts.meta_rollback --steps 1 --reason "task4f dryrun" --dry_run 1 --strict 1`

Promotion Scheduling (TASK-4H):
- Entrypoint: `python -m scripts.meta_schedule`
- Supports run-once mode:
  - `python -m scripts.meta_schedule --csv data/live/XAUUSD_M1_live.csv --run_once 1 --strict 0 --reason "task4h_run_once"`
- Supports interval loop mode:
  - `python -m scripts.meta_schedule --csv data/live/XAUUSD_M1_live.csv --interval_minutes 30 --max_runtime_seconds 7200 --reason "scheduled"`
- Locking:
  - Default lock path: `data/meta_states/.locks/meta_cycle.lock`
  - `--run_once 1` returns exit code `3` if lock is fresh (`LOCKED`)
  - Loop mode skips locked ticks and retries next interval
- Audit log:
  - Default JSONL: `data/reports/meta_schedule.jsonl`

State/Ledger Atomicity (PH4-H3):
- Critical JSON writes now use atomic replace (`write temp -> fsync -> os.replace`) and keep a rolling `.bak` sibling for decode fallback.
- Covered files include:
  - `data/meta_states/**/meta_risk_state.json`
  - `data/meta_states/**/manifest.json`
  - `data/meta_states/**/perf_history.json`
  - `data/reports/shadow_replay_report.json` and healthcheck JSON outputs generated via `live_sim_daily`.
- `data/meta_states/ledger.jsonl` appends now use crash-safe JSONL append with `fsync`.
- On next append, a trailing partial/invalid tail line is auto-truncated to the last valid newline/JSON boundary.
- Ledger readers in scheduling/rollback paths ignore only invalid trailing tail lines and still fail on non-trailing corruption.
