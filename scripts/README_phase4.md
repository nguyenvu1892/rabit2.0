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
