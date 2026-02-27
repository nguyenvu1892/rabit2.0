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
