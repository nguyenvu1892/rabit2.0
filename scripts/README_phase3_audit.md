# Phase 3 Audit (Engineering)

## Summary
- Extracted IO helpers into `scripts/_io_utils.py` and kept wrappers in `scripts/live_sim_daily.py`.
- Extracted deterministic snapshot helpers into `scripts/_deterministic.py`.
- Added per-trade risk cap helper in `rabit/risk/risk_cap.py`.
- Added regime ledger canonicalization + hashing in `rabit/meta/regime_ledger.py`.
- Updated `.gitignore` so generated reports and snapshots do not dirty the repo.

## Phase 3 Gates
Run from repo root:

1. Deterministic check (run twice):
   `python -m scripts.live_sim_daily --csv data/live/XAUUSD_M1_live.csv --meta_risk 1 --meta_feedback 1 --risk_per_trade 0.02 --account_equity_start 500 --deterministic_check 1`
   `python -m scripts.live_sim_daily --csv data/live/XAUUSD_M1_live.csv --meta_risk 1 --meta_feedback 1 --risk_per_trade 0.02 --account_equity_start 500 --deterministic_check 1`
2. Healthcheck strict:
   `python -m scripts.meta_healthcheck --summary data/reports/live_daily_summary.json --equity data/reports/live_daily_equity.csv --meta_state data/reports/meta_risk_state.json --regime data/reports/live_daily_regime.json --strict 1`
3. Freeze snapshot strict:
   `python -m scripts.prod_freeze_snapshot --reports_dir data/reports --out_dir data/snapshots/phase3 --strict 1`
4. Stress layer (must not crash):
   `python -m scripts.live_sim_daily --csv data/live/XAUUSD_M1_live.csv --meta_risk 1 --meta_feedback 1 --risk_per_trade 0.02 --account_equity_start 500 --guardrail_stress 1 --debug 1`
   Note: `--guardrail_stress` is a no-op toggle for the stress gate (it only emits a `[stress]` log when debug is on).

## Determinism Hashing
- Canonical JSON uses `sort_keys=True`, separators `(',', ':')`, and `ensure_ascii=False`.
- `input_hash` = sha256 of the input CSV.
- `equity_hash` = hash of the equity rows.
- `summary_hash` = hash of the summary JSON.
- `regime_hash` = hash of the regime stats JSON.
- `regime_ledger_hash` = hash of the canonicalized regime ledger.

Excluded fields from deterministic hashing:
- `regime_ledger.last_update_ts` (volatile)

When `--debug 1` or `--deterministic_check 1` is enabled, deterministic logs print the included/excluded
fields for regime ledger hashing.

## Git Cleanliness
Generated artifacts are ignored in `.gitignore`, including:
- `data/reports/**/*.csv`
- `data/reports/**/*.json`
- `data/reports/**/*.jsonl`
- `data/snapshots/**`

To clean the workspace, remove the `data/reports` and `data/snapshots` directories.
