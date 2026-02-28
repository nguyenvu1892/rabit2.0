#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys

from rabit.validation.holdout import (
    DEFAULT_META_STATE_PATH,
    DEFAULT_OUT_DIR,
    HoldoutConfig,
    run_holdout_validation,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Deterministic holdout validation runner.")
    ap.add_argument("--csv", required=True, help="Input CSV path.")
    ap.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument("--holdout_days", type=int, default=30, help="Use last N days as holdout.")
    ap.add_argument("--min_bars", type=int, default=500, help="Minimum required bars in dataset.")
    ap.add_argument("--strict", type=int, choices=[0, 1], default=1, help="Strict mode (0/1).")
    ap.add_argument(
        "--deterministic_check",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable deterministic hashes in report (0/1).",
    )
    ap.add_argument("--meta_risk", type=int, choices=[0, 1], default=1, help="Pass-through to live sim.")
    ap.add_argument(
        "--meta_feedback",
        type=int,
        choices=[0, 1],
        default=1,
        help="Pass-through to live sim.",
    )
    ap.add_argument("--risk_per_trade", type=float, default=0.02, help="Risk per trade.")
    ap.add_argument("--account_equity_start", type=float, default=500.0, help="Starting equity.")
    ap.add_argument(
        "--meta_state_path",
        default=DEFAULT_META_STATE_PATH,
        help=f"Meta state path (default: {DEFAULT_META_STATE_PATH})",
    )
    return ap


def main() -> int:
    args = _build_arg_parser().parse_args()
    cfg = HoldoutConfig(
        csv=args.csv,
        out_dir=args.out_dir,
        holdout_days=int(args.holdout_days),
        min_bars=int(args.min_bars),
        strict=int(args.strict),
        deterministic_check=int(args.deterministic_check),
        meta_risk=int(args.meta_risk),
        meta_feedback=int(args.meta_feedback),
        risk_per_trade=float(args.risk_per_trade),
        account_equity_start=float(args.account_equity_start),
        meta_state_path=str(args.meta_state_path),
    )
    rc, report = run_holdout_validation(cfg)
    status = str(report.get("status", "FAIL"))
    reason_code = str(report.get("reason_code", "UNKNOWN"))
    report_path = report.get("artifact_paths", {}).get("holdout_report_json", "")
    print(f"[holdout] STATUS={status}")
    print(f"[holdout] reason_code={reason_code}")
    if report_path:
        print(f"[holdout] report_path={report_path}")
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
