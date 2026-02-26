#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _load_json(path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False, None
        return True, data
    except Exception:
        return False, None


def _validate_summary(path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    ok, data = _load_json(path)
    if not ok or data is None:
        return False, None
    if "days" not in data or "total_pnl" not in data:
        return False, None
    try:
        days = float(data["days"])
    except Exception:
        return False, None
    if days <= 0:
        return False, None
    return True, data


def _validate_equity(path: str) -> Tuple[bool, Optional[pd.DataFrame]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return False, None
    required_cols = {"day", "start_equity", "end_equity", "day_pnl"}
    if not required_cols.issubset(set(df.columns)):
        return False, None
    if len(df) <= 0:
        return False, None
    return True, df


def _validate_pnl_consistency(
    df: Optional[pd.DataFrame], summary: Optional[Dict[str, Any]]
) -> bool:
    if df is None or summary is None:
        return False
    if "day_pnl" not in df.columns:
        return False
    try:
        series = pd.to_numeric(df["day_pnl"], errors="coerce")
        if series.isna().any():
            return False
        total_pnl = float(summary["total_pnl"])
        diff = float(series.sum()) - total_pnl
        return abs(diff) <= 1e-6
    except Exception:
        return False


def _validate_meta_state(path: str) -> bool:
    ok, data = _load_json(path)
    if not ok or data is None:
        return False
    return "regimes" in data


def _validate_regime(path: str) -> bool:
    ok, _data = _load_json(path)
    return ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate live_sim_daily artifacts.")
    parser.add_argument("--summary", required=True, help="Path to summary JSON.")
    parser.add_argument("--equity", required=True, help="Path to equity CSV.")
    parser.add_argument("--meta_state", required=False, help="Path to meta state JSON.")
    parser.add_argument("--regime", required=False, help="Path to regime JSON.")
    parser.add_argument("--strict", type=int, default=0, help="Strict mode (0/1).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    strict = int(args.strict) == 1

    summary_ok, summary = _validate_summary(args.summary)
    equity_ok, df = _validate_equity(args.equity)
    pnl_consistent = _validate_pnl_consistency(df, summary) if summary_ok and equity_ok else False

    if args.meta_state:
        meta_state_ok = _validate_meta_state(args.meta_state)
    else:
        meta_state_ok = not strict

    if args.regime:
        regime_ok = _validate_regime(args.regime)
    else:
        regime_ok = not strict

    print(f"[healthcheck] summary_ok={summary_ok}")
    print(f"[healthcheck] equity_ok={equity_ok}")
    print(f"[healthcheck] pnl_consistent={pnl_consistent}")
    print(f"[healthcheck] meta_state_ok={meta_state_ok}")
    print(f"[healthcheck] regime_ok={regime_ok}")

    all_ok = summary_ok and equity_ok and pnl_consistent and meta_state_ok and regime_ok
    if all_ok:
        print("[healthcheck] STATUS=PASS")
        return 0

    print("[healthcheck] STATUS=FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
