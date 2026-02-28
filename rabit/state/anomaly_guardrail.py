from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

from rabit.state import atomic_io

DEFAULT_DAILY_DD_LIMIT = 0.15
DEFAULT_PNL_JUMP_ABS_LIMIT = 100.0
DEFAULT_TRADES_SPIKE_LIMIT = 2000
DEFAULT_EQUITY_DRIFT_ABS_LIMIT = 5.0

_DAILY_TRADES_KEYS = (
    "trades_today",
    "trades",
    "trade_count",
    "n_trades",
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _load_json_dict(path: str, errors: List[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        payload, _ = atomic_io.load_json_with_fallback(path)
        if isinstance(payload, dict):
            return payload
        errors.append(f"json_not_object path={path}")
    except Exception as exc:
        errors.append(f"json_load_failed path={path} error={exc}")
    return {}


def _load_json_dict_strict(path: str, errors: List[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
        errors.append(f"json_not_object path={path}")
    except Exception as exc:
        errors.append(f"json_load_failed path={path} error={exc}")
    return {}


def _load_last_equity_row(path: str, errors: List[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        if path:
            errors.append(f"equity_missing path={path}")
        return {}
    try:
        last_row: Dict[str, Any] = {}
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if isinstance(row, dict):
                    last_row = row
        if not last_row:
            errors.append(f"equity_empty path={path}")
        return last_row
    except Exception as exc:
        errors.append(f"equity_read_failed path={path} error={exc}")
        return {}


def _extract_last_daily_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    table = summary.get("daily_table")
    if isinstance(table, list) and table:
        last = table[-1]
        if isinstance(last, dict):
            return last
    return {}


def _daily_drawdown_pct(last_daily_row: Dict[str, Any]) -> Optional[float]:
    if not isinstance(last_daily_row, dict) or not last_daily_row:
        return None
    start_equity = _safe_float(last_daily_row.get("start_equity"))
    if start_equity is None or start_equity <= 0.0:
        return None

    intraday_dd = _safe_float(last_daily_row.get("intraday_dd"))
    if intraday_dd is not None:
        return float(intraday_dd) / float(start_equity)

    day_pnl = _safe_float(last_daily_row.get("day_pnl"))
    if day_pnl is not None:
        return max(0.0, -float(day_pnl)) / float(start_equity)
    return None


def _extract_trades_today(
    context: Dict[str, Any],
    summary: Dict[str, Any],
    last_daily_row: Dict[str, Any],
    regime_path: str,
    errors: List[str],
) -> Optional[int]:
    direct = _safe_int(context.get("trades_today"))
    if direct is not None:
        return direct

    for key in _DAILY_TRADES_KEYS:
        value = _safe_int(last_daily_row.get(key))
        if value is not None:
            return value

    value = _safe_int(summary.get("trades_today"))
    if value is not None:
        return value

    regime_payload = _load_json_dict(regime_path, errors)
    days = regime_payload.get("days")
    if isinstance(days, list) and days:
        last_day = days[-1]
        if isinstance(last_day, dict):
            regime_trades = _safe_int(last_day.get("trades"))
            if regime_trades is not None:
                return regime_trades
            # Fallback proxy: allowed signals in the latest day.
            allowed = _safe_int(last_day.get("allowed"))
            if allowed is not None:
                return allowed
    return None


def detect_anomaly(context: Dict[str, Any]) -> Dict[str, Any]:
    ctx = dict(context or {})
    errors: List[str] = []

    summary_path = str(ctx.get("summary_path", "") or "")
    equity_path = str(ctx.get("equity_path", "") or "")
    regime_path = str(ctx.get("regime_path", "") or "")

    daily_dd_limit = _safe_float(ctx.get("daily_dd_limit"))
    if daily_dd_limit is None:
        daily_dd_limit = float(DEFAULT_DAILY_DD_LIMIT)

    pnl_jump_abs_limit = _safe_float(ctx.get("pnl_jump_abs_limit"))
    if pnl_jump_abs_limit is None:
        pnl_jump_abs_limit = float(DEFAULT_PNL_JUMP_ABS_LIMIT)

    trades_spike_limit = _safe_int(ctx.get("trades_spike_limit"))
    if trades_spike_limit is None:
        trades_spike_limit = int(DEFAULT_TRADES_SPIKE_LIMIT)

    equity_drift_abs_limit = _safe_float(ctx.get("equity_drift_abs_limit"))
    if equity_drift_abs_limit is None:
        equity_drift_abs_limit = float(DEFAULT_EQUITY_DRIFT_ABS_LIMIT)

    simulate_anomaly = int(_safe_int(ctx.get("simulate_anomaly")) or 0) == 1

    summary = _load_json_dict(summary_path, errors)
    last_daily_row = _extract_last_daily_row(summary)
    equity_last_row = _load_last_equity_row(equity_path, errors)

    daily_drawdown_pct = _safe_float(ctx.get("daily_drawdown_pct"))
    if daily_drawdown_pct is None:
        daily_drawdown_pct = _daily_drawdown_pct(last_daily_row)

    current_total_pnl = _safe_float(ctx.get("current_total_pnl"))
    if current_total_pnl is None:
        current_total_pnl = _safe_float(summary.get("total_pnl"))

    previous_total_pnl = _safe_float(ctx.get("previous_total_pnl"))
    if previous_total_pnl is None and summary_path:
        previous_payload = _load_json_dict_strict(f"{summary_path}.bak", errors)
        previous_total_pnl = _safe_float(previous_payload.get("total_pnl"))

    pnl_jump_abs = None
    if current_total_pnl is not None and previous_total_pnl is not None:
        pnl_jump_abs = abs(float(current_total_pnl) - float(previous_total_pnl))

    trades_today = _extract_trades_today(
        ctx,
        summary,
        last_daily_row,
        regime_path,
        errors,
    )

    summary_equity = _safe_float(ctx.get("summary_equity"))
    if summary_equity is None:
        summary_equity = _safe_float(last_daily_row.get("end_equity"))

    equity_csv_last = _safe_float(ctx.get("equity_csv_last"))
    if equity_csv_last is None:
        equity_csv_last = _safe_float(equity_last_row.get("equity"))
    if equity_csv_last is None:
        equity_csv_last = _safe_float(equity_last_row.get("end_equity"))

    equity_drift_abs = None
    if summary_equity is not None and equity_csv_last is not None:
        equity_drift_abs = abs(float(summary_equity) - float(equity_csv_last))

    triggered_rules: List[str] = []
    if simulate_anomaly:
        triggered_rules.append("SIMULATED_ANOMALY")

    if daily_drawdown_pct is not None and abs(float(daily_drawdown_pct)) > float(daily_dd_limit):
        triggered_rules.append("DAILY_DD_LIMIT_EXCEEDED")

    if pnl_jump_abs is not None and abs(float(pnl_jump_abs)) > float(pnl_jump_abs_limit):
        triggered_rules.append("PNL_JUMP_ABS_LIMIT_EXCEEDED")

    if trades_today is not None and int(trades_today) > int(trades_spike_limit):
        triggered_rules.append("TRADES_SPIKE_LIMIT_EXCEEDED")

    if equity_drift_abs is not None and abs(float(equity_drift_abs)) > float(equity_drift_abs_limit):
        triggered_rules.append("EQUITY_DRIFT_ABS_LIMIT_EXCEEDED")

    metrics = {
        "summary_path": summary_path,
        "equity_path": equity_path,
        "regime_path": regime_path,
        "daily_drawdown_pct": daily_drawdown_pct,
        "daily_dd_limit": float(daily_dd_limit),
        "current_total_pnl": current_total_pnl,
        "previous_total_pnl": previous_total_pnl,
        "pnl_jump_abs": pnl_jump_abs,
        "pnl_jump_abs_limit": float(pnl_jump_abs_limit),
        "trades_today": trades_today,
        "trades_spike_limit": int(trades_spike_limit),
        "summary_equity": summary_equity,
        "equity_csv_last": equity_csv_last,
        "equity_drift_abs": equity_drift_abs,
        "equity_drift_abs_limit": float(equity_drift_abs_limit),
        "simulate_anomaly": 1 if simulate_anomaly else 0,
        "triggered_rules": triggered_rules,
        "errors": errors,
    }
    reason_code = triggered_rules[0] if triggered_rules else "NONE"
    return {
        "is_anomaly": bool(triggered_rules),
        "reason_code": reason_code,
        "metrics": metrics,
    }

