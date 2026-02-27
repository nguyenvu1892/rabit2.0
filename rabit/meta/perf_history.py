from __future__ import annotations

import datetime as dt
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from scripts import deterministic_utils as det

PERF_HISTORY_FILENAME = "perf_history.json"

_MISSING_HASHES = {"", "missing", "skipped", None}


def perf_history_path(meta_state_path: str, filename: str = PERF_HISTORY_FILENAME) -> str:
    if not meta_state_path:
        return ""
    try:
        if os.path.isdir(meta_state_path):
            base_dir = meta_state_path
        else:
            base_dir = os.path.dirname(meta_state_path)
    except Exception:
        base_dir = os.path.dirname(meta_state_path)
    if not base_dir:
        base_dir = "."
    return os.path.join(base_dir, filename)


def _utc_now() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _safe_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        v = int(float(value))
    except Exception:
        return None
    return v


def _is_hash_value(value: Any) -> bool:
    if value in _MISSING_HASHES:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in _MISSING_HASHES
    return False


def _extract_day_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    s = str(value).strip()
    if not s:
        return None
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    for sep in ("T", " "):
        if sep in s:
            part = s.split(sep)[0]
            if len(part) >= 10 and part[4] == "-" and part[7] == "-":
                return part[:10]
    try:
        return dt.date.fromisoformat(s[:10]).isoformat()
    except Exception:
        return None


def compute_summary_from_daily_rows(
    daily_rows: List[Dict[str, Any]],
    total_trades: Optional[int] = None,
) -> Dict[str, Any]:
    days = int(len(daily_rows or []))
    total_pnl = 0.0
    wins = 0
    total_pos = 0.0
    total_neg = 0.0
    max_dd: Optional[float] = None
    max_loss_streak: Optional[int] = None
    start_day: Optional[str] = None
    end_day: Optional[str] = None

    for row in daily_rows or []:
        if not isinstance(row, dict):
            continue
        day_key = _extract_day_key(row.get("day"))
        if day_key:
            if start_day is None:
                start_day = day_key
            end_day = day_key
        pnl = _safe_float_or_none(row.get("day_pnl")) or 0.0
        total_pnl += pnl
        if pnl > 0.0:
            wins += 1
            total_pos += pnl
        elif pnl < 0.0:
            total_neg += pnl
        dd_val = _safe_float_or_none(row.get("intraday_dd"))
        if dd_val is not None and (max_dd is None or dd_val > max_dd):
            max_dd = dd_val
        loss_streak = _safe_int_or_none(row.get("end_loss_streak"))
        if loss_streak is not None and (max_loss_streak is None or loss_streak > max_loss_streak):
            max_loss_streak = loss_streak

    winrate: Optional[float] = None
    if days > 0:
        winrate = float(wins) / float(days)

    profit_factor: Optional[float] = None
    if total_neg < 0.0:
        profit_factor = float(total_pos) / abs(total_neg)

    summary: Dict[str, Any] = {
        "days": int(days),
        "trades": int(total_trades) if total_trades is not None else 0,
        "winrate": winrate,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "total_pnl": float(total_pnl),
    }
    if max_loss_streak is not None:
        summary["max_loss_streak"] = int(max_loss_streak)
    if start_day is not None:
        summary["start_day"] = start_day
    if end_day is not None:
        summary["end_day"] = end_day
    return summary


def build_perf_history(
    candidate_sha256: str,
    source_csv: str,
    run: Dict[str, Any],
    summary: Dict[str, Any],
    created_utc: Optional[str] = None,
    guardrails: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "candidate_sha256": str(candidate_sha256),
        "source_csv": str(source_csv),
        "run": dict(run),
        "summary": dict(summary),
        "timestamps": {"created_utc": created_utc or _utc_now()},
    }
    if guardrails:
        payload["guardrails"] = dict(guardrails)
    return payload


def build_perf_history_from_report(
    report: Dict[str, Any],
    meta_state_path: str,
    source_csv: str,
    created_utc: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(report, dict):
        return None
    determinism = report.get("determinism")
    if not isinstance(determinism, dict):
        return None
    perf_summary = report.get("perf_summary") if isinstance(report.get("perf_summary"), dict) else None
    if perf_summary is None:
        return None

    candidate_sha256 = report.get("meta_state_sha256") or det.sha256_file(meta_state_path)

    run = {
        "input_hash": determinism.get("input_hash"),
        "equity_hash": determinism.get("equity_hash"),
        "regime_ledger_hash": determinism.get("regime_ledger_hash"),
        "total_pnl": _safe_float_or_none(perf_summary.get("total_pnl")),
    }

    summary = {
        "days": _safe_int_or_none(perf_summary.get("days")) or 0,
        "trades": _safe_int_or_none(perf_summary.get("trades")) or 0,
        "winrate": _safe_float_or_none(perf_summary.get("winrate")),
        "profit_factor": _safe_float_or_none(perf_summary.get("profit_factor")),
        "max_dd": _safe_float_or_none(perf_summary.get("max_dd")),
    }
    if "max_loss_streak" in perf_summary:
        summary["max_loss_streak"] = _safe_int_or_none(perf_summary.get("max_loss_streak"))
    if "start_day" in perf_summary:
        summary["start_day"] = perf_summary.get("start_day")
    if "end_day" in perf_summary:
        summary["end_day"] = perf_summary.get("end_day")

    guardrails = None
    guard_src = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    counts_src = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    if guard_src or counts_src:
        guardrails = {
            "final_allowed": _safe_int_or_none(guard_src.get("final_allowed")) or 0,
            "trades_simulated": _safe_int_or_none(counts_src.get("trades_simulated")) or 0,
        }

    payload = build_perf_history(
        candidate_sha256=candidate_sha256,
        source_csv=source_csv,
        run=run,
        summary=summary,
        created_utc=created_utc,
        guardrails=guardrails,
    )
    perf_daily = report.get("perf_daily")
    if isinstance(perf_daily, list):
        payload["daily"] = perf_daily
    return payload


def load_perf_history(path: str) -> Optional[Dict[str, Any]]:
    return det.load_json(path)


def write_perf_history(path: str, payload: Dict[str, Any]) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dir_path or None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(det.stable_json_dumps(payload))
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def validate_perf_history(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if not isinstance(payload, dict):
        return False, ["payload"]

    candidate_sha256 = payload.get("candidate_sha256")
    if not _is_hash_value(candidate_sha256):
        missing.append("candidate_sha256")

    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    if not _is_hash_value(run.get("input_hash")):
        missing.append("run.input_hash")
    if not _is_hash_value(run.get("equity_hash")):
        missing.append("run.equity_hash")
    if not _is_hash_value(run.get("regime_ledger_hash")):
        missing.append("run.regime_ledger_hash")
    if _safe_float_or_none(run.get("total_pnl")) is None:
        missing.append("run.total_pnl")

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    if _safe_int_or_none(summary.get("days")) is None:
        missing.append("summary.days")

    return len(missing) == 0, missing


def perf_metrics_from_history(payload: Dict[str, Any]) -> Dict[str, Any]:
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "days": _safe_int_or_none(summary.get("days")),
        "trades": _safe_int_or_none(summary.get("trades")),
        "winrate": _safe_float_or_none(summary.get("winrate")),
        "profit_factor": _safe_float_or_none(summary.get("profit_factor")),
        "max_dd": _safe_float_or_none(summary.get("max_dd")),
        "max_loss_streak": _safe_int_or_none(summary.get("max_loss_streak")),
        "start_day": summary.get("start_day"),
        "end_day": summary.get("end_day"),
        "total_pnl": _safe_float_or_none(run.get("total_pnl")),
    }


def perf_metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        summary = {}
    return {
        "days": _safe_int_or_none(summary.get("days")),
        "trades": _safe_int_or_none(summary.get("trades")),
        "winrate": _safe_float_or_none(summary.get("winrate")),
        "profit_factor": _safe_float_or_none(summary.get("profit_factor")),
        "max_dd": _safe_float_or_none(summary.get("max_dd")),
        "max_loss_streak": _safe_int_or_none(summary.get("max_loss_streak")),
        "start_day": summary.get("start_day"),
        "end_day": summary.get("end_day"),
        "total_pnl": _safe_float_or_none(summary.get("total_pnl")),
    }


def guardrails_from_history(payload: Dict[str, Any]) -> Dict[str, Any]:
    guard = payload.get("guardrails") if isinstance(payload.get("guardrails"), dict) else {}
    return {
        "final_allowed": _safe_int_or_none(guard.get("final_allowed")),
        "trades_simulated": _safe_int_or_none(guard.get("trades_simulated")),
    }
