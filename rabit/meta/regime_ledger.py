from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

DET_REGIME_LEDGER_EXCLUDE: Set[str] = {"last_update_ts"}


def _value_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if isinstance(value, float) and np.isnan(value):
            return False
        if pd.isna(value):
            return False
    except Exception:
        pass
    return True


def _safe_str(value: Any, default: str = "") -> str:
    if not _value_present(value):
        return default
    s = str(value)
    if not s or s.lower() == "nan":
        return default
    return s


def _safe_optional_str(value: Any) -> Optional[str]:
    if not _value_present(value):
        return None
    s = str(value)
    if not s or s.lower() == "nan":
        return None
    return s


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int_or_none(value: Any) -> Optional[int]:
    if not _value_present(value):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def stable_json_dumps(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def canonicalize_regime_ledger(
    raw: Any,
    exclude_fields: Optional[Set[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    excluded = sorted(exclude_fields or DET_REGIME_LEDGER_EXCLUDE)
    meta: Dict[str, Any] = {
        "included_fields": ["config", "start_day", "end_day", "history", "regimes"],
        "excluded_fields": excluded,
    }
    if not isinstance(raw, dict):
        meta["status"] = "missing_or_invalid"
        return None, meta

    config_raw = raw.get("config", {})
    config_out: Dict[str, Any] = {}
    if isinstance(config_raw, dict):
        span = _safe_int_or_none(config_raw.get("span"))
        max_days = _safe_int_or_none(config_raw.get("max_days"))
        alpha = _safe_float(config_raw.get("alpha", 0.0), 0.0)
        config_out = {
            "span": int(span) if span is not None else 0,
            "max_days": int(max_days) if max_days is not None else 0,
            "alpha": float(alpha),
        }

    history_raw = raw.get("history", [])
    history_out: List[Dict[str, Any]] = []
    if isinstance(history_raw, list):
        for rec in history_raw:
            if not isinstance(rec, dict):
                continue
            day = _safe_optional_str(rec.get("day"))
            if not day:
                continue
            history_out.append(
                {
                    "day": day,
                    "regime": _safe_str(rec.get("regime"), default="unknown"),
                    "day_pnl": _safe_float(rec.get("day_pnl", 0.0), 0.0),
                    "start_equity": _safe_float(rec.get("start_equity", 0.0), 0.0),
                    "end_equity": _safe_float(rec.get("end_equity", 0.0), 0.0),
                    "intraday_dd": _safe_float(rec.get("intraday_dd", 0.0), 0.0),
                    "end_loss_streak": int(_safe_int_or_none(rec.get("end_loss_streak", 0)) or 0),
                }
            )
    history_out.sort(key=lambda r: str(r.get("day", "")))

    regimes_out: Dict[str, Dict[str, Any]] = {}
    regimes_raw = raw.get("regimes", {})
    if isinstance(regimes_raw, dict):
        for regime_key, stats in regimes_raw.items():
            if not isinstance(stats, dict):
                continue
            regimes_out[str(regime_key)] = {
                "winrate_ewm": _safe_float(stats.get("winrate_ewm", 0.0), 0.0),
                "avg_pnl_ewm": _safe_float(stats.get("avg_pnl_ewm", 0.0), 0.0),
                "loss_streak_ewm": _safe_float(stats.get("loss_streak_ewm", 0.0), 0.0),
                "n_days": int(_safe_int_or_none(stats.get("n_days", 0)) or 0),
                "last_day": _safe_optional_str(stats.get("last_day")),
            }

    start_day = _safe_optional_str(raw.get("start_day"))
    end_day = _safe_optional_str(raw.get("end_day"))
    if start_day is None and history_out:
        start_day = history_out[0].get("day")
    if end_day is None and history_out:
        end_day = history_out[-1].get("day")

    meta.update(
        {
            "history_len": int(len(history_out)),
            "regime_keys": sorted(regimes_out.keys()),
            "start_day": start_day,
            "end_day": end_day,
        }
    )

    canonical = {
        "config": config_out,
        "start_day": start_day,
        "end_day": end_day,
        "history": history_out,
        "regimes": regimes_out,
    }
    return canonical, meta


def hash_regime_ledger_dict(
    raw: Any,
    debug: bool = False,
    exclude_fields: Optional[Set[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    canonical, meta = canonicalize_regime_ledger(raw, exclude_fields=exclude_fields)
    if canonical is None:
        status = "missing"
        if isinstance(raw, dict):
            status = "invalid"
        meta["status"] = status
        if debug:
            print(f"[deterministic] regime_ledger hash skipped: status={status}")
        return status, meta

    if debug:
        print(
            "[deterministic] regime_ledger canonicalize: sort_keys=True separators=(',', ':') "
            "ensure_ascii=False"
        )
        print(
            "[deterministic] regime_ledger hash_include="
            f"{meta.get('included_fields')} hash_exclude={meta.get('excluded_fields')}"
        )
        print(
            "[deterministic] regime_ledger history_sort=day asc "
            f"history_len={meta.get('history_len')} regimes={meta.get('regime_keys')}"
        )

    canonical_json = stable_json_dumps(canonical)
    return _sha256_text(canonical_json), meta


def hash_regime_ledger_file(
    path: str,
    debug: bool = False,
    exclude_fields: Optional[Set[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    raw = None
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = None
    return hash_regime_ledger_dict(raw, debug=debug, exclude_fields=exclude_fields)
