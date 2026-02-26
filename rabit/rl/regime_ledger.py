from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if v != v:  # nan
        return default
    return v


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_regime(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return default
    return s


def _extract_date_key(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date().isoformat()
    if isinstance(value, datetime.date):
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
        return datetime.datetime.fromisoformat(s.replace("Z", "")).date().isoformat()
    except Exception:
        return None


def _date_ordinal(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return datetime.date.fromisoformat(str(value)[:10]).toordinal()
    except Exception:
        return None


def _history_sort_key(rec: Dict[str, Any]) -> tuple:
    day = str(rec.get("day", "") or "")
    ord_val = _date_ordinal(day)
    if ord_val is None:
        return (10**9, day)
    return (ord_val, day)


def _ewm(values: List[float], alpha: float) -> float:
    if not values:
        return 0.0
    alpha = float(alpha)
    if alpha <= 0.0:
        return float(values[-1])
    if alpha >= 1.0:
        return float(values[-1])
    ewm = float(values[0])
    for v in values[1:]:
        ewm = (1.0 - alpha) * ewm + alpha * float(v)
    return float(ewm)


@dataclass
class RegimeLedgerConfig:
    span: int = 30
    max_days: int = 365

    @property
    def alpha(self) -> float:
        span = max(1, int(self.span))
        return 2.0 / (float(span) + 1.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeLedgerConfig":
        if not isinstance(data, dict):
            return cls()
        span = _safe_int(data.get("span", 30), 30)
        max_days = _safe_int(data.get("max_days", 365), 365)
        return cls(span=span, max_days=max_days)


class RegimeLedgerState:
    def __init__(self, cfg: Optional[RegimeLedgerConfig] = None) -> None:
        self.cfg = cfg or RegimeLedgerConfig()
        self.history: List[Dict[str, Any]] = []
        self.last_update_ts: Optional[str] = None

    def _infer_regime_for_day(self, day_info: Optional[Dict[str, Any]]) -> str:
        if not isinstance(day_info, dict):
            return "unknown"
        if "regime" in day_info:
            return _normalize_regime(day_info.get("regime"), default="unknown")
        counts = day_info.get("regime_counts")
        if isinstance(counts, dict) and counts:
            items = []
            for k, v in counts.items():
                items.append((_safe_int(v, 0), str(k)))
            items.sort(key=lambda x: (-x[0], x[1]))
            if items:
                return _normalize_regime(items[0][1], default="unknown")
        return "unknown"

    def update_from_daily_outputs(
        self,
        daily_rows: Optional[List[Dict[str, Any]]],
        regime_stats: Optional[Dict[str, Any]],
        debug: bool = False,
    ) -> None:
        if not daily_rows:
            return

        regime_days = []
        if isinstance(regime_stats, dict):
            regime_days = regime_stats.get("days", []) or []

        regime_by_day: Dict[str, str] = {}
        if isinstance(regime_days, list):
            for day_info in regime_days:
                if not isinstance(day_info, dict):
                    continue
                day_key = _extract_date_key(day_info.get("day"))
                if not day_key:
                    continue
                regime_by_day[day_key] = self._infer_regime_for_day(day_info)

        by_day: Dict[str, Dict[str, Any]] = {}
        for rec in self.history:
            if isinstance(rec, dict) and rec.get("day"):
                by_day[str(rec["day"])] = dict(rec)

        for row in daily_rows:
            if not isinstance(row, dict):
                continue
            day_key = _extract_date_key(row.get("day"))
            if not day_key:
                continue
            regime = _normalize_regime(regime_by_day.get(day_key), default="unknown")
            by_day[day_key] = {
                "day": day_key,
                "regime": regime,
                "day_pnl": _safe_float(row.get("day_pnl", 0.0), 0.0),
                "start_equity": _safe_float(row.get("start_equity", 0.0), 0.0),
                "end_equity": _safe_float(row.get("end_equity", 0.0), 0.0),
                "intraday_dd": _safe_float(row.get("intraday_dd", 0.0), 0.0),
                "end_loss_streak": _safe_int(row.get("end_loss_streak", 0), 0),
            }

        history = list(by_day.values())
        history.sort(key=_history_sort_key)
        max_days = max(0, int(self.cfg.max_days))
        if max_days > 0 and len(history) > max_days:
            history = history[-max_days:]
        self.history = history
        self.last_update_ts = datetime.datetime.utcnow().isoformat()
        if debug:
            print(f"[regime_ledger] updated history={len(self.history)} max_days={max_days}")

    def compute_rollups(self) -> Dict[str, Dict[str, Any]]:
        alpha = float(self.cfg.alpha)
        history = sorted(self.history, key=_history_sort_key)
        per_regime: Dict[str, List[Dict[str, Any]]] = {}
        for rec in history:
            if not isinstance(rec, dict):
                continue
            regime = _normalize_regime(rec.get("regime"), default="unknown")
            per_regime.setdefault(regime, []).append(rec)

        out: Dict[str, Dict[str, Any]] = {}
        for regime, recs in per_regime.items():
            pnl_vals: List[float] = []
            win_vals: List[float] = []
            loss_vals: List[float] = []
            for rec in recs:
                pnl = _safe_float(rec.get("day_pnl", 0.0), 0.0)
                pnl_vals.append(pnl)
                win_vals.append(1.0 if pnl > 0.0 else 0.0)
                loss_vals.append(_safe_float(rec.get("end_loss_streak", 0), 0.0))
            out[regime] = {
                "winrate_ewm": _ewm(win_vals, alpha),
                "avg_pnl_ewm": _ewm(pnl_vals, alpha),
                "loss_streak_ewm": _ewm(loss_vals, alpha),
                "n_days": int(len(recs)),
                "last_day": recs[-1].get("day") if recs else None,
            }
        return out

    def compact_summary(self) -> Dict[str, Dict[str, float]]:
        rollups = self.compute_rollups()
        summary: Dict[str, Dict[str, float]] = {}
        for regime, stats in rollups.items():
            summary[regime] = {
                "winrate_ewm": _safe_float(stats.get("winrate_ewm", 0.0), 0.0),
                "avg_pnl_ewm": _safe_float(stats.get("avg_pnl_ewm", 0.0), 0.0),
                "loss_streak_ewm": _safe_float(stats.get("loss_streak_ewm", 0.0), 0.0),
            }
        return summary

    def to_dict(self) -> Dict[str, Any]:
        history = list(self.history)
        history.sort(key=_history_sort_key)
        start_day = history[0].get("day") if history else None
        end_day = history[-1].get("day") if history else None
        return {
            "config": {"span": int(self.cfg.span), "alpha": float(self.cfg.alpha), "max_days": int(self.cfg.max_days)},
            "start_day": start_day,
            "end_day": end_day,
            "history": history,
            "regimes": self.compute_rollups(),
            "last_update_ts": self.last_update_ts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], cfg: Optional[RegimeLedgerConfig] = None) -> "RegimeLedgerState":
        if not isinstance(data, dict):
            data = {}
        if cfg is None:
            cfg = RegimeLedgerConfig.from_dict(data.get("config", {}))
        state = cls(cfg)
        history = data.get("history", [])
        if isinstance(history, list):
            cleaned: List[Dict[str, Any]] = []
            for rec in history:
                if not isinstance(rec, dict):
                    continue
                day_key = _extract_date_key(rec.get("day"))
                if not day_key:
                    continue
                cleaned.append(
                    {
                        "day": day_key,
                        "regime": _normalize_regime(rec.get("regime"), default="unknown"),
                        "day_pnl": _safe_float(rec.get("day_pnl", 0.0), 0.0),
                        "start_equity": _safe_float(rec.get("start_equity", 0.0), 0.0),
                        "end_equity": _safe_float(rec.get("end_equity", 0.0), 0.0),
                        "intraday_dd": _safe_float(rec.get("intraday_dd", 0.0), 0.0),
                        "end_loss_streak": _safe_int(rec.get("end_loss_streak", 0), 0),
                    }
                )
            cleaned.sort(key=_history_sort_key)
            max_days = max(0, int(state.cfg.max_days))
            if max_days > 0 and len(cleaned) > max_days:
                cleaned = cleaned[-max_days:]
            state.history = cleaned
        state.last_update_ts = data.get("last_update_ts")
        return state

    def save(self, path: str) -> bool:
        if not path:
            return False
        try:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, path: str, cfg: Optional[RegimeLedgerConfig] = None) -> "RegimeLedgerState":
        if not path or not os.path.exists(path):
            return cls(cfg)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data, cfg=cfg)
        except Exception:
            return cls(cfg)
