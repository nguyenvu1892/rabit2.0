from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple
import datetime

_NONZERO_EPS = 1e-6


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    if hi < lo:
        hi = lo
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _normalize_regime(regime: Any, default: str = "") -> str:
    if regime is None:
        return default
    s = str(regime).strip()
    if not s or s.lower() == "nan":
        return default
    return s


def _normalize_ts(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    return str(value)


def _normalize_optional_ts(value: Any) -> Optional[str]:
    if value is None:
        return None
    ts = _normalize_ts(value)
    return ts if ts else None


@dataclass(init=False)
class MetaRiskConfig:
    min_trades_per_regime: int
    ewma_alpha: float
    score_clip: float
    min_scale: float
    max_scale: float
    k: float
    daily_dd_limit: float
    loss_streak_limit: int
    freeze_days: int
    global_floor: float
    scale_min: float
    scale_max: float
    down_power: float
    up_power: float
    hysteresis: float
    cooldown_trades: int
    cooldown_days: int
    global_fallback_scale: float
    target_pf: float
    target_return: float
    dd_soft_limit: float
    global_min_scale: float
    global_max_scale: float
    global_enter_risk_off: float
    global_exit_risk_off: float
    global_cooldown_days: int

    def __init__(
        self,
        min_trades_per_regime: int = 50,
        ewma_alpha: float = 0.25,
        score_clip: float = 3.0,
        min_scale: float = 0.60,
        max_scale: float = 1.30,
        k: float = 1.0,
        daily_dd_limit: float = 3.0,
        loss_streak_limit: int = 5,
        freeze_days: int = 3,
        global_floor: float = 0.2,
        alpha: Optional[float] = None,
        floor: Optional[float] = None,
        cap: Optional[float] = None,
        scale_min: Optional[float] = None,
        scale_max: Optional[float] = None,
        meta_scale_min: Optional[float] = None,
        meta_scale_max: Optional[float] = None,
        down_power: float = 1.20,
        up_power: float = 0.80,
        hysteresis: float = 0.05,
        cooldown_trades: int = 5,
        cooldown_days: int = 2,
        global_fallback_scale: float = 1.0,
        target_pf: float = 1.02,
        target_return: float = 0.0,
        dd_soft_limit: float = 0.0,
        global_min_scale: float = 0.6,
        global_max_scale: float = 1.2,
        global_enter_risk_off: float = -0.35,
        global_exit_risk_off: float = -0.10,
        global_cooldown_days: int = 3,
    ) -> None:
        if alpha is not None:
            ewma_alpha = alpha
        if scale_min is not None:
            min_scale = scale_min
        if scale_max is not None:
            max_scale = scale_max
        if meta_scale_min is not None:
            min_scale = meta_scale_min
        if meta_scale_max is not None:
            max_scale = meta_scale_max
        if floor is not None:
            min_scale = floor
        if cap is not None:
            max_scale = cap

        if global_fallback_scale is None:
            global_fallback_scale = 1.0

        self.min_trades_per_regime = int(min_trades_per_regime)
        self.ewma_alpha = float(ewma_alpha)
        self.score_clip = float(score_clip)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        if self.max_scale < self.min_scale:
            self.max_scale = self.min_scale
        self.k = float(k)
        self.daily_dd_limit = float(daily_dd_limit)
        self.loss_streak_limit = int(loss_streak_limit)
        self.freeze_days = int(freeze_days)
        self.global_floor = float(global_floor)
        self.scale_min = float(self.min_scale)
        self.scale_max = float(self.max_scale)
        self.down_power = float(down_power)
        self.up_power = float(up_power)
        self.hysteresis = float(hysteresis)
        self.cooldown_trades = int(cooldown_trades)
        self.cooldown_days = int(cooldown_days)
        self.global_fallback_scale = float(global_fallback_scale)
        self.target_pf = float(target_pf)
        self.target_return = float(target_return)
        self.dd_soft_limit = float(dd_soft_limit)
        self.global_min_scale = float(global_min_scale)
        self.global_max_scale = float(global_max_scale)
        if self.global_max_scale < self.global_min_scale:
            self.global_max_scale = self.global_min_scale
        self.global_enter_risk_off = float(global_enter_risk_off)
        self.global_exit_risk_off = float(global_exit_risk_off)
        self.global_cooldown_days = int(global_cooldown_days)

    @property
    def alpha(self) -> float:
        return self.ewma_alpha

    @property
    def floor(self) -> float:
        return self.min_scale

    @property
    def cap(self) -> float:
        return self.max_scale

    @property
    def meta_scale_min(self) -> float:
        return self.min_scale

    @property
    def meta_scale_max(self) -> float:
        return self.max_scale


@dataclass
class RegimeStats:
    n_trades: int = 0
    ewma_pnl: float = 0.0
    ewma_abs_pnl: float = 0.0
    ewma_win: float = 0.0
    ewma_loss: float = 0.0
    loss_streak: int = 0
    ewma_loss_streak: float = 0.0
    last_update_ts: Optional[str] = None
    last_scale: float = 1.0
    last_scale_update_n: int = 0
    last_scale_update_date: Optional[str] = None
    last_meta_reason: Optional[str] = None
    ewma_return: float = 0.0
    ewma_vol: float = 0.0
    ewma_winrate: float = 0.0


def _normalize_cfg_kwargs(cfg_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg_data, dict):
        return {}

    cfg = dict(cfg_data)
    if "alpha" in cfg and "ewma_alpha" not in cfg:
        cfg["ewma_alpha"] = cfg["alpha"]
    if "floor" in cfg and "min_scale" not in cfg:
        cfg["min_scale"] = cfg["floor"]
    if "cap" in cfg and "max_scale" not in cfg:
        cfg["max_scale"] = cfg["cap"]
    if "meta_scale_min" in cfg and "min_scale" not in cfg:
        cfg["min_scale"] = cfg["meta_scale_min"]
    if "meta_scale_max" in cfg and "max_scale" not in cfg:
        cfg["max_scale"] = cfg["meta_scale_max"]

    allowed = {
        "min_trades_per_regime",
        "ewma_alpha",
        "score_clip",
        "min_scale",
        "max_scale",
        "k",
        "daily_dd_limit",
        "loss_streak_limit",
        "freeze_days",
        "global_floor",
        "scale_min",
        "scale_max",
        "down_power",
        "up_power",
        "hysteresis",
        "cooldown_trades",
        "cooldown_days",
        "global_fallback_scale",
        "target_pf",
        "target_return",
        "dd_soft_limit",
        "global_min_scale",
        "global_max_scale",
        "global_enter_risk_off",
        "global_exit_risk_off",
        "global_cooldown_days",
        "alpha",
        "floor",
        "cap",
        "meta_scale_min",
        "meta_scale_max",
    }
    return {k: cfg[k] for k in allowed if k in cfg}


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


class MetaRiskState:
    def __init__(self, cfg: MetaRiskConfig) -> None:
        self.cfg = cfg
        self.regimes: Dict[str, RegimeStats] = {}
        self.stats = self.regimes
        self.daily_equity_peak = 0.0
        self.daily_drawdown = 0.0
        self.loss_streak = 0
        self.regime_freeze_until: Dict[str, str] = {}
        self.daily_date: Optional[str] = None
        self.last_guard_reason: str = "ok"
        self.last_guard_regime: Optional[str] = None
        self.last_guard_date: Optional[str] = None
        self.last_meta_reason_by_regime: Dict[str, str] = {}
        self.global_trades: int = 0
        self.global_pnl_ewm: float = 0.0
        self.global_abs_pnl_ewm: float = 0.0
        self.global_winrate_ewm: float = 0.0
        self.global_dd_ewm: float = 0.0
        self.global_loss_streak: int = 0
        self.global_loss_streak_ewm: float = 0.0
        self.global_equity: float = 0.0
        self.global_equity_peak: float = 0.0
        self.global_drawdown: float = 0.0
        self.global_state: str = "risk_on"
        self.global_state_change_date: Optional[str] = None
        self.global_last_risk_off_date: Optional[str] = None
        self.global_last_state_change_trade_n: int = 0
        self.global_last_update_ts: Optional[str] = None
        self.global_last_scale: float = 1.0
        self.global_last_reason: str = "global_ok"
        self.global_last_score: float = 0.0

    def _set_guard_event(self, regime: str, reason: str) -> None:
        self.last_guard_reason = str(reason) if reason else "ok"
        self.last_guard_regime = str(regime) if regime is not None else None
        self.last_guard_date = self.daily_date

    def get_guard_reason(self, regime: str) -> str:
        if regime:
            if self.last_guard_regime == regime:
                return self.last_guard_reason or "ok"
            return "ok"
        if self.last_guard_regime in (None, ""):
            return self.last_guard_reason or "ok"
        return "ok"

    def get_meta_reason(self, regime: str) -> str:
        regime_key = _normalize_regime(regime)
        if regime_key in self.last_meta_reason_by_regime:
            return self.last_meta_reason_by_regime[regime_key]
        st = self.regimes.get(regime_key)
        if st is not None and st.last_meta_reason:
            return st.last_meta_reason
        return "ok"

    def _scale_bounds(self) -> Tuple[float, float]:
        min_scale = _safe_float(
            getattr(self.cfg, "meta_scale_min", None),
            _safe_float(getattr(self.cfg, "scale_min", None), _safe_float(getattr(self.cfg, "min_scale", 0.0), 0.0)),
        )
        max_scale = _safe_float(
            getattr(self.cfg, "meta_scale_max", None),
            _safe_float(getattr(self.cfg, "scale_max", None), _safe_float(getattr(self.cfg, "max_scale", 1.0), 1.0)),
        )
        if min_scale <= 0.0:
            min_scale = _NONZERO_EPS
        if max_scale < min_scale:
            max_scale = min_scale
        return float(min_scale), float(max_scale)

    def _performance_hysteresis_bounds(self) -> Tuple[float, float]:
        gap = abs(_safe_float(getattr(self.cfg, "hysteresis", 0.0), 0.0))
        lower = -gap
        upper = gap
        if upper < lower:
            upper = lower
        return float(lower), float(upper)

    def _cooldown_days_elapsed(self, last_date: Optional[str], current_date: Optional[str]) -> bool:
        cooldown_days = max(0, _safe_int(getattr(self.cfg, "cooldown_days", 0), 0))
        if cooldown_days <= 0:
            return True
        if not last_date or not current_date:
            return True
        last_ord = _date_ordinal(last_date)
        cur_ord = _date_ordinal(current_date)
        if last_ord is None or cur_ord is None:
            return True
        return (cur_ord - last_ord) >= cooldown_days

    def _global_scale_bounds(self) -> Tuple[float, float]:
        min_scale = _safe_float(
            getattr(self.cfg, "global_min_scale", None),
            _safe_float(getattr(self.cfg, "global_floor", 0.2), 0.2),
        )
        max_scale = _safe_float(getattr(self.cfg, "global_max_scale", None), 1.2)
        if min_scale <= 0.0:
            min_scale = _NONZERO_EPS
        if max_scale < min_scale:
            max_scale = min_scale
        return float(min_scale), float(max_scale)

    def _fallback_scale(self) -> float:
        min_scale, max_scale = self._scale_bounds()
        fallback = _safe_float(getattr(self.cfg, "global_fallback_scale", 1.0), 1.0)
        if fallback <= 0.0:
            fallback = 1.0
        return _clamp(fallback, min_scale, max_scale)

    def _performance_score(self, st: RegimeStats) -> float:
        pnl = _safe_float(getattr(st, "ewma_pnl", None), _safe_float(getattr(st, "ewma_return", 0.0), 0.0))
        abs_pnl = _safe_float(getattr(st, "ewma_abs_pnl", None), _safe_float(getattr(st, "ewma_vol", 0.0), 0.0))
        if abs_pnl <= 1e-12:
            abs_pnl = _safe_float(getattr(st, "ewma_vol", 0.0), 0.0)
        if abs_pnl <= 1e-12:
            return 0.0
        perf = pnl / (abs_pnl + 1e-12)
        if perf > 1.0:
            perf = 1.0
        if perf < -1.0:
            perf = -1.0
        return float(perf)

    def _performance_score_from_vals(self, pnl: float, abs_pnl: float) -> float:
        pnl = _safe_float(pnl, 0.0)
        abs_pnl = _safe_float(abs_pnl, 0.0)
        if abs_pnl <= 1e-12:
            return 0.0
        perf = pnl / (abs_pnl + 1e-12)
        if perf > 1.0:
            perf = 1.0
        if perf < -1.0:
            perf = -1.0
        return float(perf)

    def _global_health_score(self) -> float:
        if _safe_int(getattr(self, "global_trades", 0), 0) <= 0:
            return 0.0
        perf = self._performance_score_from_vals(self.global_pnl_ewm, self.global_abs_pnl_ewm)
        winrate = _safe_float(self.global_winrate_ewm, 0.0)
        win_score = _clamp((winrate - 0.5) / 0.5, -1.0, 1.0)
        dd = _safe_float(self.global_dd_ewm, 0.0)
        denom = max(abs(self.global_pnl_ewm), self.global_abs_pnl_ewm, 1e-6)
        dd_ratio = _clamp(dd / (denom + 1e-12), 0.0, 1.0)
        loss_limit = max(1.0, float(_safe_int(getattr(self.cfg, "loss_streak_limit", 5), 5)))
        loss_ratio = _clamp(self.global_loss_streak_ewm / loss_limit, 0.0, 1.0)

        score = 0.55 * perf + 0.30 * win_score - 0.15 * dd_ratio - 0.10 * loss_ratio
        return _clamp(score, -1.0, 1.0)

    def _apply_dd_soft_limit(self, scale: float) -> float:
        limit = _safe_float(getattr(self.cfg, "dd_soft_limit", 0.0), 0.0)
        if limit <= 0.0:
            return scale
        dd = _safe_float(self.daily_drawdown, 0.0)
        if dd <= limit:
            return scale
        over = dd - limit
        denom = max(limit, 1e-12)
        penalty = min(0.5, 0.5 * (over / denom))
        scale = scale * (1.0 - penalty)
        min_scale, max_scale = self._scale_bounds()
        return _clamp(scale, min_scale, max_scale)

    def _global_cooldown_elapsed(self, date_key: Optional[str]) -> bool:
        cooldown_days = max(0, _safe_int(getattr(self.cfg, "global_cooldown_days", 0), 0))
        if cooldown_days <= 0:
            return True
        last_date = self.global_last_risk_off_date
        if not last_date or not date_key:
            return True
        last_ord = _date_ordinal(last_date)
        cur_ord = _date_ordinal(date_key)
        if last_ord is None or cur_ord is None:
            return True
        return (cur_ord - last_ord) >= cooldown_days

    def _update_global_state(self, date_key: Optional[str]) -> None:
        enter = _safe_float(getattr(self.cfg, "global_enter_risk_off", -0.35), -0.35)
        exit = _safe_float(getattr(self.cfg, "global_exit_risk_off", -0.10), -0.10)
        if exit < enter:
            exit = enter
        score = self._global_health_score()
        self.global_last_score = score

        if self.global_state != "risk_off":
            if score <= enter:
                self.global_state = "risk_off"
                self.global_state_change_date = date_key
                self.global_last_risk_off_date = date_key
                self.global_last_state_change_trade_n = _safe_int(self.global_trades, 0)
        else:
            if self._global_cooldown_elapsed(date_key) and score >= exit:
                self.global_state = "risk_on"
                self.global_state_change_date = date_key
                self.global_last_state_change_trade_n = _safe_int(self.global_trades, 0)

    def global_scale_with_reason(self, update_state: bool = False) -> Tuple[float, str]:
        min_scale, max_scale = self._global_scale_bounds()
        if _safe_int(getattr(self, "global_trades", 0), 0) <= 0:
            return 1.0, "global_fallback_no_trades"

        score = self._global_health_score()
        down_power = max(0.01, _safe_float(getattr(self.cfg, "down_power", 1.0), 1.0))
        up_power = max(0.01, _safe_float(getattr(self.cfg, "up_power", 1.0), 1.0))

        if self.global_state == "risk_off":
            scale = min_scale
            reason = "global_risk_off"
        else:
            if score < 0.0:
                scale = 1.0 - (abs(score) ** down_power) * (1.0 - min_scale)
                reason = "global_down"
            elif score > 0.0:
                scale = 1.0 + (score ** up_power) * (max_scale - 1.0)
                reason = "global_up"
            else:
                scale = 1.0
                reason = "global_ok"

        scale = _clamp(scale, min_scale, max_scale)
        if update_state:
            self.global_last_scale = scale
            self.global_last_reason = reason
            self.global_last_score = score
        return scale, reason

    def global_scale(self) -> float:
        scale, _reason = self.global_scale_with_reason(update_state=False)
        return float(scale)

    def _scale_from_perf(self, perf: float) -> float:
        min_scale, max_scale = self._scale_bounds()
        down_power = max(0.01, _safe_float(getattr(self.cfg, "down_power", 1.0), 1.0))
        up_power = max(0.01, _safe_float(getattr(self.cfg, "up_power", 1.0), 1.0))
        if perf < 0.0:
            scale = 1.0 - (abs(perf) ** down_power) * (1.0 - min_scale)
        else:
            scale = 1.0 + (perf ** up_power) * (max_scale - 1.0)
        scale = _clamp(scale, min_scale, max_scale)
        return self._apply_dd_soft_limit(scale)

    def _scale_from_stats(self, st: RegimeStats) -> float:
        perf = self._performance_score(st)
        return self._scale_from_perf(perf)

    def _compute_scale(self, regime: str, st: Optional[RegimeStats], update_state: bool) -> Tuple[float, str]:
        min_scale, max_scale = self._scale_bounds()
        base = _clamp(1.0, min_scale, max_scale)
        dd_limit = _safe_float(getattr(self.cfg, "daily_dd_limit", 0.0), 0.0)
        if dd_limit > 0.0 and self.daily_drawdown > dd_limit:
            return base, "freeze_dd"
        if not regime:
            return base, "fallback_no_regime"
        if st is None:
            return base, "fallback_no_regime"

        n_trades = _safe_int(getattr(st, "n_trades", 0), 0)
        min_trades = _safe_int(getattr(self.cfg, "min_trades_per_regime", 1), 1)
        if n_trades < min_trades:
            return base, "fallback_min_trades"

        perf = self._performance_score(st)
        candidate = self._scale_from_perf(perf)
        last_scale = _safe_float(getattr(st, "last_scale", base), base)
        last_scale = _clamp(last_scale, min_scale, max_scale)

        lower_thr, upper_thr = self._performance_hysteresis_bounds()
        if perf <= lower_thr:
            if candidate >= last_scale:
                return last_scale, "hold_hysteresis_down"
        elif perf >= upper_thr:
            if candidate <= last_scale:
                return last_scale, "hold_hysteresis_up"
        else:
            return last_scale, "hold_hysteresis_gap"

        hysteresis = max(0.0, _safe_float(getattr(self.cfg, "hysteresis", 0.0), 0.0))
        if abs(candidate - last_scale) < hysteresis:
            return last_scale, "hold_hysteresis"

        current_date = self.daily_date
        if not current_date and st.last_update_ts:
            current_date = _extract_date_key(st.last_update_ts)
        last_scale_date = _normalize_optional_ts(getattr(st, "last_scale_update_date", None))
        if not self._cooldown_days_elapsed(last_scale_date, current_date):
            return last_scale, "hold_cooldown_days"

        cooldown = max(0, _safe_int(getattr(self.cfg, "cooldown_trades", 0), 0))
        last_update_n = _safe_int(getattr(st, "last_scale_update_n", 0), 0)
        trades_since = max(0, n_trades - last_update_n)
        if cooldown > 0 and trades_since < cooldown:
            return last_scale, "hold_cooldown"

        reason = "calib_up" if candidate > last_scale else "calib_down"
        if update_state:
            st.last_scale = candidate
            st.last_scale_update_n = n_trades
            if current_date:
                st.last_scale_update_date = str(current_date)
            st.last_meta_reason = reason
        return candidate, reason

    def _update_global_metrics(self, pnl_value: float, ts_norm: str, date_key: Optional[str]) -> None:
        alpha = float(self.cfg.ewma_alpha)
        self.global_trades = _safe_int(self.global_trades, 0) + 1

        self.global_pnl_ewm = (1.0 - alpha) * self.global_pnl_ewm + alpha * pnl_value
        self.global_abs_pnl_ewm = (1.0 - alpha) * self.global_abs_pnl_ewm + alpha * abs(pnl_value)
        win = 1.0 if pnl_value > 0.0 else 0.0
        self.global_winrate_ewm = (1.0 - alpha) * self.global_winrate_ewm + alpha * win

        if pnl_value > 0.0:
            self.global_loss_streak = 0
        else:
            self.global_loss_streak = _safe_int(self.global_loss_streak, 0) + 1
        self.global_loss_streak_ewm = (1.0 - alpha) * self.global_loss_streak_ewm + alpha * self.global_loss_streak

        self.global_equity += pnl_value
        if self.global_equity_peak == 0.0 and self.global_drawdown == 0.0:
            self.global_equity_peak = self.global_equity
        self.global_equity_peak = max(self.global_equity_peak, self.global_equity)
        dd_now = self.global_equity_peak - self.global_equity
        self.global_drawdown = dd_now
        self.global_dd_ewm = (1.0 - alpha) * self.global_dd_ewm + alpha * dd_now

        if ts_norm:
            self.global_last_update_ts = ts_norm
        elif date_key:
            self.global_last_update_ts = str(date_key)

    def update_trade(
        self,
        regime: str,
        pnl_return: float,
        ts: Optional[str] = None,
        date: Optional[str] = None,
    ) -> None:
        if ts is None:
            ts = date
        ts_norm = _normalize_ts(ts)

        regime_key = _normalize_regime(regime, default="unknown")
        stats = self.regimes.get(regime_key)
        if stats is None:
            stats = RegimeStats(last_scale=self._fallback_scale())
            self.regimes[regime_key] = stats

        alpha = float(self.cfg.ewma_alpha)
        pnl_value = _safe_float(pnl_return, 0.0)

        stats.ewma_pnl = (1.0 - alpha) * stats.ewma_pnl + alpha * pnl_value
        stats.ewma_abs_pnl = (1.0 - alpha) * stats.ewma_abs_pnl + alpha * abs(pnl_value)
        win = 1.0 if pnl_value > 0.0 else 0.0
        stats.ewma_win = (1.0 - alpha) * stats.ewma_win + alpha * win
        stats.ewma_loss = (1.0 - alpha) * stats.ewma_loss + alpha * max(-pnl_value, 0.0)
        stats.n_trades += 1
        stats.last_update_ts = ts_norm

        stats.ewma_return = stats.ewma_pnl
        stats.ewma_vol = stats.ewma_abs_pnl
        stats.ewma_winrate = stats.ewma_win

        if pnl_value > 0.0:
            stats.loss_streak = 0
        else:
            stats.loss_streak += 1
        stats.ewma_loss_streak = (1.0 - alpha) * stats.ewma_loss_streak + alpha * stats.loss_streak

        date_key = _extract_date_key(ts_norm or date)
        if date_key is not None:
            self.daily_date = date_key

        self._update_global_metrics(pnl_value, ts_norm, date_key)
        self._update_global_state(date_key)

        if pnl_value > 0.0:
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            if (
                self.cfg.loss_streak_limit > 0
                and self.loss_streak >= self.cfg.loss_streak_limit
                and regime_key
                and date_key is not None
                and self.cfg.freeze_days > 0
            ):
                dt = None
                if isinstance(date_key, str) and len(date_key) >= 10:
                    try:
                        dt = datetime.date.fromisoformat(date_key[:10])
                    except Exception:
                        dt = None
                if dt is None:
                    dt = datetime.datetime.utcnow().date()
                freeze_until = dt + datetime.timedelta(days=self.cfg.freeze_days)
                self.regime_freeze_until[regime_key] = freeze_until.isoformat()

        scale, reason = self.meta_scale_with_reason(regime_key, update_state=True)
        stats.last_meta_reason = reason
        self.last_meta_reason_by_regime[regime_key] = reason

    def update_daily_equity(self, equity: float, date: str) -> None:
        equity = float(equity)
        date_key = _extract_date_key(date)
        if date_key is not None and self.daily_date != date_key:
            self.daily_date = date_key
            self.daily_equity_peak = equity
            self.daily_drawdown = 0.0
        if self.daily_equity_peak == 0.0 and self.daily_drawdown == 0.0:
            self.daily_equity_peak = equity
        self.daily_equity_peak = max(self.daily_equity_peak, equity)
        dd_now = self.daily_equity_peak - equity
        if dd_now > self.daily_drawdown:
            self.daily_drawdown = dd_now

    def _is_regime_frozen(self, regime: str) -> bool:
        if not regime:
            return False
        until = self.regime_freeze_until.get(regime)
        if not until:
            return False
        current = self.daily_date
        cur_ord = _date_ordinal(current)
        until_ord = _date_ordinal(until)
        if cur_ord is None or until_ord is None:
            return False
        if cur_ord <= until_ord:
            return True
        self.regime_freeze_until.pop(regime, None)
        return False

    def apply_guardrails(self, regime: str, size: float) -> float:
        size = float(size)
        if size <= 0.0:
            self._set_guard_event(regime, "ok")
            return 0.0
        if self.daily_drawdown > self.cfg.daily_dd_limit:
            self._set_guard_event(regime, "daily_dd_stop")
            return 0.0
        guard_reason = "ok"
        if self.cfg.loss_streak_limit > 0 and self.loss_streak >= self.cfg.loss_streak_limit:
            guard_reason = "loss_streak"
            size *= 0.5
        if self._is_regime_frozen(regime):
            self._set_guard_event(regime, "regime_frozen")
            return 0.0
        size = float(max(0.0, min(1.0, size)))
        self._set_guard_event(regime, guard_reason)
        return size

    def meta_scale_with_reason(self, regime: str, update_state: bool = False) -> Tuple[float, str]:
        regime_key = _normalize_regime(regime)
        st = self.regimes.get(regime_key)
        regime_scale, regime_reason = self._compute_scale(regime_key, st, update_state)
        global_scale, global_reason = self.global_scale_with_reason(update_state=update_state)
        min_scale, max_scale = self._scale_bounds()
        scale = _clamp(regime_scale * global_scale, min_scale, max_scale)
        reason = regime_reason
        if global_reason and global_reason != "global_ok":
            reason = f"{regime_reason}|{global_reason}"
        if regime_key:
            self.last_meta_reason_by_regime[regime_key] = reason
            if st is not None and st.last_meta_reason is None:
                st.last_meta_reason = reason
        return scale, reason

    def meta_scale(self, regime: str) -> float:
        """
        Return a risk scaling factor in [scale_min, scale_max].
        Warmup: if not enough trades yet -> return fallback scale (no interference).
        Applies regime scale * global scale with clamp.
        """
        scale, _reason = self.meta_scale_with_reason(regime, update_state=False)
        return scale

    def global_metrics_snapshot(self) -> Dict[str, Any]:
        return {
            "global_trades": int(_safe_int(self.global_trades, 0)),
            "global_pnl_ewm": float(_safe_float(self.global_pnl_ewm, 0.0)),
            "global_abs_pnl_ewm": float(_safe_float(self.global_abs_pnl_ewm, 0.0)),
            "global_winrate_ewm": float(_safe_float(self.global_winrate_ewm, 0.0)),
            "global_dd_ewm": float(_safe_float(self.global_dd_ewm, 0.0)),
            "global_loss_streak": int(_safe_int(self.global_loss_streak, 0)),
            "global_loss_streak_ewm": float(_safe_float(self.global_loss_streak_ewm, 0.0)),
            "global_equity": float(_safe_float(self.global_equity, 0.0)),
            "global_equity_peak": float(_safe_float(self.global_equity_peak, 0.0)),
            "global_drawdown": float(_safe_float(self.global_drawdown, 0.0)),
            "global_state": str(self.global_state or "risk_on"),
            "global_state_change_date": _normalize_optional_ts(self.global_state_change_date),
            "global_last_risk_off_date": _normalize_optional_ts(self.global_last_risk_off_date),
            "global_last_scale": float(_safe_float(self.global_last_scale, 1.0)),
            "global_last_reason": str(self.global_last_reason or "global_ok"),
            "global_last_score": float(_safe_float(self.global_last_score, 0.0)),
        }

    def to_dict(self) -> Dict[str, Any]:
        regimes_out: Dict[str, Dict[str, Any]] = {}
        for key, stats in self.regimes.items():
            st = asdict(stats)
            st["n_trades"] = _safe_int(st.get("n_trades", 0), 0)
            st["ewma_pnl"] = _safe_float(st.get("ewma_pnl", 0.0), 0.0)
            st["ewma_abs_pnl"] = _safe_float(st.get("ewma_abs_pnl", 0.0), 0.0)
            st["ewma_win"] = _safe_float(st.get("ewma_win", 0.0), 0.0)
            st["ewma_loss"] = _safe_float(st.get("ewma_loss", 0.0), 0.0)
            st["loss_streak"] = _safe_int(st.get("loss_streak", 0), 0)
            st["ewma_loss_streak"] = _safe_float(st.get("ewma_loss_streak", 0.0), 0.0)
            st["last_update_ts"] = _normalize_optional_ts(st.get("last_update_ts"))
            st["last_scale"] = _safe_float(st.get("last_scale", 1.0), 1.0)
            st["last_scale_update_n"] = _safe_int(st.get("last_scale_update_n", 0), 0)
            st["last_scale_update_date"] = _normalize_optional_ts(st.get("last_scale_update_date"))
            if st.get("last_meta_reason") is not None:
                st["last_meta_reason"] = str(st.get("last_meta_reason"))
            else:
                st["last_meta_reason"] = None
            st["ewma_return"] = _safe_float(st.get("ewma_return", st.get("ewma_pnl", 0.0)), 0.0)
            st["ewma_vol"] = _safe_float(st.get("ewma_vol", st.get("ewma_abs_pnl", 0.0)), 0.0)
            st["ewma_winrate"] = _safe_float(st.get("ewma_winrate", st.get("ewma_win", 0.0)), 0.0)
            regimes_out[str(key)] = st

        freeze_out: Dict[str, Optional[str]] = {}
        if isinstance(self.regime_freeze_until, dict):
            for key, value in self.regime_freeze_until.items():
                freeze_out[str(key)] = _normalize_optional_ts(value)

        last_meta: Dict[str, Optional[str]] = {}
        if isinstance(self.last_meta_reason_by_regime, dict):
            for key, value in self.last_meta_reason_by_regime.items():
                last_meta[str(key)] = str(value) if value is not None else None

        return {
            "config": asdict(self.cfg),
            "regimes": regimes_out,
            "daily_equity_peak": float(self.daily_equity_peak or 0.0),
            "daily_drawdown": float(self.daily_drawdown or 0.0),
            "loss_streak": int(self.loss_streak or 0),
            "regime_freeze_until": freeze_out,
            "daily_date": _normalize_optional_ts(self.daily_date),
            "last_guard_reason": str(self.last_guard_reason or "ok"),
            "last_guard_regime": str(self.last_guard_regime) if self.last_guard_regime is not None else None,
            "last_guard_date": _normalize_optional_ts(self.last_guard_date),
            "last_meta_reason_by_regime": last_meta,
            "global_trades": int(_safe_int(self.global_trades, 0)),
            "global_pnl_ewm": float(_safe_float(self.global_pnl_ewm, 0.0)),
            "global_abs_pnl_ewm": float(_safe_float(self.global_abs_pnl_ewm, 0.0)),
            "global_winrate_ewm": float(_safe_float(self.global_winrate_ewm, 0.0)),
            "global_dd_ewm": float(_safe_float(self.global_dd_ewm, 0.0)),
            "global_loss_streak": int(_safe_int(self.global_loss_streak, 0)),
            "global_loss_streak_ewm": float(_safe_float(self.global_loss_streak_ewm, 0.0)),
            "global_equity": float(_safe_float(self.global_equity, 0.0)),
            "global_equity_peak": float(_safe_float(self.global_equity_peak, 0.0)),
            "global_drawdown": float(_safe_float(self.global_drawdown, 0.0)),
            "global_state": str(self.global_state or "risk_on"),
            "global_state_change_date": _normalize_optional_ts(self.global_state_change_date),
            "global_last_risk_off_date": _normalize_optional_ts(self.global_last_risk_off_date),
            "global_last_state_change_trade_n": int(_safe_int(self.global_last_state_change_trade_n, 0)),
            "global_last_update_ts": _normalize_optional_ts(self.global_last_update_ts),
            "global_last_scale": float(_safe_float(self.global_last_scale, 1.0)),
            "global_last_reason": str(self.global_last_reason or "global_ok"),
            "global_last_score": float(_safe_float(self.global_last_score, 0.0)),
        }

    def to_json_dict(self) -> Dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, cfg: Optional[MetaRiskConfig], data: Dict[str, Any]) -> "MetaRiskState":
        if not isinstance(data, dict):
            data = {}

        if cfg is None:
            cfg_data = data.get("config", {})
            cfg = MetaRiskConfig(**_normalize_cfg_kwargs(cfg_data))

        state = cls(cfg)

        regimes_data = data.get("regimes", {})
        if isinstance(regimes_data, dict):
            allowed_stats = {f.name for f in fields(RegimeStats)}
            for key, stats_data in regimes_data.items():
                if isinstance(stats_data, dict):
                    filtered = {k: stats_data[k] for k in allowed_stats if k in stats_data}
                    if "ewma_pnl" not in filtered and "ewma_return" in stats_data:
                        filtered["ewma_pnl"] = stats_data.get("ewma_return", 0.0)
                    if "ewma_abs_pnl" not in filtered and "ewma_vol" in stats_data:
                        filtered["ewma_abs_pnl"] = stats_data.get("ewma_vol", 0.0)
                    if "ewma_win" not in filtered and "ewma_winrate" in stats_data:
                        filtered["ewma_win"] = stats_data.get("ewma_winrate", 0.0)
                    if "last_scale_update_n" not in filtered and "last_scale_update_trade_idx" in stats_data:
                        filtered["last_scale_update_n"] = stats_data.get("last_scale_update_trade_idx", 0)
                else:
                    filtered = {}
                stats = RegimeStats(**filtered)
                stats.n_trades = _safe_int(stats.n_trades, 0)
                stats.ewma_pnl = _safe_float(stats.ewma_pnl, 0.0)
                stats.ewma_abs_pnl = _safe_float(stats.ewma_abs_pnl, 0.0)
                stats.ewma_win = _safe_float(stats.ewma_win, 0.0)
                stats.ewma_loss = _safe_float(stats.ewma_loss, 0.0)
                stats.loss_streak = _safe_int(stats.loss_streak, 0)
                stats.ewma_loss_streak = _safe_float(stats.ewma_loss_streak, 0.0)
                stats.last_update_ts = _normalize_optional_ts(stats.last_update_ts)
                stats.last_scale = _safe_float(stats.last_scale, state._fallback_scale())
                min_scale, max_scale = state._scale_bounds()
                stats.last_scale = _clamp(stats.last_scale, min_scale, max_scale)
                stats.last_scale_update_n = _safe_int(stats.last_scale_update_n, 0)
                stats.last_scale_update_date = _normalize_optional_ts(stats.last_scale_update_date)
                if stats.last_meta_reason is not None:
                    stats.last_meta_reason = str(stats.last_meta_reason)
                stats.ewma_return = _safe_float(stats.ewma_return, stats.ewma_pnl)
                stats.ewma_vol = _safe_float(stats.ewma_vol, stats.ewma_abs_pnl)
                stats.ewma_winrate = _safe_float(stats.ewma_winrate, stats.ewma_win)
                if stats.last_scale_update_date is None and stats.last_update_ts:
                    stats.last_scale_update_date = _extract_date_key(stats.last_update_ts)
                state.regimes[str(key)] = stats

        state.daily_equity_peak = float(data.get("daily_equity_peak", 0.0) or 0.0)
        state.daily_drawdown = float(data.get("daily_drawdown", 0.0) or 0.0)
        state.loss_streak = int(data.get("loss_streak", 0) or 0)
        freeze = data.get("regime_freeze_until", {})
        if isinstance(freeze, dict):
            state.regime_freeze_until = {str(k): _normalize_optional_ts(v) for k, v in freeze.items()}
        else:
            state.regime_freeze_until = {}
        state.daily_date = _normalize_optional_ts(data.get("daily_date"))
        state.last_guard_reason = str(data.get("last_guard_reason", "ok") or "ok")
        last_regime = data.get("last_guard_regime")
        state.last_guard_regime = str(last_regime) if last_regime is not None else None
        state.last_guard_date = _normalize_optional_ts(data.get("last_guard_date"))
        last_meta = data.get("last_meta_reason_by_regime", {})
        if isinstance(last_meta, dict):
            state.last_meta_reason_by_regime = {
                str(k): str(v) if v is not None else None for k, v in last_meta.items()
            }
        else:
            state.last_meta_reason_by_regime = {}
        global_data = data.get("global", {})
        if not isinstance(global_data, dict):
            global_data = {}

        state.global_trades = _safe_int(data.get("global_trades", global_data.get("global_trades", 0)), 0)
        state.global_pnl_ewm = _safe_float(data.get("global_pnl_ewm", global_data.get("global_pnl_ewm", 0.0)), 0.0)
        state.global_abs_pnl_ewm = _safe_float(
            data.get("global_abs_pnl_ewm", global_data.get("global_abs_pnl_ewm", 0.0)), 0.0
        )
        state.global_winrate_ewm = _safe_float(
            data.get("global_winrate_ewm", global_data.get("global_winrate_ewm", 0.0)), 0.0
        )
        state.global_dd_ewm = _safe_float(data.get("global_dd_ewm", global_data.get("global_dd_ewm", 0.0)), 0.0)
        state.global_loss_streak = _safe_int(
            data.get("global_loss_streak", global_data.get("global_loss_streak", 0)), 0
        )
        state.global_loss_streak_ewm = _safe_float(
            data.get("global_loss_streak_ewm", global_data.get("global_loss_streak_ewm", 0.0)), 0.0
        )
        state.global_equity = _safe_float(data.get("global_equity", global_data.get("global_equity", 0.0)), 0.0)
        state.global_equity_peak = _safe_float(
            data.get("global_equity_peak", global_data.get("global_equity_peak", 0.0)), 0.0
        )
        state.global_drawdown = _safe_float(
            data.get("global_drawdown", global_data.get("global_drawdown", 0.0)), 0.0
        )
        state.global_state = str(data.get("global_state", global_data.get("global_state", "risk_on")) or "risk_on")
        state.global_state_change_date = _normalize_optional_ts(
            data.get("global_state_change_date", global_data.get("global_state_change_date"))
        )
        state.global_last_risk_off_date = _normalize_optional_ts(
            data.get("global_last_risk_off_date", global_data.get("global_last_risk_off_date"))
        )
        state.global_last_state_change_trade_n = _safe_int(
            data.get("global_last_state_change_trade_n", global_data.get("global_last_state_change_trade_n", 0)), 0
        )
        state.global_last_update_ts = _normalize_optional_ts(
            data.get("global_last_update_ts", global_data.get("global_last_update_ts"))
        )
        state.global_last_scale = _safe_float(
            data.get("global_last_scale", global_data.get("global_last_scale", 1.0)), 1.0
        )
        state.global_last_reason = str(
            data.get("global_last_reason", global_data.get("global_last_reason", "global_ok")) or "global_ok"
        )
        state.global_last_score = _safe_float(
            data.get("global_last_score", global_data.get("global_last_score", 0.0)), 0.0
        )
        return state

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "MetaRiskState":
        return cls.from_dict(None, data)

    def save(self, path: str) -> None:
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "MetaRiskState":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(None, data)

    def save_json(self, path: str) -> bool:
        return save_json(path, self)

    @classmethod
    def load_json(cls, cfg: Optional[MetaRiskConfig], path: Optional[str] = None) -> Optional["MetaRiskState"]:
        if path is None and isinstance(cfg, str):
            return load_json(None, cfg)
        return load_json(cfg, path)


def save_json(path: str, state: MetaRiskState) -> bool:
    if not path or state is None:
        return False

    try:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, sort_keys=True)
        return True
    except Exception:
        return False


def load_json(cfg_or_path: Optional[MetaRiskConfig], path: Optional[str] = None) -> Optional[MetaRiskState]:
    if path is None:
        path = str(cfg_or_path) if cfg_or_path is not None else ""
        cfg = None
    else:
        cfg = cfg_or_path

    if not path:
        return None
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MetaRiskState.from_dict(cfg, data)
    except Exception:
        return None


if __name__ == "__main__":
    cfg = MetaRiskConfig(min_trades_per_regime=3)
    state = MetaRiskState(cfg)
    state.update_trade("trend", 1.0, "2026-01-01")
    state.update_trade("trend", -0.5, "2026-01-02")
    state.update_trade("trend", 1.2, "2026-01-03")
    scale, reason = state.meta_scale_with_reason("trend")
    print("meta_scale:", scale, "reason:", reason)
