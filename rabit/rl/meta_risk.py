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
    global_fallback_scale: float
    target_pf: float
    target_return: float
    dd_soft_limit: float

    def __init__(
        self,
        min_trades_per_regime: int = 50,
        ewma_alpha: float = 0.10,
        score_clip: float = 3.0,
        min_scale: float = 0.40,
        max_scale: float = 1.20,
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
        down_power: float = 1.20,
        up_power: float = 0.80,
        hysteresis: float = 0.05,
        cooldown_trades: int = 5,
        global_fallback_scale: float = 1.0,
        target_pf: float = 1.02,
        target_return: float = 0.0,
        dd_soft_limit: float = 0.0,
    ) -> None:
        if alpha is not None:
            ewma_alpha = alpha
        if scale_min is not None:
            min_scale = scale_min
        if scale_max is not None:
            max_scale = scale_max
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
        self.global_fallback_scale = float(global_fallback_scale)
        self.target_pf = float(target_pf)
        self.target_return = float(target_return)
        self.dd_soft_limit = float(dd_soft_limit)

    @property
    def alpha(self) -> float:
        return self.ewma_alpha

    @property
    def floor(self) -> float:
        return self.min_scale

    @property
    def cap(self) -> float:
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
        "global_fallback_scale",
        "target_pf",
        "target_return",
        "dd_soft_limit",
        "alpha",
        "floor",
        "cap",
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
        min_scale = _safe_float(getattr(self.cfg, "scale_min", None), _safe_float(getattr(self.cfg, "min_scale", 0.0), 0.0))
        max_scale = _safe_float(getattr(self.cfg, "scale_max", None), _safe_float(getattr(self.cfg, "max_scale", 1.0), 1.0))
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

    def _scale_from_stats(self, st: RegimeStats) -> float:
        min_scale, max_scale = self._scale_bounds()
        perf = self._performance_score(st)
        down_power = max(0.01, _safe_float(getattr(self.cfg, "down_power", 1.0), 1.0))
        up_power = max(0.01, _safe_float(getattr(self.cfg, "up_power", 1.0), 1.0))
        if perf < 0.0:
            scale = 1.0 - (abs(perf) ** down_power) * (1.0 - min_scale)
        else:
            scale = 1.0 + (perf ** up_power) * (max_scale - 1.0)
        scale = _clamp(scale, min_scale, max_scale)
        return self._apply_dd_soft_limit(scale)

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

        candidate = self._scale_from_stats(st)
        last_scale = _safe_float(getattr(st, "last_scale", base), base)
        last_scale = _clamp(last_scale, min_scale, max_scale)

        hysteresis = max(0.0, _safe_float(getattr(self.cfg, "hysteresis", 0.0), 0.0))
        if abs(candidate - last_scale) < hysteresis:
            return last_scale, "hold_hysteresis"

        cooldown = max(0, _safe_int(getattr(self.cfg, "cooldown_trades", 0), 0))
        last_update_n = _safe_int(getattr(st, "last_scale_update_n", 0), 0)
        trades_since = max(0, n_trades - last_update_n)
        if cooldown > 0 and trades_since < cooldown:
            return last_scale, "hold_cooldown"

        reason = "calib_up" if candidate > last_scale else "calib_down"
        if update_state:
            st.last_scale = candidate
            st.last_scale_update_n = n_trades
            st.last_meta_reason = reason
        return candidate, reason

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
        scale, reason = self._compute_scale(regime_key, st, update_state)
        if regime_key:
            self.last_meta_reason_by_regime[regime_key] = reason
            if st is not None and st.last_meta_reason is None:
                st.last_meta_reason = reason
        return scale, reason

    def meta_scale(self, regime: str) -> float:
        """
        Return a risk scaling factor in [scale_min, scale_max].
        Warmup: if not enough trades yet -> return fallback scale (no interference).
        """
        scale, _reason = self.meta_scale_with_reason(regime, update_state=False)
        return scale

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "regimes": {key: asdict(stats) for key, stats in self.regimes.items()},
            "daily_equity_peak": float(self.daily_equity_peak),
            "daily_drawdown": float(self.daily_drawdown),
            "loss_streak": int(self.loss_streak),
            "regime_freeze_until": dict(self.regime_freeze_until),
            "daily_date": self.daily_date,
            "last_guard_reason": self.last_guard_reason,
            "last_guard_regime": self.last_guard_regime,
            "last_guard_date": self.last_guard_date,
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "MetaRiskState":
        if not isinstance(data, dict):
            data = {}

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
                stats.last_scale = _safe_float(stats.last_scale, state._fallback_scale())
                min_scale, max_scale = state._scale_bounds()
                stats.last_scale = _clamp(stats.last_scale, min_scale, max_scale)
                stats.last_scale_update_n = _safe_int(stats.last_scale_update_n, 0)
                stats.ewma_return = _safe_float(stats.ewma_return, stats.ewma_pnl)
                stats.ewma_vol = _safe_float(stats.ewma_vol, stats.ewma_abs_pnl)
                stats.ewma_winrate = _safe_float(stats.ewma_winrate, stats.ewma_win)
                state.regimes[str(key)] = stats

        state.daily_equity_peak = float(data.get("daily_equity_peak", 0.0) or 0.0)
        state.daily_drawdown = float(data.get("daily_drawdown", 0.0) or 0.0)
        state.loss_streak = int(data.get("loss_streak", 0) or 0)
        freeze = data.get("regime_freeze_until", {})
        if isinstance(freeze, dict):
            state.regime_freeze_until = {str(k): str(v) for k, v in freeze.items()}
        else:
            state.regime_freeze_until = {}
        state.daily_date = data.get("daily_date")
        state.last_guard_reason = str(data.get("last_guard_reason", "ok") or "ok")
        last_regime = data.get("last_guard_regime")
        state.last_guard_regime = str(last_regime) if last_regime is not None else None
        state.last_guard_date = data.get("last_guard_date")
        return state

    def save(self, path: str) -> None:
        data = self.to_json_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "MetaRiskState":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json_dict(data)

    def save_json(self, path: str) -> bool:
        return save_json(path, self)

    @classmethod
    def load_json(cls, path: str) -> Optional["MetaRiskState"]:
        return load_json(path)


def save_json(path: str, state: MetaRiskState) -> bool:
    if not path or state is None:
        return False

    try:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_json_dict(), f, indent=2, sort_keys=True)
        return True
    except Exception:
        return False


def load_json(path: str) -> Optional[MetaRiskState]:
    if not path:
        return None
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MetaRiskState.from_json_dict(data)
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
