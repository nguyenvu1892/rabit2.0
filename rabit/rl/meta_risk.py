from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import datetime

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

    def __init__(
        self,
        min_trades_per_regime: int = 50,
        ewma_alpha: float = 0.05,
        score_clip: float = 3.0,
        min_scale: float = 0.25,
        max_scale: float = 1.25,
        k: float = 1.0,
        daily_dd_limit: float = 3.0,
        loss_streak_limit: int = 5,
        freeze_days: int = 3,
        global_floor: float = 0.2,
        alpha: Optional[float] = None,
        floor: Optional[float] = None,
        cap: Optional[float] = None,
    ) -> None:
        if alpha is not None:
            ewma_alpha = alpha
        if floor is not None:
            min_scale = floor
        if cap is not None:
            max_scale = cap

        self.min_trades_per_regime = int(min_trades_per_regime)
        self.ewma_alpha = float(ewma_alpha)
        self.score_clip = float(score_clip)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.k = float(k)
        self.daily_dd_limit = float(daily_dd_limit)
        self.loss_streak_limit = int(loss_streak_limit)
        self.freeze_days = int(freeze_days)
        self.global_floor = float(global_floor)

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
    ewma_return: float = 0.0
    ewma_vol: float = 0.0
    ewma_winrate: float = 0.0
    last_update_ts: Optional[str] = None


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
        "alpha",
        "floor",
        "cap",
    }
    return {k: cfg[k] for k in allowed if k in cfg}


def _extract_date_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            return value.date().isoformat()
        except Exception:
            return None
    s = value.strip()
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
        return datetime.fromisoformat(s.replace("Z", "")).date().isoformat()
    except Exception:
        return None


def _date_ordinal(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10]).toordinal()
    except Exception:
        return None


class MetaRiskState:
    def __init__(self, cfg: MetaRiskConfig) -> None:
        self.cfg = cfg
        self.regimes: Dict[str, RegimeStats] = {}
        self.daily_equity_peak = 0.0
        self.daily_drawdown = 0.0
        self.loss_streak = 0
        self.regime_freeze_until: Dict[str, str] = {}
        self.daily_date: Optional[str] = None
        self.last_guard_reason: str = "ok"
        self.last_guard_regime: Optional[str] = None
        self.last_guard_date: Optional[str] = None

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

    def update_trade(
        self,
        regime: str,
        pnl_return: float,
        ts: Optional[str] = None,
        date: Optional[str] = None,
    ) -> None:
        if ts is None:
            ts = date
        if ts is None:
            ts = ""

        stats = self.regimes.get(regime)
        if stats is None:
            stats = RegimeStats()
            self.regimes[regime] = stats

        alpha = self.cfg.ewma_alpha
        stats.ewma_return = (1.0 - alpha) * stats.ewma_return + alpha * pnl_return
        stats.ewma_vol = (1.0 - alpha) * stats.ewma_vol + alpha * abs(pnl_return)
        win = 1.0 if pnl_return > 0.0 else 0.0
        stats.ewma_winrate = (1.0 - alpha) * stats.ewma_winrate + alpha * win
        stats.n_trades += 1
        stats.last_update_ts = ts

        date_key = _extract_date_key(ts or date)
        if date_key is not None:
            self.daily_date = date_key

        if pnl_return > 0.0:
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            if (
                self.cfg.loss_streak_limit > 0
                and self.loss_streak >= self.cfg.loss_streak_limit
                and regime
                and date_key is not None
                and self.cfg.freeze_days > 0
            ):
                if isinstance(date_key, str):
                    try:
                        dt = datetime.date.fromisoformat(date_key[:10])
                    except Exception:
                        dt = datetime.datetime.utcnow().date()
                else:
                    dt = datetime.datetime.utcnow().date()

            dt = None

            if isinstance(date_key, str) and len(date_key) >= 10:
                try:
                    dt = datetime.date.fromisoformat(date_key[:10])
                except Exception:
                    dt = None

            if dt is None:
                dt = datetime.datetime.utcnow().date()

            freeze_until = dt + datetime.timedelta(days=self.cfg.freeze_days)
            self.regime_freeze_until[regime] = freeze_until.isoformat()

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

    def meta_scale(self, regime: str) -> float:
        """
        Return a risk scaling factor in [min_scale, max_scale].
        Warmup: if not enough trades yet -> return 1.0 (no interference).
        Robust to internal attribute names (state dict) to avoid crashing.
        """

        # --- Find regime-state map robustly (avoid AttributeError) ---
        states = (
            getattr(self, "state_by_regime", None)
            or getattr(self, "by_regime", None)
            or getattr(self, "regime_states", None)
            or getattr(self, "states_by_regime", None)
            or getattr(self, "states", None)
        )

        if states is None or not hasattr(states, "get"):
            # If structure unknown, never crash; just don't interfere.
            return 1.0

        st = states.get(regime)
        if st is None:
            return 1.0

        # --- Warmup guard ---
        n_trades = getattr(st, "n_trades", None)
        if n_trades is None:
            # Some implementations store trades list
            trades_list = getattr(st, "trades", None)
            if trades_list is not None:
                try:
                    n_trades = len(trades_list)
                except Exception:
                    n_trades = 0
            else:
                n_trades = 0

        if n_trades < getattr(self.cfg, "min_trades_per_regime", 1):
            return 1.0

        # --- Compute scale: keep your existing logic if it exists ---
        # Try to call an existing compute function if you have one.
        scale = None
        if hasattr(self, "_compute_scale_from_state"):
            scale = self._compute_scale_from_state(st)
        elif hasattr(self, "_scale_from_state"):
            scale = self._scale_from_state(st)
        elif hasattr(self, "compute_scale"):
            scale = self.compute_scale(st)
        else:
            # Fallback: if you don't have compute logic here, default to 1.0
            scale = 1.0

        # --- Clamp to avoid "deny all" ---
        try:
            scale = float(scale)
        except Exception:
            scale = 1.0

        if not (scale == scale):  # NaN
            scale = 1.0

        min_scale = float(getattr(self.cfg, "min_scale", 0.2))
        max_scale = float(getattr(self.cfg, "max_scale", 1.0))
        if max_scale < min_scale:
            max_scale = min_scale

        if scale < min_scale:
            scale = min_scale
        if scale > max_scale:
            scale = max_scale

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
            allowed_stats = {
                "n_trades",
                "ewma_return",
                "ewma_vol",
                "ewma_winrate",
                "last_update_ts",
            }
            for key, stats_data in regimes_data.items():
                if isinstance(stats_data, dict):
                    filtered = {k: stats_data[k] for k in allowed_stats if k in stats_data}
                else:
                    filtered = {}
                state.regimes[str(key)] = RegimeStats(**filtered)

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
    print("meta_scale:", state.meta_scale("trend"))
