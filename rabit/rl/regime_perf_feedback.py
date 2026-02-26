from __future__ import annotations

import datetime
import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple


_EPS = 1e-9


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


def _normalize_regime(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return default
    return s


def _normalize_ts(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time.min).isoformat()
    s = str(value).strip()
    return s if s else None


def _extract_date_key(value: Any) -> Optional[str]:
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


def _ts_sort_key(value: Any) -> Tuple[int, Any]:
    if value is None:
        return (1, "")
    if isinstance(value, datetime.datetime):
        return (0, value)
    if isinstance(value, datetime.date):
        return (0, datetime.datetime.combine(value, datetime.time.min))
    s = str(value).strip()
    if not s:
        return (1, "")
    try:
        return (0, datetime.datetime.fromisoformat(s.replace("Z", "")))
    except Exception:
        return (1, s)


def _ewm_update(prev: float, value: float, alpha: float, n_obs: int) -> float:
    alpha = float(alpha)
    if n_obs <= 1 or not math.isfinite(prev):
        return float(value)
    if alpha <= 0.0:
        return float(prev)
    if alpha >= 1.0:
        return float(value)
    return (1.0 - alpha) * float(prev) + alpha * float(value)


@dataclass
class RegimePerfConfig:
    version: int = 1
    ewm_alpha: float = 0.05
    min_trades_per_regime: int = 5
    min_scale: float = 0.60
    max_scale: float = 1.15
    max_scale_step: float = 0.08
    down_bad_winrate: float = 0.45
    up_good_winrate: float = 0.55
    down_bad_pnl: float = 0.0
    up_good_pnl: float = 0.02
    loss_streak_bad: int = 3
    dd_bad: float = 0.0
    cooldown_days: int = 3
    pf_cap: float = 10.0
    small_account_enabled: bool = False
    small_account_threshold: float = 0.0
    small_account_floor: float = 0.75

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimePerfConfig":
        if not isinstance(data, dict):
            return cls()
        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = data.get(f.name)
        return cls(**kwargs)


@dataclass
class RegimePerfStats:
    n_trades: int = 0
    winrate_ewm: float = 0.0
    pnl_mean_ewm: float = 0.0
    pnl_abs_ewm: float = 0.0
    gain_ewm: float = 0.0
    loss_ewm: float = 0.0
    loss_streak: int = 0
    loss_streak_ewm: float = 0.0
    equity: float = 0.0
    equity_peak: float = 0.0
    dd_ewm: float = 0.0
    state: str = "neutral"
    cooldown_left: int = 0
    last_update_ts: Optional[str] = None
    last_day: Optional[str] = None
    last_scale: float = 1.0
    last_reason: Optional[str] = None


class RegimePerfFeedbackEngine:
    """
    Rolling, regime-specific performance tracker with stable meta scaling.
    Designed to be deterministic for identical inputs.
    """

    def __init__(self, cfg: Optional[RegimePerfConfig] = None, debug: bool = False) -> None:
        self.cfg = cfg or RegimePerfConfig()
        self.debug = bool(debug)
        self.regimes: Dict[str, RegimePerfStats] = {}
        self.last_update_day: Optional[str] = None
        self.last_update_ts: Optional[str] = None
        self.read_only: bool = False

    def _advance_day(self, day_key: Optional[str]) -> None:
        day_key = _extract_date_key(day_key)
        if not day_key:
            return
        if self.last_update_day is None:
            self.last_update_day = day_key
            return
        if day_key == self.last_update_day:
            return
        ord_prev = _date_ordinal(self.last_update_day)
        ord_now = _date_ordinal(day_key)
        diff = 1
        if ord_prev is not None and ord_now is not None:
            diff = max(1, int(ord_now - ord_prev))
        for stats in self.regimes.values():
            if stats.cooldown_left > 0:
                stats.cooldown_left = max(0, int(stats.cooldown_left) - diff)
        self.last_update_day = day_key

    def _bad_conditions(self, stats: RegimePerfStats) -> Tuple[bool, Dict[str, float]]:
        cfg = self.cfg
        winrate = _safe_float(stats.winrate_ewm, 0.0)
        pnl = _safe_float(stats.pnl_mean_ewm, 0.0)
        loss_streak = _safe_int(stats.loss_streak, 0)
        dd = _safe_float(stats.dd_ewm, 0.0)

        win_sev = 0.0
        if winrate < cfg.down_bad_winrate:
            denom = max(_EPS, cfg.down_bad_winrate)
            win_sev = (cfg.down_bad_winrate - winrate) / denom

        pnl_sev = 0.0
        if pnl < cfg.down_bad_pnl:
            denom = max(_EPS, abs(cfg.down_bad_pnl) if cfg.down_bad_pnl != 0.0 else 1.0)
            pnl_sev = (cfg.down_bad_pnl - pnl) / denom

        streak_sev = 0.0
        if loss_streak >= cfg.loss_streak_bad and cfg.loss_streak_bad > 0:
            streak_sev = min(1.0, (loss_streak - cfg.loss_streak_bad + 1) / float(cfg.loss_streak_bad))

        dd_sev = 0.0
        if cfg.dd_bad > 0.0 and dd > cfg.dd_bad:
            dd_sev = min(1.0, (dd - cfg.dd_bad) / max(_EPS, cfg.dd_bad))

        sev = {
            "winrate": win_sev,
            "pnl": pnl_sev,
            "loss_streak": streak_sev,
            "drawdown": dd_sev,
        }
        is_bad = max(sev.values()) > 0.0
        return is_bad, sev

    def _good_conditions(self, stats: RegimePerfStats) -> Tuple[bool, Dict[str, float]]:
        cfg = self.cfg
        winrate = _safe_float(stats.winrate_ewm, 0.0)
        pnl = _safe_float(stats.pnl_mean_ewm, 0.0)
        loss_streak = _safe_int(stats.loss_streak, 0)

        win_sev = 0.0
        if winrate > cfg.up_good_winrate:
            denom = max(_EPS, 1.0 - cfg.up_good_winrate)
            win_sev = min(1.0, (winrate - cfg.up_good_winrate) / denom)

        pnl_sev = 0.0
        if pnl > cfg.up_good_pnl:
            denom = max(_EPS, abs(cfg.up_good_pnl) if cfg.up_good_pnl != 0.0 else 1.0)
            pnl_sev = min(1.0, (pnl - cfg.up_good_pnl) / denom)

        sev = {"winrate": win_sev, "pnl": pnl_sev}
        is_good = max(sev.values()) > 0.0 and loss_streak == 0
        return is_good, sev

    def _dominant_reason(self, sev_map: Dict[str, float], prefix: str) -> str:
        if not sev_map:
            return f"{prefix}_neutral"
        items = sorted(sev_map.items(), key=lambda x: (-x[1], x[0]))
        if items and items[0][1] > 0.0:
            return f"{prefix}_{items[0][0]}"
        return f"{prefix}_neutral"

    def _small_account_factor(self, account_equity: Optional[float]) -> float:
        cfg = self.cfg
        if not cfg.small_account_enabled:
            return 1.0
        threshold = _safe_float(cfg.small_account_threshold, 0.0)
        if threshold <= 0.0 or account_equity is None:
            return 1.0
        equity = _safe_float(account_equity, 0.0)
        if equity >= threshold:
            return 1.0
        ratio = equity / max(_EPS, threshold)
        floor = _clamp(_safe_float(cfg.small_account_floor, 0.0), 0.0, 1.0)
        return _clamp(max(floor, ratio), 0.0, 1.0)

    def _slew_scale(self, prev_scale: float, target_scale: float) -> Tuple[float, bool]:
        step = _safe_float(getattr(self.cfg, "max_scale_step", 0.0), 0.0)
        if step <= 0.0 or not math.isfinite(prev_scale):
            return float(target_scale), False
        delta = float(target_scale) - float(prev_scale)
        if abs(delta) <= step:
            return float(target_scale), False
        return float(prev_scale + math.copysign(step, delta)), True

    def _compute_scale(self, stats: RegimePerfStats) -> Tuple[float, str, bool]:
        cfg = self.cfg
        if stats.n_trades < max(1, int(cfg.min_trades_per_regime)):
            prev = stats.last_scale if stats.last_scale is not None else 1.0
            prev_reason = stats.last_reason if stats.last_reason else "warmup"
            return float(prev), str(prev_reason), False

        is_bad, bad_sev = self._bad_conditions(stats)
        is_good, good_sev = self._good_conditions(stats)

        state = stats.state or "neutral"
        candidate = state
        if state == "bad":
            if is_good:
                candidate = "good"
            elif not is_bad:
                candidate = "neutral"
        elif state == "good":
            if is_bad:
                candidate = "bad"
            elif not is_good and (
                _safe_float(stats.winrate_ewm, 0.0) < cfg.down_bad_winrate
                or _safe_float(stats.pnl_mean_ewm, 0.0) < cfg.down_bad_pnl
            ):
                candidate = "neutral"
        else:
            if is_bad:
                candidate = "bad"
            elif is_good:
                candidate = "good"

        if stats.cooldown_left > 0 and candidate != state:
            return float(stats.last_scale), f"cooldown_{state}", False

        changed = candidate != state
        if changed:
            stats.state = candidate
            stats.cooldown_left = int(cfg.cooldown_days)

        if stats.state == "bad":
            severity = max(bad_sev.values()) if bad_sev else 0.0
            severity = _clamp(severity, 0.0, 1.0)
            scale = 1.0 - severity * (1.0 - cfg.min_scale)
            reason = self._dominant_reason(bad_sev, "bad")
        elif stats.state == "good":
            severity = max(good_sev.values()) if good_sev else 0.0
            severity = _clamp(severity, 0.0, 1.0)
            scale = 1.0 + severity * (cfg.max_scale - 1.0)
            reason = self._dominant_reason(good_sev, "good")
        else:
            scale = 1.0
            reason = "neutral"

        scale = _clamp(scale, cfg.min_scale, cfg.max_scale)
        return float(scale), str(reason), changed

    def update_from_trades(
        self,
        trades_df: Any,
        day_key: Any,
        regime_col: str = "regime",
        pnl_col: str = "pnl",
        account_equity: Optional[float] = None,
    ) -> Dict[str, Any]:
        day_key = _extract_date_key(day_key)
        self._advance_day(day_key)
        cfg = self.cfg

        updated: List[str] = []
        changes: List[Dict[str, Any]] = []
        if trades_df is None:
            return {"day": day_key, "updated_regimes": [], "scale_changes": []}

        rows: List[Tuple[Any, int, str, float]] = []

        if hasattr(trades_df, "iterrows") and hasattr(trades_df, "columns"):
            cols = set(trades_df.columns)
            ts_col = None
            for cand in ["exit_time", "entry_time", "close_time", "open_time", "time", "timestamp", "datetime", "ts"]:
                if cand in cols:
                    ts_col = cand
                    break
            pnl_key = pnl_col if pnl_col in cols else None
            if pnl_key is None:
                for cand in ["pnl", "net_pnl", "profit", "pnl_return", "return"]:
                    if cand in cols:
                        pnl_key = cand
                        break
            for pos, (_idx, row) in enumerate(trades_df.iterrows()):
                ts_val = row[ts_col] if ts_col and ts_col in cols else None
                regime_val = row[regime_col] if regime_col in cols else "unknown"
                pnl_val = row[pnl_key] if pnl_key and pnl_key in cols else 0.0
                rows.append((ts_val, int(pos), _normalize_regime(regime_val), _safe_float(pnl_val, 0.0)))
        elif isinstance(trades_df, list):
            for idx, row in enumerate(trades_df):
                if not isinstance(row, dict):
                    continue
                ts_val = row.get("exit_time") or row.get("entry_time") or row.get("time") or row.get("timestamp")
                regime_val = row.get(regime_col, row.get("regime", "unknown"))
                pnl_val = row.get(pnl_col)
                if pnl_val is None:
                    pnl_val = row.get("pnl", row.get("net_pnl", row.get("profit", 0.0)))
                rows.append((ts_val, int(idx), _normalize_regime(regime_val), _safe_float(pnl_val, 0.0)))
        else:
            return {"day": day_key, "updated_regimes": [], "scale_changes": []}

        if not rows:
            return {"day": day_key, "updated_regimes": [], "scale_changes": []}

        rows.sort(key=lambda rec: (_ts_sort_key(rec[0]), rec[1]))

        for ts_val, _idx, regime, pnl in rows:
            stats = self.regimes.get(regime)
            if stats is None:
                stats = RegimePerfStats()
                self.regimes[regime] = stats
            stats.n_trades = _safe_int(stats.n_trades, 0) + 1
            n_obs = int(stats.n_trades)
            win = 1.0 if pnl > 0.0 else 0.0
            stats.winrate_ewm = _ewm_update(stats.winrate_ewm, win, self.cfg.ewm_alpha, n_obs)
            stats.pnl_mean_ewm = _ewm_update(stats.pnl_mean_ewm, pnl, self.cfg.ewm_alpha, n_obs)
            stats.pnl_abs_ewm = _ewm_update(stats.pnl_abs_ewm, abs(pnl), self.cfg.ewm_alpha, n_obs)
            stats.gain_ewm = _ewm_update(stats.gain_ewm, max(pnl, 0.0), self.cfg.ewm_alpha, n_obs)
            stats.loss_ewm = _ewm_update(stats.loss_ewm, max(-pnl, 0.0), self.cfg.ewm_alpha, n_obs)
            if pnl < 0.0:
                stats.loss_streak = _safe_int(stats.loss_streak, 0) + 1
            else:
                stats.loss_streak = 0
            stats.loss_streak_ewm = _ewm_update(
                stats.loss_streak_ewm, float(stats.loss_streak), self.cfg.ewm_alpha, n_obs
            )
            stats.equity = _safe_float(stats.equity, 0.0) + pnl
            stats.equity_peak = max(_safe_float(stats.equity_peak, 0.0), stats.equity)
            drawdown = max(0.0, stats.equity_peak - stats.equity)
            stats.dd_ewm = _ewm_update(stats.dd_ewm, drawdown, self.cfg.ewm_alpha, n_obs)
            stats.last_update_ts = _normalize_ts(ts_val) or stats.last_update_ts
            stats.last_day = day_key or stats.last_day
            self.last_update_ts = stats.last_update_ts or self.last_update_ts
            if regime not in updated:
                updated.append(regime)

        for regime in sorted(updated):
            stats = self.regimes.get(regime)
            if stats is None:
                continue
            before = _safe_float(stats.last_scale, 1.0)
            scale, reason, changed = self._compute_scale(stats)
            scale = _clamp(float(scale), cfg.min_scale, cfg.max_scale)

            small_factor = self._small_account_factor(account_equity)
            if small_factor < 1.0:
                scale = _clamp(float(scale) * float(small_factor), cfg.min_scale, cfg.max_scale)
                reason = f"{reason}|small_account"

            scale, slewed = self._slew_scale(before, scale)
            if slewed:
                reason = f"{reason}|slew"
            scale = _clamp(float(scale), cfg.min_scale, cfg.max_scale)

            stats.last_scale = float(scale)
            stats.last_reason = str(reason)
            if changed or abs(scale - before) > 1e-9:
                changes.append(
                    {
                        "regime": regime,
                        "prev_scale": float(before),
                        "new_scale": float(scale),
                        "reason": str(reason),
                        "state": str(stats.state),
                    }
                )

        return {"day": day_key, "updated_regimes": sorted(updated), "scale_changes": changes}

    def get_scale(self, regime: str) -> float:
        scale, _reason = self.get_scale_with_reason(regime)
        return scale

    def get_scale_with_reason(self, regime: str) -> Tuple[float, Optional[str]]:
        key = _normalize_regime(regime, default="unknown")
        stats = self.regimes.get(key)
        if stats is None:
            return 1.0, "no_stats"
        if stats.last_scale is None:
            scale, reason, _changed = self._compute_scale(stats)
            return float(scale), str(reason)
        return float(stats.last_scale), stats.last_reason

    def get_debug(self, regime: str) -> Dict[str, Any]:
        key = _normalize_regime(regime, default="unknown")
        stats = self.regimes.get(key)
        if stats is None:
            return {"regime": key, "meta_scale": 1.0, "reason": "no_stats"}
        return {
            "regime": key,
            "n_trades": int(stats.n_trades),
            "winrate_ewm": float(stats.winrate_ewm),
            "pnl_mean_ewm": float(stats.pnl_mean_ewm),
            "dd_ewm": float(stats.dd_ewm),
            "loss_streak": int(stats.loss_streak),
            "meta_scale": float(stats.last_scale),
            "reason": stats.last_reason,
            "cooldown_left": int(stats.cooldown_left),
            "last_update_ts": stats.last_update_ts,
        }

    def regimes_seen(self) -> List[str]:
        return sorted(self.regimes.keys())

    def global_meta_scale(self) -> Optional[float]:
        if not self.regimes:
            return None
        total = 0.0
        weight = 0.0
        for stats in self.regimes.values():
            n = max(0, int(stats.n_trades))
            if n <= 0:
                continue
            total += float(stats.last_scale) * n
            weight += n
        if weight <= 0:
            return None
        return float(total / weight)

    def _regime_summary(self, regime: str, stats: RegimePerfStats) -> Dict[str, Any]:
        pf = 1.0
        gain = _safe_float(stats.gain_ewm, 0.0)
        loss = _safe_float(stats.loss_ewm, 0.0)
        if gain <= _EPS and loss <= _EPS:
            pf = 1.0
        else:
            pf = gain / max(_EPS, loss)
            pf = min(float(pf), float(self.cfg.pf_cap))
        return {
            "regime": str(regime),
            "n_trades": int(stats.n_trades),
            "winrate_ewm": float(stats.winrate_ewm),
            "pnl_mean_ewm": float(stats.pnl_mean_ewm),
            "pnl_abs_ewm": float(stats.pnl_abs_ewm),
            "loss_streak": int(stats.loss_streak),
            "loss_streak_ewm": float(stats.loss_streak_ewm),
            "dd_ewm": float(stats.dd_ewm),
            "profit_factor_ewm": float(pf),
            "meta_scale": float(stats.last_scale),
            "state": str(stats.state),
            "cooldown_left": int(stats.cooldown_left),
            "last_update_ts": stats.last_update_ts,
        }

    def summary(self) -> Dict[str, Any]:
        regimes_out: Dict[str, Dict[str, Any]] = {}
        for key in sorted(self.regimes.keys()):
            regimes_out[key] = self._regime_summary(key, self.regimes[key])
        return {
            "version": int(self.cfg.version),
            "config": asdict(self.cfg),
            "last_update_day": self.last_update_day,
            "last_update_ts": self.last_update_ts,
            "regimes": regimes_out,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.summary()

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        cfg: Optional[RegimePerfConfig] = None,
    ) -> "RegimePerfFeedbackEngine":
        if not isinstance(data, dict):
            data = {}
        if cfg is None:
            cfg = RegimePerfConfig.from_dict(data.get("config", {}))
        engine = cls(cfg=cfg)
        engine.last_update_day = _extract_date_key(data.get("last_update_day"))
        engine.last_update_ts = _normalize_ts(data.get("last_update_ts"))
        regimes_data = data.get("regimes", {})
        if isinstance(regimes_data, dict):
            allowed = {f.name for f in fields(RegimePerfStats)}
            for key, st_data in regimes_data.items():
                if isinstance(st_data, dict):
                    filtered = {k: st_data[k] for k in allowed if k in st_data}
                else:
                    filtered = {}
                stats = RegimePerfStats(**filtered)
                stats.n_trades = _safe_int(stats.n_trades, 0)
                stats.winrate_ewm = _safe_float(stats.winrate_ewm, 0.0)
                stats.pnl_mean_ewm = _safe_float(stats.pnl_mean_ewm, 0.0)
                stats.pnl_abs_ewm = _safe_float(stats.pnl_abs_ewm, 0.0)
                stats.gain_ewm = _safe_float(stats.gain_ewm, 0.0)
                stats.loss_ewm = _safe_float(stats.loss_ewm, 0.0)
                stats.loss_streak = _safe_int(stats.loss_streak, 0)
                stats.loss_streak_ewm = _safe_float(stats.loss_streak_ewm, 0.0)
                stats.equity = _safe_float(stats.equity, 0.0)
                stats.equity_peak = _safe_float(stats.equity_peak, 0.0)
                stats.dd_ewm = _safe_float(stats.dd_ewm, 0.0)
                stats.state = str(stats.state or "neutral")
                stats.cooldown_left = _safe_int(stats.cooldown_left, 0)
                stats.last_update_ts = _normalize_ts(stats.last_update_ts)
                stats.last_day = _extract_date_key(stats.last_day)
                stats.last_scale = _clamp(_safe_float(stats.last_scale, 1.0), cfg.min_scale, cfg.max_scale)
                if stats.last_reason is not None:
                    stats.last_reason = str(stats.last_reason)
                engine.regimes[str(key)] = stats
        return engine

    def save(self, path: str) -> bool:
        if self.read_only:
            return False
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

    def load(self, path: str, keep_cfg: bool = True) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = self.cfg if keep_cfg else None
        other = self.from_dict(data, cfg=cfg)
        self.cfg = other.cfg
        self.regimes = other.regimes
        self.last_update_day = other.last_update_day
        self.last_update_ts = other.last_update_ts


def _self_check() -> bool:
    cfg = RegimePerfConfig(min_trades_per_regime=1)
    eng = RegimePerfFeedbackEngine(cfg=cfg)
    trades = [
        {"regime": "trend", "pnl": 1.0, "exit_time": "2026-01-01T00:00:00"},
        {"regime": "trend", "pnl": -0.5, "exit_time": "2026-01-01T01:00:00"},
    ]
    eng.update_from_trades(trades, day_key="2026-01-01")
    state1 = eng.to_dict()
    eng2 = RegimePerfFeedbackEngine(cfg=cfg)
    eng2.update_from_trades(trades, day_key="2026-01-01")
    state2 = eng2.to_dict()
    return state1 == state2 and eng.get_scale("trend") == eng2.get_scale("trend")
