from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import math
import numpy as np


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class ConfidenceGateConfig:
    w_mom: float = 1.0
    w_trend: float = 0.7
    w_osc: float = 0.7

    scale: float = 0.9
    power: float = 1.3

    spread_cap: float = 0.50
    spread_power: float = 1.5

    highvol_th: float = 1.4

    # will be calibrated
    threshold: float = 0.60

    # optional (default off)
    slope_min: float = 0.0


class ConfidenceGate:
    def __init__(self, cfg: ConfidenceGateConfig | None = None):
        self.cfg = cfg or ConfidenceGateConfig()

    def _get(self, feat_row: Any, key: str, default: float = 0.0) -> float:
        try:
            v = feat_row[key]
        except Exception:
            v = default
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    def _confidence_only(self, feat_row: Any) -> float:
        cfg = self.cfg

        atr_norm = self._get(feat_row, "atr_norm", 0.0)
        spread_over_atr = self._get(feat_row, "spread_over_atr", 0.0)
        ema_fast_slope = self._get(feat_row, "ema_fast_slope", 0.0)
        ema_spread = self._get(feat_row, "ema_spread", 0.0)
        macd_hist_norm = self._get(feat_row, "macd_hist_norm", 0.0)

        # highvol => 0
        if atr_norm > cfg.highvol_th:
            return 0.0

        mom = abs(ema_fast_slope)
        trend = abs(ema_spread)
        osc = abs(macd_hist_norm)

        if cfg.slope_min > 0.0 and mom < cfg.slope_min:
            return 0.0

        raw = cfg.w_mom * mom + cfg.w_trend * trend + cfg.w_osc * osc
        raw_nl = raw ** cfg.power if raw > 0 else 0.0

        score_raw = math.tanh(raw_nl / (cfg.scale + 1e-12))
        score_raw = _clamp(score_raw, 0.0, 1.0)

        penalty_base = _clamp(spread_over_atr / (cfg.spread_cap + 1e-12), 0.0, 1.5)
        noise_penalty = _clamp(penalty_base ** cfg.spread_power, 0.0, 1.0)

        confidence = score_raw * (1.0 - noise_penalty)
        return _clamp(confidence, 0.0, 1.0)

    def calibrate_slope_min(self, feat_df, q: float = 0.60, mom_col: str = "ema_fast_slope") -> "ConfidenceGate":
        if mom_col not in feat_df.columns:
            return self
        s = feat_df[mom_col].astype(float).to_numpy()
        s = np.abs(s)
        s = s[np.isfinite(s)]
        if len(s) == 0:
            return self
        self.cfg.slope_min = float(np.quantile(s, q))
        return self

    def calibrate_threshold_from_features(self, feat_df, target_keep: float = 0.70) -> "ConfidenceGate":
        """
        Calibrate threshold so that only top `target_keep` fraction of confidence passes.
        Example:
          target_keep=0.70 => threshold = quantile(confidence, 1-0.70)=q0.30
        """
        confs = []
        # iterate with minimal overhead
        for _, row in feat_df.iterrows():
            confs.append(self._confidence_only(row))

        confs = np.asarray(confs, dtype=np.float64)
        confs = confs[np.isfinite(confs)]
        if len(confs) == 0:
            return self

        target_keep = float(_clamp(target_keep, 0.05, 0.95))
        q = 1.0 - target_keep
        self.cfg.threshold = float(np.quantile(confs, q))
        return self

    def evaluate(self, feat_row: Any) -> Tuple[float, bool, Dict[str, float]]:
        cfg = self.cfg

        atr_norm = self._get(feat_row, "atr_norm", 0.0)
        spread_over_atr = self._get(feat_row, "spread_over_atr", 0.0)
        ema_fast_slope = self._get(feat_row, "ema_fast_slope", 0.0)
        ema_spread = self._get(feat_row, "ema_spread", 0.0)
        macd_hist_norm = self._get(feat_row, "macd_hist_norm", 0.0)

        if atr_norm > cfg.highvol_th:
            reason = {"note_highvol": 1.0, "atr_norm": atr_norm, "highvol_th": cfg.highvol_th}
            return 0.0, False, reason

        mom = abs(ema_fast_slope)
        trend = abs(ema_spread)
        osc = abs(macd_hist_norm)

        if cfg.slope_min > 0.0 and mom < cfg.slope_min:
            reason = {"note_slope_floor": 1.0, "mom": mom, "slope_min": cfg.slope_min}
            return 0.0, False, reason

        raw = cfg.w_mom * mom + cfg.w_trend * trend + cfg.w_osc * osc
        raw_nl = raw ** cfg.power if raw > 0 else 0.0

        score_raw = math.tanh(raw_nl / (cfg.scale + 1e-12))
        score_raw = _clamp(score_raw, 0.0, 1.0)

        penalty_base = _clamp(spread_over_atr / (cfg.spread_cap + 1e-12), 0.0, 1.5)
        noise_penalty = _clamp(penalty_base ** cfg.spread_power, 0.0, 1.0)

        confidence = _clamp(score_raw * (1.0 - noise_penalty), 0.0, 1.0)
        allow_trade = confidence >= cfg.threshold

        reason = {
            "atr_norm": atr_norm,
            "spread_over_atr": spread_over_atr,
            "mom": mom,
            "trend": trend,
            "osc": osc,
            "raw": raw,
            "raw_nl": raw_nl,
            "score_raw": score_raw,
            "noise_penalty": noise_penalty,
            "confidence": confidence,
            "threshold": cfg.threshold,
            "slope_min": cfg.slope_min,
            "allow_trade": 1.0 if allow_trade else 0.0,
        }
        return confidence, allow_trade, reason


def make_default_gate() -> ConfidenceGate:
    return ConfidenceGate(ConfidenceGateConfig())