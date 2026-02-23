from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeDetector:
    """
    Lightweight regime classifier for XAU M1 scalping.

    Inputs expected in feat df:
      - atr_norm
      - ema_spread
      - ema_fast_slope
      - ema_slow_slope (optional)

    Regimes:
      - highvol: atr_norm >= q_highvol
      - trend: abs(ema_fast_slope) >= q_slope_hi AND ema_spread >= q_spread_hi
      - range: ema_spread <= q_spread_lo AND atr_norm <= q_atr_lo AND abs(ema_fast_slope) <= q_slope_lo
      - mixed: otherwise
    """

    def __init__(
        self,
        q_highvol: float = 0.85,
        q_atr_lo: float = 0.35,
        q_spread_hi: float = 0.65,
        q_spread_lo: float = 0.35,
        q_slope_hi: float = 0.75,
        q_slope_lo: float = 0.35,
    ):
        self.q_highvol = q_highvol
        self.q_atr_lo = q_atr_lo
        self.q_spread_hi = q_spread_hi
        self.q_spread_lo = q_spread_lo
        self.q_slope_hi = q_slope_hi
        self.q_slope_lo = q_slope_lo

        # fitted thresholds
        self.th_highvol = None
        self.th_atr_lo = None
        self.th_spread_hi = None
        self.th_spread_lo = None
        self.th_slope_hi = None
        self.th_slope_lo = None

    def fit(self, feat: pd.DataFrame) -> "RegimeDetector":
        atrn = feat["atr_norm"].to_numpy(dtype=np.float64)
        spr = feat["ema_spread"].to_numpy(dtype=np.float64)
        slope = np.abs(feat["ema_fast_slope"].to_numpy(dtype=np.float64))

        def q(x, qq):
            x = x[np.isfinite(x)]
            if len(x) == 0:
                return 0.0
            return float(np.quantile(x, qq))

        self.th_highvol = q(atrn, self.q_highvol)
        self.th_atr_lo = q(atrn, self.q_atr_lo)
        self.th_spread_hi = q(spr, self.q_spread_hi)
        self.th_spread_lo = q(spr, self.q_spread_lo)
        self.th_slope_hi = q(slope, self.q_slope_hi)
        self.th_slope_lo = q(slope, self.q_slope_lo)

        return self

    def predict(self, feat: pd.DataFrame) -> np.ndarray:
        atrn = feat["atr_norm"].to_numpy(dtype=np.float64)
        spr = feat["ema_spread"].to_numpy(dtype=np.float64)
        slope = np.abs(feat["ema_fast_slope"].to_numpy(dtype=np.float64))

        # safety if not fitted
        th_highvol = 1.0 if self.th_highvol is None else float(self.th_highvol)
        th_atr_lo = 0.8 if self.th_atr_lo is None else float(self.th_atr_lo)
        th_spread_hi = 0.0 if self.th_spread_hi is None else float(self.th_spread_hi)
        th_spread_lo = 0.0 if self.th_spread_lo is None else float(self.th_spread_lo)
        th_slope_hi = 0.0 if self.th_slope_hi is None else float(self.th_slope_hi)
        th_slope_lo = 0.0 if self.th_slope_lo is None else float(self.th_slope_lo)

        out = np.full((len(feat),), "mixed", dtype=object)

        # highvol first (priority)
        highvol = atrn >= th_highvol
        out[highvol] = "highvol"

        # trend (excluding highvol)
        trend = (~highvol) & (slope >= th_slope_hi) & (spr >= th_spread_hi)
        out[trend] = "trend"

        # range (excluding highvol)
        range_ = (~highvol) & (spr <= th_spread_lo) & (atrn <= th_atr_lo) & (slope <= th_slope_lo)
        out[range_] = "range"

        return out

    def thresholds(self) -> dict:
        return {
            "th_highvol": self.th_highvol,
            "th_atr_lo": self.th_atr_lo,
            "th_spread_hi": self.th_spread_hi,
            "th_spread_lo": self.th_spread_lo,
            "th_slope_hi": self.th_slope_hi,
            "th_slope_lo": self.th_slope_lo,
        }