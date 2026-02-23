from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    dropna: bool = True

    # --- Knowledge features toggles ---
    add_atr: bool = True
    atr_period: int = 14

    add_ema: bool = True
    ema_fast: int = 9
    ema_slow: int = 21

    add_rsi: bool = True
    rsi_period: int = 14

    add_macd: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    add_stoch: bool = True
    stoch_k: int = 14
    stoch_d: int = 3

    add_volume: bool = True
    vol_ema: int = 20

    # Normalization / stability
    eps: float = 1e-12


class FeatureBuilder:
    """
    Build "knowledge" features from OHLCV:
      - ATR (volatility)
      - EMA fast/slow + spread + slopes
      - RSI
      - MACD (line/signal/hist)
      - Stoch %K/%D
      - Volume (tickvol) ema + z-ish
      - Spread raw + normalized by ATR (optional)
    """

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    # ---------- Indicators ----------
    def _ema(self, s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    def _rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = roll_up / (roll_down + self.cfg.eps)
        return 100.0 - (100.0 / (1.0 + rs))

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder-like smoothing via EMA alpha=1/period
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        return atr

    def _macd(self, close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    def _stoch(self, df: pd.DataFrame, k: int, d: int) -> tuple[pd.Series, pd.Series]:
        low_min = df["low"].rolling(k).min()
        high_max = df["high"].rolling(k).max()
        k_pct = 100.0 * (df["close"] - low_min) / ((high_max - low_min) + self.cfg.eps)
        d_pct = k_pct.rolling(d).mean()
        return k_pct, d_pct

    # ---------- Public API ----------
    def build(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Input df must contain:
          open, high, low, close, tickvol, spread
        Index: timestamp
        """
        required = {"open", "high", "low", "close", "tickvol", "spread"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"FeatureBuilder.build missing columns: {sorted(missing)}")

        out = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        tickvol = df["tickvol"]
        spread = df["spread"]

        # Always include basic “raw” knowledge channels
        out[prefix + "close"] = close
        out[prefix + "ret_1"] = close.pct_change().fillna(0.0)

        # ATR (core volatility knowledge)
        if self.cfg.add_atr:
            atr = self._atr(df, self.cfg.atr_period)
            out[prefix + "atr"] = atr
            out[prefix + "atr_norm"] = atr / (close.abs() + self.cfg.eps)  # scale-free
        else:
            atr = None

        # Spread knowledge
        out[prefix + "spread"] = spread.astype("float64")
        if atr is not None:
            # spread normalized by ATR (how expensive market is vs volatility)
            out[prefix + "spread_over_atr"] = (spread.astype("float64") * 0.01) / (atr + self.cfg.eps)

        # EMA knowledge
        if self.cfg.add_ema:
            ema_f = self._ema(close, self.cfg.ema_fast)
            ema_s = self._ema(close, self.cfg.ema_slow)
            out[prefix + f"ema_{self.cfg.ema_fast}"] = ema_f
            out[prefix + f"ema_{self.cfg.ema_slow}"] = ema_s
            out[prefix + "ema_spread"] = ema_f - ema_s
            out[prefix + "ema_spread_norm"] = (ema_f - ema_s) / (close.abs() + self.cfg.eps)
            out[prefix + "ema_fast_slope"] = ema_f.diff().fillna(0.0)
            out[prefix + "ema_slow_slope"] = ema_s.diff().fillna(0.0)

        # RSI knowledge
        if self.cfg.add_rsi:
            rsi = self._rsi(close, self.cfg.rsi_period)
            out[prefix + f"rsi_{self.cfg.rsi_period}"] = rsi
            # normalized RSI in [-1,1] (centered at 50)
            out[prefix + "rsi_c"] = (rsi - 50.0) / 50.0

        # MACD knowledge
        if self.cfg.add_macd:
            macd_line, sig_line, hist = self._macd(close, self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal)
            out[prefix + "macd"] = macd_line
            out[prefix + "macd_signal"] = sig_line
            out[prefix + "macd_hist"] = hist
            out[prefix + "macd_hist_norm"] = hist / (close.abs() + self.cfg.eps)

        # Stoch knowledge
        if self.cfg.add_stoch:
            k_pct, d_pct = self._stoch(df, self.cfg.stoch_k, self.cfg.stoch_d)
            out[prefix + f"stoch_k_{self.cfg.stoch_k}"] = k_pct
            out[prefix + f"stoch_d_{self.cfg.stoch_d}"] = d_pct
            out[prefix + "stoch_k_c"] = (k_pct - 50.0) / 50.0
            out[prefix + "stoch_d_c"] = (d_pct - 50.0) / 50.0

        # Volume knowledge
        if self.cfg.add_volume:
            vol = tickvol.astype("float64")
            vol_ema = self._ema(vol, self.cfg.vol_ema)
            out[prefix + "vol"] = vol
            out[prefix + f"vol_ema_{self.cfg.vol_ema}"] = vol_ema
            out[prefix + "vol_rel"] = vol / (vol_ema + self.cfg.eps)  # >1 = above normal
            out[prefix + "vol_log1p"] = (vol + 1.0).apply(lambda x: float(pd.NA) if pd.isna(x) else __import__("math").log1p(x))
            # fix possible NAs from lambda
            out[prefix + "vol_log1p"] = out[prefix + "vol_log1p"].astype("float64").fillna(0.0)

        # Candle structure knowledge
        out[prefix + "range"] = (high - low).abs()
        out[prefix + "body"] = (df["close"] - df["open"]).abs()
        out[prefix + "upper_wick"] = (high - df[["open", "close"]].max(axis=1)).clip(lower=0.0)
        out[prefix + "lower_wick"] = (df[["open", "close"]].min(axis=1) - low).clip(lower=0.0)

        # Normalize candle stuff by ATR if available
        if atr is not None:
            out[prefix + "range_over_atr"] = out[prefix + "range"] / (atr + self.cfg.eps)
            out[prefix + "body_over_atr"] = out[prefix + "body"] / (atr + self.cfg.eps)
            out[prefix + "uw_over_atr"] = out[prefix + "upper_wick"] / (atr + self.cfg.eps)
            out[prefix + "lw_over_atr"] = out[prefix + "lower_wick"] / (atr + self.cfg.eps)

        # Clean
        out = out.replace([float("inf"), float("-inf")], pd.NA)

        if self.cfg.dropna:
            out = out.dropna()

        return out

    def align_mtf(self, feat_m1: pd.DataFrame, feat_m5: pd.DataFrame, suffix_m5: str = "m5_") -> pd.DataFrame:
        """
        Align multi-timeframe features by forward-filling higher timeframe onto lower timeframe index.

        Example:
          feat_m1 index = every minute
          feat_m5 index = every 5 minutes (right-closed)
          -> reindex feat_m5 to feat_m1.index with forward fill
        """
        if feat_m1.empty:
            return feat_m1.copy()

        if feat_m5.empty:
            return feat_m1.copy()

        m5 = feat_m5.copy()
        m5.columns = [suffix_m5 + c for c in m5.columns]

        m5_aligned = m5.reindex(feat_m1.index, method="ffill")
        out = pd.concat([feat_m1, m5_aligned], axis=1)

        if self.cfg.dropna:
            out = out.dropna()

        return out