from __future__ import annotations
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


class BaselinePolicy:
    """
    Soft-score policy (not hard setup):
    - Score from EMA trend + RSI momentum
    - Acts only when score passes threshold
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        score_threshold: float = 0.6,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.score_threshold = score_threshold

        self._cache_ready = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema_fast"] = _ema(out["close"], self.ema_fast)
        out["ema_slow"] = _ema(out["close"], self.ema_slow)
        out["rsi"] = _rsi(out["close"], self.rsi_period)
        out["atr"] = _atr(out, 14)
        self._cache_ready = True
        return out

    def act(self, row: pd.Series) -> int:
        """
        Return action:
          0 hold, 1 long, 2 short
        """
        if not self._cache_ready:
            raise RuntimeError("Call policy.prepare(df) first")

        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        rsi = float(row["rsi"])

        # Trend component
        trend = 1.0 if ema_fast > ema_slow else -1.0

        # Momentum component (soft)
        # Long bias when RSI > 55, Short bias when RSI < 45
        mom = 0.0
        if rsi > 55:
            mom = (rsi - 55) / 45  # scale 0..1
        elif rsi < 45:
            mom = -((45 - rsi) / 45)

        # Score in [-1, 1]
        score = 0.6 * trend + 0.4 * mom

        if score >= self.score_threshold:
            return 1
        if score <= -self.score_threshold:
            return 2
        return 0

    def tp_sl_points(self, row: pd.Series, tp_atr: float = 0.6, sl_atr: float = 0.6) -> tuple[int, int]:
        """
        Dynamic TP/SL in points using ATR (XAU 1 point=0.01)
        tp_atr/sl_atr are multipliers of ATR in price units.
        """
        atr = float(row["atr"])
        tp_price = max(0.1, tp_atr * atr)
        sl_price = max(0.1, sl_atr * atr)

        tp_points = int(round(tp_price / 0.01))
        sl_points = int(round(sl_price / 0.01))
        return tp_points, sl_points