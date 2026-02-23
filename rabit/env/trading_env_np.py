from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any

import numpy as np
import pandas as pd

from rabit.env.ledger import Ledger
from rabit.env.execution import ExecutionEngine


@dataclass
class NPWindow:
    index: pd.Index
    close: np.ndarray  # float64
    spread: np.ndarray  # int64 (MT5 points)
    atr: np.ndarray  # float64 (price units)
    gap_min: np.ndarray  # int32 gap minutes from prev bar within window (gap_min[0]=0)


def _build_np_window(df_env: pd.DataFrame, start: int, end: int) -> NPWindow:
    """
    Build a numpy window view with precomputed gap minutes.
    df_env must have:
      - index: timestamp (pd.Timestamp)
      - columns: close, spread, atr
    """
    df_w = df_env.iloc[start:end]

    idx = df_w.index
    close = df_w["close"].to_numpy(dtype=np.float64, copy=False)
    spread = df_w["spread"].to_numpy(dtype=np.int64, copy=False)
    atr = df_w["atr"].to_numpy(dtype=np.float64, copy=False)

    # Precompute gap minutes
    # Convert to int64 nanoseconds for fast diff
    t_ns = idx.view("int64")
    d_ns = np.diff(t_ns, prepend=t_ns[0])
    gap_min = (d_ns // (60 * 1_000_000_000)).astype(np.int32)
    gap_min[0] = 0

    return NPWindow(index=idx, close=close, spread=spread, atr=atr, gap_min=gap_min)


class TradingEnvNP:
    """
    Numpy-optimized env for training (fast loop).
    Action: (dir, tp_mult, sl_mult, hold_max)
      dir: 0 hold, 1 long, 2 short
      tp/sl: ATR * mult (price units)
      hold_max: bars (minutes) to time-stop
    """

    def __init__(
        self,
        win: NPWindow,
        gap_close_minutes: int = 60,
        gap_skip_minutes: int = 180,
        spread_open_cap: int = 200,
        force_close_on_spread: bool = False,
        spread_force_close_cap: int = 500,
    ):
        self.win = win

        self.gap_close_minutes = gap_close_minutes
        self.gap_skip_minutes = gap_skip_minutes
        self.spread_open_cap = spread_open_cap
        self.force_close_on_spread = force_close_on_spread
        self.spread_force_close_cap = spread_force_close_cap

        self.ledger = Ledger()
        self.execution = ExecutionEngine()

        self.position = 0  # 0 flat, 1 long, -1 short
        self.entry_price: float | None = None
        self.entry_i: int | None = None
        self.tp_price: float | None = None
        self.sl_price: float | None = None
        self.hold_max: int | None = None

    def _reset_pos(self) -> None:
        self.position = 0
        self.entry_price = None
        self.entry_i = None
        self.tp_price = None
        self.sl_price = None
        self.hold_max = None

    def run_backtest(self, policy_func: Callable[[Any], Tuple[int, float, float, int]]) -> Ledger:
        close = self.win.close
        spread = self.win.spread
        atr = self.win.atr
        gap_min = self.win.gap_min
        idx = self.win.index

        n = close.shape[0]

        for i in range(n):
            # Skip huge data holes entirely (no trades, no management)
            if gap_min[i] > self.gap_skip_minutes:
                # Equivalent to "time passes" but we choose to ignore this bar
                continue

            ts = idx[i]
            c = float(close[i])
            sp = int(spread[i])

            # Force close if currently in position and gap is too large
            if self.position != 0 and gap_min[i] > self.gap_close_minutes:
                self.ledger.close_trade(ts, c)
                self._reset_pos()
                continue

            # Optional force close on spread spike
            if self.position != 0 and self.force_close_on_spread and sp > self.spread_force_close_cap:
                self.ledger.close_trade(ts, c)
                self._reset_pos()
                continue

            # ATR safety floor
            a = float(atr[i])
            if not np.isfinite(a) or a <= 0:
                a = 0.05
            a = max(a, 0.05)

            # ENTRY
            if self.position == 0:
                dir_, tp_mult, sl_mult, hold_max = policy_func(None)

                # Only open if spread acceptable
                if sp <= self.spread_open_cap:
                    if dir_ == 1:
                        fill = self.execution.market_fill(1, c, sp)
                        self.position = 1
                        self.entry_price = float(fill)
                        self.entry_i = i
                        self.hold_max = int(hold_max)

                        tp = max(0.05, float(tp_mult) * a)
                        sl = max(0.05, float(sl_mult) * a)
                        self.tp_price = self.entry_price + tp
                        self.sl_price = self.entry_price - sl

                        self.ledger.open_trade(ts, 1, self.entry_price, volume=1)

                    elif dir_ == 2:
                        fill = self.execution.market_fill(-1, c, sp)
                        self.position = -1
                        self.entry_price = float(fill)
                        self.entry_i = i
                        self.hold_max = int(hold_max)

                        tp = max(0.05, float(tp_mult) * a)
                        sl = max(0.05, float(sl_mult) * a)
                        self.tp_price = self.entry_price - tp
                        self.sl_price = self.entry_price + sl

                        self.ledger.open_trade(ts, -1, self.entry_price, volume=1)

            # MANAGE
            if self.position != 0 and self.entry_price is not None:
                # time stop
                if self.entry_i is not None and self.hold_max is not None:
                    if (i - self.entry_i) >= self.hold_max:
                        self.ledger.close_trade(ts, c)
                        self._reset_pos()
                        continue

                if self.position == 1:
                    if self.tp_price is not None and c >= self.tp_price:
                        self.ledger.close_trade(ts, c)
                        self._reset_pos()
                        continue
                    if self.sl_price is not None and c <= self.sl_price:
                        self.ledger.close_trade(ts, c)
                        self._reset_pos()
                        continue
                else:  # short
                    if self.tp_price is not None and c <= self.tp_price:
                        self.ledger.close_trade(ts, c)
                        self._reset_pos()
                        continue
                    if self.sl_price is not None and c >= self.sl_price:
                        self.ledger.close_trade(ts, c)
                        self._reset_pos()
                        continue

        return self.ledger


def make_np_window(df_env: pd.DataFrame, start: int, end: int) -> NPWindow:
    return _build_np_window(df_env, start, end)