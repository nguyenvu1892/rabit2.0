from __future__ import annotations

import pandas as pd

from rabit.env.ledger import Ledger
from rabit.env.execution import ExecutionEngine

# Optional realism modules (TASK-2E)
try:
    from rabit.env.execution_model import ExecutionModel
except Exception:
    ExecutionModel = None  # type: ignore

try:
    from rabit.env.session_filter import SessionFilter
except Exception:
    SessionFilter = None  # type: ignore


class TradingEnv:
    """
    TradingEnv v1.3 (execution realism ready):
      - action = (dir, tp_mult, sl_mult, hold_max[, size])
      - TP/SL computed from ATR * multiplier (knowledge-driven, no fixed setup)
      - spread/gap filters preserved
      - one position at a time (scalping)
      - optional:
          * execution_model: slippage + commission
          * session_filter: allow entries only in sessions
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gap_close_minutes: int = 60,
        gap_skip_minutes: int = 180,
        spread_open_cap: int = 200,
        force_close_on_spread: bool = False,
        spread_force_close_cap: int = 500,
        point_value: float = 0.01,   # XAU: 1 point = 0.01
        execution_model=None,
        session_filter=None,
    ):
        self.df = df
        self.point_value = point_value

        self.gap_close_minutes = gap_close_minutes
        self.gap_skip_minutes = gap_skip_minutes
        self.spread_open_cap = spread_open_cap
        self.force_close_on_spread = force_close_on_spread
        self.spread_force_close_cap = spread_force_close_cap

        self.ledger = Ledger()

        # Legacy fill engine (kept for backward compatibility)
        self.execution = ExecutionEngine()

        # New realism modules (can be None)
        self.execution_model = execution_model  # ExecutionModel | None
        self.session_filter = session_filter    # SessionFilter | None

        self.position = 0
        self.entry_price: float | None = None
        self.entry_ts: pd.Timestamp | None = None
        self.entry_i: int | None = None

        self.tp_price: float | None = None
        self.sl_price: float | None = None
        self.hold_max: int | None = None

        self.last_ts: pd.Timestamp | None = None

    def _gap_minutes(self, ts: pd.Timestamp) -> int:
        if self.last_ts is None:
            return 0
        dt = ts - self.last_ts
        return int(dt.total_seconds() // 60)

    def _apply_commission(self, volume: float) -> None:
        """
        Commission is applied directly to ledger balance if possible.
        This is intentionally defensive to avoid breaking older Ledger versions.
        """
        if self.execution_model is None:
            return
        if not hasattr(self.execution_model, "commission_cost_per_side"):
            return

        cost = float(self.execution_model.commission_cost_per_side(volume))
        if cost <= 0:
            return

        # Try common ledger patterns
        if hasattr(self.ledger, "balance"):
            try:
                self.ledger.balance -= cost
                return
            except Exception:
                pass

        # If ledger has a method
        for fn in ["apply_cost", "add_cost", "charge_cost"]:
            if hasattr(self.ledger, fn):
                try:
                    getattr(self.ledger, fn)(cost)
                    return
                except Exception:
                    pass
        # If nothing works, silently ignore (keeps backward compatible)

    def step(
        self,
        i: int,
        action: tuple[int, float, float, int] | tuple[int, float, float, int, float],
    ) -> None:
        """
        action:
          dir: 0 hold, 1 long, 2 short
          tp_mult: e.g. 0.6 (ATR multiple)
          sl_mult: e.g. 0.6
          hold_max: bars/minutes (time stop)
          size: optional [0.0, 1.0]
        """
        ts = self.df.index[i]
        row = self.df.iloc[i]
        gap_min = self._gap_minutes(ts)

        # skip huge data holes
        if gap_min > self.gap_skip_minutes:
            self.last_ts = ts
            return

        close_price = float(row["close"])
        spread_points = int(row["spread"])
        spread_price = float(spread_points) * float(self.point_value)

        # force close on big gap
        if self.position != 0 and gap_min > self.gap_close_minutes:
            # close at mid (or could also use execution_model, but keep simple for now)
            self._apply_commission(volume=float(getattr(self.ledger, "current_volume", 1.0)))
            self.ledger.close_trade(ts, close_price)
            self._reset_position()
            self.last_ts = ts
            return

        # optional force close on spread spike
        if self.position != 0 and self.force_close_on_spread and spread_points > self.spread_force_close_cap:
            self._apply_commission(volume=float(getattr(self.ledger, "current_volume", 1.0)))
            self.ledger.close_trade(ts, close_price)
            self._reset_position()
            self.last_ts = ts
            return

        # compute atr (knowledge input)
        if "atr" not in row.index:
            raise ValueError("df must contain column 'atr' for TradingEnv v1.3")
        atr = float(row["atr"])
        atr = max(atr, 0.05)  # safety floor (price units)

        # ENTRY
        if self.position == 0:
            if len(action) == 4:
                dir_, tp_mult, sl_mult, hold_max = action
                size = 1.0
            elif len(action) == 5:
                dir_, tp_mult, sl_mult, hold_max, size = action
            else:
                raise ValueError("action must be (dir, tp, sl, hold) or (dir, tp, sl, hold, size)")

            size = float(size)
            if size < 0.0:
                size = 0.0
            elif size > 1.0:
                size = 1.0

            # session filter (ONLY entry)
            if dir_ != 0 and self.session_filter is not None:
                try:
                    if not self.session_filter.is_open(ts):
                        self.last_ts = ts
                        return
                except Exception:
                    # if session filter errors, do not block
                    pass

            # must pass spread filter
            if spread_points <= self.spread_open_cap:
                if dir_ == 1:
                    # fill with realism if available
                    if self.execution_model is not None and hasattr(self.execution_model, "market_fill"):
                        fill = float(self.execution_model.market_fill(
                            direction=+1,
                            mid_price=close_price,
                            spread_price=spread_price,
                            atr_price=atr,
                        ))
                    else:
                        fill = self.execution.market_fill(1, close_price, spread_points)

                    self.position = 1
                    self.entry_price = fill
                    self.entry_ts = ts
                    self.entry_i = i
                    self.hold_max = int(hold_max)

                    tp_dist = max(0.05, float(tp_mult) * atr)
                    sl_dist = max(0.05, float(sl_mult) * atr)
                    self.tp_price = fill + tp_dist
                    self.sl_price = fill - sl_dist

                    self.ledger.open_trade(ts, 1, fill, volume=size)
                    self._apply_commission(volume=size)  # entry commission

                elif dir_ == 2:
                    if self.execution_model is not None and hasattr(self.execution_model, "market_fill"):
                        fill = float(self.execution_model.market_fill(
                            direction=-1,
                            mid_price=close_price,
                            spread_price=spread_price,
                            atr_price=atr,
                        ))
                    else:
                        fill = self.execution.market_fill(-1, close_price, spread_points)

                    self.position = -1
                    self.entry_price = fill
                    self.entry_ts = ts
                    self.entry_i = i
                    self.hold_max = int(hold_max)

                    tp_dist = max(0.05, float(tp_mult) * atr)
                    sl_dist = max(0.05, float(sl_mult) * atr)
                    self.tp_price = fill - tp_dist
                    self.sl_price = fill + sl_dist

                    # keep legacy ledger signature (do not break)
                    try:
                        self.ledger.open_trade(ts, -1, fill, volume=size)
                    except TypeError:
                        # fallback if older ledger expects different args
                        self.ledger.open_trade(ts, -1, fill)

                    self._apply_commission(volume=size)  # entry commission

        # MANAGE
        if self.position != 0 and self.entry_price is not None:
            # time stop
            if self.entry_i is not None and self.hold_max is not None:
                if (i - self.entry_i) >= self.hold_max:
                    self._apply_commission(volume=float(getattr(self.ledger, "current_volume", 1.0)))
                    self.ledger.close_trade(ts, close_price)
                    self._reset_position()
                    self.last_ts = ts
                    return

            # price stop/take (close at mid for now; can add exit slippage later)
            if self.position == 1:
                if self.tp_price is not None and close_price >= self.tp_price:
                    self._apply_commission(volume=float(getattr(self.ledger, "current_volume", 1.0)))
                    self.ledger.close_trade(ts, close_price)
                    self._reset_position()
                elif self.sl_price is not None and close_price <= self.sl_price:
                    self._apply_commission(volume=float(getattr(self.ledger, "current_volume", 1.0)))
                    self.ledger.close_trade(ts, close_price)
                    self._reset_position()

            elif self.position == -1:
                if self.tp_price is not None and close_price <= self.tp_price:
                    self._apply_commission(volume=float(getattr(self.ledger, "current_volume", 1.0)))
                    self.ledger.close_trade(ts, close_price)
                    self._reset_position()
                elif self.sl_price is not None and close_price >= self.sl_price:
                    self._apply_commission(volume=float(getattr(self.ledger, "current_volume", 1.0)))
                    self.ledger.close_trade(ts, close_price)
                    self._reset_position()

        self.last_ts = ts

    def _reset_position(self) -> None:
        self.position = 0
        self.entry_price = None
        self.entry_ts = None
        self.entry_i = None
        self.tp_price = None
        self.sl_price = None
        self.hold_max = None

    def run_backtest(self, policy_func):
        for i in range(len(self.df)):
            action = policy_func(self.df.iloc[i])
            self.step(i, action)
        return self.ledger