from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: int  # 1 long, -1 short
    entry_price: float
    exit_price: Optional[float]
    volume: float
    size: float
    pnl: Optional[float]


class Ledger:
    def __init__(self):
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.balance = 0.0

    def open_trade(self, ts, direction, price, volume, size=None):
        if size is None:
            size = float(volume)
        trade = Trade(
            entry_time=ts,
            exit_time=None,
            direction=direction,
            entry_price=price,
            exit_price=None,
            volume=volume,
            size=float(size),
            pnl=None,
        )
        self.trades.append(trade)

    def close_trade(self, ts, price):
        trade = self.trades[-1]
        trade.exit_time = ts
        trade.exit_price = price

        trade.pnl = (price - trade.entry_price) * trade.direction * trade.size
        self.balance += trade.pnl
        self.equity_curve.append((ts, self.balance))
