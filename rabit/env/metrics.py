from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class PerfMetrics:
    trades: int
    balance: float
    winrate: float
    profit_factor: float
    avg_pnl: float
    max_drawdown: float


def compute_metrics(ledger) -> PerfMetrics:
    pnls = [t.pnl for t in ledger.trades if t.pnl is not None]
    pnls = np.array(pnls, dtype=np.float64)

    trades = int(len(pnls))
    balance = float(np.sum(pnls)) if trades else 0.0

    if trades == 0:
        return PerfMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    winrate = float(len(wins) / trades)

    gross_win = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(-np.sum(losses)) if len(losses) else 0.0
    profit_factor = float(gross_win / (gross_loss + 1e-12))

    avg_pnl = float(np.mean(pnls))

    # drawdown from equity curve (ledger.balance is final, but equity_curve has timestamps)
    eq = [0.0]
    running = 0.0
    for p in pnls:
        running += float(p)
        eq.append(running)
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(np.max(dd))

    return PerfMetrics(
        trades=trades,
        balance=balance,
        winrate=winrate,
        profit_factor=profit_factor,
        avg_pnl=avg_pnl,
        max_drawdown=max_dd,
    )


def metrics_to_dict(m: PerfMetrics) -> Dict[str, Any]:
    return {
        "trades": m.trades,
        "balance": m.balance,
        "winrate": m.winrate,
        "profit_factor": m.profit_factor,
        "avg_pnl": m.avg_pnl,
        "max_drawdown": m.max_drawdown,
    }