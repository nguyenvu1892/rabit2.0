import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rabit.env.trading_env import TradingEnv


def _make_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="min")
    return pd.DataFrame(
        {
            "close": [100.0, 101.0, 101.0],
            "spread": [0, 0, 0],
            "atr": [1.0, 1.0, 1.0],
        },
        index=idx,
    )


def _run_size(size: float) -> float:
    df = _make_df()
    env = TradingEnv(df, spread_open_cap=10_000)
    opened = False

    def policy(_row):
        nonlocal opened
        if not opened:
            opened = True
            return (1, 1.0, 1.0, 1, size)
        return (0, 0.0, 0.0, 1)

    ledger = env.run_backtest(policy)
    if not ledger.trades or ledger.trades[0].pnl is None:
        raise AssertionError("Expected a single closed trade with pnl.")
    return float(ledger.trades[0].pnl)


def main() -> None:
    pnl_full = _run_size(1.0)
    pnl_half = _run_size(0.5)

    if abs(pnl_half - pnl_full * 0.5) > 1e-9:
        raise AssertionError(f"Expected half pnl. full={pnl_full}, half={pnl_half}")

    pnl_zero = _run_size(0.0)
    if abs(pnl_zero) > 1e-9:
        raise AssertionError(f"Expected zero pnl for size=0. got={pnl_zero}")

    print("test_env_size OK")


if __name__ == "__main__":
    main()
