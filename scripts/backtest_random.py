import numpy as np

from rabit.data.loader import MT5DataLoader
from rabit.env.trading_env import TradingEnv


def random_policy(_row):
    return np.random.choice([0, 1, 2], p=[0.85, 0.075, 0.075])


loader = MT5DataLoader()
df = loader.load_m1("data/XAUUSD_M1.csv")
df = loader.to_numpy_ready(df)

env = TradingEnv(
    df,
    tp_points=80,
    sl_points=80,
    gap_close_minutes=60,
    gap_skip_minutes=180,
    spread_open_cap=200,
    force_close_on_spread=False,
)

ledger = env.run_backtest(random_policy)

print("Trades:", len(ledger.trades))
print("Final balance:", ledger.balance)