import numpy as np

from rabit.data.loader import MT5DataLoader
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig
from rabit.env.trading_env import TradingEnv


def random_policy(_row):
    # (dir, tp_mult, sl_mult, hold_max)
    dir_ = np.random.choice([0, 1, 2], p=[0.85, 0.075, 0.075])
    tp_mult = float(np.random.uniform(0.3, 1.2))
    sl_mult = float(np.random.uniform(0.3, 1.2))
    hold_max = int(np.random.randint(5, 60))
    return (dir_, tp_mult, sl_mult, hold_max)


loader = MT5DataLoader()
df = loader.load_m1("data/XAUUSD_M1.csv")
df = loader.to_numpy_ready(df)

fb = FeatureBuilder(FeatureConfig(dropna=False, add_atr=True))
feat = fb.build(df, prefix="")

# Merge ATR into df for env v1.2 (env requires column 'atr')
df2 = df.join(feat[["atr"]], how="left")

env = TradingEnv(
    df2,
    gap_close_minutes=60,
    gap_skip_minutes=180,
    spread_open_cap=200,
    force_close_on_spread=False,
)

ledger = env.run_backtest(random_policy)

print("Trades:", len(ledger.trades))
print("Final balance:", ledger.balance)