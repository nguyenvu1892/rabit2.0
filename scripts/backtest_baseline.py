from rabit.data.loader import MT5DataLoader
from rabit.env.trading_env import TradingEnv
from rabit.strategies.baseline import BaselinePolicy


def main():
    loader = MT5DataLoader()
    df = loader.load_m1("data/XAUUSD_M1.csv")
    df = loader.to_numpy_ready(df)

    policy = BaselinePolicy(score_threshold=0.6)
    df2 = policy.prepare(df)

    # default tp/sl placeholder; nếu chưa nâng env v1.2 thì set tay
    env = TradingEnv(
        df2,
        tp_points=80,
        sl_points=80,
        gap_close_minutes=60,
        gap_skip_minutes=180,
        spread_open_cap=200,
        force_close_on_spread=False,
    )

    ledger = env.run_backtest(lambda row: policy.act(row))

    print("Trades:", len(ledger.trades))
    print("Final balance:", ledger.balance)


if __name__ == "__main__":
    main()