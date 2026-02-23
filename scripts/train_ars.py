import json
import numpy as np

from rabit.data.loader import MT5DataLoader
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig
from rabit.rl.policy_linear import LinearPolicy
from rabit.rl.ars_trainer_parallel import ARSParallelTrainer, make_obs_matrix
from rabit.env.metrics import compute_metrics, metrics_to_dict
from rabit.env.trading_env import TradingEnv


def eval_full(df_env, X, policy) -> dict:
    env = TradingEnv(
        df_env,
        gap_close_minutes=60,
        gap_skip_minutes=180,
        spread_open_cap=200,
        force_close_on_spread=False,
    )

    idx = {"i": 0}

    def policy_func(_row):
        i = idx["i"]
        idx["i"] += 1
        return policy.act(X[i])

    ledger = env.run_backtest(policy_func)
    m = compute_metrics(ledger)
    return metrics_to_dict(m)


def main():
    loader = MT5DataLoader()
    df = loader.load_m1("data/XAUUSD_M1.csv")
    df = loader.to_numpy_ready(df)

    fb = FeatureBuilder(FeatureConfig(dropna=False, add_atr=True))
    feat = fb.build(df, prefix="")

    df_env_all = df.join(feat[["atr"]], how="left")

    feature_cols = [
        "atr",
        "atr_norm",
        "spread_over_atr",
        "ema_spread",
        "ema_fast_slope",
        "ema_slow_slope",
        "rsi_c",
        "macd_hist_norm",
        "stoch_k_c",
        "vol_rel",
        "range_over_atr",
        "body_over_atr",
        "uw_over_atr",
        "lw_over_atr",
    ]
    feature_cols = [c for c in feature_cols if c in feat.columns]
    df_feat = feat[feature_cols].copy()
    X_all = make_obs_matrix(df_feat, feature_cols)

    # ---- Time split (OOS) ----
    n = len(df_env_all)
    cut = int(n * 0.7)
    df_train = df_env_all.iloc[:cut].copy()
    X_train = X_all[:cut]

    df_test = df_env_all.iloc[cut:].copy()
    X_test = X_all[cut:]

    # ---- Train with windowed eval (fast) on train slice ----
    policy = LinearPolicy(n_features=X_all.shape[1])

    trainer = ARSParallelTrainer(
        policy,
        sigma=0.03,
        alpha=0.02,
        n_directions=8,
        top_k=4,
        eval_windows=6,
        window_size=1500,
        max_workers=6,  # bạn có thể chỉnh = số core - 2
        dd_lambda=0.25,
        trade_lambda=0.05,
        dd_scale=300.0,
        trade_scale=250.0,
    )
    history, best_theta, best_reward = trainer.train(df_train, X_train, iters=15)

    # Save best theta
    np.save("data/ars_best_theta.npy", best_theta)

    # Evaluate full-slice metrics
    policy.set_params_flat(best_theta)
    train_metrics = eval_full(df_train, X_train, policy)
    test_metrics = eval_full(df_test, X_test, policy)

    payload = {
        "best_window_reward": float(best_reward),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_cols": feature_cols,
        "history": history,
    }

    with open("data/ars_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved: data/ars_best_theta.npy and data/ars_report.json")
    print("TRAIN:", train_metrics)
    print("TEST :", test_metrics)


if __name__ == "__main__":
    main()