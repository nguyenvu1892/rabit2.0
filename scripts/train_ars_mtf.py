import json
import numpy as np

from rabit.data.loader import MT5DataLoader
from rabit.data.resampler import ResampleConfig, resample_ohlcv_m1_to_higher
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig
from rabit.rl.policy_linear import LinearPolicy
from rabit.rl.ars_trainer import ARSTrainer, make_obs_matrix
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
    df_m1 = loader.load_m1("data/XAUUSD_M1.csv")
    df_m1 = loader.to_numpy_ready(df_m1)

    # Resample M5 from M1
    df_m5 = resample_ohlcv_m1_to_higher(df_m1, ResampleConfig(rule="5min", spread_agg="max"))
    df_m5 = loader.to_numpy_ready(df_m5)

    fb = FeatureBuilder(FeatureConfig(dropna=False, add_atr=True))
    feat_m1 = fb.build(df_m1, prefix="m1_")
    feat_m5 = fb.build(df_m5, prefix="m5_")

    feat = fb.align_mtf(feat_m1, feat_m5, suffix_m5="")

    # Env needs ATR (use m1 atr)
    df_env_all = df_m1.join(feat[["m1_atr"]], how="left")
    df_env_all = df_env_all.rename(columns={"m1_atr": "atr"})

    # Feature columns (M1 + M5 knowledge)
    feature_cols = [
        # M1
        "m1_atr_norm",
        "m1_spread_over_atr",
        "m1_ema_spread",
        "m1_ema_fast_slope",
        "m1_ema_slow_slope",
        "m1_rsi_c",
        "m1_macd_hist_norm",
        "m1_stoch_k_c",
        "m1_vol_rel",
        "m1_range_over_atr",
        "m1_body_over_atr",
        "m1_uw_over_atr",
        "m1_lw_over_atr",
        # M5 context
        "m5_atr_norm",
        "m5_spread_over_atr",
        "m5_ema_spread",
        "m5_ema_fast_slope",
        "m5_ema_slow_slope",
        "m5_rsi_c",
        "m5_macd_hist_norm",
        "m5_stoch_k_c",
        "m5_vol_rel",
        "m5_range_over_atr",
    ]
    feature_cols = [c for c in feature_cols if c in feat.columns]

    df_feat = feat[feature_cols].copy()
    X_all = make_obs_matrix(df_feat, feature_cols)

    # Time split (train past, test future)
    n = len(df_env_all)
    cut = int(n * 0.7)

    df_train = df_env_all.iloc[:cut].copy()
    X_train = X_all[:cut]

    df_test = df_env_all.iloc[cut:].copy()
    X_test = X_all[cut:]

    policy = LinearPolicy(n_features=X_all.shape[1])

    trainer = ARSTrainer(
        policy,
        sigma=0.03,
        alpha=0.02,
        n_directions=10,
        top_k=5,
        eval_windows=10,
        window_size=2500,
        verbose=False,
        dd_lambda=0.25,
        trade_lambda=0.05,
        dd_scale=300.0,
        trade_scale=250.0,
    )

    history, best_theta, best_reward = trainer.train(df_train, X_train, iters=25)

    np.save("data/ars_best_theta_mtf.npy", best_theta)

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

    with open("data/ars_report_mtf.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved: data/ars_best_theta_mtf.npy and data/ars_report_mtf.json")
    print("TRAIN:", train_metrics)
    print("TEST :", test_metrics)


if __name__ == "__main__":
    main()