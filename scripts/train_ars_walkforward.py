import json
import numpy as np

from rabit.data.loader import MT5DataLoader
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig
from rabit.rl.policy_linear import LinearPolicy
from rabit.rl.ars_trainer import ARSTrainer, make_obs_matrix
from rabit.env.metrics import compute_metrics, metrics_to_dict
from rabit.env.trading_env import TradingEnv


def eval_metrics(df_env, X, policy):
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


def robust_score(metrics):
    # A single scalar to compare across folds (higher better)
    # PF term centered at 1, penalize drawdown and overtrading.
    pf = metrics["profit_factor"]
    dd = metrics["max_drawdown"]
    tr = metrics["trades"]
    return (pf - 1.0) - 0.25 * (dd / 300.0) - 0.05 * (tr / 250.0)


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

    n = len(df_env_all)

    # Holdout last 15% as final test
    test_cut = int(n * 0.85)
    df_main = df_env_all.iloc[:test_cut].copy()
    X_main = X_all[:test_cut]
    df_hold = df_env_all.iloc[test_cut:].copy()
    X_hold = X_all[test_cut:]

    # Walk-forward folds on main (e.g. 6 folds)
    folds = 6
    fold_size = len(df_main) // folds

    # store best across folds
    best_theta_global = None
    best_val_score_global = -1e9
    report = {"folds": [], "feature_cols": feature_cols}

    for f in range(1, folds):
        # train = [0 : f*fold_size), val = [f*fold_size : (f+1)*fold_size)
        tr_end = f * fold_size
        val_start = tr_end
        val_end = min((f + 1) * fold_size, len(df_main))

        df_train = df_main.iloc[:tr_end].copy()
        X_train = X_main[:tr_end]

        df_val = df_main.iloc[val_start:val_end].copy()
        X_val = X_main[val_start:val_end]

        policy = LinearPolicy(n_features=X_all.shape[1])

        trainer = ARSTrainer(
            policy,
            sigma=0.03,
            alpha=0.02,
            n_directions=8,
            top_k=4,
            eval_windows=10,
            window_size=2000,
            verbose=False,
            dd_lambda=0.25,
            trade_lambda=0.05,
            dd_scale=300.0,
            trade_scale=250.0,
        )

        hist, best_theta, best_reward = trainer.train(df_train, X_train, iters=15)

        # Evaluate on VAL (full slice)
        policy.set_params_flat(best_theta)
        val_metrics = eval_metrics(df_val, X_val, policy)
        val_score = robust_score(val_metrics)

        fold_rec = {
            "fold": f,
            "train_len": int(len(df_train)),
            "val_len": int(len(df_val)),
            "best_window_reward": float(best_reward),
            "val_metrics": val_metrics,
            "val_score": float(val_score),
        }
        report["folds"].append(fold_rec)

        print(f"[WF] fold={f} val_score={val_score:.4f} val={val_metrics}")

        if val_score > best_val_score_global:
            best_val_score_global = val_score
            best_theta_global = best_theta.copy()

    # Final holdout evaluation
    final_policy = LinearPolicy(n_features=X_all.shape[1])
    final_policy.set_params_flat(best_theta_global)

    hold_metrics = eval_metrics(df_hold, X_hold, final_policy)
    hold_score = robust_score(hold_metrics)

    report["best_val_score_global"] = float(best_val_score_global)
    report["hold_metrics"] = hold_metrics
    report["hold_score"] = float(hold_score)

    np.save("data/ars_best_theta_walkforward.npy", best_theta_global)
    with open("data/ars_report_walkforward.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved: data/ars_best_theta_walkforward.npy and data/ars_report_walkforward.json")
    print("HOLD:", hold_metrics, "hold_score:", hold_score)


if __name__ == "__main__":
    main()