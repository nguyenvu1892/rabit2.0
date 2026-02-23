import json
import numpy as np

from rabit.data.loader import MT5DataLoader
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig
from rabit.regime.regime_detector import RegimeDetector
from rabit.rl.policy_linear import LinearPolicy
from rabit.rl.ars_trainer_regime import ARSTrainerRegime, make_obs_matrix
from rabit.rl.regime_policy_bank import RegimePolicyBank
from rabit.env.metrics import compute_metrics, metrics_to_dict
from rabit.env.trading_env import TradingEnv


def eval_metrics_switch(df_env, X, regime_arr, bank: RegimePolicyBank):
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
        r = str(regime_arr[i])
        return bank.act(X[i], r)

    ledger = env.run_backtest(policy_func)
    m = compute_metrics(ledger)
    return metrics_to_dict(m)


def robust_score(metrics):
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
    X_all = make_obs_matrix(feat[feature_cols].copy(), feature_cols)

    n = len(df_env_all)

    # Holdout last 15%
    test_cut = int(n * 0.85)
    df_main = df_env_all.iloc[:test_cut].copy()
    X_main = X_all[:test_cut]
    feat_main = feat.iloc[:test_cut].copy()

    df_hold = df_env_all.iloc[test_cut:].copy()
    X_hold = X_all[test_cut:]
    feat_hold = feat.iloc[test_cut:].copy()

    # ✅ Train only trend + range. Highvol => HOLD
    regimes_to_train = ["trend", "range"]

    # ✅ Fit detector once on MAIN to stabilize thresholds
    det = RegimeDetector().fit(feat_main)
    R_main = det.predict(feat_main)
    R_hold = det.predict(feat_hold)

    folds = 6
    fold_size = len(df_main) // folds

    report = {
        "feature_cols": feature_cols,
        "regimes": regimes_to_train,
        "detector_thresholds": det.thresholds(),
        "folds": [],
    }

    best_theta_by_regime = {r: None for r in regimes_to_train}
    best_val_score_by_regime = {r: -1e9 for r in regimes_to_train}

    for f in range(1, folds):
        tr_end = f * fold_size
        val_start = tr_end
        val_end = min((f + 1) * fold_size, len(df_main))

        df_train = df_main.iloc[:tr_end].copy()
        X_train = X_main[:tr_end]
        R_train = R_main[:tr_end]

        df_val = df_main.iloc[val_start:val_end].copy()
        X_val = X_main[val_start:val_end]
        R_val = R_main[val_start:val_end]

        fold_rec = {"fold": f, "regimes": {}}

        for regime_name in regimes_to_train:
            policy = LinearPolicy(n_features=X_all.shape[1])

            trainer = ARSTrainerRegime(
                policy,
                sigma=0.03,
                alpha=0.02,
                n_directions=8,
                top_k=4,
                eval_windows=8,
                window_size=2000,
                verbose=False,
                dd_lambda=0.25,
                trade_lambda=0.05,
                dd_scale=300.0,
                trade_scale=250.0,
                target_regime=regime_name,
            )

            hist, theta_r, best_r = trainer.train(df_train, X_train, R_train, iters=12)

            bank = RegimePolicyBank({regime_name: policy}, fallback=None)
            val_metrics = eval_metrics_switch(df_val, X_val, R_val, bank)
            val_score = robust_score(val_metrics)

            fold_rec["regimes"][regime_name] = {
                "best_window_reward": float(best_r),
                "val_metrics": val_metrics,
                "val_score": float(val_score),
            }

            print(f"[WF-Regime] fold={f} regime={regime_name} val_score={val_score:.4f} val={val_metrics}")

            if val_score > best_val_score_by_regime[regime_name]:
                best_val_score_by_regime[regime_name] = float(val_score)
                best_theta_by_regime[regime_name] = theta_r.copy()

        report["folds"].append(fold_rec)

    # Final bank (trend + range). Others HOLD.
    final_policies = {}
    for r in regimes_to_train:
        p = LinearPolicy(n_features=X_all.shape[1])
        p.set_params_flat(best_theta_by_regime[r])
        final_policies[r] = p

    bank = RegimePolicyBank(final_policies, fallback=None)

    hold_metrics = eval_metrics_switch(df_hold, X_hold, R_hold, bank)
    hold_score = robust_score(hold_metrics)

    report["best_val_score_by_regime"] = best_val_score_by_regime
    report["hold_metrics"] = hold_metrics
    report["hold_score"] = float(hold_score)

    np.savez(
        "data/ars_best_theta_regime_bank_v2.npz",
        trend=best_theta_by_regime["trend"],
        range=best_theta_by_regime["range"],
    )
    with open("data/ars_report_walkforward_regime_v2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved: data/ars_best_theta_regime_bank_v2.npz and data/ars_report_walkforward_regime_v2.json")
    print("HOLD:", hold_metrics, "hold_score:", hold_score)


if __name__ == "__main__":
    main()