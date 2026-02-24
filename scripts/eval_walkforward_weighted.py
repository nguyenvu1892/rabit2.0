import json
import os
import numpy as np

from rabit.data.loader import MT5DataLoader
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig
from rabit.env.metrics import compute_metrics, metrics_to_dict
from rabit.env.trading_env import TradingEnv

from rabit.rl.confidence_gate import make_default_gate
from rabit.rl.confidence_weighting import ConfidenceWeighter, ConfidenceWeighterConfig
from rabit.rl.policy_linear import LinearPolicy

from rabit.regime.regime_detector import RegimeDetector
from rabit.rl.regime_policy_bank import RegimePolicyBank


def _ensure_reports_dir():
    os.makedirs("data/reports", exist_ok=True)


def _pick_existing(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def make_obs_matrix(feat_df, feature_cols: list[str], mu=None, sd=None):
    X = feat_df[feature_cols].to_numpy(dtype=np.float64)
    if mu is None:
        mu = np.nanmean(X, axis=0)
    if sd is None:
        sd = np.nanstd(X, axis=0) + 1e-12
    Xn = (X - mu) / (sd + 1e-12)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    return Xn, mu, sd


def load_policy_system(n_features: int):
    p_bank = _pick_existing(["data/ars_best_theta_regime_bank_v2.npz", "data/ars_best_theta_regime_bank.npz"])
    if p_bank:
        z = np.load(p_bank)
        policies = {}
        for k in ["trend", "range", "highvol"]:
            if k in z:
                p = LinearPolicy(n_features=n_features)
                p.set_params_flat(z[k].astype(np.float64))
                policies[k] = p
        return "regime_bank", RegimePolicyBank(policies, fallback=None), p_bank

    p_wf = _pick_existing(["data/ars_best_theta_walkforward.npy", "data/ars_best_theta_mtf.npy"])
    if p_wf:
        theta = np.load(p_wf).astype(np.float64)
        p = LinearPolicy(n_features=n_features)
        p.set_params_flat(theta)
        return "linear", p, p_wf

    p_lin = _pick_existing(["data/ars_best_theta.npy"])
    if p_lin:
        theta = np.load(p_lin).astype(np.float64)
        p = LinearPolicy(n_features=n_features)
        p.set_params_flat(theta)
        return "linear", p, p_lin

    raise FileNotFoundError("No model found in data/.")


def run_env(df_env, feat_env, X_env, mode, model, regime_arr=None, gate=None, weighter=None):
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

        # model action
        if mode == "linear":
            dir_, tp_mult, sl_mult, hold_max = model.act(X_env[i])
        else:
            r = "mixed"
            if regime_arr is not None:
                r = str(regime_arr[i])
            dir_, tp_mult, sl_mult, hold_max = model.act(X_env[i], r)

        if dir_ == 0:
            return (0, 0.8, 0.8, 20)

        # baseline: no weighting
        if gate is None or weighter is None:
            return (dir_, tp_mult, sl_mult, hold_max)

        # weighted
        feat_row = feat_env.iloc[i]
        confidence, _allow, _reason = gate.evaluate(feat_row)
        size = weighter.size(confidence)
        if size <= 1e-6:
            return (0, 0.8, 0.8, 20)
        return (dir_, tp_mult, sl_mult, hold_max, size)

    ledger = env.run_backtest(policy_func)
    return metrics_to_dict(compute_metrics(ledger))


def main():
    _ensure_reports_dir()

    # params
    n_folds = 8
    test_frac = 0.10   # 10% each fold test (rolling)
    min_train_frac = 0.30
    power = 2.5

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

    n = len(df_env_all)
    test_len = int(n * test_frac)
    min_train_len = int(n * min_train_frac)

    fold_starts = []
    # last fold ends at end of series
    last_start = n - test_len
    if n_folds > 1:
        step = max(1, (last_start - min_train_len) // (n_folds - 1))
    else:
        step = last_start - min_train_len

    s = min_train_len
    while s <= last_start and len(fold_starts) < n_folds:
        fold_starts.append(s)
        s += step

    # prepare gate + weighter
    gate = make_default_gate()
    if hasattr(gate, "cfg"):
        gate.cfg.slope_min = 0.0  # ensure no slope floor
        # IMPORTANT: no hard threshold usage in weighting; we only use confidence scalar.
    weighter = ConfidenceWeighter(ConfidenceWeighterConfig(power=power))

    results = []

    for k, train_end in enumerate(fold_starts):
        test_start = train_end
        test_end = min(n, test_start + test_len)

        df_train = df_env_all.iloc[:train_end].copy()
        feat_train = feat.iloc[:train_end].copy()

        df_test = df_env_all.iloc[test_start:test_end].copy()
        feat_test = feat.iloc[test_start:test_end].copy()

        # fit normalizer on TRAIN only
        X_train, mu, sd = make_obs_matrix(feat_train, feature_cols)
        X_test, _, _ = make_obs_matrix(feat_test, feature_cols, mu=mu, sd=sd)

        # load policy based on feature dim
        mode, model, model_path = load_policy_system(n_features=X_test.shape[1])

        # fit regime detector on TRAIN only (if needed)
        regime_test = None
        if mode == "regime_bank":
            det = RegimeDetector().fit(feat_train)
            regime_test = det.predict(feat_test)

        base_m = run_env(df_test, feat_test, X_test, mode, model, regime_arr=regime_test, gate=None, weighter=None)
        w_m = run_env(df_test, feat_test, X_test, mode, model, regime_arr=regime_test, gate=gate, weighter=weighter)

        rec = {
            "fold": k,
            "train_end": int(train_end),
            "test_range": [int(test_start), int(test_end)],
            "ts_start": str(df_test.index[0]) if len(df_test) else None,
            "ts_end": str(df_test.index[-1]) if len(df_test) else None,
            "baseline": base_m,
            "weighted": w_m,
            "pf_change": float(w_m["profit_factor"] - base_m["profit_factor"]),
            "maxdd_change": float((w_m["max_drawdown"] - base_m["max_drawdown"]) / max(1e-9, base_m["max_drawdown"])),
        }
        results.append(rec)

        print(f"[Fold {k}] {rec['ts_start']} → {rec['ts_end']}")
        print("  Baseline:", base_m)
        print("  Weighted:", w_m)

    report = {
        "model_path": model_path,
        "mode": mode,
        "power": power,
        "n_folds": len(results),
        "test_frac": test_frac,
        "min_train_frac": min_train_frac,
        "results": results,
    }

    out_path = "data/reports/walkforward_weighted.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()