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


def make_obs_matrix(feat_df, feature_cols: list[str]) -> np.ndarray:
    X = feat_df[feature_cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    X = (X - mu) / sd
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def load_policy_system(n_features: int):
    p_bank = _pick_existing(["data/ars_best_theta_regime_bank_v2.npz", "data/ars_best_theta_regime_bank.npz"])
    if p_bank and p_bank.endswith(".npz"):
        z = np.load(p_bank)
        policies = {}
        for k in ["trend", "range", "highvol"]:
            if k in z:
                p = LinearPolicy(n_features=n_features)
                p.set_params_flat(z[k].astype(np.float64))
                policies[k] = p
        bank = RegimePolicyBank(policies, fallback=None)
        return "regime_bank", bank, p_bank

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

    raise FileNotFoundError("No model found in data/. Expected theta.npy or regime_bank.npz")


def run_backtest(df_env, feat_df, X: np.ndarray, regime_arr: np.ndarray | None, mode: str, model_obj, gate, weighter):
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

        # base action from model
        if mode == "linear":
            dir_, tp_mult, sl_mult, hold_max = model_obj.act(X[i])
        else:
            r = "mixed"
            if regime_arr is not None:
                r = str(regime_arr[i])
            dir_, tp_mult, sl_mult, hold_max = model_obj.act(X[i], r)

        if dir_ == 0:
            return (0, 0.8, 0.8, 20)

        # confidence -> size
        feat_row = feat_df.iloc[i]
        confidence, _allow, _reason = gate.evaluate(feat_row)
        size = weighter.size(confidence)

        # If size is basically 0 => HOLD
        if size <= 1e-6:
            return (0, 0.8, 0.8, 20)

        return (dir_, tp_mult, sl_mult, hold_max, size)

    ledger = env.run_backtest(policy_func)
    m = compute_metrics(ledger)
    return metrics_to_dict(m)


def main():
    _ensure_reports_dir()

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
    test_cut = int(n * 0.85)

    feat_main = feat.iloc[:test_cut].copy()
    df_hold = df_env_all.iloc[test_cut:].copy()
    feat_hold = feat.iloc[test_cut:].copy()
    X_hold = X_all[test_cut:]

    mode, model_obj, model_path = load_policy_system(n_features=X_hold.shape[1])

    regime_arr_hold = None
    if mode == "regime_bank":
        det = RegimeDetector().fit(feat_main)
        regime_arr_hold = det.predict(feat_hold)

    gate = make_default_gate()
    # IMPORTANT: weighting should not depend on hard threshold, so ensure slope_min=0
    if hasattr(gate, "cfg"):
        gate.cfg.slope_min = 0.0

    weighter = ConfidenceWeighter(ConfidenceWeighterConfig(power=2.5, min_size=0.0, deadzone=0.0))

    # Baseline: no sizing => size=1 always (use original action format)
    # We'll run baseline by calling TradingEnv directly with same model actions (no size).
    env0 = TradingEnv(
        df_hold,
        gap_close_minutes=60,
        gap_skip_minutes=180,
        spread_open_cap=200,
        force_close_on_spread=False,
    )
    idx0 = {"i": 0}

    def baseline_policy(_row):
        i = idx0["i"]
        idx0["i"] += 1
        if mode == "linear":
            return model_obj.act(X_hold[i])
        r = "mixed"
        if regime_arr_hold is not None:
            r = str(regime_arr_hold[i])
        return model_obj.act(X_hold[i], r)

    ledger0 = env0.run_backtest(baseline_policy)
    base = metrics_to_dict(compute_metrics(ledger0))

    # Weighted
    weighted = run_backtest(df_hold, feat_hold, X_hold, regime_arr_hold, mode, model_obj, gate, weighter)

    report = {
        "model_mode": mode,
        "model_path": model_path,
        "weighter_cfg": weighter.cfg.__dict__,
        "baseline": base,
        "weighted": weighted,
        "pf_change": float(weighted["profit_factor"] - base["profit_factor"]),
        "maxdd_change": float((weighted["max_drawdown"] - base["max_drawdown"]) / max(1e-9, base["max_drawdown"])),
        "trades_change": int(weighted["trades"] - base["trades"]),
    }

    out_path = "data/reports/weighted_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Model:", mode, model_path)
    print("Weighter:", weighter.cfg)
    print("Baseline:", base)
    print("Weighted:", weighted)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()