import json
import os

import numpy as np

from rabit.data.loader import MT5DataLoader
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig
from rabit.env.metrics import compute_metrics, metrics_to_dict
from rabit.env.trading_env import TradingEnv

from rabit.rl.confidence_gate import make_default_gate
from rabit.rl.policy_linear import LinearPolicy

from rabit.regime.regime_detector import RegimeDetector
from rabit.rl.regime_policy_bank import RegimePolicyBank


def _ensure_reports_dir():
    os.makedirs("data/reports", exist_ok=True)


def _robust_score(m: dict) -> float:
    pf = float(m["profit_factor"])
    dd = float(m["max_drawdown"])
    tr = float(m["trades"])
    return (pf - 1.0) - 0.25 * (dd / 300.0) - 0.05 * (tr / 250.0)


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


def run_backtest(df_env, feat_df, X: np.ndarray, regime_arr: np.ndarray | None, mode: str, model_obj, gate):
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

        # Gate: TRADE vs HOLD
        if gate is not None:
            feat_row = feat_df.iloc[i]
            _conf, allow_trade, _reason = gate.evaluate(feat_row)
            if not allow_trade:
                return (0, 0.8, 0.8, 20)

        # Real policy
        if mode == "linear":
            return model_obj.act(X[i])

        if mode == "regime_bank":
            r = "mixed"
            if regime_arr is not None:
                r = str(regime_arr[i])
            return model_obj.act(X[i], r)

        return (0, 0.8, 0.8, 20)

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

    df_main = df_env_all.iloc[:test_cut].copy()
    feat_main = feat.iloc[:test_cut].copy()

    df_hold = df_env_all.iloc[test_cut:].copy()
    feat_hold = feat.iloc[test_cut:].copy()
    X_hold = X_all[test_cut:]

    mode, model_obj, model_path = load_policy_system(n_features=X_hold.shape[1])

    regime_arr_hold = None
    if mode == "regime_bank":
        det = RegimeDetector().fit(feat_main)
        regime_arr_hold = det.predict(feat_hold)

    # ✅ Baseline: no gate
    base = run_backtest(df_hold, feat_hold, X_hold, regime_arr_hold, mode, model_obj, gate=None)

    # ✅ Gated: strict gate + slope_min calibrated from MAIN
    gate = make_default_gate()
    gate.cfg.slope_min = 0.0
    gate = gate.calibrate_threshold_from_features(feat_main, target_keep=0.70)
    gated = run_backtest(df_hold, feat_hold, X_hold, regime_arr_hold, mode, model_obj, gate=gate)

    trade_reduction = (base["trades"] - gated["trades"]) / max(1, base["trades"])
    maxdd_change = (gated["max_drawdown"] - base["max_drawdown"]) / max(1e-9, base["max_drawdown"])
    pf_change = gated["profit_factor"] - base["profit_factor"]

    criteria = {
        "trades_ok": trade_reduction >= 0.20,
        "maxdd_ok": maxdd_change <= 0.10,
        "pf_ok": pf_change >= -0.02,
        "trade_reduction": float(trade_reduction),
        "maxdd_change": float(maxdd_change),
        "pf_change": float(pf_change),
        "pass_all": False,
    }
    criteria["pass_all"] = bool(criteria["trades_ok"] and criteria["maxdd_ok"] and criteria["pf_ok"])

    report = {
        "model_mode": mode,
        "model_path": model_path,
        "feature_cols": feature_cols,
        "baseline": base,
        "gated": gated,
        "baseline_score": float(_robust_score(base)),
        "gated_score": float(_robust_score(gated)),
        "criteria": criteria,
        "gate_cfg": gate.cfg.__dict__,
    }

    out_path = "data/reports/gate_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Model:", mode, model_path)
    print("Gate slope_min:", gate.cfg.slope_min, "threshold:", gate.cfg.threshold, "spread_cap:", gate.cfg.spread_cap)
    print("Baseline:", base)
    print("Gated:", gated)
    print("Criteria:", criteria)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()