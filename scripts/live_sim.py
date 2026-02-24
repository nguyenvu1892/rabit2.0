import os
import json
import numpy as np
import pandas as pd

from rabit.data.loader import MT5DataLoader
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig
from rabit.env.trading_env import TradingEnv
from rabit.env.metrics import compute_metrics, metrics_to_dict

from rabit.rl.confidence_gate import make_default_gate
from rabit.rl.confidence_weighting import ConfidenceWeighter, ConfidenceWeighterConfig
from rabit.rl.policy_linear import LinearPolicy

from rabit.regime.regime_detector import RegimeDetector
from rabit.rl.regime_policy_bank import RegimePolicyBank

from rabit.env.execution_model import ExecutionModel, ExecutionConfig
from rabit.env.session_filter import SessionFilter, SessionConfig

def _ensure_reports_dir():
    os.makedirs("data/reports", exist_ok=True)


def _pick_existing(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


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


def make_obs_matrix(feat_df, feature_cols: list[str]):
    X = feat_df[feature_cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, mu, sd


def build_equity_curve(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make per-trade equity curve (not per-bar) from ledger.
    Requires ledger to have at least: close_ts/ts, pnl, balance.
    We'll be defensive and infer from available columns.
    """
    if ledger_df is None or len(ledger_df) == 0:
        return pd.DataFrame(columns=["ts", "equity", "pnl"])

    ts_col = None
    for c in ["close_ts", "exit_ts", "ts", "timestamp"]:
        if c in ledger_df.columns:
            ts_col = c
            break

    if ts_col is None:
        # fallback: use index
        out = pd.DataFrame({"ts": ledger_df.index.astype(str)})
    else:
        out = pd.DataFrame({"ts": ledger_df[ts_col].astype(str)})

    pnl_col = None
    for c in ["pnl", "net_pnl", "profit"]:
        if c in ledger_df.columns:
            pnl_col = c
            break

    if pnl_col is None:
        out["pnl"] = 0.0
        out["equity"] = 0.0
        return out

    out["pnl"] = ledger_df[pnl_col].astype(float).to_numpy()

    if "balance" in ledger_df.columns:
        out["equity"] = ledger_df["balance"].astype(float).to_numpy()
    else:
        out["equity"] = np.cumsum(out["pnl"].to_numpy())

    return out


def main():
    _ensure_reports_dir()

    # ===== Config (production-ish defaults) =====
    m1_path = "data/XAUUSD_M1.csv"
    warmup_bars = 300          # skip early bars for indicator stability
    power = 2.5                # chosen from tuning
    deadzone = 0.0             # can tune later (0.05..0.15)
    min_size = 0.0             # no forced min allocation

    # TradingEnv risk / execution params (keep consistent)
    env_cfg = dict(
        gap_close_minutes=60,
        gap_skip_minutes=180,
        spread_open_cap=200,
        force_close_on_spread=False,
    )

    # ===== Load + features =====
    loader = MT5DataLoader()
    df = loader.load_m1(m1_path)
    df = loader.to_numpy_ready(df)

    fb = FeatureBuilder(FeatureConfig(dropna=False, add_atr=True))
    feat = fb.build(df, prefix="")

    # env dataframe needs 'atr' for TP/SL distances
    df_env = df.join(feat[["atr"]], how="left")

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

    X_all, mu, sd = make_obs_matrix(feat[feature_cols].copy(), feature_cols)

    # ===== Load policy =====
    mode, model, model_path = load_policy_system(n_features=X_all.shape[1])

    # ===== Regime detector (fit on past only) =====
    # For offline live-sim, we fit on warmup window only (acts like having historical context).
    regime_arr = None
    if mode == "regime_bank":
        det = RegimeDetector().fit(feat.iloc[:warmup_bars].copy())
        regime_arr = det.predict(feat)

    # ===== Confidence weighting =====
    gate = make_default_gate()
    if hasattr(gate, "cfg"):
        gate.cfg.slope_min = 0.0  # ensure no hidden hard floor

    weighter = ConfidenceWeighter(
        ConfidenceWeighterConfig(power=power, min_size=min_size, deadzone=deadzone)
    )

    # ===== Run TradingEnv as a streaming sim =====
    exec_model = ExecutionModel(ExecutionConfig(
    slip_base=0.0,
    slip_k_spread=0.10,
    slip_k_atr=0.02,
    slip_noise_std=0.02,
    commission_per_side=0.0,
    ))

    sess = SessionFilter(SessionConfig())

    env = TradingEnv(
        df_env,
        execution_model=exec_model,
        session_filter=sess,
        **env_cfg
    )

    idx = {"i": 0}

    def policy_func(_row):
        i = idx["i"]
        idx["i"] += 1

        # warmup => HOLD
        if i < warmup_bars:
            return (0, 0.8, 0.8, 20)

        # base action from model
        if mode == "linear":
            dir_, tp_mult, sl_mult, hold_max = model.act(X_all[i])
        else:
            r = "mixed"
            if regime_arr is not None:
                r = str(regime_arr[i])
            dir_, tp_mult, sl_mult, hold_max = model.act(X_all[i], r)

        if dir_ == 0:
            return (0, 0.8, 0.8, 20)

        # confidence -> size
        feat_row = feat.iloc[i]
        confidence, _allow, _reason = gate.evaluate(feat_row)
        size = weighter.size(confidence)

        if size <= 1e-6:
            return (0, 0.8, 0.8, 20)

        return (dir_, tp_mult, sl_mult, hold_max, size)

    ledger = env.run_backtest(policy_func)

    # ===== Reports =====
    m = compute_metrics(ledger)
    md = metrics_to_dict(m)

    out_trades = "data/reports/live_sim_trades.csv"
    out_equity = "data/reports/live_sim_equity.csv"
    out_summary = "data/reports/live_sim_summary.json"

    # ledger is likely a pandas df already; if not, try to convert
    if hasattr(ledger, "to_csv"):
        ledger.to_csv(out_trades, index=False)
        eq = build_equity_curve(ledger)
        eq.to_csv(out_equity, index=False)
    else:
        # fallback: store raw repr
        with open(out_trades, "w", encoding="utf-8") as f:
            f.write(str(ledger))
        eq = pd.DataFrame(columns=["ts", "equity", "pnl"])
        eq.to_csv(out_equity, index=False)

    summary = {
        "mode": mode,
        "model_path": model_path,
        "env_cfg": env_cfg,
        "warmup_bars": warmup_bars,
        "feature_cols": feature_cols,
        "weighter": weighter.cfg.__dict__,
        "metrics": md,
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Model:", mode, model_path)
    print("Weighter:", weighter.cfg)
    print("Metrics:", md)
    print("Saved:", out_trades, out_equity, out_summary)


if __name__ == "__main__":
    main()