import argparse
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
from rabit.rl.meta_risk import MetaRiskConfig, MetaRiskState, load_json, save_json

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run live simulator")
    parser.add_argument("--meta_risk", type=int, default=1, help="Enable meta-risk scaling (1/0)")
    parser.add_argument(
        "--meta_risk_state",
        type=str,
        default="data/reports/meta_risk_state.json",
        help="Path to load meta-risk state",
    )
    parser.add_argument(
        "--meta_risk_out",
        type=str,
        default="data/reports/meta_risk_state.json",
        help="Path to save meta-risk state",
    )
    parser.add_argument("--meta_risk_floor", type=float, default=0.2, help="Meta-risk minimum scale")
    parser.add_argument("--meta_risk_cap", type=float, default=1.2, help="Meta-risk maximum scale")
    parser.add_argument("--meta_risk_alpha", type=float, default=0.05, help="Meta-risk EWMA alpha")
    parser.add_argument(
        "--meta_risk_min_trades",
        type=int,
        default=30,
        help="Min trades per regime before scaling",
    )
    return parser.parse_args(argv)


def load_policy_system(n_features: int):
    """
    Load one of:
    - regime_bank .npz with per-regime theta
    - walkforward/mtf .npy
    - linear .npy
    """
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

    raise FileNotFoundError("No model found in data/. (ars_best_theta*.npy or ars_best_theta_regime_bank*.npz)")


def make_obs_matrix(feat_df: pd.DataFrame, feature_cols: list[str]):
    X = feat_df[feature_cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, mu, sd


def _ledger_to_trades_df(ledger) -> pd.DataFrame:
    """
    Robust exporter across multiple Ledger variants:
    - trades() method
    - trades attribute (list/dict/df)
    - __dict__['trades']
    """
    if ledger is None:
        raise TypeError("ledger is None")

    # 1) trades() method
    if hasattr(ledger, "trades") and callable(getattr(ledger, "trades")):
        t = ledger.trades()
        if isinstance(t, pd.DataFrame):
            return t
        if isinstance(t, list):
            return pd.DataFrame(t)
        if isinstance(t, dict):
            return pd.DataFrame(t)
        return pd.DataFrame([{"_trades_raw": str(t)}])

    # 2) trades attribute (list/dict/df)
    if hasattr(ledger, "trades"):
        t = getattr(ledger, "trades")
        if isinstance(t, pd.DataFrame):
            return t
        if isinstance(t, list):
            return pd.DataFrame(t)
        if isinstance(t, dict):
            return pd.DataFrame(t)

    # 3) __dict__ fallback
    if hasattr(ledger, "__dict__") and "trades" in ledger.__dict__:
        t = ledger.__dict__["trades"]
        if isinstance(t, pd.DataFrame):
            return t
        if isinstance(t, list):
            return pd.DataFrame(t)
        if isinstance(t, dict):
            return pd.DataFrame(t)

    raise TypeError("Cannot export trades: no trades() method or trades attribute found.")


def _build_equity_from_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build equity curve from trades_df.
    Expect common columns: pnl, exit_time/exit_ts/ts, and optionally balance.
    """
    if trades_df is None or len(trades_df) == 0:
        return pd.DataFrame(columns=["ts", "equity", "pnl"])

    ts_col = None
    for c in ["exit_time", "exit_ts", "close_ts", "ts", "timestamp", "time"]:
        if c in trades_df.columns:
            ts_col = c
            break

    pnl_col = None
    for c in ["pnl", "net_pnl", "profit", "pnl_usd", "pnl_points"]:
        if c in trades_df.columns:
            pnl_col = c
            break

    if ts_col is None:
        ts = [f"row_{i}" for i in range(len(trades_df))]
    else:
        ts = trades_df[ts_col].astype(str).fillna("").tolist()

    if pnl_col is None:
        pnl = np.zeros(len(trades_df), dtype=np.float64)
    else:
        pnl = pd.to_numeric(trades_df[pnl_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    if "balance" in trades_df.columns:
        equity = pd.to_numeric(trades_df["balance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        equity = np.cumsum(pnl)

    return pd.DataFrame({"ts": ts, "equity": equity, "pnl": pnl})


def main(argv=None):
    _ensure_reports_dir()
    args = parse_args(argv)

    # ===== Config (production-ish defaults) =====
    m1_path = "data/XAUUSD_M1.csv"
    warmup_bars = 300
    power = 2.5
    deadzone = 0.0
    min_size = 0.0

    env_cfg = dict(
        gap_close_minutes=60,
        gap_skip_minutes=180,
        spread_open_cap=200,
        force_close_on_spread=False,
    )

    meta_enabled = int(args.meta_risk) != 0
    meta_state = None
    if meta_enabled:
        cfg = MetaRiskConfig(
            alpha=args.meta_risk_alpha,
            floor=args.meta_risk_floor,
            cap=args.meta_risk_cap,
            min_trades_per_regime=args.meta_risk_min_trades,
        )
        meta_state = MetaRiskState(cfg)
        loaded = load_json(args.meta_risk_state)
        if loaded is not None:
            meta_state = loaded
            meta_state.cfg = cfg

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

    # ===== Regime detector =====
    regime_arr = None
    if mode == "regime_bank":
        det = RegimeDetector().fit(feat.iloc[:warmup_bars].copy())
        regime_arr = det.predict(feat)

    # ===== Confidence weighting =====
    gate = make_default_gate()
    if hasattr(gate, "cfg"):
        gate.cfg.slope_min = 0.0

    weighter = ConfidenceWeighter(
        ConfidenceWeighterConfig(power=power, min_size=min_size, deadzone=deadzone)
    )

    # ===== Execution + session realism =====
    exec_model = ExecutionModel(
        ExecutionConfig(
            slip_base=0.0,
            slip_k_spread=0.10,
            slip_k_atr=0.02,
            slip_noise_std=0.02,
            commission_per_side=0.0,
        )
    )
    sess = SessionFilter(SessionConfig())

    env = TradingEnv(
        df_env,
        execution_model=exec_model,
        session_filter=sess,
        **env_cfg
    )

    decision_meta = {
        "regime": "unknown",
        "meta_scale": 1.0,
        "size_conf": 0.0,
        "size": 0.0,
    }
    trade_meta = []
    closed_idx = 0

    def policy_func(i: int):
        if i < warmup_bars:
            return (0, 0.8, 0.8, 20)

        # base action from model
        if mode == "linear":
            dir_, tp_mult, sl_mult, hold_max = model.act(X_all[i])
            regime = "unknown"
        else:
            regime = "unknown"
            if regime_arr is not None:
                regime = str(regime_arr[i])
            dir_, tp_mult, sl_mult, hold_max = model.act(X_all[i], regime)

        if dir_ == 0:
            return (0, 0.8, 0.8, 20)

        # confidence -> size
        feat_row = feat.iloc[i]
        confidence, _allow, _reason = gate.evaluate(feat_row)
        size_conf = float(weighter.size(float(confidence)))

        meta_scale = 1.0
        size_final = size_conf
        if meta_state is not None:
            meta_scale = float(meta_state.meta_scale(regime))
            size_final = float(np.clip(size_conf * meta_scale, 0.0, 1.0))

        decision_meta["regime"] = regime
        decision_meta["meta_scale"] = meta_scale
        decision_meta["size_conf"] = size_conf
        decision_meta["size"] = size_final

        if size_final <= 1e-6:
            return (0, 0.8, 0.8, 20)

        # TradingEnv supports size as 5th element (TASK-2A)
        return (dir_, tp_mult, sl_mult, hold_max, size_final)

    for i in range(len(env.df)):
        prev_trade_count = len(env.ledger.trades)
        action = policy_func(i)
        env.step(i, action)

        new_trades = len(env.ledger.trades) - prev_trade_count
        if new_trades > 0:
            for _ in range(new_trades):
                trade_meta.append(
                    {
                        "regime": decision_meta.get("regime", "unknown"),
                        "meta_scale": float(decision_meta.get("meta_scale", 1.0)),
                        "size_conf": float(decision_meta.get("size_conf", 0.0)),
                        "size": float(decision_meta.get("size", 0.0)),
                    }
                )

        if meta_state is not None:
            while closed_idx < len(env.ledger.trades):
                trade = env.ledger.trades[closed_idx]
                if trade.exit_time is None or trade.pnl is None:
                    break
                if closed_idx < len(trade_meta):
                    regime = str(trade_meta[closed_idx].get("regime", "unknown"))
                else:
                    regime = "unknown"
                exit_ts_str = str(trade.exit_time) if trade.exit_time is not None else ""
                meta_state.update_trade(regime, float(trade.pnl), date=exit_ts_str)
                closed_idx += 1

    ledger = env.ledger

    # ===== Metrics =====
    m = compute_metrics(ledger)
    md = metrics_to_dict(m)

    # ===== Write reports (NO duplicate writes) =====
    out_trades_csv = "data/reports/live_sim_trades.csv"
    out_equity_csv = "data/reports/live_sim_equity.csv"
    out_summary_json = "data/reports/live_sim_summary.json"

    os.makedirs(os.path.dirname(out_trades_csv), exist_ok=True)

    trades_df = _ledger_to_trades_df(ledger)
    if len(trade_meta) > 0:
        meta_df = pd.DataFrame(trade_meta)
        if len(meta_df) < len(trades_df):
            meta_df = meta_df.reindex(range(len(trades_df)))
        for col in ["regime", "meta_scale", "size_conf", "size"]:
            if col in meta_df.columns:
                trades_df[col] = meta_df[col].values

    trades_df.to_csv(out_trades_csv, index=False)

    # sanity: ensure CSV is real table, not "Ledger object ..."
    chk = pd.read_csv(out_trades_csv)
    if chk.shape[0] == 0 or chk.shape[1] <= 1:
        raise RuntimeError(
            f"Export trades failed (file looks wrong): shape={chk.shape}, cols={chk.columns.tolist()}"
        )

    equity_df = _build_equity_from_trades(trades_df)
    equity_df.to_csv(out_equity_csv, index=False)

    summary = {
        "mode": mode,
        "model_path": model_path,
        "env_cfg": env_cfg,
        "warmup_bars": warmup_bars,
        "feature_cols": feature_cols,
        "weighter": weighter.cfg.__dict__,
        "metrics": md,
    }
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if meta_state is not None:
        save_json(args.meta_risk_out, meta_state)

    print("Model:", mode, model_path)
    print("Weighter:", weighter.cfg)
    print("Metrics:", md)
    print("Saved:", out_trades_csv, out_equity_csv, out_summary_json)

    if meta_enabled:
        regimes_n = len(meta_state.regimes) if meta_state is not None else 0
        if meta_state is None:
            global_scale = "unknown"
        elif "unknown" in meta_state.regimes:
            global_scale = f"{meta_state.meta_scale('unknown'):.4f}"
        elif len(meta_state.regimes) == 1:
            only_regime = next(iter(meta_state.regimes))
            global_scale = f"{meta_state.meta_scale(only_regime):.4f}"
        else:
            global_scale = "unknown"
        print(f"[meta_risk] enabled=1 regimes={regimes_n} global_scale={global_scale}")
    else:
        print("[meta_risk] enabled=0")


if __name__ == "__main__":
    main()
