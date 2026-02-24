from __future__ import annotations

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

# Execution realism (TASK-2E)
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


def make_obs_matrix(feat_df: pd.DataFrame, feature_cols: list[str]):
    X = feat_df[feature_cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, mu, sd


def ledger_to_df(ledger) -> pd.DataFrame:
    """
    Try to normalize ledger into DataFrame for reporting.
    """
    if ledger is None:
        return pd.DataFrame()
    if isinstance(ledger, pd.DataFrame):
        return ledger
    # many versions store trades in ledger.trades
    if hasattr(ledger, "trades"):
        t = getattr(ledger, "trades")
        if isinstance(t, pd.DataFrame):
            return t
        try:
            return pd.DataFrame(t)
        except Exception:
            pass
    # fallback: empty
    return pd.DataFrame()


def get_balance(ledger) -> float:
    if ledger is None:
        return 0.0
    if hasattr(ledger, "balance"):
        try:
            return float(getattr(ledger, "balance"))
        except Exception:
            pass
    # fallback from trades
    df = ledger_to_df(ledger)
    for c in ["balance", "equity"]:
        if c in df.columns and len(df) > 0:
            try:
                return float(df[c].iloc[-1])
            except Exception:
                pass
    for c in ["pnl", "net_pnl", "profit"]:
        if c in df.columns and len(df) > 0:
            try:
                return float(df[c].astype(float).sum())
            except Exception:
                pass
    return 0.0


def count_closed_trades(ledger) -> int:
    df = ledger_to_df(ledger)
    if len(df) == 0:
        return 0
    # if each row is a trade, rows count is trades
    return int(len(df))


def last_trade_pnl(ledger) -> float | None:
    df = ledger_to_df(ledger)
    if len(df) == 0:
        return None
    for c in ["pnl", "net_pnl", "profit"]:
        if c in df.columns:
            try:
                return float(df[c].iloc[-1])
            except Exception:
                return None
    return None


def main():
    _ensure_reports_dir()

    # ======================
    # CONFIG (production-ish)
    # ======================
    m1_path = "data/XAUUSD_M1.csv"
    warmup_bars = 300
    power = 2.5

    # Fail-safe
    daily_dd_limit = 200.0        # if intraday DD exceeds this (balance units) => stop entry rest of day
    # Volatility target sizing (TASK-2G)
    atr_window = 240          # rolling window bars
    min_vol_scale = 0.30
    max_vol_scale = 1.00
    atr_floor = 0.05          # avoid division blow-up
    max_loss_streak = 4           # consecutive losing trades => stop entry rest of day
    spread_hard_cap = 200         # keep your old spread_open_cap
    spread_spike_cap = 300        # additional fail-safe (if spread >= this => no entry)

    # Env config
    env_cfg = dict(
        gap_close_minutes=60,
        gap_skip_minutes=180,
        spread_open_cap=spread_hard_cap,
        force_close_on_spread=False,
    )

    # Execution realism config (calibration can be done later)
    exec_model = ExecutionModel(ExecutionConfig(
        slip_base=0.0,
        slip_k_spread=0.10,
        slip_k_atr=0.02,
        slip_noise_std=0.02,
        commission_per_side=0.0,
    ))

    sess = SessionFilter(SessionConfig(
        enable_london=True,
        enable_ny=True,
    ))

    # ======================
    # LOAD + FEATURES
    # ======================
    loader = MT5DataLoader()
    df = loader.load_m1(m1_path)
    df = loader.to_numpy_ready(df)

    fb = FeatureBuilder(FeatureConfig(dropna=False, add_atr=True))
    feat = fb.build(df, prefix="")

    # env df needs atr
    df_env = df.join(feat[["atr"]], how="left").copy()

    # rolling median ATR as target (shifted to avoid lookahead)
    atr_series = (
    df_env["atr"]
    .astype(float)
    .ffill()
    .bfill()
)
    target_atr = atr_series.rolling(atr_window, min_periods=max(30, atr_window // 4)).median().shift(1)
    target_atr = target_atr.fillna(target_atr.median())
    df_env["target_atr"] = target_atr

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

    # ======================
    # MODEL + REGIME
    # ======================
    mode, model, model_path = load_policy_system(n_features=X_all.shape[1])

    regime_arr = None
    if mode == "regime_bank":
        det = RegimeDetector().fit(feat.iloc[:warmup_bars].copy())
        regime_arr = det.predict(feat)

    gate = make_default_gate()
    if hasattr(gate, "cfg"):
        gate.cfg.slope_min = 0.0

    weighter = ConfidenceWeighter(ConfidenceWeighterConfig(power=power, min_size=0.0, max_size=1.0, deadzone=0.0))

    # ======================
    # DAILY LOOP
    # ======================
    # group by date (based on df index)
    dates = pd.to_datetime(df_env.index).date
    unique_days = pd.Series(dates).unique().tolist()

    daily_rows = []
    equity_rows = []
    regime_stats = {}

    global_start_balance = 0.0
    global_balance = 0.0

    for day_i, day in enumerate(unique_days):
        # indices for this day
        mask = (dates == day)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue

        day_start_i = int(idxs[0])
        day_end_i = int(idxs[-1])

        # create env over full df (streaming), but we will step only this day slice.
        env = TradingEnv(
            df_env,
            execution_model=exec_model,
            session_filter=sess,
            **env_cfg
        )

        # fast-forward warmup + previous days WITHOUT trading (to set last_ts continuity)
        # We just update env.last_ts by stepping HOLD so gap logic behaves consistently.
        for i in range(0, min(day_start_i, warmup_bars)):
            env.step(i, (0, 0.8, 0.8, 20))

        # daily state
        dd_stop = False
        loss_streak = 0

        day_equity_start = global_balance
        day_peak = global_balance
        day_min = global_balance

        trades_before = count_closed_trades(env.ledger)

        # run day
        for i in range(day_start_i, day_end_i + 1):
            ts = df_env.index[i]
            row = df_env.iloc[i]

            # bar-level equity snapshot (before action)
            bal_now = get_balance(env.ledger) + global_balance  # ledger local + global offset
            day_peak = max(day_peak, bal_now)
            day_min = min(day_min, bal_now)
            dd_now = day_peak - bal_now

            # fail-safe: daily DD
            if dd_now >= daily_dd_limit:
                dd_stop = True

            spread_pts = int(row["spread"])
            if spread_pts >= spread_spike_cap:
                # spike => block entry (do not set dd_stop)
                spike_block = True
            else:
                spike_block = False

            # make action (default HOLD)
            action = (0, 0.8, 0.8, 20)

            # warmup
            if i >= warmup_bars and (not dd_stop) and (loss_streak < max_loss_streak) and (not spike_block):
                # model decision
                if mode == "linear":
                    dir_, tp_mult, sl_mult, hold_max = model.act(X_all[i])
                else:
                    r = "mixed"
                    if regime_arr is not None:
                        r = str(regime_arr[i])
                    dir_, tp_mult, sl_mult, hold_max = model.act(X_all[i], r)

                if dir_ != 0:
                    # confidence -> size
                    feat_row = feat.iloc[i]
                    confidence, _allow, _reason = gate.evaluate(feat_row)
                    base_size = float(weighter.size(confidence))

                    # volatility scaling
                    atr_i = float(df_env.iloc[i]["atr"])
                    tatr_i = float(df_env.iloc[i].get("target_atr", atr_i))
                    atr_i = max(atr_floor, atr_i)
                    tatr_i = max(atr_floor, tatr_i)

                    vol_scale = tatr_i / atr_i
                    vol_scale = float(np.clip(vol_scale, min_vol_scale, max_vol_scale))

                    final_size = base_size * vol_scale

                    if final_size > 1e-6:
                        action = (dir_, float(tp_mult), float(sl_mult), int(hold_max), float(final_size))

            # step env
            env.step(i, action)

            # after step, check if a trade closed and update loss streak
            trades_after = count_closed_trades(env.ledger)
            if trades_after > trades_before:
                pnl = last_trade_pnl(env.ledger)
                if pnl is not None:
                    if pnl < 0:
                        loss_streak += 1
                    else:
                        loss_streak = 0
                trades_before = trades_after

            # equity snapshot after step
            bal_after = get_balance(env.ledger) + global_balance
            equity_rows.append({
                "day": str(day),
                "ts": str(ts),
                "equity": float(bal_after),
                "dd_stop": bool(dd_stop),
                "loss_streak": int(loss_streak),
                "spread": int(spread_pts),
                "regime": str(regime_arr[i]) if regime_arr is not None else "na",
                "vol_scale": float(vol_scale) if "vol_scale" in locals() else 1.0,
                "base_size": float(base_size) if "base_size" in locals() else 0.0,
            })

        # day summary
        bal_end = get_balance(env.ledger) + global_balance
        day_pnl = bal_end - day_equity_start

        # store trades df with day tag (optional)
        tdf = ledger_to_df(env.ledger)
        if len(tdf) > 0 and "day" not in tdf.columns:
            tdf = tdf.copy()
            tdf["day"] = str(day)

        # regime stats by day (counts only)
        if regime_arr is not None:
            regs = pd.Series(regime_arr[day_start_i:day_end_i+1]).astype(str)
            counts = regs.value_counts().to_dict()
        else:
            counts = {"na": int(day_end_i - day_start_i + 1)}

        regime_stats[str(day)] = counts

        daily_rows.append({
            "day": str(day),
            "bars": int(day_end_i - day_start_i + 1),
            "start_equity": float(day_equity_start),
            "end_equity": float(bal_end),
            "day_pnl": float(day_pnl),
            "day_peak": float(day_peak),
            "day_min": float(day_min),
            "intraday_dd": float(day_peak - day_min),
            "dd_stop": bool(dd_stop),
            "end_loss_streak": int(loss_streak),
        })

        # roll global balance (carry PnL forward)
        global_balance = bal_end

    # ======================
    # FINAL REPORTS
    # ======================
    daily_df = pd.DataFrame(daily_rows)
    equity_df = pd.DataFrame(equity_rows)

    out_daily_equity = "data/reports/live_daily_equity.csv"
    out_daily_summary = "data/reports/live_daily_summary.json"
    out_regime = "data/reports/live_daily_regime.json"

    equity_df.to_csv(out_daily_equity, index=False)

    # aggregate summary
    total_pnl = float(daily_df["day_pnl"].sum()) if len(daily_df) else 0.0
    best_day = None
    worst_day = None
    if len(daily_df):
        best_day = daily_df.sort_values("day_pnl", ascending=False).iloc[0].to_dict()
        worst_day = daily_df.sort_values("day_pnl", ascending=True).iloc[0].to_dict()

    summary = {
        "model_path": model_path,
        "mode": mode,
        "power": power,
        "execution": exec_model.cfg.__dict__,
        "session": sess.cfg.__dict__,
        "fail_safe": {
            "daily_dd_limit": daily_dd_limit,
            "max_loss_streak": max_loss_streak,
            "spread_spike_cap": spread_spike_cap,
            "spread_open_cap": spread_hard_cap,
        },
        "days": int(len(daily_df)),
        "total_pnl": total_pnl,
        "best_day": best_day,
        "worst_day": worst_day,
        "daily_table": daily_df.to_dict(orient="records"),
        "vol_target_sizing": {
            "atr_window": atr_window,
            "min_vol_scale": min_vol_scale,
            "max_vol_scale": max_vol_scale,
            "atr_floor": atr_floor,
        },
    }

    with open(out_daily_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_regime, "w", encoding="utf-8") as f:
        json.dump(regime_stats, f, indent=2)

    print("Saved:", out_daily_equity, out_daily_summary, out_regime)
    print("Days:", len(daily_df), "Total PnL:", total_pnl)


if __name__ == "__main__":
    main()