from __future__ import annotations

import os
import time
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from rabit.data.loader import MT5DataLoader
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig

from rabit.rl.confidence_gate import make_default_gate
from rabit.rl.confidence_weighting import ConfidenceWeighter, ConfidenceWeighterConfig
from rabit.rl.policy_linear import LinearPolicy
from rabit.rl.regime_policy_bank import RegimePolicyBank
from rabit.regime.regime_detector import RegimeDetector

from rabit.env.execution_model import ExecutionModel, ExecutionConfig
from rabit.env.session_filter import SessionFilter, SessionConfig


@dataclass
class ReplayConfig:
    csv_path: str
    speed_bars_per_sec: float = 50.0
    max_bars: int = 0  # 0 = all
    start_at: int = 0  # skip first N rows (after parsing)
    warmup_bars: int = 300

    atr_window: int = 240
    atr_floor: float = 0.05
    min_vol_scale: float = 0.30
    max_vol_scale: float = 1.00

    gate_slope_min: float = 0.0
    weight_power: float = 2.5

    spread_spike_cap: int = 300

    out_dir: str = "data/reports/paper_live_replay"
    out_jsonl: str = "paper_live_replay_log.jsonl"
    out_orders: str = "paper_live_replay_orders.csv"
    out_summary: str = "paper_live_replay_summary.json"

    slip_k_spread: float = 0.10
    slip_k_atr: float = 0.02
    slip_noise_std: float = 0.02
    commission_per_side: float = 0.0

    enable_london: bool = True
    enable_ny: bool = True


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_orders_csv(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=(not exists), index=False)


def load_policy_system(n_features: int):
    def pick(paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    p_bank = pick(["data/ars_best_theta_regime_bank_v2.npz", "data/ars_best_theta_regime_bank.npz"])
    if p_bank:
        z = np.load(p_bank)
        policies = {}
        for k in ["trend", "range", "highvol"]:
            if k in z:
                p = LinearPolicy(n_features=n_features)
                p.set_params_flat(z[k].astype(np.float64))
                policies[k] = p
        return "regime_bank", RegimePolicyBank(policies, fallback=None), p_bank

    p_wf = pick(["data/ars_best_theta_walkforward.npy", "data/ars_best_theta_mtf.npy"])
    if p_wf:
        theta = np.load(p_wf).astype(np.float64)
        p = LinearPolicy(n_features=n_features)
        p.set_params_flat(theta)
        return "linear", p, p_wf

    p_lin = pick(["data/ars_best_theta.npy"])
    if p_lin:
        theta = np.load(p_lin).astype(np.float64)
        p = LinearPolicy(n_features=n_features)
        p.set_params_flat(theta)
        return "linear", p, p_lin

    raise FileNotFoundError("No model found in data/ (ars_best_theta*.npy/npz).")


def make_obs_matrix(feat_df: pd.DataFrame, feature_cols: list[str]):
    X = feat_df[feature_cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", dest="csv_path", required=True, help="MT5 CSV/TSV file path (append export format)")
    ap.add_argument("--speed", dest="speed", type=float, default=50.0, help="Replay speed in bars/sec")
    ap.add_argument("--max_bars", dest="max_bars", type=int, default=0, help="Limit bars (0=all)")
    ap.add_argument("--start_at", dest="start_at", type=int, default=0, help="Skip first N parsed rows")
    args = ap.parse_args()

    cfg = ReplayConfig(
        csv_path=args.csv_path,
        speed_bars_per_sec=float(args.speed),
        max_bars=int(args.max_bars),
        start_at=int(args.start_at),
    )

    ensure_dir(cfg.out_dir)
    out_jsonl = os.path.join(cfg.out_dir, cfg.out_jsonl)
    out_orders = os.path.join(cfg.out_dir, cfg.out_orders)
    out_summary = os.path.join(cfg.out_dir, cfg.out_summary)

    loader = MT5DataLoader(expected_freq="1min", tz=None)
    fb = FeatureBuilder(FeatureConfig(dropna=False, add_atr=True))

    gate = make_default_gate()
    if hasattr(gate, "cfg"):
        gate.cfg.slope_min = cfg.gate_slope_min

    weighter = ConfidenceWeighter(ConfidenceWeighterConfig(
        power=cfg.weight_power, min_size=0.0, max_size=1.0, deadzone=0.0
    ))

    exec_model = ExecutionModel(ExecutionConfig(
        slip_base=0.0,
        slip_k_spread=cfg.slip_k_spread,
        slip_k_atr=cfg.slip_k_atr,
        slip_noise_std=cfg.slip_noise_std,
        commission_per_side=cfg.commission_per_side,
    ))

    sess = SessionFilter(SessionConfig(enable_london=cfg.enable_london, enable_ny=cfg.enable_ny))

    append_jsonl(out_jsonl, {"event": "START_REPLAY", "csv": cfg.csv_path, "ts": str(pd.Timestamp.now(tz="UTC")),
                             "speed_bars_per_sec": cfg.speed_bars_per_sec, "max_bars": cfg.max_bars, "start_at": cfg.start_at})

    # Load full file once (replay)
    df_all = loader.load_m1(cfg.csv_path)
    df_all = loader.to_numpy_ready(df_all)

    if cfg.start_at > 0:
        df_all = df_all.iloc[cfg.start_at:].copy()

    if cfg.max_bars and len(df_all) > cfg.max_bars:
        df_all = df_all.iloc[:cfg.max_bars].copy()

    if len(df_all) < cfg.warmup_bars + 10:
        raise ValueError(f"Not enough bars after slicing. bars={len(df_all)} warmup={cfg.warmup_bars}")

    # Feature cols expected
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

    # Paper position state (very light, for auditing)
    pos = 0
    entry_price = None
    entry_ts = None

    # Replay timing
    dt_sleep = 0.0
    if cfg.speed_bars_per_sec > 0:
        dt_sleep = 1.0 / cfg.speed_bars_per_sec

    # rolling window for speed: just enough for indicator stability
    keep_bars = max(cfg.warmup_bars + 2000, cfg.atr_window + 1000, 4000)

    # Build model after first feature matrix known
    model_mode = None
    model = None
    model_path = None
    detector: Optional[RegimeDetector] = None

    # Metrics accumulators
    n_bars = 0
    n_signals = 0
    n_allowed = 0
    sum_size = 0.0

    # Replay loop: feed increasing history window
    for t in range(cfg.warmup_bars, len(df_all)):
        hist = df_all.iloc[max(0, t - keep_bars): t + 1].copy()

        # features on window
        feat = fb.build(hist, prefix="")
        df_env = hist.join(feat[["atr"]], how="left").copy()

        # target_atr
        atr_series = df_env["atr"].astype(float).ffill().bfill()
        target_atr = atr_series.rolling(cfg.atr_window, min_periods=max(30, cfg.atr_window // 4)).median().shift(1)
        target_atr = target_atr.fillna(target_atr.median())
        df_env["target_atr"] = target_atr

        fcols = [c for c in feature_cols if c in feat.columns]
        if len(fcols) < 6:
            append_jsonl(out_jsonl, {"event": "FEATURE_TOO_FEW", "t": int(t), "n": len(fcols), "cols": fcols})
            if dt_sleep > 0:
                time.sleep(dt_sleep)
            continue

        X, mu, sd = make_obs_matrix(feat[fcols].copy(), fcols)

        if model is None:
            model_mode, model, model_path = load_policy_system(n_features=X.shape[1])
            append_jsonl(out_jsonl, {"event": "MODEL_LOADED", "mode": model_mode, "path": model_path, "n_features": int(X.shape[1])})

        if model_mode == "regime_bank" and detector is None:
            detector = RegimeDetector().fit(feat.iloc[: min(cfg.warmup_bars, len(feat))].copy())

        regime_arr = None
        if model_mode == "regime_bank" and detector is not None:
            try:
                regime_arr = detector.predict(feat)
            except Exception as e:
                append_jsonl(out_jsonl, {"event": "REGIME_FAIL", "t": int(t), "err": str(e)})
                regime_arr = None

        i = len(df_env) - 1
        ts = df_env.index[i]
        row = df_env.iloc[i]
        spread_pts = int(row["spread"])
        atr_i = float(row["atr"])
        tatr_i = float(row["target_atr"])
        close_price = float(row["close"])

        session_ok = sess.is_open(ts)
        spike_block = spread_pts >= cfg.spread_spike_cap

        action = {
            "dir": 0, "tp_mult": 0.0, "sl_mult": 0.0, "hold_max": 0,
            "confidence": 0.0, "base_size": 0.0, "vol_scale": 1.0, "final_size": 0.0,
            "allow": False, "reason": "hold"
        }

        if session_ok and (not spike_block):
            if model_mode == "linear":
                dir_, tp_mult, sl_mult, hold_max = model.act(X[i])
            else:
                r = str(regime_arr[i]) if regime_arr is not None else "mixed"
                dir_, tp_mult, sl_mult, hold_max = model.act(X[i], r)

            feat_row = feat.iloc[i]
            confidence, allow_trade, reason = gate.evaluate(feat_row)

            base_size = float(weighter.size(confidence))
            atr_use = max(cfg.atr_floor, float(atr_i))
            tatr_use = max(cfg.atr_floor, float(tatr_i))
            vol_scale = float(np.clip(tatr_use / atr_use, cfg.min_vol_scale, cfg.max_vol_scale))
            final_size = float(base_size * vol_scale)

            if allow_trade and dir_ != 0 and final_size > 1e-6:
                action.update({
                    "dir": int(dir_),
                    "tp_mult": float(tp_mult),
                    "sl_mult": float(sl_mult),
                    "hold_max": int(hold_max),
                    "confidence": float(confidence),
                    "base_size": float(base_size),
                    "vol_scale": float(vol_scale),
                    "final_size": float(final_size),
                    "allow": True,
                    "reason": str(reason),
                })

        fill_price = None
        spread_price = float(spread_pts) * 0.01
        if action["allow"] and action["dir"] != 0:
            fill_price = float(exec_model.market_fill(
                direction=+1 if action["dir"] == 1 else -1,
                mid_price=close_price,
                spread_price=spread_price,
                atr_price=max(cfg.atr_floor, float(atr_i)),
            ))

        order_event = None
        if pos == 0 and action["allow"] and action["dir"] != 0:
            pos = 1 if action["dir"] == 1 else -1
            entry_price = fill_price
            entry_ts = ts
            order_event = {"type": "OPEN", "pos": int(pos), "entry_price": float(entry_price), "size": float(action["final_size"])}
        elif pos != 0:
            order_event = {"type": "HOLD", "pos": int(pos), "entry_price": float(entry_price) if entry_price is not None else None}

        log = {
            "event": "BAR",
            "t": int(t),
            "ts": str(ts),
            "close": float(close_price),
            "spread": int(spread_pts),
            "atr": float(atr_i),
            "target_atr": float(tatr_i),
            "session_ok": bool(session_ok),
            "spike_block": bool(spike_block),
            "regime": str(regime_arr[i]) if regime_arr is not None else "na",
            "action": action,
            "paper_fill": float(fill_price) if fill_price is not None else None,
            "paper_pos": int(pos),
            "paper_entry_ts": str(entry_ts) if entry_ts is not None else None,
            "paper_entry_price": float(entry_price) if entry_price is not None else None,
            "model": {"mode": model_mode, "path": model_path, "n_features": int(X.shape[1])},
            "window": {"bars": int(len(hist)), "keep_bars": int(keep_bars)},
        }
        append_jsonl(out_jsonl, log)

        if order_event is not None:
            append_orders_csv(out_orders, {
                "ts": str(ts),
                "event": order_event["type"],
                "pos": order_event.get("pos"),
                "price": order_event.get("entry_price"),
                "size": order_event.get("size"),
            })

        # stats
        n_bars += 1
        if action["dir"] != 0:
            n_signals += 1
        if action["allow"]:
            n_allowed += 1
            sum_size += float(action["final_size"])

        if dt_sleep > 0:
            time.sleep(dt_sleep)

    summary = {
        "bars_processed": int(n_bars),
        "signals": int(n_signals),
        "allowed": int(n_allowed),
        "allow_rate": float(n_allowed / max(1, n_signals)),
        "avg_final_size_when_allowed": float(sum_size / max(1, n_allowed)),
        "csv": cfg.csv_path,
        "speed_bars_per_sec": float(cfg.speed_bars_per_sec),
        "max_bars": int(cfg.max_bars),
        "start_at": int(cfg.start_at),
        "model_mode": model_mode,
        "model_path": model_path,
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_jsonl, out_orders, out_summary)
    print("Summary:", summary)


if __name__ == "__main__":
    main()