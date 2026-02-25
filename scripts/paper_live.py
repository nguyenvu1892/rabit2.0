from __future__ import annotations

import os
import time
import json
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
from rabit.rl.meta_risk import MetaRiskConfig, MetaRiskState, load_json, save_json
from rabit.regime.regime_detector import RegimeDetector

from rabit.env.execution_model import ExecutionModel, ExecutionConfig
from rabit.env.session_filter import SessionFilter, SessionConfig


# -------------------------
# Config
# -------------------------

@dataclass
class PaperLiveConfig:
    live_csv: str = "data/live/XAUUSD_M1_live.csv"
    poll_seconds: float = 1.0

    # Speed controls
    keep_bars_extra: int = 2000  # keep warmup + this many bars
    keep_bars_min: int = 4000    # minimum window regardless

    warmup_bars: int = 300
    atr_window: int = 240
    atr_floor: float = 0.05
    min_vol_scale: float = 0.30
    max_vol_scale: float = 1.00

    # Gate / Weighting
    gate_slope_min: float = 0.0
    weight_power: float = 2.5

    # Session + spread safety (entry block only)
    spread_spike_cap: int = 300

    # Outputs
    out_dir: str = "data/reports/paper_live"
    out_jsonl: str = "paper_live_log.jsonl"
    out_state: str = "paper_live_state.json"
    out_orders: str = "paper_live_orders.csv"
    meta_risk_enabled: bool = True
    meta_risk_state: str = "meta_risk_state.json"
    meta_risk_save_every: int = 50

    # Execution realism (for paper fill estimate)
    slip_k_spread: float = 0.10
    slip_k_atr: float = 0.02
    slip_noise_std: float = 0.02
    commission_per_side: float = 0.0

    # Session (assuming timestamps are in the same timezone as data)
    enable_london: bool = True
    enable_ny: bool = True


# -------------------------
# Utilities
# -------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


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


def tail_read_csv_incremental(path: str, last_pos: int) -> tuple[pd.DataFrame, int]:
    """
    Incremental tail read.
    Assumes file is append-only CSV with header present at top.
    Returns (new_rows_df, new_pos).
    """
    if not os.path.exists(path):
        return pd.DataFrame(), last_pos

    size = os.path.getsize(path)
    if size <= last_pos:
        return pd.DataFrame(), last_pos

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(last_pos)
        chunk = f.read()
        new_pos = f.tell()

    chunk = chunk.strip()
    if not chunk:
        return pd.DataFrame(), new_pos

    # If first time, just read whole file via pandas (simplest, robust).
    if last_pos == 0:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            try:
                df = pd.read_csv(path, sep=",")
            except Exception:
                df = pd.read_csv(path, sep=";")
        return df, new_pos

    # parse chunk lines; some lines may be partial at append time → keep only complete lines
    lines = chunk.splitlines()
    if len(lines) <= 0:
        return pd.DataFrame(), new_pos

    # choose separator based on first line
    sample = lines[0]
    if "\t" in sample:
        sep = "\t"
    elif ";" in sample:
        sep = ";"
    else:
        sep = ","

    def is_complete(line: str) -> bool:
        # require at least 6 separators (DATE,TIME,OPEN,HIGH,LOW,CLOSE,...)
        return line.count(sep) >= 5

    lines = [ln for ln in lines if is_complete(ln)]
    if not lines:
        return pd.DataFrame(), new_pos

    # Need columns: load header from file start quickly
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
    cols = header.split(sep)

    from io import StringIO
    buf = StringIO("\n".join(lines))
    df = pd.read_csv(buf, sep=sep, header=None)
    if df.shape[1] == len(cols):
        df.columns = cols
    return df, new_pos


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_orders_csv(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=(not exists), index=False)


def normalize_chunk_to_ohlc(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw MT5 export chunk to standardized OHLCV DataFrame with datetime index.
    Expected columns variants:
      - DATE/TIME/OPEN/HIGH/LOW/CLOSE/TICKVOL/VOL/SPREAD
      - or <DATE>/<TIME>/<OPEN>... (some exports)
    """
    if len(df_raw) == 0:
        return pd.DataFrame()

    cols = [c.strip() for c in df_raw.columns]
    df_raw.columns = cols

    # map column names (strip <>)
    def clean_col(c: str) -> str:
        c2 = c.strip()
        if c2.startswith("<") and c2.endswith(">"):
            c2 = c2[1:-1]
        return c2

    df = df_raw.copy()
    df.columns = [clean_col(c).upper() for c in df.columns]

    needed = {"DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    ts = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"].astype(str), errors="coerce")
    df.index = ts
    df = df.dropna(subset=[df.index.name] if df.index.name else [])  # safe

    # optional
    if "TICKVOL" not in df.columns:
        df["TICKVOL"] = 0
    if "VOL" not in df.columns:
        df["VOL"] = 0
    if "SPREAD" not in df.columns:
        df["SPREAD"] = 0

    out = pd.DataFrame(index=df.index)
    out["open"] = pd.to_numeric(df["OPEN"], errors="coerce")
    out["high"] = pd.to_numeric(df["HIGH"], errors="coerce")
    out["low"] = pd.to_numeric(df["LOW"], errors="coerce")
    out["close"] = pd.to_numeric(df["CLOSE"], errors="coerce")
    out["tickvol"] = pd.to_numeric(df["TICKVOL"], errors="coerce").fillna(0).astype(int)
    out["vol"] = pd.to_numeric(df["VOL"], errors="coerce").fillna(0).astype(int)
    out["spread"] = pd.to_numeric(df["SPREAD"], errors="coerce").fillna(0).astype(int)

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


# -------------------------
# Main loop
# -------------------------

def main():
    cfg = PaperLiveConfig()
    ensure_dir(cfg.out_dir)

    out_jsonl = os.path.join(cfg.out_dir, cfg.out_jsonl)
    out_state = os.path.join(cfg.out_dir, cfg.out_state)
    out_orders = os.path.join(cfg.out_dir, cfg.out_orders)
    out_meta_state = os.path.join(cfg.out_dir, cfg.meta_risk_state)

    loader = MT5DataLoader(expected_freq="1min", tz=None)
    fb = FeatureBuilder(FeatureConfig(dropna=False, add_atr=True))

    # Gate + Weighter
    gate = make_default_gate()
    if hasattr(gate, "cfg"):
        gate.cfg.slope_min = cfg.gate_slope_min

    weighter = ConfidenceWeighter(ConfidenceWeighterConfig(
        power=cfg.weight_power, min_size=0.0, max_size=1.0, deadzone=0.0
    ))

    # Execution realism
    exec_model = ExecutionModel(ExecutionConfig(
        slip_base=0.0,
        slip_k_spread=cfg.slip_k_spread,
        slip_k_atr=cfg.slip_k_atr,
        slip_noise_std=cfg.slip_noise_std,
        commission_per_side=cfg.commission_per_side,
    ))

    # Session filter
    sess = SessionFilter(SessionConfig(enable_london=cfg.enable_london, enable_ny=cfg.enable_ny))

    # State
    last_pos = 0
    df_all = pd.DataFrame()
    detector: Optional[RegimeDetector] = None
    regime_arr: Optional[np.ndarray] = None

    # keep last processed timestamp to avoid rebuilding features when no new bar
    last_ts_seen: Optional[pd.Timestamp] = None

    # Features list must match training expectations
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

    # load policy after we have feature dim
    model_mode = None
    model = None
    model_path = None

    # Paper position state (for logging only; no execution)
    pos = 0  # -1/0/1
    entry_price = None
    entry_ts = None
    entry_size = None
    entry_regime = None
    entry_hold_max = None
    bars_in_pos = 0
    paper_equity = 0.0

    meta_state = None
    if cfg.meta_risk_enabled:
        meta_cfg = MetaRiskConfig()
        meta_state = MetaRiskState(meta_cfg)
        loaded = load_json(out_meta_state)
        if loaded is not None:
            meta_state = loaded
            meta_state.cfg = meta_cfg
    meta_save_counter = 0

    append_jsonl(out_jsonl, {"event": "START", "live_csv": cfg.live_csv, "ts": str(pd.Timestamp.now(tz="UTC"))})

    while True:
        new_df_raw, last_pos = tail_read_csv_incremental(cfg.live_csv, last_pos)
        if len(new_df_raw) == 0:
            time.sleep(cfg.poll_seconds)
            continue

        # try normalize incremental chunk
        df_chunk = normalize_chunk_to_ohlc(new_df_raw)

        if len(df_chunk) == 0:
            # fallback: load full file if incremental chunk parsing failed
            try:
                df_all = loader.load_m1(cfg.live_csv)
                df_all = loader.to_numpy_ready(df_all)
            except Exception as e:
                append_jsonl(out_jsonl, {"event": "READ_FAIL", "err": str(e)})
                time.sleep(cfg.poll_seconds)
                continue
        else:
            if len(df_all) == 0:
                df_all = df_chunk.copy()
            else:
                df_all = pd.concat([df_all, df_chunk], axis=0)
                df_all = df_all[~df_all.index.duplicated(keep="last")]
                df_all = df_all.sort_index()

        # SPEED (Level 1): keep only last N bars (sliding window)
        keep_bars = max(cfg.keep_bars_min, cfg.warmup_bars + cfg.keep_bars_extra, cfg.atr_window + 1000)
        if len(df_all) > keep_bars:
            df_all = df_all.iloc[-keep_bars:].copy()

        # need warmup
        if len(df_all) < max(cfg.warmup_bars, 50):
            append_jsonl(out_jsonl, {"event": "WARMUP_WAIT", "bars": int(len(df_all))})
            time.sleep(cfg.poll_seconds)
            continue

        # SPEED (Level 2): only rebuild if there is a truly new bar timestamp
        ts_last = pd.Timestamp(df_all.index[-1])
        if last_ts_seen is not None and ts_last == last_ts_seen:
            time.sleep(cfg.poll_seconds)
            continue
        last_ts_seen = ts_last

        # build features on window (MVP). Later: incremental indicator state.
        feat = fb.build(df_all, prefix="")
        df_env = df_all.join(feat[["atr"]], how="left").copy()

        # compute target_atr
        atr_series = df_env["atr"].astype(float).ffill().bfill()
        target_atr = atr_series.rolling(cfg.atr_window, min_periods=max(30, cfg.atr_window // 4)).median().shift(1)
        target_atr = target_atr.fillna(target_atr.median())
        df_env["target_atr"] = target_atr

        # select feature cols present
        fcols = [c for c in feature_cols if c in feat.columns]
        if len(fcols) < 6:
            append_jsonl(out_jsonl, {"event": "FEATURE_TOO_FEW", "n": len(fcols), "cols": fcols})
            time.sleep(cfg.poll_seconds)
            continue

        X_all, mu, sd = make_obs_matrix(feat[fcols].copy(), fcols)

        # load model if first time
        if model is None:
            model_mode, model, model_path = load_policy_system(n_features=X_all.shape[1])
            append_jsonl(out_jsonl, {
                "event": "MODEL_LOADED",
                "mode": model_mode,
                "path": model_path,
                "n_features": int(X_all.shape[1]),
                "ts": str(ts_last),
            })

        # regime detector init
        if model_mode == "regime_bank" and detector is None:
            detector = RegimeDetector().fit(feat.iloc[:cfg.warmup_bars].copy())

        if model_mode == "regime_bank" and detector is not None:
            try:
                regime_arr = detector.predict(feat)
            except Exception as e:
                append_jsonl(out_jsonl, {"event": "REGIME_FAIL", "err": str(e)})
                regime_arr = None

        # operate on latest bar only
        i = len(df_env) - 1
        ts = df_env.index[i]
        row = df_env.iloc[i]
        spread_pts = int(row["spread"])
        atr_i = float(row["atr"])
        tatr_i = float(row["target_atr"])
        close_price = float(row["close"])

        # session gate + spread spike
        session_ok = sess.is_open(ts)
        spike_block = spread_pts >= cfg.spread_spike_cap

        # default hold
        action = {
            "dir": 0, "tp_mult": 0.0, "sl_mult": 0.0, "hold_max": 0,
            "confidence": 0.0, "base_size": 0.0, "vol_scale": 1.0, "final_size": 0.0,
            "size_conf": 0.0, "meta_scale": 1.0, "size_pre_guard": 0.0, "size": 0.0,
            "guard_reason": "ok",
            "allow": False, "reason": "hold"
        }

        if session_ok and (not spike_block):
            if model_mode == "linear":
                dir_, tp_mult, sl_mult, hold_max = model.act(X_all[i])
                regime = "na"
            else:
                regime = str(regime_arr[i]) if regime_arr is not None else "mixed"
                dir_, tp_mult, sl_mult, hold_max = model.act(X_all[i], regime)

            feat_row = feat.iloc[i]
            confidence, allow_trade, reason = gate.evaluate(feat_row)

            size_conf = float(weighter.size(confidence))
            base_size = size_conf
            meta_scale = 1.0
            if meta_state is not None:
                meta_scale = float(meta_state.meta_scale(regime))
            size_pre_guard = float(size_conf * meta_scale)
            atr_use = max(cfg.atr_floor, float(atr_i))
            tatr_use = max(cfg.atr_floor, float(tatr_i))
            vol_scale = float(np.clip(tatr_use / atr_use, cfg.min_vol_scale, cfg.max_vol_scale))
            size_pre_guard_vol = float(size_pre_guard * vol_scale)
            final_size = float(size_pre_guard_vol)
            guard_reason = "ok"
            if meta_state is not None:
                final_size = meta_state.apply_guardrails(regime, size_pre_guard_vol)
                guard_reason = meta_state.get_guard_reason(regime)
            final_size = float(np.clip(final_size, 0.0, 1.0))

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
                    "size_conf": float(size_conf),
                    "meta_scale": float(meta_scale),
                    "size_pre_guard": float(size_pre_guard),
                    "size": float(final_size),
                    "guard_reason": str(guard_reason),
                    "allow": True,
                    "reason": str(reason),
                })

        # paper fill estimate (not a real order)
        spread_price = float(spread_pts) * 0.01  # XAU point_value assumption; align later
        fill_price = None
        if action["allow"] and action["dir"] != 0:
            fill_price = float(exec_model.market_fill(
                direction=+1 if action["dir"] == 1 else -1,
                mid_price=close_price,
                spread_price=spread_price,
                atr_price=max(cfg.atr_floor, float(atr_i)),
            ))

        # paper position handling (MVP): log “signal”
        order_event = None
        if pos == 0 and action["allow"] and action["dir"] != 0:
            pos = 1 if action["dir"] == 1 else -1
            entry_price = fill_price
            entry_ts = ts
            entry_size = float(action["final_size"])
            entry_regime = str(regime_arr[i]) if regime_arr is not None else "na"
            entry_hold_max = int(action["hold_max"]) if action["hold_max"] else None
            bars_in_pos = 0
            order_event = {"type": "OPEN", "pos": int(pos), "entry_price": float(entry_price), "size": float(entry_size)}
        elif pos != 0:
            order_event = {"type": "HOLD", "pos": int(pos), "entry_price": float(entry_price) if entry_price is not None else None}

        if pos != 0:
            bars_in_pos += 1
            if entry_hold_max is not None and entry_hold_max > 0 and bars_in_pos >= entry_hold_max:
                pnl = 0.0
                if entry_price is not None and entry_size is not None:
                    pnl = (float(close_price) - float(entry_price)) * float(pos) * float(entry_size)
                paper_equity += float(pnl)
                if meta_state is not None:
                    regime_close = entry_regime or "na"
                    meta_state.update_trade(regime_close, float(pnl), date=str(ts))
                    meta_state.update_daily_equity(float(paper_equity), str(ts))
                order_event = {"type": "CLOSE", "pos": 0, "entry_price": float(close_price)}
                pos = 0
                entry_price = None
                entry_ts = None
                entry_size = None
                entry_regime = None
                entry_hold_max = None
                bars_in_pos = 0

        log = {
            "event": "BAR",
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
            "model": {"mode": model_mode, "path": model_path, "n_features": int(X_all.shape[1])},
            "window": {"bars": int(len(df_all)), "keep_bars": int(keep_bars)},
        }
        append_jsonl(out_jsonl, log)

        if order_event is not None:
            o = {
                "ts": str(ts),
                "event": order_event["type"],
                "pos": order_event.get("pos"),
                "price": order_event.get("entry_price"),
                "size": order_event.get("size"),
            }
            append_orders_csv(out_orders, o)

        if meta_state is not None:
            meta_state.update_daily_equity(float(paper_equity), str(ts))
            meta_save_counter += 1
            if cfg.meta_risk_save_every > 0 and meta_save_counter >= cfg.meta_risk_save_every:
                save_json(out_meta_state, meta_state)
                meta_save_counter = 0

        # persist state periodically (for resume)
        state = {"last_pos": int(last_pos), "bars": int(len(df_all)), "last_ts": str(ts)}
        with open(out_state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
