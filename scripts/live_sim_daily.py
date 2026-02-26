# scripts/live_sim_daily.py
# Execution-realism daily loop + report (equity per day, regime stats, fail-safe rules)
# TASK-3H fixes:
#   - MT5 bars CSV reader + DATE/TIME resolution
#   - RegimeBank theta loader infers n_features correctly
#   - Bars mode fails hard on total_trades==0 with diagnostics
#   - Output equity report includes "equity" alias

from __future__ import annotations

import os
import re
import json
import argparse
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from rabit.data.feature_builder import FeatureBuilder, FeatureConfig
from rabit.rl.confidence_weighting import ConfidenceWeighter, ConfidenceWeighterConfig
from rabit.rl.meta_risk import MetaRiskConfig, MetaRiskState
from rabit.rl.regime_ledger import RegimeLedgerConfig, RegimeLedgerState
from rabit.env.trading_env import TradingEnv
from rabit.env.session_filter import SessionFilter, SessionConfig
from rabit.rl.regime_policy_bank import RegimePolicyBank
from rabit.rl.policy_linear import LinearPolicy
from rabit.rl.ars_trainer import make_obs_matrix

_FEATURE_DIAG_PRINTED = False
_FEATURE_DIM_SLICE_WARNED = False

_TRAIN_FEATURE_COLS = [
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

_EQUITY_COLUMNS = [
    "day",
    "start_equity",
    "end_equity",
    "day_pnl",
    "day_peak",
    "day_min",
    "intraday_dd",
    "dd_stop",
    "end_loss_streak",
]

_MT5_TRADE_COLUMNS = [
    "ts",
    "symbol",
    "timeframe",
    "action",
    "size",
    "entry_price",
    "sl",
    "tp",
    "hold_bars",
    "hold_minutes",
    "regime",
    "meta_scale",
    "size_conf",
    "meta_reason",
    "guard_reason",
    "model_path",
    "power",
    "seed",
    "spread_open_cap",
    "spread_spike_cap",
    "enable_london",
    "enable_ny",
    "london_start",
    "london_end",
    "ny_start",
    "ny_end",
    "no_session_filter",
    "no_spread_filter",
]

_MT5_DECISION_COLUMNS = [
    "ts",
    "action",
    "raw_size",
    "final_size",
    "regime",
    "meta_scale",
    "size_conf",
    "confidence",
    "spread",
    "session_ok",
    "guardrail_ok",
]


# ----------------------------
# (1) CSV IO & normalization
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _debug_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg)


def _read_sample_line(path: str, max_chars: int = 200) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    return line.strip()[:max_chars]
    except Exception:
        return ""
    return ""


def _normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    mapping: Dict[str, str] = {}
    new_cols: List[str] = []
    for c in df.columns:
        orig = str(c)
        s = orig.strip()
        if s.startswith("<") and s.endswith(">"):
            s = s[1:-1].strip()
        s = s.lower()
        mapping[orig] = s
        new_cols.append(s)
    df.columns = new_cols
    return df, mapping


def _alias_column(df: pd.DataFrame, target: str, aliases: List[str]) -> pd.DataFrame:
    if target in df.columns:
        return df
    for c in aliases:
        if c in df.columns:
            df[target] = df[c]
            return df
    return df


def _ensure_mt5_bars_columns(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Ensure MT5-style bars columns exist and are numeric:
      - required: open, high, low, close
      - tickvol from tickvol/tick_vol/tick vol/vol/volume
      - spread default 0 if missing
    """
    out = df.copy()

    out = _alias_column(
        out,
        "tickvol",
        ["tickvol", "tick_vol", "tick vol", "tickvolume", "tick volume", "ticks"],
    )
    if "tickvol" not in out.columns:
        out = _alias_column(out, "tickvol", ["vol", "volume", "real_volume", "real volume"])
        if debug and "tickvol" in out.columns:
            _debug_print(debug, "[debug] tickvol missing -> using vol/volume as tickvol")

    if "tickvol" not in out.columns:
        out["tickvol"] = 0

    if "spread" not in out.columns:
        out["spread"] = 0

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}. detected_cols={list(out.columns)}")

    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["tickvol"] = pd.to_numeric(out["tickvol"], errors="coerce").fillna(0).astype("int64")
    out["spread"] = pd.to_numeric(out["spread"], errors="coerce").fillna(0).astype("int64")
    return out


def _prepare_bars_df(df: pd.DataFrame, time_col: str, debug: bool = False) -> pd.DataFrame:
    out = _ensure_mt5_bars_columns(df, debug=debug)
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col])
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.sort_values(time_col)
    out = out.drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)
    out = out.set_index(time_col, drop=False)
    return out


def read_csv_mt5_first(path: str, sample_line: str, debug: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Legacy CSV reader (kept for backward compatibility).
    """
    detected_sep = ","
    df = pd.read_csv(path)
    if len(df.columns) == 1:
        only = str(df.columns[0])
        if "\t" in only or "\t" in sample_line:
            df = pd.read_csv(path, sep="\t")
            detected_sep = "\t"
        elif ";" in only or ";" in sample_line:
            df = pd.read_csv(path, sep=";")
            detected_sep = ";"
    _debug_print(debug, f"[debug] legacy detected_sep={repr(detected_sep)}")
    return df, detected_sep


def read_csv_smart(path: str, debug: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    CSV smart reader:
      - first try pd.read_csv(path)
      - if 1 col and header contains '\\t' -> re-read with sep='\\t'
      - else if header contains ';' -> re-read with sep=';'
    """
    detected_sep = ","
    df = pd.read_csv(path)
    if len(df.columns) == 1:
        header = str(df.columns[0])
        if "\t" in header:
            df = pd.read_csv(path, sep="\t")
            detected_sep = "\t"
        elif ";" in header:
            df = pd.read_csv(path, sep=";")
            detected_sep = ";"

    # Legacy fallback: sample-line sniff if still 1 column
    if len(df.columns) == 1 and detected_sep == ",":
        sample_line = _read_sample_line(path)
        if "\t" in sample_line or ";" in sample_line:
            df, detected_sep = read_csv_mt5_first(path, sample_line=sample_line, debug=debug)

    _debug_print(debug, f"[debug] detected_sep={repr(detected_sep)}")
    return df, detected_sep


def _col_present(cols: set[str], name: str) -> bool:
    if name in cols:
        return True
    return name.replace("_", " ") in cols


def resolve_timestamp(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Resolve timestamp for bars:
      - if timestamp/datetime/ts exists -> parse that
      - else if date+time -> build timestamp
      - else if date -> timestamp from date
      - else if time -> parse time
    Drops NaT rows and sorts by timestamp.
    """
    if df is None or len(df) == 0:
        return df, None

    out = df.copy()
    cols = set(out.columns)
    time_col: Optional[str] = None

    for c in ["timestamp", "datetime", "ts"]:
        if c in cols:
            time_col = c
            break

    if time_col is None and "date" in cols and "time" in cols:
        out["timestamp"] = pd.to_datetime(
            out["date"].astype(str).str.strip() + " " + out["time"].astype(str).str.strip(),
            errors="coerce",
        )
        time_col = "timestamp"
    elif time_col is None and "date" in cols:
        out["timestamp"] = pd.to_datetime(out["date"].astype(str).str.strip(), errors="coerce")
        time_col = "timestamp"
    elif time_col is None and "time" in cols:
        time_col = "time"

    if time_col is None:
        return out, None

    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return out, time_col


def detect_input_mode(df: pd.DataFrame) -> str:
    """
    Detect bars vs trades CSV based on normalized columns.
    """
    cols = set(df.columns)
    is_bars = all(_col_present(cols, c) for c in ["open", "high", "low", "close"])
    is_trades = all(_col_present(cols, c) for c in ["entry_time", "exit_time", "entry_price", "exit_price"])

    if is_bars and not is_trades:
        return "bars"
    if is_trades and not is_bars:
        return "trades"
    raise ValueError(f"Ambiguous or unknown CSV columns. detected_cols={','.join(sorted(cols))}")


# ----------------------------
# (2) Policy/model loader
# ----------------------------

def _linear_policy_n_out() -> int:
    try:
        return int(LinearPolicy(n_features=1).n_out)
    except Exception:
        return 6


def _infer_n_features_from_theta(theta: np.ndarray, n_out: int, regime: Optional[str] = None) -> int:
    theta_len = int(len(theta))
    bias = n_out
    if (theta_len - bias) % n_out != 0:
        reg = f"regime={regime} " if regime else ""
        raise ValueError(f"Invalid theta length for {reg}theta_len={theta_len}, n_out={n_out}")
    n_features = int((theta_len - bias) // n_out)
    if n_features <= 0:
        reg = f"regime={regime} " if regime else ""
        raise ValueError(f"Invalid n_features for {reg}theta_len={theta_len}, n_out={n_out}")
    return n_features


def _infer_model_n_features(model: Any) -> int:
    if hasattr(model, "policies"):
        policies = getattr(model, "policies", None)
        if isinstance(policies, dict) and len(policies) > 0:
            nset = {int(getattr(p, "n_features", 0) or 0) for p in policies.values()}
            nset = {n for n in nset if n > 0}
            if len(nset) > 1:
                raise ValueError(f"Policy bank has inconsistent n_features: {sorted(nset)}")
            if len(nset) == 1:
                return int(list(nset)[0])
    if hasattr(model, "n_features"):
        try:
            return int(getattr(model, "n_features"))
        except Exception:
            return 0
    return 0


def _load_regime_bank_legacy(z: Any, debug: bool = False) -> RegimePolicyBank:
    """
    Legacy loader (kept for compatibility; used only if new inference fails).
    """
    policies: Dict[str, LinearPolicy] = {}
    for k in ["trend", "range", "highvol"]:
        if k in z:
            theta = np.asarray(z[k]).astype(np.float64).reshape(-1)
            n_features = int(z[k].shape[0]) if hasattr(z[k], "shape") else int(len(theta))
            p = LinearPolicy(n_features=n_features)
            p.set_params_flat(theta)
            policies[k] = p
    _debug_print(debug, f"[debug] legacy loader policies={list(policies.keys())}")
    return RegimePolicyBank(policies, fallback=None)


def load_regime_bank(model_path: str, debug: bool = False) -> RegimePolicyBank:
    z = np.load(model_path)
    policies: Dict[str, LinearPolicy] = {}
    try:
        for k in ["trend", "range", "highvol"]:
            if k in z:
                theta = np.asarray(z[k]).astype(np.float64).reshape(-1)
                n_out = _linear_policy_n_out()
                n_features = _infer_n_features_from_theta(theta, n_out=n_out, regime=k)
                _debug_print(
                    debug,
                    f"[debug] regime={k} theta_len={len(theta)} n_out={n_out} n_features={n_features}",
                )
                p = LinearPolicy(n_features=n_features)
                p.set_params_flat(theta)
                policies[k] = p
        return RegimePolicyBank(policies, fallback=None)
    except Exception as e:
        _debug_print(debug, f"[debug] new loader failed: {e}; using legacy loader")
        return _load_regime_bank_legacy(z, debug=debug)


def _build_feature_pipeline(df_bars: pd.DataFrame, debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray]:
    """
    Build the SAME feature pipeline used in training:
      FeatureBuilder(FeatureConfig(dropna=False, add_atr=True))
      + make_obs_matrix normalization.
    Returns (df_env, feat_df, feature_cols, X_all).
    """
    fb = FeatureBuilder(FeatureConfig(dropna=False, add_atr=True))
    feat = fb.build(df_bars, prefix="")

    missing = [c for c in _TRAIN_FEATURE_COLS if c not in feat.columns]
    if missing:
        raise ValueError(f"FeatureBuilder missing columns: {missing}. detected_cols={list(feat.columns)}")

    feature_cols = [c for c in _TRAIN_FEATURE_COLS if c in feat.columns]
    X_all = make_obs_matrix(feat[feature_cols].copy(), feature_cols)
    X_all = np.asarray(X_all, dtype=np.float64)

    df_env = df_bars.join(feat[["atr"]], how="left")
    if "atr" not in df_env.columns:
        raise ValueError("Feature pipeline failed to attach 'atr' to env dataframe")

    if len(df_env) != len(X_all):
        raise ValueError(f"Feature pipeline length mismatch: df_env={len(df_env)} X={len(X_all)}")

    _debug_print(
        debug,
        f"[debug] feature_source=TradingEnv pipeline (FeatureBuilder+make_obs_matrix) features={len(feature_cols)}",
    )
    return df_env, feat, feature_cols, X_all


# ----------------------------
# (3) Daily simulation core
# ----------------------------

def parse_ts(x: Any) -> pd.Timestamp:
    try:
        return pd.to_datetime(x, utc=False)
    except Exception:
        return pd.Timestamp(x)


def ledger_to_df(ledger: Any) -> pd.DataFrame:
    if ledger is None:
        return pd.DataFrame()

    for attr in ["to_frame", "to_df", "to_dataframe"]:
        if hasattr(ledger, attr) and callable(getattr(ledger, attr)):
            try:
                df = getattr(ledger, attr)()
                if isinstance(df, pd.DataFrame):
                    return df
            except Exception:
                pass

    for attr in ["records", "rows", "trades"]:
        if hasattr(ledger, attr):
            try:
                obj = getattr(ledger, attr)
                if isinstance(obj, list) and len(obj) > 0:
                    if isinstance(obj[0], dict):
                        return pd.DataFrame(obj)
                    if hasattr(obj[0], "__dict__"):
                        return pd.DataFrame([getattr(t, "__dict__", {}) for t in obj])
            except Exception:
                pass

    return pd.DataFrame()


def get_balance(ledger: Any) -> float:
    if ledger is None:
        return 0.0
    if hasattr(ledger, "balance"):
        try:
            return float(getattr(ledger, "balance"))
        except Exception:
            pass
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


def count_closed_trades(ledger: Any) -> int:
    df = ledger_to_df(ledger)
    if len(df) == 0:
        return 0
    if "entry_time" in df.columns and "exit_time" in df.columns:
        return int(len(df))
    if "pnl" in df.columns:
        return int(len(df))
    return int(len(df))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _compute_pnl_from_trades_df(trades_df: pd.DataFrame) -> float:
    if trades_df is None or len(trades_df) == 0:
        return 0.0
    for c in ["pnl", "net_pnl", "profit"]:
        if c in trades_df.columns:
            return float(pd.to_numeric(trades_df[c], errors="coerce").fillna(0.0).sum())
    return 0.0


def resolve_time_col(df: pd.DataFrame, prefer_timestamp_if_date_time: bool = False) -> Optional[str]:
    if df is None or df.columns is None or len(df.columns) == 0:
        return None

    cols = [str(c).strip().lower() for c in df.columns]
    if prefer_timestamp_if_date_time and ("date" in cols and "time" in cols):
        for c in ["timestamp", "datetime", "ts"]:
            if c in cols:
                return df.columns[cols.index(c)]

    for c in ["time", "timestamp", "ts", "datetime"]:
        if c in cols:
            return df.columns[cols.index(c)]

    for c in ["entry_time", "exit_time", "open_time", "close_time"]:
        if c in cols:
            return df.columns[cols.index(c)]

    return None


def _intraday_equity_from_equity_df(equity_df: pd.DataFrame) -> Optional[pd.Series]:
    if equity_df is None or len(equity_df) == 0:
        return None

    df = equity_df.copy()
    df, _ = _normalize_columns(df)
    cols = [c.lower() for c in df.columns]

    time_col = resolve_time_col(df)
    val_col = None
    for c in ["equity", "balance"]:
        if c in cols:
            val_col = df.columns[cols.index(c)]
            break

    if time_col is None and val_col is None and len(df.columns) == 2:
        time_col = df.columns[0]
        val_col = df.columns[1]

    if time_col is None or val_col is None:
        return None

    s = pd.to_numeric(df[val_col], errors="coerce")
    t = pd.to_datetime(df[time_col], errors="coerce")
    out = pd.Series(s.values, index=t).dropna()
    if len(out) == 0:
        return None
    return out.sort_index()


def _row_to_feature_vector(row: Any, n_features: int) -> np.ndarray:
    try:
        s = pd.to_numeric(row, errors="coerce")
    except Exception:
        s = pd.Series(row)
        s = pd.to_numeric(s, errors="coerce")
    vals = np.asarray(s, dtype=np.float64)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    if n_features <= 0:
        n_features = len(vals) if len(vals) > 0 else 1

    if len(vals) >= n_features:
        return vals[:n_features]
    pad = np.zeros((n_features - len(vals),), dtype=np.float64)
    return np.concatenate([vals, pad], axis=0)


def _align_feature_vector(x: np.ndarray, n_features: int, debug: bool = False) -> np.ndarray:
    global _FEATURE_DIM_SLICE_WARNED
    if n_features <= 0:
        return x
    if len(x) > int(n_features):
        if debug and not _FEATURE_DIM_SLICE_WARNED:
            print(
                f"[warn] feature_dim: data={len(x)} > model={int(n_features)}; slicing to match"
            )
            _FEATURE_DIM_SLICE_WARNED = True
        return x[: int(n_features)]
    if len(x) < int(n_features):
        raise RuntimeError(
            f"Feature dimension mismatch: got {len(x)}, expected {int(n_features)}"
        )
    return x


def _decide_compat(model: Any, row: Any, debug: bool = False) -> Tuple[int, float, Dict[str, Any]]:
    global _FEATURE_DIAG_PRINTED
    policies = getattr(model, "policies", None)
    if isinstance(policies, dict) and len(policies) > 0:
        regime = None
        if isinstance(row, pd.Series) and "regime" in row.index:
            regime = str(row.get("regime"))
        if regime is None or regime not in policies:
            regime = "trend" if "trend" in policies else list(policies.keys())[0]
        policy = policies.get(regime, list(policies.values())[0])
        n_features = getattr(policy, "n_features", 0)
        x = _row_to_feature_vector(row, n_features)
        if len(x) != int(n_features):
            raise RuntimeError(f"Feature dimension mismatch: got {len(x)}, expected {n_features}")
        if debug and not _FEATURE_DIAG_PRINTED:
            x_min = float(np.nanmin(x)) if len(x) > 0 else float("nan")
            x_max = float(np.nanmax(x)) if len(x) > 0 else float("nan")
            x_mean = float(np.nanmean(x)) if len(x) > 0 else float("nan")
            x_nans = int(np.isnan(x).sum()) if len(x) > 0 else 0
            print(
                "[feature_diag] shape="
                f"{np.asarray(x).shape} expected_n_features={int(n_features)} "
                f"min={x_min} max={x_max} mean={x_mean} n_nans={x_nans}"
            )
            _FEATURE_DIAG_PRINTED = True
        dir_, _tp_mult, _sl_mult, _hold_max = policy.act(x)
        direction = 0 if dir_ == 0 else (1 if dir_ == 1 else -1)
        raw_conf = 1.0 if dir_ != 0 else 0.0
        return direction, raw_conf, {"regime": str(regime)}

    if hasattr(model, "act"):
        n_features = getattr(model, "n_features", 0)
        x = _row_to_feature_vector(row, n_features)
        if len(x) != int(n_features):
            raise RuntimeError(f"Feature dimension mismatch: got {len(x)}, expected {n_features}")
        if debug and not _FEATURE_DIAG_PRINTED:
            x_min = float(np.nanmin(x)) if len(x) > 0 else float("nan")
            x_max = float(np.nanmax(x)) if len(x) > 0 else float("nan")
            x_mean = float(np.nanmean(x)) if len(x) > 0 else float("nan")
            x_nans = int(np.isnan(x).sum()) if len(x) > 0 else 0
            print(
                "[feature_diag] shape="
                f"{np.asarray(x).shape} expected_n_features={int(n_features)} "
                f"min={x_min} max={x_max} mean={x_mean} n_nans={x_nans}"
            )
            _FEATURE_DIAG_PRINTED = True
        dir_, _tp_mult, _sl_mult, _hold_max = model.act(x)
        direction = 0 if dir_ == 0 else (1 if dir_ == 1 else -1)
        raw_conf = 1.0 if dir_ != 0 else 0.0
        return direction, raw_conf, {"regime": "unknown"}

    return 0, 0.0, {"regime": "unknown", "reason": "no_decide"}


def _decide_from_features(
    model: Any,
    x: np.ndarray,
    row: Optional[pd.Series] = None,
    debug: bool = False,
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Decide using pre-built feature vector (training pipeline).
    Keeps regime selection logic compatible with legacy path.
    """
    global _FEATURE_DIAG_PRINTED
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    policies = getattr(model, "policies", None)
    if isinstance(policies, dict) and len(policies) > 0:
        regime = None
        if isinstance(row, pd.Series) and "regime" in row.index:
            regime = str(row.get("regime"))
        if regime is None or regime not in policies:
            regime = "trend" if "trend" in policies else list(policies.keys())[0]

        policy = policies.get(regime, list(policies.values())[0])
        n_features = int(getattr(policy, "n_features", 0) or 0)
        if n_features > 0 and len(x) != n_features:
            raise RuntimeError(f"Feature dimension mismatch: got {len(x)}, expected {n_features}")

        if debug and not _FEATURE_DIAG_PRINTED:
            x_min = float(np.nanmin(x)) if len(x) > 0 else float("nan")
            x_max = float(np.nanmax(x)) if len(x) > 0 else float("nan")
            x_mean = float(np.nanmean(x)) if len(x) > 0 else float("nan")
            x_nans = int(np.isnan(x).sum()) if len(x) > 0 else 0
            print(
                "[feature_diag] source=training_pipeline "
                f"shape={np.asarray(x).shape} expected_n_features={n_features} "
                f"min={x_min} max={x_max} mean={x_mean} n_nans={x_nans}"
            )
            _FEATURE_DIAG_PRINTED = True

        dir_, _tp_mult, _sl_mult, _hold_max = policy.act(x)
        direction = 0 if dir_ == 0 else (1 if dir_ == 1 else -1)
        raw_conf = 1.0 if dir_ != 0 else 0.0
        return direction, raw_conf, {"regime": str(regime)}

    if hasattr(model, "act"):
        n_features = int(getattr(model, "n_features", 0) or 0)
        if n_features > 0 and len(x) != n_features:
            raise RuntimeError(f"Feature dimension mismatch: got {len(x)}, expected {n_features}")

        if debug and not _FEATURE_DIAG_PRINTED:
            x_min = float(np.nanmin(x)) if len(x) > 0 else float("nan")
            x_max = float(np.nanmax(x)) if len(x) > 0 else float("nan")
            x_mean = float(np.nanmean(x)) if len(x) > 0 else float("nan")
            x_nans = int(np.isnan(x).sum()) if len(x) > 0 else 0
            print(
                "[feature_diag] source=training_pipeline "
                f"shape={np.asarray(x).shape} expected_n_features={n_features} "
                f"min={x_min} max={x_max} mean={x_mean} n_nans={x_nans}"
            )
            _FEATURE_DIAG_PRINTED = True

        dir_, _tp_mult, _sl_mult, _hold_max = model.act(x)
        direction = 0 if dir_ == 0 else (1 if dir_ == 1 else -1)
        raw_conf = 1.0 if dir_ != 0 else 0.0
        return direction, raw_conf, {"regime": "unknown"}

    return 0, 0.0, {"regime": "unknown", "reason": "no_decide"}


def build_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series([], dtype=float)

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period, min_periods=1).mean()
    atr = atr.astype(float).ffill().bfill().fillna(0.0)
    return atr


def run_one_day(
    df_day: pd.DataFrame,
    model: Any,
    weighter: ConfidenceWeighter,
    meta_state: Optional[MetaRiskState],
    meta_feedback: bool,
    session_filter: Optional[SessionFilter],
    cfg: Dict[str, Any],
    global_balance: float,
    X_day: Optional[np.ndarray] = None,
    n_features: int = 0,
    legacy_features: bool = False,
    bypass_session: bool = False,
    bypass_spread: bool = False,
    debug: bool = False,
    collect_decisions: bool = False,
    collect_trade_meta: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[str, Any]]:
    if legacy_features:
        atr_series = build_atr_series(df_day, period=int(cfg.get("atr_period", 14)))
        if "atr" not in df_day.columns:
            df_day = df_day.copy()
            df_day["atr"] = atr_series
    else:
        if "atr" not in df_day.columns:
            raise ValueError("Feature pipeline missing 'atr' in df_day; cannot run TradingEnv")
        if X_day is None:
            raise ValueError("Feature pipeline missing X_day; set --legacy_features 1 to use legacy features")
        if len(X_day) != len(df_day):
            raise ValueError(f"X_day length mismatch: X_day={len(X_day)} df_day={len(df_day)}")

    if not isinstance(df_day.index, pd.DatetimeIndex):
        df_day = df_day.set_index("time", drop=False)

    try:
        env = TradingEnv(
            df_day,
            spread_cap=cfg.get("spread_open_cap", 200),
            spread_spike_cap=cfg.get("spread_spike_cap", 300),
            session_filter=session_filter,
        )
    except TypeError:
        env = TradingEnv(
            df_day,
            spread_open_cap=cfg.get("spread_open_cap", 200),
            spread_force_close_cap=cfg.get("spread_spike_cap", 300),
            session_filter=session_filter,
        )

    spread_open_cap = int(cfg.get("spread_open_cap", 200))
    spread_spike_cap = int(cfg.get("spread_spike_cap", 300))
    max_loss_streak = int(cfg.get("max_loss_streak", 4))
    daily_dd_limit = float(cfg.get("daily_dd_limit", 200.0))

    debug_info: Dict[str, Any] = {
        "bars": int(len(df_day)),
        "signals": 0,
        "allowed": 0,
        "dd_stop": False,
        "end_loss_streak": 0,
        "bars_total": int(len(df_day)),
        "bars_after_time_parse": int(len(df_day)),
        "bars_after_session_filter": 0,
        "bars_after_spread_filter": 0,
        "signals_total": 0,
        "signals_after_guardrails": 0,
        "final_allowed": 0,
        "pass_session": 0,
        "pass_spread_open": 0,
        "pass_spread_spike": 0,
        "pass_guardrails": 0,
        "session_reject": 0,
        "spread_open_reject": 0,
        "spread_spike_reject": 0,
        "guardrail_reject": 0,
        "policy_hold": 0,
        "size_zero": 0,
    }

    decision_rows: Optional[List[Dict[str, Any]]] = [] if collect_decisions else None
    trade_meta: Optional[List[Dict[str, Any]]] = [] if collect_trade_meta else None
    last_decision_meta: Dict[str, Any] = {}

    loss_streak = 0
    dd_stop = False
    equity_start = global_balance
    equity_peak = global_balance
    equity_min = global_balance

    idx_ptr = {"i": 0}

    def _session_ok(ts: pd.Timestamp) -> bool:
        if bypass_session or session_filter is None:
            return True
        try:
            return bool(session_filter.is_open(ts))
        except Exception:
            return True

    def _record_decision(payload: Dict[str, Any]) -> None:
        if decision_rows is None:
            return
        decision_rows.append(payload)

    def policy_func(i: int):
        nonlocal loss_streak, dd_stop, equity_peak, equity_min, last_decision_meta

        if isinstance(i, (int, np.integer)):
            row = df_day.iloc[int(i)]
            i_idx = int(i)
        else:
            row = i
            i_idx = int(idx_ptr["i"])
            idx_ptr["i"] += 1

        ts = parse_ts(row.get("time", row.get("timestamp", row.name)))
        spread_points = int(row.get("spread", 0))
        session_ok = _session_ok(ts)
        spread_open_ok = True
        spread_spike_ok = True
        if not bypass_spread:
            spread_open_ok = spread_points <= spread_open_cap
            spread_spike_ok = spread_points <= spread_spike_cap
        if session_ok:
            debug_info["bars_after_session_filter"] += 1
            if spread_open_ok and spread_spike_ok:
                debug_info["bars_after_spread_filter"] += 1
        if hasattr(model, "decide"):
            direction, raw_conf, info = model.decide(row)
        else:
            if legacy_features:
                direction, raw_conf, info = _decide_compat(model, row, debug=debug)
            else:
                if hasattr(env, "get_obs") and callable(getattr(env, "get_obs")):
                    x = env.get_obs(i_idx)
                elif hasattr(env, "obs") and callable(getattr(env, "obs")):
                    x = env.obs(i_idx)
                else:
                    x = X_day[i_idx]
                x = np.asarray(x, dtype=np.float64).reshape(-1)
                x = _align_feature_vector(x, n_features, debug=debug)
                direction, raw_conf, info = _decide_from_features(model, x, row=row, debug=debug)
        debug_info["signals"] += 1

        if hasattr(weighter, "size_from_confidence"):
            size_conf = float(weighter.size_from_confidence(float(raw_conf)))
        else:
            size_conf = float(weighter.size(float(raw_conf)))
        allow = size_conf > 0.0

        regime = info.get("regime", "unknown") if isinstance(info, dict) else "unknown"

        meta_scale = 1.0
        meta_reason = None
        if meta_state is not None:
            try:
                meta_scale, meta_reason = meta_state.meta_scale_with_reason(regime, update_state=False)
            except Exception:
                meta_scale = float(meta_state.meta_scale(regime))
                meta_reason = None

        final_size = size_conf * meta_scale

        cur_equity = global_balance + get_balance(env.ledger)
        equity_peak = max(equity_peak, cur_equity)
        equity_min = min(equity_min, cur_equity)
        intraday_dd = equity_peak - cur_equity

        if intraday_dd >= daily_dd_limit:
            dd_stop = True

        guardrails_ok = allow and (not dd_stop) and (loss_streak < max_loss_streak)
        guard_reason = "ok"
        if int(direction) == 0:
            guard_reason = "policy_hold"
        elif not allow:
            guard_reason = "size_zero"
        elif dd_stop:
            guard_reason = "dd_stop"
        elif loss_streak >= max_loss_streak:
            guard_reason = "loss_streak"
        elif not session_ok:
            guard_reason = "session_closed"
        elif not spread_open_ok:
            guard_reason = "spread_open_cap"
        elif not spread_spike_ok:
            guard_reason = "spread_spike_cap"

        decision_action = "HOLD"
        exec_size = 0.0
        if int(direction) != 0 and guardrails_ok and session_ok and spread_open_ok and spread_spike_ok:
            if float(direction) > 0:
                decision_action = "BUY"
            else:
                decision_action = "SELL"
            exec_size = float(np.clip(final_size, 0.0, 1.0))

        _record_decision(
            {
                "ts": parse_ts(ts).isoformat() if ts is not None else None,
                "action": decision_action,
                "raw_size": float(size_conf),
                "final_size": float(exec_size),
                "regime": str(regime),
                "meta_scale": float(meta_scale),
                "size_conf": float(size_conf),
                "confidence": float(raw_conf),
                "spread": int(spread_points),
                "session_ok": bool(session_ok),
                "guardrail_ok": bool(guardrails_ok),
                "meta_reason": str(meta_reason) if meta_reason is not None else None,
                "guard_reason": str(guard_reason) if guard_reason else None,
            }
        )

        last_decision_meta = {
            "regime": str(regime),
            "meta_scale": float(meta_scale),
            "meta_reason": str(meta_reason) if meta_reason is not None else None,
            "size_conf": float(size_conf),
            "size": float(exec_size),
            "final_size": float(exec_size),
            "guard_reason": str(guard_reason) if guard_reason else None,
            "hold_bars": 20,
        }

        if int(direction) == 0:
            debug_info["policy_hold"] += 1
            return (0, 0.8, 0.8, 20)

        debug_info["signals_total"] += 1
        if session_ok:
            debug_info["pass_session"] += 1
        if spread_open_ok:
            debug_info["pass_spread_open"] += 1
        if spread_spike_ok:
            debug_info["pass_spread_spike"] += 1
        if guardrails_ok:
            debug_info["pass_guardrails"] += 1
            debug_info["signals_after_guardrails"] += 1

        if guardrails_ok and session_ok and spread_open_ok and spread_spike_ok:
            debug_info["final_allowed"] += 1

        if not session_ok:
            debug_info["session_reject"] += 1
        elif not spread_open_ok:
            debug_info["spread_open_reject"] += 1
        elif not spread_spike_ok:
            debug_info["spread_spike_reject"] += 1
        elif not allow:
            debug_info["size_zero"] += 1
        elif dd_stop or loss_streak >= max_loss_streak:
            debug_info["guardrail_reject"] += 1

        if dd_stop or not allow:
            return (0, 0.8, 0.8, 20)

        act_dir = int(direction)
        if act_dir == -1:
            act_dir = 2
        elif act_dir not in (0, 1, 2):
            act_dir = int(np.sign(act_dir))
            if act_dir == -1:
                act_dir = 2

        if act_dir == 0:
            return (0, 0.8, 0.8, 20)

        return (act_dir, 0.8, 0.8, 20, float(final_size))

    for i in range(len(env.df)):
        prev_trades = len(env.ledger.trades)
        action = policy_func(env.df.iloc[i])
        env.step(i, action)
        if trade_meta is not None:
            new_trades = len(env.ledger.trades) - prev_trades
            if new_trades > 0:
                for _ in range(new_trades):
                    trade_meta.append(dict(last_decision_meta))

    ledger = env.ledger

    trades_df = pd.DataFrame()
    equity_df = pd.DataFrame()

    if hasattr(env, "trades") and isinstance(env.trades, list) and len(env.trades) > 0:
        try:
            trades_df = pd.DataFrame(env.trades)
        except Exception:
            trades_df = pd.DataFrame()

    if hasattr(env, "equity_curve") and isinstance(env.equity_curve, list) and len(env.equity_curve) > 0:
        try:
            equity_df = pd.DataFrame(env.equity_curve)
        except Exception:
            equity_df = pd.DataFrame()

    if len(trades_df) == 0:
        try:
            trades_df = ledger_to_df(ledger)
        except Exception:
            trades_df = pd.DataFrame()

    regime_counts: Dict[str, int] = {}
    if len(trades_df) > 0 and "regime" in trades_df.columns:
        try:
            regime_counts = (
                trades_df["regime"]
                .astype(str)
                .value_counts(dropna=True)
                .to_dict()
            )
        except Exception:
            regime_counts = {}

    if len(equity_df) == 0 and hasattr(ledger, "equity_curve"):
        try:
            eq_obj = getattr(ledger, "equity_curve")
            if isinstance(eq_obj, list) and len(eq_obj) > 0:
                if isinstance(eq_obj[0], (list, tuple)) and len(eq_obj[0]) == 2:
                    equity_df = pd.DataFrame(eq_obj, columns=["time", "equity"])
                else:
                    equity_df = pd.DataFrame(eq_obj)
        except Exception:
            equity_df = pd.DataFrame()

    bal_end = get_balance(ledger)
    day_pnl = float(bal_end)

    eq_series = _intraday_equity_from_equity_df(equity_df)
    if eq_series is not None and len(eq_series) >= 2:
        eq_delta = float(eq_series.iloc[-1] - eq_series.iloc[0])
        if abs(day_pnl) < 1e-12 and abs(eq_delta) > 1e-12:
            day_pnl = eq_delta

    if abs(day_pnl) < 1e-12 and len(trades_df) > 0:
        day_pnl = _compute_pnl_from_trades_df(trades_df)

    if meta_state is not None and meta_feedback and len(trades_df) > 0:
        if "exit_time" in trades_df.columns:
            ts_col = "exit_time"
        elif "entry_time" in trades_df.columns:
            ts_col = "entry_time"
        else:
            ts_col = None

        pnl_col = "pnl" if "pnl" in trades_df.columns else None

        if ts_col is not None and pnl_col is not None:
            for _, tr in trades_df.iterrows():
                tts = parse_ts(tr[ts_col])
                date_key = tts.date().isoformat()
                pnl = _safe_float(tr[pnl_col], 0.0)
                regime = tr["regime"] if "regime" in trades_df.columns else "unknown"
        try:
            # ưu tiên positional để tránh mismatch keyword
            meta_state.update_trade(str(regime), float(pnl), date_key)
        except TypeError:
            # fallback: một số version dùng date_str / date / ts_col khác tên
            try:
                meta_state.update_trade(str(regime), float(pnl), date_str=date_key)
            except TypeError:
                try:
                    meta_state.update_trade(str(regime), float(pnl), date=date_key)
                except TypeError:
                    # cuối cùng: gọi đúng 2 tham số nếu bản MetaRiskState không nhận date
                    meta_state.update_trade(str(regime), float(pnl))

    debug_info["allowed"] = int(len(trades_df)) if len(trades_df) > 0 else int(count_closed_trades(ledger))
    debug_info["total_trades"] = int(debug_info["allowed"])
    debug_info["dd_stop"] = bool(dd_stop)
    debug_info["equity_start"] = float(equity_start)
    debug_info["equity_end"] = float(global_balance + day_pnl)
    debug_info["equity_peak"] = float(equity_peak)
    debug_info["equity_min"] = float(equity_min)
    debug_info["intraday_dd"] = float(max(0.0, equity_peak - (global_balance + day_pnl)))
    debug_info["regime_counts"] = regime_counts
    if decision_rows is not None:
        debug_info["decision_rows"] = decision_rows
    if trade_meta is not None:
        debug_info["trade_meta"] = trade_meta

    return trades_df, equity_df, day_pnl, debug_info


# ----------------------------
# (4) Reporting writers
# ----------------------------

def _default_execution_settings() -> Dict[str, Any]:
    return {
        "slip_base": 0.0,
        "slip_k_spread": 0.1,
        "slip_k_atr": 0.02,
        "slip_noise_std": 0.02,
        "commission_per_side": 0.0,
        "seed": 7,
    }


def _default_session_settings() -> Dict[str, Any]:
    return {
        "london_start": 7,
        "london_end": 16,
        "ny_start": 13,
        "ny_end": 21,
        "enable_london": True,
        "enable_ny": True,
    }


def _parse_bool_flag(val: Any, default: bool = True) -> bool:
    try:
        return bool(int(val))
    except Exception:
        return bool(default)


def _make_session_filter(session_cfg: Dict[str, Any]) -> Optional[SessionFilter]:
    enable_london = bool(session_cfg.get("enable_london", True))
    enable_ny = bool(session_cfg.get("enable_ny", True))
    if not enable_london and not enable_ny:
        return None
    cfg = SessionConfig(
        london_start=int(session_cfg.get("london_start", 7)),
        london_end=int(session_cfg.get("london_end", 16)),
        ny_start=int(session_cfg.get("ny_start", 13)),
        ny_end=int(session_cfg.get("ny_end", 21)),
        enable_london=enable_london,
        enable_ny=enable_ny,
    )
    return SessionFilter(cfg)


def write_equity_csv(path: str, equity_rows: List[Dict[str, Any]]) -> None:
    eq_df = pd.DataFrame(equity_rows)
    for c in _EQUITY_COLUMNS:
        if c not in eq_df.columns:
            if c in ("dd_stop",):
                eq_df[c] = False
            else:
                eq_df[c] = 0
    if "equity" not in eq_df.columns and "end_equity" in eq_df.columns:
        eq_df["equity"] = eq_df["end_equity"]
    ordered = _EQUITY_COLUMNS + [c for c in eq_df.columns if c not in _EQUITY_COLUMNS]
    eq_df = eq_df[ordered]
    eq_df.to_csv(path, index=False)


def write_summary_json(path: str, summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_regime_json(path: str, regime_stats: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(regime_stats, f, ensure_ascii=False, indent=2)


def _value_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if isinstance(value, float) and np.isnan(value):
            return False
        if pd.isna(value):
            return False
    except Exception:
        pass
    return True


def _safe_str(value: Any, default: str = "") -> str:
    if not _value_present(value):
        return default
    s = str(value)
    if not s or s.lower() == "nan":
        return default
    return s


def _safe_optional_str(value: Any) -> Optional[str]:
    if not _value_present(value):
        return None
    s = str(value)
    if not s or s.lower() == "nan":
        return None
    return s


def _safe_float_or_none(value: Any) -> Optional[float]:
    if not _value_present(value):
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _safe_int_or_none(value: Any) -> Optional[int]:
    if not _value_present(value):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _row_get(row: Any, keys: List[str]) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        for k in keys:
            if k in row and _value_present(row.get(k)):
                return row.get(k)
            kl = k.lower()
            if kl in row and _value_present(row.get(kl)):
                return row.get(kl)
            ks = k.replace("_", " ")
            if ks in row and _value_present(row.get(ks)):
                return row.get(ks)
            ksl = ks.lower()
            if ksl in row and _value_present(row.get(ksl)):
                return row.get(ksl)
            kn = k.replace("_", "")
            if kn in row and _value_present(row.get(kn)):
                return row.get(kn)
            knl = kn.lower()
            if knl in row and _value_present(row.get(knl)):
                return row.get(knl)
        return None
    if isinstance(row, pd.Series):
        for k in keys:
            if k in row.index and _value_present(row.get(k)):
                return row.get(k)
            kl = k.lower()
            if kl in row.index and _value_present(row.get(kl)):
                return row.get(kl)
            ks = k.replace("_", " ")
            if ks in row.index and _value_present(row.get(ks)):
                return row.get(ks)
            ksl = ks.lower()
            if ksl in row.index and _value_present(row.get(ksl)):
                return row.get(ksl)
            kn = k.replace("_", "")
            if kn in row.index and _value_present(row.get(kn)):
                return row.get(kn)
            knl = kn.lower()
            if knl in row.index and _value_present(row.get(knl)):
                return row.get(knl)
    return None


def _normalize_action(value: Any) -> Optional[str]:
    if not _value_present(value):
        return None
    if isinstance(value, str):
        v = value.strip().upper()
        if v in ("BUY", "LONG", "B", "L"):
            return "BUY"
        if v in ("SELL", "SHORT", "S"):
            return "SELL"
        if v in ("CLOSE", "EXIT", "C"):
            return "CLOSE"
        if v in ("HOLD", "FLAT", "NONE", "0"):
            return "HOLD"
    try:
        iv = int(float(value))
        if iv == 1:
            return "BUY"
        if iv in (-1, 2):
            return "SELL"
        if iv == 0:
            return "HOLD"
    except Exception:
        pass
    try:
        fv = float(value)
        if fv > 0:
            return "BUY"
        if fv < 0:
            return "SELL"
    except Exception:
        pass
    return None


def _format_ts(value: Any) -> Optional[str]:
    if not _value_present(value):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.isoformat()
    except Exception:
        return _safe_optional_str(value)


def _day_start_from_value(value: Any) -> Optional[str]:
    ts = _format_ts(value)
    if ts is None:
        return None
    try:
        dt = pd.to_datetime(ts, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.normalize().isoformat()
    except Exception:
        return None


def _infer_symbol_timeframe_from_path(path: str) -> Tuple[str, str]:
    base = os.path.basename(path or "")
    stem, _ext = os.path.splitext(base)
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", stem) if t]
    symbol = None
    timeframe = None
    for t in tokens:
        ut = t.upper()
        if timeframe is None and re.match(r"^(M|H|D|W|MN)\\d+$", ut):
            timeframe = ut
        if symbol is None and re.match(r"^[A-Z]{3,10}$", ut):
            if ut not in ("LIVE", "HIST", "DATA", "BARS", "TRADES"):
                symbol = ut
    if symbol is None:
        symbol = "XAUUSD"
    if timeframe is None:
        timeframe = "M1"
    return symbol, timeframe


def _build_mt5_provenance(
    model_path: str,
    power: float,
    execution_settings: Dict[str, Any],
    session_settings: Dict[str, Any],
    spread_open_cap: float,
    spread_spike_cap: float,
    no_session_filter: bool,
    no_spread_filter: bool,
) -> Dict[str, Any]:
    seed = execution_settings.get("seed", 7) if isinstance(execution_settings, dict) else 7
    return {
        "model_path": str(model_path),
        "power": float(power),
        "seed": int(seed) if _value_present(seed) else 7,
        "spread_open_cap": float(spread_open_cap),
        "spread_spike_cap": float(spread_spike_cap),
        "enable_london": bool(session_settings.get("enable_london", True)),
        "enable_ny": bool(session_settings.get("enable_ny", True)),
        "london_start": int(session_settings.get("london_start", 7)),
        "london_end": int(session_settings.get("london_end", 16)),
        "ny_start": int(session_settings.get("ny_start", 13)),
        "ny_end": int(session_settings.get("ny_end", 21)),
        "no_session_filter": int(bool(no_session_filter)),
        "no_spread_filter": int(bool(no_spread_filter)),
    }


def _build_mt5_trade_records(
    trades_df: Optional[pd.DataFrame],
    trade_meta: Optional[List[Dict[str, Any]]],
    default_symbol: str,
    default_timeframe: str,
    day_start_fallback: Optional[str],
    provenance: Dict[str, Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if trades_df is None or len(trades_df) == 0:
        return records

    meta_list = trade_meta or []

    for idx, row in trades_df.iterrows():
        meta = meta_list[idx] if idx < len(meta_list) else {}
        ts = _format_ts(_row_get(row, ["entry_time", "entry_ts", "open_time", "open_ts", "time", "timestamp", "datetime", "ts"]))
        if ts is None:
            ts = _format_ts(_row_get(row, ["exit_time", "exit_ts", "close_time", "close_ts"]))
        if ts is None:
            ts = day_start_fallback

        symbol = _safe_str(_row_get(row, ["symbol", "pair", "instrument"]), default_symbol)
        timeframe = _safe_str(_row_get(row, ["timeframe", "tf"]), default_timeframe)

        action = _normalize_action(_row_get(row, ["action", "side", "direction", "dir"]))
        if action is None:
            action = "HOLD"

        size = _safe_float_or_none(_row_get(row, ["size", "final_size", "volume", "qty", "lot", "lots"]))
        if size is None:
            size = _safe_float_or_none(meta.get("final_size") if isinstance(meta, dict) else None)
        if size is None:
            size = 0.0

        entry_price = _safe_float_or_none(
            _row_get(row, ["entry_price", "open_price", "price", "entry", "fill_price"])
        )
        sl = _safe_float_or_none(
            _row_get(row, ["sl", "stop_loss", "sl_price", "stop", "stop_price"])
        )
        tp = _safe_float_or_none(
            _row_get(row, ["tp", "take_profit", "tp_price", "target", "target_price"])
        )
        hold_bars = _safe_int_or_none(_row_get(row, ["hold_bars", "hold_max"]))
        if hold_bars is None and isinstance(meta, dict):
            hold_bars = _safe_int_or_none(meta.get("hold_bars"))
        hold_minutes = _safe_int_or_none(_row_get(row, ["hold_minutes", "hold_min"]))

        regime = _safe_str(_row_get(row, ["regime", "state"]), "unknown")
        if isinstance(meta, dict) and _value_present(meta.get("regime")):
            regime = _safe_str(meta.get("regime"), regime)

        meta_scale = _safe_float_or_none(_row_get(row, ["meta_scale"]))
        if meta_scale is None and isinstance(meta, dict):
            meta_scale = _safe_float_or_none(meta.get("meta_scale"))
        if meta_scale is None:
            meta_scale = 1.0

        size_conf = _safe_float_or_none(_row_get(row, ["size_conf", "confidence"]))
        if size_conf is None and isinstance(meta, dict):
            size_conf = _safe_float_or_none(meta.get("size_conf"))

        meta_reason = None
        guard_reason = None
        if isinstance(meta, dict):
            meta_reason = meta.get("meta_reason")
            guard_reason = meta.get("guard_reason")
        if meta_reason is None:
            meta_reason = _row_get(row, ["meta_reason"])
        if guard_reason is None:
            guard_reason = _row_get(row, ["guard_reason"])

        record = {
            "ts": ts,
            "symbol": symbol,
            "timeframe": timeframe,
            "action": action,
            "size": float(size),
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "hold_bars": hold_bars,
            "hold_minutes": hold_minutes,
            "regime": regime,
            "meta_scale": float(meta_scale),
            "size_conf": size_conf,
            "meta_reason": _safe_optional_str(meta_reason),
            "guard_reason": _safe_optional_str(guard_reason),
        }
        record.update(provenance)
        for k in _MT5_TRADE_COLUMNS:
            if k not in record:
                record[k] = None
        records.append(record)

    return records


def _build_mt5_decision_records(
    decisions: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not decisions:
        return []
    records: List[Dict[str, Any]] = []
    for row in decisions:
        rec = dict(row) if isinstance(row, dict) else {}
        if "ts" in rec:
            rec["ts"] = _format_ts(rec.get("ts"))
        for k in _MT5_DECISION_COLUMNS:
            if k not in rec:
                rec[k] = None
        records.append(rec)
    return records


def _write_json_lines(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_csv(path: str, records: List[Dict[str, Any]], columns: List[str]) -> None:
    if not records:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    df = pd.DataFrame(records)
    ordered = columns + [c for c in df.columns if c not in columns]
    df = df[ordered]
    df.to_csv(path, index=False)


def _write_mt5_export_summary(path: str, summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def export_mt5_signals(
    out_dir: str,
    trades_df: Optional[pd.DataFrame],
    trade_meta: Optional[List[Dict[str, Any]]],
    decisions: Optional[List[Dict[str, Any]]],
    fmt: str,
    mode: str,
    input_csv: str,
    model_path: str,
    power: float,
    execution_settings: Dict[str, Any],
    session_settings: Dict[str, Any],
    spread_open_cap: float,
    spread_spike_cap: float,
    no_session_filter: bool,
    no_spread_filter: bool,
    no_trades_reason: Optional[str] = None,
    no_decisions_reason: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    fmt_norm = str(fmt or "json").strip().lower()
    if fmt_norm not in ("json", "csv"):
        fmt_norm = "json"
    mode_norm = str(mode or "trades").strip().lower()
    if mode_norm not in ("trades", "decisions", "both"):
        mode_norm = "trades"

    default_symbol, default_timeframe = _infer_symbol_timeframe_from_path(input_csv)
    day_start = None
    if trades_df is not None and len(trades_df) > 0:
        ts_guess = _row_get(trades_df.iloc[0], ["entry_time", "exit_time", "time", "timestamp", "datetime", "ts", "day"])
        day_start = _day_start_from_value(ts_guess)

    provenance = _build_mt5_provenance(
        model_path=model_path,
        power=power,
        execution_settings=execution_settings,
        session_settings=session_settings,
        spread_open_cap=spread_open_cap,
        spread_spike_cap=spread_spike_cap,
        no_session_filter=no_session_filter,
        no_spread_filter=no_spread_filter,
    )

    trades_records: List[Dict[str, Any]] = []
    decisions_records: List[Dict[str, Any]] = []

    trades_records = _build_mt5_trade_records(
        trades_df=trades_df,
        trade_meta=trade_meta,
        default_symbol=default_symbol,
        default_timeframe=default_timeframe,
        day_start_fallback=day_start,
        provenance=provenance,
    )
    trades_path = os.path.join(out_dir, f"mt5_signals_trades.{fmt_norm}")
    if fmt_norm == "json":
        _write_json_lines(trades_path, trades_records)
    else:
        _write_csv(trades_path, trades_records, _MT5_TRADE_COLUMNS)
    if debug:
        print(f"[mt5_export] wrote {trades_path} records={len(trades_records)}")

    if mode_norm in ("decisions", "both"):
        decisions_records = _build_mt5_decision_records(decisions)
        decisions_path = os.path.join(out_dir, f"mt5_signals_decisions.{fmt_norm}")
        if fmt_norm == "json":
            _write_json_lines(decisions_path, decisions_records)
        else:
            _write_csv(decisions_path, decisions_records, _MT5_DECISION_COLUMNS)
        if debug:
            print(f"[mt5_export] wrote {decisions_path} records={len(decisions_records)}")

    summary = {
        "mt5_export": True,
        "format": fmt_norm,
        "mode": mode_norm,
        "total_trades": int(len(trades_records)),
        "total_decisions": int(len(decisions_records)),
        "no_trades_reason": _safe_optional_str(no_trades_reason),
        "no_decisions_reason": _safe_optional_str(no_decisions_reason),
    }
    summary_path = os.path.join(out_dir, "mt5_export_summary.json")
    _write_mt5_export_summary(summary_path, summary)
    if debug:
        print(f"[mt5_export] wrote {summary_path}")

    return summary


def update_regime_ledger(
    out_dir: str,
    daily_rows: List[Dict[str, Any]],
    regime_stats: Dict[str, Any],
    span: int,
    max_days: int,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    ledger_path = os.path.join(out_dir, "regime_ledger.json")
    cfg = RegimeLedgerConfig(span=int(span), max_days=int(max_days))
    state = RegimeLedgerState.load(ledger_path, cfg=cfg)
    state.update_from_daily_outputs(daily_rows, regime_stats, debug=debug)
    state.save(ledger_path)
    return state.compact_summary()


def _build_fail_hard_message(
    input_csv: str,
    model_path: str,
    bars_loaded: int,
    time_col: str,
    first_ts: Optional[pd.Timestamp],
    last_ts: Optional[pd.Timestamp],
    detected_cols: List[str],
    session: Dict[str, Any],
    no_session_filter: bool,
    spread_open_cap: float,
    spread_spike_cap: float,
    no_spread_filter: bool,
    daily_dd_limit: float,
    max_loss_streak: int,
    power: float,
    min_size: float,
    max_size: float,
    deadzone: float,
    meta_risk: int,
    meta_feedback: int,
    meta_state_path: str,
    breakdown: Optional[Dict[str, Any]] = None,
    reason_hint: str = "all signals filtered by session/spread/guardrails",
    notes: str = "Try disabling session filter or increasing spread caps to verify pipeline",
    short: bool = False,
) -> str:
    first_s = str(first_ts) if first_ts is not None else "None"
    last_s = str(last_ts) if last_ts is not None else "None"
    session_line = (
        f"session_filter: enable_london={session.get('enable_london')} "
        f"london=[{session.get('london_start')},{session.get('london_end')}] "
        f"enable_ny={session.get('enable_ny')} "
        f"ny=[{session.get('ny_start')},{session.get('ny_end')}] "
        f"no_session_filter={int(no_session_filter)}"
    )
    spread_line = (
        f"spread_caps: spread_open_cap={spread_open_cap} "
        f"spread_spike_cap={spread_spike_cap} no_spread_filter={int(no_spread_filter)}"
    )

    def _breakdown_line(bd: Optional[Dict[str, Any]]) -> str:
        if not isinstance(bd, dict) or not bd:
            return ""
        parts = [
            f"bars_total={bd.get('bars_total', 0)}",
            f"bars_after_time_parse={bd.get('bars_after_time_parse', 0)}",
            f"bars_after_session_filter={bd.get('bars_after_session_filter', 0)}",
            f"bars_after_spread_filter={bd.get('bars_after_spread_filter', 0)}",
            f"signals_total={bd.get('signals_total', 0)}",
            f"signals_after_guardrails={bd.get('signals_after_guardrails', 0)}",
            f"final_allowed={bd.get('final_allowed', 0)}",
            f"total_trades={bd.get('total_trades', 0)}",
        ]
        return "breakdown: " + " ".join(parts)

    def _reasons_line(bd: Optional[Dict[str, Any]]) -> str:
        if not isinstance(bd, dict) or not bd:
            return ""
        parts = [
            f"session_reject={bd.get('session_reject', 0)}",
            f"spread_open_reject={bd.get('spread_open_reject', 0)}",
            f"spread_spike_reject={bd.get('spread_spike_reject', 0)}",
            f"guardrail_reject={bd.get('guardrail_reject', 0)}",
            f"policy_hold={bd.get('policy_hold', 0)}",
            f"size_zero={bd.get('size_zero', 0)}",
        ]
        return "reasons: " + " ".join(parts)

    if short:
        lines = [
            "Task-3I FAIL: total_trades==0 (bars mode)",
            _breakdown_line(breakdown),
            _reasons_line(breakdown),
            f"model_path={model_path}",
            session_line,
            spread_line,
            f"fail_safe: daily_dd_limit={daily_dd_limit} max_loss_streak={max_loss_streak}",
            f"meta_risk: enabled={int(meta_risk)} meta_feedback={int(meta_feedback)} state_path={meta_state_path}",
        ]
        return "\n".join([l for l in lines if l])

    lines = [
        "Task-3I FAIL: total_trades==0 (bars mode)",
        f"reason_hint={reason_hint}",
        f"input_csv={input_csv}",
        f"model_path={model_path}",
        f"bars_loaded={bars_loaded}",
        f"time_col={time_col}  first_ts={first_s}  last_ts={last_s}",
        f"detected_cols={','.join(detected_cols)}",
        session_line,
        spread_line,
        f"gating: power={power} min_size={min_size} max_size={max_size} deadzone={deadzone}",
        f"fail_safe: daily_dd_limit={daily_dd_limit} max_loss_streak={max_loss_streak}",
        f"meta_risk: enabled={int(meta_risk)}  meta_feedback={int(meta_feedback)}  state_path={meta_state_path}",
    ]
    breakdown_line = _breakdown_line(breakdown)
    if breakdown_line:
        lines.append(breakdown_line)
    reasons_line = _reasons_line(breakdown)
    if reasons_line:
        lines.append(reasons_line)
    lines.append(f"notes: {notes}")
    return "\n".join(lines)


# ----------------------------
# (5) CLI main()
# ----------------------------

def _select_trade_time_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    out = df.copy()
    cols = set(out.columns)

    for c in ["exit_time", "entry_time", "close_time", "open_time", "time", "timestamp", "datetime", "ts"]:
        if c in cols:
            out[c] = pd.to_datetime(out[c], errors="coerce")
            out = out.dropna(subset=[c]).sort_values(c).reset_index(drop=True)
            return out, c

    if "date" in cols and "time" in cols:
        out["timestamp"] = pd.to_datetime(
            out["date"].astype(str).str.strip() + " " + out["time"].astype(str).str.strip(),
            errors="coerce",
        )
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return out, "timestamp"

    if "date" in cols:
        out["timestamp"] = pd.to_datetime(out["date"].astype(str).str.strip(), errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return out, "timestamp"

    return out, None


def _aggregate_trades_daily(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], int]:
    out = df.copy()
    out, time_col = _select_trade_time_col(out)
    if time_col is None:
        raise ValueError(f"Trades CSV missing time columns. detected_cols={','.join(out.columns)}")

    pnl_col = None
    for c in ["pnl", "net_pnl", "profit"]:
        if c in out.columns:
            pnl_col = c
            break
    if pnl_col is None:
        out["pnl"] = 0.0
        pnl_col = "pnl"

    out["day"] = out[time_col].dt.date.astype(str)
    days = out["day"].unique().tolist()

    daily_rows: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []
    regime_stats: Dict[str, Any] = {"days": []}
    total_trades = 0
    best_day = None
    worst_day = None
    global_balance = 0.0

    has_regime = "regime" in out.columns
    for day in days:
        g = out[out["day"] == day]
        day_pnl = float(pd.to_numeric(g[pnl_col], errors="coerce").fillna(0.0).sum())
        start_equity = float(global_balance)
        end_equity = float(global_balance + day_pnl)

        row = {
            "day": day,
            "bars": 0,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "day_pnl": float(end_equity - start_equity),
            "day_peak": float(max(start_equity, end_equity)),
            "day_min": float(min(start_equity, end_equity)),
            "intraday_dd": float(max(0.0, max(start_equity, end_equity) - end_equity)),
            "dd_stop": False,
            "end_loss_streak": 0,
        }
        daily_rows.append(row)
        eq_row = dict(row)
        eq_row["equity"] = end_equity
        equity_rows.append(eq_row)

        if best_day is None or row["day_pnl"] > best_day["day_pnl"]:
            best_day = dict(row)
        if worst_day is None or row["day_pnl"] < worst_day["day_pnl"]:
            worst_day = dict(row)

        day_trades = int(len(g))
        total_trades += day_trades
        regime_day = {
            "day": day,
            "signals": day_trades,
            "allowed": day_trades,
            "dd_stop": False,
        }
        if has_regime:
            try:
                regime_counts = g["regime"].astype(str).value_counts(dropna=True).to_dict()
                if regime_counts:
                    regime_day["regime_counts"] = regime_counts
            except Exception:
                pass
        regime_stats["days"].append(regime_day)

        global_balance = end_equity

    regime_stats["best_day"] = best_day or {}
    regime_stats["worst_day"] = worst_day or {}
    return daily_rows, equity_rows, regime_stats, int(total_trades)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/live/XAUUSD_M1_live.csv")
    ap.add_argument("--model_path", default="data/ars_best_theta_regime_bank.npz")
    ap.add_argument("--out_dir", default="data/reports")
    ap.add_argument("--power", type=float, default=2.5)
    ap.add_argument("--min_size", type=float, default=0.0)
    ap.add_argument("--max_size", type=float, default=1.0)
    ap.add_argument("--deadzone", type=float, default=0.0)

    ap.add_argument("--meta_risk", type=int, default=0)
    ap.add_argument("--meta_feedback", type=int, default=0)
    ap.add_argument("--meta_state_path", default="data/reports/meta_risk_state.json")
    ap.add_argument("--regime_ledger_span", type=int, default=30)
    ap.add_argument("--regime_ledger_max_days", type=int, default=365)

    ap.add_argument("--daily_dd_limit", type=float, default=200.0)
    ap.add_argument("--max_loss_streak", type=int, default=4)
    ap.add_argument("--spread_spike_cap", type=float, default=300.0)
    ap.add_argument("--spread_open_cap", type=float, default=200.0)
    ap.add_argument("--atr_period", type=int, default=14)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--enable_london", type=int, default=1)
    ap.add_argument("--enable_ny", type=int, default=1)
    ap.add_argument("--london_start", type=int, default=None)
    ap.add_argument("--london_end", type=int, default=None)
    ap.add_argument("--ny_start", type=int, default=None)
    ap.add_argument("--ny_end", type=int, default=None)
    ap.add_argument("--no_session_filter", type=int, default=0)
    ap.add_argument("--no_spread_filter", type=int, default=0)
    ap.add_argument("--legacy_features", type=int, default=0)
    ap.add_argument("--mt5_export", type=int, default=0)
    ap.add_argument("--mt5_out_dir", default=None)
    ap.add_argument("--mt5_format", default="json")
    ap.add_argument("--mt5_mode", default="trades")

    args = ap.parse_args()

    mt5_export = _parse_bool_flag(args.mt5_export, default=False)
    mt5_format = str(args.mt5_format or "json").strip().lower()
    mt5_mode = str(args.mt5_mode or "trades").strip().lower()
    mt5_out_dir = args.mt5_out_dir if args.mt5_out_dir not in (None, "") else args.out_dir

    ensure_dir(args.out_dir)
    out_equity = os.path.join(args.out_dir, "live_daily_equity.csv")
    out_summary = os.path.join(args.out_dir, "live_daily_summary.json")
    out_regime = os.path.join(args.out_dir, "live_daily_regime.json")

    debug_enabled = bool(int(args.debug)) if str(args.debug).strip() != "" else False
    debug_enabled = debug_enabled or (str(os.getenv("LIVE_SIM_DAILY_DEBUG", "")).strip() == "1")

    if mt5_format not in ("json", "csv"):
        mt5_format = "json"
    if mt5_mode not in ("trades", "decisions", "both"):
        mt5_mode = "trades"
    mt5_collect_trades = bool(mt5_export)
    mt5_collect_decisions = bool(mt5_export and mt5_mode in ("decisions", "both"))
    if mt5_export:
        ensure_dir(mt5_out_dir)

    df_raw, detected_sep = read_csv_smart(args.csv, debug=debug_enabled)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    bars_total_raw = int(len(df_raw))
    df, _ = _normalize_columns(df_raw)

    _debug_print(debug_enabled, f"[debug] normalized_cols={list(df.columns)}")

    try:
        mode = detect_input_mode(df)
    except ValueError as e:
        raise ValueError(f"{e} sep={repr(detected_sep)}") from e

    session_settings = _default_session_settings()
    session_settings["enable_london"] = _parse_bool_flag(args.enable_london, default=True)
    session_settings["enable_ny"] = _parse_bool_flag(args.enable_ny, default=True)
    if args.london_start is not None:
        session_settings["london_start"] = int(args.london_start)
    if args.london_end is not None:
        session_settings["london_end"] = int(args.london_end)
    if args.ny_start is not None:
        session_settings["ny_start"] = int(args.ny_start)
    if args.ny_end is not None:
        session_settings["ny_end"] = int(args.ny_end)
    no_session_filter = _parse_bool_flag(args.no_session_filter, default=False)
    no_spread_filter = _parse_bool_flag(args.no_spread_filter, default=False)
    execution_settings = _default_execution_settings()

    if mode == "trades":
        daily_rows, equity_rows, regime_stats, total_trades = _aggregate_trades_daily(df)
        best_day = max(daily_rows, key=lambda r: r["day_pnl"]) if daily_rows else {}
        worst_day = min(daily_rows, key=lambda r: r["day_pnl"]) if daily_rows else {}

        summary = {
            "model_path": args.model_path,
            "mode": "trades_agg",
            "power": float(args.power),
            "execution": execution_settings,
            "session": session_settings,
            "fail_safe": {
                "daily_dd_limit": float(args.daily_dd_limit),
                "max_loss_streak": int(args.max_loss_streak),
                "spread_spike_cap": float(args.spread_spike_cap),
                "spread_open_cap": float(args.spread_open_cap),
            },
            "meta_risk": {
                "enabled": int(args.meta_risk),
                "meta_feedback": int(args.meta_feedback),
                "state_path": args.meta_state_path,
            },
            "days": int(len(daily_rows)),
            "total_trades": int(total_trades),
            "total_pnl": float(sum(r["day_pnl"] for r in daily_rows)),
            "best_day": best_day or {},
            "worst_day": worst_day or {},
            "daily_table": daily_rows,
        }

        if int(args.meta_risk) == 1:
            ledger_summary = update_regime_ledger(
                args.out_dir,
                daily_rows,
                regime_stats,
                span=int(args.regime_ledger_span),
                max_days=int(args.regime_ledger_max_days),
                debug=debug_enabled,
            )
            summary["regime_ledger"] = ledger_summary or {}

        write_equity_csv(out_equity, equity_rows)
        write_summary_json(out_summary, summary)
        write_regime_json(out_regime, regime_stats)

        if mt5_export:
            trades_export = df.copy()
            trades_export, _ = _select_trade_time_col(trades_export)
            no_trades_reason = None
            if int(total_trades) == 0:
                no_trades_reason = "no_trades_in_input"
            no_decisions_reason = None
            if mt5_collect_decisions:
                no_decisions_reason = "decisions_unavailable_in_trades_mode"
            export_mt5_signals(
                out_dir=mt5_out_dir,
                trades_df=trades_export,
                trade_meta=None,
                decisions=None,
                fmt=mt5_format,
                mode=mt5_mode,
                input_csv=args.csv,
                model_path=args.model_path,
                power=float(args.power),
                execution_settings=execution_settings,
                session_settings=session_settings,
                spread_open_cap=float(args.spread_open_cap),
                spread_spike_cap=float(args.spread_spike_cap),
                no_session_filter=bool(no_session_filter),
                no_spread_filter=bool(no_spread_filter),
                no_trades_reason=no_trades_reason,
                no_decisions_reason=no_decisions_reason,
                debug=debug_enabled,
            )

        print(f"Saved: {out_equity} {out_summary} {out_regime}")
        print(f"Days: {len(daily_rows)} Total PnL: {summary['total_pnl']}")
        return

    # --- bars mode ---
    df, time_col = resolve_timestamp(df)
    if time_col is None:
        raise ValueError(
            f"CSV must have time-like column. detected_cols={','.join(df.columns)} sep={repr(detected_sep)}"
        )
    df = _prepare_bars_df(df, time_col=time_col, debug=debug_enabled)
    df["time"] = df[time_col]
    if df["time"].isna().all():
        raise ValueError(
            f"Time column parse failed. col={time_col} detected_cols={','.join(df.columns)} sep={repr(detected_sep)}"
        )

    model = load_regime_bank(args.model_path, debug=debug_enabled)
    model_n_features = _infer_model_n_features(model)

    legacy_features = bool(int(args.legacy_features))
    if legacy_features:
        _debug_print(debug_enabled, "[debug] feature_source=legacy_row_features (debug/backward-compat)")
        X_all = None
    else:
        df, _feat, feature_cols, X_all = _build_feature_pipeline(df, debug=debug_enabled)
        if model_n_features <= 0:
            raise ValueError("Model n_features could not be inferred; check model file")
        data_n_features = int(X_all.shape[1])
        if data_n_features > int(model_n_features):
            if debug_enabled:
                _debug_print(
                    debug_enabled,
                    f"[warn] feature_dim: data={data_n_features} > model={model_n_features}; slicing",
                )
            X_all = X_all[:, : int(model_n_features)]
            feature_cols = feature_cols[: int(model_n_features)]
            data_n_features = int(X_all.shape[1])
        if data_n_features < int(model_n_features):
            raise ValueError(
                "Feature dimension mismatch: FeatureBuilder produced "
                f"{data_n_features} < model_n_features={model_n_features}"
            )
        _debug_print(
            debug_enabled,
            f"[debug] feature_source=TradingEnv feature_dims model={model_n_features} "
            f"data={data_n_features} match={int(data_n_features) == int(model_n_features)}",
        )
    weighter_cfg = ConfidenceWeighterConfig(
        power=float(args.power),
        min_size=float(args.min_size),
        max_size=float(args.max_size),
        deadzone=float(args.deadzone),
    )
    weighter = ConfidenceWeighter(weighter_cfg)

    meta_state = None
    if int(args.meta_risk) == 1:
        mcfg = MetaRiskConfig()
        meta_state = MetaRiskState(mcfg)
        if os.path.exists(args.meta_state_path):
            try:
                meta_state.load(args.meta_state_path)
                print(f"[meta_risk] loaded state from {args.meta_state_path}")
            except Exception as e:
                print(f"[meta_risk] warn: cannot load state: {e}")

    spread_open_cap = float(args.spread_open_cap)
    spread_spike_cap = float(args.spread_spike_cap)
    spread_open_cap_used = spread_open_cap
    spread_spike_cap_used = spread_spike_cap
    if no_spread_filter:
        spread_open_cap_used = 1e9
        spread_spike_cap_used = 1e9

    cfg = {
        "daily_dd_limit": float(args.daily_dd_limit),
        "max_loss_streak": int(args.max_loss_streak),
        "spread_spike_cap": float(spread_spike_cap_used),
        "spread_open_cap": float(spread_open_cap_used),
        "atr_period": int(args.atr_period),
    }
    session_filter = None if no_session_filter else _make_session_filter(session_settings)

    df["day"] = df["time"].dt.date.astype(str)
    days = df["day"].unique().tolist()

    daily_rows: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []
    regime_stats: Dict[str, Any] = {"days": []}

    global_balance = 0.0
    best_day = None
    worst_day = None
    total_trades = 0
    breakdown_totals = {
        "bars_total": int(bars_total_raw),
        "bars_after_time_parse": int(len(df)),
        "bars_after_session_filter": 0,
        "bars_after_spread_filter": 0,
        "signals_total": 0,
        "signals_after_guardrails": 0,
        "final_allowed": 0,
        "total_trades": 0,
        "session_reject": 0,
        "spread_open_reject": 0,
        "spread_spike_reject": 0,
        "guardrail_reject": 0,
        "policy_hold": 0,
        "size_zero": 0,
    }

    mt5_trades_frames: Optional[List[pd.DataFrame]] = [] if mt5_collect_trades else None
    mt5_trade_meta: Optional[List[Dict[str, Any]]] = [] if mt5_collect_trades else None
    mt5_decisions: Optional[List[Dict[str, Any]]] = [] if mt5_collect_decisions else None

    for day in days:
        mask = df["day"] == day
        df_day = df[mask].copy().reset_index(drop=True)
        X_day = None
        if not legacy_features:
            X_day = X_all[mask.values]
        trades_df, equity_df, day_pnl, dbg = run_one_day(
            df_day=df_day,
            model=model,
            weighter=weighter,
            meta_state=meta_state,
            meta_feedback=bool(int(args.meta_feedback)),
            session_filter=session_filter,
            cfg=cfg,
            global_balance=global_balance,
            X_day=X_day,
            n_features=model_n_features,
            legacy_features=legacy_features,
            bypass_session=no_session_filter,
            bypass_spread=no_spread_filter,
            debug=debug_enabled,
            collect_decisions=mt5_collect_decisions,
            collect_trade_meta=mt5_collect_trades,
        )
        if mt5_trades_frames is not None and trades_df is not None and len(trades_df) > 0:
            mt5_trades_frames.append(trades_df)
            if mt5_trade_meta is not None:
                day_meta = dbg.get("trade_meta", [])
                if isinstance(day_meta, list) and day_meta:
                    mt5_trade_meta.extend(day_meta)
        if mt5_decisions is not None:
            day_decisions = dbg.get("decision_rows", [])
            if isinstance(day_decisions, list) and day_decisions:
                mt5_decisions.extend(day_decisions)
        total_trades += int(dbg.get("allowed", 0))
        breakdown_totals["bars_after_session_filter"] += int(dbg.get("bars_after_session_filter", 0))
        breakdown_totals["bars_after_spread_filter"] += int(dbg.get("bars_after_spread_filter", 0))
        breakdown_totals["signals_total"] += int(dbg.get("signals_total", 0))
        breakdown_totals["signals_after_guardrails"] += int(dbg.get("signals_after_guardrails", 0))
        breakdown_totals["final_allowed"] += int(dbg.get("final_allowed", 0))
        breakdown_totals["session_reject"] += int(dbg.get("session_reject", 0))
        breakdown_totals["spread_open_reject"] += int(dbg.get("spread_open_reject", 0))
        breakdown_totals["spread_spike_reject"] += int(dbg.get("spread_spike_reject", 0))
        breakdown_totals["guardrail_reject"] += int(dbg.get("guardrail_reject", 0))
        breakdown_totals["policy_hold"] += int(dbg.get("policy_hold", 0))
        breakdown_totals["size_zero"] += int(dbg.get("size_zero", 0))

        start_equity = float(global_balance)
        end_equity = float(global_balance + day_pnl)

        row = {
            "day": day,
            "bars": int(dbg.get("bars", len(df_day))),
            "start_equity": start_equity,
            "end_equity": end_equity,
            "day_pnl": float(end_equity - start_equity),
            "day_peak": float(dbg.get("equity_peak", max(start_equity, end_equity))),
            "day_min": float(dbg.get("equity_min", min(start_equity, end_equity))),
            "intraday_dd": float(dbg.get("intraday_dd", max(0.0, (dbg.get("equity_peak", start_equity) - end_equity)))),
            "dd_stop": bool(dbg.get("dd_stop", False)),
            "end_loss_streak": int(dbg.get("end_loss_streak", 0)),
        }
        daily_rows.append(row)

        eq_row = dict(row)
        eq_row["equity"] = end_equity
        equity_rows.append(eq_row)

        if best_day is None or row["day_pnl"] > best_day["day_pnl"]:
            best_day = dict(row)
        if worst_day is None or row["day_pnl"] < worst_day["day_pnl"]:
            worst_day = dict(row)

        global_balance = end_equity

        regime_day = {
            "day": day,
            "signals": int(dbg.get("signals", 0)),
            "allowed": int(dbg.get("allowed", 0)),
            "dd_stop": bool(dbg.get("dd_stop", False)),
        }
        regime_counts = dbg.get("regime_counts", {})
        if isinstance(regime_counts, dict) and regime_counts:
            regime_day["regime_counts"] = regime_counts
        regime_stats["days"].append(regime_day)

    if int(total_trades) == 0:
        if int(breakdown_totals.get("signals_total", 0)) == 0:
            reason_hint = "policy never signaled (all hold)"
        elif int(breakdown_totals.get("final_allowed", 0)) == 0:
            reason_hint = "all signals filtered by session/spread/guardrails"
        else:
            reason_hint = "signals allowed but no trades recorded by env"
        if mt5_export:
            trades_export_df = pd.DataFrame()
            if mt5_trades_frames is not None and len(mt5_trades_frames) > 0:
                try:
                    trades_export_df = pd.concat(mt5_trades_frames, ignore_index=True)
                except Exception:
                    trades_export_df = pd.DataFrame()
            no_decisions_reason = None
            if mt5_collect_decisions and (not mt5_decisions):
                no_decisions_reason = "no_decisions_collected"
            export_mt5_signals(
                out_dir=mt5_out_dir,
                trades_df=trades_export_df,
                trade_meta=mt5_trade_meta,
                decisions=mt5_decisions,
                fmt=mt5_format,
                mode=mt5_mode,
                input_csv=args.csv,
                model_path=args.model_path,
                power=float(args.power),
                execution_settings=execution_settings,
                session_settings=session_settings,
                spread_open_cap=float(args.spread_open_cap),
                spread_spike_cap=float(args.spread_spike_cap),
                no_session_filter=bool(no_session_filter),
                no_spread_filter=bool(no_spread_filter),
                no_trades_reason=reason_hint,
                no_decisions_reason=no_decisions_reason,
                debug=debug_enabled,
            )
        breakdown_totals["total_trades"] = int(total_trades)
        full_msg = _build_fail_hard_message(
            input_csv=args.csv,
            model_path=args.model_path,
            bars_loaded=int(len(df)),
            time_col=str(time_col),
            first_ts=df["time"].iloc[0] if len(df) > 0 else None,
            last_ts=df["time"].iloc[-1] if len(df) > 0 else None,
            detected_cols=list(df.columns),
            session=session_settings,
            no_session_filter=bool(no_session_filter),
            spread_open_cap=float(spread_open_cap),
            spread_spike_cap=float(spread_spike_cap),
            no_spread_filter=bool(no_spread_filter),
            daily_dd_limit=float(args.daily_dd_limit),
            max_loss_streak=int(args.max_loss_streak),
            power=float(args.power),
            min_size=float(args.min_size),
            max_size=float(args.max_size),
            deadzone=float(args.deadzone),
            meta_risk=int(args.meta_risk),
            meta_feedback=int(args.meta_feedback),
            meta_state_path=args.meta_state_path,
            breakdown=breakdown_totals,
            reason_hint=reason_hint,
        )
        short_msg = _build_fail_hard_message(
            input_csv=args.csv,
            model_path=args.model_path,
            bars_loaded=int(len(df)),
            time_col=str(time_col),
            first_ts=df["time"].iloc[0] if len(df) > 0 else None,
            last_ts=df["time"].iloc[-1] if len(df) > 0 else None,
            detected_cols=list(df.columns),
            session=session_settings,
            no_session_filter=bool(no_session_filter),
            spread_open_cap=float(spread_open_cap),
            spread_spike_cap=float(spread_spike_cap),
            no_spread_filter=bool(no_spread_filter),
            daily_dd_limit=float(args.daily_dd_limit),
            max_loss_streak=int(args.max_loss_streak),
            power=float(args.power),
            min_size=float(args.min_size),
            max_size=float(args.max_size),
            deadzone=float(args.deadzone),
            meta_risk=int(args.meta_risk),
            meta_feedback=int(args.meta_feedback),
            meta_state_path=args.meta_state_path,
            breakdown=breakdown_totals,
            reason_hint=reason_hint,
            short=True,
        )
        if debug_enabled:
            print(full_msg)
        raise RuntimeError(short_msg)

    summary = {
        "model_path": args.model_path,
        "mode": "regime_bank",
        "power": float(args.power),
        "execution": execution_settings,
        "session": session_settings,
        "fail_safe": {
            "daily_dd_limit": float(args.daily_dd_limit),
            "max_loss_streak": int(args.max_loss_streak),
            "spread_spike_cap": float(args.spread_spike_cap),
            "spread_open_cap": float(args.spread_open_cap),
        },
        "meta_risk": {
            "enabled": int(args.meta_risk),
            "meta_feedback": int(args.meta_feedback),
            "state_path": args.meta_state_path,
        },
        "days": int(len(daily_rows)),
        "total_trades": int(total_trades),
        "total_pnl": float(sum(r["day_pnl"] for r in daily_rows)),
        "best_day": best_day or {},
        "worst_day": worst_day or {},
        "daily_table": daily_rows,
    }

    if int(args.meta_risk) == 1:
        ledger_summary = update_regime_ledger(
            args.out_dir,
            daily_rows,
            regime_stats,
            span=int(args.regime_ledger_span),
            max_days=int(args.regime_ledger_max_days),
            debug=debug_enabled,
        )
        summary["regime_ledger"] = ledger_summary or {}

    write_equity_csv(out_equity, equity_rows)
    write_summary_json(out_summary, summary)
    write_regime_json(out_regime, regime_stats)

    if mt5_export:
        trades_export_df = pd.DataFrame()
        if mt5_trades_frames is not None and len(mt5_trades_frames) > 0:
            try:
                trades_export_df = pd.concat(mt5_trades_frames, ignore_index=True)
            except Exception:
                trades_export_df = pd.DataFrame()
        no_decisions_reason = None
        if mt5_collect_decisions and (not mt5_decisions):
            no_decisions_reason = "no_decisions_collected"
        export_mt5_signals(
            out_dir=mt5_out_dir,
            trades_df=trades_export_df,
            trade_meta=mt5_trade_meta,
            decisions=mt5_decisions,
            fmt=mt5_format,
            mode=mt5_mode,
            input_csv=args.csv,
            model_path=args.model_path,
            power=float(args.power),
            execution_settings=execution_settings,
            session_settings=session_settings,
            spread_open_cap=float(args.spread_open_cap),
            spread_spike_cap=float(args.spread_spike_cap),
            no_session_filter=bool(no_session_filter),
            no_spread_filter=bool(no_spread_filter),
            no_trades_reason=None,
            no_decisions_reason=no_decisions_reason,
            debug=debug_enabled,
        )

    print(f"Saved: {out_equity} {out_summary} {out_regime}")
    print(f"Days: {len(daily_rows)} Total PnL: {summary['total_pnl']}")


# How to test:
#   (a) Bars mode (MT5 CSV):
#       python -m scripts.live_sim_daily --csv data/live/XAUUSD_M1_live.csv --meta_risk 0 --meta_feedback 0
#   (b) Trades aggregation mode:
#       python -m scripts.live_sim_daily --csv data/reports/live_sim_trades.csv --meta_risk 0 --meta_feedback 0

if __name__ == "__main__":
    main()
