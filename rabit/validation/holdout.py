from __future__ import annotations

import math
import os
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from rabit.state import atomic_io
from scripts import live_sim_daily as live

DEFAULT_OUT_DIR = os.path.join("data", "reports", "holdout")
DEFAULT_META_STATE_PATH = os.path.join(
    "data", "meta_states", "current_approved", "meta_risk_state.json"
)

REPORT_FILENAME = "holdout_report.json"
EQUITY_FILENAME = "holdout_equity.csv"
SUMMARY_FILENAME = "holdout_summary.json"

EXIT_PASS = 0
EXIT_REJECT = 10
EXIT_INVALID_INPUT = 20
EXIT_RUNTIME_FAIL = 30


class HoldoutInputError(RuntimeError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)


@dataclass(frozen=True)
class HoldoutConfig:
    csv: str
    out_dir: str = DEFAULT_OUT_DIR
    holdout_days: int = 30
    min_bars: int = 500
    strict: int = 1
    deterministic_check: int = 1
    meta_risk: int = 1
    meta_feedback: int = 1
    risk_per_trade: float = 0.02
    account_equity_start: float = 500.0
    meta_state_path: str = DEFAULT_META_STATE_PATH


@dataclass(frozen=True)
class HoldoutSplit:
    mode: str
    bars_total: int
    bars_holdout: int
    days_total: int
    days_holdout: int
    train_bars: int
    holdout_start_day: str
    start_ts: str
    end_ts: str


def _is_enabled(flag: Any, *, default: bool) -> bool:
    try:
        return bool(int(flag))
    except Exception:
        return bool(default)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _stable_float(value: float, digits: int = 10) -> float:
    return float(f"{float(value):.{int(digits)}f}")


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _stable_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stable_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [_stable_json_value(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return _stable_float(value)
    return value


def stable_json_dumps(data: Any) -> str:
    import json

    payload = _stable_json_value(data)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def sha256_text(text: str) -> str:
    h = sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def sha256_file(path: str) -> str:
    if not path or not os.path.exists(path):
        return "missing"
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json_canonical(path: str, payload: Dict[str, Any]) -> None:
    text = stable_json_dumps(payload)
    atomic_io.atomic_write_text(path, text, suffix=".json", create_backup=True)


def atomic_write_csv_deterministic(path: str, df: pd.DataFrame) -> None:
    csv_text = df.to_csv(index=False, lineterminator="\n", float_format="%.10f")
    atomic_io.atomic_write_text(path, csv_text, suffix=".csv", create_backup=True)


def _resolve_mode_and_time(df_norm: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    mode = live.detect_input_mode(df_norm)
    if mode == "bars":
        resolved, time_col = live.resolve_timestamp(df_norm)
    elif mode == "trades":
        resolved, time_col = live._select_trade_time_col(df_norm)
    else:
        raise HoldoutInputError("UNSUPPORTED_MODE", f"Unsupported input mode: {mode}")

    if time_col is None:
        raise HoldoutInputError(
            "MISSING_TIMESTAMP",
            f"Unable to resolve timestamp column. detected_cols={list(df_norm.columns)}",
        )

    out = resolved.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if len(out) == 0:
        raise HoldoutInputError("NO_VALID_TIMESTAMPS", "No rows with valid timestamps after parsing.")
    return out, str(time_col), str(mode)


def split_holdout_window(config: HoldoutConfig) -> Tuple[pd.DataFrame, HoldoutSplit]:
    if not config.csv or not os.path.exists(config.csv):
        raise HoldoutInputError("MISSING_CSV", f"CSV file not found: {config.csv}")

    holdout_days = int(config.holdout_days)
    if holdout_days <= 0:
        raise HoldoutInputError("INVALID_HOLDOUT_DAYS", "--holdout_days must be > 0")

    min_bars = int(config.min_bars)
    if min_bars <= 0:
        raise HoldoutInputError("INVALID_MIN_BARS", "--min_bars must be > 0")

    df_raw, _ = live.read_csv_smart(config.csv, debug=False)
    if len(df_raw) < min_bars:
        raise HoldoutInputError(
            "INSUFFICIENT_BARS",
            f"Need at least {min_bars} rows, found {len(df_raw)}",
        )

    df_norm, _ = live._normalize_columns(df_raw.copy())
    df_norm["__row_id__"] = pd.RangeIndex(start=0, stop=len(df_norm), step=1, dtype="int64")

    resolved, time_col, mode = _resolve_mode_and_time(df_norm)
    bars_total = int(len(resolved))
    if bars_total < min_bars:
        raise HoldoutInputError(
            "INSUFFICIENT_BARS_AFTER_PARSE",
            f"Need at least {min_bars} timestamp-valid rows, found {bars_total}",
        )

    resolved["__day__"] = resolved[time_col].dt.normalize()
    unique_days = sorted(resolved["__day__"].dropna().unique().tolist())
    days_total = int(len(unique_days))
    if days_total == 0:
        raise HoldoutInputError("NO_VALID_DAYS", "No valid calendar days detected in input.")

    holdout_start_day = pd.Timestamp(unique_days[max(0, days_total - holdout_days)])
    holdout_mask = resolved["__day__"] >= holdout_start_day
    holdout_ids = sorted(set(resolved.loc[holdout_mask, "__row_id__"].astype(int).tolist()))
    train_ids = sorted(set(resolved.loc[~holdout_mask, "__row_id__"].astype(int).tolist()))

    bars_holdout = int(len(holdout_ids))
    train_bars = int(len(train_ids))
    if bars_holdout <= 0:
        raise HoldoutInputError("HOLDOUT_EMPTY", "Computed holdout window is empty.")
    if train_bars <= 0:
        raise HoldoutInputError(
            "TRAIN_EMPTY",
            "Train window is empty; increase dataset size or reduce --holdout_days.",
        )

    holdout_df = df_raw.iloc[holdout_ids].copy().reset_index(drop=True)
    days_holdout = int(resolved.loc[holdout_mask, "__day__"].nunique())
    start_ts = resolved[time_col].iloc[0].isoformat()
    end_ts = resolved[time_col].iloc[-1].isoformat()

    split = HoldoutSplit(
        mode=mode,
        bars_total=bars_total,
        bars_holdout=bars_holdout,
        days_total=days_total,
        days_holdout=days_holdout,
        train_bars=train_bars,
        holdout_start_day=holdout_start_day.date().isoformat(),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    return holdout_df, split


def _build_live_args(
    config: HoldoutConfig,
    holdout_csv_path: str,
    holdout_out_dir: str,
) -> Tuple[Any, str]:
    args = live._build_arg_parser().parse_args([])
    args.csv = holdout_csv_path
    args.out_dir = holdout_out_dir
    args.meta_risk = int(config.meta_risk)
    args.meta_feedback = int(config.meta_feedback)
    args.risk_per_trade = float(config.risk_per_trade)
    args.account_equity_start = float(config.account_equity_start)
    args.meta_state_path = str(config.meta_state_path)
    args.deterministic_check = 0
    args.mt5_export = 0
    args.debug = 0
    model_path = str(args.model_path)
    return args, model_path


def _run_live_on_holdout(
    config: HoldoutConfig,
    holdout_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    with tempfile.TemporaryDirectory(prefix="holdout_live_") as tmp_root:
        holdout_csv_path = os.path.join(tmp_root, "holdout_input.csv")
        live_out_dir = os.path.join(tmp_root, "out")
        os.makedirs(live_out_dir, exist_ok=True)
        atomic_write_csv_deterministic(holdout_csv_path, holdout_df)

        args, model_path = _build_live_args(
            config=config,
            holdout_csv_path=holdout_csv_path,
            holdout_out_dir=live_out_dir,
        )
        live._run_live_sim_daily_once(
            args,
            deterministic_enabled=False,
            write_outputs=True,
            read_only_state=True,
        )

        summary_path = os.path.join(live_out_dir, "live_daily_summary.json")
        equity_path = os.path.join(live_out_dir, "live_daily_equity.csv")
        summary_payload, _ = atomic_io.load_json_with_fallback(summary_path)
        if not isinstance(summary_payload, dict):
            raise RuntimeError(f"Invalid live summary payload type: {type(summary_payload)}")
        equity_df = pd.read_csv(equity_path)
        return summary_payload, equity_df, model_path


def _equity_series(equity_df: pd.DataFrame) -> pd.Series:
    for col in ("equity", "end_equity", "balance"):
        if col in equity_df.columns:
            s = pd.to_numeric(equity_df[col], errors="coerce")
            return s.dropna()
    return pd.Series(dtype="float64")


def compute_metrics(
    summary_payload: Dict[str, Any],
    equity_df: pd.DataFrame,
    account_equity_start: float,
) -> Dict[str, Optional[float]]:
    total_pnl = _safe_float(summary_payload.get("total_pnl"), 0.0) or 0.0
    pnl_pct = None
    start_equity = _safe_float(account_equity_start, None)
    if start_equity is not None and abs(start_equity) > 1e-12:
        pnl_pct = (float(total_pnl) / float(start_equity)) * 100.0

    eq = _equity_series(equity_df)
    max_dd_pct = None
    if len(eq) > 0:
        peaks = eq.cummax()
        dd_pct = ((eq - peaks) / peaks.replace(0.0, pd.NA)) * 100.0
        dd_min = pd.to_numeric(dd_pct, errors="coerce").min()
        if dd_min is not None and not pd.isna(dd_min):
            max_dd_pct = abs(float(dd_min))

    winrate = None
    for key in ("winrate", "win_rate", "trade_winrate"):
        candidate = _safe_float(summary_payload.get(key), None)
        if candidate is not None:
            winrate = candidate
            break

    trades_count = _safe_float(summary_payload.get("total_trades"), None)
    return {
        "total_pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "max_dd_pct": max_dd_pct,
        "winrate": winrate,
        "trades_count": trades_count,
    }


def _report_hash_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(report)
    det_hashes = payload.get("deterministic_hashes")
    if isinstance(det_hashes, dict):
        copied = dict(det_hashes)
        copied.pop("report_hash", None)
        payload["deterministic_hashes"] = copied
    return payload


def _report_paths(out_dir: str) -> Dict[str, str]:
    base = os.path.normpath(out_dir)
    return {
        "holdout_report_json": os.path.normpath(os.path.join(base, REPORT_FILENAME)),
        "holdout_equity_csv": os.path.normpath(os.path.join(base, EQUITY_FILENAME)),
        "holdout_summary_json": os.path.normpath(os.path.join(base, SUMMARY_FILENAME)),
    }


def _base_metrics() -> Dict[str, Optional[float]]:
    return {
        "total_pnl": None,
        "pnl_pct": None,
        "max_dd_pct": None,
        "winrate": None,
        "trades_count": None,
    }


def _base_dataset_stats() -> Dict[str, Any]:
    return {
        "bars_total": 0,
        "bars_holdout": 0,
        "days_total": 0,
        "days_holdout": 0,
        "start_ts": None,
        "end_ts": None,
    }


def _config_payload(config: HoldoutConfig, model_path: str) -> Dict[str, Any]:
    payload = asdict(config)
    payload["model_path"] = str(model_path)
    return payload


def run_holdout_validation(config: HoldoutConfig) -> Tuple[int, Dict[str, Any]]:
    os.makedirs(config.out_dir, exist_ok=True)

    artifact_paths = _report_paths(config.out_dir)
    summary_path = artifact_paths["holdout_summary_json"]
    equity_path = artifact_paths["holdout_equity_csv"]
    report_path = artifact_paths["holdout_report_json"]

    status = "FAIL"
    reason_code = "RUNTIME_EXCEPTION"
    exit_code = EXIT_RUNTIME_FAIL
    error_message: Optional[str] = None
    dataset_stats = _base_dataset_stats()
    metrics = _base_metrics()
    model_path = str(live._build_arg_parser().parse_args([]).model_path)

    try:
        if int(config.meta_risk) == 1 and _is_enabled(config.strict, default=True):
            if not os.path.exists(config.meta_state_path):
                raise HoldoutInputError(
                    "META_STATE_MISSING",
                    f"Meta state not found: {config.meta_state_path}",
                )

        holdout_df, split = split_holdout_window(config)
        dataset_stats = {
            "bars_total": int(split.bars_total),
            "bars_holdout": int(split.bars_holdout),
            "days_total": int(split.days_total),
            "days_holdout": int(split.days_holdout),
            "start_ts": split.start_ts,
            "end_ts": split.end_ts,
        }

        summary_payload, equity_df, model_path = _run_live_on_holdout(config, holdout_df)
        atomic_write_csv_deterministic(equity_path, equity_df)
        atomic_write_json_canonical(summary_path, summary_payload)

        metrics = compute_metrics(
            summary_payload=summary_payload,
            equity_df=equity_df,
            account_equity_start=float(config.account_equity_start),
        )
        status = "PASS"
        reason_code = "OK"
        exit_code = EXIT_PASS
    except HoldoutInputError as exc:
        reason_code = str(exc.reason_code)
        status = "FAIL"
        exit_code = EXIT_INVALID_INPUT
        error_message = str(exc)
    except Exception as exc:
        reason_code = "RUNTIME_EXCEPTION"
        status = "FAIL"
        exit_code = EXIT_RUNTIME_FAIL
        error_message = f"{type(exc).__name__}: {exc}"

    report: Dict[str, Any] = {
        "status": status,
        "reason_code": reason_code,
        "config": _config_payload(config, model_path=model_path),
        "dataset_stats": dataset_stats,
        "metrics": metrics,
        "artifact_paths": artifact_paths,
    }
    if error_message:
        report["error_message"] = error_message

    if _is_enabled(config.deterministic_check, default=True):
        det_hashes = {
            "input_hash": sha256_file(config.csv),
            "equity_hash": sha256_file(equity_path),
        }
        report["deterministic_hashes"] = det_hashes
        det_hashes["report_hash"] = sha256_text(stable_json_dumps(_report_hash_payload(report)))

    atomic_write_json_canonical(report_path, report)
    return int(exit_code), report
