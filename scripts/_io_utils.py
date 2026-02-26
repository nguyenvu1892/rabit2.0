from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_write_json(
    path: str,
    payload: Any,
    ensure_ascii: bool = False,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)


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


def read_csv_sniff(path: str, debug: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Alias for read_csv_smart (kept for clarity around separator sniffing).
    """
    return read_csv_smart(path, debug=debug)


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
