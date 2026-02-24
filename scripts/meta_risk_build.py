from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from rabit.rl.meta_risk import MetaRiskConfig, MetaRiskState


def _ensure_dataframe(obj: Any) -> pd.DataFrame:
    """
    Ensure we end up with a pandas DataFrame.
    Accept:
    - DataFrame
    - list[dict]
    - dict of lists
    - Ledger-like objects (common patterns: to_frame(), records, trades, rows)
    """
    if isinstance(obj, pd.DataFrame):
        return obj

    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        return pd.DataFrame(obj)

    if isinstance(obj, dict):
        return pd.DataFrame(obj)

    if hasattr(obj, "to_frame") and callable(getattr(obj, "to_frame")):
        df = obj.to_frame()
        if isinstance(df, pd.DataFrame):
            return df

    for attr in ("records", "trades", "rows"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            try:
                return pd.DataFrame(val)
            except Exception:
                pass

    raise TypeError(f"Trades input cannot be converted to DataFrame. Got: {type(obj).__name__}")


def _read_trades_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trades file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep="\t")
        return _ensure_dataframe(df)

    if ext in (".json", ".jsonl"):
        if ext == ".jsonl":
            rows: List[dict] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return _ensure_dataframe(rows)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "trades" in data:
            data = data["trades"]
        return _ensure_dataframe(data)

    raise ValueError(f"Unsupported trades file extension: {ext}. Use .csv/.tsv/.json/.jsonl")


def _resolve_column(
    df: pd.DataFrame,
    label: str,
    candidates: List[str],
    required: bool = True,
    default: Optional[str] = None,
) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if not required:
        return default
    raise ValueError(
        f"Could not find {label} column. Tried: {', '.join(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def _make_ts(df: pd.DataFrame, ts_col: Optional[str]) -> List[str]:
    if ts_col is None or ts_col not in df.columns:
        return [f"row_{i}" for i in range(len(df))]
    return df[ts_col].astype(str).fillna("").tolist()


def _to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.fillna(0.0)
    return s.astype(float)


def _guard_trades_df(df: pd.DataFrame, path: str) -> None:
    """
    Detect common corruption: trades.csv overwritten by `str(ledger)` causing
    0 rows and 1 weird column like "<rabit.env.ledger.Ledger object at ...>"
    """
    if df is None:
        raise RuntimeError("Trades DF is None")

    if df.shape[0] == 0 and df.shape[1] == 1:
        col0 = str(df.columns[0])
        if "Ledger object" in col0 or "rabit.env.ledger" in col0:
            raise RuntimeError(
                "Trades CSV looks corrupted (was likely overwritten by a Ledger string). "
                f"File: {path}\n"
                "Fix: re-run `python -m scripts.live_sim` to regenerate "
                "`data/reports/live_sim_trades.csv` as a real CSV table."
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="Path to trades CSV/JSON/JSONL")
    ap.add_argument("--out_dir", default="data/reports", help="Output directory")

    ap.add_argument("--min_trades_per_regime", type=int, default=50)
    ap.add_argument("--ewma_alpha", type=float, default=0.05)
    ap.add_argument("--score_clip", type=float, default=3.0)
    ap.add_argument("--min_scale", type=float, default=0.25)
    ap.add_argument("--max_scale", type=float, default=1.25)
    ap.add_argument("--k", type=float, default=1.0)

    ap.add_argument("--regime_col", default="", help="Override regime column name")
    ap.add_argument("--return_col", default="", help="Override return column name")
    ap.add_argument("--ts_col", default="", help="Override timestamp column name")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_raw = _read_trades_any(args.trades)
    df = _ensure_dataframe(df_raw)
    df.columns = [str(c).strip() for c in df.columns]

    _guard_trades_df(df, args.trades)

    regime_col = args.regime_col.strip() or _resolve_column(
        df,
        "regime",
        ["regime", "regime_label", "regime_name", "market_regime"],
        required=False,
        default=None,
    )

    return_col = args.return_col.strip()
    if not return_col:
        return_col = _resolve_column(
            df,
            "return",
            ["pnl", "net_pnl", "profit", "pnl_usd", "pnl_points", "ret", "return"],
            required=True,
        )
    if return_col not in df.columns:
        raise ValueError(f"return_col='{return_col}' not found. columns={list(df.columns)}")

    ts_col = args.ts_col.strip() or _resolve_column(
        df,
        "timestamp",
        ["exit_time", "exit_ts", "close_ts", "ts", "timestamp", "time"],
        required=False,
        default=None,
    )

    cfg = MetaRiskConfig(
        min_trades_per_regime=int(args.min_trades_per_regime),
        ewma_alpha=float(args.ewma_alpha),
        score_clip=float(args.score_clip),
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
        k=float(args.k),
    )
    state = MetaRiskState(cfg)

    regimes = (
        df[regime_col].astype(str).fillna("unknown")
        if regime_col and regime_col in df.columns
        else pd.Series(["unknown"] * len(df))
    )
    rets = _to_float_series(df, return_col)
    tss = _make_ts(df, ts_col)

    for i in range(len(df)):
        regime = str(regimes.iloc[i]) if hasattr(regimes, "iloc") else str(regimes[i])
        pnl_return = float(rets.iloc[i])
        ts = str(tss[i])
        if not regime or regime == "nan":
            regime = "unknown"
        state.update_trade(regime=regime, pnl_return=pnl_return, ts=ts)

    out_state = os.path.join(args.out_dir, "meta_risk_state.json")
    state.save(out_state)

    stats_map = getattr(state, "stats", None)
    if stats_map is None:
        stats_map = getattr(state, "regimes", None)
    if stats_map is None:
        raise AttributeError("MetaRiskState has no 'stats' mapping. Please expose state.stats dict in TASK-3A.")

    per_regime: List[Dict[str, Any]] = []
    total_trades = 0

    for regime, st in stats_map.items():
        total_trades += int(getattr(st, "n_trades", 0))
        per_regime.append(
            {
                "regime": str(regime),
                "n_trades": int(getattr(st, "n_trades", 0)),
                "ewma_return": float(getattr(st, "ewma_return", 0.0)),
                "ewma_vol": float(getattr(st, "ewma_vol", 0.0)),
                "ewma_winrate": float(getattr(st, "ewma_winrate", 0.0)),
                "meta_scale": float(state.meta_scale(str(regime))),
                "last_update_ts": getattr(st, "last_update_ts", None),
            }
        )

    per_regime.sort(key=lambda x: x["n_trades"], reverse=True)

    summary = {
        "source_trades": args.trades,
        "rows": int(len(df)),
        "config": asdict(cfg),
        "resolved_columns": {
            "regime_col": regime_col,
            "return_col": return_col,
            "ts_col": ts_col,
        },
        "per_regime": per_regime,
        "global": {"regimes": int(len(per_regime)), "total_trades": int(total_trades)},
    }

    out_summary = os.path.join(args.out_dir, "meta_risk_summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[meta_risk_build] rows={len(df)} regimes={len(per_regime)} total_trades={total_trades}")
    print("[meta_risk_build] top regimes:")
    for row in per_regime[:5]:
        print(f"  - {row['regime']}: n={row['n_trades']} meta_scale={row['meta_scale']:.4f}")
    print(f"[meta_risk_build] saved: {out_state}")
    print(f"[meta_risk_build] saved: {out_summary}")


if __name__ == "__main__":
    main()