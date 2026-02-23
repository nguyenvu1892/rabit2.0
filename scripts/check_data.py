from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import pandas as pd

import rabit.data.loader as loader_mod
from rabit.data.loader import MT5DataLoader
from rabit.data.resampler import ResampleConfig, resample_ohlcv_m1_to_higher
from rabit.data.feature_builder import FeatureBuilder, FeatureConfig


def _json_safe(obj: Any) -> Any:
    """Convert pandas/numpy objects into JSON-serializable primitives."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    # pandas Index, Series, numpy scalars
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True, help="Path to MT5 M1 file (tsv/csv)")
    ap.add_argument("--out_json", default="", help="Optional output json path")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    if args.debug:
        print("=== PYTHON ===")
        print("sys.executable:", sys.executable)
        print("sys.version:", sys.version)
        print("cwd:", __import__("os").getcwd())
        print()
        print("=== IMPORT CHECK ===")
        print("loader module file:", loader_mod.__file__)
        print()

    loader = MT5DataLoader(expected_freq="1min", tz=None, debug=args.debug)
    df_m1 = loader.load_m1(args.m1)
    df_m1 = loader.to_numpy_ready(df_m1)

    rep = loader.integrity_report(df_m1)
    summ = loader.summarize(df_m1)

    df_m5 = resample_ohlcv_m1_to_higher(df_m1, ResampleConfig(rule="5min", spread_agg="max"))

    fb = FeatureBuilder(FeatureConfig(dropna=True))
    feat_m1 = fb.build(df_m1, prefix="m1_")
    feat_m5 = fb.build(df_m5, prefix="m5_")
    feat = fb.align_mtf(feat_m1, feat_m5, suffix_m5="mtf_")

    report = {
        "summary_m1": summ,
        "integrity_m1": rep.__dict__,
        "rows_m5": int(len(df_m5)),
        "features_m1_rows": int(len(feat_m1)),
        "features_m5_rows": int(len(feat_m5)),
        "features_mtf_rows": int(len(feat)),
        "features_mtf_cols": int(feat.shape[1]),
        "first_feat_ts": feat.index[0].isoformat() if len(feat) else None,
        "last_feat_ts": feat.index[-1].isoformat() if len(feat) else None,
    }

    report = _json_safe(report)

    print(json.dumps(report, indent=2))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()