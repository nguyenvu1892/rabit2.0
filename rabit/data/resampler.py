from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


SpreadAgg = Literal["last", "max", "median"]


@dataclass(frozen=True)
class ResampleConfig:
    rule: str = "5min"
    label: str = "right"
    closed: str = "right"
    spread_agg: SpreadAgg = "max"
    dropna: bool = True


def resample_ohlcv_m1_to_higher(df_m1: pd.DataFrame, cfg: ResampleConfig) -> pd.DataFrame:
    if df_m1.empty:
        return df_m1.copy()

    ohlc = df_m1[["open", "high", "low", "close"]].resample(
        cfg.rule, label=cfg.label, closed=cfg.closed
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last"})

    tickvol = df_m1["tickvol"].resample(cfg.rule, label=cfg.label, closed=cfg.closed).sum().rename("tickvol")

    if cfg.spread_agg == "last":
        spread = df_m1["spread"].resample(cfg.rule, label=cfg.label, closed=cfg.closed).last().rename("spread")
    elif cfg.spread_agg == "median":
        spread = df_m1["spread"].resample(cfg.rule, label=cfg.label, closed=cfg.closed).median().rename("spread")
    else:
        spread = df_m1["spread"].resample(cfg.rule, label=cfg.label, closed=cfg.closed).max().rename("spread")

    out = pd.concat([ohlc, tickvol, spread], axis=1)

    if cfg.dropna:
        out = out.dropna(subset=["open", "high", "low", "close"]).copy()

    out["tickvol"] = out["tickvol"].fillna(0).astype("int64")
    out["spread"] = out["spread"].ffill().fillna(0).astype("int64")
    return out