from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import pandas as pd


LOADER_VERSION = "v0.3.0-unblock-index"


@dataclass(frozen=True)
class DataIntegrityReport:
    rows: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    duplicates: int
    missing_bars: int
    expected_freq: str
    max_gap_minutes: int
    gaps_top5: list[tuple[str, int]]


class MT5DataLoader:
    """
    MT5 TSV with header:
      <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    """

    def __init__(self, expected_freq: str = "1min", tz: Optional[str] = None, debug: bool = False) -> None:
        self.expected_freq = expected_freq
        self.tz = tz
        self.debug = debug

    @staticmethod
    def _normalize_columns(cols) -> list[str]:
        out = []
        for c in cols:
            c2 = str(c).strip().replace("\ufeff", "")
            c2 = c2.replace("<", "").replace(">", "")
            out.append(c2.upper())
        return out

    @staticmethod
    def _read_mt5_tsv(path: str) -> pd.DataFrame:
        # Read all as string -> we control numeric parsing
        df = pd.read_csv(
            path,
            sep="\t",
            header=0,
            engine="python",
            encoding="utf-8-sig",
            dtype=str,
        )
        df.columns = MT5DataLoader._normalize_columns(df.columns)
        return df

    @staticmethod
    def _best_parse_datetime(dt_str: pd.Series) -> pd.Series:
        s = dt_str.astype(str).str.strip()

        ts_fmt = pd.to_datetime(s, format="%Y.%m.%d %H:%M:%S", errors="coerce")
        nat_fmt = float(ts_fmt.isna().mean()) if len(ts_fmt) else 1.0

        ts_inf = pd.to_datetime(s, errors="coerce")
        nat_inf = float(ts_inf.isna().mean()) if len(ts_inf) else 1.0

        return ts_inf if nat_inf < nat_fmt else ts_fmt

    @staticmethod
    def _clean_numeric_str(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        s = s.str.replace("\u00A0", "", regex=False)  # NBSP
        s = s.str.strip()
        s = s.str.replace(" ", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
        return s

    @classmethod
    def _to_float(cls, series: pd.Series) -> pd.Series:
        s = cls._clean_numeric_str(series)
        return pd.to_numeric(s, errors="coerce")

    @classmethod
    def _to_int(cls, series: pd.Series, default: int = 0) -> pd.Series:
        s = cls._clean_numeric_str(series)
        out = pd.to_numeric(s, errors="coerce").fillna(default)
        return out.astype("int64")

    def load_m1(self, path: str) -> pd.DataFrame:
        df_raw = self._read_mt5_tsv(path)

        required = ["DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE"]
        for c in required:
            if c not in df_raw.columns:
                raise ValueError(f"Missing required column '{c}'. Got columns={list(df_raw.columns)}")

        if "TICKVOL" not in df_raw.columns:
            df_raw["TICKVOL"] = "0"
        if "SPREAD" not in df_raw.columns:
            df_raw["SPREAD"] = "0"

        dt_str = df_raw["DATE"].astype(str).str.strip() + " " + df_raw["TIME"].astype(str).str.strip()
        ts = self._best_parse_datetime(dt_str)

        open_ = self._to_float(df_raw["OPEN"])
        high = self._to_float(df_raw["HIGH"])
        low = self._to_float(df_raw["LOW"])
        close = self._to_float(df_raw["CLOSE"])

        tickvol = self._to_int(df_raw["TICKVOL"], default=0)
        spread = self._to_int(df_raw["SPREAD"], default=0)

        if self.tz is not None:
            ts = ts.dt.tz_localize(self.tz, ambiguous="infer", nonexistent="shift_forward")

        # Build with numpy arrays to avoid any index alignment issues
        df = pd.DataFrame(
            {
                "open": open_.to_numpy(),
                "high": high.to_numpy(),
                "low": low.to_numpy(),
                "close": close.to_numpy(),
                "tickvol": tickvol.to_numpy(),
                "spread": spread.to_numpy(),
            }
        )

        # ✅ IMPORTANT: set index using DatetimeIndex(ts) (most stable)
        df.index = pd.DatetimeIndex(ts.to_numpy(), name="timestamp")

        # DO NOT drop rows here (unblock); let downstream feature builder dropna if needed
        # Just sort + dedup.
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if self.debug:
            print(f"[LOADER DEBUG] loader_version: {LOADER_VERSION}")
            print("[LOADER DEBUG] dt_str head:", dt_str.head(3).tolist())
            print("[LOADER DEBUG] ts head:", ts.head(3).tolist())
            print("[LOADER DEBUG] NaT ratio:", float(pd.isna(ts).mean()))
            print("[LOADER DEBUG] OPEN numeric sample:", open_.head(3).tolist())
            print(
                "[LOADER DEBUG] OHLC NaN ratios:",
                float(open_.isna().mean()),
                float(high.isna().mean()),
                float(low.isna().mean()),
                float(close.isna().mean()),
            )
            print("[LOADER DEBUG] index isna ratio:", float(df.index.isna().mean()))
            print("[LOADER DEBUG] final df shape:", df.shape)
            print("[LOADER DEBUG] final head:", df.head(2))

        return df

    def integrity_report(self, df: pd.DataFrame) -> DataIntegrityReport:
        if df.empty:
            return DataIntegrityReport(
                rows=0,
                start=None,
                end=None,
                duplicates=0,
                missing_bars=0,
                expected_freq=self.expected_freq,
                max_gap_minutes=0,
                gaps_top5=[],
            )

        idx = df.index
        rows = len(df)
        start = idx[0]
        end = idx[-1]
        duplicates = int(idx.duplicated().sum())

        expected = pd.date_range(start=start, end=end, freq=self.expected_freq)
        missing = int(len(expected.difference(idx)))

        diffs = idx.to_series().diff().dropna()
        diffs_min = (diffs / pd.Timedelta(minutes=1)).astype("int64")
        max_gap = int(diffs_min.max()) if len(diffs_min) else 0

        gaps = diffs_min[diffs_min > 1].sort_values(ascending=False).head(5)
        gaps_top5: list[Tuple[str, int]] = []
        if not gaps.empty:
            for ts_gap, gap in gaps.items():
                gaps_top5.append((ts_gap.isoformat(), int(gap)))

        return DataIntegrityReport(
            rows=rows,
            start=start,
            end=end,
            duplicates=duplicates,
            missing_bars=missing,
            expected_freq=self.expected_freq,
            max_gap_minutes=max_gap,
            gaps_top5=gaps_top5,
        )

    @staticmethod
    def to_numpy_ready(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out[["open", "high", "low", "close", "tickvol", "spread"]]
        out["open"] = out["open"].astype("float64")
        out["high"] = out["high"].astype("float64")
        out["low"] = out["low"].astype("float64")
        out["close"] = out["close"].astype("float64")
        out["tickvol"] = out["tickvol"].astype("int64")
        out["spread"] = out["spread"].astype("int64")
        return out

    @staticmethod
    def summarize(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"rows": 0}
        return {
            "rows": int(len(df)),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "spread_min": int(df["spread"].min()),
            "spread_p50": int(df["spread"].median()),
            "spread_p95": int(df["spread"].quantile(0.95)),
            "spread_max": int(df["spread"].max()),
            "tickvol_p50": int(df["tickvol"].median()),
            "tickvol_p95": int(df["tickvol"].quantile(0.95)),
        }