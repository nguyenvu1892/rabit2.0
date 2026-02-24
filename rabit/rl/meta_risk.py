from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class MetaRiskConfig:
    min_trades_per_regime: int = 50
    ewma_alpha: float = 0.05
    score_clip: float = 3.0
    min_scale: float = 0.25
    max_scale: float = 1.25
    k: float = 1.0


@dataclass
class RegimeStats:
    n_trades: int = 0
    ewma_return: float = 0.0
    ewma_vol: float = 0.0
    ewma_winrate: float = 0.0
    last_update_ts: Optional[str] = None


class MetaRiskState:
    def __init__(self, cfg: MetaRiskConfig) -> None:
        self.cfg = cfg
        self.regimes: Dict[str, RegimeStats] = {}

    def update_trade(self, regime: str, pnl_return: float, ts: str) -> None:
        stats = self.regimes.get(regime)
        if stats is None:
            stats = RegimeStats()
            self.regimes[regime] = stats

        alpha = self.cfg.ewma_alpha
        stats.ewma_return = (1.0 - alpha) * stats.ewma_return + alpha * pnl_return
        stats.ewma_vol = (1.0 - alpha) * stats.ewma_vol + alpha * abs(pnl_return)
        win = 1.0 if pnl_return > 0.0 else 0.0
        stats.ewma_winrate = (1.0 - alpha) * stats.ewma_winrate + alpha * win
        stats.n_trades += 1
        stats.last_update_ts = ts

    def meta_scale(self, regime: str) -> float:
        stats = self.regimes.get(regime)
        if stats is None:
            return 1.0
        if stats.n_trades < self.cfg.min_trades_per_regime:
            return 1.0

        score = stats.ewma_return / (stats.ewma_vol + 1e-8)
        score = max(-self.cfg.score_clip, min(self.cfg.score_clip, score))
        s = 1.0 / (1.0 + math.exp(-self.cfg.k * score))
        scale = self.cfg.min_scale + s * (self.cfg.max_scale - self.cfg.min_scale)
        return float(scale)

    def save(self, path: str) -> None:
        data = {
            "config": asdict(self.cfg),
            "regimes": {key: asdict(stats) for key, stats in self.regimes.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "MetaRiskState":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg_data = data.get("config", {})
        cfg = MetaRiskConfig(**cfg_data)
        state = cls(cfg)

        regimes_data = data.get("regimes", {})
        for key, stats_data in regimes_data.items():
            state.regimes[key] = RegimeStats(**stats_data)
        return state


if __name__ == "__main__":
    cfg = MetaRiskConfig(min_trades_per_regime=3)
    state = MetaRiskState(cfg)
    state.update_trade("trend", 1.0, "2026-01-01")
    state.update_trade("trend", -0.5, "2026-01-02")
    state.update_trade("trend", 1.2, "2026-01-03")
    print("meta_scale:", state.meta_scale("trend"))
