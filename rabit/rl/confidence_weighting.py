from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ConfidenceWeighterConfig:
    # size = max(min_size, confidence**power)
    power: float = 1.3
    min_size: float = 0.0
    max_size: float = 1.0

    # optional deadzone: confidence below this => size = 0
    deadzone: float = 0.0


class ConfidenceWeighter:
    """
    Convert confidence in [0,1] into position size in [0,1].

    - No direction decision.
    - No TP/SL decision.
    - Only risk allocation (volume scaling).
    """

    def __init__(self, cfg: ConfidenceWeighterConfig | None = None):
        self.cfg = cfg or ConfidenceWeighterConfig()

    def size(self, confidence: float) -> float:
        c = float(confidence)
        if not np.isfinite(c):
            return 0.0

        c = float(np.clip(c, 0.0, 1.0))

        if self.cfg.deadzone > 0.0 and c < self.cfg.deadzone:
            return 0.0

        s = c ** float(self.cfg.power)
        s = float(np.clip(s, self.cfg.min_size, self.cfg.max_size))
        return s