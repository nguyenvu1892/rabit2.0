from __future__ import annotations

import numpy as np


class RegimePolicyBank:
    """
    Holds multiple policies keyed by regime name.
    If regime not found -> HOLD
    """

    def __init__(self, policies: dict[str, object], fallback: object | None = None):
        self.policies = policies
        self.fallback = fallback

    def act(self, x: np.ndarray, regime: str):
        p = self.policies.get(regime, None)
        if p is None:
            if self.fallback is not None:
                return self.fallback.act(x)
            return (0, 0.8, 0.8, 20)
        return p.act(x)