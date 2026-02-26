from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def risk_cap_position_size(
    model_size: float,
    sl_mult: float,
    atr: float,
    equity_at_entry: float,
    risk_per_trade: float,
    sl_min_price: float,
    contract_value_per_1_0_move: float,
    min_lot: float,
    max_lot: float,
    eps: float = 1e-9,
) -> Tuple[float, Dict[str, float]]:
    """
    Apply a hard risk cap to the model size.
    Size is interpreted in the same volume units used by the ledger (lot-equivalent).
    """
    model_size = max(0.0, _safe_float(model_size, 0.0))
    sl_mult = max(0.0, _safe_float(sl_mult, 0.0))
    atr = max(0.0, _safe_float(atr, 0.0))
    equity_at_entry = max(0.0, _safe_float(equity_at_entry, 0.0))
    risk_per_trade = max(0.0, _safe_float(risk_per_trade, 0.0))
    sl_min_price = max(0.0, _safe_float(sl_min_price, 0.0))
    contract_value_per_1_0_move = max(0.0, _safe_float(contract_value_per_1_0_move, 0.0))
    min_lot = max(0.0, _safe_float(min_lot, 0.0))
    max_lot = max(min_lot, _safe_float(max_lot, 0.0))

    risk_budget = float(risk_per_trade * equity_at_entry)
    sl_distance_price = float(max(sl_mult * atr, sl_min_price))
    usd_loss_per_1lot = float(sl_distance_price * contract_value_per_1_0_move)
    cap_lots = float(risk_budget / max(usd_loss_per_1lot, eps)) if risk_budget > 0.0 else 0.0

    if model_size <= 0.0 or cap_lots <= 0.0 or risk_budget <= 0.0:
        final_size = 0.0
    else:
        capped = min(model_size, cap_lots)
        # Preserve risk cap when cap_lots < min_lot (do not lift above cap).
        if cap_lots < min_lot:
            final_size = float(max(0.0, capped))
        else:
            final_size = float(np.clip(capped, min_lot, max_lot))

    debug = {
        "risk_budget_usd": float(risk_budget),
        "sl_distance_price": float(sl_distance_price),
        "usd_loss_per_1lot_if_sl": float(usd_loss_per_1lot),
        "cap_lots": float(cap_lots),
        "model_size": float(model_size),
        "final_size": float(final_size),
    }
    return float(final_size), debug
