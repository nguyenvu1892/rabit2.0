#!/usr/bin/env python
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    if out in (float("inf"), float("-inf")):
        return None
    return float(out)


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return float(lo)
    if value > hi:
        return float(hi)
    return float(value)


def _as_key_set(values: Any) -> set[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return set()
    out: set[str] = set()
    for item in values:
        if item is None:
            continue
        out.add(str(item))
    return out


def _extract_row_perf_score(row: Any) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    for key in ("perf_score", "score", "perf"):
        maybe = _safe_float(row.get(key))
        if maybe is not None:
            return _clamp(float(maybe), -1.0, 1.0)
    return None


def _extract_perf_score(regime: str, regime_perf: Any) -> float:
    if not isinstance(regime_perf, dict):
        return 0.0
    reg = str(regime or "unknown").strip().lower() or "unknown"

    row = _extract_row_perf_score(regime_perf)
    if row is not None:
        return float(row)

    per_regime = regime_perf.get("per_regime")
    if isinstance(per_regime, dict):
        if reg in per_regime:
            row = _extract_row_perf_score(per_regime.get(reg))
            if row is not None:
                return float(row)
        for key in sorted(per_regime.keys()):
            if str(key).strip().lower() != reg:
                continue
            row = _extract_row_perf_score(per_regime.get(key))
            if row is not None:
                return float(row)
        global_row = _extract_row_perf_score(regime_perf.get("global"))
        if global_row is not None:
            return float(global_row)
        global_row = _extract_row_perf_score(per_regime.get("_global"))
        if global_row is not None:
            return float(global_row)

    global_row = _extract_row_perf_score(regime_perf.get("global"))
    if global_row is not None:
        return float(global_row)
    return 0.0


def _regime_multiplier(regime: str, cfg: Dict[str, Any]) -> float:
    multipliers = cfg.get("regime_multipliers")
    if not isinstance(multipliers, dict):
        return 1.0
    reg = str(regime or "unknown").strip().lower() or "unknown"
    value = multipliers.get(reg)
    if value is None:
        value = multipliers.get("*")
    maybe = _safe_float(value)
    if maybe is None:
        return 1.0
    return _clamp(float(maybe), 0.5, 1.5)


def compute_regime_mutation_intensity(regime: str, regime_perf: Any, conf: float, cfg: Dict[str, Any]) -> float:
    config = dict(cfg or {})
    base = _safe_float(config.get("base_intensity"))
    if base is None:
        base = 1.0

    min_intensity = _safe_float(config.get("min_intensity"))
    if min_intensity is None:
        min_intensity = 0.75
    max_intensity = _safe_float(config.get("max_intensity"))
    if max_intensity is None:
        max_intensity = 1.35
    if max_intensity < min_intensity:
        max_intensity = min_intensity

    perf_weight = _safe_float(config.get("perf_weight"))
    if perf_weight is None:
        perf_weight = 0.85
    perf_weight = _clamp(float(perf_weight), 0.0, 2.0)

    conf_floor = _safe_float(config.get("confidence_floor"))
    if conf_floor is None:
        conf_floor = 0.25
    conf_floor = _clamp(float(conf_floor), 0.0, 1.0)

    conf_clamped = _safe_float(conf)
    if conf_clamped is None:
        conf_clamped = 0.0
    conf_clamped = _clamp(float(conf_clamped), 0.0, 1.0)
    confidence_scale = float(conf_floor) + (1.0 - float(conf_floor)) * float(conf_clamped)

    perf_score = _extract_perf_score(regime=regime, regime_perf=regime_perf)
    exploration_push = -float(perf_score) * float(confidence_scale)

    raw = float(base) * (1.0 + float(perf_weight) * float(exploration_push))
    raw *= _regime_multiplier(regime=regime, cfg=config)
    return round(_clamp(raw, float(min_intensity), float(max_intensity)), 8)


def apply_intensity_to_mutation_cfg(base_cfg: Dict[str, Any], intensity: float) -> Dict[str, Any]:
    out = deepcopy(base_cfg) if isinstance(base_cfg, dict) else {}
    if not out:
        return {"applied_intensity": 1.0}

    intensity_value = _safe_float(intensity)
    if intensity_value is None:
        intensity_value = 1.0
    intensity_value = max(0.0, float(intensity_value))

    scalable_keys = _as_key_set(out.get("scalable_keys"))
    for key, value in list(out.items()):
        if key.endswith("_min") or key.endswith("_max"):
            continue
        if key == "scalable_keys":
            continue
        if isinstance(value, bool):
            continue

        min_key = f"{key}_min"
        max_key = f"{key}_max"
        has_bounds = min_key in out or max_key in out
        if key not in scalable_keys and not has_bounds:
            continue

        numeric = _safe_float(value)
        if numeric is None:
            continue
        scaled = float(numeric) * float(intensity_value)

        lower = _safe_float(out.get(min_key))
        upper = _safe_float(out.get(max_key))
        if lower is not None:
            scaled = max(float(lower), float(scaled))
        if upper is not None:
            scaled = min(float(upper), float(scaled))

        if isinstance(value, int):
            out[key] = int(round(float(scaled)))
        else:
            out[key] = round(float(scaled), 8)

    out["applied_intensity"] = round(float(intensity_value), 8)
    return out

