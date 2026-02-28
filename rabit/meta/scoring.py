from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

ROUND_DIGITS = 8
_EPS = 1e-12
_DEFAULT_METHOD = "pnl=signed_squash;winrate=clamp01;dd=clamp01_or_percent"


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _round8(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ROUND_DIGITS)


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _normalize_pnl(pnl: Optional[float]) -> float:
    if pnl is None:
        return 0.0
    # Bounded signed normalization in [-1, 1] without division-by-zero risk.
    return float(pnl) / (abs(float(pnl)) + 1.0 + _EPS)


def _normalize_winrate(winrate: Optional[float]) -> tuple[float, int]:
    if winrate is None:
        return 0.0, 1
    return _clamp01(float(winrate)), 0


def _normalize_dd(max_dd: Optional[float]) -> tuple[float, int]:
    if max_dd is None:
        return 0.0, 1
    dd = abs(float(max_dd))
    if dd <= 1.0:
        return _clamp01(dd), 0
    if dd <= 100.0:
        # Commonly reported as percentage points; convert to [0, 1].
        return _clamp01(dd / 100.0), 0
    return 1.0, 0


def _holdout_metrics_from_report(holdout_report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(holdout_report, dict):
        return None
    metrics = holdout_report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {
        "total_pnl": metrics.get("total_pnl"),
        "winrate": metrics.get("winrate"),
        "max_dd": metrics.get("max_dd_pct"),
    }


@dataclass(frozen=True)
class ScoreResult:
    score_total: float
    components: Dict[str, float]
    notes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_total": _round8(self.score_total),
            "score_components": {
                "dd_norm": _round8(self.components.get("dd_norm", 0.0)),
                "pnl_norm": _round8(self.components.get("pnl_norm", 0.0)),
                "w_dd": _round8(self.components.get("w_dd", 1.0)),
                "w_pnl": _round8(self.components.get("w_pnl", 1.0)),
                "w_win": _round8(self.components.get("w_win", 1.0)),
                "winrate_norm": _round8(self.components.get("winrate_norm", 0.0)),
            },
            "scoring_notes": {
                "dd_missing": int(self.notes.get("dd_missing", 0)),
                "normalization_method": str(self.notes.get("normalization_method") or _DEFAULT_METHOD),
                "winrate_missing": int(self.notes.get("winrate_missing", 0)),
            },
        }


def compute_composite_score(
    metrics: Dict[str, Any],
    *,
    w_pnl: float = 1.0,
    w_win: float = 1.0,
    w_dd: float = 1.0,
    normalization_method: str = _DEFAULT_METHOD,
) -> ScoreResult:
    pnl_raw = _safe_float(metrics.get("total_pnl"))
    winrate_raw = _safe_float(metrics.get("winrate"))
    dd_raw = _safe_float(metrics.get("max_dd"))

    pnl_norm = _normalize_pnl(pnl_raw)
    winrate_norm, winrate_missing = _normalize_winrate(winrate_raw)
    dd_norm, dd_missing = _normalize_dd(dd_raw)

    score_total = (
        (float(w_pnl) * float(pnl_norm))
        + (float(w_win) * float(winrate_norm))
        - (float(w_dd) * float(dd_norm))
    )

    return ScoreResult(
        score_total=float(score_total),
        components={
            "pnl_norm": float(pnl_norm),
            "winrate_norm": float(winrate_norm),
            "dd_norm": float(dd_norm),
            "w_pnl": float(w_pnl),
            "w_win": float(w_win),
            "w_dd": float(w_dd),
        },
        notes={
            "dd_missing": int(dd_missing),
            "winrate_missing": int(winrate_missing),
            "normalization_method": str(normalization_method or _DEFAULT_METHOD),
        },
    )


def compute_scores(
    in_sample_metrics: Dict[str, Any],
    *,
    holdout_report: Optional[Dict[str, Any]] = None,
    w_pnl: float = 1.0,
    w_win: float = 1.0,
    w_dd: float = 1.0,
) -> Dict[str, Any]:
    in_sample = compute_composite_score(
        in_sample_metrics,
        w_pnl=float(w_pnl),
        w_win=float(w_win),
        w_dd=float(w_dd),
    )
    in_sample_payload = in_sample.to_dict()

    payload: Dict[str, Any] = {
        "score_in_sample": in_sample_payload.get("score_total"),
        "score_total": in_sample_payload.get("score_total"),
        "score_components": in_sample_payload.get("score_components", {}),
        "scoring_notes": in_sample_payload.get("scoring_notes", {}),
        "score_holdout": None,
    }

    holdout_metrics = _holdout_metrics_from_report(holdout_report)
    if holdout_metrics is not None:
        holdout = compute_composite_score(
            holdout_metrics,
            w_pnl=float(w_pnl),
            w_win=float(w_win),
            w_dd=float(w_dd),
        )
        holdout_payload = holdout.to_dict()
        payload["score_holdout"] = holdout_payload.get("score_total")

    return payload
