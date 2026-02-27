from __future__ import annotations

import datetime as dt
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from scripts import _deterministic as detx
from scripts import deterministic_utils as det

DEFAULT_PERF_DAYS = 30
DEFAULT_MIN_WINRATE = 0.5
DEFAULT_PNL_EPS = 1e-6
DEFAULT_DD_LIMIT_MULT = 2.0
DEFAULT_LOSS_STREAK_MULT = 2.0
DEFAULT_REGRESSION_PNL_RATIO = 0.9
DEFAULT_REGRESSION_WINRATE_DELTA = 0.05


@dataclass(frozen=True)
class PromotionGateConfig:
    perf_days: int = DEFAULT_PERF_DAYS
    min_winrate: float = DEFAULT_MIN_WINRATE
    pnl_epsilon: float = DEFAULT_PNL_EPS
    dd_limit_mult: float = DEFAULT_DD_LIMIT_MULT
    loss_streak_mult: float = DEFAULT_LOSS_STREAK_MULT
    regression_pnl_ratio: float = DEFAULT_REGRESSION_PNL_RATIO
    regression_winrate_delta: float = DEFAULT_REGRESSION_WINRATE_DELTA


@dataclass
class PromotionGateResult:
    ok: bool
    reason: str
    candidate_hash: str
    approved_hash: str
    replay_hash: str
    performance_snapshot: Dict[str, Any]
    details: Dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_day_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    s = str(value).strip()
    if not s:
        return None
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    for sep in ("T", " "):
        if sep in s:
            part = s.split(sep)[0]
            if len(part) >= 10 and part[4] == "-" and part[7] == "-":
                return part[:10]
    try:
        return dt.date.fromisoformat(s[:10]).isoformat()
    except Exception:
        return None


def _date_ordinal(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10]).toordinal()
    except Exception:
        return None


def _select_window_rows(rows: List[Dict[str, Any]], window_days: int) -> List[Dict[str, Any]]:
    if not rows:
        return []
    normalized: List[Tuple[int, str, Dict[str, Any]]] = []
    fallback: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        day_key = _extract_day_key(row.get("day"))
        if not day_key:
            fallback.append(row)
            continue
        ord_val = _date_ordinal(day_key)
        if ord_val is None:
            fallback.append(row)
            continue
        normalized.append((ord_val, day_key, row))

    if normalized:
        normalized.sort(key=lambda r: r[0])
        ordered = [dict(item[2], day=item[1]) for item in normalized]
    else:
        ordered = list(fallback)

    if window_days <= 0:
        return ordered
    if len(ordered) <= window_days:
        return ordered
    return ordered[-int(window_days) :]


def _compute_perf(rows: List[Dict[str, Any]], window_days: int) -> Dict[str, Any]:
    window = _select_window_rows(rows, window_days)
    if not window:
        return {
            "days": 0,
            "winrate": 0.0,
            "total_pnl": 0.0,
            "max_dd": 0.0,
            "max_loss_streak": 0,
            "start_day": None,
            "end_day": None,
        }

    total_pnl = 0.0
    wins = 0
    max_dd = 0.0
    max_loss_streak = 0
    start_day = None
    end_day = None

    for idx, row in enumerate(window):
        if not isinstance(row, dict):
            continue
        day_key = _extract_day_key(row.get("day"))
        if day_key:
            if start_day is None:
                start_day = day_key
            end_day = day_key
        pnl = _safe_float(row.get("day_pnl", 0.0), 0.0)
        total_pnl += pnl
        if pnl > 0.0:
            wins += 1
        dd_val = _safe_float(row.get("intraday_dd", 0.0), 0.0)
        if dd_val > max_dd:
            max_dd = dd_val
        loss_streak = _safe_int(row.get("end_loss_streak", 0), 0)
        if loss_streak > max_loss_streak:
            max_loss_streak = loss_streak

    days = len(window)
    winrate = float(wins) / float(days) if days > 0 else 0.0

    return {
        "days": int(days),
        "winrate": float(winrate),
        "total_pnl": float(total_pnl),
        "max_dd": float(max_dd),
        "max_loss_streak": int(max_loss_streak),
        "start_day": start_day,
        "end_day": end_day,
    }


def _find_nonfinite(obj: Any, path: str = "") -> Optional[str]:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return path or "<root>"
        return None
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            found = _find_nonfinite(value, next_path)
            if found is not None:
                return found
        return None
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_path = f"{path}[{idx}]"
            found = _find_nonfinite(value, next_path)
            if found is not None:
                return found
        return None
    return None


def _same_path(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    try:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))
    except Exception:
        return left == right


def _build_shadow_args(
    csv_path: str,
    model_path: str,
    meta_state_path: str,
    deterministic_check: bool,
    debug: bool,
):
    from scripts import shadow_replay

    cli = [
        "--csv",
        csv_path,
        "--model_path",
        model_path,
        "--meta_state_path",
        meta_state_path,
        "--deterministic_check",
        str(int(bool(deterministic_check))),
        "--debug",
        str(int(bool(debug))),
    ]
    cli_args = shadow_replay._build_arg_parser().parse_args(cli)
    return shadow_replay._merge_live_defaults(cli_args)


def _run_shadow_replay_once(
    args,
    deterministic_enabled: bool,
    debug: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    from scripts import shadow_replay

    return shadow_replay._run_shadow_replay_once(
        args,
        debug_enabled=debug,
        deterministic_enabled=deterministic_enabled,
        read_only_state=True,
    )


def evaluate_candidate(
    candidate_path: str,
    approved_path: str,
    csv_path: str,
    model_path: str,
    strict: bool = True,
    replay_check: bool = True,
    cfg: Optional[PromotionGateConfig] = None,
    debug: bool = False,
) -> PromotionGateResult:
    config = cfg or PromotionGateConfig()
    details: Dict[str, Any] = {}
    performance_snapshot: Dict[str, Any] = {}

    if not candidate_path or not os.path.exists(candidate_path):
        return PromotionGateResult(
            ok=False,
            reason=f"candidate_missing path={candidate_path}",
            candidate_hash="missing",
            approved_hash=det.sha256_file(approved_path) if approved_path else "missing",
            replay_hash="missing",
            performance_snapshot={},
            details=details,
        )

    if _same_path(candidate_path, approved_path):
        return PromotionGateResult(
            ok=False,
            reason="candidate_path_equals_approved_path",
            candidate_hash=det.sha256_file(candidate_path),
            approved_hash=det.sha256_file(approved_path),
            replay_hash="missing",
            performance_snapshot={},
            details=details,
        )

    candidate_hash = det.sha256_file(candidate_path)
    approved_hash = det.sha256_file(approved_path) if approved_path else "missing"

    candidate_data = det.load_json(candidate_path)
    if not isinstance(candidate_data, dict):
        return PromotionGateResult(
            ok=False,
            reason="candidate_json_invalid",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash="missing",
            performance_snapshot={},
            details=details,
        )

    missing_keys = [key for key in ("config", "regimes") if key not in candidate_data]
    if missing_keys:
        return PromotionGateResult(
            ok=False,
            reason=f"candidate_missing_keys keys={','.join(missing_keys)}",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash="missing",
            performance_snapshot={},
            details=details,
        )

    history_present = "history" in candidate_data
    if history_present:
        history_val = candidate_data.get("history")
        if not isinstance(history_val, list):
            return PromotionGateResult(
                ok=False,
                reason="candidate_history_invalid",
                candidate_hash=candidate_hash,
                approved_hash=approved_hash,
                replay_hash="missing",
                performance_snapshot={},
                details=details,
            )
        if len(history_val) < 1 and strict:
            return PromotionGateResult(
                ok=False,
                reason="candidate_history_empty",
                candidate_hash=candidate_hash,
                approved_hash=approved_hash,
                replay_hash="missing",
                performance_snapshot={},
                details=details,
            )

    nonfinite_path = _find_nonfinite(candidate_data)
    if nonfinite_path is not None:
        return PromotionGateResult(
            ok=False,
            reason=f"candidate_nonfinite value_path={nonfinite_path}",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash="missing",
            performance_snapshot={},
            details=details,
        )

    config_data = candidate_data.get("config", {}) if isinstance(candidate_data.get("config"), dict) else {}
    daily_dd_limit = _safe_float(config_data.get("daily_dd_limit", 3.0), 3.0)
    cooldown_trades = _safe_int(config_data.get("cooldown_trades", 5), 5)
    dd_limit = float(daily_dd_limit) * float(config.dd_limit_mult)
    loss_streak_limit = int(cooldown_trades * config.loss_streak_mult)

    details.update(
        {
            "daily_dd_limit": daily_dd_limit,
            "cooldown_trades": cooldown_trades,
            "dd_limit": dd_limit,
            "loss_streak_limit": loss_streak_limit,
        }
    )

    report = None
    snapshot1 = None
    snapshot2 = None
    replay_hash = "missing"

    try:
        if replay_check:
            args = _build_shadow_args(csv_path, model_path, candidate_path, deterministic_check=True, debug=debug)
            report, snapshot1 = _run_shadow_replay_once(args, deterministic_enabled=True, debug=debug)
            _report2, snapshot2 = _run_shadow_replay_once(args, deterministic_enabled=True, debug=debug)
            try:
                detx.compare_deterministic_snapshots(snapshot1, snapshot2)
            except Exception as exc:
                return PromotionGateResult(
                    ok=False,
                    reason=f"deterministic_mismatch {exc}",
                    candidate_hash=candidate_hash,
                    approved_hash=approved_hash,
                    replay_hash="missing",
                    performance_snapshot={},
                    details=details,
                )
            replay_hash = det.hash_json(snapshot1) if snapshot1 is not None else "missing"
        else:
            args = _build_shadow_args(csv_path, model_path, candidate_path, deterministic_check=False, debug=debug)
            report, snapshot1 = _run_shadow_replay_once(args, deterministic_enabled=False, debug=debug)
            replay_hash = "skipped"
    except Exception as exc:
        return PromotionGateResult(
            ok=False,
            reason=f"shadow_replay_failed {exc}",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash="missing",
            performance_snapshot={},
            details=details,
        )

    if not isinstance(report, dict):
        return PromotionGateResult(
            ok=False,
            reason="shadow_replay_report_missing",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    guardrails = report.get("guardrails", {}) if isinstance(report.get("guardrails"), dict) else {}
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    final_allowed = _safe_int(guardrails.get("final_allowed", 0), 0)
    trades_simulated = _safe_int(counts.get("trades_simulated", 0), 0)
    details["final_allowed"] = final_allowed
    details["trades_simulated"] = trades_simulated
    if final_allowed <= 0:
        return PromotionGateResult(
            ok=False,
            reason="guardrail_final_allowed_zero",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )
    if trades_simulated <= 0:
        return PromotionGateResult(
            ok=False,
            reason="guardrail_trades_simulated_zero",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    daily_rows = report.get("daily_table", []) if isinstance(report.get("daily_table"), list) else []
    candidate_perf = _compute_perf(daily_rows, int(config.perf_days))
    if candidate_perf.get("days", 0) < 1:
        return PromotionGateResult(
            ok=False,
            reason="performance_history_empty",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    if candidate_perf["winrate"] < float(config.min_winrate):
        return PromotionGateResult(
            ok=False,
            reason=f"performance_winrate_low winrate={candidate_perf['winrate']}",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    if candidate_perf["total_pnl"] < float(config.pnl_epsilon):
        return PromotionGateResult(
            ok=False,
            reason=f"performance_pnl_low total_pnl={candidate_perf['total_pnl']}",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    if candidate_perf["max_dd"] > dd_limit:
        return PromotionGateResult(
            ok=False,
            reason=f"performance_dd_breach max_dd={candidate_perf['max_dd']}",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    if candidate_perf["max_loss_streak"] > loss_streak_limit:
        return PromotionGateResult(
            ok=False,
            reason=f"performance_loss_streak_breach max_loss_streak={candidate_perf['max_loss_streak']}",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    if not approved_path or not os.path.exists(approved_path):
        if strict:
            return PromotionGateResult(
                ok=False,
                reason=f"approved_missing path={approved_path}",
                candidate_hash=candidate_hash,
                approved_hash=approved_hash,
                replay_hash=replay_hash,
                performance_snapshot={},
                details=details,
            )
        approved_perf = {
            "days": 0,
            "winrate": 0.0,
            "total_pnl": 0.0,
            "max_dd": 0.0,
            "max_loss_streak": 0,
            "start_day": None,
            "end_day": None,
        }
    else:
        try:
            approved_args = _build_shadow_args(
                csv_path, model_path, approved_path, deterministic_check=False, debug=debug
            )
            approved_report, _ = _run_shadow_replay_once(
                approved_args,
                deterministic_enabled=False,
                debug=debug,
            )
        except Exception as exc:
            if strict:
                return PromotionGateResult(
                    ok=False,
                    reason=f"approved_shadow_replay_failed {exc}",
                    candidate_hash=candidate_hash,
                    approved_hash=approved_hash,
                    replay_hash=replay_hash,
                    performance_snapshot={},
                    details=details,
                )
            approved_report = {}

        approved_daily = (
            approved_report.get("daily_table", []) if isinstance(approved_report, dict) else []
        )
        approved_perf = _compute_perf(approved_daily, int(config.perf_days))

    regression_pnl_ratio = float(config.regression_pnl_ratio)
    regression_win_delta = float(config.regression_winrate_delta)
    pnl_required = approved_perf["total_pnl"] * regression_pnl_ratio
    winrate_required = approved_perf["winrate"] - regression_win_delta

    if approved_perf.get("days", 0) > 0:
        if candidate_perf["total_pnl"] < pnl_required:
            return PromotionGateResult(
                ok=False,
                reason=(
                    "regression_pnl "
                    f"candidate={candidate_perf['total_pnl']} approved={approved_perf['total_pnl']}"
                ),
                candidate_hash=candidate_hash,
                approved_hash=approved_hash,
                replay_hash=replay_hash,
                performance_snapshot={},
                details=details,
            )
        if candidate_perf["winrate"] < winrate_required:
            return PromotionGateResult(
                ok=False,
                reason=(
                    "regression_winrate "
                    f"candidate={candidate_perf['winrate']} approved={approved_perf['winrate']}"
                ),
                candidate_hash=candidate_hash,
                approved_hash=approved_hash,
                replay_hash=replay_hash,
                performance_snapshot={},
                details=details,
            )
    elif strict:
        return PromotionGateResult(
            ok=False,
            reason="approved_performance_missing",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    performance_snapshot = {
        "window_days": int(config.perf_days),
        "thresholds": {
            "min_winrate": float(config.min_winrate),
            "pnl_epsilon": float(config.pnl_epsilon),
            "dd_limit": float(dd_limit),
            "loss_streak_limit": int(loss_streak_limit),
        },
        "guardrails": {
            "final_allowed": int(final_allowed),
            "trades_simulated": int(trades_simulated),
        },
        "candidate": candidate_perf,
        "approved": approved_perf,
        "regression": {
            "pnl_ratio_required": float(regression_pnl_ratio),
            "winrate_delta_allowed": float(regression_win_delta),
            "pnl_required": float(pnl_required),
            "winrate_required": float(winrate_required),
        },
        "replay": {
            "csv": csv_path,
            "model_path": model_path,
            "replay_check": int(bool(replay_check)),
        },
    }

    return PromotionGateResult(
        ok=True,
        reason="ok",
        candidate_hash=candidate_hash,
        approved_hash=approved_hash,
        replay_hash=replay_hash,
        performance_snapshot=performance_snapshot,
        details=details,
    )
