from __future__ import annotations

import datetime as dt
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from scripts import deterministic_utils as det
from rabit.meta import perf_history

DEFAULT_PERF_DAYS = 30
DEFAULT_MIN_WINRATE = 0.25
DEFAULT_MIN_TRADES_FOR_GATE = 20
DEFAULT_MIN_DAYS_FOR_GATE = 10
DEFAULT_PNL_EPS = 1e-6
DEFAULT_DD_LIMIT_MULT = 2.0
DEFAULT_LOSS_STREAK_MULT = 2.0
DEFAULT_REGRESSION_PNL_RATIO = 0.9
DEFAULT_REGRESSION_WINRATE_DELTA = 0.05


@dataclass(frozen=True)
class PromotionGateConfig:
    perf_days: int = DEFAULT_PERF_DAYS
    min_winrate: float = DEFAULT_MIN_WINRATE
    min_trades_for_gate: int = DEFAULT_MIN_TRADES_FOR_GATE
    min_days_for_gate: int = DEFAULT_MIN_DAYS_FOR_GATE
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


def _safe_float_or_none(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _safe_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


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


def _extract_pnl(row: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for key in keys:
        if key not in row:
            continue
        val = _safe_float_or_none(row.get(key))
        if val is not None:
            return val
    return None


def _extract_trade_day(row: Dict[str, Any]) -> Optional[str]:
    for key in (
        "day",
        "date",
        "exit_day",
        "exit_date",
        "entry_date",
        "exit_time",
        "entry_time",
        "time",
        "timestamp",
        "ts",
    ):
        day_key = _extract_day_key(row.get(key))
        if day_key is not None:
            return day_key
    return None


def _select_window_trade_rows(rows: List[Dict[str, Any]], window_days: int) -> List[Dict[str, Any]]:
    if not rows:
        return []
    normalized: List[Tuple[int, Dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        day_key = _extract_trade_day(row)
        ord_val = _date_ordinal(day_key)
        if ord_val is None:
            continue
        normalized.append((ord_val, row))

    if not normalized:
        return [row for row in rows if isinstance(row, dict)]

    normalized.sort(key=lambda item: item[0])
    if window_days <= 0:
        return [item[1] for item in normalized]

    max_ord = normalized[-1][0]
    min_ord = max_ord - int(window_days) + 1
    return [row for ord_val, row in normalized if ord_val >= min_ord]


def _count_win_loss_from_rows(rows: List[Dict[str, Any]], pnl_keys: Tuple[str, ...]) -> Dict[str, Any]:
    wins = 0
    losses = 0
    flats = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        pnl = _extract_pnl(row, pnl_keys)
        if pnl is None:
            continue
        if pnl > 0.0:
            wins += 1
        elif pnl < 0.0:
            losses += 1
        else:
            flats += 1
    outcomes = wins + losses
    winrate = float(wins) / float(outcomes) if outcomes > 0 else None
    return {
        "wins": int(wins),
        "losses": int(losses),
        "flats": int(flats),
        "outcomes": int(outcomes),
        "winrate": winrate,
    }


def _trade_winrate_from_payload(payload: Dict[str, Any], window_days: int) -> Optional[Dict[str, Any]]:
    trade_summary = payload.get("trade_summary") if isinstance(payload.get("trade_summary"), dict) else {}
    if trade_summary:
        wins = _safe_int_or_none(
            trade_summary.get("win_trades", trade_summary.get("wins", trade_summary.get("winning_trades")))
        )
        losses = _safe_int_or_none(
            trade_summary.get("loss_trades", trade_summary.get("losses", trade_summary.get("losing_trades")))
        )
        flats = _safe_int_or_none(trade_summary.get("flat_trades"))
        trades = _safe_int_or_none(trade_summary.get("trades"))
        if wins is not None and losses is not None:
            wins = max(0, int(wins))
            losses = max(0, int(losses))
            flats = max(0, int(flats or 0))
            outcomes = wins + losses
            winrate = float(wins) / float(outcomes) if outcomes > 0 else None
            if trades is None:
                trades = outcomes + flats
            return {
                "winrate": winrate,
                "trades": int(max(0, trades)),
                "win_trades": int(wins),
                "loss_trades": int(losses),
                "flat_trades": int(flats),
                "source": "trade_summary",
            }

    trade_rows = payload.get("trades") if isinstance(payload.get("trades"), list) else []
    if not trade_rows:
        return None

    window = _select_window_trade_rows(trade_rows, int(window_days))
    outcome_stats = _count_win_loss_from_rows(
        window,
        ("pnl", "net_pnl", "profit", "profit_loss", "realized_pnl"),
    )
    total_trades = outcome_stats["wins"] + outcome_stats["losses"] + outcome_stats["flats"]
    return {
        "winrate": outcome_stats.get("winrate"),
        "trades": int(total_trades),
        "win_trades": int(outcome_stats["wins"]),
        "loss_trades": int(outcome_stats["losses"]),
        "flat_trades": int(outcome_stats["flats"]),
        "source": "trade_rows",
    }


def _compute_perf(rows: List[Dict[str, Any]], window_days: int, min_days_for_winrate: int) -> Dict[str, Any]:
    window = _select_window_rows(rows, window_days)
    if not window:
        return {
            "days": 0,
            "winrate": None,
            "total_pnl": 0.0,
            "max_dd": 0.0,
            "max_loss_streak": 0,
            "start_day": None,
            "end_day": None,
            "win_days": 0,
            "loss_days": 0,
            "flat_days": 0,
            "winrate_source": "day",
        }

    total_pnl = 0.0
    win_days = 0
    loss_days = 0
    flat_days = 0
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
            win_days += 1
        elif pnl < 0.0:
            loss_days += 1
        else:
            flat_days += 1
        dd_val = _safe_float(row.get("intraday_dd", 0.0), 0.0)
        if dd_val > max_dd:
            max_dd = dd_val
        loss_streak = _safe_int(row.get("end_loss_streak", 0), 0)
        if loss_streak > max_loss_streak:
            max_loss_streak = loss_streak

    days = len(window)
    outcome_days = win_days + loss_days
    min_days = max(0, int(min_days_for_winrate))
    winrate: Optional[float] = None
    if days >= min_days and outcome_days > 0:
        winrate = float(win_days) / float(outcome_days)

    return {
        "days": int(days),
        "winrate": winrate,
        "total_pnl": float(total_pnl),
        "max_dd": float(max_dd),
        "max_loss_streak": int(max_loss_streak),
        "start_day": start_day,
        "end_day": end_day,
        "win_days": int(win_days),
        "loss_days": int(loss_days),
        "flat_days": int(flat_days),
        "winrate_source": "day",
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

    perf_path = perf_history.perf_history_path(candidate_path)
    perf_payload = perf_history.load_perf_history(perf_path)
    if perf_payload is None:
        if not os.path.exists(perf_path):
            reason = (
                f"perf_history_missing path={perf_path} "
                f"hint=run_shadow_replay --meta_state_path {candidate_path}"
            )
        else:
            reason = f"perf_history_invalid path={perf_path}"
        if debug:
            print(f"[perf_history] {reason}")
        return PromotionGateResult(
            ok=False,
            reason=reason,
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash="missing",
            performance_snapshot={},
            details=details,
        )

    perf_ok, perf_missing = perf_history.validate_perf_history(perf_payload)
    if not perf_ok:
        reason = f"perf_history_missing_fields fields={','.join(perf_missing)} path={perf_path}"
        if debug:
            print(f"[perf_history] {reason}")
        return PromotionGateResult(
            ok=False,
            reason=reason,
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash="missing",
            performance_snapshot={},
            details=details,
        )

    perf_candidate_sha = perf_payload.get("candidate_sha256")
    if perf_candidate_sha != candidate_hash:
        reason = f"perf_history_candidate_sha_mismatch perf={perf_candidate_sha} file={candidate_hash}"
        if debug:
            print(f"[perf_history] {reason}")
        return PromotionGateResult(
            ok=False,
            reason=reason,
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash="missing",
            performance_snapshot={},
            details=details,
        )

    details["perf_history_path"] = perf_path
    details["perf_history_source_csv"] = perf_payload.get("source_csv")

    replay_payload = {k: v for k, v in perf_payload.items() if k != "timestamps"}
    replay_hash = det.hash_json(replay_payload)

    guardrails = perf_history.guardrails_from_history(perf_payload)
    final_allowed = guardrails.get("final_allowed")
    trades_simulated = guardrails.get("trades_simulated")
    if final_allowed is not None:
        details["final_allowed"] = int(final_allowed)
        if int(final_allowed) <= 0:
            return PromotionGateResult(
                ok=False,
                reason="guardrail_final_allowed_zero",
                candidate_hash=candidate_hash,
                approved_hash=approved_hash,
                replay_hash=replay_hash,
                performance_snapshot={},
                details=details,
            )
    if trades_simulated is not None:
        details["trades_simulated"] = int(trades_simulated)
        if int(trades_simulated) <= 0:
            return PromotionGateResult(
                ok=False,
                reason="guardrail_trades_simulated_zero",
                candidate_hash=candidate_hash,
                approved_hash=approved_hash,
                replay_hash=replay_hash,
                performance_snapshot={},
                details=details,
            )

    perf_daily = perf_payload.get("daily") if isinstance(perf_payload.get("daily"), list) else []
    summary_perf = perf_history.perf_metrics_from_history(perf_payload)
    if perf_daily:
        candidate_perf = _compute_perf(
            perf_daily,
            int(config.perf_days),
            min_days_for_winrate=int(config.min_days_for_gate),
        )
        summary_trades = _safe_int_or_none(summary_perf.get("trades"))
        if summary_trades is not None:
            candidate_perf["trades"] = int(max(0, summary_trades))
        if _safe_float_or_none(candidate_perf.get("max_dd")) is None:
            summary_max_dd = _safe_float_or_none(summary_perf.get("max_dd"))
            if summary_max_dd is not None:
                candidate_perf["max_dd"] = float(summary_max_dd)
        if _safe_int_or_none(candidate_perf.get("max_loss_streak")) is None:
            summary_max_loss = _safe_int_or_none(summary_perf.get("max_loss_streak"))
            if summary_max_loss is not None:
                candidate_perf["max_loss_streak"] = int(summary_max_loss)
        if _safe_float_or_none(candidate_perf.get("total_pnl")) is None:
            summary_total_pnl = _safe_float_or_none(summary_perf.get("total_pnl"))
            if summary_total_pnl is not None:
                candidate_perf["total_pnl"] = float(summary_total_pnl)
    else:
        candidate_perf = perf_history.perf_metrics_from_history(perf_payload)
        candidate_perf["winrate_source"] = "summary"

    trade_winrate = _trade_winrate_from_payload(perf_payload, int(config.perf_days))
    if trade_winrate is not None:
        candidate_perf["winrate"] = trade_winrate.get("winrate")
        candidate_perf["winrate_source"] = trade_winrate.get("source")
        if _safe_int_or_none(trade_winrate.get("trades")) is not None:
            candidate_perf["trades"] = int(_safe_int_or_none(trade_winrate.get("trades")) or 0)
        candidate_perf["win_trades"] = int(_safe_int_or_none(trade_winrate.get("win_trades")) or 0)
        candidate_perf["loss_trades"] = int(_safe_int_or_none(trade_winrate.get("loss_trades")) or 0)
        candidate_perf["flat_trades"] = int(_safe_int_or_none(trade_winrate.get("flat_trades")) or 0)
    elif "winrate_source" not in candidate_perf:
        candidate_perf["winrate_source"] = "day" if perf_daily else "summary"

    days_val = _safe_int_or_none(candidate_perf.get("days"))
    if days_val is None:
        days_val = 0
    candidate_perf["days"] = int(max(0, days_val))

    total_trades_val = _safe_int_or_none(candidate_perf.get("trades"))
    if total_trades_val is None:
        total_trades_val = _safe_int_or_none(trades_simulated)
        if total_trades_val is not None:
            candidate_perf["trades"] = int(max(0, total_trades_val))

    total_pnl_val = _safe_float_or_none(candidate_perf.get("total_pnl"))
    if total_pnl_val is None:
        return PromotionGateResult(
            ok=False,
            reason="perf_history_missing_total_pnl",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    winrate_val = _safe_float_or_none(candidate_perf.get("winrate"))
    max_dd_val = _safe_float_or_none(candidate_perf.get("max_dd"))
    max_loss_val = _safe_int_or_none(candidate_perf.get("max_loss_streak"))

    min_trades_for_gate = max(0, int(config.min_trades_for_gate))
    min_days_for_gate = max(0, int(config.min_days_for_gate))
    has_min_trades = total_trades_val is not None and int(total_trades_val) >= min_trades_for_gate
    has_min_days = int(candidate_perf.get("days", 0)) >= min_days_for_gate
    winrate_gate_reason = "ok"
    if not has_min_trades or not has_min_days or winrate_val is None:
        winrate_gate_reason = "insufficient_sample"
    winrate_gate_applied = winrate_gate_reason == "ok"

    details.update(
        {
            "performance_days": int(candidate_perf.get("days", 0)),
            "performance_trades": int(total_trades_val) if total_trades_val is not None else None,
            "performance_winrate": winrate_val,
            "performance_winrate_source": candidate_perf.get("winrate_source"),
            "performance_total_pnl": total_pnl_val,
            "performance_reason": winrate_gate_reason,
            "promote_min_winrate": float(config.min_winrate),
            "promote_min_trades": int(min_trades_for_gate),
            "promote_min_days": int(min_days_for_gate),
        }
    )

    if winrate_gate_applied and winrate_val is not None and winrate_val < float(config.min_winrate):
        return PromotionGateResult(
            ok=False,
            reason=(
                "performance_winrate_low "
                f"winrate={winrate_val} min_winrate={float(config.min_winrate)} "
                f"trades={total_trades_val} days={candidate_perf.get('days')}"
            ),
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    if total_pnl_val < float(config.pnl_epsilon):
        return PromotionGateResult(
            ok=False,
            reason=f"performance_pnl_low total_pnl={total_pnl_val}",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    if max_dd_val is not None and max_dd_val > dd_limit:
        return PromotionGateResult(
            ok=False,
            reason=f"performance_dd_breach max_dd={max_dd_val}",
            candidate_hash=candidate_hash,
            approved_hash=approved_hash,
            replay_hash=replay_hash,
            performance_snapshot={},
            details=details,
        )

    if max_loss_val is not None and max_loss_val > loss_streak_limit:
        return PromotionGateResult(
            ok=False,
            reason=f"performance_loss_streak_breach max_loss_streak={max_loss_val}",
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
        approved_perf_source = "missing"
    else:
        approved_perf = None
        approved_perf_source = "perf_history"
        approved_perf_path = perf_history.perf_history_path(approved_path)
        approved_payload = perf_history.load_perf_history(approved_perf_path)
        if approved_payload is not None:
            approved_ok, approved_missing = perf_history.validate_perf_history(approved_payload)
            approved_sha = approved_payload.get("candidate_sha256")
            if approved_ok and approved_sha == approved_hash:
                approved_perf = perf_history.perf_metrics_from_history(approved_payload)
            elif debug:
                print(
                    f"[perf_history] approved_perf_history_invalid path={approved_perf_path} "
                    f"missing={approved_missing} sha_ok={approved_sha == approved_hash}"
                )

        if approved_perf is None:
            approved_perf_source = "shadow_replay"
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

            approved_summary = (
                approved_report.get("perf_summary", {}) if isinstance(approved_report, dict) else {}
            )
            approved_perf = perf_history.perf_metrics_from_summary(approved_summary)

    regression_pnl_ratio = float(config.regression_pnl_ratio)
    regression_win_delta = float(config.regression_winrate_delta)
    approved_days = _safe_int_or_none(approved_perf.get("days")) or 0
    approved_total_pnl = _safe_float_or_none(approved_perf.get("total_pnl"))
    approved_winrate = _safe_float_or_none(approved_perf.get("winrate"))
    pnl_required = (approved_total_pnl or 0.0) * regression_pnl_ratio
    winrate_required = (approved_winrate or 0.0) - regression_win_delta

    if approved_days > 0:
        if approved_total_pnl is None or (winrate_gate_applied and approved_winrate is None):
            if strict:
                return PromotionGateResult(
                    ok=False,
                    reason="approved_performance_missing",
                    candidate_hash=candidate_hash,
                    approved_hash=approved_hash,
                    replay_hash=replay_hash,
                    performance_snapshot={},
                    details=details,
                )
        elif total_pnl_val < pnl_required:
            return PromotionGateResult(
                ok=False,
                reason=(
                    "regression_pnl "
                    f"candidate={total_pnl_val} approved={approved_total_pnl}"
                ),
                candidate_hash=candidate_hash,
                approved_hash=approved_hash,
                replay_hash=replay_hash,
                performance_snapshot={},
                details=details,
            )
        if (
            winrate_gate_applied
            and winrate_val is not None
            and approved_winrate is not None
            and winrate_val < winrate_required
        ):
            return PromotionGateResult(
                ok=False,
                reason=(
                    "regression_winrate "
                    f"candidate={winrate_val} approved={approved_winrate}"
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
            "min_trades_for_gate": int(min_trades_for_gate),
            "min_days_for_gate": int(min_days_for_gate),
            "pnl_epsilon": float(config.pnl_epsilon),
            "dd_limit": float(dd_limit),
            "loss_streak_limit": int(loss_streak_limit),
        },
        "winrate_gate": {
            "applied": int(bool(winrate_gate_applied)),
            "reason": winrate_gate_reason,
            "winrate_source": candidate_perf.get("winrate_source"),
            "trades": total_trades_val,
            "days": candidate_perf.get("days"),
        },
        "guardrails": {
            "final_allowed": final_allowed,
            "trades_simulated": trades_simulated,
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
            "perf_history_path": perf_path,
            "perf_history_source_csv": perf_payload.get("source_csv"),
            "perf_history_input_hash": perf_payload.get("run", {}).get("input_hash"),
            "perf_history_equity_hash": perf_payload.get("run", {}).get("equity_hash"),
            "perf_history_regime_ledger_hash": perf_payload.get("run", {}).get("regime_ledger_hash"),
            "approved_perf_source": approved_perf_source,
        },
    }

    pass_reason = "ok"
    if winrate_gate_reason == "insufficient_sample":
        pass_reason = "insufficient_sample"

    return PromotionGateResult(
        ok=True,
        reason=pass_reason,
        candidate_hash=candidate_hash,
        approved_hash=approved_hash,
        replay_hash=replay_hash,
        performance_snapshot=performance_snapshot,
        details=details,
    )
