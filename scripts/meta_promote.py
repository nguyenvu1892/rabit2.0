#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional

from rabit.meta import perf_history
from rabit.meta import scoring
from rabit.state import atomic_io
from rabit.state import promotion_gate
from rabit.state.exit_codes import ExitCode
from rabit.utils import get_logger
from scripts import deterministic_utils as det

DEFAULT_APPROVED_DIR = os.path.join("data", "meta_states", "current_approved")
DEFAULT_HISTORY_DIR = os.path.join("data", "meta_states", "history")
DEFAULT_REJECTED_DIR = os.path.join("data", "meta_states", "rejected")
DEFAULT_LEDGER_PATH = os.path.join("data", "meta_states", "ledger.jsonl")
DEFAULT_CSV = os.path.join("data", "live", "XAUUSD_M1_live.csv")
DEFAULT_MODEL = os.path.join("data", "ars_best_theta_regime_bank.npz")
DEFAULT_HOLDOUT_REPORT_PATH = os.path.join("data", "reports", "holdout", "holdout_report.json")

EXIT_OK = ExitCode.SUCCESS
EXIT_REJECT = ExitCode.BUSINESS_REJECT
EXIT_ERROR = ExitCode.INTERNAL_ERROR
EXIT_CODE_SCORING_FAIL = ExitCode.DATA_INVALID

SCORING_ERROR_HOLDOUT_REPORT_MISSING = "SCORING_ERROR: HOLDOUT_REPORT_MISSING"
SCORING_ERROR_HOLDOUT_REPORT_INVALID = "SCORING_ERROR: HOLDOUT_REPORT_INVALID"
SCORING_ERROR_SCORE_COMPUTE_FAILED = "SCORING_ERROR: SCORE_COMPUTE_FAILED"

_REJECT_REASON_PREFIXES = (
    "guardrail_",
    "performance_",
    "regression_",
)

_DATA_INVALID_REASON_PREFIXES = (
    "candidate_",
    "perf_history_",
    "approved_missing",
    "approved_shadow_replay_failed",
    "approved_performance_missing",
)

_STATE_CORRUPT_REASON_PREFIXES = (
    "history_archive_failed",
    "atomic_replace_failed",
    "ledger_write_failed",
    "ledger_entry_failed",
    "promotion_move_failed",
    "rejected_move_failed",
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Promotion gate for meta-risk state.")
    ap.add_argument("--candidate_path", required=True, help="Path to candidate meta_risk_state.json")
    ap.add_argument(
        "--approved_dir",
        default=DEFAULT_APPROVED_DIR,
        help="Directory containing current approved meta_risk_state.json",
    )
    ap.add_argument(
        "--history_dir",
        default=DEFAULT_HISTORY_DIR,
        help="Directory for archived approved states",
    )
    ap.add_argument(
        "--rejected_dir",
        default=DEFAULT_REJECTED_DIR,
        help="Directory for rejected candidates",
    )
    ap.add_argument(
        "--ledger_path",
        default=DEFAULT_LEDGER_PATH,
        help="Append-only promotion ledger path (JSONL)",
    )
    ap.add_argument("--strict", type=int, default=1, help="Strict mode (0/1)")
    ap.add_argument("--reason", required=True, help="Promotion reason string")
    ap.add_argument("--replay_check", type=int, default=1, help="Determinism replay check (0/1)")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Bars CSV for shadow replay")
    ap.add_argument("--model_path", default=DEFAULT_MODEL, help="Model path for shadow replay")
    ap.add_argument("--perf_days", type=int, default=promotion_gate.DEFAULT_PERF_DAYS, help="Perf window days")
    ap.add_argument(
        "--promote_min_winrate",
        type=float,
        default=0.25,
        help="Minimum winrate required by promotion gate when sample is sufficient",
    )
    ap.add_argument(
        "--promote_min_trades",
        type=int,
        default=20,
        help="Minimum trades required before enforcing winrate gate",
    )
    ap.add_argument(
        "--promote_min_days",
        type=int,
        default=10,
        help="Minimum days required before enforcing winrate gate",
    )
    ap.add_argument(
        "--no_exit_on_reject",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, return code 0 on reject (dev/test mode). Errors still return 2.",
    )
    ap.add_argument("--debug", type=int, default=0, help="Debug mode (0/1)")
    ap.add_argument("--cycle_id", default="", help="Optional correlation id propagated by meta_cycle")
    ap.add_argument(
        "--eval_only",
        type=int,
        choices=[0, 1],
        default=0,
        help="Evaluate gate/scoring only (no promotion, no reject move, no ledger write).",
    )
    ap.add_argument(
        "--eval_report_path",
        default="",
        help="Optional JSON path for eval-only payload.",
    )
    ap.add_argument(
        "--ledger_extra_path",
        default="",
        help="Optional JSON path for additive ledger fields.",
    )
    ap.add_argument(
        "--holdout_report_path",
        default=DEFAULT_HOLDOUT_REPORT_PATH,
        help="Optional holdout report path for score_holdout.",
    )
    ap.add_argument(
        "--enable_scoring",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable composite scoring layer after gate pass (0/1, default: 0).",
    )
    ap.add_argument("--w_pnl", type=float, default=1.0, help="Composite score weight for normalized PnL.")
    ap.add_argument("--w_win", type=float, default=1.0, help="Composite score weight for normalized winrate.")
    ap.add_argument("--w_dd", type=float, default=1.0, help="Composite score weight for normalized drawdown.")
    return ap.parse_args()


def _safe_summary_value(value: Any, default: str = "missing") -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else default
    return value


def _normalize_reason(reason: Any) -> str:
    text = str(reason or "unknown_error")
    return " ".join(text.split())


def _print_summary_line(
    status: str,
    reason: Any,
    result: Optional[promotion_gate.PromotionGateResult],
    perf_summary: Optional[Dict[str, Any]],
    **extra: Any,
) -> None:
    perf = perf_summary or {}
    parts = [
        f"[promotion] STATUS={status}",
        f"reason={_normalize_reason(reason)}",
        f"winrate={_safe_summary_value(perf.get('winrate'))}",
        f"trades={_safe_summary_value(perf.get('trades'))}",
        f"days={_safe_summary_value(perf.get('days'))}",
        f"total_pnl={_safe_summary_value(perf.get('total_pnl'))}",
        f"candidate_hash={_safe_summary_value(result.candidate_hash if result is not None else None)}",
        f"approved_hash={_safe_summary_value(result.approved_hash if result is not None else None)}",
        f"replay_hash={_safe_summary_value(result.replay_hash if result is not None else None)}",
        f"perf_history_path={_safe_summary_value(perf.get('perf_history_path'))}",
    ]
    for key, value in extra.items():
        parts.append(f"{key}={_safe_summary_value(value)}")
    print(" ".join(parts))


def _processing_error_exit_code(reason: Any) -> int | None:
    reason_text = _normalize_reason(reason)
    for prefix in _REJECT_REASON_PREFIXES:
        if reason_text.startswith(prefix):
            return None
    for prefix in _DATA_INVALID_REASON_PREFIXES:
        if reason_text.startswith(prefix):
            return ExitCode.DATA_INVALID
    for prefix in _STATE_CORRUPT_REASON_PREFIXES:
        if reason_text.startswith(prefix):
            return ExitCode.STATE_CORRUPT
    if "corrupt" in reason_text:
        return ExitCode.STATE_CORRUPT
    # Preserve legacy behavior for unknown gate failures by treating them as rejection.
    return None


def _utc_iso() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _round_float(value: Any, default: Optional[float] = None, digits: int = 8) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return round(out, int(digits))


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    atomic_io.atomic_write_text(
        path,
        det.stable_json_dumps(payload),
        suffix=".json",
        create_backup=True,
    )


def _atomic_copy_file(src: str, dest: str) -> None:
    if not src or not os.path.exists(src):
        raise RuntimeError(f"source_missing path={src}")
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest):
        raise RuntimeError(f"destination_exists path={dest}")

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dest_dir or None)
    try:
        with open(src, "rb") as fsrc, os.fdopen(fd, "wb") as fdst:
            for chunk in iter(lambda: fsrc.read(1024 * 1024), b""):
                fdst.write(chunk)
            fdst.flush()
            os.fsync(fdst.fileno())
        os.replace(tmp_path, dest)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_replace_from_source(src: str, dest: str) -> None:
    if not src or not os.path.exists(src):
        raise RuntimeError(f"source_missing path={src}")
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_promote_", suffix=".json", dir=dest_dir or None)
    try:
        with open(src, "rb") as fsrc, os.fdopen(fd, "wb") as fdst:
            for chunk in iter(lambda: fsrc.read(1024 * 1024), b""):
                fdst.write(chunk)
            fdst.flush()
            os.fsync(fdst.fileno())
        # Atomic replace on same filesystem because temp file is in destination directory.
        os.replace(tmp_path, dest)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _timestamp() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def _unique_path(dir_path: str, filename: str, stamp: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dir_path, f"{base}_{stamp}{ext}")
    if not os.path.exists(candidate):
        return candidate
    for idx in range(1, 1000):
        alt = os.path.join(dir_path, f"{base}_{stamp}_{idx}{ext}")
        if not os.path.exists(alt):
            return alt
    raise RuntimeError(f"unique_path_failed dir={dir_path} base={filename}")


def _resolve_approved_path(approved_dir: str) -> str:
    if approved_dir.lower().endswith(".json") or os.path.isfile(approved_dir):
        return approved_dir
    return os.path.join(approved_dir, "meta_risk_state.json")


def _move_file(src: str, dest: str) -> None:
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest):
        raise RuntimeError(f"destination_exists path={dest}")
    os.replace(src, dest)


def _handle_rejection(candidate_path: str, rejected_dir: str, stamp: str) -> str:
    if not candidate_path or not os.path.exists(candidate_path):
        return ""
    rejected_path = _unique_path(rejected_dir, os.path.basename(candidate_path), stamp)
    _move_file(candidate_path, rejected_path)
    return rejected_path


def _perf_summary_fields(result: promotion_gate.PromotionGateResult) -> Dict[str, Any]:
    details = result.details if isinstance(result.details, dict) else {}
    snap = result.performance_snapshot if isinstance(result.performance_snapshot, dict) else {}
    candidate = snap.get("candidate") if isinstance(snap.get("candidate"), dict) else {}
    replay = snap.get("replay") if isinstance(snap.get("replay"), dict) else {}
    winrate_gate = snap.get("winrate_gate") if isinstance(snap.get("winrate_gate"), dict) else {}
    return {
        "perf_reason": details.get("performance_reason", winrate_gate.get("reason", "n/a")),
        "winrate": candidate.get("winrate", details.get("performance_winrate")),
        "trades": candidate.get("trades", details.get("performance_trades")),
        "days": candidate.get("days", details.get("performance_days")),
        "total_pnl": candidate.get("total_pnl", details.get("performance_total_pnl")),
        "drawdown": candidate.get("max_dd"),
        "perf_history_path": details.get("perf_history_path", replay.get("perf_history_path", "missing")),
    }


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = (result.stdout or "").strip()
        return commit or "unknown"
    except Exception:
        return "unknown"


def _load_valid_perf_history(candidate_path: str, candidate_hash: str) -> tuple[str, Dict[str, Any]]:
    perf_path = perf_history.perf_history_path(candidate_path)
    payload = perf_history.load_perf_history(perf_path)
    if payload is None:
        if not os.path.exists(perf_path):
            raise RuntimeError(f"perf_history_missing path={perf_path}")
        raise RuntimeError(f"perf_history_invalid path={perf_path}")

    ok, missing = perf_history.validate_perf_history(payload)
    if not ok:
        raise RuntimeError(f"perf_history_missing_fields fields={','.join(missing)} path={perf_path}")

    perf_candidate_hash = payload.get("candidate_sha256")
    if perf_candidate_hash != candidate_hash:
        raise RuntimeError(
            f"perf_history_candidate_sha_mismatch perf={perf_candidate_hash} file={candidate_hash}"
        )
    return perf_path, payload


def _load_holdout_report(
    path: str = DEFAULT_HOLDOUT_REPORT_PATH,
    *,
    required: bool = False,
) -> Optional[Dict[str, Any]]:
    source = str(path or "").strip()
    if not source:
        if required:
            raise RuntimeError(f"{SCORING_ERROR_HOLDOUT_REPORT_MISSING} path={path}")
        return None
    if not os.path.exists(source):
        if required:
            raise RuntimeError(f"{SCORING_ERROR_HOLDOUT_REPORT_MISSING} path={source}")
        return None
    try:
        payload, _ = atomic_io.load_json_with_fallback(source)
    except Exception as exc:
        if required:
            raise RuntimeError(f"{SCORING_ERROR_HOLDOUT_REPORT_INVALID} path={source} detail={exc}") from exc
        return None
    if isinstance(payload, dict):
        if required and not isinstance(payload.get("metrics"), dict):
            raise RuntimeError(f"{SCORING_ERROR_HOLDOUT_REPORT_INVALID} path={source} metrics=missing")
        return payload
    if required:
        raise RuntimeError(f"{SCORING_ERROR_HOLDOUT_REPORT_INVALID} path={source} type={type(payload).__name__}")
    return None


def _require_score_float(field_name: str, value: Any) -> float:
    out = _round_float(value, default=None)
    if out is None:
        raise RuntimeError(f"{SCORING_ERROR_SCORE_COMPUTE_FAILED} field={field_name} value={value}")
    return float(out)


def _normalize_scoring_error(exc: Exception) -> str:
    reason = str(exc)
    if reason.startswith("SCORING_ERROR:"):
        return reason
    return f"{SCORING_ERROR_SCORE_COMPUTE_FAILED} detail={reason}"


def _log_scoring_error(
    logger: Optional[Any],
    *,
    cycle_id: str,
    candidate_hash: str,
    reason: str,
) -> None:
    if logger is None:
        return
    logger.error(
        event="scoring_error",
        stage="meta_promote",
        cycle_id=cycle_id,
        candidate_hash=candidate_hash,
        reason=reason,
    )
    return None


def _load_optional_extra_fields(path: str) -> Dict[str, Any]:
    source = str(path or "").strip()
    if not source:
        return {}
    if not os.path.exists(source):
        raise RuntimeError(f"ledger_extra_missing path={source}")
    payload, _ = atomic_io.load_json_with_fallback(source)
    if not isinstance(payload, dict):
        raise RuntimeError(f"ledger_extra_invalid type={type(payload).__name__}")
    return dict(payload)


def _deterministic_hash_from_perf_history(payload: Dict[str, Any]) -> str:
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    data = {
        "candidate_sha256": payload.get("candidate_sha256"),
        "input_hash": run.get("input_hash"),
        "equity_hash": run.get("equity_hash"),
        "regime_ledger_hash": run.get("regime_ledger_hash"),
    }
    return det.hash_json(data)


def _ledger_metrics(
    perf_payload: Dict[str, Any],
    perf_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = perf_history.perf_metrics_from_history(perf_payload)
    summary = perf_summary or {}
    return {
        "winrate": metrics.get("winrate", summary.get("winrate")),
        "total_pnl": metrics.get("total_pnl", summary.get("total_pnl")),
        "drawdown": metrics.get("max_dd", summary.get("drawdown")),
        "trades": metrics.get("trades", summary.get("trades")),
    }


def _append_ledger_entry(ledger_path: str, entry: Dict[str, Any]) -> None:
    atomic_io.safe_append_jsonl(
        ledger_path,
        entry,
        ensure_ascii=False,
        sort_keys=True,
    )


def _base_scoring_fields(
    *,
    enabled: bool,
    w_pnl: float,
    w_win: float,
    w_dd: float,
) -> Dict[str, Any]:
    return {
        "scoring_enabled": int(bool(enabled)),
        "score_total": None,
        "score_in_sample": None,
        "score_holdout": None,
        "score_components": {
            "dd_norm": None,
            "pnl_norm": None,
            "w_dd": _round_float(w_dd, default=1.0),
            "w_pnl": _round_float(w_pnl, default=1.0),
            "w_win": _round_float(w_win, default=1.0),
            "winrate_norm": None,
        },
        "scoring_notes": {
            "dd_missing": 0,
            "normalization_method": "disabled",
            "winrate_missing": 0,
        },
    }


def _compute_scoring_fields(
    *,
    perf_payload: Dict[str, Any],
    enable_scoring: bool,
    w_pnl: float,
    w_win: float,
    w_dd: float,
    holdout_report_path: str = DEFAULT_HOLDOUT_REPORT_PATH,
    require_holdout_report: bool = False,
) -> Dict[str, Any]:
    fields = _base_scoring_fields(enabled=enable_scoring, w_pnl=w_pnl, w_win=w_win, w_dd=w_dd)
    if not enable_scoring:
        return fields

    in_sample_metrics = perf_history.perf_metrics_from_history(perf_payload)
    holdout_report = _load_holdout_report(
        holdout_report_path,
        required=bool(require_holdout_report),
    )
    score_payload = scoring.compute_scores(
        in_sample_metrics,
        holdout_report=holdout_report,
        w_pnl=float(w_pnl),
        w_win=float(w_win),
        w_dd=float(w_dd),
    )

    score_total = _require_score_float("score_total", score_payload.get("score_total"))
    score_in_sample = _require_score_float("score_in_sample", score_payload.get("score_in_sample"))
    score_holdout_raw = score_payload.get("score_holdout")
    if require_holdout_report:
        score_holdout: Optional[float] = _require_score_float("score_holdout", score_holdout_raw)
    else:
        score_holdout = _round_float(score_holdout_raw, default=None)

    fields.update(
        {
            "score_total": score_total,
            "score_in_sample": score_in_sample,
            "score_holdout": score_holdout,
            "score_components": score_payload.get("score_components", fields.get("score_components", {})),
            "scoring_notes": score_payload.get("scoring_notes", fields.get("scoring_notes", {})),
        }
    )
    return fields


def _build_ledger_entry(
    result: promotion_gate.PromotionGateResult,
    perf_payload: Dict[str, Any],
    perf_path: str,
    base_hash: str,
    decision: str,
    reason: str = "",
    scoring_fields: Optional[Dict[str, Any]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    replay_hash = result.replay_hash
    if not replay_hash or replay_hash == "missing":
        replay_payload = {k: v for k, v in perf_payload.items() if k != "timestamps"}
        replay_hash = det.hash_json(replay_payload)
    entry: Dict[str, Any] = {
        "timestamp_utc": _utc_iso(),
        "git_commit_hash": _git_commit_hash(),
        "base_hash": base_hash or "missing",
        "candidate_hash": result.candidate_hash,
        "replay_hash": replay_hash,
        "deterministic_hash": _deterministic_hash_from_perf_history(perf_payload),
        "decision": decision,
        "metrics": _ledger_metrics(perf_payload, _perf_summary_fields(result)),
        "reason": reason if decision == "rejected" else "",
        "perf_history_path": perf_path,
    }
    if isinstance(scoring_fields, dict):
        entry.update(scoring_fields)
    if isinstance(extra_fields, dict):
        for key, value in extra_fields.items():
            if key in entry:
                continue
            entry[str(key)] = value
    return entry


def _archive_approved(approved_path: str, history_dir: str, stamp: str) -> str:
    if not approved_path or not os.path.exists(approved_path):
        raise RuntimeError(f"approved_missing path={approved_path}")
    os.makedirs(history_dir, exist_ok=True)
    archived_path = _unique_path(history_dir, os.path.basename(approved_path), stamp)
    _atomic_copy_file(approved_path, archived_path)
    return archived_path


def _atomic_promote(
    candidate_path: str,
    approved_path: str,
    history_dir: str,
    stamp: str,
) -> str:
    archived_path = _archive_approved(approved_path, history_dir, stamp)
    _atomic_replace_from_source(candidate_path, approved_path)
    return archived_path


def run(
    args: argparse.Namespace,
    *,
    logger: Optional[Any] = None,
    cycle_id: str = "",
) -> int:
    strict = int(args.strict) == 1
    replay_check = int(args.replay_check) == 1
    no_exit_on_reject = int(args.no_exit_on_reject) == 1
    debug = int(args.debug) == 1
    eval_only = int(getattr(args, "eval_only", 0)) == 1
    eval_report_path = str(getattr(args, "eval_report_path", "") or "").strip()
    holdout_report_path = str(
        getattr(args, "holdout_report_path", DEFAULT_HOLDOUT_REPORT_PATH) or DEFAULT_HOLDOUT_REPORT_PATH
    )
    enable_scoring = int(getattr(args, "enable_scoring", 0)) == 1
    w_pnl = float(getattr(args, "w_pnl", 1.0))
    w_win = float(getattr(args, "w_win", 1.0))
    w_dd = float(getattr(args, "w_dd", 1.0))

    candidate_path = args.candidate_path
    approved_path = _resolve_approved_path(args.approved_dir)
    if str(getattr(args, "cycle_id", "") or "").strip():
        print(f"[promote] cycle_id={str(args.cycle_id).strip()}")
    print(
        f"[promote] candidate_path={candidate_path} "
        f"perf_history={perf_history.perf_history_path(candidate_path)}"
    )

    gate_cfg = promotion_gate.PromotionGateConfig(
        perf_days=int(args.perf_days),
        min_winrate=float(args.promote_min_winrate),
        min_trades_for_gate=int(args.promote_min_trades),
        min_days_for_gate=int(args.promote_min_days),
    )

    result = promotion_gate.evaluate_candidate(
        candidate_path=candidate_path,
        approved_path=approved_path,
        csv_path=args.csv,
        model_path=args.model_path,
        strict=strict,
        replay_check=replay_check,
        cfg=gate_cfg,
        debug=debug,
    )

    stamp = _timestamp()
    perf_summary = _perf_summary_fields(result)
    base_hash = result.approved_hash if result is not None else "missing"
    scoring_fields = _base_scoring_fields(
        enabled=enable_scoring,
        w_pnl=w_pnl,
        w_win=w_win,
        w_dd=w_dd,
    )
    try:
        ledger_extra_fields = _load_optional_extra_fields(str(getattr(args, "ledger_extra_path", "") or ""))
    except Exception as exc:
        _print_summary_line(
            "FAIL",
            str(exc),
            result,
            perf_summary,
            perf_reason=perf_summary.get("perf_reason"),
        )
        return ExitCode.DATA_INVALID

    if eval_only:
        gate_status = "PASS"
        gate_reason = str(result.reason)
        eval_rc = int(EXIT_OK)
        if not result.ok:
            processing_error_exit_code = _processing_error_exit_code(result.reason)
            if processing_error_exit_code is not None:
                gate_status = "FAIL"
                eval_rc = int(processing_error_exit_code)
            else:
                gate_status = "REJECT"
                eval_rc = int(EXIT_OK)

        if gate_status in ("PASS", "REJECT"):
            try:
                perf_path, perf_payload = _load_valid_perf_history(candidate_path, result.candidate_hash)
            except Exception as exc:
                gate_status = "FAIL"
                gate_reason = str(exc)
                eval_rc = int(ExitCode.DATA_INVALID)
            else:
                try:
                    scoring_fields = _compute_scoring_fields(
                        perf_payload=perf_payload,
                        enable_scoring=enable_scoring,
                        w_pnl=w_pnl,
                        w_win=w_win,
                        w_dd=w_dd,
                        holdout_report_path=holdout_report_path,
                        require_holdout_report=enable_scoring,
                    )
                except Exception as exc:
                    gate_status = "FAIL"
                    gate_reason = _normalize_scoring_error(exc)
                    eval_rc = int(EXIT_CODE_SCORING_FAIL)
                    _log_scoring_error(
                        logger,
                        cycle_id=cycle_id,
                        candidate_hash=result.candidate_hash,
                        reason=gate_reason,
                    )
                else:
                    if logger is not None:
                        logger.info(
                            event="scoring_computed",
                            stage="meta_promote",
                            cycle_id=cycle_id,
                            candidate_hash=result.candidate_hash,
                            score_total=scoring_fields.get("score_total"),
                            score_holdout=scoring_fields.get("score_holdout"),
                            gate_status=gate_status,
                        )

        eval_payload: Dict[str, Any] = {
            "gate_status": gate_status,
            "gate_reason": gate_reason,
            "candidate_hash": result.candidate_hash,
            "approved_hash": result.approved_hash,
            "replay_hash": result.replay_hash,
            "score_total": scoring_fields.get("score_total"),
            "score_in_sample": scoring_fields.get("score_in_sample"),
            "score_holdout": scoring_fields.get("score_holdout"),
            "score_components": scoring_fields.get("score_components"),
            "scoring_notes": scoring_fields.get("scoring_notes"),
            "scoring_enabled": int(enable_scoring),
        }
        if eval_report_path:
            try:
                _atomic_write_json(eval_report_path, eval_payload)
            except Exception as exc:
                _print_summary_line(
                    "FAIL",
                    f"eval_report_write_failed {exc}",
                    result,
                    perf_summary,
                    perf_reason=perf_summary.get("perf_reason"),
                )
                return ExitCode.STATE_CORRUPT

        _print_summary_line(
            gate_status,
            gate_reason,
            result,
            perf_summary,
            perf_reason=perf_summary.get("perf_reason"),
            score_total=_safe_summary_value(scoring_fields.get("score_total")),
            score_holdout=_safe_summary_value(scoring_fields.get("score_holdout")),
            scoring_enabled=int(enable_scoring),
            eval_only=1,
        )
        return int(eval_rc)

    if not result.ok:
        processing_error_exit_code = _processing_error_exit_code(result.reason)
        if processing_error_exit_code is not None:
            _print_summary_line(
                "FAIL",
                result.reason,
                result,
                perf_summary,
                perf_reason=perf_summary.get("perf_reason"),
            )
            return int(processing_error_exit_code)

        try:
            perf_path, perf_payload = _load_valid_perf_history(candidate_path, result.candidate_hash)
        except Exception as exc:
            _print_summary_line(
                "FAIL",
                f"ledger_entry_failed {exc}",
                result,
                perf_summary,
                perf_reason=perf_summary.get("perf_reason"),
            )
            return ExitCode.STATE_CORRUPT
        try:
            scoring_fields = _compute_scoring_fields(
                perf_payload=perf_payload,
                enable_scoring=enable_scoring,
                w_pnl=w_pnl,
                w_win=w_win,
                w_dd=w_dd,
                holdout_report_path=holdout_report_path,
                require_holdout_report=enable_scoring,
            )
        except Exception as exc:
            scoring_reason = _normalize_scoring_error(exc)
            _print_summary_line(
                "FAIL",
                scoring_reason,
                result,
                perf_summary,
                perf_reason=perf_summary.get("perf_reason"),
                score_total=_safe_summary_value(scoring_fields.get("score_total")),
                score_holdout=_safe_summary_value(scoring_fields.get("score_holdout")),
                scoring_enabled=int(enable_scoring),
            )
            _log_scoring_error(
                logger,
                cycle_id=cycle_id,
                candidate_hash=result.candidate_hash,
                reason=scoring_reason,
            )
            return int(EXIT_CODE_SCORING_FAIL)
        try:
            ledger_entry = _build_ledger_entry(
                result=result,
                perf_payload=perf_payload,
                perf_path=perf_path,
                base_hash=base_hash,
                decision="rejected",
                reason=result.reason,
                scoring_fields=scoring_fields,
                extra_fields=ledger_extra_fields,
            )
        except Exception as exc:
            _print_summary_line(
                "FAIL",
                f"ledger_entry_failed {exc}",
                result,
                perf_summary,
                perf_reason=perf_summary.get("perf_reason"),
            )
            return ExitCode.STATE_CORRUPT

        rejected_path = ""
        try:
            rejected_path = _handle_rejection(candidate_path, args.rejected_dir, stamp)
        except Exception as exc:
            _print_summary_line(
                "FAIL",
                f"{result.reason} rejected_move_failed={exc}",
                result,
                perf_summary,
                perf_reason=perf_summary.get("perf_reason"),
            )
            return ExitCode.STATE_CORRUPT

        try:
            _append_ledger_entry(args.ledger_path, ledger_entry)
        except Exception as exc:
            rollback_err = None
            try:
                if rejected_path and os.path.exists(rejected_path) and not os.path.exists(candidate_path):
                    _move_file(rejected_path, candidate_path)
            except Exception as rollback_exc:
                rollback_err = rollback_exc
            reason = f"ledger_write_failed {exc}"
            if rollback_err is not None:
                reason = f"{reason} rollback_failed={rollback_err}"
            _print_summary_line(
                "FAIL",
                reason,
                result,
                perf_summary,
                perf_reason=perf_summary.get("perf_reason"),
            )
            return ExitCode.STATE_CORRUPT

        _print_summary_line(
            "REJECT",
            result.reason,
            result,
            perf_summary,
            perf_reason=perf_summary.get("perf_reason"),
            rejected_path=rejected_path or "missing",
            ledger_path=args.ledger_path,
            score_total=_safe_summary_value(scoring_fields.get("score_total")),
            score_holdout=_safe_summary_value(scoring_fields.get("score_holdout")),
            scoring_enabled=int(enable_scoring),
        )
        if no_exit_on_reject:
            return EXIT_OK
        return EXIT_REJECT

    if not approved_path or not os.path.exists(approved_path):
        _print_summary_line(
            "FAIL",
            f"approved_missing path={approved_path}",
            result,
            perf_summary,
            perf_reason=perf_summary.get("perf_reason"),
        )
        return ExitCode.DATA_INVALID

    try:
        perf_path, perf_payload = _load_valid_perf_history(candidate_path, result.candidate_hash)
    except Exception as exc:
        _print_summary_line(
            "FAIL",
            str(exc),
            result=result,
            perf_summary=perf_summary,
            perf_reason=perf_summary.get("perf_reason"),
        )
        return ExitCode.DATA_INVALID

    try:
        scoring_fields = _compute_scoring_fields(
            perf_payload=perf_payload,
            enable_scoring=enable_scoring,
            w_pnl=w_pnl,
            w_win=w_win,
            w_dd=w_dd,
            holdout_report_path=holdout_report_path,
            require_holdout_report=enable_scoring,
        )
    except Exception as exc:
        scoring_reason = _normalize_scoring_error(exc)
        _print_summary_line(
            "FAIL",
            scoring_reason,
            result=result,
            perf_summary=perf_summary,
            perf_reason=perf_summary.get("perf_reason"),
            score_total=_safe_summary_value(scoring_fields.get("score_total")),
            score_holdout=_safe_summary_value(scoring_fields.get("score_holdout")),
            scoring_enabled=int(enable_scoring),
        )
        _log_scoring_error(
            logger,
            cycle_id=cycle_id,
            candidate_hash=result.candidate_hash,
            reason=scoring_reason,
        )
        return int(EXIT_CODE_SCORING_FAIL)
    if enable_scoring and logger is not None:
        logger.info(
            event="scoring_computed",
            stage="meta_promote",
            cycle_id=cycle_id,
            candidate_hash=result.candidate_hash,
            score_total=scoring_fields.get("score_total"),
            score_holdout=scoring_fields.get("score_holdout"),
        )

    try:
        archived_path = _atomic_promote(
            candidate_path=candidate_path,
            approved_path=approved_path,
            history_dir=args.history_dir,
            stamp=stamp,
        )
    except Exception as exc:
        reason = f"atomic_replace_failed {exc}"
        _print_summary_line(
            "FAIL",
            reason=reason,
            result=result,
            perf_summary=perf_summary,
            perf_reason=perf_summary.get("perf_reason"),
        )
        return ExitCode.STATE_CORRUPT

    try:
        ledger_entry = _build_ledger_entry(
            result=result,
            perf_payload=perf_payload,
            perf_path=perf_path,
            base_hash=base_hash,
            decision="approved",
            scoring_fields=scoring_fields,
            extra_fields=ledger_extra_fields,
        )
        _append_ledger_entry(args.ledger_path, ledger_entry)
    except Exception as exc:
        restore_err = None
        try:
            _atomic_replace_from_source(archived_path, approved_path)
        except Exception as restore_exc:
            restore_err = restore_exc
        reason = f"ledger_write_failed {exc}"
        if restore_err is not None:
            reason = f"{reason} restore_failed={restore_err}"
        _print_summary_line(
            "FAIL",
            reason=reason,
            result=result,
            perf_summary=perf_summary,
            perf_reason=perf_summary.get("perf_reason"),
        )
        return ExitCode.STATE_CORRUPT

    manifest = {
        "promoted_at": _utc_iso(),
        "reason": str(args.reason),
        "approved_hash": result.approved_hash,
        "candidate_hash": result.candidate_hash,
        "replay_hash": result.replay_hash,
        "deterministic_hash": _deterministic_hash_from_perf_history(perf_payload),
        "performance_snapshot": result.performance_snapshot,
        "candidate_path": candidate_path,
        "approved_path": approved_path,
        "archived_path": archived_path,
        "perf_history_path": perf_path,
        "ledger_path": args.ledger_path,
    }
    manifest_path = _unique_path(args.history_dir, "promotion_manifest.json", stamp)
    _atomic_write_json(manifest_path, manifest)

    _print_summary_line(
        "PASS",
        result.reason,
        result,
        perf_summary,
        perf_reason=perf_summary.get("perf_reason"),
        archived_path=archived_path,
        manifest_path=manifest_path,
        ledger_path=args.ledger_path,
        score_total=_safe_summary_value(scoring_fields.get("score_total")),
        score_holdout=_safe_summary_value(scoring_fields.get("score_holdout")),
        scoring_enabled=int(enable_scoring),
    )
    return EXIT_OK


def main() -> int:
    args = _parse_args()
    cycle_id = str(getattr(args, "cycle_id", "") or "").strip()
    logger = get_logger("meta_promote").bind(cycle_id=cycle_id)
    logger.info(
        event="stage_start",
        stage="meta_promote",
        cycle_id=cycle_id,
        candidate_path=args.candidate_path,
        strict=int(args.strict),
        reason=str(args.reason),
    )
    t0 = time.perf_counter()
    rc = int(ExitCode.INTERNAL_ERROR)
    try:
        rc = int(run(args, logger=logger, cycle_id=cycle_id))
        return int(rc)
    except Exception as exc:
        logger.error(
            event="exception",
            stage="meta_promote",
            cycle_id=cycle_id,
            exc_type=type(exc).__name__,
            message=str(exc),
        )
        _print_summary_line("FAIL", str(exc), result=None, perf_summary=None)
        rc = int(ExitCode.INTERNAL_ERROR)
        return int(rc)
    finally:
        logger.info(
            event="stage_end",
            stage="meta_promote",
            cycle_id=cycle_id,
            rc=int(rc),
            duration_s=round(time.perf_counter() - t0, 6),
        )


if __name__ == "__main__":
    sys.exit(main())
