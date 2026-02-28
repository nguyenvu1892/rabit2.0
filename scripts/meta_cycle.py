#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

from rabit.state import atomic_io
from rabit.state.anomaly_guardrail import detect_anomaly
from rabit.state.exit_codes import ExitCode
from rabit.state.file_lock import acquire_exclusive_lock, lock_owner_summary, release_exclusive_lock
from rabit.utils import StructuredLogger, generate_cycle_id, get_logger

EXIT_OK = ExitCode.SUCCESS
EXIT_REJECT = ExitCode.BUSINESS_REJECT
EXIT_LOCK_RETRYABLE = ExitCode.BUSINESS_SKIP
EXIT_ERROR = ExitCode.INTERNAL_ERROR
EXIT_ANOMALY_HALT = ExitCode.ANOMALY_HALT

DEFAULT_APPROVED_STATE_PATH = os.path.join("data", "meta_states", "current_approved", "meta_risk_state.json")
DEFAULT_CANDIDATE_PATH = os.path.join("data", "meta_states", "candidate", "meta_risk_state.json")
DEFAULT_CANDIDATE_MANIFEST = os.path.join("data", "meta_states", "candidate", "manifest.json")
DEFAULT_SHADOW_REPORT = os.path.join("data", "reports", "shadow_replay_report.json")
DEFAULT_LEDGER_PATH = os.path.join("data", "meta_states", "ledger.jsonl")
DEFAULT_HEALTH_OUT_DIR = os.path.join("data", "reports", "meta_cycle_healthcheck")
DEFAULT_INCIDENT_DIR = os.path.join("data", "reports", "incidents")
DEFAULT_LOCK_PATH = os.path.join("data", "meta_states", ".locks", "meta_cycle.lock")

_PROMOTION_STATUS_RE = re.compile(r"\[promotion\]\s+STATUS=([A-Z]+)")
_CANDIDATE_HASH_RE = re.compile(r"\bcandidate_hash=([0-9a-f]{64}|missing)\b")
_LEDGER_PATH_RE = re.compile(r"\bledger_path=([^\s]+)")
_DETERMINISTIC_STATUS_RE = re.compile(r"\[deterministic\]\s+STATUS=(PASS|FAIL)")
_HEALTHCHECK_STATUS_RE = re.compile(r"\[healthcheck\]\s+STATUS=(PASS|FAIL)")


@dataclass
class StageResult:
    name: str
    return_code: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def combined_output(self) -> str:
        if self.stdout and self.stderr:
            return self.stdout + "\n" + self.stderr
        return self.stdout or self.stderr or ""


@dataclass
class CyclePaths:
    approved_state_path: str
    candidate_path: str
    candidate_manifest_path: str
    shadow_out_json: str
    promote_approved_dir: str
    promote_history_dir: str
    promote_rejected_dir: str
    promote_ledger_path: str
    health_out_dir: str


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Auto Cycle Orchestrator for Phase 4 governance pipeline.")
    ap.add_argument("--csv", required=True, help="Input bars CSV path")
    ap.add_argument("--reason", required=True, help="Cycle reason string")
    ap.add_argument("--strict", type=int, choices=[0, 1], default=1, help="Strict mode (0/1)")
    ap.add_argument(
        "--lock_path",
        default=DEFAULT_LOCK_PATH,
        help=f"Global lock path (default: {DEFAULT_LOCK_PATH})",
    )
    ap.add_argument(
        "--lock_ttl_sec",
        type=float,
        default=1800.0,
        help="Lock TTL seconds for stale detection (default: 1800)",
    )
    ap.add_argument(
        "--force_lock_break",
        type=int,
        choices=[0, 1],
        default=0,
        help="Allow stale lock takeover when age > lock_ttl_sec (0/1)",
    )
    ap.add_argument(
        "--on_lock_held",
        choices=["skip", "retryable", "fail"],
        default=None,
        help="Lock-held action override. Default: strict=1->retryable, strict=0->skip",
    )
    ap.add_argument(
        "--skip_global_lock",
        type=int,
        choices=[0, 1],
        default=0,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--risk_per_trade", type=float, default=None, help="Optional passthrough to validation sims")
    ap.add_argument(
        "--account_equity_start",
        type=float,
        default=None,
        help="Optional passthrough to validation sims",
    )
    ap.add_argument("--dry_run", type=int, choices=[0, 1], default=0, help="Dry-run mode (0/1)")
    ap.add_argument(
        "--no_exit_on_reject",
        type=int,
        choices=[0, 1],
        default=0,
        help="Pass-through to meta_promote (0/1)",
    )
    ap.add_argument(
        "--enable_scoring",
        type=int,
        choices=[0, 1],
        default=0,
        help="Pass-through to meta_promote scoring layer (0/1)",
    )
    ap.add_argument("--w_pnl", type=float, default=1.0, help="Pass-through scoring weight for PnL.")
    ap.add_argument("--w_win", type=float, default=1.0, help="Pass-through scoring weight for winrate.")
    ap.add_argument("--w_dd", type=float, default=1.0, help="Pass-through scoring weight for drawdown.")
    ap.add_argument(
        "--enable_anomaly_guard",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable anomaly guardrail between meta_promote and healthcheck (0/1)",
    )
    ap.add_argument(
        "--auto_rollback",
        type=int,
        choices=[0, 1],
        default=0,
        help="On anomaly, auto-run rollback --steps 1 in protected mode (0/1)",
    )
    ap.add_argument(
        "--daily_dd_limit",
        type=float,
        default=0.15,
        help="Anomaly threshold for |daily_drawdown_pct| (default: 0.15)",
    )
    ap.add_argument(
        "--pnl_jump_abs_limit",
        type=float,
        default=100.0,
        help="Anomaly threshold for |current_total_pnl - previous_total_pnl| (default: 100)",
    )
    ap.add_argument(
        "--trades_spike_limit",
        type=int,
        default=2000,
        help="Anomaly threshold for trades_today (default: 2000)",
    )
    ap.add_argument(
        "--equity_drift_abs_limit",
        type=float,
        default=5.0,
        help="Anomaly threshold for |summary_equity - equity_csv_last| (default: 5)",
    )
    ap.add_argument(
        "--simulate_anomaly",
        type=int,
        choices=[0, 1],
        default=0,
        help="Force anomaly guardrail detection for test validation (0/1)",
    )
    ap.add_argument("--skip_replay", type=int, choices=[0, 1], default=0, help="Skip shadow replay step")
    ap.add_argument("--skip_promote", type=int, choices=[0, 1], default=0, help="Skip promotion gate step")
    ap.add_argument("--cycle_id", default="", help="Optional correlation id propagated across cycle stages")
    return ap.parse_args()


def _log(message: str) -> None:
    print(f"[cycle] {message}")


def _lock_log(message: str) -> None:
    print(f"[lock] {message}")


def _effective_lock_policy(args: argparse.Namespace) -> str:
    requested = args.on_lock_held
    if requested:
        return str(requested).strip().lower()
    strict = int(args.strict) == 1
    return "retryable" if strict else "skip"


def _lock_policy_exit_code(policy: str) -> int:
    normalized = str(policy or "skip").strip().lower()
    if normalized == "retryable":
        return int(EXIT_LOCK_RETRYABLE)
    if normalized == "fail":
        return int(EXIT_ERROR)
    return int(EXIT_OK)


def _shell_join(parts: List[str]) -> str:
    try:
        return shlex.join(parts)
    except Exception:
        return " ".join(parts)


def _run_module(
    stage: str,
    module_name: str,
    module_args: List[str],
    *,
    logger: StructuredLogger,
    cycle_id: str,
) -> StageResult:
    cmd = [sys.executable, "-m", module_name] + module_args
    _log(f"stage={stage} start cmd={_shell_join(cmd)}")
    logger.info(
        event="stage_start",
        stage=stage,
        cycle_id=cycle_id,
        module_name=module_name,
        cmd=_shell_join(cmd),
    )
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        dt_s = time.perf_counter() - t0
        logger.error(
            event="exception",
            stage=stage,
            cycle_id=cycle_id,
            exc_type=type(exc).__name__,
            message=str(exc),
            duration_s=round(dt_s, 6),
        )
        raise
    dt_s = time.perf_counter() - t0
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout:
        for line in stdout.splitlines():
            print(f"[cycle][{stage}][stdout] {line}")
    if stderr:
        for line in stderr.splitlines():
            print(f"[cycle][{stage}][stderr] {line}")
    _log(f"stage={stage} end rc={proc.returncode} duration_s={dt_s:.3f}")
    logger.info(
        event="stage_end",
        stage=stage,
        cycle_id=cycle_id,
        rc=int(proc.returncode),
        duration_s=round(dt_s, 6),
    )
    return StageResult(
        name=stage,
        return_code=int(proc.returncode),
        stdout=stdout,
        stderr=stderr,
        duration_s=float(dt_s),
    )


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_sha256(text: Optional[str]) -> bool:
    if not isinstance(text, str):
        return False
    if len(text) != 64:
        return False
    return all(c in "0123456789abcdef" for c in text.lower())


def _load_candidate_hash(candidate_manifest_path: str, candidate_path: str) -> str:
    if os.path.exists(candidate_manifest_path):
        try:
            payload, _ = atomic_io.load_json_with_fallback(candidate_manifest_path)
            if isinstance(payload, dict):
                candidate_sha = str(payload.get("candidate_sha256", "")).strip().lower()
                if _is_sha256(candidate_sha):
                    return candidate_sha
        except Exception:
            pass
    if os.path.exists(candidate_path):
        try:
            return _sha256_file(candidate_path)
        except Exception:
            return "missing"
    return "missing"


def _last_status(text: str, pattern: re.Pattern[str]) -> str:
    status = ""
    for match in pattern.finditer(text):
        status = str(match.group(1)).strip().upper()
    return status


def _parse_promotion_stage(result: StageResult) -> tuple[str, str, str]:
    combined = result.combined_output
    status = _last_status(combined, _PROMOTION_STATUS_RE)
    candidate_hash = ""
    ledger_path = ""

    lines = combined.splitlines()
    for line in reversed(lines):
        if "[promotion]" not in line:
            continue
        if not status:
            status_match = _PROMOTION_STATUS_RE.search(line)
            if status_match:
                status = str(status_match.group(1)).strip().upper()
        if not candidate_hash:
            hash_match = _CANDIDATE_HASH_RE.search(line)
            if hash_match:
                candidate_hash = str(hash_match.group(1)).strip().lower()
        if not ledger_path:
            ledger_match = _LEDGER_PATH_RE.search(line)
            if ledger_match:
                ledger_path = str(ledger_match.group(1)).strip()
        if status and candidate_hash and ledger_path:
            break

    if not candidate_hash:
        hash_match = _CANDIDATE_HASH_RE.search(combined)
        if hash_match:
            candidate_hash = str(hash_match.group(1)).strip().lower()
    if not ledger_path:
        ledger_match = _LEDGER_PATH_RE.search(combined)
        if ledger_match:
            ledger_path = str(ledger_match.group(1)).strip()

    return status, candidate_hash, ledger_path


def _build_paths(dry_run: bool, temp_root: Optional[str]) -> CyclePaths:
    if not dry_run:
        return CyclePaths(
            approved_state_path=DEFAULT_APPROVED_STATE_PATH,
            candidate_path=DEFAULT_CANDIDATE_PATH,
            candidate_manifest_path=DEFAULT_CANDIDATE_MANIFEST,
            shadow_out_json=DEFAULT_SHADOW_REPORT,
            promote_approved_dir=os.path.join("data", "meta_states", "current_approved"),
            promote_history_dir=os.path.join("data", "meta_states", "history"),
            promote_rejected_dir=os.path.join("data", "meta_states", "rejected"),
            promote_ledger_path=DEFAULT_LEDGER_PATH,
            health_out_dir=DEFAULT_HEALTH_OUT_DIR,
        )

    if not temp_root:
        raise RuntimeError("dry_run_temp_root_missing")

    return CyclePaths(
        approved_state_path=DEFAULT_APPROVED_STATE_PATH,
        candidate_path=os.path.join(temp_root, "candidate", "meta_risk_state.json"),
        candidate_manifest_path=os.path.join(temp_root, "candidate", "manifest.json"),
        shadow_out_json=os.path.join(temp_root, "shadow_replay_report.json"),
        promote_approved_dir=os.path.join(temp_root, "current_approved"),
        promote_history_dir=os.path.join(temp_root, "history"),
        promote_rejected_dir=os.path.join(temp_root, "rejected"),
        promote_ledger_path=os.path.join(temp_root, "ledger.jsonl"),
        health_out_dir=os.path.join(temp_root, "healthcheck"),
    )


def _prepare_dry_run_state(paths: CyclePaths) -> None:
    approved_src = paths.approved_state_path
    approved_dst = os.path.join(paths.promote_approved_dir, "meta_risk_state.json")
    if not os.path.exists(approved_src):
        raise RuntimeError(f"approved_state_missing path={approved_src}")
    os.makedirs(paths.promote_approved_dir, exist_ok=True)
    shutil.copy2(approved_src, approved_dst)


def _append_optional_risk_args(cli: List[str], args: argparse.Namespace) -> None:
    if args.risk_per_trade is not None:
        cli.extend(["--risk_per_trade", str(float(args.risk_per_trade))])
    if args.account_equity_start is not None:
        cli.extend(["--account_equity_start", str(float(args.account_equity_start))])


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_incident_report(
    *,
    cycle_id: str,
    reason_code: str,
    metrics_snapshot: dict,
    incident_dir: str = DEFAULT_INCIDENT_DIR,
) -> str:
    path = os.path.join(incident_dir, f"{cycle_id}.json")
    payload = {
        "ts_utc": _utc_iso(),
        "cycle_id": str(cycle_id),
        "reason_code": str(reason_code),
        "metrics_snapshot": dict(metrics_snapshot or {}),
    }
    atomic_io.atomic_write_json(path, payload, ensure_ascii=False, sort_keys=True, indent=2)
    return path


def _run_auto_rollback_protected(
    *,
    logger: StructuredLogger,
    cycle_id: str,
    reason_code: str,
) -> None:
    rollback_reason = f"anomaly_auto_rollback cycle_id={cycle_id} reason_code={reason_code}"
    try:
        rollback_stage = _run_module(
            "anomaly_auto_rollback",
            "scripts.meta_rollback",
            [
                "--steps",
                "1",
                "--reason",
                rollback_reason,
                "--strict",
                "1",
                "--dry_run",
                "0",
            ],
            logger=logger,
            cycle_id=cycle_id,
        )
        logger.info(
            event="anomaly_auto_rollback_result",
            stage="anomaly_guardrail",
            cycle_id=cycle_id,
            rollback_rc=int(rollback_stage.return_code),
            reason_code=str(reason_code),
        )
    except Exception as exc:
        logger.error(
            event="anomaly_auto_rollback_exception",
            stage="anomaly_guardrail",
            cycle_id=cycle_id,
            reason_code=str(reason_code),
            exc_type=type(exc).__name__,
            message=str(exc),
        )


def _print_final_summary(
    *,
    cycle_id: str,
    cycle_status: str,
    candidate_hash: str,
    decision: str,
    ledger_path: str,
    deterministic_status: str,
    healthcheck_status: str,
    total_runtime_seconds: float,
) -> None:
    _log("summary_begin")
    print(f"[cycle] cycle_id={cycle_id}")
    print(f"[cycle] cycle_status={cycle_status}")
    print(f"[cycle] candidate_hash={candidate_hash}")
    print(f"[cycle] decision={decision}")
    print(f"[cycle] ledger_path={ledger_path}")
    print(f"[cycle] deterministic_status={deterministic_status}")
    print(f"[cycle] healthcheck_status={healthcheck_status}")
    print(f"[cycle] total_runtime_seconds={total_runtime_seconds:.3f}")
    _log("summary_end")


def _run_cycle(
    args: argparse.Namespace,
    paths: CyclePaths,
    dry_run: bool,
    *,
    logger: StructuredLogger,
    cycle_id: str,
) -> int:
    strict = int(args.strict) == 1
    skip_replay = int(args.skip_replay) == 1
    skip_promote = int(args.skip_promote) == 1

    t_total = time.perf_counter()
    candidate_hash = "missing"
    decision = "SKIPPED" if skip_promote else "UNKNOWN"
    deterministic_status = "SKIPPED"
    healthcheck_status = "SKIPPED"
    cycle_status = "FAIL"
    ledger_path = paths.promote_ledger_path
    final_rc = int(EXIT_ERROR)

    def _finish(code: int) -> int:
        nonlocal final_rc
        final_rc = int(code)
        return final_rc

    try:
        if not os.path.exists(paths.approved_state_path):
            _log(f"approved_state_missing path={paths.approved_state_path}")
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)

        # Step 1: candidate generation
        step1 = _run_module(
            "candidate_generate",
            "scripts.meta_candidate_generate",
            [
                "--approved_state_path",
                paths.approved_state_path,
                "--candidate_out_path",
                paths.candidate_path,
                "--candidate_manifest",
                paths.candidate_manifest_path,
                "--reason",
                args.reason,
                "--strict",
                str(int(args.strict)),
            ],
            logger=logger,
            cycle_id=cycle_id,
        )
        if step1.return_code == EXIT_ERROR:
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)
        if step1.return_code != EXIT_OK:
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)

        candidate_hash = _load_candidate_hash(paths.candidate_manifest_path, paths.candidate_path)

        # Step 2: shadow replay
        if skip_replay:
            _log("stage=shadow_replay skipped=1")
            logger.info(event="stage_skip", stage="shadow_replay", cycle_id=cycle_id, skipped=1)
        else:
            step2 = _run_module(
                "shadow_replay",
                "scripts.shadow_replay",
                [
                    "--csv",
                    args.csv,
                    "--meta_state_path",
                    paths.candidate_path,
                    "--out_json",
                    paths.shadow_out_json,
                    "--strict",
                    str(int(args.strict)),
                    "--deterministic_check",
                    "1",
                    "--cycle_id",
                    cycle_id,
                ],
                logger=logger,
                cycle_id=cycle_id,
            )
            if step2.return_code == EXIT_ERROR:
                cycle_status = "FAIL"
                return _finish(EXIT_ERROR)
            if step2.return_code != EXIT_OK:
                cycle_status = "FAIL"
                return _finish(EXIT_ERROR)

        # Step 3: promotion gate
        if skip_promote:
            decision = "SKIPPED"
            _log("stage=meta_promote skipped=1")
            logger.info(event="stage_skip", stage="meta_promote", cycle_id=cycle_id, skipped=1)
        else:
            promote_cli = [
                "--candidate_path",
                paths.candidate_path,
                "--reason",
                args.reason,
                "--strict",
                str(int(args.strict)),
                "--csv",
                args.csv,
                "--no_exit_on_reject",
                "1" if dry_run else str(int(args.no_exit_on_reject)),
                "--enable_scoring",
                str(int(args.enable_scoring)),
                "--w_pnl",
                str(float(args.w_pnl)),
                "--w_win",
                str(float(args.w_win)),
                "--w_dd",
                str(float(args.w_dd)),
                "--cycle_id",
                cycle_id,
            ]
            if dry_run:
                promote_cli.extend(
                    [
                        "--approved_dir",
                        paths.promote_approved_dir,
                        "--history_dir",
                        paths.promote_history_dir,
                        "--rejected_dir",
                        paths.promote_rejected_dir,
                        "--ledger_path",
                        paths.promote_ledger_path,
                    ]
                )
            step3 = _run_module(
                "meta_promote",
                "scripts.meta_promote",
                promote_cli,
                logger=logger,
                cycle_id=cycle_id,
            )
            if step3.return_code >= ExitCode.DATA_INVALID:
                cycle_status = "FAIL"
                return _finish(int(step3.return_code))
            if step3.return_code == EXIT_ERROR:
                cycle_status = "FAIL"
                return _finish(EXIT_ERROR)

            promote_status, promote_hash, promote_ledger_path = _parse_promotion_stage(step3)
            if _is_sha256(promote_hash):
                candidate_hash = promote_hash
            if promote_ledger_path:
                ledger_path = promote_ledger_path

            if promote_status == "REJECT" or step3.return_code == EXIT_REJECT:
                decision = "REJECTED"
            elif promote_status == "PASS":
                decision = "APPROVED"
            elif promote_status == "FAIL":
                decision = "ERROR"
            elif step3.return_code == EXIT_OK:
                # meta_promote can return 0 on reject when no_exit_on_reject=1; fallback is optimistic only
                # when no explicit status can be parsed.
                decision = "APPROVED"
            else:
                decision = "ERROR"

            if decision == "ERROR":
                cycle_status = "FAIL"
                return _finish(EXIT_ERROR)

            if strict and not dry_run and decision == "REJECTED":
                _log("strict_reject=1 -> stop_before_post_validation")
                cycle_status = "SUCCESS_WITH_REJECT"
                return _finish(EXIT_REJECT)

            if decision == "REJECTED" and not strict:
                _log("non_strict_reject_allowed=1 -> continue")
            if decision == "REJECTED" and dry_run:
                _log("dry_run_reject_simulated=1 -> continue")

        # Step 4: post-validation
        det_cli = [
            "--csv",
            args.csv,
            "--meta_risk",
            "1",
            "--meta_feedback",
            "1",
            "--meta_state_path",
            paths.approved_state_path,
            "--deterministic_check",
            "1",
        ]
        _append_optional_risk_args(det_cli, args)
        det_stage = _run_module(
            "deterministic_check",
            "scripts.live_sim_daily",
            det_cli,
            logger=logger,
            cycle_id=cycle_id,
        )
        if det_stage.return_code == EXIT_ERROR:
            deterministic_status = "FAIL"
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)
        if det_stage.return_code != EXIT_OK:
            deterministic_status = "FAIL"
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)

        det_status = _last_status(det_stage.combined_output, _DETERMINISTIC_STATUS_RE)
        deterministic_status = det_status if det_status else "PASS"
        if deterministic_status != "PASS":
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)

        materialize_cli = [
            "--csv",
            args.csv,
            "--meta_risk",
            "0",
            "--meta_feedback",
            "0",
            "--out_dir",
            paths.health_out_dir,
        ]
        _append_optional_risk_args(materialize_cli, args)
        materialize_stage = _run_module(
            "health_materialize",
            "scripts.live_sim_daily",
            materialize_cli,
            logger=logger,
            cycle_id=cycle_id,
        )
        if materialize_stage.return_code == EXIT_ERROR:
            healthcheck_status = "FAIL"
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)
        if materialize_stage.return_code != EXIT_OK:
            healthcheck_status = "FAIL"
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)

        enable_anomaly_guard = int(args.enable_anomaly_guard) == 1
        if enable_anomaly_guard:
            summary_path = os.path.join(paths.health_out_dir, "live_daily_summary.json")
            equity_path = os.path.join(paths.health_out_dir, "live_daily_equity.csv")
            regime_path = os.path.join(paths.health_out_dir, "live_daily_regime.json")
            anomaly_payload = detect_anomaly(
                {
                    "summary_path": summary_path,
                    "equity_path": equity_path,
                    "regime_path": regime_path,
                    "daily_dd_limit": float(args.daily_dd_limit),
                    "pnl_jump_abs_limit": float(args.pnl_jump_abs_limit),
                    "trades_spike_limit": int(args.trades_spike_limit),
                    "equity_drift_abs_limit": float(args.equity_drift_abs_limit),
                    "simulate_anomaly": int(args.simulate_anomaly),
                }
            )
            is_anomaly = bool(anomaly_payload.get("is_anomaly", False))
            if is_anomaly:
                reason_code = str(anomaly_payload.get("reason_code", "UNKNOWN_ANOMALY"))
                metrics_snapshot = anomaly_payload.get("metrics", {})
                incident_path = _write_incident_report(
                    cycle_id=cycle_id,
                    reason_code=reason_code,
                    metrics_snapshot=metrics_snapshot if isinstance(metrics_snapshot, dict) else {},
                )
                _log(f"event=anomaly_detected reason_code={reason_code} incident_path={incident_path}")
                logger.warn(
                    event="anomaly_detected",
                    stage="anomaly_guardrail",
                    cycle_id=cycle_id,
                    reason_code=reason_code,
                    metrics_snapshot=metrics_snapshot,
                    incident_path=incident_path,
                )

                if int(args.auto_rollback) == 1:
                    if dry_run:
                        logger.info(
                            event="anomaly_auto_rollback_skip",
                            stage="anomaly_guardrail",
                            cycle_id=cycle_id,
                            reason="dry_run=1",
                            reason_code=reason_code,
                        )
                    else:
                        _run_auto_rollback_protected(
                            logger=logger,
                            cycle_id=cycle_id,
                            reason_code=reason_code,
                        )

                healthcheck_status = "SKIPPED"
                cycle_status = "HALT_ANOMALY"
                return _finish(EXIT_ANOMALY_HALT)
            logger.info(
                event="anomaly_guardrail_pass",
                stage="anomaly_guardrail",
                cycle_id=cycle_id,
                reason_code=str(anomaly_payload.get("reason_code", "NONE")),
            )
        else:
            _log("stage=anomaly_guardrail skipped=1")
            logger.info(event="stage_skip", stage="anomaly_guardrail", cycle_id=cycle_id, skipped=1)

        health_stage = _run_module(
            "healthcheck",
            "scripts.meta_healthcheck",
            [
                "--summary",
                os.path.join(paths.health_out_dir, "live_daily_summary.json"),
                "--equity",
                os.path.join(paths.health_out_dir, "live_daily_equity.csv"),
                "--meta_state",
                paths.approved_state_path,
                "--regime",
                os.path.join(paths.health_out_dir, "live_daily_regime.json"),
                "--strict",
                "1",
                "--cycle_id",
                cycle_id,
            ],
            logger=logger,
            cycle_id=cycle_id,
        )
        if health_stage.return_code == EXIT_ERROR:
            healthcheck_status = "FAIL"
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)
        if health_stage.return_code != EXIT_OK:
            healthcheck_status = "FAIL"
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)

        health_status = _last_status(health_stage.combined_output, _HEALTHCHECK_STATUS_RE)
        healthcheck_status = health_status if health_status else "PASS"
        if healthcheck_status != "PASS":
            cycle_status = "FAIL"
            return _finish(EXIT_ERROR)

        if dry_run:
            cycle_status = "DRY_RUN_SUCCESS"
        elif decision == "REJECTED":
            cycle_status = "SUCCESS_WITH_REJECT"
        elif decision == "SKIPPED":
            cycle_status = "SUCCESS_PROMOTION_SKIPPED"
        else:
            cycle_status = "SUCCESS"
        return _finish(EXIT_OK)

    except Exception as exc:
        _log(f"unexpected_error={exc}")
        logger.error(
            event="exception",
            stage="meta_cycle",
            cycle_id=cycle_id,
            exc_type=type(exc).__name__,
            message=str(exc),
        )
        cycle_status = "FAIL"
        return _finish(EXIT_ERROR)
    finally:
        elapsed = time.perf_counter() - t_total
        _print_final_summary(
            cycle_id=cycle_id,
            cycle_status=cycle_status,
            candidate_hash=candidate_hash,
            decision=decision,
            ledger_path=ledger_path,
            deterministic_status=deterministic_status,
            healthcheck_status=healthcheck_status,
            total_runtime_seconds=elapsed,
        )
        logger.info(
            event="stage_end",
            stage="meta_cycle",
            cycle_id=cycle_id,
            rc=int(final_rc),
            duration_s=round(elapsed, 6),
            cycle_status=cycle_status,
        )


def main() -> int:
    args = _parse_args()
    main_started = time.perf_counter()
    cycle_id = str(getattr(args, "cycle_id", "") or "").strip() or generate_cycle_id(seed=str(args.reason))
    logger = get_logger("meta_cycle").bind(cycle_id=cycle_id)
    _log(f"cycle_id={cycle_id}")
    logger.info(
        event="stage_start",
        stage="meta_cycle",
        cycle_id=cycle_id,
        reason=str(args.reason),
        strict=int(args.strict),
        dry_run=int(args.dry_run),
        enable_scoring=int(args.enable_scoring),
        w_pnl=float(args.w_pnl),
        w_win=float(args.w_win),
        w_dd=float(args.w_dd),
        enable_anomaly_guard=int(args.enable_anomaly_guard),
        auto_rollback=int(args.auto_rollback),
        simulate_anomaly=int(args.simulate_anomaly),
    )
    dry_run = int(args.dry_run) == 1
    skip_global_lock = int(args.skip_global_lock) == 1
    lock_policy = _effective_lock_policy(args)

    def _run_with_paths() -> int:
        if dry_run:
            with tempfile.TemporaryDirectory(prefix="meta_cycle_") as temp_root:
                paths = _build_paths(dry_run=True, temp_root=temp_root)
                _prepare_dry_run_state(paths)
                return _run_cycle(args, paths, dry_run=True, logger=logger, cycle_id=cycle_id)
        paths = _build_paths(dry_run=False, temp_root=None)
        return _run_cycle(args, paths, dry_run=False, logger=logger, cycle_id=cycle_id)

    if skip_global_lock:
        _lock_log("status=skip reason=skip_global_lock=1")
        logger.info(event="lock_skip", stage="lock", cycle_id=cycle_id, reason="skip_global_lock=1")
        return _run_with_paths()

    lock_result = acquire_exclusive_lock(
        lock_path=args.lock_path,
        reason=args.reason,
        ttl_sec=max(1.0, float(args.lock_ttl_sec)),
        force_lock_break=(int(args.force_lock_break) == 1),
    )
    if not lock_result.acquired:
        _lock_log(
            f"held_by={lock_owner_summary(lock_result.owner)} age_s={lock_result.age_s:.3f} action={lock_policy}"
        )
        logger.warn(
            event="lock_held",
            stage="lock",
            cycle_id=cycle_id,
            action=lock_policy,
            owner=lock_owner_summary(lock_result.owner),
            age_s=round(lock_result.age_s, 6),
        )
        rc = _lock_policy_exit_code(lock_policy)
        logger.info(
            event="stage_end",
            stage="meta_cycle",
            cycle_id=cycle_id,
            rc=int(rc),
            duration_s=round(time.perf_counter() - main_started, 6),
            cycle_status="LOCK_HELD",
        )
        return int(rc)

    lock_handle = lock_result.handle
    if lock_handle is None:
        _lock_log("status=error reason=missing_lock_handle")
        logger.error(event="exception", stage="lock", cycle_id=cycle_id, message="missing_lock_handle")
        logger.info(
            event="stage_end",
            stage="meta_cycle",
            cycle_id=cycle_id,
            rc=int(EXIT_ERROR),
            duration_s=round(time.perf_counter() - main_started, 6),
            cycle_status="FAIL",
        )
        return int(EXIT_ERROR)

    _lock_log(
        f"acquired path={args.lock_path} status={lock_result.status} "
        f"stale_break={1 if lock_handle.stale_break else 0} owner={lock_owner_summary(lock_result.owner)}"
    )
    logger.info(
        event="lock_acquired",
        stage="lock",
        cycle_id=cycle_id,
        path=args.lock_path,
        status=lock_result.status,
        stale_break=1 if lock_handle.stale_break else 0,
        owner=lock_owner_summary(lock_result.owner),
    )
    try:
        return _run_with_paths()
    finally:
        released = release_exclusive_lock(lock_handle)
        _lock_log(f"released path={args.lock_path} ok={1 if released else 0}")
        logger.info(
            event="lock_released",
            stage="lock",
            cycle_id=cycle_id,
            path=args.lock_path,
            ok=1 if released else 0,
        )


if __name__ == "__main__":
    sys.exit(main())
