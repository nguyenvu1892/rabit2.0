#!/usr/bin/env python
from __future__ import annotations

"""
TASK-4H scheduler for automated meta-cycle promotion runs.

Usage examples:
  python -m scripts.meta_schedule --csv data/live/XAUUSD_M1_live.csv --run_once 1
  python -m scripts.meta_schedule --csv data/live/XAUUSD_M1_live.csv --interval_minutes 30 --max_runtime_seconds 7200
"""

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rabit.state.exit_codes import ExitCode
from rabit.state.file_lock import acquire_exclusive_lock, lock_owner_summary, release_exclusive_lock

EXIT_OK = ExitCode.SUCCESS
EXIT_REJECT = ExitCode.BUSINESS_REJECT
EXIT_LOCK_RETRYABLE = ExitCode.BUSINESS_SKIP
EXIT_ERROR = ExitCode.INTERNAL_ERROR

DEFAULT_LOCK_PATH = os.path.join("data", "meta_states", ".locks", "meta_cycle.lock")
DEFAULT_AUDIT_LOG_PATH = os.path.join("data", "reports", "meta_schedule.jsonl")
DEFAULT_LOCK_TTL_SEC = 30 * 60

_CYCLE_STATUS_RE = re.compile(r"^\[cycle\]\s+cycle_status=([^\s]+)", re.MULTILINE)
_CYCLE_DECISION_RE = re.compile(r"^\[cycle\]\s+decision=([^\s]+)", re.MULTILINE)
_LEDGER_PATH_RE = re.compile(r"^\[cycle\]\s+ledger_path=([^\s]+)", re.MULTILINE)
_CYCLE_RUNTIME_RE = re.compile(r"^\[cycle\]\s+total_runtime_seconds=([0-9]+(?:\.[0-9]+)?)", re.MULTILINE)
_CANDIDATE_HASH_RE = re.compile(r"\bcandidate_hash=([0-9a-f]{64}|missing)\b")
_APPROVED_HASH_RE = re.compile(r"\bapproved_hash=([0-9a-f]{64}|missing)\b")
_BASE_HASH_RE = re.compile(r"\bbase_hash=([0-9a-f]{64}|missing)\b")


@dataclass
class CycleExecResult:
    return_code: int
    cycle_status: str
    decision: str
    ledger_path: str
    candidate_hash: str
    approved_hash: str
    cycle_runtime_seconds: float


def _utc_iso() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(message: str) -> None:
    print(f"[schedule] {message}")


def _lock_log(message: str) -> None:
    print(f"[lock] {message}")


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Schedule and run scripts.meta_cycle with lock guardrails.")
    ap.add_argument("--csv", required=True, help="Input bars CSV path passed to meta_cycle.")
    ap.add_argument("--reason", default="scheduled", help='Cycle reason string (default: "scheduled").')
    ap.add_argument("--strict", type=int, choices=[0, 1], default=1, help="Strict mode (0/1).")
    ap.add_argument("--dry_run", type=int, choices=[0, 1], default=0, help="Dry-run mode (0/1).")
    ap.add_argument(
        "--interval_minutes",
        type=float,
        default=None,
        help="Loop interval in minutes. Use with loop mode.",
    )
    ap.add_argument("--run_once", type=int, choices=[0, 1], default=0, help="Run exactly one cycle then exit.")
    ap.add_argument(
        "--lock_path",
        default=DEFAULT_LOCK_PATH,
        help=f"Single-flight lock file path (default: {DEFAULT_LOCK_PATH}).",
    )
    ap.add_argument(
        "--lock_ttl_sec",
        type=float,
        default=float(DEFAULT_LOCK_TTL_SEC),
        help="Lock TTL in seconds for stale detection (default: 1800).",
    )
    ap.add_argument(
        "--stale_lock_seconds",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--force_lock_break",
        type=int,
        choices=[0, 1],
        default=0,
        help="Allow stale lock takeover when age > lock_ttl_sec (0/1).",
    )
    ap.add_argument(
        "--on_lock_held",
        choices=["skip", "retryable", "fail"],
        default="skip",
        help="Action when lock is held: skip=0, retryable=11, fail=50.",
    )
    ap.add_argument(
        "--max_runtime_seconds",
        type=float,
        default=None,
        help="Optional max wall-clock runtime for interval mode.",
    )
    ap.add_argument(
        "--audit_log_path",
        default=DEFAULT_AUDIT_LOG_PATH,
        help=f"JSONL audit output path (default: {DEFAULT_AUDIT_LOG_PATH}).",
    )
    return ap.parse_args()


def _last_match(text: str, pattern: re.Pattern[str]) -> str:
    out = ""
    for match in pattern.finditer(text):
        out = str(match.group(1)).strip()
    return out


def _is_sha_like(text: str) -> bool:
    s = str(text or "").strip().lower()
    if s == "missing":
        return True
    if len(s) != 64:
        return False
    return all(c in "0123456789abcdef" for c in s)


def _normalize_cycle_status(raw_status: str, decision: str, dry_run: bool, return_code: int) -> str:
    status = str(raw_status or "").strip().upper()
    dec = str(decision or "").strip().upper()

    if status == "SUCCESS":
        return "SUCCESS_WITH_PROMOTE"
    if status:
        return status

    if return_code == EXIT_OK:
        if dry_run:
            return "DRY_RUN_SUCCESS"
        if dec == "REJECTED":
            return "SUCCESS_WITH_REJECT"
        if dec == "APPROVED":
            return "SUCCESS_WITH_PROMOTE"
        return "SUCCESS"
    if return_code == EXIT_REJECT:
        return "SUCCESS_WITH_REJECT"
    return "FAIL"


def _is_business_reject_outcome(cycle_result: Optional[CycleExecResult]) -> bool:
    if cycle_result is None:
        return False
    if int(cycle_result.return_code) == ExitCode.BUSINESS_REJECT:
        return True
    if str(cycle_result.decision).strip().upper() == "REJECTED":
        return True
    if str(cycle_result.cycle_status).strip().upper() == "SUCCESS_WITH_REJECT":
        return True
    return False


def _scheduler_status_from_return_code(return_code: int) -> str:
    rc = int(return_code)
    if rc in (ExitCode.SUCCESS, ExitCode.BUSINESS_REJECT):
        return "OK"
    if rc == ExitCode.BUSINESS_SKIP:
        return "RETRYABLE"
    return "ERROR"


def _effective_lock_ttl_sec(args: argparse.Namespace) -> float:
    if args.stale_lock_seconds is not None:
        return max(1.0, float(args.stale_lock_seconds))
    return max(1.0, float(args.lock_ttl_sec))


def _lock_policy_exit_code(policy: str) -> int:
    normalized = str(policy or "skip").strip().lower()
    if normalized == "retryable":
        return int(EXIT_LOCK_RETRYABLE)
    if normalized == "fail":
        return int(EXIT_ERROR)
    return int(EXIT_OK)


def _lock_policy_scheduler_status(policy: str) -> str:
    normalized = str(policy or "skip").strip().lower()
    if normalized == "retryable":
        return "LOCKED_RETRYABLE"
    if normalized == "fail":
        return "LOCKED_FAIL"
    return "LOCKED_SKIP"


def _shell_join(parts: List[str]) -> str:
    try:
        return shlex.join(parts)
    except Exception:
        return " ".join(parts)


def _load_last_ledger_entry(ledger_path: str) -> Dict[str, Any]:
    if not ledger_path or not os.path.exists(ledger_path):
        return {}
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return {}
        payload = json.loads(lines[-1])
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _run_cycle(csv: str, reason: str, strict: int, dry_run: int) -> CycleExecResult:
    cmd = [
        sys.executable,
        "-m",
        "scripts.meta_cycle",
        "--csv",
        csv,
        "--reason",
        reason,
        "--strict",
        str(int(strict)),
        "--dry_run",
        str(int(dry_run)),
        "--skip_global_lock",
        "1",
    ]
    _log(f"cycle_start cmd={_shell_join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt_s = time.perf_counter() - t0

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout:
        for line in stdout.splitlines():
            print(f"[schedule][meta_cycle][stdout] {line}")
    if stderr:
        for line in stderr.splitlines():
            print(f"[schedule][meta_cycle][stderr] {line}")

    combined = stdout
    if combined and stderr:
        combined = combined + "\n" + stderr
    elif not combined:
        combined = stderr

    raw_cycle_status = _last_match(combined, _CYCLE_STATUS_RE)
    decision = _last_match(combined, _CYCLE_DECISION_RE)
    ledger_path = _last_match(combined, _LEDGER_PATH_RE)
    runtime_text = _last_match(combined, _CYCLE_RUNTIME_RE)
    cycle_runtime_seconds = dt_s
    if runtime_text:
        try:
            cycle_runtime_seconds = float(runtime_text)
        except Exception:
            cycle_runtime_seconds = dt_s

    candidate_hash = _last_match(combined, _CANDIDATE_HASH_RE).lower()
    approved_hash = _last_match(combined, _APPROVED_HASH_RE).lower()

    if not _is_sha_like(candidate_hash):
        candidate_hash = "missing"
    if not _is_sha_like(approved_hash):
        approved_hash = "missing"

    ledger_entry = _load_last_ledger_entry(ledger_path)
    if candidate_hash in ("", "missing"):
        c_hash = str(ledger_entry.get("candidate_hash", "")).strip().lower()
        if _is_sha_like(c_hash):
            candidate_hash = c_hash
    if approved_hash in ("", "missing"):
        a_hash = str(ledger_entry.get("base_hash", "")).strip().lower()
        if _is_sha_like(a_hash):
            approved_hash = a_hash
    if approved_hash in ("", "missing"):
        a_hash_alt = _last_match(combined, _BASE_HASH_RE).lower()
        if _is_sha_like(a_hash_alt):
            approved_hash = a_hash_alt

    cycle_status = _normalize_cycle_status(
        raw_status=raw_cycle_status,
        decision=decision,
        dry_run=(int(dry_run) == 1),
        return_code=int(proc.returncode),
    )

    _log(
        "cycle_end "
        f"rc={int(proc.returncode)} "
        f"cycle_status={cycle_status} "
        f"decision={decision or 'UNKNOWN'} "
        f"runtime_seconds={cycle_runtime_seconds:.3f}"
    )
    _log(f"decision={decision or 'UNKNOWN'}")
    _log(f"status={cycle_status}")
    _log(f"runtime_seconds={cycle_runtime_seconds:.3f}")

    return CycleExecResult(
        return_code=int(proc.returncode),
        cycle_status=cycle_status,
        decision=decision or "UNKNOWN",
        ledger_path=ledger_path or "",
        candidate_hash=candidate_hash or "missing",
        approved_hash=approved_hash or "missing",
        cycle_runtime_seconds=float(cycle_runtime_seconds),
    )


def _append_audit_record(path: str, record: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    line = _json_dumps(record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _build_audit_record(
    *,
    args: argparse.Namespace,
    mode: str,
    cycle_result: Optional[CycleExecResult],
    scheduler_status: str,
    lock_status: str,
    lock_age_seconds: float,
    scheduler_runtime_seconds: float,
    lock_return_code: int = EXIT_OK,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "timestamp": _utc_iso(),
        "mode": mode,
        "csv": args.csv,
        "reason": str(args.reason),
        "strict": int(args.strict),
        "dry_run": int(args.dry_run),
        "lock_path": args.lock_path,
        "lock_status": lock_status,
        "lock_age_seconds": float(lock_age_seconds),
        "scheduler_status": scheduler_status,
        "durations": {
            "cycle_runtime_seconds": float(cycle_result.cycle_runtime_seconds) if cycle_result else 0.0,
            "scheduler_runtime_seconds": float(scheduler_runtime_seconds),
        },
        "cycle_status": cycle_result.cycle_status if cycle_result else scheduler_status,
        "decision": cycle_result.decision if cycle_result else "LOCKED",
        "ledger_path": cycle_result.ledger_path if cycle_result else "",
        "candidate_hash": cycle_result.candidate_hash if cycle_result else "missing",
        "approved_hash": cycle_result.approved_hash if cycle_result else "missing",
        "cycle_return_code": int(cycle_result.return_code) if cycle_result else int(lock_return_code),
    }
    return entry


def _sleep_interval(interval_seconds: float, started_at: float, max_runtime_seconds: Optional[float], base_time: float) -> bool:
    elapsed_this_tick = time.monotonic() - started_at
    sleep_s = max(0.0, interval_seconds - elapsed_this_tick)
    if max_runtime_seconds is not None:
        remaining = max_runtime_seconds - (time.monotonic() - base_time)
        if remaining <= 0:
            return False
        sleep_s = min(sleep_s, remaining)
    if sleep_s > 0:
        time.sleep(sleep_s)
    return True


def _run_once(args: argparse.Namespace, lock_ttl_sec: float) -> int:
    sched_t0 = time.perf_counter()
    policy = str(args.on_lock_held).strip().lower()
    lock_result = acquire_exclusive_lock(
        lock_path=args.lock_path,
        reason=args.reason,
        ttl_sec=lock_ttl_sec,
        force_lock_break=(int(args.force_lock_break) == 1),
    )
    if not lock_result.acquired:
        lock_exit_code = _lock_policy_exit_code(policy)
        _lock_log(
            f"held_by={lock_owner_summary(lock_result.owner)} age_s={lock_result.age_s:.3f} action={policy}"
        )
        record = _build_audit_record(
            args=args,
            mode="run_once",
            cycle_result=None,
            scheduler_status=_lock_policy_scheduler_status(policy),
            lock_status=lock_result.status,
            lock_age_seconds=lock_result.age_s,
            scheduler_runtime_seconds=time.perf_counter() - sched_t0,
            lock_return_code=lock_exit_code,
        )
        _append_audit_record(args.audit_log_path, record)
        return int(lock_exit_code)

    lock_handle = lock_result.handle
    if lock_handle is None:
        _lock_log("status=error reason=missing_lock_handle")
        return int(EXIT_ERROR)

    _lock_log(
        f"acquired path={args.lock_path} status={lock_result.status} stale_break={1 if lock_handle.stale_break else 0} "
        f"owner={lock_owner_summary(lock_result.owner)}"
    )
    cycle_result: Optional[CycleExecResult] = None
    try:
        cycle_result = _run_cycle(
            csv=args.csv,
            reason=args.reason,
            strict=int(args.strict),
            dry_run=int(args.dry_run),
        )
        scheduler_status = _scheduler_status_from_return_code(cycle_result.return_code)
        record = _build_audit_record(
            args=args,
            mode="run_once",
            cycle_result=cycle_result,
            scheduler_status=scheduler_status,
            lock_status=lock_result.status,
            lock_age_seconds=lock_result.age_s,
            scheduler_runtime_seconds=time.perf_counter() - sched_t0,
        )
        _append_audit_record(args.audit_log_path, record)
        if _is_business_reject_outcome(cycle_result):
            _log("business_reject_not_error")
            return EXIT_OK
        return int(cycle_result.return_code)
    finally:
        released = release_exclusive_lock(lock_handle)
        _lock_log(f"released path={args.lock_path} ok={1 if released else 0}")


def _run_interval_loop(args: argparse.Namespace, lock_ttl_sec: float) -> int:
    if args.interval_minutes is None or float(args.interval_minutes) <= 0:
        raise ValueError("interval_minutes must be > 0 for loop mode")

    interval_seconds = float(args.interval_minutes) * 60.0
    max_runtime_seconds = None
    if args.max_runtime_seconds is not None:
        max_runtime_seconds = max(0.0, float(args.max_runtime_seconds))

    policy = str(args.on_lock_held).strip().lower()
    loop_t0 = time.monotonic()
    cycle_idx = 0
    while True:
        if max_runtime_seconds is not None and (time.monotonic() - loop_t0) >= max_runtime_seconds:
            _log("max_runtime_reached=1 -> stop")
            return EXIT_OK

        cycle_idx += 1
        tick_t0 = time.monotonic()
        cycle_reason = args.reason
        _log(f"cycle_tick index={cycle_idx} reason={cycle_reason}")

        sched_t0 = time.perf_counter()
        lock_result = acquire_exclusive_lock(
            lock_path=args.lock_path,
            reason=cycle_reason,
            ttl_sec=lock_ttl_sec,
            force_lock_break=(int(args.force_lock_break) == 1),
        )
        if not lock_result.acquired:
            lock_exit_code = _lock_policy_exit_code(policy)
            _lock_log(
                f"held_by={lock_owner_summary(lock_result.owner)} age_s={lock_result.age_s:.3f} action={policy}"
            )
            record = _build_audit_record(
                args=args,
                mode="interval",
                cycle_result=None,
                scheduler_status=_lock_policy_scheduler_status(policy),
                lock_status=lock_result.status,
                lock_age_seconds=lock_result.age_s,
                scheduler_runtime_seconds=time.perf_counter() - sched_t0,
                lock_return_code=lock_exit_code,
            )
            _append_audit_record(args.audit_log_path, record)
            if policy != "skip":
                return int(lock_exit_code)
            if not _sleep_interval(interval_seconds, tick_t0, max_runtime_seconds, loop_t0):
                _log("max_runtime_reached=1 -> stop")
                return EXIT_OK
            continue

        lock_handle = lock_result.handle
        if lock_handle is None:
            _lock_log("status=error reason=missing_lock_handle")
            return int(EXIT_ERROR)

        _lock_log(
            f"acquired path={args.lock_path} status={lock_result.status} stale_break={1 if lock_handle.stale_break else 0} "
            f"owner={lock_owner_summary(lock_result.owner)}"
        )
        cycle_result: Optional[CycleExecResult] = None
        try:
            cycle_result = _run_cycle(
                csv=args.csv,
                reason=cycle_reason,
                strict=int(args.strict),
                dry_run=int(args.dry_run),
            )
            scheduler_status = _scheduler_status_from_return_code(cycle_result.return_code)
            record = _build_audit_record(
                args=args,
                mode="interval",
                cycle_result=cycle_result,
                scheduler_status=scheduler_status,
                lock_status=lock_result.status,
                lock_age_seconds=lock_result.age_s,
                scheduler_runtime_seconds=time.perf_counter() - sched_t0,
            )
            _append_audit_record(args.audit_log_path, record)
        finally:
            released = release_exclusive_lock(lock_handle)
            _lock_log(f"released path={args.lock_path} ok={1 if released else 0}")

        if _is_business_reject_outcome(cycle_result):
            _log("business_reject_not_error")
        if cycle_result is not None and int(cycle_result.return_code) >= ExitCode.DATA_INVALID:
            return int(cycle_result.return_code)

        if not _sleep_interval(interval_seconds, tick_t0, max_runtime_seconds, loop_t0):
            _log("max_runtime_reached=1 -> stop")
            return EXIT_OK


def main() -> int:
    args = _parse_args()
    mode = "run_once" if int(args.run_once) == 1 else "interval"

    if int(args.run_once) != 1 and args.interval_minutes is None:
        raise SystemExit("Either set --run_once 1 or provide --interval_minutes N.")
    if int(args.run_once) == 1 and args.interval_minutes is not None:
        _log("run_once=1 and interval_minutes provided -> interval ignored")

    lock_ttl_sec = _effective_lock_ttl_sec(args)
    _log(
        f"start mode={mode} csv={args.csv} reason={args.reason} strict={int(args.strict)} "
        f"dry_run={int(args.dry_run)} lock_path={args.lock_path} lock_ttl_sec={lock_ttl_sec:.3f} "
        f"force_lock_break={int(args.force_lock_break)} on_lock_held={args.on_lock_held}"
    )

    try:
        if mode == "run_once":
            return _run_once(args, lock_ttl_sec=lock_ttl_sec)
        return _run_interval_loop(args, lock_ttl_sec=lock_ttl_sec)
    except KeyboardInterrupt:
        _log("status=INTERRUPTED")
        return 130


if __name__ == "__main__":
    sys.exit(main())
