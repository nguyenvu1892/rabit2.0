#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from rabit.state.exit_codes import ExitCode

try:
    from rabit.utils import StructuredLogger, generate_cycle_id, get_logger
except Exception:  # pragma: no cover - keep smoke usable even if logger package import fails
    StructuredLogger = Any  # type: ignore[assignment]
    generate_cycle_id = None  # type: ignore[assignment]
    get_logger = None  # type: ignore[assignment]


OK_EXIT_CODES = {
    int(ExitCode.SUCCESS),
    int(ExitCode.BUSINESS_REJECT),
    int(ExitCode.BUSINESS_SKIP),
}
FAIL_TAXONOMY_CODES = {
    int(ExitCode.DATA_INVALID),
    int(ExitCode.STATE_CORRUPT),
    int(ExitCode.LOCK_TIMEOUT),
    int(ExitCode.INTERNAL_ERROR),
    int(ExitCode.ANOMALY_HALT),
}

DEFAULT_META_STATE_PATH = os.path.join("data", "meta_states", "current_approved", "meta_risk_state.json")


@dataclass
class StepResult:
    name: str
    return_code: int
    duration_s: float
    stdout: str
    stderr: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production one-line smoke test for Phase 4 hardening.")
    parser.add_argument("--csv", required=True, help="Input bars CSV path.")
    parser.add_argument("--reason", default="prod_smoke_test", help="Reason passed to meta_cycle.")
    parser.add_argument("--simulate_anomaly", type=int, choices=[0, 1], default=0, help="Pass-through for anomaly simulation.")
    parser.add_argument(
        "--meta_state_path",
        default=DEFAULT_META_STATE_PATH,
        help=f"Meta state used by deterministic and health checks (default: {DEFAULT_META_STATE_PATH}).",
    )
    parser.add_argument(
        "--health_out_dir",
        default="",
        help="Optional output directory for health artifacts. Defaults to a temporary directory.",
    )
    parser.add_argument("--cycle_id", default="", help="Optional correlation id for structured logs.")
    return parser.parse_args()


def _shell_join(parts: Sequence[str]) -> str:
    try:
        return shlex.join(list(parts))
    except Exception:
        return " ".join(parts)


def _emit_log(logger: Optional[StructuredLogger], level: str, *, event: str, **fields: Any) -> None:
    if logger is None:
        return
    method = getattr(logger, str(level).strip().lower(), None)
    if callable(method):
        method(event=event, **fields)


def _build_logger(cycle_id: str) -> Optional[StructuredLogger]:
    if get_logger is None:
        return None
    try:
        return get_logger("prod_smoke_test").bind(cycle_id=cycle_id)
    except Exception:
        return None


def _run_module(
    name: str,
    module_name: str,
    module_args: List[str],
    *,
    logger: Optional[StructuredLogger],
    cycle_id: str,
) -> StepResult:
    cmd = [sys.executable, "-m", module_name] + list(module_args)
    print(f"[smoke] step={name} start cmd={_shell_join(cmd)}")
    _emit_log(
        logger,
        "info",
        event="stage_start",
        stage=name,
        cycle_id=cycle_id,
        module_name=module_name,
        cmd=_shell_join(cmd),
    )
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration_s = time.perf_counter() - t0
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout:
        for line in stdout.splitlines():
            print(f"[smoke][{name}][stdout] {line}")
    if stderr:
        for line in stderr.splitlines():
            print(f"[smoke][{name}][stderr] {line}")
    print(f"[smoke] step={name} end rc={int(proc.returncode)} duration_s={duration_s:.3f}")
    _emit_log(
        logger,
        "info",
        event="stage_end",
        stage=name,
        cycle_id=cycle_id,
        rc=int(proc.returncode),
        duration_s=round(duration_s, 6),
    )
    return StepResult(
        name=name,
        return_code=int(proc.returncode),
        duration_s=float(duration_s),
        stdout=stdout,
        stderr=stderr,
    )


def _classify_exit_code(return_code: int) -> tuple[bool, int, str]:
    rc = int(return_code)
    if rc in OK_EXIT_CODES:
        return True, int(ExitCode.SUCCESS), "OK"
    if rc in FAIL_TAXONOMY_CODES:
        return False, rc, f"TAXONOMY_{rc}"
    return False, int(ExitCode.INTERNAL_ERROR), f"UNMAPPED_{rc}"


def _compute_outcome(
    *,
    meta_cycle_exit: int,
    deterministic_exit: int,
    healthcheck_exit: int,
) -> tuple[str, int, str]:
    ordered = [
        ("meta_cycle_exit", int(meta_cycle_exit)),
        ("deterministic_exit", int(deterministic_exit)),
        ("healthcheck_exit", int(healthcheck_exit)),
    ]
    for key, rc in ordered:
        ok, mapped_rc, label = _classify_exit_code(rc)
        if not ok:
            return "FAIL", int(mapped_rc), f"{key}:{label}"
    return "PASS", int(ExitCode.SUCCESS), "ALL_CHECKS_OK"


def _materialize_health_artifacts(
    *,
    csv_path: str,
    out_dir: str,
    logger: Optional[StructuredLogger],
    cycle_id: str,
) -> StepResult:
    return _run_module(
        "health_materialize",
        "scripts.live_sim_daily",
        [
            "--csv",
            csv_path,
            "--meta_risk",
            "0",
            "--meta_feedback",
            "0",
            "--out_dir",
            out_dir,
        ],
        logger=logger,
        cycle_id=cycle_id,
    )


def _run_smoke(args: argparse.Namespace, *, logger: Optional[StructuredLogger], cycle_id: str) -> int:
    health_root = str(args.health_out_dir).strip()
    use_temp_health_root = not bool(health_root)

    temp_dir_cm = tempfile.TemporaryDirectory(prefix="prod_smoke_") if use_temp_health_root else None
    try:
        if temp_dir_cm is not None:
            temp_root = temp_dir_cm.__enter__()
            health_root = os.path.join(temp_root, "healthcheck")
        os.makedirs(health_root, exist_ok=True)

        meta_cycle_result = _run_module(
            "meta_cycle_dry_run",
            "scripts.meta_cycle",
            [
                "--csv",
                args.csv,
                "--reason",
                args.reason,
                "--strict",
                "0",
                "--dry_run",
                "1",
                "--on_lock_held",
                "retryable",
                "--simulate_anomaly",
                str(int(args.simulate_anomaly)),
                "--cycle_id",
                cycle_id,
            ],
            logger=logger,
            cycle_id=cycle_id,
        )
        deterministic_result = _run_module(
            "deterministic_check",
            "scripts.live_sim_daily",
            [
                "--csv",
                args.csv,
                "--meta_risk",
                "1",
                "--meta_feedback",
                "1",
                "--meta_state_path",
                args.meta_state_path,
                "--deterministic_check",
                "1",
            ],
            logger=logger,
            cycle_id=cycle_id,
        )

        materialize_result = _materialize_health_artifacts(
            csv_path=args.csv,
            out_dir=health_root,
            logger=logger,
            cycle_id=cycle_id,
        )
        if int(materialize_result.return_code) != 0:
            healthcheck_exit = int(materialize_result.return_code)
        else:
            healthcheck_result = _run_module(
                "meta_healthcheck",
                "scripts.meta_healthcheck",
                [
                    "--summary",
                    os.path.join(health_root, "live_daily_summary.json"),
                    "--equity",
                    os.path.join(health_root, "live_daily_equity.csv"),
                    "--meta_state",
                    args.meta_state_path,
                    "--regime",
                    os.path.join(health_root, "live_daily_regime.json"),
                    "--strict",
                    "1",
                    "--cycle_id",
                    cycle_id,
                ],
                logger=logger,
                cycle_id=cycle_id,
            )
            healthcheck_exit = int(healthcheck_result.return_code)

        smoke_status, final_exit_code, final_reason = _compute_outcome(
            meta_cycle_exit=int(meta_cycle_result.return_code),
            deterministic_exit=int(deterministic_result.return_code),
            healthcheck_exit=int(healthcheck_exit),
        )
        if smoke_status == "PASS":
            print("STATUS=PASS")
        else:
            print(f"STATUS=FAIL reason={final_reason}")

        summary = {
            "smoke_status": smoke_status,
            "meta_cycle_exit": int(meta_cycle_result.return_code),
            "deterministic_exit": int(deterministic_result.return_code),
            "healthcheck_exit": int(healthcheck_exit),
        }
        print(json.dumps(summary, indent=2))
        _emit_log(
            logger,
            "info" if smoke_status == "PASS" else "warn",
            event="summary",
            stage="prod_smoke_test",
            cycle_id=cycle_id,
            smoke_status=smoke_status,
            final_exit_code=int(final_exit_code),
            reason=final_reason,
            meta_cycle_exit=int(meta_cycle_result.return_code),
            deterministic_exit=int(deterministic_result.return_code),
            healthcheck_exit=int(healthcheck_exit),
            health_out_dir=health_root,
        )
        return int(final_exit_code)
    finally:
        if temp_dir_cm is not None:
            temp_dir_cm.__exit__(None, None, None)


def main() -> int:
    args = _parse_args()
    cycle_id = str(args.cycle_id or "").strip()
    if not cycle_id:
        if generate_cycle_id is not None:
            cycle_id = generate_cycle_id(seed=f"prod_smoke:{args.reason}")
        else:
            cycle_id = f"prod_smoke:{int(time.time())}"

    logger = _build_logger(cycle_id)
    started = time.perf_counter()
    _emit_log(
        logger,
        "info",
        event="stage_start",
        stage="prod_smoke_test",
        cycle_id=cycle_id,
        csv=args.csv,
        reason=args.reason,
        simulate_anomaly=int(args.simulate_anomaly),
        meta_state_path=args.meta_state_path,
    )

    rc = int(ExitCode.INTERNAL_ERROR)
    try:
        rc = int(_run_smoke(args, logger=logger, cycle_id=cycle_id))
        return int(rc)
    except Exception as exc:
        traceback.print_exc()
        print(f"STATUS=FAIL reason=EXCEPTION:{type(exc).__name__}")
        summary = {
            "smoke_status": "FAIL",
            "meta_cycle_exit": int(ExitCode.INTERNAL_ERROR),
            "deterministic_exit": int(ExitCode.INTERNAL_ERROR),
            "healthcheck_exit": int(ExitCode.INTERNAL_ERROR),
        }
        print(json.dumps(summary, indent=2))
        _emit_log(
            logger,
            "error",
            event="exception",
            stage="prod_smoke_test",
            cycle_id=cycle_id,
            exc_type=type(exc).__name__,
            message=str(exc),
        )
        rc = int(ExitCode.INTERNAL_ERROR)
        return int(rc)
    finally:
        _emit_log(
            logger,
            "info",
            event="stage_end",
            stage="prod_smoke_test",
            cycle_id=cycle_id,
            rc=int(rc),
            duration_s=round(time.perf_counter() - started, 6),
        )


if __name__ == "__main__":
    sys.exit(main())
