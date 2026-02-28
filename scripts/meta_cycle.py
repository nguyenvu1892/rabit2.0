#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

_MUTATION_K_MIN = 3
_MUTATION_K_MAX = 5
_MUTATION_K_DEFAULT = 5


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


@dataclass
class MutationCandidate:
    index: int
    candidate_path: str
    candidate_manifest_path: str
    shadow_out_json: str
    eval_report_path: str
    candidate_hash: str
    param_diff_summary: List[str]
    gate_status: str
    gate_reason: str
    score_total: Optional[float]
    score_holdout: Optional[float]


@dataclass
class MutationSelection:
    candidates: List[MutationCandidate]
    selected: Optional[MutationCandidate]
    k_candidates: int
    mutation_seed: int
    mutation_seed_basis: str
    mutation_profile: str
    selection_reason: str
    ledger_extra_path: str


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
    ap.add_argument(
        "--opt_mutation",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable bounded deterministic multi-candidate mutation selection (0/1).",
    )
    ap.add_argument(
        "--k_candidates",
        type=int,
        default=_MUTATION_K_DEFAULT,
        help=f"Candidate count for mutation search (clamped to [{_MUTATION_K_MIN},{_MUTATION_K_MAX}]).",
    )
    ap.add_argument("--mutation_seed", default="", help="Optional deterministic mutation seed string.")
    ap.add_argument(
        "--mutation_profile",
        default="safe",
        help='Mutation profile (currently only "safe" is implemented).',
    )
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


def _safe_float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    if out in (float("inf"), float("-inf")):
        return None
    return float(out)


def _round8_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 8)


def _clamp_k_candidates(raw_value: Any) -> int:
    try:
        out = int(raw_value)
    except Exception:
        out = _MUTATION_K_DEFAULT
    if out < _MUTATION_K_MIN:
        return _MUTATION_K_MIN
    if out > _MUTATION_K_MAX:
        return _MUTATION_K_MAX
    return int(out)


def _resolve_mutation_seed(
    *,
    approved_hash: str,
    reason: str,
    k_candidates: int,
    mutation_profile: str,
    mutation_seed_arg: str,
) -> Tuple[int, str]:
    provided = str(mutation_seed_arg or "").strip()
    if provided:
        seed_basis = provided
    else:
        seed_basis = f"{approved_hash}|{reason}|{int(k_candidates)}|{mutation_profile}"
    seed = int(hashlib.sha1(seed_basis.encode("utf-8")).hexdigest()[:8], 16)
    return int(seed), seed_basis


def _load_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    payload, _ = atomic_io.load_json_with_fallback(path)
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _manifest_param_diff_summary(path: str) -> List[str]:
    payload = _load_json_dict(path)
    summary = payload.get("param_diff_summary")
    if isinstance(summary, list):
        return [str(item) for item in summary]
    if isinstance(summary, str) and summary.strip():
        return [summary.strip()]
    return ["BASE"]


def _run_mutation_selection(
    *,
    args: argparse.Namespace,
    paths: CyclePaths,
    dry_run: bool,
    logger: StructuredLogger,
    cycle_id: str,
    approved_hash: str,
) -> MutationSelection:
    profile = str(getattr(args, "mutation_profile", "safe") or "safe").strip().lower()
    if profile != "safe":
        raise RuntimeError(f"unsupported_mutation_profile profile={profile}")

    k_candidates = _clamp_k_candidates(getattr(args, "k_candidates", _MUTATION_K_DEFAULT))
    seed, seed_basis = _resolve_mutation_seed(
        approved_hash=str(approved_hash),
        reason=str(args.reason),
        k_candidates=int(k_candidates),
        mutation_profile=profile,
        mutation_seed_arg=str(getattr(args, "mutation_seed", "") or ""),
    )
    rng = random.Random(int(seed))
    candidate_parent = os.path.dirname(paths.candidate_path) or os.path.join("data", "meta_states", "candidate")
    mutation_root = os.path.join(candidate_parent, "mutation", str(cycle_id))
    os.makedirs(mutation_root, exist_ok=True)

    _log(
        "mutation_config "
        f"enabled=1 k_candidates={k_candidates} mutation_seed={seed} mutation_profile={profile}"
    )
    logger.info(
        event="mutation_candidates_generated",
        stage="candidate_generate",
        cycle_id=cycle_id,
        approved_hash=approved_hash,
        candidate_hash="pending",
        score_total=None,
        gate_status="PENDING",
        k_candidates=int(k_candidates),
        mutation_seed=int(seed),
        mutation_profile=profile,
    )

    candidates: List[MutationCandidate] = []
    seen_candidate_hashes: set[str] = set()
    for idx in range(int(k_candidates)):
        candidate_dir = os.path.join(mutation_root, f"candidate_{idx}")
        candidate_path = os.path.join(candidate_dir, "meta_risk_state.json")
        candidate_manifest = os.path.join(candidate_dir, "manifest.json")
        shadow_out_json = os.path.join(candidate_dir, "shadow_replay_report.json")
        eval_report_path = os.path.join(candidate_dir, "promotion_eval.json")
        mode = "copy" if idx == 0 else "safe_mutate"
        candidate_hash = "missing"
        param_diff_summary: List[str] = ["BASE"]
        attempts = 0
        while True:
            candidate_seed = int(rng.randint(0, 2**31 - 1))
            step = _run_module(
                f"candidate_generate_c{idx}",
                "scripts.meta_candidate_generate",
                [
                    "--approved_state_path",
                    paths.approved_state_path,
                    "--candidate_out_path",
                    candidate_path,
                    "--candidate_manifest",
                    candidate_manifest,
                    "--reason",
                    args.reason,
                    "--strict",
                    str(int(args.strict)),
                    "--mode",
                    mode,
                    "--seed",
                    str(candidate_seed),
                    "--candidate_index",
                    str(int(idx)),
                    "--mutation_profile",
                    profile,
                ],
                logger=logger,
                cycle_id=cycle_id,
            )
            if step.return_code != EXIT_OK:
                raise RuntimeError(f"mutation_candidate_generate_failed index={idx} rc={step.return_code}")
            candidate_hash = _load_candidate_hash(candidate_manifest, candidate_path)
            param_diff_summary = _manifest_param_diff_summary(candidate_manifest)
            if idx == 0 or candidate_hash not in seen_candidate_hashes or attempts >= 4:
                break
            attempts += 1
            _log(
                "mutation_candidate_regenerate "
                f"index={idx} duplicate_hash={candidate_hash} attempt={attempts}"
            )

        seen_candidate_hashes.add(candidate_hash)
        _log(
            "mutation_candidate_generated "
            f"index={idx} candidate_hash={candidate_hash} "
            f"param_diff_summary={'|'.join(param_diff_summary)}"
        )
        logger.info(
            event="mutation_candidates_generated",
            stage="candidate_generate",
            cycle_id=cycle_id,
            approved_hash=approved_hash,
            candidate_hash=candidate_hash,
            score_total=None,
            gate_status="PENDING",
            candidate_index=int(idx),
        )
        candidates.append(
            MutationCandidate(
                index=int(idx),
                candidate_path=candidate_path,
                candidate_manifest_path=candidate_manifest,
                shadow_out_json=shadow_out_json,
                eval_report_path=eval_report_path,
                candidate_hash=candidate_hash,
                param_diff_summary=param_diff_summary,
                gate_status="PENDING",
                gate_reason="pending",
                score_total=None,
                score_holdout=None,
            )
        )

    for candidate in candidates:
        replay_stage = _run_module(
            f"shadow_replay_c{candidate.index}",
            "scripts.shadow_replay",
            [
                "--csv",
                args.csv,
                "--meta_state_path",
                candidate.candidate_path,
                "--out_json",
                candidate.shadow_out_json,
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
        if replay_stage.return_code != EXIT_OK:
            candidate.gate_status = "FAIL"
            candidate.gate_reason = f"shadow_replay_failed rc={replay_stage.return_code}"
            candidate.score_total = None
            candidate.score_holdout = None
            _log(
                "mutation_candidate_evaluated "
                f"index={candidate.index} candidate_hash={candidate.candidate_hash} "
                f"gate_status={candidate.gate_status} gate_reason={candidate.gate_reason}"
            )
            logger.info(
                event="mutation_candidate_evaluated",
                stage="meta_promote_eval",
                cycle_id=cycle_id,
                approved_hash=approved_hash,
                candidate_hash=candidate.candidate_hash,
                score_total=None,
                gate_status=candidate.gate_status,
            )
            continue

        eval_cli = [
            "--candidate_path",
            candidate.candidate_path,
            "--reason",
            args.reason,
            "--strict",
            str(int(args.strict)),
            "--csv",
            args.csv,
            "--no_exit_on_reject",
            "1",
            "--enable_scoring",
            "1",
            "--w_pnl",
            str(float(args.w_pnl)),
            "--w_win",
            str(float(args.w_win)),
            "--w_dd",
            str(float(args.w_dd)),
            "--cycle_id",
            cycle_id,
            "--eval_only",
            "1",
            "--eval_report_path",
            candidate.eval_report_path,
            "--holdout_report_path",
            os.path.join("data", "reports", "holdout", "holdout_report.json"),
        ]
        if dry_run:
            eval_cli.extend(
                [
                    "--approved_dir",
                    paths.promote_approved_dir,
                ]
            )

        eval_stage = _run_module(
            f"meta_promote_eval_c{candidate.index}",
            "scripts.meta_promote",
            eval_cli,
            logger=logger,
            cycle_id=cycle_id,
        )
        if eval_stage.return_code >= ExitCode.DATA_INVALID or eval_stage.return_code == EXIT_ERROR:
            raise RuntimeError(f"mutation_eval_failed index={candidate.index} rc={eval_stage.return_code}")

        eval_payload = _load_json_dict(candidate.eval_report_path)
        if not eval_payload:
            raise RuntimeError(f"mutation_eval_report_missing index={candidate.index} path={candidate.eval_report_path}")

        candidate.gate_status = str(eval_payload.get("gate_status", "FAIL")).strip().upper() or "FAIL"
        candidate.gate_reason = str(eval_payload.get("gate_reason", "missing")).strip() or "missing"
        candidate.score_total = _safe_float_or_none(eval_payload.get("score_total"))
        candidate.score_holdout = _safe_float_or_none(eval_payload.get("score_holdout"))
        _log(
            "mutation_candidate_evaluated "
            f"index={candidate.index} candidate_hash={candidate.candidate_hash} "
            f"gate_status={candidate.gate_status} score_total={candidate.score_total}"
        )
        logger.info(
            event="mutation_candidate_evaluated",
            stage="meta_promote_eval",
            cycle_id=cycle_id,
            approved_hash=approved_hash,
            candidate_hash=candidate.candidate_hash,
            score_total=_round8_or_none(candidate.score_total),
            gate_status=candidate.gate_status,
        )

    selected: Optional[MutationCandidate] = None
    best_score: Optional[float] = None
    pass_count = 0
    for candidate in candidates:
        if candidate.gate_status != "PASS":
            continue
        pass_count += 1
        score = candidate.score_total
        if selected is None:
            selected = candidate
            best_score = score
            continue
        if score is None:
            continue
        if best_score is None or float(score) > float(best_score):
            selected = candidate
            best_score = float(score)

    if selected is None:
        selection_reason = "no_pass_candidates"
        selected_hash = None
        selection_score = None
        selection_gate = "REJECT"
    else:
        selected_hash = selected.candidate_hash
        selection_score = _round8_or_none(best_score)
        selection_gate = "PASS"
        if best_score is None:
            selection_reason = f"pass_without_score_fallback candidate_index={selected.index}"
        else:
            selection_reason = (
                f"best_score_total={_round8_or_none(best_score)} "
                f"candidate_index={selected.index} pass_count={pass_count}"
            )

    logger.info(
        event="mutation_selection",
        stage="meta_promote_eval",
        cycle_id=cycle_id,
        approved_hash=approved_hash,
        candidate_hash=selected_hash or "missing",
        score_total=selection_score,
        gate_status=selection_gate,
    )
    _log(
        "mutation_selection "
        f"selected_candidate_hash={selected_hash or 'missing'} "
        f"selection_reason={selection_reason}"
    )

    ledger_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        item: Dict[str, Any] = {
            "candidate_hash": candidate.candidate_hash,
            "param_diff_summary": list(candidate.param_diff_summary),
            "gate_status": candidate.gate_status,
            "gate_reason": candidate.gate_reason,
            "score_total": _round8_or_none(candidate.score_total),
        }
        if candidate.score_holdout is not None:
            item["score_holdout"] = _round8_or_none(candidate.score_holdout)
        ledger_candidates.append(item)

    ledger_extra = {
        "mutation_enabled": 1,
        "k_candidates": int(k_candidates),
        "mutation_seed": int(seed),
        "mutation_seed_basis": seed_basis,
        "mutation_profile": profile,
        "candidates": ledger_candidates,
        "selected_candidate_hash": selected_hash,
        "selection_reason": selection_reason,
    }
    ledger_extra_path = os.path.join(mutation_root, "mutation_ledger_extra.json")
    atomic_io.atomic_write_json(
        ledger_extra_path,
        ledger_extra,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )

    return MutationSelection(
        candidates=candidates,
        selected=selected,
        k_candidates=int(k_candidates),
        mutation_seed=int(seed),
        mutation_seed_basis=seed_basis,
        mutation_profile=profile,
        selection_reason=selection_reason,
        ledger_extra_path=ledger_extra_path,
    )


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
    opt_mutation = int(getattr(args, "opt_mutation", 0)) == 1

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

        if opt_mutation and (skip_replay or skip_promote):
            _log("opt_mutation_disabled reason=skip_replay_or_skip_promote")
            logger.warn(
                event="mutation_disabled",
                stage="meta_cycle",
                cycle_id=cycle_id,
                reason="skip_replay_or_skip_promote",
            )
            opt_mutation = False

        if opt_mutation:
            approved_hash = _sha256_file(paths.approved_state_path)
            mutation_selection = _run_mutation_selection(
                args=args,
                paths=paths,
                dry_run=dry_run,
                logger=logger,
                cycle_id=cycle_id,
                approved_hash=approved_hash,
            )
            if not mutation_selection.candidates:
                cycle_status = "FAIL"
                return _finish(EXIT_ERROR)

            final_candidate = mutation_selection.selected or mutation_selection.candidates[0]
            candidate_hash = final_candidate.candidate_hash
            promote_cli = [
                "--candidate_path",
                final_candidate.candidate_path,
                "--reason",
                args.reason,
                "--strict",
                str(int(args.strict)),
                "--csv",
                args.csv,
                "--no_exit_on_reject",
                "1",
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
                "--ledger_extra_path",
                mutation_selection.ledger_extra_path,
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
                decision = "APPROVED"
            else:
                decision = "ERROR"

            if decision == "ERROR":
                cycle_status = "FAIL"
                return _finish(EXIT_ERROR)

            if decision == "REJECTED":
                _log("mutation_business_reject=1 -> stop_before_post_validation")
                cycle_status = "SUCCESS_WITH_REJECT"
                return _finish(EXIT_OK)
        else:
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
        opt_mutation=int(getattr(args, "opt_mutation", 0)),
        k_candidates=int(_clamp_k_candidates(getattr(args, "k_candidates", _MUTATION_K_DEFAULT))),
        mutation_profile=str(getattr(args, "mutation_profile", "safe")),
        mutation_seed=str(getattr(args, "mutation_seed", "") or ""),
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
