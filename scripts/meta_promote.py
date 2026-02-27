#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import tempfile
from typing import Any, Dict

from rabit.state import promotion_gate
from scripts import deterministic_utils as det

DEFAULT_APPROVED_DIR = os.path.join("data", "meta_states", "current_approved")
DEFAULT_HISTORY_DIR = os.path.join("data", "meta_states", "history")
DEFAULT_REJECTED_DIR = os.path.join("data", "meta_states", "rejected")
DEFAULT_CSV = os.path.join("data", "live", "XAUUSD_M1_live.csv")
DEFAULT_MODEL = os.path.join("data", "ars_best_theta_regime_bank.npz")


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
    ap.add_argument("--strict", type=int, default=1, help="Strict mode (0/1)")
    ap.add_argument("--reason", required=True, help="Promotion reason string")
    ap.add_argument("--replay_check", type=int, default=1, help="Determinism replay check (0/1)")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Bars CSV for shadow replay")
    ap.add_argument("--model_path", default=DEFAULT_MODEL, help="Model path for shadow replay")
    ap.add_argument("--perf_days", type=int, default=promotion_gate.DEFAULT_PERF_DAYS, help="Perf window days")
    ap.add_argument("--debug", type=int, default=0, help="Debug mode (0/1)")
    return ap.parse_args()


def _print_status(status: str, **kwargs: Any) -> None:
    parts = [f"[promotion] status={status}"]
    for key, value in kwargs.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts))


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dir_path or None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(det.stable_json_dumps(payload))
        os.replace(tmp_path, path)
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
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        raise RuntimeError(f"destination_exists path={dest}")
    os.replace(src, dest)


def _handle_rejection(candidate_path: str, rejected_dir: str, stamp: str) -> str:
    if not candidate_path or not os.path.exists(candidate_path):
        return ""
    rejected_path = _unique_path(rejected_dir, os.path.basename(candidate_path), stamp)
    _move_file(candidate_path, rejected_path)
    return rejected_path


def run(args: argparse.Namespace) -> int:
    strict = int(args.strict) == 1
    replay_check = int(args.replay_check) == 1
    debug = int(args.debug) == 1

    candidate_path = args.candidate_path
    approved_path = _resolve_approved_path(args.approved_dir)

    gate_cfg = promotion_gate.PromotionGateConfig(perf_days=int(args.perf_days))

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

    if not result.ok:
        hashes = f"candidate={result.candidate_hash} approved={result.approved_hash} replay={result.replay_hash}"
        rejected_path = ""
        try:
            rejected_path = _handle_rejection(candidate_path, args.rejected_dir, stamp)
        except Exception as exc:
            _print_status("FAIL", reason=f"{result.reason} rejected_move_failed={exc}", hashes=hashes)
            return 1

        _print_status(
            "FAIL",
            reason=result.reason,
            hashes=hashes,
            rejected_path=rejected_path or "missing",
        )
        return 1

    if not approved_path or not os.path.exists(approved_path):
        hashes = f"candidate={result.candidate_hash} approved={result.approved_hash} replay={result.replay_hash}"
        _print_status("FAIL", reason=f"approved_missing path={approved_path}", hashes=hashes)
        return 1

    history_dir = args.history_dir
    os.makedirs(history_dir, exist_ok=True)

    archived_path = _unique_path(history_dir, os.path.basename(approved_path), stamp)
    try:
        _move_file(approved_path, archived_path)
        os.makedirs(os.path.dirname(approved_path), exist_ok=True)
        _move_file(candidate_path, approved_path)
    except Exception as exc:
        restore_err = None
        if os.path.exists(archived_path) and not os.path.exists(approved_path):
            try:
                _move_file(archived_path, approved_path)
            except Exception as restore_exc:
                restore_err = restore_exc
        reason = f"promotion_move_failed {exc}"
        if restore_err is not None:
            reason = f"{reason} restore_failed={restore_err}"
        hashes = f"candidate={result.candidate_hash} approved={result.approved_hash} replay={result.replay_hash}"
        _print_status("FAIL", reason=reason, hashes=hashes)
        return 1

    manifest = {
        "promoted_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": str(args.reason),
        "approved_hash": result.approved_hash,
        "candidate_hash": result.candidate_hash,
        "replay_hash": result.replay_hash,
        "performance_snapshot": result.performance_snapshot,
        "candidate_path": candidate_path,
        "approved_path": approved_path,
        "archived_path": archived_path,
    }
    manifest_path = _unique_path(history_dir, "promotion_manifest.json", stamp)
    _atomic_write_json(manifest_path, manifest)

    hashes = f"candidate={result.candidate_hash} approved={result.approved_hash} replay={result.replay_hash}"
    _print_status(
        "PASS",
        reason=result.reason,
        hashes=hashes,
        archived_path=archived_path,
        manifest_path=manifest_path,
    )
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return run(args)
    except Exception as exc:
        _print_status("FAIL", reason=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
