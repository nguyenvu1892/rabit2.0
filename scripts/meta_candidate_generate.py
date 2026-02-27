#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import random
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from scripts import deterministic_utils as det

MUTATE_KEYS: Dict[str, Tuple[float, float]] = {
    "meta_scale": (0.5, 2.0),
    "up_cap": (1.0, 1.5),
    "down_floor": (0.3, 1.0),
    "cooldown_days": (0.0, 30.0),
    "ewma_alpha": (0.01, 0.5),
}

INT_KEYS = {"cooldown_days"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate candidate meta-risk state from approved state.")
    ap.add_argument(
        "--approved_state_path",
        default="data/meta_states/current_approved/meta_risk_state.json",
        help="Path to approved meta_risk_state.json",
    )
    ap.add_argument(
        "--candidate_out_path",
        default="data/meta_states/candidate/meta_risk_state.json",
        help="Output candidate meta_risk_state.json path",
    )
    ap.add_argument(
        "--candidate_manifest",
        default="data/meta_states/candidate/manifest.json",
        help="Output manifest.json path",
    )
    ap.add_argument(
        "--mode",
        choices=["copy", "copy_plus_mutate"],
        default="copy",
        help="Candidate generation mode",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for mutations")
    ap.add_argument("--reason", default="task4b", help="Reason string for manifest metadata")
    ap.add_argument("--strict", type=int, default=1, help="Strict mode (0/1)")
    ap.add_argument("--debug", type=int, default=0, help="Debug mode (0/1)")
    return ap.parse_args()


def _print_status(status: str, **kwargs: Any) -> None:
    parts = [f"[candidate_gen] STATUS={status}"]
    for key, value in kwargs.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts))


def _atomic_write_text(path: str, text: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dir_path or None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_write_json(path: str, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    _atomic_write_text(path, text)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"approved_state_invalid type={type(data).__name__}")
    return data


def _perturb(value: float, key: str, rng: random.Random) -> float:
    lo, hi = MUTATE_KEYS[key]
    delta = value * rng.uniform(-0.05, 0.05)
    new_val = value + delta
    if new_val < lo:
        new_val = lo
    if new_val > hi:
        new_val = hi
    if key in INT_KEYS:
        new_val = float(int(round(new_val)))
        if new_val < lo:
            new_val = lo
        if new_val > hi:
            new_val = hi
    return new_val


def _apply_mutations(
    obj: Any,
    rng: random.Random,
    changes: List[Tuple[str, float, float]],
    path: str = "",
) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            if key in MUTATE_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                base = float(value)
                new_val = _perturb(base, key, rng)
                if key in INT_KEYS:
                    new_val = int(round(new_val))
                obj[key] = new_val
                changes.append((next_path, base, float(new_val)))
            else:
                _apply_mutations(value, rng, changes, next_path)
        return
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            _apply_mutations(item, rng, changes, f"{path}[{idx}]" if path else f"[{idx}]")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def _ensure_distinct_paths(approved_path: str, candidate_path: str) -> None:
    try:
        approved_norm = os.path.normcase(os.path.abspath(approved_path))
        candidate_norm = os.path.normcase(os.path.abspath(candidate_path))
    except Exception:
        approved_norm = approved_path
        candidate_norm = candidate_path
    if approved_norm == candidate_norm:
        raise RuntimeError("candidate_out_path must differ from approved_state_path")


def run(args: argparse.Namespace) -> int:
    strict = int(args.strict) == 1
    debug = int(args.debug) == 1

    approved_path = args.approved_state_path
    candidate_out_path = args.candidate_out_path
    candidate_manifest = args.candidate_manifest
    mode = args.mode

    if not os.path.exists(approved_path):
        msg = f"approved_state_missing path={approved_path}"
        if strict:
            raise RuntimeError(msg)
        _print_status("FAIL", reason=msg)
        return 1

    _ensure_distinct_paths(approved_path, candidate_out_path)

    approved_raw: str
    with open(approved_path, "r", encoding="utf-8") as f:
        approved_raw = f.read()
    approved_data = _load_json(approved_path)

    source_sha256 = det.sha256_file(approved_path)

    if debug:
        print(f"[candidate_gen] debug mode={mode} source_sha256={source_sha256}")

    if mode == "copy":
        _atomic_write_text(candidate_out_path, approved_raw)
        changes: List[Tuple[str, float, float]] = []
    else:
        rng = random.Random(int(args.seed))
        candidate_data = copy.deepcopy(approved_data)
        changes = []
        _apply_mutations(candidate_data, rng, changes)
        _atomic_write_json(candidate_out_path, candidate_data)

    candidate_sha256 = det.sha256_file(candidate_out_path)

    now = dt.datetime.now(dt.timezone.utc)
    created_ts_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    version_id = now.strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "created_ts_utc": created_ts_utc,
        "version_id": version_id,
        "approved_state_path": approved_path,
        "candidate_out_path": candidate_out_path,
        "mode": mode,
        "seed": int(args.seed),
        "reason": str(args.reason),
        "source_sha256": source_sha256,
        "candidate_sha256": candidate_sha256,
        "git_commit": _git_commit(),
    }

    _atomic_write_json(candidate_manifest, manifest)

    if debug and changes:
        for path, before, after in changes:
            print(f"[candidate_gen] debug mutate {path}: {before} -> {after}")

    _print_status(
        "PASS",
        candidate_out_path=candidate_out_path,
        source_sha256=source_sha256,
        candidate_sha256=candidate_sha256,
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
