#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import random
import subprocess
import sys
from typing import Any, Dict, List, Tuple

from rabit.state import atomic_io
from scripts import deterministic_utils as det

MUTATE_KEYS: Dict[str, Tuple[float, float]] = {
    "meta_scale": (0.5, 2.0),
    "up_cap": (1.0, 1.5),
    "down_floor": (0.3, 1.0),
    "cooldown_days": (0.0, 30.0),
    "ewma_alpha": (0.01, 0.5),
}

INT_KEYS = {"cooldown_days"}
_SAFE_MODE = "safe_mutate"
_SAFE_PROFILE = "safe"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _round8(value: Any) -> float:
    return round(float(value), 8)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    if out in (float("inf"), float("-inf")):
        return None
    return out


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
        choices=["copy", "copy_plus_mutate", _SAFE_MODE],
        default="copy",
        help="Candidate generation mode",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for mutations")
    ap.add_argument("--candidate_index", type=int, default=0, help="Candidate index (0-based)")
    ap.add_argument(
        "--mutation_profile",
        default=_SAFE_PROFILE,
        help='Mutation profile (currently only "safe" is implemented)',
    )
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
    atomic_io.atomic_write_text(path, text, suffix=".json", create_backup=True)


def _atomic_write_json(path: str, payload: Any) -> None:
    atomic_io.atomic_write_json(
        path,
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _load_json(path: str) -> Tuple[Dict[str, Any], str]:
    data, loaded_from = atomic_io.load_json_with_fallback(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"approved_state_invalid type={type(data).__name__}")
    return data, loaded_from


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


def _is_threshold_like_key(key: str) -> bool:
    lowered = str(key).strip().lower()
    if "threshold" in lowered:
        return True
    return lowered.endswith("_th") or lowered.endswith("_threshold")


def _existing_cap(parent: Dict[str, Any], *keys: str) -> float | None:
    for cap_key in keys:
        if cap_key not in parent:
            continue
        val = _safe_float(parent.get(cap_key))
        if val is not None:
            return float(val)
    return None


def _risk_per_trade_bounds(parent: Dict[str, Any], key: str) -> tuple[float, float]:
    lo = 0.005
    hi = 0.02
    existing_lo = _existing_cap(parent, f"{key}_min", f"min_{key}", "risk_per_trade_min")
    existing_hi = _existing_cap(parent, f"{key}_max", f"max_{key}", "risk_per_trade_max", f"{key}_cap")
    if existing_lo is not None:
        lo = max(lo, float(existing_lo))
    if existing_hi is not None:
        hi = min(hi, float(existing_hi))
    if hi < lo:
        hi = lo
    return lo, hi


def _clamp(value: float, lo: float | None, hi: float | None) -> float:
    out = float(value)
    if lo is not None and out < float(lo):
        out = float(lo)
    if hi is not None and out > float(hi):
        out = float(hi)
    return out


def _safe_mutate_value(
    *,
    parent: Dict[str, Any],
    key: str,
    base_value: float,
    rng: random.Random,
) -> float:
    lowered = str(key).strip().lower()
    if lowered == "risk_per_trade":
        factor = float(rng.choice([0.9, 1.1]))
        lo, hi = _risk_per_trade_bounds(parent, key)
        return _round8(_clamp(base_value * factor, lo, hi))

    if lowered == "meta_scale":
        delta = float(rng.choice([-0.05, 0.05]))
        return _round8(_clamp(base_value + delta, 0.5, 1.5))

    if lowered in {"cooldown_days", "cooldown_steps"}:
        delta = int(rng.choice([-1, 1]))
        out = int(round(base_value)) + delta
        if out < 0:
            out = 0
        return float(out)

    if _is_threshold_like_key(lowered):
        factor = float(rng.choice([0.95, 1.05]))
        lo = 0.0 if 0.0 <= float(base_value) <= 1.0 else None
        hi = 1.0 if 0.0 <= float(base_value) <= 1.0 else None
        return _round8(_clamp(base_value * factor, lo, hi))

    return _round8(base_value)


def _collect_safe_mutable_fields(
    obj: Any,
    out: List[Tuple[Dict[str, Any], str, str, float]],
    path: str = "",
) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            lowered = str(key).strip().lower()
            if _is_number(value) and (
                lowered in {"risk_per_trade", "meta_scale", "cooldown_days", "cooldown_steps"}
                or _is_threshold_like_key(lowered)
            ):
                out.append((obj, str(key), next_path, float(value)))
            if isinstance(value, (dict, list)):
                _collect_safe_mutable_fields(value, out, next_path)
        return
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            next_path = f"{path}[{idx}]" if path else f"[{idx}]"
            _collect_safe_mutable_fields(item, out, next_path)


def _apply_safe_mutations(
    obj: Dict[str, Any],
    rng: random.Random,
    changes: List[Tuple[str, float, float]],
    *,
    candidate_index: int,
    mutation_profile: str,
) -> None:
    profile = str(mutation_profile or _SAFE_PROFILE).strip().lower()
    if profile != _SAFE_PROFILE:
        raise RuntimeError(f"unsupported_mutation_profile profile={profile}")
    if int(candidate_index) <= 0:
        return

    mutable_fields: List[Tuple[Dict[str, Any], str, str, float]] = []
    _collect_safe_mutable_fields(obj, mutable_fields)
    if not mutable_fields:
        return

    rng.shuffle(mutable_fields)
    max_mutations = min(3, len(mutable_fields))
    target_mutations = int(rng.randint(1, max_mutations))
    applied = 0
    for parent, key, value_path, base in mutable_fields:
        before = float(base)
        mutated = _safe_mutate_value(parent=parent, key=key, base_value=before, rng=rng)
        if key in {"cooldown_days", "cooldown_steps"}:
            after: float = float(int(round(mutated)))
            parent[key] = int(round(after))
        else:
            after = float(mutated)
            parent[key] = float(after)
        if after != before:
            changes.append((value_path, _round8(before), _round8(after)))
            applied += 1
        if applied >= target_mutations:
            break


def _changes_payload(changes: List[Tuple[str, float, float]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for path, before, after in changes:
        payload.append(
            {
                "path": str(path),
                "before": _round8(before),
                "after": _round8(after),
            }
        )
    return payload


def _summarize_changes(changes: List[Tuple[str, float, float]], max_items: int = 8) -> List[str]:
    if not changes:
        return ["BASE"]
    summary: List[str] = []
    for path, before, after in changes[:max_items]:
        summary.append(f"{path}:{_round8(before)}->{_round8(after)}")
    if len(changes) > max_items:
        summary.append(f"...(+{len(changes) - max_items} more)")
    return summary


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
    candidate_index = int(getattr(args, "candidate_index", 0))
    mutation_profile = str(getattr(args, "mutation_profile", _SAFE_PROFILE) or _SAFE_PROFILE)

    if not os.path.exists(approved_path):
        msg = f"approved_state_missing path={approved_path}"
        if strict:
            raise RuntimeError(msg)
        _print_status("FAIL", reason=msg)
        return 1

    _ensure_distinct_paths(approved_path, candidate_out_path)

    approved_data, approved_loaded_from = _load_json(approved_path)
    with open(approved_loaded_from, "r", encoding="utf-8") as f:
        approved_raw = f.read()

    source_sha256 = det.sha256_file(approved_loaded_from)

    if debug:
        print(f"[candidate_gen] debug mode={mode} source_sha256={source_sha256}")

    if mode == "copy":
        _atomic_write_text(candidate_out_path, approved_raw)
        changes: List[Tuple[str, float, float]] = []
    elif mode == _SAFE_MODE:
        if candidate_index <= 0:
            _atomic_write_text(candidate_out_path, approved_raw)
            changes = []
        else:
            rng = random.Random(int(args.seed))
            candidate_data = copy.deepcopy(approved_data)
            changes = []
            _apply_safe_mutations(
                candidate_data,
                rng,
                changes,
                candidate_index=candidate_index,
                mutation_profile=mutation_profile,
            )
            if changes:
                _atomic_write_json(candidate_out_path, candidate_data)
            else:
                _atomic_write_text(candidate_out_path, approved_raw)
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
        "candidate_index": int(candidate_index),
        "mutation_profile": str(mutation_profile),
        "seed": int(args.seed),
        "reason": str(args.reason),
        "source_sha256": source_sha256,
        "candidate_sha256": candidate_sha256,
        "changes": _changes_payload(changes),
        "param_diff_summary": _summarize_changes(changes),
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
