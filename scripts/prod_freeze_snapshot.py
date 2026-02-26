#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from scripts._freeze_utils import read_json, safe_mkdir, sha256_file, stable_json_dump

REQUIRED_FILES = [
    "live_daily_equity.csv",
    "live_daily_summary.json",
    "live_daily_regime.json",
    "meta_risk_state.json",
    "regime_perf_state.json",
]

HASH_KEYS = {
    "input_hash",
    "equity_hash",
    "regime_ledger_hash",
    "summary_hash",
    "regime_hash",
}

RUN_ID_KEYS = [
    ("input_hash", "in"),
    ("equity_hash", "eq"),
    ("regime_ledger_hash", "rl"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a production freeze snapshot.")
    parser.add_argument("--reports_dir", default="data/reports", help="Reports directory.")
    parser.add_argument("--out_dir", default="data/snapshots/phase3", help="Snapshot output root.")
    parser.add_argument("--strict", type=int, default=1, help="Strict mode (0/1).")
    parser.add_argument("--tag", default="", help="Optional tag.")
    return parser.parse_args()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if value:
            return value
    except Exception:
        return "unknown"
    return "unknown"


def _is_valid_hash(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return False
    val = value.strip()
    if not val:
        return False
    if val.lower() in {"missing", "skipped", "none", "null"}:
        return False
    return True


def _scan_hashes(obj: Any, found: Dict[str, str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in HASH_KEYS and _is_valid_hash(value) and key not in found:
                found[key] = str(value).strip()
            _scan_hashes(value, found)
    elif isinstance(obj, list):
        for item in obj:
            _scan_hashes(item, found)


def _collect_hashes(payloads: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for payload in payloads:
        if isinstance(payload, dict):
            _scan_hashes(payload, found)
    return found


def _build_run_id(hashes: Dict[str, str]) -> str:
    parts: List[str] = []
    for key, prefix in RUN_ID_KEYS:
        value = hashes.get(key)
        if _is_valid_hash(value):
            parts.append(f"{prefix}-{str(value)[:12]}")
    if parts:
        return "_".join(parts)
    return _timestamp_id()


def _extract_config(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not summary or not isinstance(summary, dict):
        return None
    config: Dict[str, Any] = {}
    for key in ("mode", "model_path", "power"):
        if key in summary:
            config[key] = summary.get(key)
    for key in ("config", "execution", "session", "fail_safe", "meta_risk"):
        value = summary.get(key)
        if isinstance(value, dict):
            config[key] = value
    return config or None


def _load_json_best_effort(path: str, label: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    data = read_json(path)
    if data is None:
        print(f"[freeze] WARN failed to parse {label}: {path}")
    return data


def main() -> int:
    args = _parse_args()
    reports_dir = os.path.abspath(args.reports_dir)
    out_dir = os.path.abspath(args.out_dir)
    strict = int(args.strict) == 1
    tag = args.tag.strip()

    print(f"[freeze] reports_dir={reports_dir}")
    print(f"[freeze] out_dir={out_dir}")
    print(f"[freeze] strict={int(strict)}")
    if tag:
        print(f"[freeze] tag={tag}")

    if not os.path.isdir(reports_dir):
        raise RuntimeError(f"[freeze] STATUS=FAIL reports_dir missing: {reports_dir}")

    required_paths = {name: os.path.join(reports_dir, name) for name in REQUIRED_FILES}
    missing = [name for name, path in required_paths.items() if not os.path.exists(path)]
    if missing:
        msg = "missing required artifacts: " + ", ".join(missing)
        if strict:
            raise RuntimeError(f"[freeze] STATUS=FAIL {msg}")
        print(f"[freeze] WARN {msg} (strict=0, skipping)")

    summary = _load_json_best_effort(required_paths.get("live_daily_summary.json", ""), "summary")
    meta_state = _load_json_best_effort(required_paths.get("meta_risk_state.json", ""), "meta_state")
    regime_state = _load_json_best_effort(
        required_paths.get("regime_perf_state.json", ""), "regime_perf_state"
    )
    regime_json = _load_json_best_effort(required_paths.get("live_daily_regime.json", ""), "regime")

    hashes = _collect_hashes([summary, meta_state, regime_state, regime_json])
    run_id = _build_run_id(hashes)

    snapshot_dir = os.path.join(out_dir, run_id)
    if os.path.exists(snapshot_dir):
        suffix = _timestamp_id()
        snapshot_dir = os.path.join(out_dir, f"{run_id}_{suffix}")
        print(f"[freeze] WARN snapshot_dir exists, using {snapshot_dir}")

    safe_mkdir(snapshot_dir)

    files_manifest: List[Dict[str, Any]] = []
    for name in REQUIRED_FILES:
        src = required_paths[name]
        if not os.path.exists(src):
            continue
        dst = os.path.join(snapshot_dir, name)
        shutil.copy2(src, dst)
        relpath = os.path.relpath(dst, snapshot_dir)
        files_manifest.append(
            {
                "name": name,
                "relpath": relpath,
                "sha256": sha256_file(dst),
                "bytes": int(os.path.getsize(dst)),
            }
        )
        print(f"[freeze] copied {name}")

    manifest: Dict[str, Any] = {
        "created_ts_utc": _utc_now().isoformat(),
        "git_commit": _git_commit(),
        "source_reports_dir": reports_dir,
        "run_id": os.path.basename(snapshot_dir),
        "files": files_manifest,
    }
    if tag:
        manifest["tag"] = tag
    config = _extract_config(summary)
    if config:
        manifest["config"] = config
    if hashes:
        manifest["deterministic_hashes"] = hashes

    manifest_path = os.path.join(snapshot_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(stable_json_dump(manifest))
        f.write("\n")

    print(f"[freeze] wrote {manifest_path}")
    print(f"[freeze] STATUS=PASS snapshot_dir={snapshot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
