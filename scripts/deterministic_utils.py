from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Set


def _json_default(value: Any) -> str:
    return str(value)


def stable_json_dumps(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def sha256_file(path: str) -> str:
    if not path or not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_json(data: Any) -> str:
    return sha256_text(stable_json_dumps(data))


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)


def _strip_keys(data: Any, ignore_keys: Optional[Set[str]]) -> Any:
    if not ignore_keys or not isinstance(data, dict):
        return data
    return {k: v for k, v in data.items() if k not in ignore_keys}


def _diff_values(expected: Any, current: Any, path: str, diffs: List[str]) -> None:
    if isinstance(expected, dict) and isinstance(current, dict):
        keys = sorted(set(expected) | set(current))
        for key in keys:
            next_path = f"{path}.{key}" if path else str(key)
            if key not in expected:
                diffs.append(f"{next_path}: expected=<missing> actual={current.get(key)!r}")
                continue
            if key not in current:
                diffs.append(f"{next_path}: expected={expected.get(key)!r} actual=<missing>")
                continue
            _diff_values(expected[key], current[key], next_path, diffs)
        return

    if isinstance(expected, list) and isinstance(current, list):
        if len(expected) != len(current):
            diffs.append(f"{path}: expected_len={len(expected)} actual_len={len(current)}")
        limit = min(len(expected), len(current))
        for idx in range(limit):
            next_path = f"{path}[{idx}]"
            _diff_values(expected[idx], current[idx], next_path, diffs)
        return

    if expected != current:
        diffs.append(f"{path}: expected={expected!r} actual={current!r}")


def diff_snapshots(
    expected: Dict[str, Any],
    current: Dict[str, Any],
    ignore_keys: Optional[Set[str]] = None,
) -> List[str]:
    exp = _strip_keys(expected, ignore_keys)
    cur = _strip_keys(current, ignore_keys)
    diffs: List[str] = []
    _diff_values(exp, cur, "", diffs)
    return diffs
