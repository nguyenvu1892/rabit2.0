from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from rabit.state import atomic_io

DEFAULT_BASE_DIR = os.path.join("data", "meta_states")
DEFAULT_STATE_FILE = "meta_risk_state.json"


def _norm_path(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return os.path.normcase(os.path.abspath(path))
    except Exception:
        return path


def _same_path(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    return _norm_path(left) == _norm_path(right)


def ensure_state_dirs(base_dir: str = DEFAULT_BASE_DIR) -> Dict[str, str]:
    paths = {
        "base": base_dir,
        "current_approved": os.path.join(base_dir, "current_approved"),
        "candidate": os.path.join(base_dir, "candidate"),
        "rejected": os.path.join(base_dir, "rejected"),
        "history": os.path.join(base_dir, "history"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def approved_state_path(base_dir: str = DEFAULT_BASE_DIR, filename: str = DEFAULT_STATE_FILE) -> str:
    return os.path.join(base_dir, "current_approved", filename)


def history_dir(base_dir: str = DEFAULT_BASE_DIR) -> str:
    return os.path.join(base_dir, "history")


def list_versions(base_dir: str = DEFAULT_BASE_DIR) -> List[str]:
    hist = history_dir(base_dir)
    if not os.path.exists(hist):
        return []
    try:
        entries = [name for name in os.listdir(hist) if name and not name.startswith(".")]
    except Exception:
        return []
    return sorted(entries)


def legacy_state_path() -> str:
    return os.path.join("data", "reports", DEFAULT_STATE_FILE)


def _stable_json_text(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _atomic_write_text(path: str, text: str) -> None:
    atomic_io.atomic_write_text(path, text, suffix=".json")


def _atomic_write_json(path: str, data: Any) -> bool:
    try:
        payload = _stable_json_text(data)
        _atomic_write_text(path, payload)
        return True
    except Exception:
        return False


def load_approved_state(
    cfg: Any,
    base_dir: str = DEFAULT_BASE_DIR,
    legacy_path: Optional[str] = None,
    mirror_on_load: bool = True,
) -> Tuple[Optional[Any], str, Optional[str]]:
    from rabit.rl.meta_risk import MetaRiskState

    ensure_state_dirs(base_dir)
    approved_path = approved_state_path(base_dir)
    legacy_path = legacy_path or legacy_state_path()
    legacy_default = legacy_state_path()

    explicit_override = False
    if legacy_path and not _same_path(legacy_path, approved_path) and not _same_path(legacy_path, legacy_default):
        explicit_override = True

    load_errors: List[str] = []
    if explicit_override:
        if not os.path.exists(legacy_path):
            return None, "", f"explicit_state_missing path={legacy_path}"
        try:
            state = MetaRiskState.load_json(cfg, legacy_path)
            if state is not None:
                return state, legacy_path, None
            return None, "", "explicit_state_invalid"
        except Exception as exc:
            return None, "", f"explicit_state_error={exc}"

    if approved_path and os.path.exists(approved_path):
        try:
            state = MetaRiskState.load_json(cfg, approved_path)
            if state is not None:
                return state, approved_path, None
            load_errors.append("approved_state_invalid")
        except Exception as exc:
            load_errors.append(f"approved_state_error={exc}")

    if legacy_path and os.path.exists(legacy_path):
        try:
            state = MetaRiskState.load_json(cfg, legacy_path)
            if state is not None:
                if mirror_on_load:
                    _atomic_write_json(approved_path, state.to_dict())
                return state, legacy_path, None
            load_errors.append("legacy_state_invalid")
        except Exception as exc:
            load_errors.append(f"legacy_state_error={exc}")

    if load_errors:
        return None, "", "; ".join(load_errors)
    return None, "", None


def save_approved_state(
    state: Any,
    base_dir: str = DEFAULT_BASE_DIR,
    legacy_path: Optional[str] = None,
    debug: bool = False,
) -> bool:
    if state is None:
        return False
    if getattr(state, "read_only", False):
        return False

    ensure_state_dirs(base_dir)
    approved_path = approved_state_path(base_dir)
    legacy_path = legacy_path or legacy_state_path()
    payload = state.to_dict() if hasattr(state, "to_dict") else state

    legacy_ok = _atomic_write_json(legacy_path, payload)
    if legacy_ok:
        _atomic_write_json(approved_path, payload)

    if debug and legacy_ok and not os.path.exists(approved_path):
        raise RuntimeError(
            "TASK-4A FAIL: approved meta state missing after save "
            f"(approved_path={approved_path})"
        )

    return legacy_ok
