#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rabit.rl.meta_risk import MetaRiskState
from rabit.state import atomic_io
from scripts import deterministic_utils as det

DEFAULT_BASE_DIR = os.path.join("data", "meta_states")
DEFAULT_APPROVED_PATH = os.path.join(DEFAULT_BASE_DIR, "current_approved", "meta_risk_state.json")
DEFAULT_HISTORY_DIR = os.path.join(DEFAULT_BASE_DIR, "history")
DEFAULT_LEDGER_PATH = os.path.join(DEFAULT_BASE_DIR, "ledger.jsonl")

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_ERROR = 2

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RollbackRejected(Exception):
    """User/input/state rejection (exit 1)."""


class RollbackInternalError(Exception):
    """Operational/IO/parse failure (exit 2 in strict mode)."""


@dataclass
class ApprovedEvent:
    line_no: int
    hash_value: str
    path_hint: str
    event_kind: str
    ts: str


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rollback current approved meta-risk state to a prior approved version.")
    ap.add_argument("--reason", required=True, help="Rollback reason string (required).")
    ap.add_argument("--strict", type=int, choices=[0, 1], default=1, help="Strict mode (0/1).")
    ap.add_argument(
        "--ledger_path",
        default=DEFAULT_LEDGER_PATH,
        help="Append-only promotion/rollback ledger path (JSONL).",
    )
    ap.add_argument(
        "--to_hash",
        default="",
        help="Target approved sha256 hash to rollback to.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Rollback N approved steps backward (1=previous approved).",
    )
    ap.add_argument("--dry_run", type=int, choices=[0, 1], default=0, help="Dry-run mode (0/1).")
    return ap.parse_args()


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_status(status: str, **fields: Any) -> None:
    parts = [f"[rollback] STATUS={status}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts))


def _normalize_hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    if _SHA256_RE.match(text):
        return text
    return ""


def _normalize_path(path: Any) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    return os.path.normpath(text)


def _atomic_copy_file(src: str, dest: str) -> None:
    if not src or not os.path.exists(src):
        raise RollbackInternalError(f"source_missing path={src}")
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_rollback_copy_", suffix=".json", dir=dest_dir or None)
    try:
        with open(src, "rb") as fsrc, os.fdopen(fd, "wb") as fdst:
            for chunk in iter(lambda: fsrc.read(1024 * 1024), b""):
                fdst.write(chunk)
            fdst.flush()
            os.fsync(fdst.fileno())
        if os.path.exists(dest):
            raise RollbackInternalError(f"destination_exists path={dest}")
        os.replace(tmp_path, dest)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_replace_from_source(src: str, dest: str) -> None:
    if not src or not os.path.exists(src):
        raise RollbackInternalError(f"source_missing path={src}")
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_rollback_replace_", suffix=".json", dir=dest_dir or None)
    try:
        with open(src, "rb") as fsrc, os.fdopen(fd, "wb") as fdst:
            for chunk in iter(lambda: fsrc.read(1024 * 1024), b""):
                fdst.write(chunk)
            fdst.flush()
            os.fsync(fdst.fileno())
        os.replace(tmp_path, dest)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _append_ledger_entry(ledger_path: str, entry: Dict[str, Any]) -> None:
    atomic_io.safe_append_jsonl(
        ledger_path,
        entry,
        ensure_ascii=False,
        sort_keys=True,
    )


def _load_ledger_events(ledger_path: str) -> List[Dict[str, Any]]:
    if not ledger_path or not os.path.exists(ledger_path):
        raise RollbackInternalError(f"ledger_missing path={ledger_path}")

    events: List[Dict[str, Any]] = []
    try:
        rows, skipped = atomic_io.read_jsonl_best_effort(ledger_path, return_skipped=True)
    except Exception as exc:
        raise RollbackInternalError(f"ledger_parse_error path={ledger_path} error={exc}") from exc
    for line_no, payload in enumerate(rows, start=1):
        if not isinstance(payload, dict):
            raise RollbackInternalError(f"ledger_parse_error line={line_no} reason=non_object_json")
        payload["_line_no"] = int(line_no)
        events.append(payload)
    if skipped > 0:
        _print_status("WARN", reason=f"ledger_tail_recovered path={ledger_path} skipped={int(skipped)}")
    return events


def _approved_hash_from_entry(entry: Dict[str, Any]) -> str:
    event_kind = str(entry.get("event", "")).strip().upper()
    status = str(entry.get("status", "")).strip().upper()
    decision = str(entry.get("decision", "")).strip().lower()

    if event_kind == "ROLLBACK":
        if status == "PASS":
            return _normalize_hash(entry.get("to_hash"))
        return ""

    if decision == "approved":
        return _normalize_hash(entry.get("candidate_hash"))
    if decision in {"rejected", "reject"}:
        return ""

    if event_kind in {"PROMOTE", "PROMOTION", "APPROVE", "APPROVED"}:
        if status and status not in {"PASS", "APPROVED", "OK", "SUCCESS"}:
            return ""
        for key in ("candidate_hash", "to_hash", "approved_hash", "state_hash", "hash"):
            value = _normalize_hash(entry.get(key))
            if value:
                return value
    return ""


def _path_hint_from_entry(entry: Dict[str, Any]) -> str:
    for key in ("to_path", "target_path", "approved_path", "state_path", "candidate_path", "path"):
        path = _normalize_path(entry.get(key))
        if path:
            return path
    return ""


def _extract_approved_events(events: Sequence[Dict[str, Any]]) -> List[ApprovedEvent]:
    out: List[ApprovedEvent] = []
    for row in events:
        hash_value = _approved_hash_from_entry(row)
        if not hash_value:
            continue
        out.append(
            ApprovedEvent(
                line_no=int(row.get("_line_no", 0)),
                hash_value=hash_value,
                path_hint=_path_hint_from_entry(row),
                event_kind=str(row.get("event") or row.get("decision") or "unknown"),
                ts=str(row.get("ts") or row.get("timestamp_utc") or ""),
            )
        )
    return out


def _scan_history_state_files(history_dir: str) -> List[Tuple[str, str, float]]:
    if not history_dir or not os.path.exists(history_dir):
        return []

    files: List[str] = []
    for root, dirs, names in os.walk(history_dir):
        dirs.sort()
        for name in sorted(names):
            lower = name.lower()
            if not lower.endswith(".json"):
                continue
            if not lower.startswith("meta_risk_state"):
                continue
            files.append(os.path.join(root, name))

    out: List[Tuple[str, str, float]] = []
    for path in files:
        hash_value = _normalize_hash(det.sha256_file(path))
        if not hash_value:
            continue
        try:
            mtime = float(os.path.getmtime(path))
        except Exception:
            mtime = 0.0
        out.append((hash_value, path, mtime))

    out.sort(key=lambda item: (-item[2], item[1]))
    return out


def _build_fallback_chain(current_hash: str, history_entries: Sequence[Tuple[str, str, float]]) -> List[str]:
    chain: List[str] = []
    seen: set[str] = set()
    if current_hash:
        chain.append(current_hash)
        seen.add(current_hash)
    for hash_value, _path, _mtime in history_entries:
        if hash_value in seen:
            continue
        seen.add(hash_value)
        chain.append(hash_value)
    return chain


def _expected_history_path(history_dir: str, hash_value: str) -> str:
    return os.path.join(history_dir, hash_value, "meta_risk_state.json")


def _resolve_target_path(
    target_hash: str,
    current_hash: str,
    current_path: str,
    history_dir: str,
    approved_events: Sequence[ApprovedEvent],
    history_entries: Sequence[Tuple[str, str, float]],
) -> str:
    candidates: List[str] = []
    if target_hash == current_hash and os.path.exists(current_path):
        candidates.append(current_path)

    for event in reversed(list(approved_events)):
        if event.hash_value != target_hash:
            continue
        if event.path_hint:
            candidates.append(event.path_hint)

    candidates.append(_expected_history_path(history_dir, target_hash))
    for hash_value, path, _mtime in history_entries:
        if hash_value == target_hash:
            candidates.append(path)

    seen: set[str] = set()
    for candidate in candidates:
        norm = _normalize_path(candidate)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if not os.path.exists(norm):
            continue
        if _normalize_hash(det.sha256_file(norm)) != target_hash:
            continue
        return norm
    return ""


def _validate_meta_state_file(path: str) -> bool:
    return MetaRiskState.load_json(None, path) is not None


def _build_rollback_event(
    reason: str,
    from_hash: str,
    to_hash: str,
    strict: bool,
    dry_run: bool,
    status: str,
    error: str = "",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event": "ROLLBACK",
        "ts": _utc_iso(),
        "reason": str(reason),
        "from_hash": from_hash or "missing",
        "to_hash": to_hash or "missing",
        "strict": int(bool(strict)),
        "dry_run": int(bool(dry_run)),
        "status": str(status),
        "error": str(error or ""),
    }
    return payload


def _validate_selection_inputs(to_hash: str, steps: Optional[int], dry_run: bool) -> None:
    if to_hash and steps is not None:
        raise RollbackRejected("invalid_args mutually_exclusive to_hash_and_steps")
    if not to_hash and steps is None and not dry_run:
        raise RollbackRejected("invalid_args exactly_one_of_to_hash_or_steps_required")
    if steps is not None and int(steps) <= 0:
        raise RollbackRejected("invalid_args steps_must_be_positive")
    if to_hash and not _normalize_hash(to_hash):
        raise RollbackRejected("invalid_args to_hash_must_be_sha256")


def _find_target_hash(
    *,
    to_hash: str,
    steps: Optional[int],
    current_hash: str,
    approved_events: Sequence[ApprovedEvent],
    source_mode: str,
) -> str:
    if to_hash:
        target_hash = _normalize_hash(to_hash)
        if not target_hash:
            raise RollbackRejected("invalid_target_hash")
        if target_hash == current_hash:
            return target_hash
        if source_mode == "ledger":
            approved_set = {event.hash_value for event in approved_events}
            if target_hash not in approved_set:
                raise RollbackRejected(f"target_not_approved hash={target_hash}")
        return target_hash

    if steps is None:
        raise RollbackRejected("invalid_args missing_target_selector")

    n = int(steps)
    if source_mode == "ledger":
        lineage = [event.hash_value for event in approved_events]
        if not lineage:
            raise RollbackRejected("no_approved_entries_in_ledger")
        current_index = -1
        for idx, hash_value in enumerate(lineage):
            if hash_value == current_hash:
                current_index = idx
        if current_index < 0:
            raise RollbackInternalError(
                f"current_hash_not_in_ledger from_hash={current_hash} approved_entries={len(lineage)}"
            )
        target_index = current_index - n
        if target_index < 0:
            raise RollbackRejected(
                f"no_prior_approved from_index={current_index} steps={n} approved_entries={len(lineage)}"
            )
        return lineage[target_index]

    raise RollbackInternalError(f"invalid_source_mode source={source_mode}")


def _ensure_backup_current(current_path: str, history_dir: str, current_hash: str) -> Tuple[str, bool]:
    backup_path = _expected_history_path(history_dir, current_hash)
    if os.path.exists(backup_path):
        return backup_path, False
    _atomic_copy_file(current_path, backup_path)
    return backup_path, True


def run(args: argparse.Namespace) -> int:
    strict = int(args.strict) == 1
    dry_run = int(args.dry_run) == 1
    reason = str(args.reason or "").strip()
    ledger_path = str(args.ledger_path or DEFAULT_LEDGER_PATH)
    to_hash_arg = str(args.to_hash or "").strip()
    steps = args.steps

    from_hash_for_log = "missing"
    to_hash_for_log = _normalize_hash(to_hash_arg) or "missing"
    allow_fail_log = not dry_run

    try:
        if not reason:
            raise RollbackRejected("reason_empty")
        _validate_selection_inputs(to_hash=to_hash_arg, steps=steps, dry_run=dry_run)

        current_path = DEFAULT_APPROVED_PATH
        history_dir = DEFAULT_HISTORY_DIR

        if not os.path.exists(current_path):
            raise RollbackInternalError(f"current_approved_missing path={current_path}")
        current_hash = _normalize_hash(det.sha256_file(current_path))
        if not current_hash:
            raise RollbackInternalError(f"current_hash_invalid path={current_path}")
        from_hash_for_log = current_hash

        history_entries = _scan_history_state_files(history_dir)

        source_mode = "ledger"
        approved_events: List[ApprovedEvent] = []
        try:
            ledger_events = _load_ledger_events(ledger_path)
            approved_events = _extract_approved_events(ledger_events)
        except RollbackInternalError as exc:
            if strict:
                allow_fail_log = False
                raise
            source_mode = "fallback"
            approved_events = []
            _print_status("WARN", reason=str(exc), mode="fallback_history_scan")

        discovery_only = dry_run and not to_hash_arg and steps is None
        if discovery_only:
            lineage_count = len(approved_events) if source_mode == "ledger" else 0
            fallback_chain = _build_fallback_chain(current_hash=current_hash, history_entries=history_entries)
            _print_status(
                "PASS",
                mode="discovery",
                source=source_mode,
                current_hash=current_hash,
                approved_entries=lineage_count,
                fallback_entries=max(0, len(fallback_chain) - 1),
            )
            if source_mode == "ledger":
                for event in approved_events[-5:]:
                    _print_status(
                        "DISCOVERY",
                        line=event.line_no,
                        hash=event.hash_value,
                        event=event.event_kind,
                        ts=event.ts or "missing",
                    )
            else:
                for idx, hash_value in enumerate(fallback_chain[:6]):
                    _print_status("DISCOVERY", idx=idx, hash=hash_value, source="history_scan")
            return EXIT_OK

        if source_mode == "fallback":
            fallback_chain = _build_fallback_chain(current_hash=current_hash, history_entries=history_entries)
            if to_hash_arg:
                selected_hash = _normalize_hash(to_hash_arg)
                if not selected_hash:
                    raise RollbackRejected("invalid_target_hash")
                if selected_hash not in set(fallback_chain):
                    raise RollbackRejected(f"target_not_found_in_fallback hash={selected_hash}")
                target_hash = selected_hash
            elif steps is not None:
                n = int(steps)
                if n >= len(fallback_chain):
                    raise RollbackRejected(
                        f"no_prior_approved_fallback steps={n} fallback_entries={len(fallback_chain) - 1}"
                    )
                target_hash = fallback_chain[n]
            else:
                target_hash = ""
        else:
            target_hash = _find_target_hash(
                to_hash=to_hash_arg,
                steps=steps,
                current_hash=current_hash,
                approved_events=approved_events,
                source_mode=source_mode,
            )

        if target_hash:
            to_hash_for_log = target_hash

        if not target_hash:
            raise RollbackRejected("target_unresolved")

        target_path = _resolve_target_path(
            target_hash=target_hash,
            current_hash=current_hash,
            current_path=current_path,
            history_dir=history_dir,
            approved_events=approved_events,
            history_entries=history_entries,
        )
        if not target_path:
            raise RollbackRejected(f"target_path_missing hash={target_hash}")
        if not _validate_meta_state_file(target_path):
            raise RollbackRejected(f"target_meta_state_invalid path={target_path}")

        target_hash_actual = _normalize_hash(det.sha256_file(target_path))
        if target_hash_actual != target_hash:
            raise RollbackInternalError(
                f"target_hash_mismatch expected={target_hash} actual={target_hash_actual} path={target_path}"
            )

        backup_path = _expected_history_path(history_dir, current_hash)

        if dry_run:
            _print_status(
                "PASS",
                dry_run=1,
                source=source_mode,
                from_hash=current_hash,
                to_hash=target_hash,
                current_path=current_path,
                target_path=target_path,
                backup_path=backup_path,
                action="rollback_preview",
            )
            return EXIT_OK

        if target_hash == current_hash:
            no_op_event = _build_rollback_event(
                reason=reason,
                from_hash=current_hash,
                to_hash=target_hash,
                strict=strict,
                dry_run=False,
                status="PASS",
                error="",
            )
            _append_ledger_entry(ledger_path, no_op_event)
            _print_status(
                "PASS",
                action="no_op",
                message="target_equals_current",
                from_hash=current_hash,
                to_hash=target_hash,
                target_path=target_path,
            )
            return EXIT_OK

        backup_created = False
        backup_path, backup_created = _ensure_backup_current(
            current_path=current_path,
            history_dir=history_dir,
            current_hash=current_hash,
        )

        _atomic_replace_from_source(target_path, current_path)
        replaced_hash = _normalize_hash(det.sha256_file(current_path))
        if replaced_hash != target_hash:
            if os.path.exists(backup_path):
                _atomic_replace_from_source(backup_path, current_path)
            raise RollbackInternalError(
                f"post_replace_hash_mismatch expected={target_hash} actual={replaced_hash} restored=1"
            )

        pass_event = _build_rollback_event(
            reason=reason,
            from_hash=current_hash,
            to_hash=target_hash,
            strict=strict,
            dry_run=False,
            status="PASS",
            error="",
        )

        try:
            _append_ledger_entry(ledger_path, pass_event)
        except Exception as exc:
            restore_err = None
            try:
                if os.path.exists(backup_path):
                    _atomic_replace_from_source(backup_path, current_path)
            except Exception as restore_exc:
                restore_err = restore_exc
            msg = f"ledger_write_failed error={exc}"
            if restore_err is not None:
                msg = f"{msg} restore_failed={restore_err}"
            raise RollbackInternalError(msg) from exc

        _print_status(
            "PASS",
            from_hash=current_hash,
            to_hash=target_hash,
            current_path=current_path,
            target_path=target_path,
            backup_path=backup_path,
            backup_created=int(backup_created),
            source=source_mode,
        )
        return EXIT_OK

    except RollbackRejected as exc:
        err = str(exc)
        _print_status("FAIL", reason=err, strict=int(strict))
        if allow_fail_log:
            fail_event = _build_rollback_event(
                reason=reason or "unknown",
                from_hash=from_hash_for_log,
                to_hash=to_hash_for_log,
                strict=strict,
                dry_run=dry_run,
                status="FAIL",
                error=err,
            )
            try:
                _append_ledger_entry(ledger_path, fail_event)
            except Exception as log_exc:
                _print_status("WARN", reason=f"fail_event_log_failed error={log_exc}")
        return EXIT_REJECT
    except RollbackInternalError as exc:
        err = str(exc)
        _print_status("FAIL", reason=err, strict=int(strict))
        if allow_fail_log:
            fail_event = _build_rollback_event(
                reason=reason or "unknown",
                from_hash=from_hash_for_log,
                to_hash=to_hash_for_log,
                strict=strict,
                dry_run=dry_run,
                status="FAIL",
                error=err,
            )
            try:
                _append_ledger_entry(ledger_path, fail_event)
            except Exception as log_exc:
                _print_status("WARN", reason=f"fail_event_log_failed error={log_exc}")
        return EXIT_ERROR if strict else EXIT_REJECT
    except Exception as exc:
        err = f"unexpected_error error={exc}"
        _print_status("FAIL", reason=err, strict=int(strict))
        if allow_fail_log:
            fail_event = _build_rollback_event(
                reason=reason or "unknown",
                from_hash=from_hash_for_log,
                to_hash=to_hash_for_log,
                strict=strict,
                dry_run=dry_run,
                status="FAIL",
                error=err,
            )
            try:
                _append_ledger_entry(ledger_path, fail_event)
            except Exception as log_exc:
                _print_status("WARN", reason=f"fail_event_log_failed error={log_exc}")
        return EXIT_ERROR if strict else EXIT_REJECT


def main() -> int:
    args = _parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
