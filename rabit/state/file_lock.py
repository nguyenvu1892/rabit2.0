from __future__ import annotations

import datetime as dt
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, TextIO

try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover - non-Windows
    msvcrt = None

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - Windows
    fcntl = None


def _utc_iso(epoch: Optional[float] = None) -> str:
    now_epoch = float(epoch if epoch is not None else time.time())
    ts = dt.datetime.fromtimestamp(now_epoch, tz=dt.timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if out < 0:
        return float(fallback)
    return out


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _build_cmd_text(cmd: Optional[str]) -> str:
    if cmd:
        return str(cmd)
    try:
        return " ".join(sys.argv)
    except Exception:
        return "unknown"


def _read_payload(lock_path: str) -> Dict[str, Any]:
    if not os.path.exists(lock_path):
        return {}
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        raw = raw.lstrip("\ufeff")
        if not raw:
            return {}
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _write_payload(file_obj: TextIO, payload: Dict[str, Any]) -> None:
    file_obj.seek(0)
    file_obj.truncate(0)
    file_obj.write(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    )
    file_obj.write("\n")
    file_obj.flush()
    os.fsync(file_obj.fileno())


def _try_lock_nonblocking(file_obj: TextIO) -> bool:
    fd = file_obj.fileno()
    if msvcrt is not None:
        try:
            file_obj.seek(0)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    return True


def _unlock_file(file_obj: TextIO) -> None:
    fd = file_obj.fileno()
    if msvcrt is not None:
        try:
            file_obj.seek(0)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


@dataclass
class LockOwner:
    pid: int
    start_ts_utc: str
    start_epoch: float
    host: str
    cmd: str
    reason: str
    token: str
    active: int


@dataclass
class FileLockHandle:
    path: str
    owner: LockOwner
    file_obj: TextIO
    stale_break: bool


@dataclass
class LockAcquireResult:
    acquired: bool
    status: str
    age_s: float
    owner: LockOwner
    handle: Optional[FileLockHandle] = None


def _owner_from_payload(payload: Dict[str, Any], fallback_epoch: float) -> LockOwner:
    start_epoch = _safe_float(payload.get("start_epoch"), fallback_epoch)
    start_ts_utc = str(payload.get("start_ts_utc") or _utc_iso(start_epoch))
    return LockOwner(
        pid=_safe_int(payload.get("pid"), -1),
        start_ts_utc=start_ts_utc,
        start_epoch=float(start_epoch),
        host=str(payload.get("host") or "unknown"),
        cmd=str(payload.get("cmd") or "unknown"),
        reason=str(payload.get("reason") or "unknown"),
        token=str(payload.get("token") or ""),
        active=_safe_int(payload.get("active"), 1),
    )


def _new_owner(reason: str, cmd: Optional[str]) -> LockOwner:
    now_epoch = float(time.time())
    return LockOwner(
        pid=int(os.getpid()),
        start_ts_utc=_utc_iso(now_epoch),
        start_epoch=now_epoch,
        host=str(socket.gethostname() or "unknown"),
        cmd=_build_cmd_text(cmd),
        reason=str(reason or "unknown"),
        token=str(uuid.uuid4()),
        active=1,
    )


def _owner_to_payload(owner: LockOwner) -> Dict[str, Any]:
    return {
        "pid": int(owner.pid),
        "start_ts_utc": owner.start_ts_utc,
        "start_epoch": float(owner.start_epoch),
        "host": owner.host,
        "cmd": owner.cmd,
        "reason": owner.reason,
        "token": owner.token,
        "active": int(owner.active),
    }


def lock_owner_summary(owner: LockOwner) -> str:
    cmd = owner.cmd.replace("\n", " ").strip()
    if len(cmd) > 120:
        cmd = cmd[:117] + "..."
    return (
        f"pid={owner.pid} host={owner.host} start_ts_utc={owner.start_ts_utc} "
        f"reason={owner.reason} cmd={cmd}"
    )


def acquire_exclusive_lock(
    *,
    lock_path: str,
    reason: str,
    ttl_sec: float = 1800.0,
    force_lock_break: bool = False,
    cmd: Optional[str] = None,
) -> LockAcquireResult:
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    ttl = max(1.0, float(ttl_sec))
    while True:
        now_epoch = float(time.time())
        if not os.path.exists(lock_path):
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except FileExistsError:
                continue
            file_obj = os.fdopen(fd, "r+", encoding="utf-8")
            if not _try_lock_nonblocking(file_obj):
                file_obj.close()
                continue
            owner = _new_owner(reason=reason, cmd=cmd)
            payload = _owner_to_payload(owner)
            _write_payload(file_obj, payload)
            handle = FileLockHandle(path=lock_path, owner=owner, file_obj=file_obj, stale_break=False)
            return LockAcquireResult(
                acquired=True,
                status="acquired",
                age_s=0.0,
                owner=owner,
                handle=handle,
            )

        existing_payload = _read_payload(lock_path)
        existing_owner = _owner_from_payload(existing_payload, fallback_epoch=now_epoch)
        age_s = max(0.0, now_epoch - float(existing_owner.start_epoch))

        if int(existing_owner.active) == 0:
            try:
                file_obj = open(lock_path, "r+", encoding="utf-8")
            except FileNotFoundError:
                continue
            if not _try_lock_nonblocking(file_obj):
                file_obj.close()
                return LockAcquireResult(
                    acquired=False,
                    status="held",
                    age_s=age_s,
                    owner=existing_owner,
                    handle=None,
                )
            owner = _new_owner(reason=reason, cmd=cmd)
            payload = _owner_to_payload(owner)
            payload["recovered_from"] = "inactive"
            _write_payload(file_obj, payload)
            handle = FileLockHandle(path=lock_path, owner=owner, file_obj=file_obj, stale_break=False)
            return LockAcquireResult(
                acquired=True,
                status="acquired",
                age_s=age_s,
                owner=owner,
                handle=handle,
            )

        if age_s <= ttl:
            return LockAcquireResult(
                acquired=False,
                status="held",
                age_s=age_s,
                owner=existing_owner,
                handle=None,
            )

        if not force_lock_break:
            return LockAcquireResult(
                acquired=False,
                status="held_stale",
                age_s=age_s,
                owner=existing_owner,
                handle=None,
            )

        try:
            file_obj = open(lock_path, "r+", encoding="utf-8")
        except FileNotFoundError:
            continue

        if not _try_lock_nonblocking(file_obj):
            file_obj.close()
            return LockAcquireResult(
                acquired=False,
                status="held_stale_active",
                age_s=age_s,
                owner=existing_owner,
                handle=None,
            )

        owner = _new_owner(reason=reason, cmd=cmd)
        payload = _owner_to_payload(owner)
        payload["stale_break"] = 1
        payload["previous_owner"] = _owner_to_payload(existing_owner)
        _write_payload(file_obj, payload)
        handle = FileLockHandle(path=lock_path, owner=owner, file_obj=file_obj, stale_break=True)
        return LockAcquireResult(
            acquired=True,
            status="acquired_stale_break",
            age_s=age_s,
            owner=owner,
            handle=handle,
        )


def release_exclusive_lock(handle: FileLockHandle) -> bool:
    if handle is None:
        return False

    file_obj = handle.file_obj
    if file_obj is None:
        return False

    released = False
    try:
        payload = _owner_to_payload(handle.owner)
        payload["active"] = 0
        payload["released_ts_utc"] = _utc_iso()
        _write_payload(file_obj, payload)
        released = True
    except Exception:
        released = False
    finally:
        try:
            _unlock_file(file_obj)
        finally:
            try:
                file_obj.close()
            except Exception:
                pass
    return released
