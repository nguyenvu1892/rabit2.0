from __future__ import annotations

import json
import os
import tempfile
from typing import Any, List, Tuple


def _fsync_file_obj(file_obj: Any) -> None:
    file_obj.flush()
    os.fsync(file_obj.fileno())


def _try_fsync_dir(path: str) -> None:
    if not path:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except Exception:
        return
    try:
        os.fsync(fd)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def _atomic_copy_file(src: str, dest: str) -> None:
    if not src or not dest or not os.path.exists(src):
        return
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".bak", dir=dest_dir or None)
    try:
        with open(src, "rb") as fsrc, os.fdopen(fd, "wb") as fdst:
            for chunk in iter(lambda: fsrc.read(1024 * 1024), b""):
                fdst.write(chunk)
            _fsync_file_obj(fdst)
        os.replace(tmp_path, dest)
        _try_fsync_dir(dest_dir)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _canonical_json_dumps(
    payload: Any,
    *,
    ensure_ascii: bool,
    sort_keys: bool,
    indent: int | None,
) -> str:
    separators = (",", ":") if indent is None else (",", ": ")
    return json.dumps(
        payload,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        default=str,
    )


def atomic_write_text(
    path: str,
    text: str,
    *,
    encoding: str = "utf-8",
    suffix: str = ".tmp",
    create_backup: bool = False,
) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    if create_backup and os.path.exists(path):
        try:
            _atomic_copy_file(path, f"{path}.bak")
        except Exception:
            # Backup is best-effort. Main write should still succeed.
            pass

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=suffix, dir=dir_path or None)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            _fsync_file_obj(f)
        os.replace(tmp_path, path)
        _try_fsync_dir(dir_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def atomic_write_json(
    path: str,
    obj: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    indent: int | None = None,
) -> None:
    text = _canonical_json_dumps(
        obj,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
    )
    atomic_write_text(path, text, suffix=".json", create_backup=True)


def load_json_with_fallback(
    path: str,
    *,
    encoding: str = "utf-8",
    allow_backup: bool = True,
) -> Tuple[Any, str]:
    if not path:
        raise RuntimeError("json_path_missing")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        with open(path, "r", encoding=encoding) as f:
            return json.load(f), path
    except json.JSONDecodeError as exc:
        if not allow_backup:
            raise RuntimeError(f"json_decode_error path={path} error={exc}") from exc
        bak_path = f"{path}.bak"
        if not os.path.exists(bak_path):
            raise RuntimeError(
                f"json_decode_error path={path} backup_missing path={bak_path} error={exc}"
            ) from exc
        try:
            with open(bak_path, "r", encoding=encoding) as f:
                return json.load(f), bak_path
        except Exception as bak_exc:
            raise RuntimeError(
                f"json_decode_error path={path} backup_load_failed path={bak_path} error={bak_exc}"
            ) from bak_exc


def _truncate_jsonl_tail_if_needed(file_obj: Any, raw: bytes) -> int:
    size = int(len(raw))
    if size <= 0:
        return 0

    target_size = size
    if raw[-1:] != b"\n":
        last_nl = raw.rfind(b"\n")
        target_size = 0 if last_nl < 0 else int(last_nl + 1)

    # Drop invalid trailing complete line(s) until tail is valid JSON or file is empty.
    while target_size > 0:
        line_end = target_size
        if raw[line_end - 1 : line_end] != b"\n":
            break
        prev_nl = raw.rfind(b"\n", 0, line_end - 1)
        line_start = 0 if prev_nl < 0 else int(prev_nl + 1)
        line_bytes = raw[line_start : line_end - 1].rstrip(b"\r")
        if not line_bytes.strip():
            target_size = line_start
            continue
        try:
            json.loads(line_bytes.decode("utf-8"))
            break
        except Exception:
            target_size = line_start

    if target_size != size:
        file_obj.seek(target_size)
        file_obj.truncate()
        _fsync_file_obj(file_obj)
    return target_size


def safe_append_jsonl(
    path: str,
    obj: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    encoding: str = "utf-8",
) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    payload = _canonical_json_dumps(
        obj,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=None,
    )
    raw_line = payload.encode(encoding, errors="strict") + b"\n"

    created = not os.path.exists(path)
    with open(path, "a+b") as f:
        f.seek(0)
        existing = f.read()
        _truncate_jsonl_tail_if_needed(f, existing)
        f.seek(0, os.SEEK_END)
        f.write(raw_line)
        _fsync_file_obj(f)
    if created:
        _try_fsync_dir(dir_path)


def read_jsonl_best_effort(
    path: str,
    *,
    encoding: str = "utf-8",
    return_skipped: bool = False,
) -> List[Any] | Tuple[List[Any], int]:
    rows: List[Any] = []
    skipped = 0
    if not path or not os.path.exists(path):
        return (rows, skipped) if return_skipped else rows

    with open(path, "rb") as f:
        raw = f.read()
    lines = raw.splitlines(keepends=False)

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            text = line.decode(encoding, errors="strict")
            rows.append(json.loads(text))
        except Exception:
            trailing_nonempty = any(rest.strip() for rest in lines[idx:])
            if trailing_nonempty:
                raise RuntimeError(f"jsonl_invalid_non_trailing path={path} line={idx}")
            skipped += 1
            break

    if return_skipped:
        return rows, int(skipped)
    return rows


def append_jsonl_line(path: str, line: str, *, encoding: str = "utf-8") -> None:
    normalized = line.rstrip("\n")
    safe_append_jsonl(path, json.loads(normalized), ensure_ascii=False, sort_keys=False, encoding=encoding)


def append_jsonl_record(
    path: str,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
) -> None:
    safe_append_jsonl(
        path,
        payload,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
    )
