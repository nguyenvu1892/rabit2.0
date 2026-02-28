from __future__ import annotations

import json
import os
import tempfile
from typing import Any


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


def atomic_write_text(path: str, text: str, *, encoding: str = "utf-8", suffix: str = ".tmp") -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

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
    payload: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        default=str,
    )
    atomic_write_text(path, text, suffix=".json")


def append_jsonl_line(path: str, line: str, *, encoding: str = "utf-8") -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    normalized = line.rstrip("\n")
    with open(path, "a", encoding=encoding) as f:
        f.write(normalized)
        f.write("\n")
        _fsync_file_obj(f)


def append_jsonl_record(
    path: str,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    separators: tuple[str, str] = (",", ":"),
) -> None:
    line = json.dumps(
        payload,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        separators=separators,
        default=str,
    )
    append_jsonl_line(path, line)
