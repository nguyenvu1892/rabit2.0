from __future__ import annotations

import datetime as dt
import hashlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict

from rabit.state import atomic_io

DEFAULT_JSONL_PATH = os.path.join("logs", "meta_cycle.jsonl")
_WRITE_LOCK = threading.Lock()


def _utc_iso() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def generate_cycle_id(seed: str = "") -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    entropy = f"{stamp}:{os.getpid()}:{seed}"
    short_hash = hashlib.sha256(entropy.encode("utf-8")).hexdigest()[:10]
    return f"{stamp}-{short_hash}"


def _format_console_field(value: Any) -> str:
    text = str(value)
    return " ".join(text.split())


@dataclass(frozen=True)
class StructuredLogger:
    module: str
    jsonl_path: str = DEFAULT_JSONL_PATH
    context: Dict[str, Any] = field(default_factory=dict)

    def bind(self, **fields: Any) -> "StructuredLogger":
        merged = dict(self.context)
        for key, value in fields.items():
            if value is None:
                continue
            merged[str(key)] = value
        return StructuredLogger(module=self.module, jsonl_path=self.jsonl_path, context=merged)

    def _emit(self, level: str, event: str, fields: Dict[str, Any]) -> None:
        merged = dict(self.context)
        merged.update(fields)

        stage = str(merged.pop("stage", "") or "")
        cycle_id = str(merged.pop("cycle_id", "") or "")
        payload = {
            "ts_utc": _utc_iso(),
            "level": str(level).upper(),
            "module": self.module,
            "stage": stage,
            "event": str(event),
            "cycle_id": cycle_id,
            "fields": merged,
        }

        console_fields = dict(merged)
        if stage:
            console_fields["stage"] = stage
        if cycle_id:
            console_fields["cycle_id"] = cycle_id

        tail = " ".join(
            f"{key}={_format_console_field(console_fields[key])}" for key in sorted(console_fields.keys())
        )
        line = f"[struct][{payload['level']}][{self.module}] event={event}"
        if tail:
            line = f"{line} {tail}"

        with _WRITE_LOCK:
            print(line)
            try:
                atomic_io.safe_append_jsonl(
                    self.jsonl_path,
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except Exception as exc:
                print(f"[struct][WARN][{self.module}] event=log_write_failed error={_format_console_field(exc)}")

    def debug(self, *, event: str, **fields: Any) -> None:
        self._emit("DEBUG", event, fields)

    def info(self, *, event: str, **fields: Any) -> None:
        self._emit("INFO", event, fields)

    def warn(self, *, event: str, **fields: Any) -> None:
        self._emit("WARN", event, fields)

    def error(self, *, event: str, **fields: Any) -> None:
        self._emit("ERROR", event, fields)


def get_logger(name: str, *, jsonl_path: str = DEFAULT_JSONL_PATH) -> StructuredLogger:
    return StructuredLogger(module=str(name), jsonl_path=jsonl_path)
