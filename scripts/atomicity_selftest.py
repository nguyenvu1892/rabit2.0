#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List

from rabit.state import atomic_io

TEST_ROOT = os.path.join("data", "tmp_atomicity_test")


def _print(msg: str) -> None:
    print(f"[atomicity_selftest] {msg}")


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _clean_test_root(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _test_atomic_json_write(path: str) -> None:
    payload = {"name": "atomic", "version": 1, "ok": True}
    atomic_io.atomic_write_json(path, payload, indent=2, sort_keys=True, ensure_ascii=False)
    loaded, loaded_from = atomic_io.load_json_with_fallback(path)
    _assert(isinstance(loaded, dict), "atomic_json_write_invalid_type")
    _assert(loaded == payload, "atomic_json_write_mismatch")
    _assert(os.path.normpath(loaded_from) == os.path.normpath(path), "atomic_json_write_unexpected_source")
    _print("test_atomic_json_write=PASS")


def _test_json_fallback(path: str) -> None:
    atomic_io.atomic_write_json(path, {"version": 1}, indent=2, sort_keys=True, ensure_ascii=False)
    atomic_io.atomic_write_json(path, {"version": 2}, indent=2, sort_keys=True, ensure_ascii=False)
    bak_path = f"{path}.bak"
    _assert(os.path.exists(bak_path), "json_fallback_missing_backup")

    with open(path, "w", encoding="utf-8") as f:
        f.write('{"version":')
        f.flush()
        os.fsync(f.fileno())

    loaded, loaded_from = atomic_io.load_json_with_fallback(path)
    _assert(isinstance(loaded, dict), "json_fallback_invalid_type")
    _assert(int(loaded.get("version", -1)) == 1, "json_fallback_wrong_payload")
    _assert(os.path.normpath(loaded_from) == os.path.normpath(bak_path), "json_fallback_wrong_source")
    _print("test_json_fallback=PASS")


def _read_json_lines(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise RuntimeError(f"ledger_non_object line={idx}")
            rows.append(obj)
    return rows


def _test_ledger_partial_recovery(path: str) -> None:
    with open(path, "wb") as f:
        f.write(b'{"seq":1}\n')
        f.write(b'{"seq":2')
        f.flush()
        os.fsync(f.fileno())

    rows_before, skipped_before = atomic_io.read_jsonl_best_effort(path, return_skipped=True)
    _assert(skipped_before == 1, "ledger_partial_expected_skip_before_append")
    _assert(len(rows_before) == 1, "ledger_partial_expected_one_valid_before_append")

    atomic_io.safe_append_jsonl(path, {"seq": 3}, ensure_ascii=False, sort_keys=True)

    with open(path, "rb") as f:
        raw = f.read()
    _assert(raw.endswith(b"\n"), "ledger_missing_final_newline")

    rows_after, skipped_after = atomic_io.read_jsonl_best_effort(path, return_skipped=True)
    _assert(skipped_after == 0, "ledger_unexpected_skip_after_append")
    _assert(rows_after == [{"seq": 1}, {"seq": 3}], "ledger_rows_after_recovery_mismatch")

    rows_strict = _read_json_lines(path)
    _assert(rows_strict == rows_after, "ledger_strict_parse_mismatch")
    _print("test_ledger_partial_recovery=PASS")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Self-test for crash-safe atomic state/ledger IO.")
    ap.add_argument("--strict", type=int, choices=[0, 1], default=1, help="Strict mode (0/1)")
    return ap.parse_args()


def main() -> int:
    _parse_args()  # reserved for compatibility
    try:
        _clean_test_root(TEST_ROOT)

        state_path = os.path.join(TEST_ROOT, "meta_risk_state.json")
        ledger_path = os.path.join(TEST_ROOT, "ledger.jsonl")

        _test_atomic_json_write(state_path)
        _test_json_fallback(state_path)
        _test_ledger_partial_recovery(ledger_path)

        _print("STATUS=PASS")
        return 0
    except Exception as exc:
        _print(f"STATUS=FAIL reason={exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
