from __future__ import annotations

import datetime
from typing import Any, Dict, Optional, Set, List

from scripts import deterministic_utils as det
from rabit.meta import regime_ledger as regime_ledger_meta

DEFAULT_IGNORE_KEYS: Set[str] = {"timestamp", "regime_ledger_meta"}


def build_deterministic_snapshot(
    base: Optional[Dict[str, Any]],
    payload: Dict[str, Any],
    include_timestamp: bool = False,
) -> Optional[Dict[str, Any]]:
    if base is None:
        return None
    snapshot = dict(base)
    snapshot.update(payload)
    if include_timestamp:
        snapshot["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return snapshot


def print_regime_ledger_diff_hint(expected: Dict[str, Any], current: Dict[str, Any]) -> None:
    if not isinstance(expected, dict) or not isinstance(current, dict):
        return
    if expected.get("regime_ledger_hash") == current.get("regime_ledger_hash"):
        return

    exp_meta = expected.get("regime_ledger_meta")
    cur_meta = current.get("regime_ledger_meta")

    def _fmt_meta(meta: Any) -> str:
        if not isinstance(meta, dict):
            return "meta=n/a"
        return (
            "history_len="
            f"{meta.get('history_len')} start_day={meta.get('start_day')} "
            f"end_day={meta.get('end_day')} regimes={meta.get('regime_keys')}"
        )

    print(
        "[deterministic] regime_ledger diff hint: "
        f"expected({_fmt_meta(exp_meta)}) actual({_fmt_meta(cur_meta)})"
    )


def compare_deterministic_snapshots(
    first: Optional[Dict[str, Any]],
    second: Optional[Dict[str, Any]],
    ignore_keys: Optional[Set[str]] = None,
) -> None:
    if first is None or second is None:
        raise RuntimeError("TASK-3M FAIL: deterministic snapshot missing")

    print("[deterministic] compare run#1 vs run#2")
    print(f"[deterministic] input_hash={first.get('input_hash')}")
    print(f"[deterministic] equity_hash={first.get('equity_hash')}")
    print(f"[deterministic] regime_ledger_hash={first.get('regime_ledger_hash')}")
    print(f"[deterministic] total_pnl={first.get('total_pnl')}")

    ignore = ignore_keys or DEFAULT_IGNORE_KEYS
    diffs = det.diff_snapshots(first, second, ignore_keys=ignore)
    if diffs:
        for diff in diffs:
            print(f"[deterministic] diff {diff}")
        print_regime_ledger_diff_hint(first, second)
        print("[deterministic] STATUS=FAIL")
        raise RuntimeError("TASK-3M FAIL: deterministic violation")
    print("[deterministic] STATUS=PASS")


def build_deterministic_context(
    args,
    execution_settings: Dict[str, Any],
) -> Dict[str, Any]:
    args_payload = vars(args).copy()
    args_json = det.stable_json_dumps(args_payload)
    args_hash = det.sha256_text(args_json)
    seed = execution_settings.get("seed", 7) if isinstance(execution_settings, dict) else 7
    input_hash = det.sha256_file(args.csv)
    model_hash = det.sha256_file(args.model_path)
    meta_state_hash = "skipped"
    if int(args.meta_risk) == 1:
        meta_state_hash = det.sha256_file(args.meta_state_path)
    return {
        "input_hash": input_hash,
        "model_hash": model_hash,
        "args_hash": args_hash,
        "args_json": args_json,
        "seed": int(seed),
        "meta_state_hash": meta_state_hash,
    }


def hash_equity_rows(rows: List[Dict[str, Any]]) -> str:
    return det.hash_json(rows)


def hash_summary(summary: Dict[str, Any]) -> str:
    return det.hash_json(summary)


def hash_regime_stats(regime_stats: Dict[str, Any]) -> str:
    return det.hash_json(regime_stats)


def hash_regime_ledger_dict(raw: Any, debug: bool = False) -> tuple[str, Dict[str, Any]]:
    return regime_ledger_meta.hash_regime_ledger_dict(raw, debug=debug)


def hash_regime_ledger_file(path: str, debug: bool = False) -> tuple[str, Dict[str, Any]]:
    return regime_ledger_meta.hash_regime_ledger_file(path, debug=debug)
