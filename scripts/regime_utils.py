#!/usr/bin/env python
from __future__ import annotations

import datetime as dt
import math
import os
import re
from typing import Any, Dict, Optional, Tuple

from rabit.state import atomic_io

_DECISION_PASS = {"approved", "approve", "pass", "accepted"}
_DECISION_REJECT = {"rejected", "reject", "fail", "failed", "denied"}
_REASON_REGIME_RE = re.compile(r"\bregime=([a-zA-Z0-9_.-]+)\b")


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    if out in (float("inf"), float("-inf")):
        return None
    return float(out)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return float(lo)
    if value > hi:
        return float(hi)
    return float(value)


def _parse_ts_utc(value: Any) -> Optional[dt.datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _normalize_regime(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def _extract_regime_from_reason(reason: Any) -> Optional[str]:
    text = str(reason or "")
    match = _REASON_REGIME_RE.search(text)
    if not match:
        return None
    return _normalize_regime(match.group(1))


def _extract_regime(entry: Dict[str, Any]) -> str:
    direct_keys = ("regime", "current_regime", "market_regime")
    for key in direct_keys:
        if key in entry and str(entry.get(key, "")).strip():
            return _normalize_regime(entry.get(key))
    metrics = entry.get("metrics")
    if isinstance(metrics, dict):
        for key in direct_keys:
            if key in metrics and str(metrics.get(key, "")).strip():
                return _normalize_regime(metrics.get(key))
    from_reason = _extract_regime_from_reason(entry.get("reason"))
    if from_reason:
        return from_reason
    return "unknown"


def _decision_sign(decision: Any) -> int:
    text = str(decision or "").strip().lower()
    if text in _DECISION_PASS:
        return 1
    if text in _DECISION_REJECT:
        return -1
    return 0


def _empty_bucket() -> Dict[str, Any]:
    return {
        "n": 0,
        "approved": 0,
        "rejected": 0,
        "score_sum": 0.0,
        "score_n": 0,
        "pnl_sum": 0.0,
        "pnl_n": 0,
    }


def _finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    n = int(bucket.get("n", 0))
    approved = int(bucket.get("approved", 0))
    rejected = int(bucket.get("rejected", 0))
    pass_rate = (float(approved) / float(n)) if n > 0 else 0.0
    reject_rate = (float(rejected) / float(n)) if n > 0 else 0.0
    decision_score = pass_rate - reject_rate

    score_n = int(bucket.get("score_n", 0))
    avg_score_total: Optional[float]
    if score_n > 0:
        avg_score_total = float(bucket.get("score_sum", 0.0)) / float(score_n)
        score_signal = math.tanh(float(avg_score_total))
    else:
        avg_score_total = None
        score_signal = 0.0

    pnl_n = int(bucket.get("pnl_n", 0))
    avg_total_pnl: Optional[float]
    if pnl_n > 0:
        avg_total_pnl = float(bucket.get("pnl_sum", 0.0)) / float(pnl_n)
        pnl_signal = math.tanh(float(avg_total_pnl) / 100.0)
    else:
        avg_total_pnl = None
        pnl_signal = 0.0

    perf_score = _clamp(0.75 * decision_score + 0.15 * score_signal + 0.10 * pnl_signal, -1.0, 1.0)
    out: Dict[str, Any] = {
        "n": int(n),
        "approved": int(approved),
        "rejected": int(rejected),
        "pass_rate": round(pass_rate, 8),
        "reject_rate": round(reject_rate, 8),
        "perf_score": round(float(perf_score), 8),
    }
    if avg_score_total is not None:
        out["avg_score_total"] = round(float(avg_score_total), 8)
    if avg_total_pnl is not None:
        out["avg_total_pnl"] = round(float(avg_total_pnl), 8)
    return out


def load_live_regime_report(path: str) -> Dict[str, Any]:
    normalized_path = str(path or "").strip()
    if not normalized_path or not os.path.exists(normalized_path):
        return {}
    try:
        payload, _ = atomic_io.load_json_with_fallback(normalized_path)
    except Exception:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def summarize_regime_perf_from_ledger(ledger_path: str, lookback_days: int) -> Dict[str, Any]:
    normalized_path = str(ledger_path or "").strip()
    try:
        lookback = int(max(0, int(lookback_days)))
    except Exception:
        lookback = 0
    out: Dict[str, Any] = {
        "ledger_path": normalized_path,
        "lookback_days": int(lookback),
        "anchor_ts_utc": None,
        "window_start_ts_utc": None,
        "records_total": 0,
        "records_considered": 0,
        "per_regime": {},
        "global": _finalize_bucket(_empty_bucket()),
    }
    if not normalized_path or not os.path.exists(normalized_path):
        return out

    parsed_rows: list[tuple[Optional[dt.datetime], Dict[str, Any]]] = []
    rows, skipped = atomic_io.read_jsonl_best_effort(normalized_path, return_skipped=True)
    out["records_total"] = int(len(rows) + int(skipped))
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = _parse_ts_utc(row.get("timestamp_utc"))
        if ts is None:
            ts = _parse_ts_utc(row.get("ts"))
        parsed_rows.append((ts, row))

    anchor: Optional[dt.datetime] = None
    for ts, _row in parsed_rows:
        if ts is None:
            continue
        if anchor is None or ts > anchor:
            anchor = ts
    cutoff: Optional[dt.datetime] = None
    if lookback > 0 and anchor is not None:
        cutoff = anchor - dt.timedelta(days=int(lookback))
        out["anchor_ts_utc"] = anchor.strftime("%Y-%m-%dT%H:%M:%SZ")
        out["window_start_ts_utc"] = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif anchor is not None:
        out["anchor_ts_utc"] = anchor.strftime("%Y-%m-%dT%H:%M:%SZ")

    buckets: Dict[str, Dict[str, Any]] = {}
    for ts, row in parsed_rows:
        if cutoff is not None:
            if ts is None:
                continue
            if ts < cutoff:
                continue

        decision = str(row.get("decision", "")).strip().lower()
        if not decision:
            continue

        sign = _decision_sign(decision)
        regime = _extract_regime(row)
        for bucket_key in (regime, "_global"):
            bucket = buckets.setdefault(bucket_key, _empty_bucket())
            bucket["n"] = int(bucket["n"]) + 1
            if sign > 0:
                bucket["approved"] = int(bucket["approved"]) + 1
            elif sign < 0:
                bucket["rejected"] = int(bucket["rejected"]) + 1

            score = _safe_float(row.get("score_total"))
            if score is not None:
                bucket["score_sum"] = float(bucket["score_sum"]) + float(score)
                bucket["score_n"] = int(bucket["score_n"]) + 1

            total_pnl = _safe_float(row.get("total_pnl"))
            metrics = row.get("metrics")
            if total_pnl is None and isinstance(metrics, dict):
                total_pnl = _safe_float(metrics.get("total_pnl"))
            if total_pnl is not None:
                bucket["pnl_sum"] = float(bucket["pnl_sum"]) + float(total_pnl)
                bucket["pnl_n"] = int(bucket["pnl_n"]) + 1

        out["records_considered"] = int(out["records_considered"]) + 1

    per_regime = {
        key: _finalize_bucket(value)
        for key, value in sorted(buckets.items(), key=lambda kv: kv[0])
        if key != "_global"
    }
    global_bucket = buckets.get("_global", _empty_bucket())
    out["per_regime"] = per_regime
    out["global"] = _finalize_bucket(global_bucket)
    return out


def _extract_confidence(payload: Dict[str, Any], default: float) -> float:
    for key in ("confidence", "conf", "probability", "prob"):
        value = _safe_float(payload.get(key))
        if value is not None:
            return round(_clamp(value, 0.0, 1.0), 8)
    return round(_clamp(float(default), 0.0, 1.0), 8)


def pick_current_regime(regime_report: Dict[str, Any]) -> Tuple[str, float]:
    report = regime_report if isinstance(regime_report, dict) else {}

    for key in ("current", "latest", "state"):
        current = report.get(key)
        if not isinstance(current, dict):
            continue
        regime = current.get("regime")
        if regime is None:
            regime = current.get("current_regime")
        if regime is None:
            regime = current.get("name")
        if regime is None:
            continue
        return _normalize_regime(regime), _extract_confidence(current, default=0.5)

    explicit = report.get("current_regime")
    if explicit is None:
        explicit = report.get("regime")
    if explicit is not None and str(explicit).strip():
        return _normalize_regime(explicit), _extract_confidence(report, default=0.5)

    days = report.get("days")
    if isinstance(days, list):
        for day in reversed(days):
            if not isinstance(day, dict):
                continue
            counts = day.get("regime_counts")
            if not isinstance(counts, dict) or not counts:
                continue
            ranked = []
            total = 0
            for key, value in counts.items():
                count = _safe_int(value)
                if count is None or count < 0:
                    continue
                ranked.append((_normalize_regime(key), int(count)))
                total += int(count)
            if not ranked:
                continue
            ranked.sort(key=lambda item: (-item[1], item[0]))
            regime, top_count = ranked[0]
            conf = (float(top_count) / float(total)) if total > 0 else 0.0
            return regime, round(_clamp(conf, 0.0, 1.0), 8)

    regimes = report.get("regimes")
    if isinstance(regimes, dict) and regimes:
        ranked = []
        total = 0
        for key, payload in regimes.items():
            count = 0
            if isinstance(payload, dict):
                maybe_count = _safe_int(payload.get("n_trades"))
                if maybe_count is None:
                    maybe_count = _safe_int(payload.get("count"))
                if maybe_count is not None and maybe_count > 0:
                    count = int(maybe_count)
                regime_name = _normalize_regime(payload.get("regime", key))
            else:
                regime_name = _normalize_regime(key)
            ranked.append((regime_name, int(max(0, count))))
            total += int(max(0, count))
        if ranked:
            ranked.sort(key=lambda item: (-item[1], item[0]))
            regime, top_count = ranked[0]
            if total > 0:
                conf = float(top_count) / float(total)
            else:
                conf = 0.5
            return regime, round(_clamp(conf, 0.0, 1.0), 8)

    return "unknown", 0.0
