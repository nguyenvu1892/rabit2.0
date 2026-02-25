from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import datetime
from typing import Any, Dict


def _ensure_reports_dir() -> None:
    os.makedirs("data/reports", exist_ok=True)


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _count_trades(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        if lines <= 1:
            return 0
        return int(lines - 1)
    except Exception:
        return 0


def _pick_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    preferred = [
        "total_pnl",
        "total_return",
        "sharpe",
        "profit_factor",
        "win_rate",
        "max_drawdown",
        "avg_pnl",
    ]
    picked: Dict[str, Any] = {}
    for key in preferred:
        if key in metrics:
            picked[key] = metrics[key]
    if picked:
        return picked
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            picked[key] = val
    return picked


def _run_live_sim(meta_risk: int, tag: str) -> Dict[str, Any]:
    state_path = f"data/reports/meta_risk_state_{tag}.json"
    out_state_path = state_path
    cmd = [
        sys.executable,
        "-m",
        "scripts.live_sim",
        "--meta_risk",
        str(meta_risk),
        "--meta_risk_state",
        state_path,
        "--meta_risk_out",
        out_state_path,
    ]
    print("[run]", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

    trades_src = "data/reports/live_sim_trades.csv"
    summary_src = "data/reports/live_sim_summary.json"
    trades_dst = f"data/reports/live_sim_trades_{tag}.csv"
    summary_dst = f"data/reports/live_sim_summary_{tag}.json"

    if os.path.exists(trades_src):
        shutil.copyfile(trades_src, trades_dst)
    if os.path.exists(summary_src):
        shutil.copyfile(summary_src, summary_dst)

    summary = _load_json(summary_src)
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    trades_count = _count_trades(trades_src)

    return {
        "meta_risk": int(meta_risk),
        "tag": tag,
        "trades": trades_count,
        "trades_path": trades_dst if os.path.exists(trades_dst) else trades_src,
        "summary_path": summary_dst if os.path.exists(summary_dst) else summary_src,
        "metrics": metrics,
        "picked_metrics": _pick_metrics(metrics) if isinstance(metrics, dict) else {},
    }


def _diff_metrics(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    for key in set(a.keys()) & set(b.keys()):
        av = a.get(key)
        bv = b.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            diff[key] = float(bv) - float(av)
    return diff


def main() -> None:
    _ensure_reports_dir()

    baseline = _run_live_sim(0, "meta0")
    calibrated = _run_live_sim(1, "meta1")

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runs": {
            "meta_risk_0": baseline,
            "meta_risk_1": calibrated,
        },
        "diff": {
            "trades": int(calibrated.get("trades", 0)) - int(baseline.get("trades", 0)),
            "metrics": _diff_metrics(
                baseline.get("picked_metrics", {}),
                calibrated.get("picked_metrics", {}),
            ),
        },
    }

    out_path = "data/reports/meta_risk_calibration_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[meta_risk_calibration] baseline trades:", baseline.get("trades", 0))
    print("[meta_risk_calibration] calibrated trades:", calibrated.get("trades", 0))
    print("[meta_risk_calibration] baseline metrics:", baseline.get("picked_metrics", {}))
    print("[meta_risk_calibration] calibrated metrics:", calibrated.get("picked_metrics", {}))
    print("[meta_risk_calibration] saved:", out_path)


if __name__ == "__main__":
    main()
