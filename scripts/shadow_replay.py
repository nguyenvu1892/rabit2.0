from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

from scripts import _deterministic as detx
from scripts import deterministic_utils as det
from scripts import live_sim_daily as live
from rabit.rl.confidence_weighting import ConfidenceWeighter, ConfidenceWeighterConfig
from rabit.rl.meta_risk import MetaRiskConfig, MetaRiskState
from rabit.rl.regime_perf_feedback import RegimePerfConfig, RegimePerfFeedbackEngine
from rabit.meta import perf_history
from rabit.state import atomic_io
from rabit.state import versioned_state as vstate


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model_path", default="data/ars_best_theta_regime_bank.npz")
    ap.add_argument(
        "--meta_state_path",
        default=vstate.approved_state_path(vstate.DEFAULT_BASE_DIR),
    )
    ap.add_argument("--out_json", default=os.path.join("data", "reports", "shadow_replay_report.json"))
    ap.add_argument("--strict", type=int, default=1)
    ap.add_argument("--deterministic_check", type=int, default=1)
    ap.add_argument("--debug", type=int, default=0)
    return ap


def _parse_bool_flag(value: Any, default: bool = True) -> bool:
    try:
        return bool(int(value))
    except Exception:
        return bool(default)


def _merge_live_defaults(cli_args: argparse.Namespace) -> argparse.Namespace:
    base_args = live._build_arg_parser().parse_args([])
    base_args.csv = cli_args.csv
    base_args.model_path = cli_args.model_path
    base_args.meta_state_path = cli_args.meta_state_path
    base_args.debug = cli_args.debug
    base_args.deterministic_check = cli_args.deterministic_check

    base_args.meta_risk = 1
    base_args.meta_feedback = 0
    base_args.legacy_features = 0
    base_args.mt5_export = 0

    return base_args


def _theta_len_from_model(model: Any, n_features: Optional[int]) -> Optional[int]:
    if model is None:
        return None
    if hasattr(model, "policies") and isinstance(getattr(model, "policies"), dict):
        for policy in getattr(model, "policies").values():
            if hasattr(policy, "get_params_flat"):
                try:
                    return int(len(policy.get_params_flat()))
                except Exception:
                    continue
    if hasattr(model, "get_params_flat"):
        try:
            return int(len(model.get_params_flat()))
        except Exception:
            return None
    if n_features is not None:
        try:
            n_out = int(live._linear_policy_n_out())
            return int(n_out * (int(n_features) + 1))
        except Exception:
            return None
    return None


def _load_meta_state(
    args: argparse.Namespace,
    debug_enabled: bool,
) -> Tuple[Optional[MetaRiskState], Optional[str], Optional[str]]:
    meta_state = None
    loaded_path = None
    if int(getattr(args, "meta_risk", 0)) == 1:
        mcfg = MetaRiskConfig()
        meta_state = MetaRiskState(mcfg)
        loaded_state, loaded_path, load_err = vstate.load_approved_state(
            mcfg,
            base_dir=vstate.DEFAULT_BASE_DIR,
            legacy_path=args.meta_state_path,
            mirror_on_load=True,
        )
        if loaded_state is not None:
            meta_state = loaded_state
            meta_state.cfg = mcfg
            meta_state.daily_drawdown = 0.0
            meta_state.daily_equity_peak = 0.0
            meta_state.daily_date = None
            meta_state.loss_streak = 0
            meta_state.regime_freeze_until = {}
            if debug_enabled:
                print(f"[meta_risk] loaded state from {loaded_path}")
        elif load_err:
            strict_enabled = _parse_bool_flag(getattr(args, "strict", 1), default=True)
            if strict_enabled:
                raise RuntimeError(f"meta_state_load_failed {load_err}")
            if debug_enabled:
                print(f"[meta_risk] warn: cannot load state: {load_err}")
        meta_state.read_only = True
        if loaded_path is None:
            loaded_path = args.meta_state_path
    return meta_state, loaded_path, None


def _load_perf_engine(
    args: argparse.Namespace,
    debug_enabled: bool,
    read_only_state: bool,
) -> Optional[RegimePerfFeedbackEngine]:
    meta_risk_enabled = int(getattr(args, "meta_risk", 0)) == 1
    meta_feedback_enabled = int(getattr(args, "meta_feedback", 0)) == 1
    if not (meta_risk_enabled and meta_feedback_enabled):
        return None

    small_account_threshold = args.small_account_threshold
    if small_account_threshold is None:
        small_account_threshold = float(args.account_equity_start) * 0.9
    small_account_enabled = bool(meta_risk_enabled and float(small_account_threshold) > 0.0)
    perf_cfg = RegimePerfConfig(
        ewm_alpha=float(args.perf_ewm_alpha),
        min_trades_per_regime=int(args.perf_min_trades),
        min_scale=float(args.meta_scale_min),
        max_scale=float(args.meta_scale_max),
        max_scale_step=float(args.meta_scale_step),
        down_bad_winrate=float(args.down_bad_winrate),
        up_good_winrate=float(args.up_good_winrate),
        down_bad_pnl=float(args.down_bad_pnl),
        up_good_pnl=float(args.up_good_pnl),
        loss_streak_bad=int(args.loss_streak_bad),
        dd_bad=float(args.dd_bad),
        cooldown_days=int(args.cooldown_days),
        small_account_enabled=bool(small_account_enabled),
        small_account_threshold=float(small_account_threshold),
        small_account_floor=float(args.small_account_floor),
    )
    perf_engine = RegimePerfFeedbackEngine(cfg=perf_cfg, debug=debug_enabled)
    perf_state_path = args.regime_perf_state_path
    if os.path.exists(perf_state_path):
        try:
            perf_engine.load(perf_state_path, keep_cfg=True)
            if debug_enabled:
                print(f"[regime_perf] loaded state from {perf_state_path}")
        except Exception as exc:
            if debug_enabled:
                print(f"[regime_perf] warn: cannot load state: {exc}")
    if read_only_state and perf_engine is not None:
        perf_engine.read_only = True
    return perf_engine


def _build_guardrail_summary(breakdown: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bars_total": int(breakdown.get("bars_total", 0)),
        "bars_after_time_parse": int(breakdown.get("bars_after_time_parse", 0)),
        "bars_after_session_filter": int(breakdown.get("bars_after_session_filter", 0)),
        "bars_after_spread_filter": int(breakdown.get("bars_after_spread_filter", 0)),
        "signals_total": int(breakdown.get("signals_total", 0)),
        "signals_after_guardrails": int(breakdown.get("signals_after_guardrails", 0)),
        "final_allowed": int(breakdown.get("final_allowed", 0)),
        "session_reject": int(breakdown.get("session_reject", 0)),
        "spread_open_reject": int(breakdown.get("spread_open_reject", 0)),
        "spread_spike_reject": int(breakdown.get("spread_spike_reject", 0)),
        "guardrail_reject": int(breakdown.get("guardrail_reject", 0)),
        "policy_hold": int(breakdown.get("policy_hold", 0)),
        "size_zero": int(breakdown.get("size_zero", 0)),
    }


def _count_decisions(decisions: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = {"decisions_total": 0, "holds": 0, "long": 0, "short": 0}
    for row in decisions:
        if not isinstance(row, dict):
            continue
        totals["decisions_total"] += 1
        action = str(row.get("action", "")).upper()
        if action == "HOLD":
            totals["holds"] += 1
        elif action == "BUY":
            totals["long"] += 1
        elif action == "SELL":
            totals["short"] += 1
    return totals


def _run_shadow_replay_once(
    args: argparse.Namespace,
    debug_enabled: bool,
    deterministic_enabled: bool,
    read_only_state: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    execution_settings = live._default_execution_settings()
    session_settings = live._default_session_settings()
    session_settings["enable_london"] = _parse_bool_flag(args.enable_london, default=True)
    session_settings["enable_ny"] = _parse_bool_flag(args.enable_ny, default=True)
    if args.london_start is not None:
        session_settings["london_start"] = int(args.london_start)
    if args.london_end is not None:
        session_settings["london_end"] = int(args.london_end)
    if args.ny_start is not None:
        session_settings["ny_start"] = int(args.ny_start)
    if args.ny_end is not None:
        session_settings["ny_end"] = int(args.ny_end)
    no_session_filter = _parse_bool_flag(args.no_session_filter, default=False)
    no_spread_filter = _parse_bool_flag(args.no_spread_filter, default=False)

    deterministic_ctx: Optional[Dict[str, Any]] = None
    if deterministic_enabled:
        deterministic_ctx = detx.build_deterministic_context(args, execution_settings)

    df_raw, detected_sep = live.read_csv_smart(args.csv, debug=debug_enabled)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    bars_total_raw = int(len(df_raw))
    df, _ = live._normalize_columns(df_raw)

    mode = live.detect_input_mode(df)
    if mode != "bars":
        raise ValueError(f"Shadow replay requires bars CSV. detected_mode={mode}")

    df, time_col = live.resolve_timestamp(df)
    if time_col is None:
        raise ValueError(
            f"CSV must have time-like column. detected_cols={','.join(df.columns)} sep={repr(detected_sep)}"
        )
    df = live._prepare_bars_df(df, time_col=time_col, debug=debug_enabled)
    df["time"] = df[time_col]
    if df["time"].isna().all():
        raise ValueError(
            f"Time column parse failed. col={time_col} detected_cols={','.join(df.columns)} sep={repr(detected_sep)}"
        )

    model = live.load_regime_bank(args.model_path, debug=debug_enabled)
    model_n_features = live._infer_model_n_features(model)
    data_n_features = None
    n_features_match = None
    feature_shape: Optional[List[int]] = None

    df, _feat, feature_cols, X_all = live._build_feature_pipeline(df, debug=debug_enabled)
    if model_n_features <= 0:
        raise ValueError("Model n_features could not be inferred; check model file")
    data_n_features = int(X_all.shape[1])
    if data_n_features > int(model_n_features):
        if debug_enabled:
            live._debug_print(
                debug_enabled,
                f"[warn] feature_dim: data={data_n_features} > model={model_n_features}; slicing",
            )
        X_all = X_all[:, : int(model_n_features)]
        feature_cols = feature_cols[: int(model_n_features)]
        data_n_features = int(X_all.shape[1])
    if data_n_features < int(model_n_features):
        raise ValueError(
            "Feature dimension mismatch: FeatureBuilder produced "
            f"{data_n_features} < model_n_features={model_n_features}"
        )
    n_features_match = int(data_n_features) == int(model_n_features)
    feature_shape = [int(X_all.shape[0]), int(X_all.shape[1])]

    weighter_cfg = ConfidenceWeighterConfig(
        power=float(args.power),
        min_size=float(args.min_size),
        max_size=float(args.max_size),
        deadzone=float(args.deadzone),
    )
    weighter = ConfidenceWeighter(weighter_cfg)

    meta_state, meta_state_path_used, _ = _load_meta_state(args, debug_enabled)
    perf_engine = _load_perf_engine(args, debug_enabled, read_only_state=read_only_state)

    spread_open_cap = float(args.spread_open_cap)
    spread_spike_cap = float(args.spread_spike_cap)
    spread_open_cap_used = spread_open_cap if not no_spread_filter else 1e9
    spread_spike_cap_used = spread_spike_cap if not no_spread_filter else 1e9

    cfg = {
        "daily_dd_limit": float(args.daily_dd_limit),
        "max_loss_streak": int(args.max_loss_streak),
        "spread_spike_cap": float(spread_spike_cap_used),
        "spread_open_cap": float(spread_open_cap_used),
        "atr_period": int(args.atr_period),
        "risk_per_trade": float(args.risk_per_trade),
        "sl_min_price": float(args.sl_min_price),
        "contract_value_per_1_0_move": float(args.contract_value_per_1_0_move),
        "min_lot": float(args.min_lot),
        "max_lot": float(args.max_lot),
    }
    session_filter = None if no_session_filter else live._make_session_filter(session_settings)

    df["day"] = df["time"].dt.date.astype(str)
    days = df["day"].unique().tolist()
    ledger_last_ts = live._format_ts(df["time"].iloc[-1]) if len(df) > 0 else ""

    daily_rows: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []
    regime_stats: Dict[str, Any] = {"days": []}
    decisions_all: List[Dict[str, Any]] = []

    global_balance = float(args.account_equity_start)
    best_day = None
    worst_day = None
    total_trades = 0
    perf_scale_changes_total = 0
    breakdown_totals = {
        "bars_total": int(bars_total_raw),
        "bars_after_time_parse": int(len(df)),
        "bars_after_session_filter": 0,
        "bars_after_spread_filter": 0,
        "signals_total": 0,
        "signals_after_guardrails": 0,
        "final_allowed": 0,
        "total_trades": 0,
        "session_reject": 0,
        "spread_open_reject": 0,
        "spread_spike_reject": 0,
        "guardrail_reject": 0,
        "policy_hold": 0,
        "size_zero": 0,
    }

    for day in days:
        mask = df["day"] == day
        df_day = df[mask].copy().reset_index(drop=True)
        X_day = X_all[mask.values]
        trades_df, equity_df, day_pnl, dbg = live.run_one_day(
            df_day=df_day,
            model=model,
            weighter=weighter,
            meta_state=meta_state,
            meta_feedback=bool(int(args.meta_feedback)),
            regime_feedback=perf_engine,
            session_filter=session_filter,
            cfg=cfg,
            global_balance=global_balance,
            X_day=X_day,
            n_features=model_n_features,
            legacy_features=False,
            bypass_session=no_session_filter,
            bypass_spread=no_spread_filter,
            debug=debug_enabled,
            collect_decisions=True,
            collect_trade_meta=False,
        )
        start_equity = float(global_balance)
        end_equity = float(global_balance + day_pnl)

        if perf_engine is not None:
            update_info = perf_engine.update_from_trades(
                trades_df,
                day_key=day,
                account_equity=end_equity,
            )
            perf_scale_changes_total += len(update_info.get("scale_changes", []))
            live._debug_perf_update(debug_enabled, day, update_info, perf_engine)

        day_decisions = dbg.get("decision_rows", [])
        if isinstance(day_decisions, list) and day_decisions:
            decisions_all.extend(day_decisions)

        total_trades += int(dbg.get("allowed", 0))
        breakdown_totals["bars_after_session_filter"] += int(dbg.get("bars_after_session_filter", 0))
        breakdown_totals["bars_after_spread_filter"] += int(dbg.get("bars_after_spread_filter", 0))
        breakdown_totals["signals_total"] += int(dbg.get("signals_total", 0))
        breakdown_totals["signals_after_guardrails"] += int(dbg.get("signals_after_guardrails", 0))
        breakdown_totals["final_allowed"] += int(dbg.get("final_allowed", 0))
        breakdown_totals["session_reject"] += int(dbg.get("session_reject", 0))
        breakdown_totals["spread_open_reject"] += int(dbg.get("spread_open_reject", 0))
        breakdown_totals["spread_spike_reject"] += int(dbg.get("spread_spike_reject", 0))
        breakdown_totals["guardrail_reject"] += int(dbg.get("guardrail_reject", 0))
        breakdown_totals["policy_hold"] += int(dbg.get("policy_hold", 0))
        breakdown_totals["size_zero"] += int(dbg.get("size_zero", 0))

        row = {
            "day": day,
            "bars": int(dbg.get("bars", len(df_day))),
            "start_equity": start_equity,
            "end_equity": end_equity,
            "day_pnl": float(end_equity - start_equity),
            "day_peak": float(dbg.get("equity_peak", max(start_equity, end_equity))),
            "day_min": float(dbg.get("equity_min", min(start_equity, end_equity))),
            "intraday_dd": float(dbg.get("intraday_dd", max(0.0, (dbg.get("equity_peak", start_equity) - end_equity)))),
            "dd_stop": bool(dbg.get("dd_stop", False)),
            "end_loss_streak": int(dbg.get("end_loss_streak", 0)),
        }
        daily_rows.append(row)

        eq_row = dict(row)
        eq_row["equity"] = end_equity
        equity_rows.append(eq_row)

        if best_day is None or row["day_pnl"] > best_day["day_pnl"]:
            best_day = dict(row)
        if worst_day is None or row["day_pnl"] < worst_day["day_pnl"]:
            worst_day = dict(row)

        global_balance = end_equity

        regime_day = {
            "day": day,
            "signals": int(dbg.get("signals", 0)),
            "allowed": int(dbg.get("allowed", 0)),
            "dd_stop": bool(dbg.get("dd_stop", False)),
        }
        regime_counts = dbg.get("regime_counts", {})
        if isinstance(regime_counts, dict) and regime_counts:
            regime_day["regime_counts"] = regime_counts
        regime_stats["days"].append(regime_day)

    if perf_engine is not None:
        regime_stats["regimes"] = perf_engine.summary().get("regimes", {})
        regime_stats["regime_perf_meta"] = {
            "version": int(perf_engine.cfg.version),
            "last_update_ts": perf_engine.last_update_ts,
            "state_path": args.regime_perf_state_path,
        }

    breakdown_totals["total_trades"] = int(total_trades)

    summary = {
        "model_path": args.model_path,
        "mode": "regime_bank",
        "power": float(args.power),
        "config": {
            "account_equity_start": float(args.account_equity_start),
            "risk_per_trade": float(args.risk_per_trade),
            "sl_min_price": float(args.sl_min_price),
            "contract_value_per_1_0_move": float(args.contract_value_per_1_0_move),
            "min_lot": float(args.min_lot),
            "max_lot": float(args.max_lot),
        },
        "execution": execution_settings,
        "session": session_settings,
        "fail_safe": {
            "daily_dd_limit": float(args.daily_dd_limit),
            "max_loss_streak": int(args.max_loss_streak),
            "spread_spike_cap": float(args.spread_spike_cap),
            "spread_open_cap": float(args.spread_open_cap),
        },
        "meta_risk": {
            "enabled": int(args.meta_risk),
            "meta_feedback": int(args.meta_feedback),
            "state_path": args.meta_state_path,
        },
        "days": int(len(daily_rows)),
        "total_trades": int(total_trades),
        "total_pnl": float(sum(r["day_pnl"] for r in daily_rows)),
        "best_day": best_day or {},
        "worst_day": worst_day or {},
        "daily_table": daily_rows,
    }

    perf_summary = perf_history.compute_summary_from_daily_rows(daily_rows, total_trades=total_trades)
    perf_daily: List[Dict[str, Any]] = []
    for row in daily_rows:
        if not isinstance(row, dict):
            continue
        perf_daily.append(
            {
                "day": row.get("day"),
                "day_pnl": row.get("day_pnl"),
                "intraday_dd": row.get("intraday_dd"),
                "end_loss_streak": row.get("end_loss_streak"),
            }
        )

    ledger_state = None
    regime_ledger_hash = "skipped"
    regime_ledger_meta = None
    if int(args.meta_risk) == 1:
        _, ledger_state = live.update_regime_ledger(
            args.out_dir,
            daily_rows,
            regime_stats,
            span=int(args.regime_ledger_span),
            max_days=int(args.regime_ledger_max_days),
            debug=debug_enabled,
            last_update_ts=ledger_last_ts,
            read_only=True,
            return_state=True,
        )
        if ledger_state is not None:
            regime_ledger_hash, regime_ledger_meta = detx.hash_regime_ledger_dict(
                ledger_state.to_dict(), debug=bool(debug_enabled or deterministic_enabled)
            )

    determinism: Dict[str, Any] = {}
    input_hash = det.sha256_file(args.csv)
    meta_state_path_used = meta_state_path_used or args.meta_state_path
    meta_state_hash = det.sha256_file(meta_state_path_used) if meta_state_path_used else "missing"
    equity_hash = detx.hash_equity_rows(equity_rows) if equity_rows else "skipped"
    determinism.update(
        {
            "input_hash": input_hash,
            "equity_hash": equity_hash,
            "regime_ledger_hash": regime_ledger_hash,
        }
    )

    model_mode = "regime_bank"
    theta_len = _theta_len_from_model(model, model_n_features)
    decisions_counts = _count_decisions(decisions_all)

    report = {
        "input_csv": args.csv,
        "bars_loaded": int(len(df)),
        "start_ts": live._format_ts(df["time"].iloc[0]) if len(df) > 0 else None,
        "end_ts": live._format_ts(df["time"].iloc[-1]) if len(df) > 0 else None,
        "feature_source": "TradingEnv",
        "n_features": int(data_n_features) if data_n_features is not None else None,
        "theta_len": int(theta_len) if theta_len is not None else None,
        "model_mode": model_mode,
        "meta_state_path": str(meta_state_path_used or args.meta_state_path),
        "meta_state_sha256": meta_state_hash,
        "determinism": determinism,
        "guardrails": _build_guardrail_summary(breakdown_totals),
        "counts": {
            "decisions_total": int(decisions_counts.get("decisions_total", 0)),
            "holds": int(decisions_counts.get("holds", 0)),
            "long": int(decisions_counts.get("long", 0)),
            "short": int(decisions_counts.get("short", 0)),
            "trades_simulated": int(total_trades),
        },
        "perf_summary": perf_summary,
        "perf_daily": perf_daily,
        "status": "PASS",
        "reasons": [],
    }

    deterministic_snapshot: Optional[Dict[str, Any]] = None
    if deterministic_enabled:
        deterministic_regime_counts = live._aggregate_regime_counts(regime_stats)
        meta_scale_history_len = live._meta_scale_history_len(decisions_all)
        snapshot_payload = {
            "equity_hash": equity_hash,
            "summary_hash": detx.hash_summary(summary),
            "regime_hash": detx.hash_regime_stats(regime_stats),
            "regime_ledger_hash": regime_ledger_hash,
            "regime_ledger_meta": regime_ledger_meta,
            "total_pnl": float(summary.get("total_pnl", 0.0)),
            "total_trades": int(total_trades),
            "days": int(len(daily_rows)),
            "first_ts": report.get("start_ts"),
            "last_ts": report.get("end_ts"),
            "feature_shape": feature_shape,
            "model_n_features": int(model_n_features) if model_n_features is not None else None,
            "data_n_features": int(data_n_features) if data_n_features is not None else None,
            "n_features_match": n_features_match,
            "regime_counts": deterministic_regime_counts,
            "meta_scale_history_len": int(meta_scale_history_len),
            "decisions_total": int(decisions_counts.get("decisions_total", 0)),
        }
        deterministic_snapshot = detx.build_deterministic_snapshot(deterministic_ctx, snapshot_payload)

    return report, deterministic_snapshot


def _write_report(path: str, report: Dict[str, Any]) -> None:
    atomic_io.atomic_write_json(
        path,
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def main() -> None:
    ap = _build_arg_parser()
    cli_args = ap.parse_args()

    strict = _parse_bool_flag(cli_args.strict, default=True)
    deterministic_enabled = _parse_bool_flag(cli_args.deterministic_check, default=True)
    debug_enabled = _parse_bool_flag(cli_args.debug, default=False)

    args = _merge_live_defaults(cli_args)

    report: Optional[Dict[str, Any]] = None
    snapshot1: Optional[Dict[str, Any]] = None
    snapshot2: Optional[Dict[str, Any]] = None
    reasons: List[str] = []

    try:
        if deterministic_enabled:
            report1, snapshot1 = _run_shadow_replay_once(
                args,
                debug_enabled=debug_enabled,
                deterministic_enabled=True,
                read_only_state=True,
            )
            report = report1
            report2, snapshot2 = _run_shadow_replay_once(
                args,
                debug_enabled=debug_enabled,
                deterministic_enabled=True,
                read_only_state=True,
            )
            try:
                detx.compare_deterministic_snapshots(snapshot1, snapshot2)
            except Exception as exc:
                reasons.append(f"deterministic_check_failed: {exc}")
        else:
            report, _ = _run_shadow_replay_once(
                args,
                debug_enabled=debug_enabled,
                deterministic_enabled=False,
                read_only_state=True,
            )
    except Exception as exc:
        reasons.append(str(exc))

    if report is None:
        report = {
            "input_csv": cli_args.csv,
            "status": "FAIL",
            "reasons": reasons or ["unknown_error"],
        }

    if report.get("counts", {}).get("decisions_total", 0) == 0:
        reasons.append("decisions_total==0")

    if reasons:
        report["status"] = "FAIL"
        report["reasons"] = reasons
    else:
        report["status"] = "PASS"
        report["reasons"] = []

    _write_report(cli_args.out_json, report)

    if report.get("status") == "PASS":
        perf_payload = perf_history.build_perf_history_from_report(
            report=report,
            meta_state_path=cli_args.meta_state_path,
            source_csv=report.get("input_csv") or cli_args.csv,
        )
        if perf_payload is not None:
            perf_path = perf_history.perf_history_path(cli_args.meta_state_path)
            perf_history.write_perf_history(perf_path, perf_payload)
            print(f"[perf_history] wrote path={perf_path}")
        else:
            print("[perf_history] skip write reason=missing_fields")
    else:
        print(f"[perf_history] skip write status={report.get('status')}")

    if strict and report["status"] != "PASS":
        raise RuntimeError("; ".join(report.get("reasons", []) or ["shadow_replay_failed"]))


if __name__ == "__main__":
    main()
