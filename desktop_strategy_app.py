"""Integrated desktop app for A-share backtesting and intraday monitoring."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
import traceback
import tempfile
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

import app as engine
from ml_decision import AccountState, DecisionConfig
from ml_decision.data_sources import fetch_external_factor_frame, merge_external_factors
from ml_decision.engine import result_to_jsonable, run_holding_decision
from ml_decision.inference import production_result_to_jsonable, run_production_holding_decision
from ml_decision.model_registry import (
    evaluate_candidate_model,
    ProductionModelLoader,
    ProductionModelNotFound,
    promote_candidate_model,
    rollback_production_model,
)
from ml_decision.actions import apply_account_constraints

MONITOR_SNAPSHOT_PATH = engine.CACHE_DIR / "monitor_snapshot.json"
ML_PORTFOLIO_SETTINGS_PATH = engine.CACHE_DIR / "ml_portfolio_settings.json"
ML_STOCK_POOL_PATH = engine.CACHE_DIR / "ml_stock_pool.json"
UI_DEBUG_LOG_PATH = engine.CACHE_DIR / "ui_debug.log"
RESIZE_REDRAW_DELAY_MS = 260
CONTENT_MIN_WIDTH = 1180
CONTENT_MIN_HEIGHT = 650
CONTENT_RESIZE_FINAL_DELAY_MS = 90
ML_BATCH_PARALLEL_WORKERS = max(1, int(os.environ.get("ML_BATCH_PARALLEL_WORKERS", "3") or "3"))

try:
    import ttkbootstrap as tb
except Exception:
    tb = None


@dataclass
class WorkerMessage:
    kind: str
    payload: Any = None
    error: str | None = None


class BacktestCancelled(Exception):
    pass


def _short_error_text(error: object) -> str:
    text = str(error or "").strip()
    if not text:
        return "未知错误"
    lower = text.lower()
    if "proxyerror" in lower or "unable to connect to proxy" in lower or "remote end closed connection" in lower:
        return "网络/代理连接失败；程序已尝试使用本地缓存，若仍失败请稍后重试或检查代理。"
    if "无法联网拉取" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else text
    if "traceback" in lower:
        for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
            if line and not line.startswith("File ") and not line.startswith("^"):
                return line[:260]
    return text[:260]


def _ui_debug(message: str) -> None:
    try:
        engine.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if UI_DEBUG_LOG_PATH.exists() and UI_DEBUG_LOG_PATH.stat().st_size > 2_000_000:
            UI_DEBUG_LOG_PATH.write_text("", encoding="utf-8")
        with UI_DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except Exception:
        pass


def _scan_row_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "strategy_type": str(row.get("strategy_type", "")),
        "strategy_label": str(row.get("strategy_label", "")),
        "fast": int(row.get("fast", 0) or 0),
        "slow": int(row.get("slow", 0) or 0),
        "total_return_pct": float(row.get("total_return_pct", np.nan)),
        "max_drawdown_pct": float(row.get("max_drawdown_pct", np.nan)),
        "sharpe": float(row.get("sharpe", np.nan)),
        "trades": int(row.get("trades", 0) or 0),
        "final_value": float(row.get("final_value", np.nan)),
        "score": float(row.get("score", np.nan)),
    }


def _daily_gate_from_backtest_payload(
    form: dict[str, str],
    symbol: str,
    data: pd.DataFrame,
    best: pd.Series,
    fast_line: pd.Series,
    slow_line: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    horizon: str,
) -> dict[str, Any]:
    risk = form.get("risk", "normal")
    strategy_filter = form.get("strategy_type", "auto")
    cash = float(form.get("cash") or 100000)
    fee = float(form.get("fee") or 0.0003)
    latest_date = data.index[-1]
    latest_close = float(data["Close"].iloc[-1])
    strategy_type = str(best.get("strategy_type", "sma"))
    strategy_label = engine.STRATEGY_TYPES.get(strategy_type, strategy_type)
    fast = int(best["fast"])
    slow = int(best["slow"])
    save_strategy = str(form.get("_save_strategy", "")).lower() in {"1", "true", "yes"}
    active_strategy = str(form.get("_active_strategy", "")).lower() in {"1", "true", "yes"}
    selected_for_left = str(form.get("_selected_for_left", "")).lower() in {"1", "true", "yes"}
    cache_strategy_filter = f"{strategy_type}_{fast}_{slow}" if save_strategy else strategy_filter
    cache_key = engine.strategy_cache_key(symbol, form.get("start", "20200101"), form.get("adjust", "qfq"), cash, fee, horizon, cache_strategy_filter, risk)
    latest_fast = float(fast_line.iloc[-1])
    latest_slow = float(slow_line.iloc[-1])
    entry_today = bool(entries.iloc[-1])
    exit_today = bool(exits.iloc[-1])
    in_trend = engine.strategy_in_trend(strategy_type, latest_fast, latest_slow, latest_close)
    last_side, last_date = engine.last_signal_date(entries, exits)
    risk_factor = {"tight": 0.8, "normal": 1.0, "loose": 1.3}[risk] if horizon == "short" else {"tight": 1.2, "normal": 1.6, "loose": 2.2}[risk]
    lookback = engine.STRATEGY_GRIDS[horizon]["lookback"]
    atr_value = float(engine.atr(data).iloc[-1])
    recent_low = float(data["Low"].tail(lookback).min())
    recent_high = float(data["High"].tail(lookback).max())
    trend_stop = max(latest_slow, latest_close - risk_factor * atr_value)
    structure_stop = recent_low
    stop_line = min(trend_stop, latest_close * (0.992 if horizon == "short" else 0.985)) if in_trend else max(latest_slow, latest_close * 1.01)
    result = {
        "strategy_label": strategy_label,
        "best_params": f"{fast}/{slow}",
        "best": _scan_row_payload(best),
        "signal_lines": [f"Latest daily: {latest_date:%Y-%m-%d}, close {engine.money(latest_close)}."],
        "daily_signal": {
            "date": latest_date.strftime("%Y-%m-%d"),
            "strategy_type": strategy_type,
            "strategy_label": strategy_label,
            "fast": fast,
            "slow": slow,
            "entry_today": entry_today,
            "exit_today": exit_today,
            "in_trend": in_trend,
            "latest_close": latest_close,
            "stop_line": stop_line,
            "structure_stop": structure_stop,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "last_side": last_side,
            "last_date": last_date,
        },
    }
    engine.attach_ml_risk_snapshot(result, data, fast, slow, strategy_type, stop_line)
    engine.DAILY_GATE_CACHE[cache_key] = result
    if save_strategy:
        if active_strategy:
            result["_active_for_trading"] = True
        if selected_for_left or active_strategy:
            result["_selected_for_left"] = True
        engine.save_daily_gate(cache_key, result)
    return result


def _save_all_scan_strategies_from_payload(
    form: dict[str, str],
    symbol: str,
    data: pd.DataFrame,
    scan: pd.DataFrame,
    horizon: str,
    active_index: Any | None,
) -> dict[str, Any] | None:
    active_gate: dict[str, Any] | None = None
    for index, row in scan.iterrows():
        strategy_type = str(row.get("strategy_type", "sma"))
        fast = int(row["fast"])
        slow = int(row["slow"])
        fast_line, slow_line, entries, exits = engine.strategy_signals(data, fast, slow, horizon, strategy_type)
        save_form = form.copy()
        save_form["_save_strategy"] = "1"
        if active_index is not None and index == active_index:
            save_form["_active_strategy"] = "1"
            save_form["_selected_for_left"] = "1"
        gate = _daily_gate_from_backtest_payload(save_form, symbol, data, row, fast_line, slow_line, entries, exits, horizon)
        if index == active_index:
            active_gate = gate
    return active_gate


def _compute_backtest_payload(form: dict[str, str]) -> dict[str, Any]:
    symbol = engine.resolve_stock_identifier(form["symbol"])
    start = form["start"]
    adjust = form.get("adjust", "qfq")
    cash = float(form.get("cash") or 100000)
    fee = float(form.get("fee") or 0.0003)
    horizon = form.get("horizon", "short")
    strategy_filter = form.get("strategy_type", "auto")
    if horizon not in engine.STRATEGY_GRIDS:
        horizon = "short"
    if strategy_filter not in engine.STRATEGY_TYPES:
        strategy_filter = "auto"

    data = engine.cached_data(symbol, start, adjust).copy()
    scan = engine.scan_strategies(data, cash, fee, horizon, strategy_filter)
    best = scan.iloc[0]
    fast = int(best["fast"])
    slow = int(best["slow"])
    strategy_type = str(best.get("strategy_type", "sma"))
    fast_line, slow_line, entries, exits = engine.strategy_signals(data, fast, slow, horizon, strategy_type)
    portfolio = engine.strategy_portfolio(data, entries, exits, cash, fee, horizon)
    trades = portfolio.trades.records_readable
    equity = portfolio.value()
    if str(form.get("_save_all_strategies", "")).lower() in {"1", "true", "yes"}:
        gate = _save_all_scan_strategies_from_payload(form, symbol, data, scan, horizon, None)
        if gate is None:
            gate = _daily_gate_from_backtest_payload(form, symbol, data, best, fast_line, slow_line, entries, exits, horizon)
    else:
        gate = _daily_gate_from_backtest_payload(form, symbol, data, best, fast_line, slow_line, entries, exits, horizon)

    return {
        "symbol": symbol,
        "name": engine.stock_display_name(symbol),
        "data": data,
        "scan": scan,
        "best": best,
        "fast_line": fast_line,
        "slow_line": slow_line,
        "trades": trades,
        "equity": equity,
        "cash": cash,
        "fee": fee,
        "horizon": horizon,
        "strategy_type": strategy_type,
        "daily_gate": gate,
    }


def _strategy_row_from_portfolio(
    strategy_type: str,
    fast: int,
    slow: int,
    portfolio: Any,
    horizon: str,
) -> pd.Series:
    grid = engine.STRATEGY_GRIDS[horizon]
    sharpe = portfolio.sharpe_ratio()
    ret = float(portfolio.total_return()) * 100
    dd = float(portfolio.max_drawdown()) * 100
    trades = int(portfolio.trades.count())
    score = ret - abs(dd) * 0.8 + (float(sharpe) if np.isfinite(sharpe) else 0) * 35
    if horizon == "short":
        score = ret * 0.55 - abs(dd) * 3.2 + (float(sharpe) if np.isfinite(sharpe) else 0) * 35
        score -= max(0, abs(dd) - 16) * 8.0
        score += min(trades, 30) * 1.5
        if abs(dd) > 28:
            score -= 500
        if strategy_type == "ml":
            score -= 15
    if trades < grid["min_trades"]:
        score -= 200
    return pd.Series(
        {
            "strategy_type": strategy_type,
            "strategy_label": engine.STRATEGY_TYPES.get(strategy_type, strategy_type),
            "fast": fast,
            "slow": slow,
            "total_return_pct": ret,
            "max_drawdown_pct": dd,
            "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
            "trades": trades,
            "final_value": float(portfolio.final_value()),
            "score": score,
        }
    )


def _compute_saved_strategy_preview_payload(key_text: str) -> dict[str, Any]:
    cache = engine.load_persistent_strategy_cache()
    record = cache.get(key_text)
    if not isinstance(record, dict):
        raise ValueError("没有找到这条保存策略")
    try:
        key = json.loads(key_text)
    except Exception as exc:
        raise ValueError("保存策略键值损坏") from exc
    if not isinstance(key, list) or len(key) < 6:
        raise ValueError("保存策略参数不完整")

    symbol, start, adjust, cash_text, fee_text, mode = key[:6]
    parts = str(mode).split(":")
    horizon = parts[0] if len(parts) > 0 and parts[0] in engine.STRATEGY_GRIDS else "short"
    strategy_filter = parts[1] if len(parts) > 1 else "auto_fast"
    cash = float(cash_text or 100000)
    fee = float(fee_text or 0.0003)

    saved_result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
    signal = saved_result.get("daily_signal", {}) if isinstance(saved_result.get("daily_signal"), dict) else {}
    if not signal:
        raise ValueError("这条保存策略没有可用信号，请重新回测并保存一次")
    strategy_type = str(signal.get("strategy_type") or strategy_filter or "sma")
    fast = int(signal.get("fast") or 5)
    slow = int(signal.get("slow") or 20)

    data = engine.cached_data(str(symbol), str(start), str(adjust)).copy()
    fast_line, slow_line, entries, exits = engine.strategy_signals(data, fast, slow, horizon, strategy_type)
    portfolio = engine.strategy_portfolio(data, entries, exits, cash, fee, horizon)
    row = _strategy_row_from_portfolio(strategy_type, fast, slow, portfolio, horizon)
    scan = pd.DataFrame([row])
    scan.index = [0]
    return {
        "symbol": engine.normalize_symbol(str(symbol)),
        "name": str(record.get("name") or engine.stock_display_name(str(symbol))),
        "data": data,
        "scan": scan,
        "best": scan.iloc[0],
        "fast_line": fast_line,
        "slow_line": slow_line,
        "trades": portfolio.trades.records_readable,
        "equity": portfolio.value(),
        "cash": cash,
        "fee": fee,
        "horizon": horizon,
        "strategy_type": strategy_type,
        "daily_gate": saved_result,
        "cache_key_text": key_text,
        "from_saved_strategy": True,
    }


def _ml_prediction_row(result: dict[str, Any]) -> dict[str, Any]:
    prediction = result["prediction"]
    decision_row = result.get("decision_row") if isinstance(result.get("decision_row"), dict) else {}
    horizons = {int(item["days"]): item for item in prediction.get("horizons", [])}
    h3 = horizons.get(3, {})
    h5 = horizons.get(5, {})
    h10 = horizons.get(10, {})
    anomaly = prediction.get("anomaly", {}) if isinstance(prediction.get("anomaly"), dict) else {}
    factor = prediction.get("factor", {}) if isinstance(prediction.get("factor"), dict) else {}
    holding_risk = prediction.get("holding_risk", {}) if isinstance(prediction.get("holding_risk"), dict) else {}
    news = prediction.get("news_sentiment", {}) if isinstance(prediction.get("news_sentiment"), dict) else {}
    decision_meta = prediction.get("model_metadata", {}) if isinstance(prediction.get("model_metadata"), dict) else {}
    events = decision_row.get("event_status", {}) if isinstance(decision_row.get("event_status"), dict) else {}
    positive = decision_row.get("top_positive_factor_details", [])
    negative = decision_row.get("top_negative_factor_details", [])
    driver_names = [item.get("factor_name", "") for item in positive[:2] if isinstance(item, dict)]
    driver_names += [item.get("factor_name", "") for item in negative[:1] if isinstance(item, dict)]
    return {
        "symbol": result["symbol"],
        "name": result["name"],
        "view": prediction.get("view", "-"),
        "holding_risk": holding_risk.get("level", "-"),
        "risk_detail": holding_risk.get("detail", "-"),
        "target_weight": float(result.get("target_weight", np.nan)),
        "current_weight": float(result.get("current_weight", np.nan)),
        "target_shares": float(result.get("target_shares", np.nan)),
        "trade_shares": float(result.get("trade_shares", np.nan)),
        "rebalance_action": result.get("rebalance_action", "-"),
        "utility": float(decision_row.get("display_score", np.nan)),
        "prob3": float(h3.get("up_prob", np.nan)) * 100,
        "prob5": float(h5.get("up_prob", np.nan)) * 100,
        "prob10": float(h10.get("up_prob", np.nan)) * 100,
        "exp10": float(h10.get("expected_return_pct", np.nan)),
        "factor": float(factor.get("score", np.nan)),
        "risk": float(prediction.get("risk_score", np.nan)),
        "volatility": float(factor.get("atr_pct", np.nan)),
        "anomaly": anomaly.get("level", "-"),
        "news_risk": news.get("level", "-"),
        "fund_flow": float(decision_row.get("main_net_ratio_3", np.nan)) * 100,
        "news_factor": float(decision_row.get("news_sentiment_mean_3", np.nan)),
        "institution": float(decision_row.get("institution_activity", np.nan)),
        "weight_probability": float(decision_row.get("score_weight_probability", np.nan)),
        "weight_expected": float(decision_row.get("score_weight_expected", np.nan)),
        "weight_risk": float(decision_row.get("score_weight_risk", np.nan)),
        "score": float(prediction.get("composite_score", np.nan)),
        "confidence": float(decision_row.get("confidence_score", np.nan)) * 100,
        "confidence_level": str(decision_row.get("confidence_level", "-")),
        "completeness": float(decision_row.get("data_completeness_score", np.nan)) * 100,
        "model_version": str(decision_row.get("model_version") or prediction.get("model_version") or "-"),
        "event_status": "/".join(str(value) for value in events.values()) if events else "-",
        "main_driver": ", ".join(name for name in driver_names if name) or "-",
        "training_end": str(decision_row.get("training_end") or decision_meta.get("training_end") or "-"),
    }


def _parse_batch_symbols(text: str) -> list[str]:
    for sep in ("\n", "\r", "\t", ",", "，", ";", "；", "、", "|"):
        text = text.replace(sep, " ")
    symbols: list[str] = []
    seen: set[str] = set()
    for token in text.split():
        try:
            symbol = engine.resolve_stock_identifier(token)
        except Exception:
            symbol = token.strip()
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def _compute_ml_decision_payload_legacy(form: dict[str, Any]) -> dict[str, Any]:
    batch_symbols = _parse_batch_symbols(str(form.get("batch_symbols", "")))
    if not batch_symbols:
        batch_symbols = [engine.resolve_stock_identifier(str(form["symbol"]))]
    start = form.get("start", "20200101")
    adjust = form.get("adjust", "qfq")
    errors: list[str] = []
    results: list[dict[str, Any]] = []
    positions = form.get("positions", {})
    if not isinstance(positions, dict):
        positions = {}
    force_external_refresh = str(form.get("refresh_external_data") or "").lower() in {"1", "true", "yes", "on"}
    account = AccountState(
        cash=float(form.get("cash") or 100000),
        total_asset=float(form.get("cash") or 100000),
        max_total_position_weight=float(form.get("target_position") or 80) / 100.0,
        commission_rate=float(form.get("fee") or 0.0003),
    )
    min_daily_rows = 55
    for symbol in batch_symbols:
        try:
            data = engine.cached_data(symbol, str(start), str(adjust)).copy()
        except Exception as exc:
            errors.append(f"{symbol}: 日线数据失败，已跳过：{_short_error_text(exc)}")
            continue
        if data is None or data.empty or len(data) < min_daily_rows:
            actual_rows = 0 if data is None else len(data)
            errors.append(f"{symbol}: 日线有效数据不足，仅 {actual_rows} 行，至少需要约 {min_daily_rows} 行；开始日期可能太近，已跳过")
            continue
        frame = data.reset_index()
        frame = frame.rename(
            columns={frame.columns[0]: "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        )
        frame["code"] = symbol
        frame["amount"] = frame["volume"] * frame["close"]
        market_df = frame[["date", "code", "open", "high", "low", "close", "volume", "amount"]]
        try:
            external_df, source_notes = fetch_external_factor_frame(
                [symbol], force_refresh=force_external_refresh, market_df=market_df
            )
        except Exception as exc:
            external_df, source_notes = pd.DataFrame(), []
            errors.append(f"{symbol}: 外部因子拉取失败，已仅用日线继续：{_short_error_text(exc)}")
        market_df = merge_external_factors(market_df, external_df)

        position = positions.get(symbol, {}) if isinstance(positions.get(symbol, {}), dict) else {}
        latest_close = float(data["Close"].iloc[-1])
        shares = int(float(position.get("shares") or form.get("shares") or 0))
        cost = float(position.get("cost") or form.get("buy_price") or latest_close)
        holdings_df = pd.DataFrame(
            [
                {
                    "code": symbol,
                    "name": str(position.get("name") or ""),
                    "shares": shares,
                    "available_shares": shares,
                    "average_cost": cost,
                    "current_price": latest_close,
                    "position_value": shares * latest_close,
                    "holding_days": 0,
                }
            ]
        )
        config = DecisionConfig(start_date=str(start), train_min_rows=max(50, min(260, len(market_df) // 2)), external_factor_lag=0)
        try:
            decision = run_holding_decision(market_df, holdings_df, account=account, config=config, source_notes=source_notes)
            decision_payload = result_to_jsonable(decision)
        except Exception as exc:
            errors.append(f"{symbol}: ML 训练失败，已跳过：{_short_error_text(exc)}")
            continue
        table_by_code = {str(row["code"]): row for row in decision_payload["table"]}
        row = table_by_code.get(symbol)
        if row is None:
            errors.append(f"{symbol}: ML 未产生结果，已跳过")
            continue
        try:
            result = _ml_decision_result_from_row(symbol, row, data, decision_payload, str(position.get("name") or ""))
            results.append(result)
        except Exception as exc:
            errors.append(f"{symbol}: ML 结果整理失败，已跳过：{_short_error_text(exc)}")
    if not results:
        detail = "；".join(errors[:6])
        raise ValueError(f"没有可展示的 ML 结果；全部股票训练失败或样本不足。{detail}")
    if len(results) > 1:
        constrained = apply_account_constraints(
            pd.DataFrame([item["decision_row"] for item in results]), account
        )
        constrained_by_code = {
            str(row["code"]).zfill(6): row for row in constrained.to_dict("records")
        }
        for item in results:
            row = constrained_by_code.get(str(item["symbol"]).zfill(6))
            if row is None:
                continue
            item["decision_row"].update(row)
            item["prediction"]["decision_row"] = item["decision_row"]
            item["target_weight"] = float(row.get("recommended_target_weight") or 0) * 100
            item["target_shares"] = int(row.get("recommended_target_shares") or 0)
            item["trade_shares"] = int(row.get("recommended_trade_shares") or 0)
            item["rebalance_action"] = str(row.get("recommended_action", "-"))
            item["row"] = _ml_prediction_row(item)
    if len(results) == 1 and len(batch_symbols) == 1:
        return results[0]
    return {"results": results, "errors": errors, "total": len(batch_symbols), "cancelled": False, "decision_meta": {}}


def _compute_ml_decision_payload(form: dict[str, Any]) -> dict[str, Any]:
    """Run one production-model inference pass for the entire requested batch."""
    batch_symbols = _parse_batch_symbols(str(form.get("batch_symbols", "")))
    if not batch_symbols:
        batch_symbols = [engine.resolve_stock_identifier(str(form["symbol"]))]
    start = str(form.get("start") or "20200101")
    adjust = str(form.get("adjust") or "qfq")
    positions = form.get("positions", {})
    if not isinstance(positions, dict):
        positions = {}
    refresh_external = str(form.get("refresh_external_data") or "").lower() in {
        "1", "true", "yes", "on"
    }
    account = AccountState(
        cash=float(form.get("cash") or 100000),
        total_asset=float(form.get("cash") or 100000),
        max_total_position_weight=float(form.get("target_position") or 80) / 100.0,
        commission_rate=float(form.get("fee") or 0.0003),
    )
    errors: list[str] = []
    market_frames: list[pd.DataFrame] = []
    holdings_rows: list[dict[str, Any]] = []
    data_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in batch_symbols:
        try:
            data = engine.cached_data(symbol, start, adjust).copy()
        except Exception as exc:
            errors.append(f"{symbol}: daily data failed; skipped: {_short_error_text(exc)}")
            continue
        if data is None or data.empty or len(data) < 55:
            errors.append(f"{symbol}: fewer than 55 daily rows; skipped")
            continue
        frame = data.reset_index()
        frame = frame.rename(
            columns={
                frame.columns[0]: "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frame["code"] = symbol
        frame["amount"] = frame["volume"] * frame["close"]
        market = frame[["date", "code", "open", "high", "low", "close", "volume", "amount"]].copy()
        market["market_data_available"] = 1.0
        market_frames.append(market)
        data_by_symbol[symbol] = data
        position = positions.get(symbol, {}) if isinstance(positions.get(symbol), dict) else {}
        latest_close = float(data["Close"].iloc[-1])
        shares = int(float(position.get("shares") or form.get("shares") or 0))
        cost = float(position.get("cost") or form.get("buy_price") or latest_close)
        holdings_rows.append(
            {
                "code": symbol,
                "name": str(position.get("name") or ""),
                "shares": shares,
                "available_shares": shares,
                "average_cost": cost,
                "current_price": latest_close,
                "position_value": shares * latest_close,
                "holding_days": 0,
            }
        )

    if not market_frames:
        raise ValueError("No usable daily data. " + "; ".join(errors[:6]))
    combined_market = pd.concat(market_frames, ignore_index=True, sort=False)
    try:
        external, source_notes = fetch_external_factor_frame(
            list(data_by_symbol),
            force_refresh=refresh_external,
            market_df=combined_market,
        )
    except Exception as exc:
        external, source_notes = pd.DataFrame(), []
        errors.append(f"External factors unavailable; using market data only: {_short_error_text(exc)}")
    combined_market = merge_external_factors(combined_market, external)
    try:
        decision = run_production_holding_decision(
            combined_market,
            pd.DataFrame(holdings_rows),
            engine.ROOT,
            account=account,
            config=DecisionConfig(start_date=start, train_min_rows=50, external_factor_lag=0),
            source_notes=source_notes,
        )
    except ProductionModelNotFound as exc:
        raise ValueError(
            "No production ML model is registered. Run monthly training from Model Diagnostics "
            "and promote the candidate before daily evaluation."
        ) from exc
    payload = production_result_to_jsonable(decision)
    rows = {str(row["code"]).zfill(6): row for row in payload["table"]}
    results: list[dict[str, Any]] = []
    for symbol, data in data_by_symbol.items():
        row = rows.get(symbol)
        if row is None:
            errors.append(f"{symbol}: production model returned no row; skipped")
            continue
        position = positions.get(symbol, {}) if isinstance(positions.get(symbol), dict) else {}
        try:
            results.append(
                _ml_decision_result_from_row(
                    symbol, row, data, payload, str(position.get("name") or "")
                )
            )
        except Exception as exc:
            errors.append(f"{symbol}: result formatting failed; skipped: {_short_error_text(exc)}")
    if not results:
        raise ValueError("Production inference returned no displayable rows. " + "; ".join(errors[:6]))
    if len(results) == 1 and len(batch_symbols) == 1:
        return results[0]
    return {
        "results": results,
        "errors": errors,
        "total": len(batch_symbols),
        "cancelled": False,
        "decision_meta": {
            "model_version": payload.get("model_version"),
            "model_metadata": payload.get("model_metadata", {}),
            "snapshot_path": payload.get("snapshot_path"),
        },
    }


def _compute_ml_prediction_payload(form: dict[str, Any]) -> dict[str, Any]:
    return _compute_ml_decision_payload(form)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(max(0.0, min(100.0, value)))


def _ml_decision_risk_score(row: dict[str, Any]) -> float:
    down_prob = max(0.0, _safe_float(row.get("probability_down_2pct"), 0.0))
    q10 = _safe_float(row.get("return_q10"), 0.0)
    downside = max(0.0, _safe_float(row.get("downside_risk"), 0.0))
    unrealized = _safe_float(row.get("unrealized_return"), 0.0)
    raw = down_prob * 55.0 + max(-q10, 0.0) * 900.0 + downside * 900.0 + max(-unrealized, 0.0) * 20.0
    return _clamp_score(raw)


def _ml_decision_factor_score(row: dict[str, Any]) -> float:
    up_prob = _safe_float(row.get("probability_up"), 0.5)
    expected = _safe_float(row.get("expected_open_to_open_return"), 0.0)
    q50 = _safe_float(row.get("return_q50"), 0.0)
    down_prob = _safe_float(row.get("probability_down_2pct"), 0.0)
    raw = up_prob * 70.0 + expected * 800.0 + q50 * 400.0 - down_prob * 30.0
    return _clamp_score(raw)


def _ml_holding_risk_level(row: dict[str, Any], risk_score: float) -> str:
    try:
        shares = int(float(row.get("shares") or 0))
    except Exception:
        shares = 0
    if shares <= 0:
        return "未持仓"
    if risk_score >= 70:
        return "高风险"
    if risk_score >= 45:
        return "风险升高"
    if risk_score >= 25:
        return "中等"
    return "低"


def _ml_display_snapshot(data: pd.DataFrame) -> dict[str, Any]:
    try:
        atr_pct = float((engine.atr(data, 14) / data["Close"]).iloc[-1]) * 100
    except Exception:
        atr_pct = np.nan
    try:
        anomaly = engine.detect_latest_anomaly(data, 20, 60)
    except Exception as exc:
        anomaly = {"level": "unknown", "detail": f"异常检测暂不可用: {exc}"}
    if isinstance(anomaly, dict) and str(anomaly.get("level", "")) == "unknown":
        detail = str(anomaly.get("detail") or "")
        anomaly["level"] = "样本不足" if "样本不足" in detail else "未判断"
    try:
        mc = engine.monte_carlo_risk(data, stop_line=None, days=10, simulations=1200)
    except Exception as exc:
        mc = {"days": 10, "detail": f"蒙特卡洛暂不可用: {exc}"}
    return {
        "factor": {"atr_pct": atr_pct},
        "anomaly": anomaly,
        "monte_carlo": mc,
    }


def _ml_decision_horizon_legacy(row: dict[str, Any], days: int) -> dict[str, Any]:
    up_prob_1 = _safe_float(row.get("probability_up"), np.nan)
    expected_1 = _safe_float(row.get("expected_open_to_open_return"), np.nan)
    q10_1 = _safe_float(row.get("return_q10"), np.nan)
    if not np.isfinite(up_prob_1):
        up_prob = np.nan
    else:
        shrink = 1.0 / np.sqrt(max(days, 1))
        up_prob = 0.5 + (up_prob_1 - 0.5) * shrink
    expected_pct = expected_1 * days * 100 if np.isfinite(expected_1) else np.nan
    q10_pct = q10_1 * np.sqrt(max(days, 1)) * 100 if np.isfinite(q10_1) else np.nan
    expected_text = f"{expected_pct:.2f}%" if np.isfinite(expected_pct) else "-"
    q10_text = f"{q10_pct:.2f}%" if np.isfinite(q10_pct) else "-"
    return {
        "days": days,
        "up_prob": up_prob,
        "expected_return_pct": expected_pct,
        "q10_return_pct": q10_pct,
        "detail": f"{days}日为基于次日模型的保守折算；预期 {expected_text}，q10 {q10_text}。",
    }


def _ml_decision_horizon(row: dict[str, Any], days: int) -> dict[str, Any]:
    """Expose the next-session model honestly; longer horizons are projections."""
    up_prob_1 = _safe_float(row.get("probability_up"), np.nan)
    expected_1 = _safe_float(row.get("expected_open_to_open_return"), np.nan)
    q10_1 = _safe_float(row.get("return_q10"), np.nan)
    shrink = 1.0 / np.sqrt(max(days, 1))
    up_prob = 0.5 + (up_prob_1 - 0.5) * shrink if np.isfinite(up_prob_1) else np.nan
    expected_pct = expected_1 * days * 100 if np.isfinite(expected_1) else np.nan
    q10_pct = q10_1 * np.sqrt(max(days, 1)) * 100 if np.isfinite(q10_1) else np.nan
    if days == 1:
        detail = "次日独立模型预测。"
    else:
        expected_text = f"{expected_pct:.2f}%" if np.isfinite(expected_pct) else "-"
        q10_text = f"{q10_pct:.2f}%" if np.isfinite(q10_pct) else "-"
        detail = f"{days}日风险投影：由次日模型保守折算，不是独立训练结果；预期 {expected_text}，q10 {q10_text}。"
    return {
        "days": days,
        "up_prob": up_prob,
        "expected_return_pct": expected_pct,
        "q10_return_pct": q10_pct,
        "detail": detail,
        "is_independent_model": days == 1,
    }


def _ml_decision_result_from_row(
    symbol: str,
    row: dict[str, Any],
    data: pd.DataFrame,
    decision_payload: dict[str, Any],
    display_name: str = "",
) -> dict[str, Any]:
    risk_score = _ml_decision_risk_score(row)
    factor_score = _ml_decision_factor_score(row)
    display = _ml_display_snapshot(data)
    factor_display = display.get("factor", {}) if isinstance(display.get("factor"), dict) else {}
    factor_display["score"] = factor_score
    prediction = {
        "date": row.get("date"),
        "latest_close": row.get("current_price"),
        "backend": str(decision_payload.get("metrics", {}).get("model_backend", "XGBoost CUDA")),
        "factor": factor_display,
        "risk_score": risk_score,
        "holding_risk": {"level": row.get("holding_risk_level") or _ml_holding_risk_level(row, risk_score), "detail": row.get("reason", "")},
        "anomaly": display.get("anomaly", {"level": "-"}),
        "news_sentiment": {"level": "-"},
        "monte_carlo": display.get("monte_carlo", {}),
        "horizons": [_ml_decision_horizon(row, days) for days in (1, 3, 10)],
        "decision_row": row,
        "metrics": decision_payload.get("metrics", {}),
        "feature_columns": decision_payload.get("feature_columns", []),
        "source_notes": decision_payload.get("source_notes", []),
        "model_version": decision_payload.get("model_version") or row.get("model_version"),
        "model_metadata": decision_payload.get("model_metadata", {}),
        "snapshot_path": decision_payload.get("snapshot_path"),
        "data_completeness_score": row.get("data_completeness_score"),
        "confidence_score": row.get("confidence_score"),
        "confidence_level": row.get("confidence_level"),
        "event_status": row.get("event_status", {}),
        "group_weights": row.get("group_weights", {}),
        "group_predictions": row.get("group_predictions", {}),
        "top_positive_factor_details": row.get("top_positive_factor_details", []),
        "top_negative_factor_details": row.get("top_negative_factor_details", []),
    }
    result = {
        "symbol": symbol,
        "name": display_name or str(row.get("name") or "") or engine.stock_display_name(symbol),
        "data": data,
        "prediction": prediction,
        "start": "",
        "adjust": "",
        "target_weight": float(row.get("recommended_target_weight") or 0) * 100,
        "target_shares": int(row.get("shares", 0) + row.get("recommended_trade_shares", 0)),
        "trade_shares": int(row.get("recommended_trade_shares", 0)),
        "current_shares": int(row.get("shares", 0)),
        "current_weight": float(row.get("position_weight") or 0) * 100,
        "rebalance_action": str(row.get("recommended_action", "-")),
        "cash_reserve": np.nan,
        "decision_row": row,
    }
    result["row"] = _ml_prediction_row(result)
    return result


def _backtest_process_entry(form: dict[str, str], result_queue: Any) -> None:
    try:
        if form.get("_job") == "ml_predict":
            result_queue.put(("ok", _compute_ml_prediction_payload(form)))
        else:
            result_queue.put(("ok", _compute_backtest_payload(form)))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))


class StrategyDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("A股策略桌面版")
        self.geometry("1480x920")
        self.minsize(720, 480)
        self.tk.call("tk", "scaling", 1.12)
        self._configure_style()

        self.queue: queue.Queue[WorkerMessage] = queue.Queue()
        self.monitor_running = False
        self.monitor_worker: threading.Thread | None = None
        self.backtest_worker: threading.Thread | None = None
        self.cache_preview_worker: threading.Thread | None = None
        self.cache_preview_key: str | None = None
        self.pending_saved_description_key: str | None = None
        self.backtest_process: Any | None = None
        self.backtest_processes: set[Any] = set()
        self.backtest_process_lock = threading.Lock()
        self.backtest_stop_event = threading.Event()
        self.backtest_target = "traditional"
        self.pending_backtest_form: dict[str, str] | None = None
        self.monitor_items: dict[str, dict[str, Any]] = {}
        self.selected_monitor_symbol: str | None = None
        self.monitor_strategy_keys: dict[str, str] = {}
        self.monitor_enabled_symbols: set[str] = set()
        self.monitor_refresh_pending = False
        self.monitor_pending_symbols: set[str] = set()
        self.monitor_last_symbol_refresh: dict[str, float] = {}
        self.monitor_select_job: str | None = None
        self.ml_monitor_running = False
        self.ml_monitor_worker: threading.Thread | None = None
        self.ml_monitor_items: dict[str, dict[str, Any]] = {}
        self.selected_ml_monitor_symbol: str | None = None
        self.ml_monitor_strategy_keys: dict[str, str] = {}
        self.backtest_result: dict[str, Any] | None = None
        self.ml_backtest_result: dict[str, Any] | None = None
        self.ml_prediction_results: dict[str, dict[str, Any]] = {}
        self.ml_sort_column: str | None = None
        self.ml_sort_reverse = False
        self.selected_ml_prediction_symbol: str | None = None
        self.selected_scan_rank: int = 0
        self.backtest_checked_symbols: set[str] = set()
        self.ml_checked_symbols: set[str] = set()
        self.backtest_chart_payload: dict[str, Any] | None = None
        self.backtest_zoom: tuple[int, int] | None = None
        self.backtest_drag_start_x: int | None = None
        self.backtest_drag_rect: int | None = None
        self.backtest_pan_start_x: int | None = None
        self.backtest_pan_start_zoom: tuple[int, int] | None = None
        self.backtest_resize_job: str | None = None
        self.monitor_resize_job: str | None = None
        self.ml_resize_job: str | None = None
        self.ml_monitor_resize_job: str | None = None
        self.backtest_fullscreen_window: tk.Toplevel | None = None
        self.backtest_fullscreen_canvas: tk.Canvas | None = None
        self.backtest_fullscreen_zoom: tuple[int, int] | None = None
        self.backtest_fullscreen_drag_start_x: int | None = None
        self.backtest_fullscreen_drag_rect: int | None = None
        self.backtest_fullscreen_pan_start_x: int | None = None
        self.backtest_fullscreen_pan_start_zoom: tuple[int, int] | None = None
        self.backtest_fullscreen_resize_job: str | None = None
        self.content_window_id: int | None = None
        self.content_resize_job: str | None = None
        self.content_last_layout_size: tuple[int, int] = (0, 0)
        self.content_pending_size: tuple[int, int] | None = None
        self.model_training_process: subprocess.Popen | None = None
        self.model_training_log_path = engine.CACHE_DIR / "monthly_training.log"
        self.model_training_poll_job: str | None = None

        self._build_ui()
        self._load_monitor_snapshot()
        self._render_saved_stock_picker()
        self._load_ml_prediction_snapshot_banner()
        self.protocol("WM_DELETE_WINDOW", self._on_app_close)
        self.after(400, self._poll_queue)
        self.after(1500, self._auto_monitor_tick)

    def _build_ui(self) -> None:
        self.configure(background="#eef3f8")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="Header.TFrame", padding=(18, 14, 18, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="A股策略桌面版", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="传统策略负责买卖点，ML负责持仓风险、异常检测和组合仓位。数据和信号仅供研究，实盘请人工复核。",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.content_canvas = tk.Canvas(self, background="#eef3f8", highlightthickness=0, bd=0)
        self.content_canvas.grid(row=1, column=0, sticky="nsew", padx=14, pady=(10, 0))
        self.content_xscroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.content_canvas.xview)
        self.content_xscroll.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.content_yscroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.content_canvas.yview)
        self.content_yscroll.grid(row=1, column=1, sticky="ns", pady=(10, 0))
        self.content_canvas.configure(xscrollcommand=self.content_xscroll.set, yscrollcommand=self.content_yscroll.set)

        self.notebook = ttk.Notebook(self.content_canvas)
        self.content_window_id = self.content_canvas.create_window(0, 0, window=self.notebook, anchor="nw")
        self.content_canvas.bind("<Configure>", self._on_content_canvas_configure)
        self.notebook.bind("<Configure>", self._on_notebook_configure)

        self.monitor_tab = ttk.Frame(self.notebook, padding=12)
        self.backtest_tab = ttk.Frame(self.notebook, padding=12)
        self.ml_tab = ttk.Frame(self.notebook, padding=12)
        self.model_diagnostics_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.backtest_tab, text="传统回测")
        self.notebook.add(self.monitor_tab, text="盘中监控")
        self.notebook.add(self.ml_tab, text="ML持仓决策")

        self._build_backtest_tab()
        self._build_monitor_tab()
        self._build_ml_tab()
        self.notebook.add(self.model_diagnostics_tab, text="模型诊断")
        self._build_model_diagnostics_tab()

        bottom = ttk.Frame(self, style="Status.TFrame", padding=(16, 8, 16, 10))
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(bottom, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")

    def _configure_style(self) -> None:
        palette = {
            "bg": "#eef3f8",
            "panel": "#ffffff",
            "panel_alt": "#f8fafc",
            "line": "#d8e2ee",
            "ink": "#14213d",
            "muted": "#607086",
            "accent": "#1d5fd1",
            "accent_dark": "#174aa3",
            "select": "#dbeafe",
        }
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.option_add("*TCombobox*Listbox.font", ("Microsoft YaHei UI", 10))
        if tb is not None:
            try:
                style = tb.Style(theme="flatly")
            except Exception:
                style = ttk.Style(self)
        else:
            style = ttk.Style(self)
        try:
            if tb is None:
                style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 10), background=palette["bg"], foreground=palette["ink"])
        style.configure("TFrame", background=palette["bg"])
        style.configure("Header.TFrame", background="#111827")
        style.configure("Status.TFrame", background=palette["panel"])
        style.configure("Title.TLabel", background="#111827", foreground="#ffffff", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background="#111827", foreground="#cbd5e1", font=("Microsoft YaHei UI", 10))
        style.configure("Status.TLabel", background=palette["panel"], foreground=palette["muted"])
        style.configure("TLabel", background=palette["bg"], foreground=palette["ink"])
        style.configure("TLabelframe", background=palette["bg"], bordercolor=palette["line"], relief="solid")
        style.configure("TLabelframe.Label", background=palette["bg"], foreground=palette["ink"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TButton", padding=(12, 7), borderwidth=0, background="#e8eef7", foreground=palette["ink"])
        style.map("TButton", background=[("active", "#dbe7f6"), ("pressed", "#c7d7ee")])
        style.configure("Primary.TButton", padding=(14, 8), background=palette["accent"], foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", palette["accent_dark"]), ("pressed", palette["accent_dark"])])
        style.configure("TEntry", fieldbackground="#ffffff", bordercolor="#cbd5e1", lightcolor="#cbd5e1", darkcolor="#cbd5e1", padding=(6, 5))
        style.configure("TCombobox", fieldbackground="#ffffff", bordercolor="#cbd5e1", padding=(6, 5), arrowsize=14)
        style.configure("TNotebook", background=palette["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), background="#e2e8f0", foreground=palette["muted"], font=("Microsoft YaHei UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", palette["panel"])], foreground=[("selected", palette["accent"])])
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground=palette["ink"], rowheight=30, borderwidth=0)
        style.configure("Treeview.Heading", background="#f1f5f9", foreground=palette["ink"], relief="flat", font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Treeview", background=[("selected", palette["accent"])], foreground=[("selected", "#ffffff")])

    def _on_content_canvas_configure(self, event: tk.Event) -> None:
        if self.content_window_id is None:
            return
        width = max(CONTENT_MIN_WIDTH, int(event.width))
        height = max(CONTENT_MIN_HEIGHT, int(event.height))
        self.content_pending_size = (width, height)
        if self.content_last_layout_size == (0, 0):
            self._apply_content_layout(width, height)
        if self.content_resize_job is not None:
            try:
                self.after_cancel(self.content_resize_job)
            except Exception:
                pass
        self.content_resize_job = self.after(CONTENT_RESIZE_FINAL_DELAY_MS, self._apply_pending_content_layout)
        if int(event.width) < CONTENT_MIN_WIDTH:
            self.content_xscroll.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        else:
            self.content_xscroll.grid_remove()
            self.content_canvas.xview_moveto(0)
        if int(event.height) < CONTENT_MIN_HEIGHT:
            self.content_yscroll.grid(row=1, column=1, sticky="ns", pady=(10, 0))
        else:
            self.content_yscroll.grid_remove()
            self.content_canvas.yview_moveto(0)

    def _apply_pending_content_layout(self) -> None:
        self.content_resize_job = None
        if self.content_pending_size is None:
            return
        width, height = self.content_pending_size
        self._apply_content_layout(width, height)

    def _apply_content_layout(self, width: int, height: int) -> None:
        if self.content_window_id is None:
            return
        width = max(CONTENT_MIN_WIDTH, int(width))
        height = max(CONTENT_MIN_HEIGHT, int(height))
        if self.content_last_layout_size == (width, height):
            return
        self.content_last_layout_size = (width, height)
        self.content_canvas.itemconfigure(self.content_window_id, width=width, height=height)
        self.content_canvas.configure(scrollregion=(0, 0, width, height))

    def _on_notebook_configure(self, _event: tk.Event | None = None) -> None:
        if self.content_window_id is None:
            return
        canvas_width = max(1, self.content_canvas.winfo_width())
        canvas_height = max(1, self.content_canvas.winfo_height())
        width = max(CONTENT_MIN_WIDTH, canvas_width)
        height = max(CONTENT_MIN_HEIGHT, canvas_height)
        if self.content_last_layout_size == (0, 0):
            self._apply_content_layout(width, height)

    def _build_monitor_tab(self) -> None:
        self.monitor_tab.columnconfigure(0, weight=1)
        self.monitor_tab.rowconfigure(1, weight=1)

        top = ttk.Frame(self.monitor_tab)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(4, weight=1)

        self.monitor_period = tk.StringVar(value="5")
        self.monitor_interval = tk.StringVar(value="30")
        self.monitor_position_shares = tk.StringVar(value="")
        self.monitor_position_cost = tk.StringVar(value="")
        self._labeled_combo(top, "周期", self.monitor_period, ("1", "5", "15"), 0)
        self._labeled_entry(top, "刷新秒", self.monitor_interval, 1, width=8)
        ttk.Button(top, text="刷新一次", command=self.refresh_monitor_once).grid(row=1, column=2, padx=(0, 8))
        ttk.Label(top, text="点击左侧股票立即显示盘中监测曲线；后台只刷新当前股票。持股/成本在下方填写。", foreground="#607086").grid(row=1, column=3, columnspan=2, sticky="w")

        body = ttk.PanedWindow(self.monitor_tab, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")
        left_frame = ttk.Frame(body)
        right_frame = ttk.Frame(body)
        body.add(left_frame, weight=2)
        body.add(right_frame, weight=5)

        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        saved_box = ttk.LabelFrame(left_frame, text="已保存策略股票")
        saved_box.grid(row=0, column=0, sticky="nsew")
        saved_box.columnconfigure(0, weight=1)
        saved_box.rowconfigure(0, weight=1)
        self.saved_stock_tree = ttk.Treeview(
            saved_box,
            columns=("symbol", "name", "action", "price", "shares", "cost", "strategy"),
            show="headings",
            height=16,
            selectmode="browse",
        )
        self._setup_tree(
            self.saved_stock_tree,
            {
                "symbol": "代码",
                "name": "名称",
                "action": "信号",
                "price": "价格",
                "shares": "持股",
                "cost": "成本",
                "strategy": "策略",
            },
            {
                "symbol": 86,
                "name": 105,
                "action": 108,
                "price": 70,
                "shares": 70,
                "cost": 70,
                "strategy": 145,
            },
        )
        self.saved_stock_tree.grid(row=0, column=0, sticky="nsew")
        saved_scroll = ttk.Scrollbar(saved_box, orient=tk.VERTICAL, command=self.saved_stock_tree.yview)
        saved_scroll.grid(row=0, column=1, sticky="ns")
        self.saved_stock_tree.configure(yscrollcommand=saved_scroll.set)
        self.saved_stock_tree.bind("<Button-1>", self._on_saved_stock_click)
        self.saved_stock_tree.bind("<<TreeviewSelect>>", self._on_monitor_saved_stock_select)
        self.saved_stock_tree.bind("<Double-1>", self._on_saved_stock_double_click)
        self.saved_stock_tree.bind("<Button-3>", self._show_saved_stock_context_menu)
        self.saved_stock_tree.bind("<space>", lambda _event: self.refresh_monitor_once())

        strategy_box = ttk.LabelFrame(left_frame, text="该股票使用的保存策略")
        strategy_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        strategy_box.columnconfigure(0, weight=1)
        monitor_strategy_columns = ("strategy", "mode", "saved")
        self.monitor_strategy_tree = ttk.Treeview(strategy_box, columns=monitor_strategy_columns, show="headings", height=4)
        self._setup_tree(
            self.monitor_strategy_tree,
            {"strategy": "策略", "mode": "模式", "saved": "保存时间"},
            {"strategy": 160, "mode": 130, "saved": 135},
        )
        self.monitor_strategy_tree.grid(row=0, column=0, sticky="ew")
        self.monitor_strategy_tree.bind("<<TreeviewSelect>>", self._on_monitor_strategy_select)

        position_row = ttk.Frame(strategy_box)
        position_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        position_row.columnconfigure(1, weight=1)
        position_row.columnconfigure(3, weight=1)
        ttk.Label(position_row, text="持股数").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.monitor_position_shares_entry = ttk.Entry(position_row, textvariable=self.monitor_position_shares, width=12)
        self.monitor_position_shares_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        ttk.Label(position_row, text="成本价").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.monitor_position_cost_entry = ttk.Entry(position_row, textvariable=self.monitor_position_cost, width=12)
        self.monitor_position_cost_entry.grid(row=0, column=3, sticky="ew", padx=(0, 10))
        ttk.Button(position_row, text="保存持股/成本", command=self._save_selected_monitor_position, style="Primary.TButton").grid(row=0, column=4, sticky="e")

        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(2, weight=1)
        xueqiu_box = ttk.LabelFrame(right_frame, text="雪球K线")
        xueqiu_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        xueqiu_box.columnconfigure(0, weight=1)
        self.monitor_xueqiu_var = tk.StringVar(value="选中左侧股票后可打开雪球K线")
        ttk.Label(xueqiu_box, textvariable=self.monitor_xueqiu_var, foreground="#405269").grid(row=0, column=0, sticky="w", padx=10, pady=10)
        ttk.Button(xueqiu_box, text="打开雪球K线", command=self._open_monitor_xueqiu).grid(row=0, column=1, sticky="e", padx=10, pady=10)

        detail_box = ttk.LabelFrame(right_frame, text="盘中监测信息")
        detail_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        detail_box.columnconfigure(0, weight=1)
        self.monitor_detail_text = tk.Text(detail_box, height=7, wrap="word", background="#ffffff", foreground="#14213d", relief=tk.FLAT, padx=12, pady=10)
        self.monitor_detail_text.grid(row=0, column=0, sticky="ew")
        self.monitor_detail_text.configure(state=tk.DISABLED)

        chart_box = ttk.LabelFrame(right_frame, text="盘中监测曲线")
        chart_box.grid(row=2, column=0, sticky="nsew")
        chart_box.columnconfigure(0, weight=1)
        chart_box.rowconfigure(0, weight=1)
        self.monitor_canvas = tk.Canvas(chart_box, height=430, background="#fbfdff", highlightthickness=1, highlightbackground="#d8e0ea")
        self.monitor_canvas.grid(row=0, column=0, sticky="nsew")
        self.monitor_canvas.bind("<Configure>", self._schedule_monitor_canvas_redraw)
        self._render_saved_stock_picker()
        self._render_monitor_detail(None)

    def _build_backtest_tab(self) -> None:
        self.backtest_tab.columnconfigure(0, weight=1)
        self.backtest_tab.rowconfigure(2, weight=1)

        top = ttk.Frame(self.backtest_tab)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for idx in range(12):
            top.columnconfigure(idx, weight=0)
        top.columnconfigure(12, weight=1)

        self.bt_symbol = tk.StringVar(value="002472")
        self.bt_start = tk.StringVar(value="20200101")
        self.bt_adjust = tk.StringVar(value="qfq")
        self.bt_cash = tk.StringVar(value="100000")
        self.bt_fee = tk.StringVar(value="0.0003")
        self.bt_risk = tk.StringVar(value="normal")
        self.bt_horizon = tk.StringVar(value="short")
        self.bt_strategy = tk.StringVar(value="auto_fast")
        self.bt_shares = tk.StringVar(value="0")
        self.bt_buy_price = tk.StringVar(value="")
        self.bt_buy_date = tk.StringVar(value="")

        self._labeled_entry(top, "股票", self.bt_symbol, 0, width=14)
        self._labeled_entry(top, "开始日期", self.bt_start, 1, width=10)
        self._labeled_combo(top, "复权", self.bt_adjust, ("qfq", "", "hfq"), 2)
        self._labeled_entry(top, "资金", self.bt_cash, 3, width=10)
        self._labeled_entry(top, "手续费", self.bt_fee, 4, width=10)
        self._labeled_combo(top, "风险", self.bt_risk, ("normal", "tight", "loose"), 5)
        self._labeled_combo(top, "周期", self.bt_horizon, ("short", "swing", "trend"), 6)
        self._labeled_combo(top, "策略", self.bt_strategy, ("auto_fast", "hybrid", "rsi_capital", "macd_kdj", "boll_wr", "breakout_capital", "macd", "breakout", "rsi", "sma"), 7)
        self._labeled_entry(top, "持股数", self.bt_shares, 8, width=8)
        self._labeled_entry(top, "成本价", self.bt_buy_price, 9, width=8)
        self._labeled_entry(top, "买入日", self.bt_buy_date, 10, width=10)
        ttk.Button(top, text="开始回测", command=self.run_backtest, style="Primary.TButton").grid(row=1, column=11, padx=(8, 0))
        self.stop_backtest_button = ttk.Button(top, text="终止回测", command=self.stop_backtest, state=tk.DISABLED)
        self.stop_backtest_button.grid(row=1, column=12, padx=(8, 0))

        self.summary_var = tk.StringVar(value="尚未回测")
        ttk.Label(self.backtest_tab, textvariable=self.summary_var, anchor="w").grid(row=1, column=0, sticky="ew", pady=(0, 8))

        body = ttk.PanedWindow(self.backtest_tab, orient=tk.HORIZONTAL)
        body.grid(row=2, column=0, sticky="nsew")
        cache_frame = ttk.Frame(body)
        right_frame = ttk.Frame(body)
        body.add(cache_frame, weight=1)
        body.add(right_frame, weight=5)

        cache_frame.columnconfigure(0, weight=1)
        cache_frame.rowconfigure(2, weight=1)
        batch_box = ttk.LabelFrame(cache_frame, text="批量股票代码")
        batch_box.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        batch_box.columnconfigure(0, weight=1)
        self.bt_batch_text = tk.Text(batch_box, height=4, wrap="word", background="#ffffff", foreground="#14213d", relief=tk.FLAT, padx=8, pady=6)
        self.bt_batch_text.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        batch_buttons = ttk.Frame(batch_box)
        batch_buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        batch_buttons.columnconfigure(0, weight=1)
        batch_buttons.columnconfigure(1, weight=1)
        ttk.Button(batch_buttons, text="批量回测输入代码", command=self.run_input_stock_backtests, style="Primary.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(batch_buttons, text="清空代码", command=lambda: self.bt_batch_text.delete("1.0", tk.END)).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(cache_frame, text="已保存策略").grid(row=1, column=0, sticky="w", pady=(0, 4))
        cache_columns = ("symbol", "name", "mode", "strategy", "date")
        self.cache_tree = ttk.Treeview(cache_frame, columns=cache_columns, show="tree headings", height=18, selectmode="extended")
        cache_headings = {"symbol": "代码", "name": "名称", "mode": "模式", "strategy": "策略", "date": "日期"}
        cache_widths = {"symbol": 70, "name": 95, "mode": 120, "strategy": 115, "date": 90}
        self._setup_tree(self.cache_tree, cache_headings, cache_widths)
        self.cache_tree.heading("#0", text="股票/策略")
        self.cache_tree.column("#0", width=165, minwidth=120, anchor="w", stretch=True)
        self.cache_tree.grid(row=2, column=0, sticky="nsew")
        cache_vscroll = ttk.Scrollbar(cache_frame, orient=tk.VERTICAL, command=self.cache_tree.yview)
        cache_vscroll.grid(row=2, column=1, sticky="ns")
        self.cache_tree.configure(yscrollcommand=cache_vscroll.set)
        self.cache_tree.bind("<Button-1>", self._on_cache_tree_click)
        self.cache_tree.bind("<<TreeviewSelect>>", self._on_cache_select)
        self.cache_tree.bind("<Double-1>", lambda _event: "break")
        self.cache_tree.bind("<Button-3>", self._show_cache_context_menu)
        cache_buttons = ttk.Frame(cache_frame)
        cache_buttons.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        cache_buttons.columnconfigure(0, weight=1)
        cache_buttons.columnconfigure(1, weight=1)
        cache_buttons.columnconfigure(2, weight=1)
        cache_buttons.columnconfigure(3, weight=1)
        cache_buttons.columnconfigure(4, weight=1)
        ttk.Button(cache_buttons, text="载入回测", command=self.run_backtest).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(cache_buttons, text="删除策略", command=self._delete_selected_cache).grid(row=0, column=1, sticky="ew", padx=(0, 4))
        ttk.Button(cache_buttons, text="刷新", command=self._render_strategy_cache_list).grid(row=0, column=2, sticky="ew", padx=(0, 4))
        ttk.Button(cache_buttons, text="回测勾选股票", command=self.run_checked_saved_stock_backtests).grid(row=0, column=3, sticky="ew", padx=(0, 4))
        ttk.Button(cache_buttons, text="回测全部股票", command=self.run_all_saved_stock_backtests).grid(row=0, column=4, sticky="ew")
        self.cache_stock_context_menu = tk.Menu(self, tearoff=0)
        self.cache_stock_context_menu.add_command(label="勾选选中股票", command=lambda: self._set_selected_cache_stock_checks(True))
        self.cache_stock_context_menu.add_command(label="取消勾选选中股票", command=lambda: self._set_selected_cache_stock_checks(False))
        self.cache_stock_context_menu.add_command(label="回测这只股票", command=self.run_selected_cache_stock_backtest)
        self.cache_stock_context_menu.add_separator()
        self.cache_stock_context_menu.add_command(label="全部勾选", command=self._check_all_cache_stocks)
        self.cache_stock_context_menu.add_command(label="全部取消勾选", command=self._uncheck_all_cache_stocks)
        self.cache_stock_context_menu.add_separator()
        self.cache_stock_context_menu.add_command(label="删除这只股票的左侧策略", command=self._delete_selected_cache)
        self.cache_strategy_context_menu = tk.Menu(self, tearoff=0)
        self.cache_strategy_context_menu.add_command(label="查看这条策略曲线", command=self._load_left_cache_strategy_preview)
        self.cache_strategy_context_menu.add_command(label="删除这条策略", command=self._delete_selected_cache)
        self.cache_bulk_context_menu = tk.Menu(self, tearoff=0)
        self.cache_bulk_context_menu.add_command(label="全部勾选", command=self._check_all_cache_stocks)
        self.cache_bulk_context_menu.add_command(label="全部取消勾选", command=self._uncheck_all_cache_stocks)

        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=4)
        right_frame.rowconfigure(1, weight=2)
        chart_frame = ttk.Frame(right_frame)
        table_frame = ttk.Frame(right_frame)
        chart_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        table_frame.grid(row=1, column=0, sticky="nsew")

        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(1, weight=1)
        chart_tools = ttk.Frame(chart_frame)
        chart_tools.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(chart_tools, text="放大", command=lambda: self._zoom_backtest(0.72)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(chart_tools, text="缩小", command=lambda: self._zoom_backtest(1.35)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(chart_tools, text="重置缩放", command=self._reset_backtest_zoom).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(chart_tools, text="全屏图", command=self._open_backtest_fullscreen).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(chart_tools, text="左键框选放大/单击看价格，右键拖动平移；右键排名行看说明", foreground="#607086").pack(side=tk.LEFT)
        self.backtest_canvas = tk.Canvas(chart_frame, height=360, background="#fbfdff", highlightthickness=1, highlightbackground="#d8e0ea")
        self.backtest_canvas.grid(row=1, column=0, sticky="nsew")
        self.backtest_canvas.bind("<Configure>", self._schedule_backtest_canvas_redraw)
        self.backtest_canvas.bind("<MouseWheel>", self._on_backtest_mousewheel)
        self.backtest_canvas.bind("<ButtonPress-1>", self._on_backtest_drag_start)
        self.backtest_canvas.bind("<B1-Motion>", self._on_backtest_drag_move)
        self.backtest_canvas.bind("<ButtonRelease-1>", self._on_backtest_drag_release)
        self.backtest_canvas.bind("<ButtonPress-3>", self._on_backtest_pan_start)
        self.backtest_canvas.bind("<B3-Motion>", self._on_backtest_pan_move)
        self.backtest_canvas.bind("<ButtonRelease-3>", self._on_backtest_pan_release)

        columns = ("rank", "strategy", "params", "return", "drawdown", "sharpe", "trades", "final", "score")
        self.bt_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=9)
        headings = {
            "rank": "排名",
            "strategy": "策略",
            "params": "参数",
            "return": "收益%",
            "drawdown": "回撤%",
            "sharpe": "夏普",
            "trades": "交易",
            "final": "最终权益",
            "score": "评分",
        }
        widths = {
            "rank": 60,
            "strategy": 140,
            "params": 90,
            "return": 90,
            "drawdown": 90,
            "sharpe": 80,
            "trades": 80,
            "final": 120,
            "score": 90,
        }
        self._setup_tree(self.bt_tree, headings, widths)
        self.bt_tree.grid(row=0, column=0, sticky="nsew")
        self.bt_tree.bind("<<TreeviewSelect>>", self._on_backtest_rank_select)
        self.bt_tree.bind("<Button-3>", self._show_backtest_context_menu)
        self.bt_tree.bind("<Double-1>", lambda _event: self._show_strategy_description_popup())
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        bt_vscroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.bt_tree.yview)
        bt_vscroll.grid(row=0, column=1, sticky="ns")
        self.bt_tree.configure(yscrollcommand=bt_vscroll.set)
        self.bt_context_menu = tk.Menu(self, tearoff=0)
        self.bt_context_menu.add_command(label="查看策略说明", command=self._show_strategy_description_popup)
        self.bt_context_menu.add_command(label="保存选中策略", command=self._save_selected_rank_strategy)
        self.bt_context_menu.add_separator()
        self.bt_context_menu.add_command(label="全屏查看曲线", command=self._open_backtest_fullscreen)
        self.saved_bt_context_menu = tk.Menu(self, tearoff=0)
        self.saved_bt_context_menu.add_command(label="查看策略说明", command=self._show_saved_strategy_description_popup)
        self.saved_bt_context_menu.add_command(label="加入左侧并设为使用策略", command=self._add_right_saved_strategy_to_left)
        self._render_strategy_cache_list()

    def _build_model_diagnostics_tab(self) -> None:
        tab = self.model_diagnostics_tab
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)

        current = ttk.LabelFrame(tab, text="当前正式模型", padding=12)
        current.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        current.columnconfigure(1, weight=1)
        current.columnconfigure(3, weight=1)
        self.model_diag_vars = {
            "version": tk.StringVar(value="未注册"),
            "range": tk.StringVar(value="-"),
            "target": tk.StringVar(value="-"),
            "backend": tk.StringVar(value="-"),
            "features": tk.StringVar(value="-"),
            "calibration": tk.StringVar(value="-"),
            "updated": tk.StringVar(value="-"),
        }
        fields = [
            ("模型版本", "version"), ("训练范围", "range"),
            ("标签定义", "target"), ("模型后端", "backend"),
            ("因子数量", "features"), ("概率校准", "calibration"),
            ("更新时间", "updated"),
        ]
        for index, (label, key) in enumerate(fields):
            row, pair = divmod(index, 2)
            ttk.Label(current, text=label).grid(row=row, column=pair * 2, sticky="w", padx=(0, 8), pady=3)
            ttk.Label(current, textvariable=self.model_diag_vars[key]).grid(
                row=row, column=pair * 2 + 1, sticky="ew", padx=(0, 24), pady=3
            )

        metrics_frame = ttk.LabelFrame(tab, text="最近样本外指标", padding=8)
        metrics_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        metrics_frame.rowconfigure(0, weight=1)
        metrics_frame.columnconfigure(0, weight=1)
        self.model_metrics_tree = ttk.Treeview(metrics_frame, columns=("metric", "value"), show="headings", height=10)
        self.model_metrics_tree.heading("metric", text="指标")
        self.model_metrics_tree.heading("value", text="数值")
        self.model_metrics_tree.column("metric", width=190, anchor="w")
        self.model_metrics_tree.column("value", width=120, anchor="e")
        self.model_metrics_tree.grid(row=0, column=0, sticky="nsew")

        weights_frame = ttk.LabelFrame(tab, text="当前因子组权重", padding=8)
        weights_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        weights_frame.rowconfigure(0, weight=1)
        weights_frame.columnconfigure(0, weight=1)
        self.model_weights_tree = ttk.Treeview(
            weights_frame, columns=("group", "weight", "status"), show="headings", height=10
        )
        for column, text, width in (("group", "因子组", 180), ("weight", "权重", 90), ("status", "状态", 120)):
            self.model_weights_tree.heading(column, text=text)
            self.model_weights_tree.column(column, width=width, anchor="center")
        self.model_weights_tree.grid(row=0, column=0, sticky="nsew")

        contribution = ttk.LabelFrame(tab, text="因子组样本外贡献（读取离线报告）", padding=8)
        contribution.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        contribution.columnconfigure(0, weight=1)
        self.model_contribution_tree = ttk.Treeview(
            contribution,
            columns=("variant", "rank_ic", "net_return", "sharpe", "positive"),
            show="headings",
            height=5,
        )
        for column, text, width in (
            ("variant", "因子组/消融变体", 230),
            ("rank_ic", "Rank IC增量", 120),
            ("net_return", "净收益增量", 120),
            ("sharpe", "夏普增量", 110),
            ("positive", "正收益窗口占比", 140),
        ):
            self.model_contribution_tree.heading(column, text=text)
            self.model_contribution_tree.column(column, width=width, anchor="center")
        self.model_contribution_tree.grid(row=0, column=0, sticky="ew")

        operations = ttk.LabelFrame(tab, text="模型操作", padding=10)
        operations.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        operations.columnconfigure(1, weight=1)
        ttk.Label(operations, text="训练数据").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.model_training_data_path = tk.StringVar(
            value=str(engine.ROOT / "data" / "ml_training_panel.parquet")
        )
        ttk.Entry(operations, textvariable=self.model_training_data_path).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.model_training_status = tk.StringVar(value="空闲")
        ttk.Label(operations, textvariable=self.model_training_status).grid(row=0, column=2, sticky="w")
        buttons = ttk.Frame(operations)
        buttons.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="刷新模型状态", command=self.refresh_model_diagnostics).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="运行月度训练", command=self.run_monthly_model_training).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="取消训练", command=self.cancel_monthly_model_training).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="查看训练日志", command=self.open_model_training_log).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="晋升候选模型", command=self.promote_candidate_from_ui).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="回退上一正式模型", command=self.rollback_model_from_ui).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="打开模型目录", command=self.open_model_directory).pack(side=tk.LEFT, padx=6)
        self.after(150, self.refresh_model_diagnostics)

    def _load_ml_prediction_snapshot_banner(self) -> None:
        """Show the last inference timestamp without pretending it is current data."""
        path = engine.CACHE_DIR / "ml_prediction_snapshot.json"
        if not path.exists() or not hasattr(self, "ml_summary_var"):
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            snapshot_time = str(payload.get("snapshot_time") or "-")
            version = str(payload.get("model_version") or "-")
            count = len(payload.get("results") or [])
            self.ml_summary_var.set(
                f"上次预测快照：{snapshot_time}，正式模型 {version}，共 {count} 只；点击评估后更新。"
            )
        except Exception:
            return

    def refresh_model_diagnostics(self) -> None:
        loader = ProductionModelLoader(engine.ROOT)
        status = loader.status()
        version = status.get("production")
        for tree in (self.model_metrics_tree, self.model_weights_tree, self.model_contribution_tree):
            for item in tree.get_children():
                tree.delete(item)
        if not version:
            self.model_diag_vars["version"].set("未注册，请先训练并晋升候选模型")
            for key in ("range", "target", "backend", "features", "calibration", "updated"):
                self.model_diag_vars[key].set("-")
            return
        version_path = engine.ROOT / "models" / str(version)
        try:
            metadata = json.loads((version_path / "model_metadata.json").read_text(encoding="utf-8"))
            weights = json.loads((version_path / "group_weights.json").read_text(encoding="utf-8"))
            factor_status = json.loads((version_path / "factor_status.json").read_text(encoding="utf-8"))
            metrics_rows = json.loads((version_path / "training_metrics.json").read_text(encoding="utf-8"))
        except Exception as exc:
            self.model_diag_vars["version"].set(f"{version}（文件异常：{_short_error_text(exc)}）")
            return
        latest = metadata.get("latest_window", {}) if isinstance(metadata.get("latest_window"), dict) else {}
        self.model_diag_vars["version"].set(str(version))
        self.model_diag_vars["range"].set(f"{latest.get('train_start', '-')} 至 {latest.get('train_end', '-')}")
        self.model_diag_vars["target"].set(str(metadata.get("target_definition") or metadata.get("label_definition") or "-"))
        self.model_diag_vars["backend"].set(str(metadata.get("model_backend", "XGBoost CUDA")))
        self.model_diag_vars["features"].set(str(metadata.get("factor_count", "-")))
        self.model_diag_vars["calibration"].set(str(metadata.get("calibration_status", "unknown")))
        self.model_diag_vars["updated"].set(str(metadata.get("training_time", "-")))
        latest_all = {}
        if isinstance(metrics_rows, list):
            candidates = [row for row in metrics_rows if isinstance(row, dict) and row.get("model_group") == "all_factor"]
            latest_all = candidates[-1] if candidates else {}
        for key in ("auc", "pr_auc", "brier", "log_loss", "rank_ic", "net_return", "sharpe", "max_drawdown", "turnover"):
            value = latest_all.get(key, "-")
            text = f"{value:.6f}" if isinstance(value, (int, float)) and np.isfinite(value) else str(value)
            self.model_metrics_tree.insert("", tk.END, values=(key, text))
        for group, weight in sorted(weights.items()):
            self.model_weights_tree.insert(
                "", tk.END, values=(group, f"{float(weight) * 100:.2f}%", factor_status.get(group, "-"))
            )
        contribution_path = engine.ROOT / "reports" / "factor_group_ablation.csv"
        if contribution_path.exists():
            try:
                contribution = pd.read_csv(contribution_path)
                for variant, frame in contribution.groupby("variant", sort=False):
                    self.model_contribution_tree.insert(
                        "",
                        tk.END,
                        values=(
                            variant,
                            self._fmt_number(pd.to_numeric(frame.get("rank_ic_change"), errors="coerce").mean(), digits=5),
                            self._fmt_number(pd.to_numeric(frame.get("net_return_change"), errors="coerce").mean(), digits=5),
                            self._fmt_number(pd.to_numeric(frame.get("sharpe_change"), errors="coerce").mean(), digits=4),
                            self._fmt_number(pd.to_numeric(frame.get("positive_window_ratio"), errors="coerce").mean() * 100),
                        ),
                    )
            except Exception as exc:
                self.model_contribution_tree.insert("", tk.END, values=(f"报告读取失败: {_short_error_text(exc)}", "-", "-", "-", "-"))
        else:
            self.model_contribution_tree.insert("", tk.END, values=("尚未运行季度因子审计", "-", "-", "-", "-"))

    def run_monthly_model_training(self) -> None:
        if self.model_training_process is not None and self.model_training_process.poll() is None:
            self.model_training_status.set("训练仍在运行")
            return
        data_path = Path(self.model_training_data_path.get().strip())
        self.model_training_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self.model_training_log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(engine.ROOT / "scripts" / "monthly_train_desktop.py"),
            "--data",
            str(data_path),
        ]
        self.model_training_process = subprocess.Popen(
            command,
            cwd=str(engine.ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log_handle.close()
        self.model_training_status.set(f"训练中，PID {self.model_training_process.pid}")
        self._poll_model_training()

    def _poll_model_training(self) -> None:
        process = self.model_training_process
        if process is None:
            return
        code = process.poll()
        if code is None:
            try:
                lines = self.model_training_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                progress_prefixes = ("[market ", "[external]", "[panel]", "[training ", "[candidate]", "[skip]")
                progress_lines = [line.strip() for line in lines if line.strip().startswith(progress_prefixes)]
                if progress_lines:
                    self.model_training_status.set(f"训练中：{progress_lines[-1][-120:]}")
                elif lines:
                    self.model_training_status.set(f"训练中，PID {process.pid}")
            except Exception:
                pass
            self.model_training_poll_job = self.after(1000, self._poll_model_training)
            return
        self.model_training_status.set("训练完成，已生成候选模型" if code == 0 else f"训练失败，退出码 {code}")
        self.model_training_process = None
        self.refresh_model_diagnostics()

    def cancel_monthly_model_training(self) -> None:
        process = self.model_training_process
        if process is None or process.poll() is not None:
            self.model_training_status.set("当前没有训练任务")
            return
        process.terminate()
        self.model_training_status.set("正在取消训练")

    def _on_app_close(self) -> None:
        self.backtest_stop_event.set()
        processes: list[Any] = []
        if self.model_training_process is not None:
            processes.append(self.model_training_process)
        if self.backtest_process is not None:
            processes.append(self.backtest_process)
        with self.backtest_process_lock:
            processes.extend(self.backtest_processes)
        for process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
            except Exception:
                pass
        self.destroy()

    def open_model_training_log(self) -> None:
        if not self.model_training_log_path.exists():
            self.model_training_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.model_training_log_path.write_text("No training log yet.\n", encoding="utf-8")
        os.startfile(str(self.model_training_log_path))

    def open_model_directory(self) -> None:
        path = engine.ROOT / "models"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def promote_candidate_from_ui(self) -> None:
        comparison_path = engine.ROOT / "reports" / "model_comparison.json"
        if not comparison_path.exists():
            messagebox.showerror("无法晋升", "缺少样本外模型比较报告，请先完成月度训练。")
            return
        self.model_training_status.set("正在运行回归测试并验收候选模型")
        threading.Thread(target=self._promote_candidate_worker, daemon=True).start()

    def _promote_candidate_worker(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=str(engine.ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            comparison = json.loads((engine.ROOT / "reports" / "model_comparison.json").read_text(encoding="utf-8"))
            comparison["regression_tests_passed"] = completed.returncode == 0
            comparison.setdefault("minimum_better_windows", 3)
            evaluation = evaluate_candidate_model(engine.ROOT, comparison)
            promoted = promote_candidate_model(engine.ROOT, comparison)
            if promoted:
                mode = "首次绝对指标" if evaluation.get("mode") == "initial_absolute" else "相对改进指标"
                detail = f"候选模型已通过{mode}并晋升为正式模型"
            else:
                failed = ", ".join(evaluation.get("failed_checks", [])) or "unknown"
                detail = f"候选模型未通过验收：{failed}；正式模型保持不变"
        except Exception as exc:
            promoted = False
            detail = f"模型晋升失败：{_short_error_text(exc)}"
        self.queue.put(WorkerMessage("model_action", payload={"ok": promoted, "detail": detail}))

    def rollback_model_from_ui(self) -> None:
        try:
            changed = rollback_production_model(engine.ROOT)
            text = "已回退到上一正式模型" if changed else "没有可回退的上一正式模型"
            self.model_training_status.set(text)
            self.refresh_model_diagnostics()
        except Exception as exc:
            messagebox.showerror("模型回退失败", _short_error_text(exc))

    def _build_ml_tab(self) -> None:
        self.ml_tab.columnconfigure(0, weight=1)
        self.ml_tab.rowconfigure(0, weight=1)

        self.ml_symbol = tk.StringVar(value=self.bt_symbol.get())
        self.ml_start = tk.StringVar(value=self.bt_start.get())
        self.ml_adjust = tk.StringVar(value=self.bt_adjust.get())
        ml_settings = self._load_ml_portfolio_settings()
        self.ml_cash = tk.StringVar(value=str(ml_settings.get("cash") or self.bt_cash.get()))
        self.ml_target_position = tk.StringVar(value=str(ml_settings.get("target_position") or "80"))
        self.ml_fee = tk.StringVar(value=self.bt_fee.get())
        self.ml_risk = tk.StringVar(value=self.bt_risk.get())
        self.ml_horizon = tk.StringVar(value=self.bt_horizon.get())
        self.ml_advice_days = tk.StringVar(value="3")
        self.ml_parallel_workers = tk.StringVar(value=str(ml_settings.get("parallel_workers") or ML_BATCH_PARALLEL_WORKERS))
        self.ml_refresh_external_data = tk.BooleanVar(value=False)
        self.ml_shares = tk.StringVar(value=self.bt_shares.get())
        self.ml_buy_price = tk.StringVar(value=self.bt_buy_price.get())
        self.ml_buy_date = tk.StringVar(value=self.bt_buy_date.get())
        self.ml_position_symbol = tk.StringVar(value="")

        note = (
            "ML 现在只做持仓风险、异常检测和组合权重分配；传统回测仍负责买入/卖出点。\n"
            "多只股票一起评估时，会按风险分、波动率和异常状态给目标仓位；新闻/资金流源未接入时会明确标注，不会假装已分析。"
        )

        ml_body = ttk.PanedWindow(self.ml_tab, orient=tk.HORIZONTAL)
        ml_body.grid(row=0, column=0, sticky="nsew")

        saved_box = ttk.LabelFrame(self.ml_tab, text="ML股票池 / 我的持仓")
        ml_result_frame = ttk.Frame(ml_body)
        ml_body.add(saved_box, weight=1)
        ml_body.add(ml_result_frame, weight=5)
        saved_box.columnconfigure(0, weight=1)
        saved_box.rowconfigure(0, weight=0)
        saved_box.rowconfigure(1, weight=0)
        saved_box.rowconfigure(2, weight=0)
        saved_box.rowconfigure(3, weight=1)

        ml_pool_box = ttk.LabelFrame(saved_box, text="添加ML股票代码")
        ml_pool_box.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 8))
        ml_pool_box.columnconfigure(0, weight=1)
        self.ml_pool_text = tk.Text(ml_pool_box, height=3, wrap="word", background="#ffffff", foreground="#14213d", relief=tk.FLAT, padx=8, pady=6)
        self.ml_pool_text.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        ml_pool_buttons = ttk.Frame(ml_pool_box)
        ml_pool_buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ml_pool_buttons.columnconfigure(0, weight=1)
        ml_pool_buttons.columnconfigure(1, weight=1)
        ttk.Button(ml_pool_buttons, text="批量加入ML股票池", command=self._add_ml_pool_symbols, style="Primary.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(ml_pool_buttons, text="清空代码", command=lambda: self.ml_pool_text.delete("1.0", tk.END)).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ml_eval_buttons = ttk.Frame(saved_box)
        ml_eval_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 8))
        ml_eval_buttons.columnconfigure(0, weight=1)
        ml_eval_buttons.columnconfigure(1, weight=1)
        ttk.Button(ml_eval_buttons, text="评估勾选股票", command=self.run_selected_saved_stock_ml_backtests, style="Primary.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(ml_eval_buttons, text="评估全部股票池", command=self.run_all_saved_stock_ml_backtests).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.ml_stop_backtest_button = ttk.Button(ml_eval_buttons, text="终止", command=self.stop_backtest, state=tk.DISABLED)
        self.ml_stop_backtest_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(ml_eval_buttons, text="ML股票池和传统回测互不影响；右键股票可填持股/成本，或删除选中的多只股票。", foreground="#607086").grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.ml_saved_stock_tree = ttk.Treeview(
            saved_box,
            columns=("checked", "symbol", "name", "shares", "cost", "added"),
            show="headings",
            height=12,
            selectmode="extended",
        )
        self._setup_tree(
            self.ml_saved_stock_tree,
            {"checked": "选择", "symbol": "代码", "name": "名称", "shares": "持股数", "cost": "成本价", "added": "加入时间"},
            {"checked": 48, "symbol": 80, "name": 100, "shares": 80, "cost": 80, "added": 135},
        )
        self.ml_saved_stock_tree.grid(row=3, column=0, sticky="nsew")
        ml_saved_vscroll = ttk.Scrollbar(saved_box, orient=tk.VERTICAL, command=self.ml_saved_stock_tree.yview)
        ml_saved_vscroll.grid(row=3, column=1, sticky="ns")
        self.ml_saved_stock_tree.configure(yscrollcommand=ml_saved_vscroll.set)
        self.ml_saved_stock_tree.bind("<Button-1>", self._on_ml_saved_stock_click)
        self.ml_saved_stock_tree.bind("<<TreeviewSelect>>", self._on_ml_saved_stock_select)
        self.ml_saved_stock_tree.bind("<Button-3>", self._show_ml_stock_context_menu)
        self.ml_saved_stock_tree.bind("<Double-1>", lambda _event: self._edit_selected_ml_position())
        self._render_saved_stock_picker()

        ml_result_frame.columnconfigure(0, weight=1)
        ml_result_frame.rowconfigure(2, weight=1)

        top = ttk.LabelFrame(ml_result_frame, text="ML 持仓次日决策参数")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for idx in range(15):
            top.columnconfigure(idx, weight=1 if idx == 14 else 0)
        ttk.Label(top, text="建议周期").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=6)
        ttk.Combobox(top, textvariable=self.ml_advice_days, values=("1", "3", "10"), width=6, state="readonly").grid(row=0, column=1, sticky="w", padx=(0, 14), pady=6)
        ttk.Label(top, text="开始日期").grid(row=0, column=2, sticky="w", padx=(0, 4), pady=6)
        ttk.Entry(top, textvariable=self.ml_start, width=10).grid(row=0, column=3, sticky="w", padx=(0, 14), pady=6)
        ttk.Label(top, text="复权").grid(row=0, column=4, sticky="w", padx=(0, 4), pady=6)
        ttk.Combobox(top, textvariable=self.ml_adjust, values=("qfq", "", "hfq"), width=7, state="readonly").grid(row=0, column=5, sticky="w", padx=(0, 14), pady=6)
        ttk.Label(top, text="手续费").grid(row=0, column=6, sticky="w", padx=(0, 4), pady=6)
        ttk.Entry(top, textvariable=self.ml_fee, width=9).grid(row=0, column=7, sticky="w", padx=(0, 14), pady=6)
        ttk.Label(top, text="总资金").grid(row=0, column=8, sticky="w", padx=(0, 4), pady=6)
        ttk.Entry(top, textvariable=self.ml_cash, width=11).grid(row=0, column=9, sticky="w", padx=(0, 14), pady=6)
        ttk.Label(top, text="目标仓位%").grid(row=0, column=10, sticky="w", padx=(0, 4), pady=6)
        ttk.Entry(top, textvariable=self.ml_target_position, width=8).grid(row=0, column=11, sticky="w", padx=(0, 14), pady=6)
        ttk.Label(top, text="并行数").grid(row=0, column=12, sticky="w", padx=(0, 4), pady=6)
        ttk.Entry(top, textvariable=self.ml_parallel_workers, width=5).grid(row=0, column=13, sticky="w", padx=(0, 14), pady=6)
        ttk.Checkbutton(top, text="刷新外部数据", variable=self.ml_refresh_external_data).grid(row=0, column=14, sticky="w", padx=(0, 8), pady=6)

        self.ml_summary_var = tk.StringVar(value="尚未 ML 持仓决策评估")
        ttk.Label(ml_result_frame, textvariable=self.ml_summary_var, anchor="w").grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ml_result_pane = ttk.PanedWindow(ml_result_frame, orient=tk.VERTICAL)
        ml_result_pane.grid(row=2, column=0, sticky="nsew")
        ml_chart_frame = ttk.Frame(ml_result_pane)
        ml_table_frame = ttk.Frame(ml_result_pane)
        ml_result_pane.add(ml_chart_frame, weight=3)
        ml_result_pane.add(ml_table_frame, weight=2)

        ml_chart_frame.columnconfigure(0, weight=4)
        ml_chart_frame.columnconfigure(1, weight=1)
        ml_chart_frame.rowconfigure(0, weight=1)
        self.ml_canvas = tk.Canvas(ml_chart_frame, background="#fbfdff", highlightthickness=1, highlightbackground="#d8e0ea")
        self.ml_canvas.grid(row=0, column=0, sticky="nsew")
        self.ml_canvas.bind("<Configure>", self._schedule_ml_canvas_redraw)

        ml_detail_frame = ttk.LabelFrame(ml_chart_frame, text="决策详情")
        ml_detail_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ml_detail_frame.columnconfigure(0, weight=1)
        ml_detail_frame.rowconfigure(0, weight=1)
        self.ml_detail_text = tk.Text(
            ml_detail_frame,
            wrap="word",
            height=8,
            background="#ffffff",
            foreground="#14213d",
            relief=tk.FLAT,
            padx=8,
            pady=8,
        )
        self.ml_detail_text.grid(row=0, column=0, sticky="nsew")
        self.ml_detail_text.configure(state=tk.DISABLED)
        ml_detail_scroll = ttk.Scrollbar(ml_detail_frame, orient=tk.VERTICAL, command=self.ml_detail_text.yview)
        ml_detail_scroll.grid(row=0, column=1, sticky="ns")
        self.ml_detail_text.configure(yscrollcommand=ml_detail_scroll.set)

        columns = (
            "rank",
            "symbol",
            "name",
            "action",
            "utility",
            "current_weight",
            "target_weight",
            "trade_shares",
            "holding_risk",
            "risk",
            "prob10",
            "exp10",
            "fund_flow",
            "news_factor",
            "institution",
            "factor",
            "confidence",
            "completeness",
            "event_status",
            "main_driver",
            "detail",
        )
        frozen_columns = ("rank", "symbol", "name")
        scroll_columns = tuple(column for column in columns if column not in frozen_columns)
        self.ml_frozen_tree = ttk.Treeview(ml_table_frame, columns=frozen_columns, show="headings", height=9, selectmode="extended")
        self.ml_tree = ttk.Treeview(
            ml_table_frame,
            columns=columns,
            displaycolumns=scroll_columns,
            show="headings",
            height=9,
            selectmode="extended",
        )
        headings = {
            "rank": "排名",
            "symbol": "代码",
            "name": "名称",
            "action": "建议",
            "utility": "综合分",
            "current_weight": "当前仓位%",
            "holding_risk": "持仓风险",
            "target_weight": "目标仓位%",
            "trade_shares": "交易股数",
            "risk": "风险分",
            "anomaly": "异常",
            "news": "新闻风险",
            "volatility": "波动%",
            "prob10": "10日涨%",
            "exp10": "10日预期%",
            "fund_flow": "资金流%",
            "news_factor": "新闻",
            "institution": "机构活跃",
            "factor": "因子分",
            "confidence": "模型置信度%",
            "completeness": "数据完整度%",
            "event_status": "事件状态",
            "main_driver": "主要驱动",
            "detail": "决策说明",
        }
        widths = {
            "rank": 55,
            "symbol": 80,
            "name": 100,
            "action": 90,
            "utility": 75,
            "current_weight": 90,
            "target_weight": 90,
            "trade_shares": 90,
            "holding_risk": 90,
            "risk": 75,
            "prob10": 80,
            "exp10": 95,
            "fund_flow": 85,
            "news_factor": 80,
            "institution": 85,
            "factor": 75,
            "confidence": 100,
            "completeness": 105,
            "event_status": 190,
            "main_driver": 220,
            "detail": 360,
        }
        self._setup_tree(self.ml_frozen_tree, headings, widths)
        for column in frozen_columns:
            self.ml_frozen_tree.heading(column, text=headings.get(column, column), command=lambda col=column: self._sort_ml_table(col))
            self.ml_frozen_tree.column(column, stretch=False)
        self.ml_frozen_tree.grid(row=0, column=0, sticky="nsew")
        self.ml_frozen_tree.bind("<<TreeviewSelect>>", self._on_ml_frozen_rank_select)
        self._setup_tree(self.ml_tree, headings, widths)
        for column in columns:
            self.ml_tree.heading(column, text=headings.get(column, column), command=lambda col=column: self._sort_ml_table(col))
            self.ml_tree.column(column, stretch=False)
        self.ml_tree.grid(row=0, column=1, sticky="nsew")
        self.ml_tree.bind("<<TreeviewSelect>>", self._on_ml_rank_select)
        ml_table_frame.columnconfigure(0, weight=0)
        ml_table_frame.columnconfigure(1, weight=1)
        ml_table_frame.rowconfigure(0, weight=1)
        ml_vscroll = ttk.Scrollbar(ml_table_frame, orient=tk.VERTICAL, command=self._ml_table_yview)
        ml_vscroll.grid(row=0, column=2, sticky="ns")
        ml_hscroll = ttk.Scrollbar(ml_table_frame, orient=tk.HORIZONTAL, command=self.ml_tree.xview)
        ml_hscroll.grid(row=1, column=1, sticky="ew")
        self.ml_tree.configure(yscrollcommand=ml_vscroll.set)
        self.ml_tree.configure(xscrollcommand=ml_hscroll.set)
        self.ml_tree.bind("<MouseWheel>", self._on_ml_table_mousewheel)
        self.ml_frozen_tree.bind("<MouseWheel>", self._on_ml_table_mousewheel)
        self.ml_advice_days.trace_add("write", lambda *_args: self._refresh_ml_advice_view())

    def _build_ml_monitor_tab(self) -> None:
        self.ml_monitor_tab.columnconfigure(0, weight=1)
        self.ml_monitor_tab.rowconfigure(1, weight=1)

        top = ttk.Frame(self.ml_monitor_tab)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(0, weight=1)

        ttk.Label(top, text="ML盘中监控股票，逗号/换行分隔").grid(row=0, column=0, sticky="w")
        self.ml_monitor_symbols = tk.Text(top, height=3)
        self.ml_monitor_symbols.grid(row=1, column=0, sticky="ew", padx=(0, 10))
        self.ml_monitor_symbols.insert("1.0", self.ml_symbol.get() if hasattr(self, "ml_symbol") else "002472")

        saved_box = ttk.LabelFrame(top, text="已有 ML 策略股票")
        saved_box.grid(row=2, column=0, sticky="ew", padx=(0, 10), pady=(8, 0))
        saved_box.columnconfigure(0, weight=1)
        self.ml_monitor_saved_stock_tree = ttk.Treeview(
            saved_box,
            columns=("symbol", "name", "count", "latest"),
            show="headings",
            height=4,
            selectmode="extended",
        )
        self._setup_tree(
            self.ml_monitor_saved_stock_tree,
            {"symbol": "代码", "name": "名称", "count": "ML数", "latest": "最近保存"},
            {"symbol": 90, "name": 120, "count": 70, "latest": 150},
        )
        self.ml_monitor_saved_stock_tree.grid(row=0, column=0, sticky="ew")

        controls = ttk.Frame(top)
        controls.grid(row=1, column=1, sticky="n")
        self.ml_monitor_period = tk.StringVar(value="5")
        self.ml_monitor_interval = tk.StringVar(value="30")
        self.ml_monitor_shares = tk.StringVar(value="")
        self.ml_monitor_buy_price = tk.StringVar(value="")
        self._labeled_combo(controls, "周期", self.ml_monitor_period, ("1", "5", "15"), 0)
        self._labeled_entry(controls, "刷新秒", self.ml_monitor_interval, 1, width=8)
        self._labeled_entry(controls, "持股数", self.ml_monitor_shares, 2, width=10)
        self._labeled_entry(controls, "成本价", self.ml_monitor_buy_price, 3, width=10)
        ttk.Button(controls, text="刷新一次", command=self.refresh_ml_monitor_once).grid(row=1, column=4, padx=(0, 8))
        self.ml_monitor_start_button = ttk.Button(controls, text="开始ML监控", command=self.toggle_ml_monitor)
        self.ml_monitor_start_button.grid(row=1, column=5)

        saved_controls = ttk.Frame(top)
        saved_controls.grid(row=2, column=1, sticky="nw", pady=(8, 0))
        ttk.Button(saved_controls, text="使用选中ML股票", command=self._use_selected_ml_monitor_symbols).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(saved_controls, text="全选ML股票", command=self._select_all_ml_monitor_stocks).grid(row=1, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(saved_controls, text="清空输入", command=lambda: self.ml_monitor_symbols.delete("1.0", "end")).grid(row=2, column=0, sticky="ew")

        body = ttk.PanedWindow(self.ml_monitor_tab, orient=tk.VERTICAL)
        body.grid(row=1, column=0, sticky="nsew")
        table_frame = ttk.Frame(body)
        chart_frame = ttk.Frame(body)
        body.add(table_frame, weight=2)
        body.add(chart_frame, weight=4)

        columns = ("symbol", "name", "action", "price", "daily_gate", "trend", "volume", "vwap", "stop", "time")
        self.ml_monitor_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=9)
        self._setup_tree(
            self.ml_monitor_tree,
            {
                "symbol": "代码",
                "name": "名称",
                "action": "ML信号",
                "price": "价格",
                "daily_gate": "ML日线",
                "trend": "分钟趋势",
                "volume": "量能比",
                "vwap": "VWAP",
                "stop": "风控线",
                "time": "行情时间",
            },
            {
                "symbol": 90,
                "name": 120,
                "action": 130,
                "price": 90,
                "daily_gate": 110,
                "trend": 90,
                "volume": 90,
                "vwap": 90,
                "stop": 90,
                "time": 160,
            },
        )
        self.ml_monitor_tree.grid(row=0, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        ml_monitor_vscroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.ml_monitor_tree.yview)
        ml_monitor_vscroll.grid(row=0, column=1, sticky="ns")
        self.ml_monitor_tree.configure(yscrollcommand=ml_monitor_vscroll.set)
        self.ml_monitor_tree.bind("<<TreeviewSelect>>", self._on_ml_monitor_select)
        self.ml_monitor_tree.bind("<Double-1>", self._open_ml_monitor_xueqiu)

        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.ml_monitor_canvas = tk.Canvas(chart_frame, height=360, background="#fbfdff", highlightthickness=1, highlightbackground="#d8e0ea")
        self.ml_monitor_canvas.grid(row=0, column=0, sticky="nsew")
        self.ml_monitor_canvas.bind("<Configure>", self._schedule_ml_monitor_canvas_redraw)
        strategy_box = ttk.LabelFrame(chart_frame, text="ML盘中监控使用的保存策略")
        strategy_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        strategy_box.columnconfigure(0, weight=1)
        self.ml_monitor_strategy_tree = ttk.Treeview(strategy_box, columns=("strategy", "mode", "saved"), show="headings", height=3)
        self._setup_tree(
            self.ml_monitor_strategy_tree,
            {"strategy": "策略", "mode": "模式", "saved": "保存时间"},
            {"strategy": 180, "mode": 180, "saved": 150},
        )
        self.ml_monitor_strategy_tree.grid(row=0, column=0, sticky="ew")
        self.ml_monitor_strategy_tree.bind("<<TreeviewSelect>>", self._on_ml_monitor_strategy_select)
        ttk.Button(chart_frame, text="打开雪球", command=self._open_ml_monitor_xueqiu).grid(row=2, column=0, sticky="e", pady=(6, 0))
        self._render_saved_stock_picker()

    def _labeled_entry(self, parent: ttk.Frame, label: str, var: tk.StringVar, col: int, width: int = 10) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=col, sticky="w", padx=(0, 6))
        ttk.Entry(parent, textvariable=var, width=width).grid(row=1, column=col, sticky="w", padx=(0, 8))

    def _labeled_combo(self, parent: ttk.Frame, label: str, var: tk.StringVar, values: tuple[str, ...], col: int) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=col, sticky="w", padx=(0, 6))
        ttk.Combobox(parent, textvariable=var, values=values, width=9, state="readonly").grid(row=1, column=col, sticky="w", padx=(0, 8))

    def _setup_tree(self, tree: ttk.Treeview, headings: dict[str, str], widths: dict[str, int]) -> None:
        columns = list(tree["columns"])
        stretch_col = columns[-1] if columns else ""
        for col in columns:
            tree.heading(col, text=headings.get(col, col))
            tree.column(col, width=widths.get(col, 90), anchor="center", stretch=(col == stretch_col))

    def _load_ml_portfolio_settings(self) -> dict[str, Any]:
        if not ML_PORTFOLIO_SETTINGS_PATH.exists():
            return {}
        try:
            with ML_PORTFOLIO_SETTINGS_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_ml_portfolio_settings(self) -> None:
        engine.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.ml_cash.get().strip(),
            "target_position": self.ml_target_position.get().strip(),
            "parallel_workers": self.ml_parallel_workers.get().strip(),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp_path = ML_PORTFOLIO_SETTINGS_PATH.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(ML_PORTFOLIO_SETTINGS_PATH)

    def _load_ml_stock_pool(self) -> dict[str, dict[str, Any]]:
        if not ML_STOCK_POOL_PATH.exists():
            return {}
        try:
            with ML_STOCK_POOL_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        items = payload.get("items", payload) if isinstance(payload, dict) else {}
        if not isinstance(items, dict):
            return {}
        pool: dict[str, dict[str, Any]] = {}
        for raw_symbol, raw_item in items.items():
            try:
                symbol = engine.normalize_symbol(str(raw_symbol))
            except Exception:
                symbol = str(raw_symbol).strip()
            if not symbol:
                continue
            item = raw_item if isinstance(raw_item, dict) else {}
            pool[symbol] = {
                "symbol": symbol,
                "name": str(item.get("name") or ""),
                "shares": str(item.get("shares") or ""),
                "cost": str(item.get("cost") or ""),
                "added_at": str(item.get("added_at") or ""),
                "updated_at": str(item.get("updated_at") or ""),
            }
        return pool

    def _save_ml_stock_pool(self, pool: dict[str, dict[str, Any]]) -> None:
        engine.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "items": pool}
        tmp_path = ML_STOCK_POOL_PATH.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(ML_STOCK_POOL_PATH)

    def _resolve_stock_display_name(self, symbol: str) -> str:
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol).strip()
        name = self._saved_stock_name(code)
        if name:
            return name
        try:
            return engine.stock_display_name(code)
        except Exception:
            return ""

    def _ml_stock_name(self, symbol: str) -> str:
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol).strip()
        item = self._load_ml_stock_pool().get(code, {})
        name = str(item.get("name") or "").strip()
        if name:
            return name
        resolved = self._resolve_stock_display_name(code)
        if resolved:
            pool = self._load_ml_stock_pool()
            if code in pool:
                pool[code]["name"] = resolved
                self._save_ml_stock_pool(pool)
        return resolved

    def _ml_stock_cached_name(self, symbol: str) -> str:
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol).strip()
        item = self._load_ml_stock_pool().get(code, {})
        name = str(item.get("name") or "").strip()
        if name:
            return name
        return self._saved_stock_name(code)

    def _ml_stock_position(self, symbol: str) -> dict[str, str]:
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol).strip()
        item = self._load_ml_stock_pool().get(code, {})
        return {"shares": str(item.get("shares") or ""), "cost": str(item.get("cost") or "")}

    def _save_ml_stock_position(self, symbol: str, shares: str, cost: str) -> None:
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol).strip()
        if not code:
            return
        pool = self._load_ml_stock_pool()
        item = pool.setdefault(
            code,
            {
                "symbol": code,
                "name": engine.stock_display_name(code),
                "shares": "",
                "cost": "",
                "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": "",
            },
        )
        item["name"] = str(item.get("name") or engine.stock_display_name(code))
        item["shares"] = shares.strip()
        item["cost"] = cost.strip()
        item["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_ml_stock_pool(pool)

    def _ml_stock_pool_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        pool = self._load_ml_stock_pool()
        changed = False
        for symbol, item in pool.items():
            shares = str(item.get("shares") or "")
            try:
                shares_count = int(float(shares or 0))
            except Exception:
                shares_count = 0
            name = str(item.get("name") or "").strip()
            if not name:
                name = self._resolve_stock_display_name(symbol)
                if name:
                    item["name"] = name
                    changed = True
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "shares": shares,
                    "cost": str(item.get("cost") or ""),
                    "added_at": str(item.get("added_at") or ""),
                    "updated_at": str(item.get("updated_at") or ""),
                    "shares_count": shares_count,
                }
            )
        rows.sort(key=lambda row: str(row.get("added_at", "")), reverse=True)
        rows.sort(key=lambda row: 0 if int(row.get("shares_count") or 0) > 0 else 1)
        if changed:
            self._save_ml_stock_pool(pool)
        return rows

    def _record_strategy_type(self, record: dict[str, Any]) -> str:
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        signal = result.get("daily_signal", {}) if isinstance(result.get("daily_signal"), dict) else {}
        return str(signal.get("strategy_type", "")).strip()

    def _record_matches_strategy_filter(
        self,
        record: dict[str, Any],
        strategy_filter: str | None = None,
        exclude_strategy_type: str | None = None,
    ) -> bool:
        strategy_type = self._record_strategy_type(record)
        if strategy_filter is not None and strategy_type != strategy_filter:
            return False
        if exclude_strategy_type is not None and strategy_type == exclude_strategy_type:
            return False
        return True

    def _saved_stock_rows(
        self,
        strategy_filter: str | None = None,
        exclude_strategy_type: str | None = None,
        selected_for_left_only: bool = False,
    ) -> list[dict[str, Any]]:
        stocks: dict[str, dict[str, Any]] = {}
        for record in engine.load_persistent_strategy_cache().values():
            if not isinstance(record, dict):
                continue
            if not self._record_matches_strategy_filter(record, strategy_filter, exclude_strategy_type):
                continue
            if selected_for_left_only and not bool(record.get("selected_for_left")):
                continue
            symbol = str(record.get("symbol", "")).strip()
            if not symbol:
                continue
            try:
                code = engine.normalize_symbol(symbol)
            except Exception:
                code = symbol
            row = stocks.setdefault(
                code,
                {
                    "symbol": code,
                    "name": str(record.get("name") or ""),
                    "count": 0,
                    "latest": "",
                },
            )
            row["count"] += 1
            saved_at = str(record.get("saved_at", ""))
            if saved_at > str(row.get("latest", "")):
                row["latest"] = saved_at
                row["name"] = str(record.get("name") or row["name"])
        return sorted(stocks.values(), key=lambda item: str(item.get("latest", "")), reverse=True)

    def _saved_stock_name(self, symbol: str) -> str:
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol)
        if hasattr(self, "saved_stock_tree") and self.saved_stock_tree.exists(code):
            values = self.saved_stock_tree.item(code, "values")
            if len(values) > 1 and values[1]:
                return str(values[1])
        item = self.monitor_items.get(code)
        if isinstance(item, dict) and item.get("name"):
            return str(item["name"])
        for _key_text, record in self._saved_records_for_symbol(code):
            name = str(record.get("name") or "")
            if name:
                return name
        return ""

    def _saved_records_for_symbol(
        self,
        symbol: str,
        strategy_filter: str | None = None,
        exclude_strategy_type: str | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol)
        rows: list[tuple[str, dict[str, Any]]] = []
        for key_text, record in engine.load_persistent_strategy_cache().items():
            if not isinstance(record, dict):
                continue
            try:
                record_code = engine.normalize_symbol(str(record.get("symbol", "")))
            except Exception:
                record_code = str(record.get("symbol", ""))
            if record_code != code:
                continue
            if not self._record_matches_strategy_filter(record, strategy_filter, exclude_strategy_type):
                continue
            rows.append((key_text, record))
        rows.sort(key=lambda item: str(item[1].get("saved_at", "")), reverse=True)
        rows.sort(key=lambda item: 0 if bool(item[1].get("active_for_trading")) else 1)
        return rows

    def _cache_stock_iid(self, symbol: str) -> str:
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol)
        return f"stock:{code}"

    def _cache_iid_symbol(self, iid: str) -> str:
        return iid.split(":", 1)[1] if iid.startswith("stock:") else ""

    def _strategy_display_name(self, record: dict[str, Any]) -> str:
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        signal = result.get("daily_signal", {}) if isinstance(result.get("daily_signal"), dict) else {}
        return f"{signal.get('strategy_label', result.get('strategy_label', ''))} {signal.get('fast', '')}/{signal.get('slow', '')}".strip()

    def _set_active_strategy_for_symbol(self, key_text: str) -> None:
        cache = engine.load_persistent_strategy_cache()
        record = cache.get(key_text)
        if not isinstance(record, dict):
            return
        try:
            code = engine.normalize_symbol(str(record.get("symbol", "")))
        except Exception:
            code = str(record.get("symbol", ""))
        if not code:
            return
        record["active_for_trading"] = True
        record["selected_for_left"] = True
        engine.save_persistent_strategy_cache()
        self.monitor_strategy_keys[code] = key_text
        self.ml_monitor_strategy_keys[code] = key_text

    def _select_strategy_for_left_from_key(self, key_text: str, refresh: bool = True) -> None:
        self._set_active_strategy_for_symbol(key_text)
        if refresh:
            self._render_strategy_cache_list()
        self.status_var.set("已把该策略加入左侧，并设为盘中监控/ML使用策略")

    def _right_table_saved_key(self, row_id: str | None = None) -> str:
        if row_id is None:
            selection = self.bt_tree.selection()
            row_id = str(selection[0]) if selection else ""
        if not row_id or not row_id.startswith("saved:"):
            return ""
        tags = self.bt_tree.item(row_id, "tags")
        return str(tags[0]) if tags else ""

    def _stock_position(self, symbol: str) -> dict[str, Any]:
        rows = self._saved_records_for_symbol(symbol, exclude_strategy_type="ml")
        for _key_text, record in rows:
            position = record.get("position")
            if isinstance(position, dict):
                return {
                    "monitor_enabled": bool(position.get("monitor_enabled")),
                    "shares": str(position.get("shares", "")),
                    "cost": str(position.get("cost", "")),
                }
        return {"monitor_enabled": False, "shares": "", "cost": ""}

    def _save_stock_position(
        self,
        symbol: str,
        shares: str | None = None,
        cost: str | None = None,
        monitor_enabled: bool | None = None,
    ) -> None:
        rows = self._saved_records_for_symbol(symbol, exclude_strategy_type="ml")
        if not rows:
            return
        cache = engine.load_persistent_strategy_cache()
        for key_text, record in rows:
            editable = cache.get(key_text)
            if not isinstance(editable, dict):
                continue
            position = editable.get("position")
            if not isinstance(position, dict):
                position = {}
                editable["position"] = position
            if shares is not None:
                position["shares"] = shares.strip()
            if cost is not None:
                position["cost"] = cost.strip()
            if monitor_enabled is not None:
                position["monitor_enabled"] = bool(monitor_enabled)
        engine.save_persistent_strategy_cache()

    def _latest_strategy_label_for_symbol(self, symbol: str) -> str:
        rows = self._saved_strategy_rows_for_symbol(symbol, exclude_strategy_type="ml")
        if not rows:
            return ""
        return str(rows[0][1][0]).lstrip("★ ").strip()

    def _monitor_enabled_symbols(self) -> list[str]:
        rows = self._saved_stock_rows(exclude_strategy_type="ml")
        return [str(row["symbol"]) for row in rows if self._stock_position(str(row["symbol"])).get("monitor_enabled")]

    def _render_saved_stock_picker(self) -> None:
        tree_rows = {
            "saved_stock_tree": self._saved_stock_rows(exclude_strategy_type="ml"),
            "ml_saved_stock_tree": self._ml_stock_pool_rows(),
            "ml_monitor_saved_stock_tree": self._saved_stock_rows(strategy_filter="ml"),
        }
        for tree_name, rows in tree_rows.items():
            if not hasattr(self, tree_name):
                continue
            tree = getattr(self, tree_name)
            selected = set(tree.selection())
            for iid in tree.get_children():
                tree.delete(iid)
            if tree_name == "ml_saved_stock_tree":
                self.ml_checked_symbols.intersection_update({str(row["symbol"]) for row in rows})
            for row in rows:
                symbol = str(row["symbol"])
                if tree_name == "saved_stock_tree":
                    position = self._stock_position(symbol)
                    item = self.monitor_items.get(symbol, {})
                    values = (
                        symbol,
                        row.get("name", ""),
                        item.get("action", "-"),
                        item.get("price", "-"),
                        position.get("shares", ""),
                        position.get("cost", ""),
                        self._latest_strategy_label_for_symbol(symbol),
                    )
                elif tree_name == "ml_saved_stock_tree":
                    values = (
                        "☑" if symbol in self.ml_checked_symbols else "☐",
                        symbol,
                        row.get("name", ""),
                        row.get("shares", ""),
                        row.get("cost", ""),
                        row.get("added_at", ""),
                    )
                else:
                    values = (symbol, row.get("name", ""), row.get("count", 0), row.get("latest", ""))
                tree.insert("", "end", iid=symbol, values=values)
            for symbol in selected:
                if tree.exists(symbol):
                    tree.selection_add(symbol)

    def _position_shares_count(self, symbol: str) -> int:
        position = self._stock_position(symbol)
        try:
            return int(float(position.get("shares", "") or 0))
        except Exception:
            return 0

    def _selected_or_all_saved_symbols(self) -> list[str]:
        rows = self._saved_stock_rows(exclude_strategy_type="ml")
        all_symbols = [str(row["symbol"]) for row in rows]
        if hasattr(self, "saved_stock_tree"):
            selected = [symbol for symbol in self.saved_stock_tree.selection() if symbol in all_symbols]
            if selected:
                return selected
        return all_symbols

    def _all_strategy_cache_symbols(self) -> list[str]:
        rows = self._saved_stock_rows(exclude_strategy_type="ml")
        return [str(row["symbol"]) for row in rows]

    def _checked_strategy_cache_symbols(self) -> list[str]:
        all_symbols = self._all_strategy_cache_symbols()
        checked = set(self.backtest_checked_symbols)
        return [symbol for symbol in all_symbols if symbol in checked]

    def _batch_input_symbols(self) -> list[str]:
        if not hasattr(self, "bt_batch_text"):
            return []
        text = self.bt_batch_text.get("1.0", tk.END)
        for sep in ("\n", "\r", "\t", ",", "，", ";", "；", "、", "|"):
            text = text.replace(sep, " ")
        symbols: list[str] = []
        seen: set[str] = set()
        for token in text.split():
            raw = token.strip().strip("'\"")
            if not raw:
                continue
            try:
                symbol = engine.resolve_stock_identifier(raw)
            except Exception:
                symbol = raw
            if symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)
        return symbols

    def _ml_pool_input_symbols(self) -> list[str]:
        if not hasattr(self, "ml_pool_text"):
            return []
        return _parse_batch_symbols(self.ml_pool_text.get("1.0", tk.END))

    def _add_ml_pool_symbols(self) -> None:
        symbols = self._ml_pool_input_symbols()
        if not symbols:
            self.status_var.set("请先输入 ML 股票代码或股票名称，多个用逗号/空格/换行分隔")
            return
        pool = self._load_ml_stock_pool()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        added = 0
        for symbol in symbols:
            try:
                code = engine.normalize_symbol(symbol)
            except Exception:
                code = str(symbol).strip()
            if not code:
                continue
            item = pool.setdefault(
                code,
                {
                    "symbol": code,
                    "name": "",
                    "shares": "",
                    "cost": "",
                    "added_at": now,
                    "updated_at": "",
                },
            )
            if not item.get("added_at"):
                item["added_at"] = now
            item["name"] = str(item.get("name") or self._resolve_stock_display_name(code))
            added += 1
            self.ml_checked_symbols.add(code)
        self._save_ml_stock_pool(pool)
        self._render_saved_stock_picker()
        if hasattr(self, "ml_pool_text"):
            self.ml_pool_text.delete("1.0", tk.END)
        self.status_var.set(f"已加入/更新 {added} 只 ML 股票，并自动勾选")

    def _selected_ml_pool_symbols_for_menu(self) -> list[str]:
        if not hasattr(self, "ml_saved_stock_tree"):
            return []
        pool_symbols = {str(row["symbol"]) for row in self._ml_stock_pool_rows()}
        selected = [str(item) for item in self.ml_saved_stock_tree.selection() if str(item) in pool_symbols]
        focused = str(self.ml_saved_stock_tree.focus() or "")
        if not selected and focused in pool_symbols:
            selected = [focused]
        return selected

    def _show_ml_stock_context_menu(self, event: tk.Event) -> str | None:
        if not hasattr(self, "ml_saved_stock_tree"):
            return None
        row_id = self.ml_saved_stock_tree.identify_row(event.y)
        if row_id:
            if row_id not in self.ml_saved_stock_tree.selection():
                self.ml_saved_stock_tree.selection_set(row_id)
            self.ml_saved_stock_tree.focus(row_id)
        symbols = self._selected_ml_pool_symbols_for_menu()
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="填写持股/成本", command=self._edit_selected_ml_position, state=tk.NORMAL if symbols else tk.DISABLED)
        menu.add_separator()
        menu.add_command(label="勾选选中股票", command=lambda: self._set_selected_ml_checked(True), state=tk.NORMAL if symbols else tk.DISABLED)
        menu.add_command(label="取消勾选选中股票", command=lambda: self._set_selected_ml_checked(False), state=tk.NORMAL if symbols else tk.DISABLED)
        menu.add_separator()
        menu.add_command(label="删除选中股票", command=self._delete_selected_ml_pool_symbols, state=tk.NORMAL if symbols else tk.DISABLED)
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _set_selected_ml_checked(self, checked: bool) -> None:
        symbols = self._selected_ml_pool_symbols_for_menu()
        if not symbols:
            return
        if checked:
            self.ml_checked_symbols.update(symbols)
            action = "勾选"
        else:
            self.ml_checked_symbols.difference_update(symbols)
            action = "取消勾选"
        self._render_saved_stock_picker()
        for symbol in symbols:
            if self.ml_saved_stock_tree.exists(symbol):
                self.ml_saved_stock_tree.selection_add(symbol)
        self.status_var.set(f"已{action} {len(symbols)} 只 ML 股票")

    def _edit_selected_ml_position(self) -> None:
        symbols = self._selected_ml_pool_symbols_for_menu()
        if not symbols:
            self.status_var.set("请先选择一只 ML 股票")
            return
        self._edit_ml_position_popup(symbols[0])

    def _edit_ml_position_popup(self, symbol: str) -> None:
        position = self._ml_stock_position(symbol)
        name = self._ml_stock_name(symbol)
        popup = tk.Toplevel(self)
        popup.title(f"填写持股信息 - {symbol} {name}")
        popup.transient(self)
        popup.grab_set()
        popup.columnconfigure(1, weight=1)

        shares_var = tk.StringVar(value=str(position.get("shares", "")))
        cost_var = tk.StringVar(value=str(position.get("cost", "")))

        ttk.Label(popup, text=f"{symbol} {name}").grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 8))
        ttk.Label(popup, text="持股数").grid(row=1, column=0, sticky="w", padx=(14, 8), pady=6)
        shares_entry = ttk.Entry(popup, textvariable=shares_var, width=18)
        shares_entry.grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=6)
        ttk.Label(popup, text="成本价").grid(row=2, column=0, sticky="w", padx=(14, 8), pady=6)
        cost_entry = ttk.Entry(popup, textvariable=cost_var, width=18)
        cost_entry.grid(row=2, column=1, sticky="ew", padx=(0, 14), pady=6)

        def is_number_or_blank(value: str) -> bool:
            if not value:
                return True
            try:
                float(value)
                return True
            except ValueError:
                return False

        def save() -> None:
            shares = shares_var.get().strip()
            cost = cost_var.get().strip()
            if not is_number_or_blank(shares):
                messagebox.showwarning("持仓格式不对", "持股数只能填数字，或留空。", parent=popup)
                return
            if not is_number_or_blank(cost):
                messagebox.showwarning("成本格式不对", "成本价只能填数字，或留空。", parent=popup)
                return
            self._save_ml_stock_position(symbol, shares, cost)
            self._render_saved_stock_picker()
            if self.ml_saved_stock_tree.exists(symbol):
                self.ml_saved_stock_tree.selection_set(symbol)
                self.ml_saved_stock_tree.focus(symbol)
            self.status_var.set(f"{symbol} {name} 的 ML 持股/成本已保存")
            popup.destroy()

        buttons = ttk.Frame(popup)
        buttons.grid(row=3, column=0, columnspan=2, sticky="ew", padx=14, pady=(8, 14))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="保存", command=save, style="Primary.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="取消", command=popup.destroy).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        shares_entry.focus_set()
        shares_entry.select_range(0, "end")
        popup.bind("<Return>", lambda _event: save())
        popup.bind("<Escape>", lambda _event: popup.destroy())

    def _delete_selected_ml_pool_symbols(self) -> None:
        symbols = self._selected_ml_pool_symbols_for_menu()
        if not symbols:
            self.status_var.set("请先选择要删除的 ML 股票")
            return
        if not messagebox.askyesno("删除ML股票", f"确定从 ML 股票池删除 {len(symbols)} 只股票吗？"):
            return
        pool = self._load_ml_stock_pool()
        for symbol in symbols:
            pool.pop(symbol, None)
            self.ml_checked_symbols.discard(symbol)
        self._save_ml_stock_pool(pool)
        self._render_saved_stock_picker()
        self.status_var.set(f"已从 ML 股票池删除 {len(symbols)} 只股票")

    def _use_selected_saved_monitor_symbols(self) -> None:
        symbols = self._selected_or_all_saved_symbols()
        if not symbols:
            self.status_var.set("还没有已保存股票，先在回测页跑一次并保存策略")
            return
        for symbol in symbols:
            self._save_stock_position(symbol, monitor_enabled=True)
        self._render_saved_stock_picker()
        self.status_var.set(f"已勾选 {len(symbols)} 只已保存股票，会按刷新秒自动监控")

    def _select_all_saved_stocks(self) -> None:
        if not hasattr(self, "saved_stock_tree"):
            return
        children = self.saved_stock_tree.get_children()
        if children:
            self.saved_stock_tree.selection_set(children)
            self.status_var.set(f"已选择 {len(children)} 只已保存股票")

    def _on_monitor_saved_stock_select(self, _event: object | None = None) -> None:
        selection = self.saved_stock_tree.selection()
        if not selection:
            return
        symbol = str(selection[0])
        self.selected_monitor_symbol = symbol
        self.monitor_xueqiu_var.set(f"{symbol} {self._saved_stock_name(symbol)} | {engine.xueqiu_url(symbol)}")
        if self.monitor_select_job is not None:
            try:
                self.after_cancel(self.monitor_select_job)
            except Exception:
                pass
        self.monitor_select_job = self.after(120, lambda stock=symbol: self._apply_monitor_saved_stock_selection(stock))

    def _apply_monitor_saved_stock_selection(self, symbol: str) -> None:
        self.monitor_select_job = None
        if not hasattr(self, "saved_stock_tree") or not self.saved_stock_tree.exists(symbol):
            return
        current = self.saved_stock_tree.selection()
        if not current or str(current[0]) != symbol:
            return
        self.selected_monitor_symbol = symbol
        position = self._stock_position(symbol)
        self.monitor_position_shares.set(str(position.get("shares", "")))
        self.monitor_position_cost.set(str(position.get("cost", "")))
        self._render_monitor_strategy_list(symbol)
        item = self.monitor_items.get(symbol)
        self._render_monitor_detail(item)
        if item:
            self._draw_intraday_chart(item)
        else:
            self._show_monitor_waiting(symbol)

    def _toggle_monitor_saved_stock(self, _event: object | None = None) -> None:
        selection = self.saved_stock_tree.selection()
        if not selection:
            self.status_var.set("请先在左侧选择一只已保存股票")
            return
        symbol = str(selection[0])
        current = bool(self._stock_position(symbol).get("monitor_enabled"))
        self._save_stock_position(symbol, monitor_enabled=not current)
        self._render_saved_stock_picker()
        self.saved_stock_tree.selection_set(symbol)
        self.selected_monitor_symbol = symbol
        self._on_monitor_saved_stock_select()
        self.status_var.set(f"{symbol} 已{'加入' if not current else '移出'}盘中监控")
        if not current:
            self._show_monitor_loading(symbol)
            self.refresh_monitor_symbol(symbol)

    def _on_saved_stock_click(self, event: tk.Event) -> str | None:
        row_id = self.saved_stock_tree.identify_row(event.y)
        if row_id:
            self.saved_stock_tree.selection_set(row_id)
            self.saved_stock_tree.focus(row_id)
            self._on_monitor_saved_stock_select()
            return "break"
        return None

    def _on_saved_stock_double_click(self, event: tk.Event) -> str | None:
        row_id = self.saved_stock_tree.identify_row(event.y)
        if not row_id:
            return None
        self.saved_stock_tree.selection_set(row_id)
        self.saved_stock_tree.focus(row_id)
        self._apply_monitor_saved_stock_selection(row_id)
        if hasattr(self, "monitor_position_shares_entry"):
            self.monitor_position_shares_entry.focus_set()
            self.monitor_position_shares_entry.select_range(0, "end")
        self.status_var.set(f"{row_id} 可在下方填写持股数和成本价")
        return "break"

    def _show_saved_stock_context_menu(self, event: tk.Event) -> str | None:
        row_id = self.saved_stock_tree.identify_row(event.y)
        if not row_id:
            return None
        self.saved_stock_tree.selection_set(row_id)
        self.saved_stock_tree.focus(row_id)
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="刷新这只股票", command=lambda symbol=row_id: self.refresh_monitor_symbol_with_loading(symbol))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def _tree_column_name(self, tree: ttk.Treeview, col_id: str) -> str:
        try:
            index = int(col_id.replace("#", "")) - 1
        except ValueError:
            return ""
        columns = list(tree["columns"])
        if 0 <= index < len(columns):
            return str(columns[index])
        return ""

    def _edit_saved_stock_position_cell(self, symbol: str, field: str = "shares", col_id: str = "") -> None:
        position = self._stock_position(symbol)
        dialog = tk.Toplevel(self)
        dialog.title(f"编辑持仓 - {symbol}")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.configure(background="#f8fafc")

        name = self._saved_stock_name(symbol)
        shares_var = tk.StringVar(value=str(position.get("shares", "")))
        cost_var = tk.StringVar(value=str(position.get("cost", "")))

        frame = ttk.Frame(dialog, padding=(18, 16, 18, 14))
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=f"{symbol} {name}", font=("Microsoft YaHei UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Label(frame, text="持股数").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        shares_entry = ttk.Entry(frame, textvariable=shares_var, width=20)
        shares_entry.grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Label(frame, text="成本价").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=6)
        cost_entry = ttk.Entry(frame, textvariable=cost_var, width=20)
        cost_entry.grid(row=2, column=1, sticky="ew", pady=6)

        def is_number_or_blank(value: str) -> bool:
            if not value.strip():
                return True
            try:
                float(value)
                return True
            except ValueError:
                return False

        def commit(_event: object | None = None) -> None:
            shares = shares_var.get().strip()
            cost = cost_var.get().strip()
            if not is_number_or_blank(shares):
                messagebox.showwarning("持仓格式不对", "持股数只能填数字，或留空。", parent=dialog)
                shares_entry.focus_set()
                shares_entry.select_range(0, "end")
                return
            if not is_number_or_blank(cost):
                messagebox.showwarning("成本格式不对", "成本价只能填数字，或留空。", parent=dialog)
                cost_entry.focus_set()
                cost_entry.select_range(0, "end")
                return
            self._save_stock_position(symbol, shares=shares, cost=cost)
            dialog.destroy()
            self._render_saved_stock_picker()
            if self.saved_stock_tree.exists(symbol):
                self.saved_stock_tree.selection_set(symbol)
                self.saved_stock_tree.focus(symbol)
            self.status_var.set(f"{symbol} 持股/成本已保存")
            if symbol == self.selected_monitor_symbol:
                self.refresh_monitor_symbol(symbol)

        button_row = ttk.Frame(frame)
        button_row.grid(row=3, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(button_row, text="取消", command=dialog.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="保存", command=commit, style="Primary.TButton").grid(row=0, column=1)

        dialog.bind("<Return>", commit)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.update_idletasks()
        parent_x = self.winfo_rootx()
        parent_y = self.winfo_rooty()
        parent_w = self.winfo_width()
        parent_h = self.winfo_height()
        dialog_w = dialog.winfo_width()
        dialog_h = dialog.winfo_height()
        dialog.geometry(f"+{parent_x + max(80, (parent_w - dialog_w) // 2)}+{parent_y + max(80, (parent_h - dialog_h) // 2)}")
        if field == "cost":
            cost_entry.focus_set()
            cost_entry.select_range(0, "end")
        else:
            shares_entry.focus_set()
            shares_entry.select_range(0, "end")
        dialog.grab_set()
        dialog.lift()
        dialog.focus_force()

    def _save_selected_monitor_position(self) -> None:
        selection = self.saved_stock_tree.selection()
        if not selection:
            self.status_var.set("请先在左侧选择一只已保存股票")
            return
        symbol = str(selection[0])
        shares = self.monitor_position_shares.get().strip()
        cost = self.monitor_position_cost.get().strip()

        def is_number_or_blank(value: str) -> bool:
            if not value:
                return True
            try:
                float(value)
                return True
            except ValueError:
                return False

        if not is_number_or_blank(shares):
            messagebox.showwarning("持仓格式不对", "持股数只能填数字，或留空。")
            return
        if not is_number_or_blank(cost):
            messagebox.showwarning("成本格式不对", "成本价只能填数字，或留空。")
            return

        self._save_stock_position(symbol, shares=shares, cost=cost)
        self._render_saved_stock_picker()
        self.saved_stock_tree.selection_set(symbol)
        self.saved_stock_tree.focus(symbol)
        self.status_var.set(f"{symbol} 持股/成本已保存，并已同步到上方股票表")
        if symbol == self.selected_monitor_symbol:
            self.refresh_monitor_symbol(symbol)

    def _save_selected_ml_position(self) -> None:
        symbol = ""
        if hasattr(self, "ml_saved_stock_tree"):
            focused = self.ml_saved_stock_tree.focus()
            selection = self.ml_saved_stock_tree.selection()
            symbol = str(focused or (selection[0] if selection else ""))
        shares = self.ml_shares.get().strip()
        cost = self.ml_buy_price.get().strip()
        cash = self.ml_cash.get().strip()
        target_position = self.ml_target_position.get().strip()

        def is_number_or_blank(value: str) -> bool:
            if not value:
                return True
            try:
                float(value)
                return True
            except ValueError:
                return False

        if not is_number_or_blank(cash) or (cash and float(cash) <= 0):
            messagebox.showwarning("总资金格式不对", "总资金只能填大于 0 的数字。")
            return
        if not is_number_or_blank(target_position):
            messagebox.showwarning("总仓位格式不对", "目标总仓位只能填 0-100 的数字。")
            return
        if target_position:
            target = float(target_position)
            if target < 0 or target > 100:
                messagebox.showwarning("总仓位格式不对", "目标总仓位只能填 0-100。")
                return
        if not is_number_or_blank(shares):
            messagebox.showwarning("持仓格式不对", "持股数只能填数字，或留空。")
            return
        if not is_number_or_blank(cost):
            messagebox.showwarning("成本格式不对", "成本价只能填数字，或留空。")
            return

        self._save_ml_portfolio_settings()
        if symbol:
            selected = list(self.ml_saved_stock_tree.selection()) if hasattr(self, "ml_saved_stock_tree") else []
            self._save_ml_stock_position(symbol, shares, cost)
            self._render_saved_stock_picker()
            if self.ml_saved_stock_tree.exists(symbol):
                keep_selected = [item for item in selected if self.ml_saved_stock_tree.exists(item)]
                if keep_selected:
                    self.ml_saved_stock_tree.selection_set(keep_selected)
                else:
                    self.ml_saved_stock_tree.selection_set(symbol)
                self.ml_saved_stock_tree.focus(symbol)
            self.status_var.set(f"{symbol} 仓位和组合资金已保存；下次 ML 组合评估会使用这些真实仓位")
        else:
            self.status_var.set("总资金和目标总仓位已保存；右键左侧股票可填写单股持仓")

    def _validate_and_save_ml_account_settings(self) -> bool:
        cash = self.ml_cash.get().strip()
        target_position = self.ml_target_position.get().strip()
        parallel_workers = self.ml_parallel_workers.get().strip()

        def is_number_or_blank(value: str) -> bool:
            if not value:
                return True
            try:
                float(value)
                return True
            except ValueError:
                return False

        if not is_number_or_blank(cash) or (cash and float(cash) <= 0):
            messagebox.showwarning("总资金格式不对", "总资金只能填大于 0 的数字。")
            return False
        if not is_number_or_blank(target_position):
            messagebox.showwarning("总仓位格式不对", "目标总仓位只能填 0-100 的数字。")
            return False
        if target_position:
            target = float(target_position)
            if target < 0 or target > 100:
                messagebox.showwarning("总仓位格式不对", "目标总仓位只能填 0-100。")
                return False
        try:
            workers = int(parallel_workers or ML_BATCH_PARALLEL_WORKERS)
        except ValueError:
            messagebox.showwarning("并行数格式不对", "并行数只能填正整数，例如 1、2、3。")
            return False
        if workers < 1:
            messagebox.showwarning("并行数格式不对", "并行数至少为 1。")
            return False
        self.ml_parallel_workers.set(str(workers))
        self._save_ml_portfolio_settings()
        return True

    def _load_monitor_snapshot(self) -> None:
        if not MONITOR_SNAPSHOT_PATH.exists():
            return
        try:
            with MONITOR_SNAPSHOT_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            items = payload.get("items", {}) if isinstance(payload, dict) else {}
            if isinstance(items, dict):
                self.monitor_items.update({str(key): value for key, value in items.items() if isinstance(value, dict)})
        except Exception:
            return

    def _save_monitor_snapshot(self) -> None:
        try:
            MONITOR_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = MONITOR_SNAPSHOT_PATH.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump({"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "items": self.monitor_items}, handle, ensure_ascii=False)
            tmp_path.replace(MONITOR_SNAPSHOT_PATH)
        except Exception:
            return

    def _write_monitor_detail(self, text: str) -> None:
        if not hasattr(self, "monitor_detail_text"):
            return
        self.monitor_detail_text.configure(state=tk.NORMAL)
        self.monitor_detail_text.delete("1.0", "end")
        self.monitor_detail_text.insert("1.0", text)
        self.monitor_detail_text.configure(state=tk.DISABLED)

    def _show_monitor_error(self, text: str) -> None:
        self._write_monitor_detail(text)
        if hasattr(self, "monitor_canvas"):
            self.monitor_canvas.delete("all")
            self.monitor_canvas.create_text(
                self.monitor_canvas.winfo_width() / 2 or 360,
                self.monitor_canvas.winfo_height() / 2 or 220,
                text=text,
                fill="#b91c1c",
                font=("Microsoft YaHei", 13),
            )

    def _show_monitor_loading(self, symbol: str) -> None:
        name = self._saved_stock_name(symbol)
        self._write_monitor_detail(f"{symbol} {name} 正在拉取实时行情和分时曲线...")
        if hasattr(self, "monitor_canvas"):
            self.monitor_canvas.delete("all")
            self.monitor_canvas.create_text(
                self.monitor_canvas.winfo_width() / 2 or 360,
                self.monitor_canvas.winfo_height() / 2 or 220,
                text="正在拉取盘中监测曲线...",
                fill="#607086",
                font=("Microsoft YaHei", 13),
            )

    def _show_monitor_waiting(self, symbol: str) -> None:
        name = self._saved_stock_name(symbol)
        self._write_monitor_detail(f"{symbol} {name} 已选中。暂无盘中缓存，点击“刷新一次”后再拉取监测曲线。")
        if hasattr(self, "monitor_canvas"):
            self.monitor_canvas.delete("all")
            self.monitor_canvas.create_text(
                self.monitor_canvas.winfo_width() / 2 or 360,
                self.monitor_canvas.winfo_height() / 2 or 220,
                text="已选中，点击“刷新一次”拉取盘中曲线",
                fill="#607086",
                font=("Microsoft YaHei", 13),
            )

    def _render_monitor_detail(self, item: dict[str, Any] | None) -> None:
        if not item:
            self._write_monitor_detail("点击左侧股票后，会立即显示已有曲线并后台刷新当前股票；也可以点“刷新一次”立即更新。")
            return
        lines = [
            f"{item.get('symbol', '')} {item.get('name', '')} | {item.get('action', '')} | 价格 {item.get('price', '-')}",
            f"日线闸门：{item.get('daily_gate', '-')} | 分钟趋势：{item.get('minute_trend', '-')} | 量能比：{item.get('volume_ratio', '-')}",
            f"VWAP：{item.get('vwap', '-')} | 风控线：{item.get('stop_line', '-')} | 时间：{item.get('updated', '-')}",
            "",
            "判断依据：",
        ]
        lines.extend(f"- {reason}" for reason in item.get("reasons", []) or [])
        self._write_monitor_detail("\n".join(lines))
        self.monitor_xueqiu_var.set(f"{item.get('symbol', '')} {item.get('name', '')} | {item.get('xueqiu_url', '')}")

    def _selected_or_all_ml_monitor_symbols(self) -> list[str]:
        rows = self._saved_stock_rows(strategy_filter="ml")
        all_symbols = [str(row["symbol"]) for row in rows]
        if hasattr(self, "ml_monitor_saved_stock_tree"):
            selected = [symbol for symbol in self.ml_monitor_saved_stock_tree.selection() if symbol in all_symbols]
            if selected:
                return selected
        return all_symbols

    def _use_selected_ml_monitor_symbols(self) -> None:
        symbols = self._selected_or_all_ml_monitor_symbols()
        if not symbols:
            self.status_var.set("还没有保存的 ML 策略，先到 ML回测 跑一次")
            return
        lines = [f"{symbol}, {engine.stock_display_name(symbol)}" for symbol in symbols]
        self.ml_monitor_symbols.delete("1.0", "end")
        self.ml_monitor_symbols.insert("1.0", "\n".join(lines))
        self.status_var.set(f"已填入 {len(symbols)} 只 ML 策略股票")

    def _select_all_ml_monitor_stocks(self) -> None:
        if not hasattr(self, "ml_monitor_saved_stock_tree"):
            return
        children = self.ml_monitor_saved_stock_tree.get_children()
        if children:
            self.ml_monitor_saved_stock_tree.selection_set(children)
            self.status_var.set(f"已选择 {len(children)} 只 ML 策略股票")

    def _on_ml_saved_stock_select(self, _event: object | None = None) -> None:
        selection = self.ml_saved_stock_tree.selection()
        if not selection:
            return
        symbol = selection[0]
        self.ml_symbol.set(symbol)
        name = self._ml_stock_name(symbol)
        position = self._ml_stock_position(symbol)
        self.ml_position_symbol.set(f"{symbol} {name}".strip())
        self.ml_shares.set(str(position.get("shares", "")))
        self.ml_buy_price.set(str(position.get("cost", "")))
        self.status_var.set(f"ML 板块已选择股票：{symbol} {name}；右键可填写持股/成本或删除")

    def _on_ml_saved_stock_click(self, event: tk.Event) -> str | None:
        row_id = self.ml_saved_stock_tree.identify_row(event.y)
        if not row_id:
            return None
        column_name = self._tree_column_name(self.ml_saved_stock_tree, self.ml_saved_stock_tree.identify_column(event.x))
        if column_name != "checked":
            return None
        symbol = str(row_id)
        if symbol in self.ml_checked_symbols:
            self.ml_checked_symbols.remove(symbol)
            state = "取消勾选"
        else:
            self.ml_checked_symbols.add(symbol)
            state = "已勾选"
        selected = list(self.ml_saved_stock_tree.selection())
        focused = self.ml_saved_stock_tree.focus()
        self._render_saved_stock_picker()
        keep_selected = [item for item in selected if self.ml_saved_stock_tree.exists(item)]
        if keep_selected:
            self.ml_saved_stock_tree.selection_set(keep_selected)
        if focused and self.ml_saved_stock_tree.exists(focused):
            self.ml_saved_stock_tree.focus(focused)
        self.status_var.set(f"ML评估范围：{symbol} {state}")
        return "break"

    def _render_strategy_cache_list(self) -> None:
        if not hasattr(self, "cache_tree"):
            return
        engine.PERSISTENT_STRATEGY_CACHE = None
        for iid in self.cache_tree.get_children():
            self.cache_tree.delete(iid)
        cache = engine.load_persistent_strategy_cache()
        cache_changed = False
        for record in cache.values():
            if isinstance(record, dict) and bool(record.get("selected_for_left")) and not bool(record.get("active_for_trading")):
                record["active_for_trading"] = True
                cache_changed = True
        if cache_changed:
            engine.save_persistent_strategy_cache()
        grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for key_text, record in cache.items():
            if isinstance(record, dict):
                if not self._record_matches_strategy_filter(record, exclude_strategy_type="ml"):
                    continue
                try:
                    code = engine.normalize_symbol(str(record.get("symbol", "")))
                except Exception:
                    code = str(record.get("symbol", ""))
                if code:
                    grouped.setdefault(code, []).append((key_text, record))
        parent_rows: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = []
        for code, rows in grouped.items():
            rows.sort(key=lambda item: str(item[1].get("saved_at", "")), reverse=True)
            rows.sort(key=lambda item: 0 if bool(item[1].get("active_for_trading")) else 1)
            parent_rows.append((code, rows))
        existing_symbols = {code for code, _rows in parent_rows}
        self.backtest_checked_symbols.intersection_update(existing_symbols)
        parent_rows.sort(key=lambda item: str(item[1][0][1].get("saved_at", "")) if item[1] else "", reverse=True)
        for code, rows in parent_rows:
            visible_rows = [item for item in rows if bool(item[1].get("selected_for_left"))]
            active_candidates = [item for item in visible_rows if bool(item[1].get("active_for_trading"))]
            active = active_candidates[0][1] if active_candidates else (visible_rows[0][1] if visible_rows else rows[0][1])
            latest = str(rows[0][1].get("saved_at", "")) if rows else ""
            name = str(active.get("name") or engine.stock_display_name(code))
            active_strategy = self._strategy_display_name(active_candidates[0][1]) if active_candidates else ""
            parent_iid = self._cache_stock_iid(code)
            check = "☑" if code in self.backtest_checked_symbols else "☐"
            self.cache_tree.insert(
                "",
                "end",
                iid=parent_iid,
                text=f"{check} {code} {name}".strip(),
                values=(code, name, f"已选 {len(visible_rows)} / 全部 {len(rows)}", active_strategy, latest[:10]),
                open=False,
            )
            for key_text, record in visible_rows:
                params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
                strategy = self._strategy_display_name(record)
                marker = "★ " if bool(record.get("active_for_trading")) else ""
                result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
                signal = result.get("daily_signal", {}) if isinstance(result.get("daily_signal"), dict) else {}
                saved_at = str(record.get("saved_at", ""))
                values = (
                    "",
                    "",
                    params.get("mode", ""),
                    strategy,
                    signal.get("date", "") or saved_at[:10],
                )
                self.cache_tree.insert(parent_iid, "end", iid=key_text, text=f"{marker}{strategy}", values=values)
        self._render_saved_stock_picker()
        self._render_monitor_strategy_list()

    def _saved_strategy_rows_for_symbol(
        self,
        symbol: str | None,
        strategy_filter: str | None = None,
        exclude_strategy_type: str | None = None,
    ) -> list[tuple[str, tuple[str, str, str]]]:
        if not symbol:
            return []
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol)
        rows: list[tuple[str, dict[str, Any]]] = []
        for key_text, record in engine.load_persistent_strategy_cache().items():
            if not isinstance(record, dict):
                continue
            if engine.normalize_symbol(str(record.get("symbol", ""))) != code:
                continue
            if not self._record_matches_strategy_filter(record, strategy_filter, exclude_strategy_type):
                continue
            if not bool(record.get("selected_for_left")):
                continue
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            signal = result.get("daily_signal", {}) if isinstance(result.get("daily_signal"), dict) else {}
            if not signal:
                continue
            rows.append((key_text, record))
        rows.sort(key=lambda item: str(item[1].get("saved_at", "")), reverse=True)
        rows.sort(key=lambda item: 0 if bool(item[1].get("active_for_trading")) else 1)

        output: list[tuple[str, tuple[str, str, str]]] = []
        for key_text, record in rows:
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            signal = result.get("daily_signal", {}) if isinstance(result.get("daily_signal"), dict) else {}
            params = record.get("params", {}) if isinstance(record.get("params"), dict) else {}
            strategy = f"{signal.get('strategy_label', result.get('strategy_label', ''))} {signal.get('fast', '')}/{signal.get('slow', '')}".strip()
            if bool(record.get("active_for_trading")):
                strategy = f"★ {strategy}"
            output.append((key_text, (strategy, str(params.get("mode", "")), str(record.get("saved_at", "")))))
        return output

    def _render_monitor_strategy_list(self, symbol: str | None = None) -> None:
        if not hasattr(self, "monitor_strategy_tree"):
            return
        if symbol is None:
            symbol = self.selected_monitor_symbol
        for iid in self.monitor_strategy_tree.get_children():
            self.monitor_strategy_tree.delete(iid)
        rows = self._saved_strategy_rows_for_symbol(symbol, exclude_strategy_type="ml")
        if not rows:
            return
        code = engine.normalize_symbol(str(symbol))
        selected_key = self.monitor_strategy_keys.get(code)
        for key_text, values in rows:
            self.monitor_strategy_tree.insert("", "end", iid=key_text, values=values)
        if not selected_key or selected_key not in {key for key, _ in rows}:
            selected_key = rows[0][0]
            self.monitor_strategy_keys[code] = selected_key
        self.monitor_strategy_tree.selection_set(selected_key)
        self.monitor_strategy_tree.focus(selected_key)

    def _on_monitor_strategy_select(self, _event: object | None = None) -> None:
        symbol = self.selected_monitor_symbol
        selection = self.monitor_strategy_tree.selection()
        if not symbol or not selection:
            return
        code = engine.normalize_symbol(symbol)
        self.monitor_strategy_keys[code] = selection[0]
        self._set_active_strategy_for_symbol(selection[0])
        values = self.monitor_strategy_tree.item(selection[0], "values")
        strategy_name = values[0] if values else "选中策略"
        self.status_var.set(f"{code} 盘中监控已切换为：{strategy_name}")

    def _cache_selected_iid(self) -> str:
        selection = self.cache_tree.selection() if hasattr(self, "cache_tree") else ()
        return str(selection[0]) if selection else ""

    def _cache_symbol_from_iid(self, iid: str) -> str:
        if iid.startswith("stock:"):
            return self._cache_iid_symbol(iid)
        record = engine.load_persistent_strategy_cache().get(iid)
        if isinstance(record, dict):
            try:
                return engine.normalize_symbol(str(record.get("symbol", "")))
            except Exception:
                return str(record.get("symbol", ""))
        parent = self.cache_tree.parent(iid) if hasattr(self, "cache_tree") and self.cache_tree.exists(iid) else ""
        return self._cache_iid_symbol(parent) if parent.startswith("stock:") else ""

    def _toggle_cache_stock_check(self, symbol: str) -> None:
        try:
            code = engine.normalize_symbol(symbol)
        except Exception:
            code = str(symbol)
        if not code:
            return
        if code in self.backtest_checked_symbols:
            self.backtest_checked_symbols.remove(code)
            state = "取消勾选"
        else:
            self.backtest_checked_symbols.add(code)
            state = "已勾选"
        self._render_strategy_cache_list()
        iid = self._cache_stock_iid(code)
        if self.cache_tree.exists(iid):
            self.cache_tree.selection_set(iid)
            self.cache_tree.focus(iid)
        self.status_var.set(f"{code} {state}，用于“回测勾选股票”")

    def _toggle_selected_cache_stock_check(self) -> None:
        iid = self._cache_selected_iid()
        symbol = self._cache_symbol_from_iid(iid)
        if not symbol:
            self.status_var.set("请先在左侧选择一只股票")
            return
        self._toggle_cache_stock_check(symbol)

    def _selected_cache_stock_symbols(self) -> list[str]:
        if not hasattr(self, "cache_tree"):
            return []
        symbols: list[str] = []
        seen: set[str] = set()
        for iid in self.cache_tree.selection():
            symbol = self._cache_symbol_from_iid(str(iid))
            if symbol and symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)
        return symbols

    def _set_selected_cache_stock_checks(self, checked: bool) -> None:
        symbols = self._selected_cache_stock_symbols()
        if not symbols:
            self.status_var.set("请先在左侧选中一只或多只股票")
            return
        if checked:
            self.backtest_checked_symbols.update(symbols)
            action = "勾选"
        else:
            self.backtest_checked_symbols.difference_update(symbols)
            action = "取消勾选"
        selected_iids = list(self.cache_tree.selection())
        focused = self.cache_tree.focus()
        self._render_strategy_cache_list()
        keep_selected = [iid for iid in selected_iids if self.cache_tree.exists(iid)]
        if keep_selected:
            self.cache_tree.selection_set(keep_selected)
        if focused and self.cache_tree.exists(focused):
            self.cache_tree.focus(focused)
        self.status_var.set(f"已{action}选中的 {len(symbols)} 只股票")

    def _check_all_cache_stocks(self) -> None:
        symbols = self._all_strategy_cache_symbols()
        if not symbols:
            self.status_var.set("还没有可勾选的保存股票")
            return
        self.backtest_checked_symbols.update(symbols)
        self._render_strategy_cache_list()
        self.status_var.set(f"已全部勾选 {len(symbols)} 只股票，可点击“回测勾选股票”")

    def _uncheck_all_cache_stocks(self) -> None:
        count = len(self.backtest_checked_symbols)
        self.backtest_checked_symbols.clear()
        self._render_strategy_cache_list()
        self.status_var.set(f"已取消全部勾选，原勾选 {count} 只")

    def _on_cache_tree_click(self, event: tk.Event) -> str | None:
        row_id = self.cache_tree.identify_row(event.y)
        if not row_id or not row_id.startswith("stock:"):
            return None
        if self.cache_tree.identify_column(event.x) != "#0":
            return None
        try:
            element = str(self.cache_tree.identify_element(event.x, event.y)).lower()
        except Exception:
            element = ""
        if "indicator" in element:
            return None
        bbox = self.cache_tree.bbox(row_id, "#0")
        if bbox and bbox[0] + 18 <= event.x <= bbox[0] + 48:
            self.cache_tree.selection_set(row_id)
            self.cache_tree.focus(row_id)
            self._toggle_cache_stock_check(self._cache_iid_symbol(row_id))
            return "break"
        return None

    def _show_cache_context_menu(self, event: tk.Event) -> str | None:
        row_id = self.cache_tree.identify_row(event.y)
        if not row_id:
            try:
                self.cache_bulk_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.cache_bulk_context_menu.grab_release()
            return "break"
        if row_id not in self.cache_tree.selection():
            self.cache_tree.selection_set(row_id)
        self.cache_tree.focus(row_id)
        try:
            if row_id.startswith("stock:"):
                self.cache_stock_context_menu.tk_popup(event.x_root, event.y_root)
            else:
                self.cache_strategy_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            if row_id.startswith("stock:"):
                self.cache_stock_context_menu.grab_release()
            else:
                self.cache_strategy_context_menu.grab_release()
        return "break"

    def _load_left_cache_strategy_preview(self) -> None:
        key_text = self._cache_selected_iid()
        if not key_text or key_text.startswith("stock:"):
            self.status_var.set("请先在左侧选择一条具体策略")
            return
        self._start_saved_strategy_preview(key_text)

    def _on_cache_select(self, _event: object | None = None) -> None:
        selection = self.cache_tree.selection()
        if not selection:
            return
        key_text = selection[0]
        if key_text.startswith("stock:"):
            symbol = self._cache_iid_symbol(key_text)
            self._show_saved_stock_strategies(symbol)
            return
        try:
            key = json.loads(key_text)
        except Exception:
            return
        if not isinstance(key, list) or len(key) < 6:
            return
        symbol, start, adjust, cash, fee, mode = key[:6]
        parts = str(mode).split(":")
        horizon = parts[0] if len(parts) > 0 else "short"
        strategy = parts[1] if len(parts) > 1 else "auto_fast"
        risk = parts[2] if len(parts) > 2 else "normal"
        self.bt_symbol.set(str(symbol))
        self.bt_start.set(str(start))
        self.bt_adjust.set(str(adjust))
        self.bt_cash.set(str(cash))
        self.bt_fee.set(str(fee))
        self.bt_horizon.set(str(horizon))
        self.bt_strategy.set(str(strategy))
        self.bt_risk.set(str(risk))
        if str(strategy) == "ml" and hasattr(self, "ml_symbol"):
            self.ml_symbol.set(str(symbol))
            self.ml_start.set(str(start))
            self.ml_adjust.set(str(adjust))
            self.ml_cash.set(str(cash))
            self.ml_fee.set(str(fee))
            self.ml_horizon.set(str(horizon))
            self.ml_risk.set(str(risk))
            self.notebook.select(self.ml_tab)
        if str(strategy) != "ml":
            self._set_active_strategy_for_symbol(key_text)
            self._start_saved_strategy_preview(key_text)
        self.status_var.set(f"已选择并载入策略：{symbol} {mode}，盘中监控和 ML 会优先使用它")

    def _show_saved_stock_strategies(self, symbol: str) -> None:
        rows = self._saved_records_for_symbol(symbol, exclude_strategy_type="ml")
        for iid in self.bt_tree.get_children():
            self.bt_tree.delete(iid)
        if not rows:
            self.summary_var.set(f"{symbol} 暂无保存策略")
            self.status_var.set("左侧展开股票后，点击具体保存策略可加载曲线")
            return
        for rank, (key_text, record) in enumerate(rows, start=1):
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            signal = result.get("daily_signal", {}) if isinstance(result.get("daily_signal"), dict) else {}
            best = result.get("best") if isinstance(result.get("best"), dict) else {}
            strategy = self._strategy_display_name(record)
            values = (
                rank,
                ("★ " if record.get("active_for_trading") else "") + strategy,
                f"{signal.get('fast', '')}/{signal.get('slow', '')}",
                self._fmt_number(best.get("total_return_pct"), 2),
                self._fmt_number(best.get("max_drawdown_pct"), 2),
                self._fmt_number(best.get("sharpe"), 2),
                int(float(best.get("trades", 0) or 0)),
                self._fmt_number(best.get("final_value"), 2),
                self._fmt_number(best.get("score"), 2),
            )
            self.bt_tree.insert("", "end", iid=f"saved:{rank}", values=values, tags=(key_text,))
        active_key = rows[0][0]
        self.summary_var.set(f"{symbol} 已保存 {len(rows)} 条策略；带 ★ 的策略会进入盘中监控和 ML")
        self.status_var.set("点击左侧某条具体策略可加载曲线；删除按钮可删单条策略或整只股票")
        self.cache_preview_key = active_key

    def _start_saved_strategy_preview(self, key_text: str) -> None:
        self.cache_preview_key = key_text
        self.summary_var.set("正在读取保存策略并刷新右侧曲线...")
        self.status_var.set("正在用保存策略生成右侧预览，不会重新扫描全部策略")
        self.cache_preview_worker = threading.Thread(target=self._saved_strategy_preview_worker, args=(key_text,), daemon=True)
        self.cache_preview_worker.start()

    def _saved_strategy_preview_worker(self, key_text: str) -> None:
        try:
            result = _compute_saved_strategy_preview_payload(key_text)
            self.queue.put(WorkerMessage("cache_preview", payload={"key_text": key_text, "result": result}))
        except Exception:
            self.queue.put(WorkerMessage("cache_preview_error", payload={"key_text": key_text}, error=traceback.format_exc()))

    def _monitor_symbols_text(self) -> str:
        return "\n".join(self._monitor_enabled_symbols())

    def _monitor_interval_seconds(self) -> int:
        try:
            return max(10, int(float(self.monitor_interval.get() or 30)))
        except ValueError:
            return 30

    def _auto_monitor_tick(self) -> None:
        self.after(self._monitor_interval_seconds() * 1000, self._auto_monitor_tick)

    def toggle_monitor(self) -> None:
        self.monitor_running = not self.monitor_running
        if hasattr(self, "monitor_start_button"):
            self.monitor_start_button.configure(text="停止监控" if self.monitor_running else "开始监控")
        if self.monitor_running:
            self._start_monitor_worker(loop=True)

    def refresh_monitor_once(self) -> None:
        if not self.selected_monitor_symbol:
            self.status_var.set("请先在左侧点选一只股票")
            return
        self._show_monitor_loading(str(self.selected_monitor_symbol))
        self._start_monitor_worker(loop=False, symbols=[str(self.selected_monitor_symbol)])

    def refresh_monitor_symbol(self, symbol: str) -> None:
        self.monitor_last_symbol_refresh[str(symbol)] = time.time()
        self._start_monitor_worker(loop=False, symbols=[symbol])

    def refresh_monitor_symbol_with_loading(self, symbol: str) -> None:
        self.selected_monitor_symbol = str(symbol)
        self._show_monitor_loading(str(symbol))
        self.refresh_monitor_symbol(str(symbol))

    def _start_monitor_worker(self, loop: bool, symbols: list[str] | None = None) -> None:
        if self.monitor_worker and self.monitor_worker.is_alive():
            self.monitor_refresh_pending = True
            if symbols:
                self.monitor_pending_symbols = {str(symbol) for symbol in symbols}
            self.status_var.set("上一轮监控刷新尚未完成，已排队下一轮刷新")
            return
        tasks = self._build_monitor_tasks(symbols)
        interval = self._monitor_interval_seconds()
        self.monitor_worker = threading.Thread(target=self._monitor_worker_loop, args=(loop, tasks, interval), daemon=True)
        self.monitor_worker.start()

    def _build_monitor_tasks(self, symbols: list[str] | None = None) -> list[dict[str, str]]:
        if symbols is None:
            symbols = [str(self.selected_monitor_symbol)] if self.selected_monitor_symbol else []
        period = self.monitor_period.get()
        tasks: list[dict[str, str]] = []
        for symbol in symbols:
            try:
                code = engine.resolve_stock_identifier(symbol)
            except Exception:
                code = str(symbol)
            strategy_key = self.monitor_strategy_keys.get(code, "")
            if not strategy_key:
                rows = self._saved_strategy_rows_for_symbol(code, exclude_strategy_type="ml")
                if rows:
                    strategy_key = rows[0][0]
                    self.monitor_strategy_keys[code] = strategy_key
            if not strategy_key:
                continue
            position = self._stock_position(code)
            tasks.append(
                {
                    "symbol": str(symbol),
                    "period": str(period),
                    "shares": str(position.get("shares", "")),
                    "cost": str(position.get("cost", "")),
                    "strategy_key": str(strategy_key),
                }
            )
        return tasks

    def _run_pending_monitor_refresh(self) -> None:
        if not self.monitor_refresh_pending:
            return
        if self.monitor_worker and self.monitor_worker.is_alive():
            self.after(200, self._run_pending_monitor_refresh)
            return
        self.monitor_refresh_pending = False
        pending_symbols = sorted(self.monitor_pending_symbols)
        self.monitor_pending_symbols.clear()
        symbols = pending_symbols or ([str(self.selected_monitor_symbol)] if self.selected_monitor_symbol else [])
        if symbols:
            self._start_monitor_worker(loop=False, symbols=symbols)

    def _monitor_worker_loop(self, loop: bool, tasks: list[dict[str, str]], interval: int) -> None:
        while True:
            self._fetch_monitor_once(tasks)
            if not loop or not self.monitor_running:
                break
            time.sleep(interval)

    def _fetch_monitor_once(self, tasks: list[dict[str, str]]) -> None:
        if not tasks:
            self.queue.put(WorkerMessage("monitor_error", error="请先在左侧点选一只股票"))
            return

        results: list[dict[str, Any]] = []
        for task in tasks:
            symbol = task["symbol"]
            try:
                results.append(
                    engine.build_monitor_item(
                        symbol,
                        task.get("period", "5"),
                        task.get("shares", ""),
                        task.get("cost", ""),
                        task.get("strategy_key", ""),
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "symbol": symbol,
                        "name": engine.stock_display_name(symbol),
                        "xueqiu_url": engine.xueqiu_url(symbol),
                        "action": "错误",
                        "action_code": "watch",
                        "price": "-",
                        "daily_gate": "-",
                        "minute_trend": "-",
                        "volume_ratio": "-",
                        "vwap": "-",
                        "stop_line": "-",
                        "updated": "-",
                        "chart_points": [],
                        "reasons": [str(exc)],
                    }
                )
        self.queue.put(WorkerMessage("monitor", payload=results))

    def _ml_monitor_symbols_text(self) -> str:
        return self.ml_monitor_symbols.get("1.0", "end").strip()

    def _ml_monitor_interval_seconds(self) -> int:
        try:
            return max(10, int(float(self.ml_monitor_interval.get() or 30)))
        except ValueError:
            return 30

    def toggle_ml_monitor(self) -> None:
        self.ml_monitor_running = not self.ml_monitor_running
        self.ml_monitor_start_button.configure(text="停止ML监控" if self.ml_monitor_running else "开始ML监控")
        if self.ml_monitor_running:
            self._start_ml_monitor_worker(loop=True)

    def refresh_ml_monitor_once(self) -> None:
        self._start_ml_monitor_worker(loop=False)

    def _start_ml_monitor_worker(self, loop: bool) -> None:
        if self.ml_monitor_worker and self.ml_monitor_worker.is_alive():
            self.status_var.set("上一轮 ML 监控刷新尚未完成")
            return
        tasks = self._build_ml_monitor_tasks()
        interval = self._ml_monitor_interval_seconds()
        self.ml_monitor_worker = threading.Thread(target=self._ml_monitor_worker_loop, args=(loop, tasks, interval), daemon=True)
        self.ml_monitor_worker.start()

    def _build_ml_monitor_tasks(self) -> list[dict[str, str]]:
        try:
            symbols = engine.parse_symbol_text(self._ml_monitor_symbols_text())
        except Exception as exc:
            self.queue.put(WorkerMessage("ml_monitor_error", error=str(exc)))
            return []
        period = self.ml_monitor_period.get()
        shares = self.ml_monitor_shares.get()
        buy_price = self.ml_monitor_buy_price.get()
        tasks: list[dict[str, str]] = []
        for symbol in symbols:
            try:
                code = engine.resolve_stock_identifier(symbol)
            except Exception:
                code = str(symbol)
            tasks.append(
                {
                    "symbol": str(symbol),
                    "period": str(period),
                    "shares": str(shares),
                    "buy_price": str(buy_price),
                    "strategy_key": str(self.ml_monitor_strategy_keys.get(code, "")),
                }
            )
        return tasks

    def _ml_monitor_worker_loop(self, loop: bool, tasks: list[dict[str, str]], interval: int) -> None:
        while True:
            self._fetch_ml_monitor_once(tasks)
            if not loop or not self.ml_monitor_running:
                break
            time.sleep(interval)

    def _fetch_ml_monitor_once(self, tasks: list[dict[str, str]]) -> None:
        results: list[dict[str, Any]] = []
        for task in tasks:
            symbol = task["symbol"]
            try:
                results.append(
                    engine.build_monitor_item(
                        symbol,
                        task.get("period", "5"),
                        task.get("shares", ""),
                        task.get("buy_price", ""),
                        task.get("strategy_key", ""),
                        strategy_type="ml",
                        exclude_strategy_type=None,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "symbol": symbol,
                        "name": engine.stock_display_name(symbol),
                        "xueqiu_url": engine.xueqiu_url(symbol),
                        "action": "错误",
                        "action_code": "watch",
                        "price": "-",
                        "daily_gate": "-",
                        "minute_trend": "-",
                        "volume_ratio": "-",
                        "vwap": "-",
                        "stop_line": "-",
                        "updated": "-",
                        "chart_points": [],
                        "chart_strategy_type": "ml",
                        "chart_strategy_label": "ML",
                        "reasons": [str(exc)],
                    }
                )
        self.queue.put(WorkerMessage("ml_monitor", payload=results))

    def run_backtest(self) -> None:
        if self.backtest_worker and self.backtest_worker.is_alive():
            self.status_var.set("回测仍在运行，请稍等")
            return
        self.backtest_stop_event.clear()
        self.backtest_target = "traditional"
        self.pending_backtest_form = self._backtest_form()
        self.pending_backtest_form["_save_all_strategies"] = "1"
        self._set_backtest_running(True)
        strategy = self.bt_strategy.get()
        if strategy in {"auto", "ml"}:
            self.summary_var.set("回测中：auto/ML 会比较慢，可能需要 30-90 秒...")
        else:
            self.summary_var.set("回测中：正在拉取数据并扫描策略...")
        self.status_var.set("回测运行中，请稍等")
        self.backtest_worker = threading.Thread(target=self._backtest_worker, daemon=True)
        self.backtest_worker.start()

    def _start_saved_stock_backtests(self, symbols: list[str], source_label: str) -> None:
        if self.backtest_worker and self.backtest_worker.is_alive():
            self.status_var.set("回测仍在运行，请稍等")
            return
        if not symbols:
            self.status_var.set("没有可回测的保存股票")
            return
        self.backtest_stop_event.clear()
        self.backtest_target = "traditional"
        self.pending_backtest_form = self._backtest_form()
        self.pending_backtest_form["_save_strategy"] = "1"
        self.pending_backtest_form["_save_all_strategies"] = "1"
        self._set_backtest_running(True)
        self.summary_var.set(f"{source_label}：准备回测 {len(symbols)} 只已保存股票...")
        self.status_var.set("批量回测运行中，请稍等")
        self.backtest_worker = threading.Thread(target=self._backtest_batch_worker, args=(symbols,), daemon=True)
        self.backtest_worker.start()

    def run_saved_stock_backtests(self) -> None:
        self.run_checked_saved_stock_backtests()

    def run_checked_saved_stock_backtests(self) -> None:
        symbols = self._checked_strategy_cache_symbols()
        if not symbols:
            self.status_var.set("请先在左侧股票名前勾选要回测的股票")
            return
        self._start_saved_stock_backtests(symbols, "勾选股票回测")

    def run_all_saved_stock_backtests(self) -> None:
        symbols = self._all_strategy_cache_symbols()
        if not symbols:
            self.status_var.set("还没有已保存股票，先单只回测并保存一个策略")
            return
        self._start_saved_stock_backtests(symbols, "全部股票回测")

    def run_selected_cache_stock_backtest(self) -> None:
        symbol = self._cache_symbol_from_iid(self._cache_selected_iid())
        if not symbol:
            self.status_var.set("请先在左侧选择一只股票")
            return
        self._start_saved_stock_backtests([symbol], f"{symbol} 单股回测")

    def run_input_stock_backtests(self) -> None:
        if self.backtest_worker and self.backtest_worker.is_alive():
            self.status_var.set("回测仍在运行，请稍等")
            return
        symbols = self._batch_input_symbols()
        if not symbols:
            self.status_var.set("请先在左侧批量股票代码框里输入代码")
            return
        self.backtest_stop_event.clear()
        self.backtest_target = "traditional"
        form = self._backtest_form()
        form["batch_symbols"] = " ".join(symbols)
        form["_save_strategy"] = "1"
        form["_save_all_strategies"] = "1"
        self.pending_backtest_form = form
        self._set_backtest_running(True)
        self.summary_var.set(f"批量回测中：准备回测 {len(symbols)} 只输入股票，成功后自动保存最佳策略...")
        self.status_var.set("批量代码回测运行中，请稍等")
        self.backtest_worker = threading.Thread(target=self._backtest_batch_worker, args=(symbols,), daemon=True)
        self.backtest_worker.start()

    def _copy_ml_form_to_backtest(self) -> None:
        self.bt_symbol.set(self.ml_symbol.get().strip())
        self.bt_start.set(self.ml_start.get().strip() or "20200101")
        self.bt_adjust.set(self.ml_adjust.get())
        self.bt_cash.set(self.ml_cash.get().strip() or "100000")
        self.bt_fee.set(self.ml_fee.get().strip() or "0.0003")
        self.bt_risk.set(self.ml_risk.get())
        self.bt_horizon.set(self.ml_horizon.get())
        self.bt_strategy.set("ml")
        self.bt_shares.set(self.ml_shares.get().strip() or "0")
        self.bt_buy_price.set(self.ml_buy_price.get().strip())
        self.bt_buy_date.set(self.ml_buy_date.get().strip())

    def run_ml_backtest(self) -> None:
        if self.backtest_worker and self.backtest_worker.is_alive():
            self.status_var.set("ML持仓决策仍在运行，请稍等")
            return
        form = self._ml_backtest_form()
        self.backtest_stop_event.clear()
        self.backtest_target = "ml"
        self.pending_backtest_form = form
        self._set_backtest_running(True)
        self.ml_summary_var.set("ML持仓决策中：正在计算次日概率、风险和仓位建议，必要时可点终止")
        self.status_var.set("ML持仓决策运行中，请稍等")
        self.backtest_worker = threading.Thread(target=self._backtest_worker, daemon=True)
        self.backtest_worker.start()

    def _start_saved_stock_ml_backtests(self, symbols: list[str], source_label: str) -> None:
        _ui_debug(f"ml_start_enter source={source_label} count={len(symbols)}")
        if self.backtest_worker and self.backtest_worker.is_alive():
            _ui_debug("ml_start_blocked_existing_worker")
            self.status_var.set("ML持仓决策仍在运行，请稍等")
            return
        if not symbols:
            _ui_debug("ml_start_no_symbols")
            self.status_var.set("没有可评估的 ML 股票")
            return
        if not self._validate_and_save_ml_account_settings():
            _ui_debug("ml_start_validate_failed")
            return
        self.backtest_stop_event.clear()
        self.backtest_target = "ml"
        _ui_debug("ml_start_build_form")
        form = self._ml_backtest_form()
        form["batch_symbols"] = " ".join(symbols)
        _ui_debug("ml_start_build_positions")
        form["positions"] = self._ml_positions_for_symbols(symbols)
        self.pending_backtest_form = form
        self._set_backtest_running(True)
        external_mode = "刷新外部数据" if form.get("refresh_external_data") == "1" else "复用外部数据缓存"
        self.ml_summary_var.set(f"{source_label}：准备评估 {len(symbols)} 只股票，{external_mode}...")
        self.status_var.set("ML持仓决策运行中，请稍等")
        _ui_debug("ml_start_schedule_worker")
        self.after(50, lambda stock_symbols=list(symbols): self._launch_ml_batch_worker(stock_symbols))

    def _launch_ml_batch_worker(self, symbols: list[str]) -> None:
        _ui_debug(f"ml_launch_worker count={len(symbols)}")
        if self.backtest_worker and self.backtest_worker.is_alive():
            _ui_debug("ml_launch_worker_already_running")
            return
        self.backtest_worker = threading.Thread(target=self._backtest_batch_worker, args=(symbols,), daemon=True)
        self.backtest_worker.start()
        _ui_debug("ml_launch_worker_started")

    def run_saved_stock_ml_backtests(self) -> None:
        self.run_selected_saved_stock_ml_backtests()

    def run_selected_saved_stock_ml_backtests(self) -> None:
        _ui_debug("ml_run_selected_enter")
        symbols = self._selected_ml_saved_symbols()
        _ui_debug(f"ml_run_selected_symbols count={len(symbols)}")
        if not symbols:
            self.status_var.set("请先在 ML 股票池的“选择”列勾选要评估的股票")
            return
        self._start_saved_stock_ml_backtests(symbols, "ML勾选股票评估")

    def run_all_saved_stock_ml_backtests(self) -> None:
        _ui_debug("ml_run_all_enter")
        symbols = self._all_ml_saved_symbols_fast()
        _ui_debug(f"ml_run_all_symbols count={len(symbols)}")
        if not symbols:
            self.status_var.set("ML 股票池还没有股票，先在左侧批量加入代码")
            return
        self._start_saved_stock_ml_backtests(symbols, "ML全部股票池评估")

    def _ml_backtest_form(self) -> dict[str, str]:
        return {
            "symbol": self.ml_symbol.get().strip(),
            "start": self.ml_start.get().strip() or "20200101",
            "adjust": self.ml_adjust.get(),
            "cash": self.ml_cash.get().strip() or "100000",
            "fee": self.ml_fee.get().strip() or "0.0003",
            "risk": self.ml_risk.get(),
            "horizon": self.ml_horizon.get(),
            "strategy_type": "ml",
            "_job": "ml_predict",
            "shares": self.ml_shares.get().strip() or "0",
            "buy_price": self.ml_buy_price.get().strip(),
            "buy_date": self.ml_buy_date.get().strip(),
            "batch_symbols": "",
            "target_position": self.ml_target_position.get().strip() or "80",
            "positions": {},
            "refresh_external_data": "1" if self.ml_refresh_external_data.get() else "0",
            "_parallel_workers": self.ml_parallel_workers.get().strip() or str(ML_BATCH_PARALLEL_WORKERS),
        }

    def _ml_positions_for_symbols(self, symbols: list[str]) -> dict[str, dict[str, str]]:
        positions: dict[str, dict[str, str]] = {}
        for symbol in symbols:
            position = self._ml_stock_position(symbol)
            positions[symbol] = {
                "shares": str(position.get("shares", "")),
                "cost": str(position.get("cost", "")),
                "name": self._ml_stock_cached_name(symbol),
            }
        return positions

    def _selected_ml_saved_symbols(self) -> list[str]:
        return [symbol for symbol in self._all_ml_saved_symbols_fast() if symbol in self.ml_checked_symbols]

    def _all_ml_saved_symbols(self) -> list[str]:
        rows = self._ml_stock_pool_rows()
        return [str(row["symbol"]) for row in rows]

    def _all_ml_saved_symbols_fast(self) -> list[str]:
        rows: list[dict[str, Any]] = []
        for symbol, item in self._load_ml_stock_pool().items():
            try:
                code = engine.normalize_symbol(symbol)
            except Exception:
                code = str(symbol).strip()
            if not code:
                continue
            try:
                shares_count = int(float(str(item.get("shares") or "") or 0))
            except Exception:
                shares_count = 0
            rows.append({"symbol": code, "shares_count": shares_count, "added_at": str(item.get("added_at") or "")})
        rows.sort(key=lambda row: str(row.get("added_at", "")), reverse=True)
        rows.sort(key=lambda row: 0 if int(row.get("shares_count") or 0) > 0 else 1)
        return [str(row["symbol"]) for row in rows]

    def _held_saved_symbols(self) -> list[str]:
        held: list[str] = []
        for row in self._saved_stock_rows(exclude_strategy_type="ml", selected_for_left_only=True):
            symbol = str(row["symbol"])
            position = self._stock_position(symbol)
            try:
                shares = int(float(position.get("shares", "") or 0))
            except Exception:
                shares = 0
            if shares > 0:
                held.append(symbol)
        return held

    def _set_backtest_running(self, running: bool) -> None:
        if hasattr(self, "stop_backtest_button"):
            self.stop_backtest_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        if hasattr(self, "ml_stop_backtest_button"):
            self.ml_stop_backtest_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def stop_backtest(self) -> None:
        process = self.backtest_process
        with self.backtest_process_lock:
            processes = list(self.backtest_processes)
        worker_alive = bool(self.backtest_worker and self.backtest_worker.is_alive())
        if hasattr(process, "is_alive"):
            process_alive = bool(process and process.is_alive())
        elif hasattr(process, "poll"):
            process_alive = bool(process and process.poll() is None)
        else:
            process_alive = False
        any_parallel_alive = any(proc.poll() is None for proc in processes if hasattr(proc, "poll"))
        if not worker_alive and not process_alive and not any_parallel_alive:
            self.status_var.set("当前没有正在运行的回测")
            self._set_backtest_running(False)
            return
        self.backtest_stop_event.set()
        if process_alive and process is not None:
            process.terminate()
        for proc in processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        self.status_var.set("正在终止回测...")

    def _backtest_worker(self) -> None:
        try:
            target = self.backtest_target
            form = self.pending_backtest_form or self._backtest_form()
            result = self._run_backtest_process(form)
            self.queue.put(WorkerMessage("backtest", payload={"target": target, "result": result}))
        except BacktestCancelled:
            self.queue.put(WorkerMessage("backtest_cancelled"))
        except Exception as exc:
            self.queue.put(WorkerMessage("backtest_error", error=_short_error_text(exc)))

    def _backtest_batch_worker(self, symbols: list[str]) -> None:
        _ui_debug(f"batch_worker_enter target={self.backtest_target} count={len(symbols)}")
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        # Tk variables must only be read on the UI thread. Batch starters set
        # pending_backtest_form before this worker starts; avoid touching widgets
        # from this background thread because it can freeze the whole window.
        if self.pending_backtest_form is not None:
            base_form = self.pending_backtest_form.copy()
        else:
            base_form = self._backtest_form()
        target = self.backtest_target
        cancelled = False

        def build_symbol_form(symbol: str) -> dict[str, Any]:
            form = base_form.copy()
            form["symbol"] = symbol
            if target == "ml":
                form["batch_symbols"] = symbol
                form["_timeout_seconds"] = "120"
                positions = base_form.get("positions", {})
                if isinstance(positions, dict):
                    form["positions"] = {symbol: positions.get(symbol, {})}
            return form

        if target == "ml":
            # Daily production inference is intentionally one batch job. This
            # lets every stock share one model load while per-stock data errors
            # are collected inside the worker instead of aborting the batch.
            batch_form = base_form.copy()
            batch_form["batch_symbols"] = " ".join(symbols)
            batch_form["_timeout_seconds"] = "240"
            try:
                payload = self._run_backtest_process(batch_form)
                if isinstance(payload, dict) and "results" in payload:
                    payload = {"target": target, **payload}
                else:
                    payload = {
                        "target": target,
                        "results": [payload],
                        "errors": [],
                        "total": len(symbols),
                        "cancelled": False,
                    }
                self.queue.put(WorkerMessage("backtest_batch", payload=payload))
            except BacktestCancelled:
                self.queue.put(WorkerMessage("backtest_cancelled"))
            except Exception as exc:
                self.queue.put(WorkerMessage("backtest_error", error=_short_error_text(exc)))
            return

        if target == "ml" and len(symbols) > 1:
            task_name = "ML持仓决策"
            try:
                requested_workers = int(float(base_form.get("_parallel_workers") or ML_BATCH_PARALLEL_WORKERS))
            except Exception:
                requested_workers = ML_BATCH_PARALLEL_WORKERS
            max_workers = min(len(symbols), max(1, requested_workers))
            _ui_debug(f"batch_worker_parallel_start workers={max_workers} count={len(symbols)}")
            self.queue.put(WorkerMessage("status", payload=f"{task_name}并发评估启动：{len(symbols)} 只股票，同时 {max_workers} 只"))
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ml-stock") as executor:
                future_to_symbol = {}
                for symbol in symbols:
                    if self.backtest_stop_event.is_set():
                        cancelled = True
                        break
                    form = build_symbol_form(symbol)
                    future_to_symbol[executor.submit(self._run_backtest_process, form)] = symbol
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1
                    if self.backtest_stop_event.is_set():
                        cancelled = True
                    try:
                        if cancelled:
                            future.cancel()
                            continue
                        results.append(future.result())
                        _ui_debug(f"batch_worker_parallel_done symbol={symbol}")
                    except BacktestCancelled as exc:
                        if not self.backtest_stop_event.is_set():
                            errors.append(f"{symbol}: {_short_error_text(exc)}")
                        else:
                            cancelled = True
                    except BaseException as exc:
                        errors.append(f"{symbol}: {_short_error_text(exc)}")
                    self.queue.put(WorkerMessage("status", payload=f"{task_name}并发进度 {completed}/{len(symbols)}：完成 {symbol}，成功 {len(results)}，失败 {len(errors)}"))
                    if cancelled:
                        break
            _ui_debug(f"batch_worker_parallel_done results={len(results)} errors={len(errors)} cancelled={cancelled}")
            self.queue.put(WorkerMessage("backtest_batch", payload={"target": target, "results": results, "errors": errors, "total": len(symbols), "cancelled": cancelled}))
            return

        for idx, symbol in enumerate(symbols, start=1):
            _ui_debug(f"batch_worker_loop idx={idx} symbol={symbol}")
            if self.backtest_stop_event.is_set():
                cancelled = True
                break
            try:
                task_name = "ML持仓决策" if target == "ml" else "批量回测"
                self.queue.put(WorkerMessage("status", payload=f"{task_name}进度 {idx}/{len(symbols)}：正在评估 {symbol}"))
                form = build_symbol_form(symbol)
                _ui_debug(f"batch_worker_run_process symbol={symbol}")
                results.append(self._run_backtest_process(form))
                _ui_debug(f"batch_worker_process_done symbol={symbol}")
            except BacktestCancelled as exc:
                if target == "ml" and not self.backtest_stop_event.is_set():
                    errors.append(f"{symbol}: {_short_error_text(exc)}")
                else:
                    cancelled = True
                    break
            except BaseException as exc:
                if target != "ml" and isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                errors.append(f"{symbol}: {_short_error_text(exc)}")
            task_name = "ML持仓决策" if target == "ml" else "批量回测"
            self.queue.put(WorkerMessage("status", payload=f"{task_name}进度 {idx}/{len(symbols)}"))
        _ui_debug(f"batch_worker_done results={len(results)} errors={len(errors)} cancelled={cancelled}")
        self.queue.put(WorkerMessage("backtest_batch", payload={"target": target, "results": results, "errors": errors, "total": len(symbols), "cancelled": cancelled}))

    def _run_backtest_process(self, form: dict[str, str]) -> dict[str, Any]:
        timeout_seconds = int(float(form.get("_timeout_seconds") or (240 if form.get("_job") == "ml_predict" else 180)))
        worker_path = engine.ROOT / "desktop_worker.py"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        with tempfile.TemporaryDirectory(prefix="strategy_worker_") as tmp_dir:
            input_path = Path(tmp_dir) / "input.json"
            output_path = Path(tmp_dir) / "output.pkl"
            with input_path.open("w", encoding="utf-8") as handle:
                json.dump(form, handle, ensure_ascii=False)
            _ui_debug(f"run_process_popen symbol={form.get('symbol', '')} timeout={timeout_seconds}")
            process = subprocess.Popen(
                [sys.executable, str(worker_path), str(input_path), str(output_path)],
                cwd=str(engine.ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            _ui_debug(f"run_process_started pid={process.pid} symbol={form.get('symbol', '')}")
            self.backtest_process = process
            with self.backtest_process_lock:
                self.backtest_processes.add(process)
            started_at = time.monotonic()
            try:
                while process.poll() is None:
                    if self.backtest_stop_event.is_set():
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise BacktestCancelled()
                    if time.monotonic() - started_at > timeout_seconds:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        symbol = form.get("symbol", "")
                        raise RuntimeError(f"{symbol} 单只评估超过 {timeout_seconds} 秒，已自动跳过；通常是网络/代理或模型进程卡住。")
                    time.sleep(0.2)
                if self.backtest_stop_event.is_set():
                    raise BacktestCancelled()
                if not output_path.exists():
                    raise RuntimeError(f"评估子进程没有返回结果，退出码 {process.returncode}")
                with output_path.open("rb") as handle:
                    message = pickle.load(handle)
                if isinstance(message, dict) and message.get("kind") == "ok":
                    return message["payload"]
                error = message.get("error") if isinstance(message, dict) else str(message)
                raise RuntimeError(str(error))
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                with self.backtest_process_lock:
                    self.backtest_processes.discard(process)
                    if self.backtest_process is process:
                        self.backtest_process = None

    def _backtest_form(self) -> dict[str, str]:
        return {
            "symbol": self.bt_symbol.get().strip(),
            "start": self.bt_start.get().strip() or "20200101",
            "adjust": self.bt_adjust.get(),
            "cash": self.bt_cash.get().strip() or "100000",
            "fee": self.bt_fee.get().strip() or "0.0003",
            "risk": self.bt_risk.get(),
            "horizon": self.bt_horizon.get(),
            "strategy_type": self.bt_strategy.get(),
            "shares": self.bt_shares.get().strip() or "0",
            "buy_price": self.bt_buy_price.get().strip(),
            "buy_date": self.bt_buy_date.get().strip(),
            "batch_symbols": "",
        }

    def _compute_backtest(self, form: dict[str, str]) -> dict[str, Any]:
        return _compute_backtest_payload(form)

    def _daily_gate_from_backtest(
        self,
        form: dict[str, str],
        symbol: str,
        data: pd.DataFrame,
        best: pd.Series,
        fast_line: pd.Series,
        slow_line: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        horizon: str,
        display_name: str = "",
    ) -> dict[str, Any]:
        risk = form.get("risk", "normal")
        strategy_filter = form.get("strategy_type", "auto")
        cash = float(form.get("cash") or 100000)
        fee = float(form.get("fee") or 0.0003)
        latest_date = data.index[-1]
        latest_close = float(data["Close"].iloc[-1])
        strategy_type = str(best.get("strategy_type", "sma"))
        strategy_label = engine.STRATEGY_TYPES.get(strategy_type, strategy_type)
        fast = int(best["fast"])
        slow = int(best["slow"])
        save_strategy = str(form.get("_save_strategy", "")).lower() in {"1", "true", "yes"}
        active_strategy = str(form.get("_active_strategy", "")).lower() in {"1", "true", "yes"}
        selected_for_left = str(form.get("_selected_for_left", "")).lower() in {"1", "true", "yes"}
        cache_strategy_filter = f"{strategy_type}_{fast}_{slow}" if save_strategy else strategy_filter
        cache_key = engine.strategy_cache_key(symbol, form.get("start", "20200101"), form.get("adjust", "qfq"), cash, fee, horizon, cache_strategy_filter, risk)
        latest_fast = float(fast_line.iloc[-1])
        latest_slow = float(slow_line.iloc[-1])
        entry_today = bool(entries.iloc[-1])
        exit_today = bool(exits.iloc[-1])
        in_trend = engine.strategy_in_trend(strategy_type, latest_fast, latest_slow, latest_close)
        last_side, last_date = engine.last_signal_date(entries, exits)
        risk_factor = {"tight": 0.8, "normal": 1.0, "loose": 1.3}[risk] if horizon == "short" else {"tight": 1.2, "normal": 1.6, "loose": 2.2}[risk]
        lookback = engine.STRATEGY_GRIDS[horizon]["lookback"]
        atr_value = float(engine.atr(data).iloc[-1])
        recent_low = float(data["Low"].tail(lookback).min())
        recent_high = float(data["High"].tail(lookback).max())
        trend_stop = max(latest_slow, latest_close - risk_factor * atr_value)
        structure_stop = recent_low
        stop_line = min(trend_stop, latest_close * (0.992 if horizon == "short" else 0.985)) if in_trend else max(latest_slow, latest_close * 1.01)
        result = {
            "name": display_name,
            "strategy_label": strategy_label,
            "best_params": f"{fast}/{slow}",
            "best": _scan_row_payload(best),
            "signal_lines": [f"Latest daily: {latest_date:%Y-%m-%d}, close {engine.money(latest_close)}."],
            "daily_signal": {
                "date": latest_date.strftime("%Y-%m-%d"),
                "strategy_type": strategy_type,
                "strategy_label": strategy_label,
                "fast": fast,
                "slow": slow,
                "entry_today": entry_today,
                "exit_today": exit_today,
                "in_trend": in_trend,
                "latest_close": latest_close,
                "stop_line": stop_line,
                "structure_stop": structure_stop,
                "recent_high": recent_high,
                "recent_low": recent_low,
                "last_side": last_side,
                "last_date": last_date,
            },
        }
        engine.attach_ml_risk_snapshot(result, data, fast, slow, strategy_type, stop_line)
        engine.DAILY_GATE_CACHE[cache_key] = result
        if save_strategy:
            if active_strategy:
                result["_active_for_trading"] = True
            if selected_for_left or active_strategy:
                result["_selected_for_left"] = True
            engine.save_daily_gate(cache_key, result)
        return result

    def _poll_queue(self) -> None:
        try:
            while True:
                message = self.queue.get_nowait()
                if message.kind == "monitor":
                    self._apply_monitor_results(message.payload)
                    self.after(200, self._run_pending_monitor_refresh)
                elif message.kind == "monitor_error":
                    error_text = message.error or "监控错误"
                    self.status_var.set(error_text)
                    self._show_monitor_error(error_text)
                    self.after(200, self._run_pending_monitor_refresh)
                elif message.kind == "ml_monitor":
                    self._apply_ml_monitor_results(message.payload)
                elif message.kind == "ml_monitor_error":
                    self.status_var.set(message.error or "ML监控错误")
                elif message.kind == "backtest":
                    self._set_backtest_running(False)
                    target = "traditional"
                    result = message.payload
                    if isinstance(message.payload, dict) and "result" in message.payload:
                        target = str(message.payload.get("target", "traditional"))
                        result = message.payload["result"]
                    if target == "ml":
                        if isinstance(result, dict) and "results" in result:
                            self._apply_ml_backtest_batch_result(result)
                        else:
                            self._apply_ml_backtest_result(result)
                    else:
                        self._apply_backtest_result(result)
                elif message.kind == "backtest_batch":
                    _ui_debug(f"poll_backtest_batch_enter target={message.payload.get('target') if isinstance(message.payload, dict) else ''}")
                    self._set_backtest_running(False)
                    if isinstance(message.payload, dict) and message.payload.get("target") == "ml":
                        self._apply_ml_backtest_batch_result(message.payload)
                    else:
                        self._apply_backtest_batch_result(message.payload)
                    _ui_debug("poll_backtest_batch_done")
                elif message.kind == "backtest_error":
                    self._set_backtest_running(False)
                    if self.backtest_target == "ml" and hasattr(self, "ml_summary_var"):
                        self.ml_summary_var.set("ML持仓决策失败")
                        self.status_var.set("ML持仓决策失败，详情已弹出")
                        messagebox.showerror("ML持仓决策失败", message.error or "")
                    else:
                        self.summary_var.set("回测失败")
                        self.status_var.set("回测失败，详情已弹出")
                        messagebox.showerror("回测失败", message.error or "")
                elif message.kind == "backtest_cancelled":
                    self._set_backtest_running(False)
                    if self.backtest_target == "ml" and hasattr(self, "ml_summary_var"):
                        self.ml_summary_var.set("ML持仓决策已终止")
                        self.status_var.set("ML持仓决策已终止，子进程已停止")
                    else:
                        self.summary_var.set("回测已终止")
                        self.status_var.set("回测已终止，子进程已停止")
                elif message.kind == "cache_preview":
                    payload = message.payload if isinstance(message.payload, dict) else {}
                    if payload.get("key_text") == self.cache_preview_key:
                        self._apply_saved_strategy_preview_chart(payload["result"])
                        if payload.get("key_text") == self.pending_saved_description_key:
                            self.pending_saved_description_key = None
                            self._show_strategy_description_popup()
                        self.status_var.set("已用保存策略刷新上方曲线；右侧全部策略表保持不变")
                elif message.kind == "cache_preview_error":
                    payload = message.payload if isinstance(message.payload, dict) else {}
                    if payload.get("key_text") == self.cache_preview_key:
                        if payload.get("key_text") == self.pending_saved_description_key:
                            self.pending_saved_description_key = None
                        self.status_var.set(message.error or "保存策略预览失败")
                elif message.kind == "strategy_saved":
                    self._apply_saved_strategy_result(message.payload)
                elif message.kind == "strategy_save_error":
                    self.status_var.set("保存策略失败，详情已弹出")
                    messagebox.showerror("保存策略失败", message.error or "")
                elif message.kind == "model_action":
                    payload = message.payload if isinstance(message.payload, dict) else {}
                    detail = str(payload.get("detail") or "模型操作完成")
                    self.model_training_status.set(detail)
                    self.refresh_model_diagnostics()
                    if not payload.get("ok"):
                        messagebox.showwarning("模型验收", detail)
                elif message.kind == "status":
                    self.status_var.set(str(message.payload))
        except queue.Empty:
            pass
        self.after(400, self._poll_queue)

    def _apply_monitor_results(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            self.monitor_items[str(item.get("symbol", ""))] = item
        self._save_monitor_snapshot()
        self._render_monitor_table()
        if self.selected_monitor_symbol in self.monitor_items:
            self._render_monitor_strategy_list(self.selected_monitor_symbol)
            self._render_monitor_detail(self.monitor_items[self.selected_monitor_symbol])
            self._draw_intraday_chart(self.monitor_items[self.selected_monitor_symbol])
        elif self.monitor_items:
            self.selected_monitor_symbol = next(iter(self.monitor_items))
            if hasattr(self, "saved_stock_tree") and self.saved_stock_tree.exists(self.selected_monitor_symbol):
                self.saved_stock_tree.selection_set(self.selected_monitor_symbol)
            self._render_monitor_strategy_list(self.selected_monitor_symbol)
            self._render_monitor_detail(self.monitor_items[self.selected_monitor_symbol])
            self._draw_intraday_chart(self.monitor_items[self.selected_monitor_symbol])
        self.status_var.set(f"监控刷新完成：{time.strftime('%H:%M:%S')}，共 {len(items)} 只")

    def _render_monitor_table(self) -> None:
        self._render_saved_stock_picker()

    def _apply_ml_monitor_results(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            self.ml_monitor_items[str(item.get("symbol", ""))] = item
        self._render_ml_monitor_table()
        if self.selected_ml_monitor_symbol in self.ml_monitor_items:
            self._render_ml_monitor_strategy_list(self.selected_ml_monitor_symbol)
            self._draw_ml_intraday_chart(self.ml_monitor_items[self.selected_ml_monitor_symbol])
        elif self.ml_monitor_items:
            self.selected_ml_monitor_symbol = next(iter(self.ml_monitor_items))
            self.ml_monitor_tree.selection_set(self.selected_ml_monitor_symbol)
            self._render_ml_monitor_strategy_list(self.selected_ml_monitor_symbol)
            self._draw_ml_intraday_chart(self.ml_monitor_items[self.selected_ml_monitor_symbol])
        self.status_var.set(f"ML监控刷新完成：{time.strftime('%H:%M:%S')}，共 {len(items)} 只")

    def _render_ml_monitor_table(self) -> None:
        existing = set(self.ml_monitor_tree.get_children())
        current = set(self.ml_monitor_items)
        for iid in existing - current:
            self.ml_monitor_tree.delete(iid)
        for symbol, item in self.ml_monitor_items.items():
            values = (
                item.get("symbol", symbol),
                item.get("name", ""),
                item.get("action", ""),
                item.get("price", ""),
                item.get("daily_gate", ""),
                item.get("minute_trend", ""),
                item.get("volume_ratio", ""),
                item.get("vwap", ""),
                item.get("stop_line", ""),
                item.get("updated", ""),
            )
            if symbol in existing:
                self.ml_monitor_tree.item(symbol, values=values)
            else:
                self.ml_monitor_tree.insert("", "end", iid=symbol, values=values)

    def _render_ml_monitor_strategy_list(self, symbol: str | None = None) -> None:
        if not hasattr(self, "ml_monitor_strategy_tree"):
            return
        if symbol is None:
            symbol = self.selected_ml_monitor_symbol
        for iid in self.ml_monitor_strategy_tree.get_children():
            self.ml_monitor_strategy_tree.delete(iid)
        rows = self._saved_strategy_rows_for_symbol(symbol, strategy_filter="ml")
        if not rows:
            return
        code = engine.normalize_symbol(str(symbol))
        selected_key = self.ml_monitor_strategy_keys.get(code)
        for key_text, values in rows:
            self.ml_monitor_strategy_tree.insert("", "end", iid=key_text, values=values)
        if not selected_key or selected_key not in {key for key, _ in rows}:
            selected_key = rows[0][0]
            self.ml_monitor_strategy_keys[code] = selected_key
        self.ml_monitor_strategy_tree.selection_set(selected_key)
        self.ml_monitor_strategy_tree.focus(selected_key)

    def _on_ml_monitor_strategy_select(self, _event: object | None = None) -> None:
        symbol = self.selected_ml_monitor_symbol
        selection = self.ml_monitor_strategy_tree.selection()
        if not symbol or not selection:
            return
        code = engine.normalize_symbol(symbol)
        self.ml_monitor_strategy_keys[code] = selection[0]
        values = self.ml_monitor_strategy_tree.item(selection[0], "values")
        strategy_name = values[0] if values else "选中ML策略"
        self.status_var.set(f"{code} ML盘中监控已切换为：{strategy_name}")

    def _apply_backtest_result(self, result: dict[str, Any]) -> None:
        self.backtest_result = result
        self.selected_scan_rank = 0
        best = result["best"]
        strategy_label = engine.STRATEGY_TYPES.get(str(best.get("strategy_type", "")), str(best.get("strategy_type", "")))
        summary = (
            f"{result['symbol']} {result['name']} | 最优策略 {strategy_label} {int(best['fast'])}/{int(best['slow'])} | "
            f"收益 {float(best['total_return_pct']):.2f}% | 最大回撤 {float(best['max_drawdown_pct']):.2f}% | "
            f"夏普 {float(best['sharpe']):.2f} | 交易 {int(best['trades'])} 次 | 最终权益 {engine.money(float(best['final_value']))}"
        )
        self._set_backtest_summary(best)
        self._render_backtest_table(result["scan"])
        selected_row = best
        if self.bt_tree.get_children():
            first = self.bt_tree.get_children()[0]
            self.bt_tree.selection_set(first)
            self.bt_tree.focus(first)
            try:
                self.selected_scan_rank = int(first)
            except ValueError:
                self.selected_scan_rank = 0
            selected_candidate = self._selected_scan_row()
            selected_row = selected_candidate if selected_candidate is not None else best
            self._set_backtest_summary(selected_row)
        if not result.get("from_saved_strategy"):
            self._render_strategy_cache_list()
        try:
            self._draw_backtest_chart(result, selected_row)
            self.status_var.set(f"回测完成：右键排名行可查看策略说明或保存策略 {time.strftime('%H:%M:%S')}")
        except Exception as exc:
            self.status_var.set(f"回测完成，但画图失败：{exc}")
            messagebox.showerror("画图失败", traceback.format_exc())

    def _apply_saved_strategy_preview_chart(self, result: dict[str, Any]) -> None:
        self.backtest_result = result
        self.selected_scan_rank = 0
        best = result["best"]
        strategy_label = engine.STRATEGY_TYPES.get(str(best.get("strategy_type", "")), str(best.get("strategy_type", "")))
        self.summary_var.set(
            f"{result['symbol']} {result['name']} | 当前查看 {strategy_label} {int(best['fast'])}/{int(best['slow'])} | "
            f"收益 {float(best['total_return_pct']):.2f}% | 最大回撤 {float(best['max_drawdown_pct']):.2f}% | "
            f"夏普 {float(best['sharpe']):.2f} | 交易 {int(best['trades'])} 次 | 最终权益 {engine.money(float(best['final_value']))}"
        )
        try:
            self._draw_backtest_chart(result, best)
        except Exception as exc:
            self.status_var.set(f"保存策略曲线绘制失败：{exc}")

    def _apply_backtest_batch_result(self, payload: dict[str, Any]) -> None:
        results: list[dict[str, Any]] = list(payload.get("results") or [])
        errors: list[str] = list(payload.get("errors") or [])
        total = int(payload.get("total") or (len(results) + len(errors)))
        cancelled = bool(payload.get("cancelled"))
        if results:
            self._apply_backtest_result(results[0])
            prefix = "批量回测已终止" if cancelled else "批量回测完成"
            self.summary_var.set(f"{prefix}：成功 {len(results)}/{total}，已自动保存每只成功股票的最佳策略，当前展示 {results[0]['symbol']} {results[0]['name']}")
            self.status_var.set(f"{prefix}：成功 {len(results)} 只，失败 {len(errors)} 只；最佳策略已保存并刷新")
        elif cancelled:
            self.summary_var.set("批量回测已终止")
            self.status_var.set("批量回测已终止，未产生新结果")
        else:
            self.summary_var.set("批量回测失败")
            self.status_var.set("批量回测没有成功结果")
        self._render_strategy_cache_list()
        if errors:
            messagebox.showwarning("部分股票回测失败", "\n".join(errors[:8]))

    def _ml_risk_summary_text(self, result: dict[str, Any]) -> str:
        gate = result.get("daily_gate", {}) if isinstance(result.get("daily_gate"), dict) else {}
        risk = gate.get("ml_risk", {}) if isinstance(gate.get("ml_risk"), dict) else {}
        signal = gate.get("daily_signal", {}) if isinstance(gate.get("daily_signal"), dict) else {}
        if not risk and isinstance(signal.get("ml_risk"), dict):
            risk = signal["ml_risk"]
        anomaly = risk.get("anomaly", {}) if isinstance(risk.get("anomaly"), dict) else {}
        mc = risk.get("monte_carlo", {}) if isinstance(risk.get("monte_carlo"), dict) else {}
        if not anomaly and not mc:
            return ""
        parts = []
        if anomaly:
            parts.append(f"异常 {anomaly.get('level', '-')}")
        if mc:
            parts.append(f"10日上涨 {float(mc.get('up_prob', 0)) * 100:.1f}%")
            if mc.get("stop_break_prob") is not None:
                parts.append(f"跌破风控 {float(mc.get('stop_break_prob', 0)) * 100:.1f}%")
        return " | " + " | ".join(parts)

    def _apply_ml_backtest_result(self, result: dict[str, Any]) -> None:
        _ui_debug("apply_ml_single_enter")
        self.ml_backtest_result = result
        self.ml_prediction_results = {str(result["symbol"]): result}
        self._render_ml_prediction_table([result])
        self._select_ml_prediction(str(result["symbol"]), force=True)
        self.status_var.set(f"ML持仓决策完成：{time.strftime('%H:%M:%S')}")
        _ui_debug("apply_ml_single_done")

    def _apply_ml_backtest_batch_result(self, payload: dict[str, Any]) -> None:
        _ui_debug("apply_ml_batch_enter")
        results: list[dict[str, Any]] = list(payload.get("results") or [])
        errors: list[str] = list(payload.get("errors") or [])
        total = int(payload.get("total") or (len(results) + len(errors)))
        cancelled = bool(payload.get("cancelled"))
        if results:
            _ui_debug(f"apply_ml_batch_weight results={len(results)}")
            weighted = results if any(isinstance(item, dict) and "decision_row" in item for item in results) else self._apply_portfolio_weights(results)
            self.ml_backtest_result = weighted[0]
            _ui_debug("apply_ml_batch_render_table")
            self._render_ml_prediction_table(weighted)
            first_symbol = self.ml_tree.get_children()[0] if self.ml_tree.get_children() else str(weighted[0]["symbol"])
            _ui_debug(f"apply_ml_batch_select {first_symbol}")
            self._select_ml_prediction(str(first_symbol), force=True)
            prefix = "ML持仓决策已终止" if cancelled else "ML持仓决策完成"
            error_hint = f"，失败 {len(errors)} 只，失败明细见状态栏" if errors else ""
            self.ml_summary_var.set(f"{prefix}：成功 {len(results)}/{total}{error_hint}，已按推荐动作和目标仓位排序")
            self.status_var.set(f"{prefix}：成功 {len(results)} 只，失败 {len(errors)} 只")
        elif cancelled:
            self.ml_summary_var.set("ML持仓决策已终止")
            self.status_var.set("ML持仓决策已终止，未产生新结果")
        else:
            self.ml_summary_var.set("ML持仓决策失败")
            self.status_var.set("ML持仓决策没有成功结果")
        if errors and not results:
            messagebox.showwarning("部分 ML 持仓决策失败", "\n".join(errors[:8]))
        elif errors:
            self.status_var.set(f"ML持仓决策完成：成功 {len(results)} 只，跳过 {len(errors)} 只；例如 {errors[0]}")
        _ui_debug("apply_ml_batch_done")

    def _fmt_number(self, value: Any, digits: int = 1) -> str:
        try:
            number = float(value)
        except Exception:
            return "-"
        if not np.isfinite(number):
            return "-"
        return f"{number:.{digits}f}"

    def _ml_external_source_summary(self, result: dict[str, Any]) -> str:
        prediction = result.get("prediction", {}) if isinstance(result.get("prediction"), dict) else {}
        notes = prediction.get("source_notes", []) if isinstance(prediction.get("source_notes"), list) else []
        status_map = {"cache": "缓存", "stale_cache": "旧缓存", "ok": "今日拉取", "failed": "拉取失败"}
        buckets = {"资金流": "-", "新闻": "-", "机构活跃": "-"}
        for note in notes:
            if not isinstance(note, dict):
                continue
            source = str(note.get("source") or "")
            status = status_map.get(str(note.get("status") or ""), str(note.get("status") or "-"))
            if "fund" in source or "资金" in source:
                buckets["资金流"] = status
            elif "news" in source or "notice" in source or "新闻" in source:
                buckets["新闻"] = status
            elif "institution" in source or "lhb" in source or "institute" in source or "机构" in source:
                current = buckets["机构活跃"]
                if current in {"-", "拉取失败"} or status == "今日拉取":
                    buckets["机构活跃"] = status
        return "；".join(f"{key}={value}" for key, value in buckets.items())

    def _apply_portfolio_weights(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not results:
            return []
        try:
            total_capital = max(float(self.ml_cash.get() or 100000), 1.0)
        except Exception:
            total_capital = 100000.0
        try:
            target_total_position = float(self.ml_target_position.get() or 80) / 100.0
        except Exception:
            target_total_position = 0.80
        target_total_position = float(np.clip(target_total_position, 0.0, 1.0))
        advice_days = self._ml_selected_advice_days()
        raw_scores: list[float] = []
        for result in results:
            symbol = str(result.get("symbol", ""))
            prediction = result.get("prediction", {})
            projection = self._ml_period_projection(result, advice_days)
            advice = self._ml_period_advice(result, advice_days)
            try:
                current_shares = int(float(self._ml_stock_position(symbol).get("shares", "") or 0))
            except Exception:
                current_shares = 0
            prob_pct = float(projection.get("prob_pct", np.nan))
            exp_pct = float(projection.get("expected_return_pct", np.nan))
            if not np.isfinite(prob_pct):
                prob_pct = 50.0
            action = str(advice.get("action", ""))
            if "暂不买入" in action and current_shares <= 0:
                raw_scores.append(0.0)
                continue
            if "卖出" in action:
                raw_scores.append(0.0)
                continue
            threshold = 45.0 if current_shares > 0 else 50.0
            edge = max(prob_pct - threshold, 0.0)
            if np.isfinite(exp_pct) and exp_pct <= -0.8:
                edge = 0.0
            elif np.isfinite(exp_pct) and exp_pct > 0:
                edge *= 1.0 + min(exp_pct, 6.0) / 20.0
            probability_score = edge ** 1.25 if edge > 0 else 0.0
            raw_scores.append(probability_score)

        invest_weight = target_total_position
        cash_reserve = max(0.0, 1.0 - invest_weight)
        total_raw = float(np.nansum(raw_scores))
        weighted: list[dict[str, Any]] = []
        for result, raw in zip(results, raw_scores):
            item = result.copy()
            symbol = str(item.get("symbol", ""))
            prediction = item.get("prediction", {}) if isinstance(item.get("prediction"), dict) else {}
            latest_price = float(prediction.get("latest_close", 0) or 0)
            position = self._ml_stock_position(symbol)
            try:
                current_shares = int(float(position.get("shares", "") or 0))
            except Exception:
                current_shares = 0
            current_value = current_shares * latest_price
            current_weight = current_value / total_capital * 100 if total_capital > 0 else 0.0
            target_weight = round(0.0 if total_raw <= 0 else raw / total_raw * invest_weight * 100, 1)
            target_value = total_capital * target_weight / 100.0
            target_shares = int(target_value // (latest_price * 100)) * 100 if latest_price > 0 else 0
            trade_shares = target_shares - current_shares
            if abs(trade_shares) < 100:
                action = "持有/观察" if current_shares > 0 else "暂不买入"
                trade_shares = 0
            elif trade_shares > 0:
                action = "买入/加仓"
            elif target_shares <= 0:
                action = "卖出/清仓"
            else:
                action = "卖出/减仓"
            item["current_shares"] = current_shares
            item["current_value"] = round(current_value, 2)
            item["current_weight"] = round(current_weight, 1)
            item["target_weight"] = target_weight
            item["target_shares"] = target_shares
            item["trade_shares"] = trade_shares
            item["rebalance_action"] = action
            item["cash_reserve"] = round(cash_reserve * 100, 1)
            item["target_total_position"] = round(target_total_position * 100, 1)
            item["risk_cash_reserve"] = round(cash_reserve * 100, 1)
            item["row"] = _ml_prediction_row(item)
            weighted.append(item)
        return sorted(weighted, key=lambda item: float(item.get("target_weight", 0)), reverse=True)

    def _ml_selected_advice_days(self) -> int:
        try:
            days = int(str(self.ml_advice_days.get()).strip() or "3")
        except Exception:
            days = 3
        return days if days in {1, 3, 10} else 3

    def _ml_period_projection(self, result: dict[str, Any], days: int) -> dict[str, Any]:
        prediction = result.get("prediction", {}) if isinstance(result.get("prediction"), dict) else {}
        horizons = {int(item.get("days", 0)): item for item in prediction.get("horizons", []) or [] if isinstance(item, dict)}
        horizon = horizons.get(days) or {}
        decision_row = result.get("decision_row")
        if not horizon and isinstance(decision_row, dict):
            horizon = _ml_decision_horizon(decision_row, days)
        if not horizon and days == 1:
            return {
                "prob_pct": np.nan,
                "expected_return_pct": np.nan,
                "label": "1日上涨",
                "detail": "旧缓存没有1日ML概率，请重新评估后再看第二天建议。",
            }
        return {
            "prob_pct": float(horizon.get("up_prob", np.nan)) * 100,
            "expected_return_pct": float(horizon.get("expected_return_pct", np.nan)),
            "label": "次日上涨" if days == 1 else f"{days}日风险投影上涨",
            "detail": str(horizon.get("detail", "")),
        }

    def _ml_position_shares_for_result(self, result: dict[str, Any]) -> int:
        symbol = str(result.get("symbol", ""))
        try:
            return int(float(self._ml_stock_position(symbol).get("shares", "") or result.get("current_shares", 0) or 0))
        except Exception:
            return int(result.get("current_shares", 0) or 0)

    def _display_ml_action(self, action: Any, shares: int) -> str:
        raw = str(action or "-").strip()
        if raw in {"HOLD", "持有"}:
            return "持有/观察" if shares > 0 else "暂不买入"
        if raw in {"ADD_25", "ADD_50", "加仓25%", "加仓50%"}:
            return "买入/试仓" if shares <= 0 else "买入/加仓"
        if raw == "ADD_LIMITED_LOW_CONFIDENCE":
            return "低置信度限量试仓" if shares <= 0 else "低置信度限量加仓"
        if raw in {"NO_TRADE", "NO_TRADE_LOW_CONFIDENCE"}:
            return "持有/观察" if shares > 0 else "暂不买入"
        if raw == "SELL_AVAILABLE":
            return "卖出全部可用股份"
        if raw in {"SELL_ALL", "清仓"}:
            return "卖出/清仓" if shares > 0 else "暂不买入"
        if raw in {"REDUCE_50", "REDUCE_25", "减仓50%", "减仓25%"}:
            return "卖出/减仓" if shares > 0 else "暂不买入"
        return raw

    def _ml_period_advice(self, result: dict[str, Any], days: int | None = None) -> dict[str, Any]:
        days = self._ml_selected_advice_days() if days is None else days
        decision_row = result.get("decision_row")
        if isinstance(decision_row, dict) and days == 1:
            shares = self._ml_position_shares_for_result(result)
            return {
                "action": self._display_ml_action(decision_row.get("recommended_action", "-"), shares),
                "prob_pct": float(decision_row.get("probability_up", np.nan)) * 100,
                "expected_return_pct": float(decision_row.get("expected_open_to_open_return", np.nan)) * 100,
                "detail": str(decision_row.get("reason", "")),
            }
        projection = self._ml_period_projection(result, days)
        shares = self._ml_position_shares_for_result(result)
        prob = float(projection.get("prob_pct", np.nan))
        exp = float(projection.get("expected_return_pct", np.nan))
        reasons: list[str] = []
        if np.isfinite(prob):
            reasons.append(f"{projection.get('label')} {prob:.1f}%")
        if np.isfinite(exp):
            reasons.append(f"{days}日预期 {exp:.1f}%")

        if not np.isfinite(prob):
            action = "持有/观察" if shares > 0 else "暂不买入"
        elif np.isfinite(exp) and exp <= -0.8:
            action = "卖出/减仓" if shares > 0 else "暂不买入"
        elif prob < 45:
            action = "卖出/减仓" if shares > 0 else "暂不买入"
        elif prob < 50:
            action = "减仓/观察" if shares > 0 else "暂不买入"
        elif prob >= 58 and (not np.isfinite(exp) or exp >= 0.2):
            action = "持有/加仓" if shares > 0 else "买入观察"
        elif shares > 0:
            action = "持有/观察"
        else:
            action = "暂不买入"
        return {
            "action": action,
            "prob_pct": prob,
            "expected_return_pct": exp,
            "detail": "；".join(reasons) if reasons else str(projection.get("detail", "")) or "暂无明确优势。",
        }

    def _refresh_ml_advice_view(self) -> None:
        if not hasattr(self, "ml_tree") or not self.ml_prediction_results:
            return
        current = str(self.ml_tree.selection()[0]) if self.ml_tree.selection() else next(iter(self.ml_prediction_results), "")
        self._render_ml_prediction_table(list(self.ml_prediction_results.values()))
        if current and current in self.ml_prediction_results:
            self._select_ml_prediction(current, force=True)

    def _ml_action_priority(self, action: str) -> int:
        text = str(action or "")
        if any(key in text for key in ("买入", "加仓")):
            return 0
        if any(key in text for key in ("卖出", "减仓", "清仓")):
            return 1
        if "持有" in text:
            return 2
        if "暂不买入" in text:
            return 3
        return 4

    def _ml_sort_value(self, result: dict[str, Any], column: str, rank: int = 0) -> Any:
        row = result.get("row") or _ml_prediction_row(result)
        advice = self._ml_period_advice(result, self._ml_selected_advice_days())
        action = self._display_ml_action(row.get("rebalance_action") or advice.get("action", "-"), self._ml_position_shares_for_result(result))
        numeric_map = {
            "rank": rank,
            "utility": row.get("utility"),
            "current_weight": row.get("current_weight"),
            "target_weight": row.get("target_weight"),
            "trade_shares": row.get("trade_shares"),
            "risk": row.get("risk"),
            "prob10": advice.get("prob_pct"),
            "exp10": advice.get("expected_return_pct"),
            "fund_flow": row.get("fund_flow"),
            "news_factor": row.get("news_factor"),
            "institution": row.get("institution"),
            "factor": row.get("factor"),
            "confidence": row.get("confidence"),
            "completeness": row.get("completeness"),
        }
        if column in numeric_map:
            try:
                value = float(numeric_map[column])
                return value if np.isfinite(value) else -np.inf
            except Exception:
                return -np.inf
        if column == "action":
            return (self._ml_action_priority(action), action)
        if column == "symbol":
            return str(row.get("symbol") or result.get("symbol") or "")
        if column == "name":
            return str(row.get("name") or result.get("name") or "")
        if column == "holding_risk":
            return {"高风险": 0, "风险升高": 1, "中等": 2, "低": 3, "未持仓": 4}.get(str(row.get("holding_risk", "")), 9)
        if column == "detail":
            return str(advice.get("detail") or row.get("risk_detail", ""))
        return str(row.get(column, ""))

    def _sort_ml_table(self, column: str) -> None:
        if not self.ml_prediction_results:
            return
        if self.ml_sort_column == column:
            self.ml_sort_reverse = not self.ml_sort_reverse
        else:
            self.ml_sort_column = column
            self.ml_sort_reverse = column not in {"rank", "symbol", "name", "action", "holding_risk", "detail"}
        self._render_ml_prediction_table(list(self.ml_prediction_results.values()))

    def _render_ml_prediction_table(self, results: list[dict[str, Any]]) -> None:
        if hasattr(self, "ml_frozen_tree"):
            for iid in self.ml_frozen_tree.get_children():
                self.ml_frozen_tree.delete(iid)
        for iid in self.ml_tree.get_children():
            self.ml_tree.delete(iid)
        ranked = list(results) if any(isinstance(item, dict) and "decision_row" in item for item in results) else self._apply_portfolio_weights(list(results))
        if self.ml_sort_column:
            indexed = list(enumerate(ranked, start=1))
            indexed.sort(key=lambda item: self._ml_sort_value(item[1], self.ml_sort_column or "", item[0]), reverse=self.ml_sort_reverse)
            ranked = [item for _rank, item in indexed]
        else:
            indexed = list(enumerate(ranked, start=1))
            indexed.sort(
                key=lambda item: (
                    self._ml_action_priority(
                        self._display_ml_action(
                            (item[1].get("row") or _ml_prediction_row(item[1])).get("rebalance_action")
                            or self._ml_period_advice(item[1], self._ml_selected_advice_days()).get("action", "-"),
                            self._ml_position_shares_for_result(item[1]),
                        )
                    ),
                    -float((item[1].get("row") or _ml_prediction_row(item[1])).get("utility") or -999),
                    item[0],
                )
            )
            ranked = [item for _rank, item in indexed]
        self.ml_prediction_results = {str(item["symbol"]): item for item in ranked}
        days = self._ml_selected_advice_days()
        action_title = "次日建议" if days == 1 else f"{days}日风险投影建议"
        probability_title = "次日上涨%" if days == 1 else f"{days}日风险投影涨%"
        expected_title = "次日预期%" if days == 1 else f"{days}日风险投影预期%"
        self.ml_tree.heading("action", text=action_title, command=lambda: self._sort_ml_table("action"))
        self.ml_tree.heading("prob10", text=probability_title, command=lambda: self._sort_ml_table("prob10"))
        self.ml_tree.heading("exp10", text=expected_title, command=lambda: self._sort_ml_table("exp10"))
        for rank, result in enumerate(ranked, start=1):
            row = result.get("row") or _ml_prediction_row(result)
            advice = self._ml_period_advice(result, days)
            symbol = str(row["symbol"])
            display_name = str(row.get("name") or result.get("name") or self._saved_stock_name(symbol) or "")
            values = (
                rank,
                symbol,
                display_name,
                self._display_ml_action(row.get("rebalance_action") or advice.get("action", "-"), self._ml_position_shares_for_result(result)),
                self._fmt_number(row.get("utility")),
                self._fmt_number(row.get("current_weight")),
                self._fmt_number(row.get("target_weight")),
                int(row.get("trade_shares", 0) or 0),
                row.get("holding_risk", "-"),
                self._fmt_number(row.get("risk")),
                self._fmt_number(advice.get("prob_pct")),
                self._fmt_number(advice.get("expected_return_pct")),
                self._fmt_number(row.get("fund_flow")),
                self._fmt_number(row.get("news_factor")),
                self._fmt_number(row.get("institution")),
                self._fmt_number(row.get("factor")),
                self._fmt_number(row.get("confidence")),
                self._fmt_number(row.get("completeness")),
                row.get("event_status", "-"),
                row.get("main_driver", "-"),
                advice.get("detail") or row.get("risk_detail", "-"),
            )
            if hasattr(self, "ml_frozen_tree"):
                self.ml_frozen_tree.insert("", "end", iid=symbol, values=(rank, symbol, display_name))
            self.ml_tree.insert("", "end", iid=symbol, values=values)

    def _select_ml_prediction(self, symbol: str, force: bool = False) -> None:
        _ui_debug(f"select_ml_prediction_enter {symbol}")
        if not force and symbol == self.selected_ml_prediction_symbol:
            _ui_debug(f"select_ml_prediction_skip_same {symbol}")
            return
        self.selected_ml_prediction_symbol = symbol
        self._syncing_ml_table_selection = True
        try:
            if hasattr(self, "ml_frozen_tree") and symbol in self.ml_frozen_tree.get_children():
                current_frozen_selection = tuple(self.ml_frozen_tree.selection())
                if current_frozen_selection != (symbol,):
                    self.ml_frozen_tree.selection_set(symbol)
                if self.ml_frozen_tree.focus() != symbol:
                    self.ml_frozen_tree.focus(symbol)
            if symbol in self.ml_tree.get_children():
                current_selection = tuple(self.ml_tree.selection())
                if current_selection != (symbol,):
                    self.ml_tree.selection_set(symbol)
                if self.ml_tree.focus() != symbol:
                    self.ml_tree.focus(symbol)
        finally:
            self._syncing_ml_table_selection = False
        result = self.ml_prediction_results.get(symbol)
        if not result:
            return
        self.ml_backtest_result = result
        row = result.get("row") or _ml_prediction_row(result)
        days = self._ml_selected_advice_days()
        advice = self._ml_period_advice(result, days)
        self.ml_summary_var.set(
            f"{row['symbol']} {row['name']} | 持仓风险 {row.get('holding_risk', '-')} | "
            f"{days}日建议 {self._display_ml_action(row.get('rebalance_action') or advice.get('action', '-'), self._ml_position_shares_for_result(result))} | "
            f"{days}日上涨 {self._fmt_number(advice.get('prob_pct'))}% | "
            f"{days}日预期 {self._fmt_number(advice.get('expected_return_pct'))}% | "
            f"目标仓位 {self._fmt_number(row.get('target_weight'))}% | 缓冲/现金 {self._fmt_number(result.get('cash_reserve', 0))}%"
        )
        try:
            self._draw_ml_prediction_chart(result, update_detail=True)
        except Exception as exc:
            self.status_var.set(f"ML预测图绘制失败：{exc}")
        _ui_debug(f"select_ml_prediction_done {symbol}")

    def _render_backtest_table(self, scan: pd.DataFrame) -> None:
        self._render_result_table(self.bt_tree, scan)

    def _render_result_table(self, tree: ttk.Treeview, scan: pd.DataFrame) -> None:
        for iid in tree.get_children():
            tree.delete(iid)
        for display_rank, (source_idx, row) in enumerate(scan.iterrows(), start=1):
            strategy_type = row.get("strategy_type")
            label = engine.STRATEGY_TYPES.get(str(strategy_type), str(strategy_type))
            values = (
                display_rank,
                label,
                f"{int(row['fast'])}/{int(row['slow'])}",
                f"{float(row['total_return_pct']):.2f}",
                f"{float(row['max_drawdown_pct']):.2f}",
                "-" if pd.isna(row.get("sharpe")) else f"{float(row['sharpe']):.2f}",
                int(row["trades"]),
                engine.money(float(row["final_value"])),
                f"{float(row['score']):.2f}",
            )
            tree.insert("", "end", iid=str(source_idx), values=values)

    def _set_backtest_summary(self, row: pd.Series) -> None:
        result = self.backtest_result
        if not result:
            return
        strategy_type = str(row.get("strategy_type", ""))
        strategy_label = engine.STRATEGY_TYPES.get(strategy_type, strategy_type)
        summary = (
            f"{result['symbol']} {result['name']} | 当前策略 {strategy_label} {int(row['fast'])}/{int(row['slow'])} | "
            f"收益 {float(row['total_return_pct']):.2f}% | 最大回撤 {float(row['max_drawdown_pct']):.2f}% | "
            f"夏普 {float(row['sharpe']):.2f} | 交易 {int(row['trades'])} 次 | 最终权益 {engine.money(float(row['final_value']))}"
        )
        self.summary_var.set(summary)

    def _strategy_description_text(self, row: pd.Series | None) -> str:
        result = self.backtest_result
        if not result or row is None:
            return "请先完成回测，并在排名表里选择一条策略。"
        try:
            shares = int(float((self.bt_shares.get() or "0").strip() or 0))
        except Exception:
            shares = 0
        try:
            buy_price = engine.parse_float(self.bt_buy_price.get(), None)
        except Exception:
            buy_price = None
        buy_date = self.bt_buy_date.get().strip()
        risk = self.bt_risk.get().strip() or "normal"
        horizon = self.bt_horizon.get().strip() or str(result.get("horizon") or "short")

        try:
            advice = engine.generate_advice(
                str(result["symbol"]),
                result["data"],
                row,
                shares,
                buy_price,
                buy_date,
                risk,
                horizon,
            )
            daily = advice.get("daily_signal", {}) if isinstance(advice.get("daily_signal"), dict) else {}
            metrics = advice.get("metrics", []) if isinstance(advice.get("metrics"), list) else []
            metric_text = " | ".join(
                f"{item.get('k')}: {item.get('v')}" for item in metrics if isinstance(item, dict)
            )
            strategy_label = daily.get("strategy_label") or engine.STRATEGY_TYPES.get(str(row.get("strategy_type", "")), str(row.get("strategy_type", "")))
            header = (
                f"{result['symbol']} {result['name']} | {strategy_label} {int(row['fast'])}/{int(row['slow'])}\n"
                f"{metric_text}\n"
                f"关键线：防守线 {engine.money(float(daily.get('stop_line', np.nan)))}，"
                f"结构低点 {engine.money(float(daily.get('structure_stop', np.nan)))}，"
                f"近期高点 {engine.money(float(daily.get('recent_high', np.nan)))}。"
            )
            signal_lines = [str(item) for item in advice.get("signal_lines", []) if item]
            buy_signal_lines = [str(item) for item in advice.get("buy_signal_lines", []) if item]
            sell_signal_lines = [str(item) for item in advice.get("sell_signal_lines", []) if item]
            action_lines = [str(item) for item in advice.get("action_lines", []) if item]
            sell_plan_lines = [str(item) for item in advice.get("sell_plan_lines", []) if item]
            reminder_lines = [str(item) for item in advice.get("reminder_lines", []) if item]
            sections = [
                header,
                "\n当前信号：\n" + "\n".join(f"- {line}" for line in signal_lines[:9]),
                "\n买入信号说明：\n" + "\n".join(f"- {line}" for line in buy_signal_lines),
                "\n卖出信号说明：\n" + "\n".join(f"- {line}" for line in sell_signal_lines),
                "\n接下来怎么做：\n" + "\n".join(f"- {line}" for line in action_lines),
                "\n卖出执行策略：\n" + "\n".join(f"- {line}" for line in sell_plan_lines),
                "\n盯盘提醒：\n" + "\n".join(f"- {line}" for line in reminder_lines[:4]),
            ]
            return "\n".join(section for section in sections if section.strip())
        except Exception as exc:
            strategy_type = str(row.get("strategy_type", ""))
            strategy_label = engine.STRATEGY_TYPES.get(strategy_type, strategy_type)
            latest_close = float(result["data"]["Close"].iloc[-1])
            return (
                f"{result['symbol']} {result['name']} | {strategy_label} {int(row['fast'])}/{int(row['slow'])}\n"
                f"最新收盘 {engine.money(latest_close)}。说明生成失败：{exc}\n"
                "可以先参考上方回测曲线和表格，稍后重新点击该策略行刷新说明。"
            )

    def _show_strategy_description_popup(self) -> None:
        row = self._selected_scan_row()
        if row is None or not self.backtest_result:
            self.status_var.set("请先完成回测并在排名表里选择一条策略")
            return
        text = self._strategy_description_text(row)
        strategy_type = str(row.get("strategy_type", ""))
        strategy_label = engine.STRATEGY_TYPES.get(strategy_type, strategy_type)
        window = tk.Toplevel(self)
        window.title(f"策略说明 - {strategy_label} {int(row['fast'])}/{int(row['slow'])}")
        window.geometry("820x560")
        window.minsize(640, 420)
        window.configure(background="#eef3f8")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        text_box = tk.Text(
            window,
            wrap="word",
            background="#ffffff",
            foreground="#14213d",
            relief=tk.FLAT,
            padx=16,
            pady=14,
            font=("Microsoft YaHei UI", 10),
        )
        scroll = ttk.Scrollbar(window, orient=tk.VERTICAL, command=text_box.yview)
        text_box.configure(yscrollcommand=scroll.set)
        text_box.grid(row=0, column=0, sticky="nsew", padx=(12, 0), pady=12)
        scroll.grid(row=0, column=1, sticky="ns", pady=12, padx=(0, 12))
        text_box.insert("1.0", text)
        text_box.configure(state=tk.DISABLED)
        buttons = ttk.Frame(window, padding=(12, 0, 12, 12))
        buttons.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(buttons, text="保存这个策略", command=self._save_selected_rank_strategy).pack(side=tk.LEFT)
        ttk.Button(buttons, text="关闭", command=window.destroy).pack(side=tk.RIGHT)

    def _show_saved_strategy_description_popup(self) -> None:
        key_text = self._right_table_saved_key()
        if not key_text:
            self.status_var.set("请先在右侧全部策略表里选择一条策略")
            return
        if self.backtest_result and self.backtest_result.get("cache_key_text") == key_text:
            self._show_strategy_description_popup()
            return
        self.pending_saved_description_key = key_text
        if self.cache_preview_key == key_text and self.cache_preview_worker and self.cache_preview_worker.is_alive():
            self.status_var.set("正在加载这条保存策略，加载完成后会自动打开策略说明")
            return
        self.status_var.set("正在读取保存策略，加载完成后会自动打开策略说明")
        self._start_saved_strategy_preview(key_text)

    def _show_backtest_context_menu(self, event: tk.Event) -> None:
        row_id = self.bt_tree.identify_row(event.y)
        if row_id.startswith("saved:"):
            self.bt_tree.selection_set(row_id)
            self.bt_tree.focus(row_id)
            try:
                self.saved_bt_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.saved_bt_context_menu.grab_release()
            return
        if row_id:
            self.bt_tree.selection_set(row_id)
            self.bt_tree.focus(row_id)
            try:
                self.selected_scan_rank = int(row_id)
            except ValueError:
                self.selected_scan_rank = 0
            self._on_backtest_rank_select()
        if not self.bt_tree.selection():
            self.status_var.set("请先完成回测并在排名表里选择一条策略")
            return
        try:
            self.bt_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.bt_context_menu.grab_release()

    def _selected_scan_row(self) -> pd.Series | None:
        result = self.backtest_result
        if not result:
            return None
        scan: pd.DataFrame = result["scan"]
        if scan.empty:
            return None
        selection = self.bt_tree.selection()
        if selection:
            try:
                self.selected_scan_rank = int(selection[0])
            except ValueError:
                self.selected_scan_rank = 0
        if self.selected_scan_rank in scan.index:
            return scan.loc[self.selected_scan_rank]
        idx = max(0, min(self.selected_scan_rank, len(scan) - 1))
        return scan.iloc[idx]

    def _on_backtest_rank_select(self, _event: object | None = None) -> None:
        selection = self.bt_tree.selection()
        if selection and str(selection[0]).startswith("saved:"):
            key_text = self._right_table_saved_key(str(selection[0]))
            if key_text:
                self._start_saved_strategy_preview(key_text)
                self.status_var.set("正在加载右侧选中策略曲线；右键可加入左侧用于盘中/ML")
            return
        row = self._selected_scan_row()
        if row is None or not self.backtest_result:
            return
        self._set_backtest_summary(row)
        try:
            self._draw_backtest_chart(self.backtest_result, row)
        except Exception as exc:
            self.status_var.set(f"选中策略画图失败：{exc}")

    def _add_right_saved_strategy_to_left(self) -> None:
        key_text = self._right_table_saved_key()
        if not key_text:
            self.status_var.set("请先在右侧全部策略表里选择一条策略")
            return
        self._select_strategy_for_left_from_key(key_text, refresh=True)

    def _load_right_saved_strategy_preview(self) -> None:
        key_text = self._right_table_saved_key()
        if not key_text:
            self.status_var.set("请先在右侧全部策略表里选择一条策略")
            return
        self._start_saved_strategy_preview(key_text)

    def _ml_table_yview(self, *args: object) -> None:
        if hasattr(self, "ml_frozen_tree"):
            self.ml_frozen_tree.yview(*args)
        if hasattr(self, "ml_tree"):
            self.ml_tree.yview(*args)

    def _on_ml_table_mousewheel(self, event: tk.Event) -> str:
        step = -1 if getattr(event, "delta", 0) > 0 else 1
        if hasattr(self, "ml_frozen_tree"):
            self.ml_frozen_tree.yview_scroll(step, "units")
        if hasattr(self, "ml_tree"):
            self.ml_tree.yview_scroll(step, "units")
        return "break"

    def _on_ml_frozen_rank_select(self, _event: object | None = None) -> None:
        _ui_debug("ml_frozen_rank_select_enter")
        if getattr(self, "_syncing_ml_table_selection", False):
            _ui_debug("ml_frozen_rank_select_skip_sync")
            return
        selection = self.ml_frozen_tree.selection()
        if not selection:
            return
        symbol = selection[0]
        if symbol == self.selected_ml_prediction_symbol:
            _ui_debug(f"ml_frozen_rank_select_skip_same {symbol}")
            return
        self._select_ml_prediction(symbol)
        _ui_debug(f"ml_frozen_rank_select_done {symbol}")

    def _on_ml_rank_select(self, _event: object | None = None) -> None:
        _ui_debug("ml_rank_select_enter")
        if getattr(self, "_syncing_ml_table_selection", False):
            _ui_debug("ml_rank_select_skip_sync")
            return
        selection = self.ml_tree.selection()
        if not selection:
            return
        symbol = selection[0]
        if symbol == self.selected_ml_prediction_symbol:
            _ui_debug(f"ml_rank_select_skip_same {symbol}")
            return
        self._select_ml_prediction(symbol)
        _ui_debug(f"ml_rank_select_done {symbol}")

    def _delete_selected_cache(self) -> None:
        selection = self.cache_tree.selection()
        key_text = selection[0] if selection else ""
        if not key_text or not self.cache_tree.exists(key_text):
            self.status_var.set("请先在左侧选择股票或已保存策略")
            return
        cache = engine.load_persistent_strategy_cache()
        if key_text.startswith("stock:"):
            symbol = self._cache_iid_symbol(key_text)
            rows = self._saved_records_for_symbol(symbol, exclude_strategy_type="ml")
            if not rows:
                self.status_var.set("这只股票没有可删除的保存策略")
                return
            if not messagebox.askyesno("移出左侧", f"确认把 {symbol} 的左侧已选策略全部移出？历史回测结果仍会保留在右侧。"):
                return
            for child_key, _record in rows:
                editable = cache.get(child_key)
                if isinstance(editable, dict):
                    editable["selected_for_left"] = False
                    editable["active_for_trading"] = False
            self.monitor_strategy_keys.pop(symbol, None)
            self.ml_monitor_strategy_keys.pop(symbol, None)
            deleted_text = f"已把 {symbol} 的已选策略移出左侧；历史策略仍保留"
        else:
            if not messagebox.askyesno("移出左侧", "确认把这条策略从左侧移出？历史回测结果仍会保留在右侧。"):
                return
            editable = cache.get(key_text)
            if isinstance(editable, dict):
                editable["selected_for_left"] = False
                editable["active_for_trading"] = False
                try:
                    code = engine.normalize_symbol(str(editable.get("symbol", "")))
                    if self.monitor_strategy_keys.get(code) == key_text:
                        self.monitor_strategy_keys.pop(code, None)
                    if self.ml_monitor_strategy_keys.get(code) == key_text:
                        self.ml_monitor_strategy_keys.pop(code, None)
                except Exception:
                    pass
            deleted_text = "已把选中策略移出左侧；历史策略仍保留"
        engine.save_persistent_strategy_cache()
        self._render_strategy_cache_list()
        self.status_var.set(deleted_text)

    def _save_selected_rank_strategy(self) -> None:
        result = self.backtest_result
        row = self._selected_scan_row()
        if not result or row is None:
            self.status_var.set("请先完成回测并在右侧选择一条策略")
            return
        form = self._backtest_form()
        form["_save_strategy"] = "1"
        form["_active_strategy"] = "1"
        form["_selected_for_left"] = "1"
        self.status_var.set("正在后台保存选中策略...")
        worker = threading.Thread(target=self._save_selected_rank_strategy_worker, args=(result, row.copy(), form), daemon=True)
        worker.start()

    def _save_selected_rank_strategy_worker(self, result: dict[str, Any], row: pd.Series, form: dict[str, str]) -> None:
        try:
            data: pd.DataFrame = result["data"]
            horizon = str(result.get("horizon") or form.get("horizon", "short"))
            fast = int(row["fast"])
            slow = int(row["slow"])
            strategy_type = str(row.get("strategy_type", "sma"))
            fast_line, slow_line, entries, exits = engine.strategy_signals(data, fast, slow, horizon, strategy_type)
            gate = self._daily_gate_from_backtest(
                form,
                str(result["symbol"]),
                data,
                row,
                fast_line,
                slow_line,
                entries,
                exits,
                horizon,
                str(result.get("name") or ""),
            )
            self.queue.put(
                WorkerMessage(
                    "strategy_saved",
                    payload={
                        "result": result,
                        "row": row,
                        "gate": gate,
                        "fast_line": fast_line,
                        "slow_line": slow_line,
                        "strategy_type": strategy_type,
                        "fast": fast,
                        "slow": slow,
                    },
                )
            )
        except Exception:
            self.queue.put(WorkerMessage("strategy_save_error", error=traceback.format_exc()))

    def _apply_saved_strategy_result(self, payload: dict[str, Any]) -> None:
        result = payload["result"]
        result["daily_gate"] = payload["gate"]
        result["best"] = payload["row"]
        result["fast_line"] = payload["fast_line"]
        result["slow_line"] = payload["slow_line"]
        self._render_strategy_cache_list()
        strategy_type = str(payload["strategy_type"])
        label = engine.STRATEGY_TYPES.get(strategy_type, strategy_type)
        self.status_var.set(f"已保存选中策略：{result['symbol']} {label} {payload['fast']}/{payload['slow']}，盘中监控会优先使用它")

    def _zoom_backtest(self, factor: float) -> None:
        payload = self.backtest_chart_payload
        if not payload:
            return
        total = len(payload.get("points", []))
        if total <= 2:
            return
        start, end = self.backtest_zoom or (0, total)
        current = max(2, end - start)
        new_len = max(30, min(total, int(current * factor)))
        center = start + current // 2
        new_start = max(0, min(total - new_len, center - new_len // 2))
        self.backtest_zoom = (new_start, new_start + new_len)
        self._draw_backtest_payload()

    def _reset_backtest_zoom(self) -> None:
        self.backtest_zoom = None
        self.backtest_drag_start_x = None
        self.backtest_pan_start_x = None
        self.backtest_pan_start_zoom = None
        if self.backtest_drag_rect is not None:
            self.backtest_canvas.delete(self.backtest_drag_rect)
            self.backtest_drag_rect = None
        self._draw_backtest_payload()

    def _open_backtest_fullscreen(self) -> None:
        if not self.backtest_chart_payload:
            self.status_var.set("还没有回测曲线，先跑一次回测")
            return
        if self.backtest_fullscreen_window is not None and self.backtest_fullscreen_window.winfo_exists():
            self.backtest_fullscreen_window.lift()
            self.backtest_fullscreen_window.focus_force()
            self._draw_backtest_fullscreen_payload()
            return

        window = tk.Toplevel(self)
        window.title("回测曲线全屏")
        window.configure(background="#eef3f8")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        self.backtest_fullscreen_window = window
        self.backtest_fullscreen_zoom = self.backtest_zoom

        toolbar = ttk.Frame(window, padding=(12, 10, 12, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="放大", command=lambda: self._zoom_backtest_fullscreen(0.72)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="缩小", command=lambda: self._zoom_backtest_fullscreen(1.35)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="重置缩放", command=self._reset_backtest_fullscreen_zoom).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(toolbar, text="退出全屏", command=self._close_backtest_fullscreen).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="滚轮缩放，左键框选/单击看点位，右键拖动横向平移", foreground="#607086").pack(side=tk.LEFT, padx=(18, 0))

        canvas = tk.Canvas(window, background="#fbfdff", highlightthickness=1, highlightbackground="#d8e0ea")
        canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        canvas.bind("<Configure>", self._schedule_backtest_fullscreen_redraw)
        canvas.bind("<MouseWheel>", lambda event: self._zoom_backtest_fullscreen(0.82 if event.delta > 0 else 1.22))
        canvas.bind("<ButtonPress-1>", self._on_backtest_fullscreen_drag_start)
        canvas.bind("<B1-Motion>", self._on_backtest_fullscreen_drag_move)
        canvas.bind("<ButtonRelease-1>", self._on_backtest_fullscreen_drag_release)
        canvas.bind("<ButtonPress-3>", self._on_backtest_fullscreen_pan_start)
        canvas.bind("<B3-Motion>", self._on_backtest_fullscreen_pan_move)
        canvas.bind("<ButtonRelease-3>", self._on_backtest_fullscreen_pan_release)
        window.protocol("WM_DELETE_WINDOW", self._close_backtest_fullscreen)
        window.bind("<Escape>", lambda _event: self._close_backtest_fullscreen())
        self.backtest_fullscreen_canvas = canvas
        try:
            window.state("zoomed")
        except tk.TclError:
            window.geometry("1280x820")
        self.after(120, self._draw_backtest_fullscreen_payload)

    def _close_backtest_fullscreen(self) -> None:
        if self.backtest_fullscreen_resize_job is not None:
            try:
                self.after_cancel(self.backtest_fullscreen_resize_job)
            except Exception:
                pass
            self.backtest_fullscreen_resize_job = None
        window = self.backtest_fullscreen_window
        self.backtest_fullscreen_window = None
        self.backtest_fullscreen_canvas = None
        self.backtest_fullscreen_drag_start_x = None
        self.backtest_fullscreen_drag_rect = None
        self.backtest_fullscreen_pan_start_x = None
        self.backtest_fullscreen_pan_start_zoom = None
        if window is not None and window.winfo_exists():
            window.destroy()

    def _zoom_backtest_fullscreen(self, factor: float) -> None:
        payload = self.backtest_chart_payload
        if not payload:
            return
        total = len(payload.get("points", []))
        if total <= 2:
            return
        start, end = self.backtest_fullscreen_zoom or (0, total)
        current = max(2, end - start)
        new_len = max(30, min(total, int(current * factor)))
        center = start + current // 2
        new_start = max(0, min(total - new_len, center - new_len // 2))
        self.backtest_fullscreen_zoom = (new_start, new_start + new_len)
        self._draw_backtest_fullscreen_payload()

    def _reset_backtest_fullscreen_zoom(self) -> None:
        self.backtest_fullscreen_zoom = None
        self.backtest_fullscreen_drag_start_x = None
        canvas = self.backtest_fullscreen_canvas
        if canvas is not None and self.backtest_fullscreen_drag_rect is not None:
            canvas.delete(self.backtest_fullscreen_drag_rect)
            self.backtest_fullscreen_drag_rect = None
        self._draw_backtest_fullscreen_payload()

    def _draw_backtest_fullscreen_payload(self) -> None:
        canvas = self.backtest_fullscreen_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        self._draw_backtest_payload_on_canvas(canvas, self.backtest_fullscreen_zoom)

    def _schedule_backtest_fullscreen_redraw(self, _event: tk.Event | None = None) -> None:
        if not self.backtest_chart_payload or self.backtest_fullscreen_canvas is None:
            return
        if self.backtest_fullscreen_resize_job is not None:
            try:
                self.after_cancel(self.backtest_fullscreen_resize_job)
            except Exception:
                pass
        self.backtest_fullscreen_resize_job = self.after(350, self._redraw_backtest_fullscreen_after_resize)

    def _redraw_backtest_fullscreen_after_resize(self) -> None:
        self.backtest_fullscreen_resize_job = None
        self._draw_backtest_fullscreen_payload()

    def _on_backtest_mousewheel(self, event: tk.Event) -> None:
        self._zoom_backtest(0.82 if event.delta > 0 else 1.22)

    def _schedule_backtest_canvas_redraw(self, _event: tk.Event | None = None) -> None:
        if not self.backtest_chart_payload:
            return
        if self.backtest_resize_job is not None:
            try:
                self.after_cancel(self.backtest_resize_job)
            except Exception:
                pass
        self.backtest_resize_job = self.after(RESIZE_REDRAW_DELAY_MS, self._redraw_backtest_canvas_after_resize)

    def _redraw_backtest_canvas_after_resize(self) -> None:
        self.backtest_resize_job = None
        self._draw_backtest_payload()

    def _schedule_monitor_canvas_redraw(self, _event: tk.Event | None = None) -> None:
        if not self.selected_monitor_symbol or self.selected_monitor_symbol not in self.monitor_items:
            return
        if self.monitor_resize_job is not None:
            try:
                self.after_cancel(self.monitor_resize_job)
            except Exception:
                pass
        self.monitor_resize_job = self.after(RESIZE_REDRAW_DELAY_MS, self._redraw_monitor_canvas_after_resize)

    def _redraw_monitor_canvas_after_resize(self) -> None:
        self.monitor_resize_job = None
        symbol = self.selected_monitor_symbol
        if symbol and symbol in self.monitor_items:
            self._draw_intraday_chart(self.monitor_items[symbol])

    def _schedule_ml_canvas_redraw(self, _event: tk.Event | None = None) -> None:
        if not self.ml_backtest_result:
            return
        if self.ml_resize_job is not None:
            try:
                self.after_cancel(self.ml_resize_job)
            except Exception:
                pass
        self.ml_resize_job = self.after(RESIZE_REDRAW_DELAY_MS, self._redraw_ml_canvas_after_resize)

    def _redraw_ml_canvas_after_resize(self) -> None:
        self.ml_resize_job = None
        if self.ml_backtest_result:
            _ui_debug("redraw_ml_canvas_after_resize")
            self._draw_ml_prediction_chart(self.ml_backtest_result, update_detail=False)

    def _schedule_ml_monitor_canvas_redraw(self, _event: tk.Event | None = None) -> None:
        if not self.selected_ml_monitor_symbol or self.selected_ml_monitor_symbol not in self.ml_monitor_items:
            return
        if self.ml_monitor_resize_job is not None:
            try:
                self.after_cancel(self.ml_monitor_resize_job)
            except Exception:
                pass
        self.ml_monitor_resize_job = self.after(RESIZE_REDRAW_DELAY_MS, self._redraw_ml_monitor_canvas_after_resize)

    def _redraw_ml_monitor_canvas_after_resize(self) -> None:
        self.ml_monitor_resize_job = None
        symbol = self.selected_ml_monitor_symbol
        if symbol and symbol in self.ml_monitor_items:
            self._draw_ml_intraday_chart(self.ml_monitor_items[symbol])

    def _backtest_plot_bounds(self) -> tuple[int, int, int, int]:
        return self._backtest_plot_bounds_for_canvas(self.backtest_canvas)

    def _backtest_plot_bounds_for_canvas(self, canvas: tk.Canvas) -> tuple[int, int, int, int]:
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        width = canvas_width if canvas_width > 100 else 900
        height = canvas_height if canvas_height > 120 else 420
        return 62, width - 24, 34, height - 44

    def _on_backtest_drag_start(self, event: tk.Event) -> None:
        if not self.backtest_chart_payload:
            return
        left, right, top, bottom = self._backtest_plot_bounds()
        x = max(left, min(right, int(event.x)))
        self.backtest_drag_start_x = x
        if self.backtest_drag_rect is not None:
            self.backtest_canvas.delete(self.backtest_drag_rect)
        self.backtest_drag_rect = self.backtest_canvas.create_rectangle(
            x,
            top,
            x,
            bottom,
            outline="#1464f4",
            dash=(4, 3),
            width=1,
            fill="",
        )

    def _on_backtest_drag_move(self, event: tk.Event) -> None:
        if self.backtest_drag_start_x is None or self.backtest_drag_rect is None:
            return
        left, right, top, bottom = self._backtest_plot_bounds()
        x = max(left, min(right, int(event.x)))
        self.backtest_canvas.coords(self.backtest_drag_rect, self.backtest_drag_start_x, top, x, bottom)

    def _on_backtest_drag_release(self, event: tk.Event) -> None:
        payload = self.backtest_chart_payload
        start_x = self.backtest_drag_start_x
        if not payload or start_x is None:
            return
        left, right, _top, _bottom = self._backtest_plot_bounds()
        end_x = max(left, min(right, int(event.x)))
        if self.backtest_drag_rect is not None:
            self.backtest_canvas.delete(self.backtest_drag_rect)
            self.backtest_drag_rect = None
        self.backtest_drag_start_x = None

        x1, x2 = sorted((start_x, end_x))
        if x2 - x1 < 12:
            self._show_backtest_point_info(self.backtest_canvas, event.x, self.backtest_zoom)
            return
        points = payload.get("points", [])
        total = len(points)
        if total <= 2 or right <= left:
            return
        current_start, current_end = self.backtest_zoom or (0, total)
        current_len = max(2, current_end - current_start)
        rel1 = (x1 - left) / (right - left)
        rel2 = (x2 - left) / (right - left)
        new_start = current_start + int(rel1 * current_len)
        new_end = current_start + int(rel2 * current_len)
        new_start = max(0, min(total - 2, new_start))
        new_end = max(new_start + 2, min(total, new_end))
        if new_end - new_start < 8:
            return
        self.backtest_zoom = (new_start, new_end)
        self._draw_backtest_payload()
        self.status_var.set(f"已按框选范围放大：{new_start + 1}-{new_end}/{total}")

    def _pan_zoom_from_drag(
        self,
        canvas: tk.Canvas,
        start_x: int | None,
        current_x: int,
        start_zoom: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        payload = self.backtest_chart_payload
        if not payload or start_x is None:
            return None
        points = payload.get("points", [])
        total = len(points)
        if total <= 2:
            return None
        left, right, _top, _bottom = self._backtest_plot_bounds_for_canvas(canvas)
        width = max(1, right - left)
        start, end = start_zoom or (0, total)
        current_len = max(2, end - start)
        if current_len >= total:
            return (0, total)
        shift = int(round(-(int(current_x) - int(start_x)) / width * current_len))
        new_start = max(0, min(total - current_len, start + shift))
        return (new_start, new_start + current_len)

    def _on_backtest_pan_start(self, event: tk.Event) -> None:
        if not self.backtest_chart_payload:
            return
        total = len(self.backtest_chart_payload.get("points", []))
        self.backtest_pan_start_x = int(event.x)
        self.backtest_pan_start_zoom = self.backtest_zoom or (0, total)
        self.backtest_canvas.configure(cursor="fleur")

    def _on_backtest_pan_move(self, event: tk.Event) -> None:
        zoom = self._pan_zoom_from_drag(self.backtest_canvas, self.backtest_pan_start_x, int(event.x), self.backtest_pan_start_zoom)
        if zoom is None:
            return
        self.backtest_zoom = zoom
        self._draw_backtest_payload()

    def _on_backtest_pan_release(self, _event: tk.Event) -> None:
        self.backtest_pan_start_x = None
        self.backtest_pan_start_zoom = None
        self.backtest_canvas.configure(cursor="")
        if self.backtest_zoom and self.backtest_chart_payload:
            total = len(self.backtest_chart_payload.get("points", []))
            self.status_var.set(f"已平移到：{self.backtest_zoom[0] + 1}-{self.backtest_zoom[1]}/{total}")

    def _on_backtest_fullscreen_drag_start(self, event: tk.Event) -> None:
        canvas = self.backtest_fullscreen_canvas
        if not self.backtest_chart_payload or canvas is None:
            return
        left, right, top, bottom = self._backtest_plot_bounds_for_canvas(canvas)
        x = max(left, min(right, int(event.x)))
        self.backtest_fullscreen_drag_start_x = x
        if self.backtest_fullscreen_drag_rect is not None:
            canvas.delete(self.backtest_fullscreen_drag_rect)
        self.backtest_fullscreen_drag_rect = canvas.create_rectangle(
            x,
            top,
            x,
            bottom,
            outline="#1464f4",
            dash=(4, 3),
            width=1,
            fill="",
        )

    def _on_backtest_fullscreen_drag_move(self, event: tk.Event) -> None:
        canvas = self.backtest_fullscreen_canvas
        if canvas is None or self.backtest_fullscreen_drag_start_x is None or self.backtest_fullscreen_drag_rect is None:
            return
        left, right, top, bottom = self._backtest_plot_bounds_for_canvas(canvas)
        x = max(left, min(right, int(event.x)))
        canvas.coords(self.backtest_fullscreen_drag_rect, self.backtest_fullscreen_drag_start_x, top, x, bottom)

    def _on_backtest_fullscreen_drag_release(self, event: tk.Event) -> None:
        payload = self.backtest_chart_payload
        canvas = self.backtest_fullscreen_canvas
        start_x = self.backtest_fullscreen_drag_start_x
        if not payload or canvas is None or start_x is None:
            return
        left, right, _top, _bottom = self._backtest_plot_bounds_for_canvas(canvas)
        end_x = max(left, min(right, int(event.x)))
        if self.backtest_fullscreen_drag_rect is not None:
            canvas.delete(self.backtest_fullscreen_drag_rect)
            self.backtest_fullscreen_drag_rect = None
        self.backtest_fullscreen_drag_start_x = None

        x1, x2 = sorted((start_x, end_x))
        if x2 - x1 < 12:
            self._show_backtest_point_info(canvas, event.x, self.backtest_fullscreen_zoom)
            return
        points = payload.get("points", [])
        total = len(points)
        if total <= 2 or right <= left:
            return
        current_start, current_end = self.backtest_fullscreen_zoom or (0, total)
        current_len = max(2, current_end - current_start)
        rel1 = (x1 - left) / (right - left)
        rel2 = (x2 - left) / (right - left)
        new_start = current_start + int(rel1 * current_len)
        new_end = current_start + int(rel2 * current_len)
        new_start = max(0, min(total - 2, new_start))
        new_end = max(new_start + 2, min(total, new_end))
        if new_end - new_start < 8:
            return
        self.backtest_fullscreen_zoom = (new_start, new_end)
        self._draw_backtest_fullscreen_payload()
        self.status_var.set(f"全屏图已按框选范围放大：{new_start + 1}-{new_end}/{total}")

    def _on_backtest_fullscreen_pan_start(self, event: tk.Event) -> None:
        canvas = self.backtest_fullscreen_canvas
        if not self.backtest_chart_payload or canvas is None:
            return
        total = len(self.backtest_chart_payload.get("points", []))
        self.backtest_fullscreen_pan_start_x = int(event.x)
        self.backtest_fullscreen_pan_start_zoom = self.backtest_fullscreen_zoom or (0, total)
        canvas.configure(cursor="fleur")

    def _on_backtest_fullscreen_pan_move(self, event: tk.Event) -> None:
        canvas = self.backtest_fullscreen_canvas
        if canvas is None:
            return
        zoom = self._pan_zoom_from_drag(canvas, self.backtest_fullscreen_pan_start_x, int(event.x), self.backtest_fullscreen_pan_start_zoom)
        if zoom is None:
            return
        self.backtest_fullscreen_zoom = zoom
        self._draw_backtest_fullscreen_payload()

    def _on_backtest_fullscreen_pan_release(self, _event: tk.Event) -> None:
        canvas = self.backtest_fullscreen_canvas
        self.backtest_fullscreen_pan_start_x = None
        self.backtest_fullscreen_pan_start_zoom = None
        if canvas is not None:
            canvas.configure(cursor="")
        if self.backtest_fullscreen_zoom and self.backtest_chart_payload:
            total = len(self.backtest_chart_payload.get("points", []))
            self.status_var.set(f"全屏图已平移到：{self.backtest_fullscreen_zoom[0] + 1}-{self.backtest_fullscreen_zoom[1]}/{total}")

    def _on_monitor_select(self, _event: object | None = None) -> None:
        self._on_monitor_saved_stock_select(_event)

    def _open_monitor_xueqiu(self, _event: object | None = None) -> None:
        symbol = self.selected_monitor_symbol
        selection = self.saved_stock_tree.selection() if hasattr(self, "saved_stock_tree") else ()
        if selection:
            symbol = selection[0]
        if not symbol:
            return
        item = self.monitor_items.get(symbol, {})
        webbrowser.open(str(item.get("xueqiu_url") or engine.xueqiu_url(symbol)))

    def _on_ml_monitor_select(self, _event: object | None = None) -> None:
        selection = self.ml_monitor_tree.selection()
        if not selection:
            return
        self.selected_ml_monitor_symbol = selection[0]
        self._render_ml_monitor_strategy_list(self.selected_ml_monitor_symbol)
        item = self.ml_monitor_items.get(self.selected_ml_monitor_symbol)
        if item:
            self._draw_ml_intraday_chart(item)

    def _open_ml_monitor_xueqiu(self, _event: object | None = None) -> None:
        symbol = self.selected_ml_monitor_symbol
        selection = self.ml_monitor_tree.selection()
        if selection:
            symbol = selection[0]
        if not symbol:
            return
        item = self.ml_monitor_items.get(symbol, {})
        webbrowser.open(str(item.get("xueqiu_url") or engine.xueqiu_url(symbol)))

    def _write_ml_detail(self, lines: list[str] | tuple[str, ...] | str) -> None:
        if not hasattr(self, "ml_detail_text"):
            return
        text = "\n".join(str(line) for line in lines) if isinstance(lines, (list, tuple)) else str(lines)
        self.ml_detail_text.configure(state=tk.NORMAL)
        self.ml_detail_text.delete("1.0", tk.END)
        self.ml_detail_text.insert("1.0", text)
        self.ml_detail_text.configure(state=tk.DISABLED)

    def _draw_ml_prediction_chart(self, result: dict[str, Any], update_detail: bool = True) -> None:
        _ui_debug(f"draw_ml_chart_enter update_detail={int(update_detail)} symbol={result.get('symbol', '')}")
        canvas = self.ml_canvas
        canvas.delete("all")
        data: pd.DataFrame = result["data"]
        prediction: dict[str, Any] = result["prediction"]
        width = canvas.winfo_width() if canvas.winfo_width() > 180 else 720
        height = canvas.winfo_height() if canvas.winfo_height() > 180 else 420
        pad_left, pad_right, pad_top, pad_bottom = 62, 36, 46, 44
        close = data["Close"].tail(180)
        ma20 = data["Close"].rolling(20).mean().reindex(close.index)
        ma60 = data["Close"].rolling(60).mean().reindex(close.index)
        points = [
            {"time": idx.strftime("%Y-%m-%d"), "price": float(value), "ma20": float(ma20.loc[idx]), "ma60": float(ma60.loc[idx])}
            for idx, value in close.items()
        ]
        values = [p["price"] for p in points]
        values += [p["ma20"] for p in points if np.isfinite(p["ma20"])]
        values += [p["ma60"] for p in points if np.isfinite(p["ma60"])]
        if not points or not values:
            canvas.create_text(width / 2, height / 2, text="暂无ML预测图", fill="#607086", font=("Microsoft YaHei", 14))
            return
        low, high = min(values), max(values)
        if high == low:
            high += 1
            low -= 1
        margin = (high - low) * 0.10
        high += margin
        low -= margin
        x_at, y_at = self._chart_scale(points, width, height, pad_left, pad_right, pad_top, pad_bottom, low, high)
        self._draw_axes(canvas, width, height, pad_left, pad_right, pad_top, pad_bottom, low, high)
        self._draw_series(canvas, points, "ma60", x_at, y_at, "#10b981", 2)
        self._draw_series(canvas, points, "ma20", x_at, y_at, "#f97316", 2)
        self._draw_series(canvas, points, "price", x_at, y_at, "#1464f4", 3)
        canvas.create_text(pad_left, pad_top - 20, text=f"{result['symbol']} {result['name']} ML持仓决策", fill="#14213d", anchor="w", font=("Microsoft YaHei", 12, "bold"))
        canvas.create_text(width - pad_right, pad_top - 20, text="蓝=收盘  橙=20日线  绿=60日线", fill="#607086", anchor="e")
        canvas.create_text(pad_left, height - 18, text=points[0]["time"], fill="#607086", anchor="w")
        canvas.create_text(width - pad_right, height - 18, text=points[-1]["time"], fill="#607086", anchor="e")

        factor = prediction.get("factor", {}) if isinstance(prediction.get("factor"), dict) else {}
        anomaly = prediction.get("anomaly", {}) if isinstance(prediction.get("anomaly"), dict) else {}
        mc = prediction.get("monte_carlo", {}) if isinstance(prediction.get("monte_carlo"), dict) else {}
        holding = prediction.get("holding_risk", {}) if isinstance(prediction.get("holding_risk"), dict) else {}
        news = prediction.get("news_sentiment", {}) if isinstance(prediction.get("news_sentiment"), dict) else {}
        days = self._ml_selected_advice_days()
        advice = self._ml_period_advice(result, days)
        decision_row = result.get("decision_row", {}) if isinstance(result.get("decision_row"), dict) else {}
        metadata = prediction.get("model_metadata", {}) if isinstance(prediction.get("model_metadata"), dict) else {}
        events = prediction.get("event_status", {}) if isinstance(prediction.get("event_status"), dict) else {}
        group_weights = prediction.get("group_weights", {}) if isinstance(prediction.get("group_weights"), dict) else {}
        group_predictions = prediction.get("group_predictions", {}) if isinstance(prediction.get("group_predictions"), dict) else {}
        availability = decision_row.get("data_availability", {}) if isinstance(decision_row.get("data_availability"), dict) else {}
        positive = prediction.get("top_positive_factor_details", [])
        negative = prediction.get("top_negative_factor_details", [])
        training_window = metadata.get("latest_window", {}) if isinstance(metadata.get("latest_window"), dict) else {}
        period_label = "次日" if days == 1 else f"{days}日风险投影"
        lines = [
            f"正式模型版本：{prediction.get('model_version', '-')}",
            f"训练截止：{training_window.get('train_end', decision_row.get('training_end', '-'))}",
            f"数据日期：{prediction.get('date', '-')}",
            f"模型后端：{prediction.get('backend', '-')}",
            f"最近样本外AUC：{self._fmt_number((prediction.get('metrics') or {}).get('auc'), digits=3) if isinstance(prediction.get('metrics'), dict) else '-'}",
            f"模型置信度：{self._fmt_number(float(prediction.get('confidence_score') or np.nan) * 100)}% ({prediction.get('confidence_level', '-')})",
            f"数据完整度：{self._fmt_number(float(prediction.get('data_completeness_score') or np.nan) * 100)}%",
            f"外部数据：{self._ml_external_source_summary(result)}",
            f"持仓风险：{holding.get('level', '-')}",
            f"{period_label}建议：{self._display_ml_action(result.get('rebalance_action') or advice.get('action', '-'), self._ml_position_shares_for_result(result))}",
            f"请求动作：{decision_row.get('requested_action', '-')}；实际可执行：{decision_row.get('effective_action', '-')}",
            f"当前仓位：{self._fmt_number(result.get('current_weight'))}%；交易股数：{int(result.get('trade_shares') or 0)}",
            f"目标仓位：{self._fmt_number(result.get('target_weight'))}%",
            f"现金/缓冲：{self._fmt_number(result.get('cash_reserve', 0))}%",
            f"风险分：{self._fmt_number(prediction.get('risk_score'))}",
            f"异常：{anomaly.get('level', '-')}",
            f"新闻风险：{news.get('level', '-')}",
            f"波动：{self._fmt_number(factor.get('atr_pct'))}%",
            f"{period_label}上涨：{self._fmt_number(advice.get('prob_pct'))}%",
            f"{period_label}预期：{self._fmt_number(advice.get('expected_return_pct'))}%",
            f"次日下跌2%以上概率：{self._fmt_number(float(decision_row.get('probability_down_2pct') or np.nan) * 100)}%",
            f"因子分：{self._fmt_number(factor.get('score'))}",
            f"展示-模拟10日上涨：{self._fmt_number(float(mc.get('up_prob', np.nan)) * 100)}%",
            f"展示-VaR 95%：{self._fmt_number(mc.get('var_95_pct'))}%",
            f"周期说明：{advice.get('detail', '-')}",
            f"决策说明：{holding.get('detail', '-')}",
            f"新闻/公告：{news.get('detail', '-')}",
        ]
        lines.extend(["", "数据源状态："])
        lines.extend(f"- {key}: {'可用' if value else '缺失/失败'}" for key, value in availability.items())
        lines.extend(["", "事件门控："])
        lines.extend(f"- {key}: {value}" for key, value in events.items())
        lines.extend(["", "当前因子组权重 / 预测："])
        for group, weight in group_weights.items():
            pack = group_predictions.get(group, {}) if isinstance(group_predictions.get(group), dict) else {}
            probability = self._fmt_number(float(pack.get("probability_up", np.nan)) * 100)
            expected = self._fmt_number(float(pack.get("expected_return", np.nan)) * 100)
            lines.append(f"- {group}: 权重 {float(weight) * 100:.1f}%，上涨 {probability}%，预期 {expected}%")
        lines.extend(["", "主要正向贡献："])
        lines.extend(
            f"- {item.get('factor_name')} [{item.get('factor_group')}], 值 {self._fmt_number(item.get('factor_value'))}, SHAP {self._fmt_number(item.get('shap_value'), digits=5)}"
            for item in positive
            if isinstance(item, dict)
        )
        lines.extend(["", "主要负向贡献："])
        lines.extend(
            f"- {item.get('factor_name')} [{item.get('factor_group')}], 值 {self._fmt_number(item.get('factor_value'))}, SHAP {self._fmt_number(item.get('shap_value'), digits=5)}"
            for item in negative
            if isinstance(item, dict)
        )
        if not positive and not negative:
            lines.append("- 当前样本 SHAP 不可用，仅保留无方向的重要因子。")
        if update_detail:
            self._write_ml_detail(lines)
        _ui_debug(f"draw_ml_chart_done update_detail={int(update_detail)} symbol={result.get('symbol', '')}")

    def _draw_intraday_chart(self, item: dict[str, Any]) -> None:
        points = item.get("chart_points") or []
        self._draw_line_chart(
            self.monitor_canvas,
            points,
            title=f"{item.get('symbol', '')} {item.get('name', '')}  {item.get('action', '')}  {item.get('chart_strategy_label', '')}",
            stop_value=item.get("stop_value"),
            stop_label=str(item.get("stop_line", "")),
            action_code=str(item.get("action_code", "")),
            strategy_type=str(item.get("chart_strategy_type", "")),
        )

    def _draw_ml_intraday_chart(self, item: dict[str, Any]) -> None:
        points = item.get("chart_points") or []
        self._draw_line_chart(
            self.ml_monitor_canvas,
            points,
            title=f"{item.get('symbol', '')} {item.get('name', '')}  {item.get('action', '')}  {item.get('chart_strategy_label', '')}",
            stop_value=item.get("stop_value"),
            stop_label=str(item.get("stop_line", "")),
            action_code=str(item.get("action_code", "")),
            strategy_type=str(item.get("chart_strategy_type", "")),
        )

    def _draw_result_chart_on_canvas(self, canvas: tk.Canvas, result: dict[str, Any], row: pd.Series | None = None) -> None:
        data: pd.DataFrame = result["data"]
        horizon = str(result.get("horizon", "short"))
        cash = float(result["cash"])
        fee = float(result["fee"])
        selected = row if row is not None else result.get("best")
        if isinstance(selected, pd.Series):
            fast = int(selected["fast"])
            slow = int(selected["slow"])
            strategy_type = str(selected.get("strategy_type", "sma"))
            fast_line, slow_line, entries, exits = engine.strategy_signals(data, fast, slow, horizon, strategy_type)
            trades: pd.DataFrame = engine.strategy_portfolio(data, entries, exits, cash, fee, horizon).trades.records_readable
        else:
            fast_line = result["fast_line"]
            slow_line = result["slow_line"]
            trades = result["trades"]
            fast = int(result["best"]["fast"])
            slow = int(result["best"]["slow"])
            strategy_type = str(result["best"].get("strategy_type", "sma"))

        plot_fast = data["Close"].rolling(fast).mean() if strategy_type in {"ml", "hybrid"} else fast_line
        plot_slow = data["Close"].rolling(slow).mean() if strategy_type in {"ml", "hybrid"} else slow_line
        frame = pd.DataFrame({"price": data["Close"], "fast": plot_fast, "slow": plot_slow}).dropna()
        points = [
            {
                "time": idx.strftime("%Y-%m-%d"),
                "price": round(float(values["price"]), 3),
                "fast": round(float(values["fast"]), 3),
                "slow": round(float(values["slow"]), 3),
            }
            for idx, values in frame.iterrows()
        ]

        indicator_frame = pd.DataFrame(index=frame.index)
        indicator_type = strategy_type
        if strategy_type == "rsi":
            indicator_frame["rsi"] = engine.rsi(data["Close"], max(2, fast)).reindex(frame.index)
            indicator_frame["low"] = 35.0
            indicator_frame["high"] = 72.0
        elif strategy_type == "macd":
            macd_line, signal_line, _entries, _exits = engine.make_macd_signals(data, fast, slow)
            indicator_frame["macd"] = macd_line.reindex(frame.index)
            indicator_frame["signal"] = signal_line.reindex(frame.index)
            indicator_frame["hist"] = indicator_frame["macd"] - indicator_frame["signal"]
        elif strategy_type == "ml":
            indicator_frame["prob"] = fast_line.reindex(frame.index)
            indicator_frame["buy"] = indicator_frame["prob"].rolling(160, min_periods=60).quantile(0.66).clip(lower=0.45, upper=0.58)
            indicator_frame["sell"] = indicator_frame["prob"].rolling(160, min_periods=60).quantile(0.32).clip(lower=0.36, upper=0.48)
        else:
            indicator_type = "spread"
            indicator_frame["spread"] = (fast_line - slow_line).reindex(frame.index)
            indicator_frame["zero"] = 0.0
        indicator_points = [
            {"time": idx.strftime("%Y-%m-%d"), **{key: round(float(value), 4) for key, value in values.items()}}
            for idx, values in indicator_frame.dropna().iterrows()
        ]

        buys: list[dict[str, Any]] = []
        sells: list[dict[str, Any]] = []
        if not trades.empty:
            for _, trade in trades.iterrows():
                entry_time = pd.Timestamp(trade["Entry Timestamp"])
                buys.append({"time": entry_time.strftime("%Y-%m-%d"), "price": float(trade["Avg Entry Price"])})
                if str(trade["Status"]) == "Closed":
                    exit_time = pd.Timestamp(trade["Exit Timestamp"])
                    sells.append({"time": exit_time.strftime("%Y-%m-%d"), "price": float(trade["Avg Exit Price"])})

        label = engine.STRATEGY_TYPES.get(strategy_type, strategy_type)
        title = f"{result['symbol']} {result['name']} {label} {fast}/{slow} 历史买卖点"
        if len(points) > 760:
            visible = [points[i] for i in np.linspace(0, len(points) - 1, 760).astype(int)]
            times = {str(p.get("time")) for p in visible}
            buys = [b for b in buys if str(b.get("time")) in times]
            sells = [s for s in sells if str(s.get("time")) in times]
            indicator_points = [p for p in indicator_points if str(p.get("time")) in times]
            points = visible
        self._draw_backtest_canvas(points, buys, sells, title, indicator_points, indicator_type, canvas=canvas)

    def _draw_backtest_chart(self, result: dict[str, Any], row: pd.Series | None = None) -> None:
        data: pd.DataFrame = result["data"]
        horizon = str(result.get("horizon", "short"))
        cash = float(result["cash"])
        fee = float(result["fee"])
        selected = row if row is not None else result.get("best")

        if isinstance(selected, pd.Series):
            fast = int(selected["fast"])
            slow = int(selected["slow"])
            strategy_type = str(selected.get("strategy_type", "sma"))
            fast_line, slow_line, entries, exits = engine.strategy_signals(data, fast, slow, horizon, strategy_type)
            trades: pd.DataFrame = engine.strategy_portfolio(data, entries, exits, cash, fee, horizon).trades.records_readable
        else:
            fast_line = result["fast_line"]
            slow_line = result["slow_line"]
            trades = result["trades"]
            fast = int(result["best"]["fast"])
            slow = int(result["best"]["slow"])
            strategy_type = str(result["best"].get("strategy_type", "sma"))

        plot_fast = data["Close"].rolling(fast).mean() if strategy_type in {"ml", "hybrid"} else fast_line
        plot_slow = data["Close"].rolling(slow).mean() if strategy_type in {"ml", "hybrid"} else slow_line
        frame = pd.DataFrame({"price": data["Close"], "fast": plot_fast, "slow": plot_slow}).dropna()
        points = [
            {
                "time": idx.strftime("%Y-%m-%d"),
                "price": round(float(values["price"]), 3),
                "fast": round(float(values["fast"]), 3),
                "slow": round(float(values["slow"]), 3),
            }
            for idx, values in frame.iterrows()
        ]

        indicator_frame = pd.DataFrame(index=frame.index)
        indicator_type = strategy_type
        if strategy_type == "rsi":
            indicator_frame["rsi"] = engine.rsi(data["Close"], max(2, fast)).reindex(frame.index)
            indicator_frame["low"] = 35.0
            indicator_frame["high"] = 72.0
        elif strategy_type == "macd":
            macd_line, signal_line, _entries, _exits = engine.make_macd_signals(data, fast, slow)
            indicator_frame["macd"] = macd_line.reindex(frame.index)
            indicator_frame["signal"] = signal_line.reindex(frame.index)
            indicator_frame["hist"] = indicator_frame["macd"] - indicator_frame["signal"]
        elif strategy_type == "ml":
            indicator_frame["prob"] = fast_line.reindex(frame.index)
            indicator_frame["buy"] = indicator_frame["prob"].rolling(160, min_periods=60).quantile(0.66).clip(lower=0.45, upper=0.58)
            indicator_frame["sell"] = indicator_frame["prob"].rolling(160, min_periods=60).quantile(0.32).clip(lower=0.36, upper=0.48)
        else:
            indicator_type = "spread"
            indicator_frame["spread"] = (fast_line - slow_line).reindex(frame.index)
            indicator_frame["zero"] = 0.0
        indicator_frame = indicator_frame.dropna()
        indicator_points = [
            {
                "time": idx.strftime("%Y-%m-%d"),
                **{key: round(float(value), 4) for key, value in values.items()},
            }
            for idx, values in indicator_frame.iterrows()
        ]

        buys: list[dict[str, Any]] = []
        sells: list[dict[str, Any]] = []
        if not trades.empty:
            for _, trade in trades.iterrows():
                entry_time = pd.Timestamp(trade["Entry Timestamp"])
                buys.append({"time": entry_time.strftime("%Y-%m-%d"), "price": float(trade["Avg Entry Price"])})
                if str(trade["Status"]) == "Closed":
                    exit_time = pd.Timestamp(trade["Exit Timestamp"])
                    sells.append({"time": exit_time.strftime("%Y-%m-%d"), "price": float(trade["Avg Exit Price"])})

        label = engine.STRATEGY_TYPES.get(strategy_type, strategy_type)
        self.backtest_chart_payload = {
            "points": points,
            "buys": buys,
            "sells": sells,
            "title": f"{result['symbol']} {result['name']} {label} {fast}/{slow} 历史买卖点",
            "indicator_points": indicator_points,
            "indicator_type": indicator_type,
        }
        self.backtest_zoom = None
        self.backtest_fullscreen_zoom = None
        self._draw_backtest_payload()
        self._draw_backtest_fullscreen_payload()

    def _draw_backtest_payload(self) -> None:
        self._draw_backtest_payload_on_canvas(self.backtest_canvas, self.backtest_zoom)

    def _backtest_visible_payload(self, zoom: tuple[int, int] | None = None) -> dict[str, Any] | None:
        payload = self.backtest_chart_payload
        if not payload:
            return None
        points: list[dict[str, Any]] = list(payload.get("points", []))
        total = len(points)
        start, end = zoom or (0, total)
        start = max(0, min(start, total))
        end = max(start, min(end, total))
        visible = points[start:end]
        if len(visible) > 760:
            visible = [visible[i] for i in np.linspace(0, len(visible) - 1, 760).astype(int)]
        visible_times = {str(p.get("time")) for p in visible}
        buys = [b for b in payload.get("buys", []) if str(b.get("time")) in visible_times]
        sells = [s for s in payload.get("sells", []) if str(s.get("time")) in visible_times]
        indicator_points = [p for p in payload.get("indicator_points", []) if str(p.get("time")) in visible_times]
        title = str(payload.get("title", "历史买卖点"))
        if total and (start > 0 or end < total):
            title = f"{title}  [{start + 1}-{end}/{total}]"
        return {
            "points": visible,
            "buys": buys,
            "sells": sells,
            "indicator_points": indicator_points,
            "indicator_type": str(payload.get("indicator_type", "")),
            "title": title,
            "start": start,
            "end": end,
            "total": total,
        }

    def _draw_backtest_payload_on_canvas(self, canvas: tk.Canvas, zoom: tuple[int, int] | None = None) -> None:
        visible_payload = self._backtest_visible_payload(zoom)
        if not visible_payload:
            self._draw_backtest_canvas([], [], [], "暂无回测曲线", canvas=canvas)
            return
        self._draw_backtest_canvas(
            visible_payload["points"],
            visible_payload["buys"],
            visible_payload["sells"],
            visible_payload["title"],
            visible_payload["indicator_points"],
            visible_payload["indicator_type"],
            canvas=canvas,
        )

    def _draw_line_chart(
        self,
        canvas: tk.Canvas,
        points: list[dict[str, Any]],
        title: str,
        stop_value: Any = None,
        stop_label: str = "",
        action_code: str = "",
        strategy_type: str = "",
    ) -> None:
        canvas.delete("all")
        width = canvas.winfo_width() if canvas.winfo_width() > 180 else 480
        height = canvas.winfo_height() if canvas.winfo_height() > 180 else 430
        pad_left, pad_right, pad_top, pad_bottom = 62, 24, 34, 58
        if not points:
            canvas.create_text(width / 2, height / 2, text="暂无曲线", fill="#607086", font=("Microsoft YaHei", 14))
            return

        has_indicator = strategy_type in {"rsi", "macd"} and any(
            isinstance(point.get("rsi" if strategy_type == "rsi" else "macd"), (int, float)) for point in points
        )
        price_bottom = int(height * 0.62) if has_indicator else height - pad_bottom
        price_pad_bottom = height - price_bottom
        values: list[float] = []
        for point in points:
            for key in ("price", "vwap", "strat_fast", "strat_slow"):
                if isinstance(point.get(key), (int, float)):
                    values.append(float(point[key]))
        if isinstance(stop_value, (int, float)):
            values.append(float(stop_value))
        low, high = min(values), max(values)
        if high == low:
            high += 1
            low -= 1
        margin = (high - low) * 0.16
        high += margin
        low -= margin
        x_at, y_at = self._chart_scale(points, width, height, pad_left, pad_right, pad_top, price_pad_bottom, low, high)

        self._draw_axes(canvas, width, height, pad_left, pad_right, pad_top, price_pad_bottom, low, high)
        if isinstance(stop_value, (int, float)):
            y = y_at(float(stop_value))
            self._dashed_line(canvas, pad_left, y, width - pad_right, y, "#bf2f2f")
            canvas.create_text(width - 88, y - 10, text=f"风控 {stop_label}", fill="#bf2f2f")

        self._draw_series(canvas, points, "vwap", x_at, y_at, "#f97316", 3)
        self._draw_series(canvas, points, "strat_slow", x_at, y_at, "#10b981", 3)
        self._draw_series(canvas, points, "strat_fast", x_at, y_at, "#7c3aed", 3)
        self._draw_series(canvas, points, "price", x_at, y_at, "#1464f4", 4)
        last = points[-1]
        last_x = x_at(len(points) - 1)
        last_y = y_at(float(last.get("price", values[-1])))
        color = "#0f8f61" if action_code == "buy" else "#bf2f2f" if action_code == "sell" else "#1464f4"
        canvas.create_oval(last_x - 5, last_y - 5, last_x + 5, last_y + 5, fill=color, outline="white", width=2)
        marker = "买" if action_code == "buy" else "卖" if action_code == "sell" else ""
        if marker:
            canvas.create_rectangle(last_x - 18, last_y - 36, last_x + 18, last_y - 12, fill=color, outline=color)
            canvas.create_text(last_x, last_y - 24, text=marker, fill="white", font=("Microsoft YaHei", 12, "bold"))

        canvas.create_text(pad_left, pad_top - 14, text=title, fill="#14213d", anchor="w", font=("Microsoft YaHei", 12, "bold"))
        canvas.create_text(width - pad_right, pad_top - 14, text="蓝=价格  橙=VWAP  紫=策略快线  绿=策略慢线", fill="#607086", anchor="e")

        if has_indicator:
            panel_top = price_bottom + 34
            panel_bottom = height - pad_bottom
            canvas.create_line(pad_left, panel_top, pad_left, panel_bottom, fill="#d8e0ea")
            canvas.create_line(pad_left, panel_bottom, width - pad_right, panel_bottom, fill="#d8e0ea")
            if strategy_type == "rsi":
                raw = [float(point["rsi"]) for point in points if isinstance(point.get("rsi"), (int, float))]
                ind_low, ind_high = 0.0, 100.0
                legend = "RSI  蓝=RSI  绿=35  红=72"
            else:
                keys = ("macd", "signal", "hist")
                raw = [float(point[key]) for point in points for key in keys if isinstance(point.get(key), (int, float))]
                ind_low, ind_high = min(raw), max(raw)
                margin = (ind_high - ind_low) * 0.12 if ind_high != ind_low else 1.0
                ind_low -= margin
                ind_high += margin
                legend = "MACD  蓝=MACD  橙=Signal  灰=柱体"
            if ind_high == ind_low:
                ind_high += 1
                ind_low -= 1

            def ind_y(value: float) -> float:
                return panel_top + (ind_high - value) * (panel_bottom - panel_top) / (ind_high - ind_low)

            def draw_indicator_line(key: str, color: str, width_px: int = 1) -> None:
                coords: list[float] = []
                for idx, point in enumerate(points):
                    if isinstance(point.get(key), (int, float)):
                        coords.extend([x_at(idx), ind_y(float(point[key]))])
                if len(coords) >= 4:
                    canvas.create_line(*coords, fill=color, width=width_px)

            canvas.create_text(pad_left, panel_top - 18, text=legend, fill="#14213d", anchor="w", font=("Microsoft YaHei", 11, "bold"))
            canvas.create_text(pad_left, panel_top + 2, text=f"{ind_high:.2f}", fill="#607086", anchor="w")
            canvas.create_text(pad_left, panel_bottom - 2, text=f"{ind_low:.2f}", fill="#607086", anchor="w")
            if strategy_type == "rsi":
                draw_indicator_line("rsi_low", "#16a34a", 2)
                draw_indicator_line("rsi_high", "#dc2626", 2)
                draw_indicator_line("rsi", "#2563eb", 3)
            else:
                zero_y = ind_y(0.0)
                canvas.create_line(pad_left, zero_y, width - pad_right, zero_y, fill="#cbd5e1")
                bar_width = max(2, (width - pad_left - pad_right) / max(1, len(points)) * 0.55)
                for idx, point in enumerate(points):
                    if not isinstance(point.get("hist"), (int, float)):
                        continue
                    x = x_at(idx)
                    y = ind_y(float(point["hist"]))
                    canvas.create_rectangle(x - bar_width, min(y, zero_y), x + bar_width, max(y, zero_y), fill="#94a3b8", outline="")
                draw_indicator_line("macd", "#2563eb", 3)
                draw_indicator_line("signal", "#f97316", 3)
        canvas.create_text(pad_left, height - 18, text=str(points[0].get("time", "")), fill="#607086", anchor="w")
        canvas.create_text(width - pad_right, height - 18, text=str(points[-1].get("time", "")), fill="#607086", anchor="e")

    def _draw_backtest_canvas(
        self,
        points: list[dict[str, Any]],
        buys: list[dict[str, Any]],
        sells: list[dict[str, Any]],
        title: str,
        indicator_points: list[dict[str, Any]] | None = None,
        indicator_type: str = "",
        canvas: tk.Canvas | None = None,
    ) -> None:
        canvas = canvas or self.backtest_canvas
        canvas.delete("all")
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        width = canvas_width if canvas_width > 100 else 900
        height = canvas_height if canvas_height > 140 else 420
        pad_left, pad_right, pad_top, pad_bottom = 62, 24, 34, 44
        if not points:
            canvas.create_text(width / 2, height / 2, text="暂无回测曲线", fill="#607086", font=("Microsoft YaHei", 14))
            return
        has_indicator = bool(indicator_points)
        price_bottom = int(height * 0.62) if has_indicator else height - pad_bottom
        price_pad_bottom = height - price_bottom
        values = [float(p["price"]) for p in points]
        values += [float(p["price"]) for p in buys + sells]
        low, high = min(values), max(values)
        if high == low:
            high += 1
            low -= 1
        margin = (high - low) * 0.08
        high += margin
        low -= margin
        x_at, y_at = self._chart_scale(points, width, height, pad_left, pad_right, pad_top, price_pad_bottom, low, high)
        time_to_x = {p["time"]: x_at(i) for i, p in enumerate(points)}

        self._draw_axes(canvas, width, height, pad_left, pad_right, pad_top, price_pad_bottom, low, high)
        self._draw_time_ticks(canvas, points, x_at, width, height, pad_left, pad_right, pad_bottom)
        self._draw_series(canvas, points, "slow", x_at, y_at, "#10b981", 1)
        self._draw_series(canvas, points, "fast", x_at, y_at, "#f97316", 1)
        self._draw_series(canvas, points, "price", x_at, y_at, "#1464f4", 2)

        for buy in buys:
            x = self._nearest_time_x(buy["time"], points, time_to_x, x_at)
            y = y_at(float(buy["price"]))
            self._triangle(canvas, x, y, "#0f8f61", up=True)
        for sell in sells:
            x = self._nearest_time_x(sell["time"], points, time_to_x, x_at)
            y = y_at(float(sell["price"]))
            self._triangle(canvas, x, y, "#bf2f2f", up=False)

        canvas.create_text(pad_left, pad_top - 16, text=title, fill="#14213d", anchor="w", font=("Microsoft YaHei", 12, "bold"))
        canvas.create_text(width - pad_right, pad_top - 16, text="蓝=收盘  橙=快线  绿=慢线  ▲买 ▼卖", fill="#607086", anchor="e")

        if indicator_points:
            panel_top = price_bottom + 34
            panel_bottom = height - pad_bottom
            canvas.create_line(pad_left, panel_top, pad_left, panel_bottom, fill="#d8e0ea")
            canvas.create_line(pad_left, panel_bottom, width - pad_right, panel_bottom, fill="#d8e0ea")
            if indicator_type == "rsi":
                keys = ("rsi",)
                ind_low, ind_high = 0.0, 100.0
                legend = "RSI  蓝=RSI  绿=35买入区  红=72过热区"
            elif indicator_type == "macd":
                keys = ("macd", "signal", "hist")
                raw = [float(point[key]) for point in indicator_points for key in keys if isinstance(point.get(key), (int, float))]
                ind_low, ind_high = min(raw), max(raw)
                legend = "MACD  蓝=MACD  橙=Signal  灰=柱体"
            elif indicator_type == "ml":
                keys = ("prob", "buy", "sell")
                raw = [float(point[key]) for point in indicator_points for key in keys if isinstance(point.get(key), (int, float))]
                ind_low, ind_high = 0.0, 1.0
                if raw:
                    ind_low = max(0.0, min(raw) - 0.06)
                    ind_high = min(1.0, max(raw) + 0.06)
                legend = "ML  蓝=上涨概率  绿=买入阈值  红=卖出阈值"
            else:
                keys = ("spread", "zero")
                raw = [float(point[key]) for point in indicator_points for key in keys if isinstance(point.get(key), (int, float))]
                ind_low, ind_high = min(raw), max(raw)
                legend = "指标副图  蓝=快慢线差值  灰=零轴"
            if ind_high == ind_low:
                ind_high += 1
                ind_low -= 1
            if indicator_type != "rsi":
                ind_margin = (ind_high - ind_low) * 0.12
                ind_high += ind_margin
                ind_low -= ind_margin

            def ind_y(value: float) -> float:
                return panel_top + (ind_high - value) * (panel_bottom - panel_top) / (ind_high - ind_low)

            canvas.create_text(pad_left, panel_top - 18, text=legend, fill="#14213d", anchor="w", font=("Microsoft YaHei", 11, "bold"))
            canvas.create_text(pad_left, panel_top + 2, text=f"{ind_high:.2f}", fill="#607086", anchor="w")
            canvas.create_text(pad_left, panel_bottom - 2, text=f"{ind_low:.2f}", fill="#607086", anchor="w")

            def draw_indicator_line(key: str, color: str, width_px: int = 1) -> None:
                coords: list[float] = []
                for point in indicator_points or []:
                    if point.get("time") in time_to_x and isinstance(point.get(key), (int, float)):
                        coords.extend([time_to_x[point["time"]], ind_y(float(point[key]))])
                if len(coords) >= 4:
                    canvas.create_line(*coords, fill=color, width=width_px)

            if indicator_type == "rsi":
                draw_indicator_line("low", "#16a34a", 1)
                draw_indicator_line("high", "#dc2626", 1)
                draw_indicator_line("rsi", "#2563eb", 2)
            elif indicator_type == "macd":
                zero_y = ind_y(0.0)
                canvas.create_line(pad_left, zero_y, width - pad_right, zero_y, fill="#cbd5e1")
                bar_width = max(2, (width - pad_left - pad_right) / max(1, len(points)) * 0.55)
                for point in indicator_points:
                    if point.get("time") not in time_to_x or not isinstance(point.get("hist"), (int, float)):
                        continue
                    x = time_to_x[point["time"]]
                    y = ind_y(float(point["hist"]))
                    canvas.create_rectangle(x - bar_width, min(y, zero_y), x + bar_width, max(y, zero_y), fill="#94a3b8", outline="")
                draw_indicator_line("macd", "#2563eb", 2)
                draw_indicator_line("signal", "#f97316", 1)
            elif indicator_type == "ml":
                draw_indicator_line("buy", "#16a34a", 1)
                draw_indicator_line("sell", "#dc2626", 1)
                draw_indicator_line("prob", "#2563eb", 2)
            else:
                draw_indicator_line("zero", "#94a3b8", 1)
                draw_indicator_line("spread", "#2563eb", 2)

    def _draw_time_ticks(
        self,
        canvas: tk.Canvas,
        points: list[dict[str, Any]],
        x_at,
        width: int,
        height: int,
        pad_left: int,
        pad_right: int,
        pad_bottom: int,
    ) -> None:
        if not points:
            return
        tick_count = max(4, min(9, int((width - pad_left - pad_right) / 115)))
        if len(points) <= tick_count:
            indexes = list(range(len(points)))
        else:
            indexes = sorted(set(int(i) for i in np.linspace(0, len(points) - 1, tick_count)))
        axis_bottom = height - pad_bottom
        for idx in indexes:
            x = x_at(idx)
            label = str(points[idx].get("time", ""))
            canvas.create_line(x, axis_bottom, x, axis_bottom + 5, fill="#cbd5e1")
            canvas.create_text(x, height - 18, text=label, fill="#607086", anchor="center", font=("Microsoft YaHei", 8))

    def _show_backtest_point_info(self, canvas: tk.Canvas, x: int, zoom: tuple[int, int] | None = None) -> None:
        visible_payload = self._backtest_visible_payload(zoom)
        if not visible_payload:
            return
        points: list[dict[str, Any]] = list(visible_payload.get("points", []))
        if not points:
            return
        left, right, top, bottom = self._backtest_plot_bounds_for_canvas(canvas)
        if right <= left:
            return
        x = max(left, min(right, int(x)))
        idx = int(round((x - left) / (right - left) * (len(points) - 1))) if len(points) > 1 else 0
        idx = max(0, min(len(points) - 1, idx))
        point = points[idx]

        width = canvas.winfo_width() if canvas.winfo_width() > 100 else 900
        height = canvas.winfo_height() if canvas.winfo_height() > 140 else 420
        has_indicator = bool(visible_payload.get("indicator_points"))
        price_bottom = int(height * 0.62) if has_indicator else height - 44
        price_pad_bottom = height - price_bottom
        values = [float(p["price"]) for p in points]
        values += [float(p["price"]) for p in visible_payload.get("buys", []) + visible_payload.get("sells", [])]
        low, high = min(values), max(values)
        if high == low:
            high += 1
            low -= 1
        margin = (high - low) * 0.08
        high += margin
        low -= margin
        x_at, y_at = self._chart_scale(points, width, height, 62, 24, 34, price_pad_bottom, low, high)
        px = x_at(idx)
        py = y_at(float(point.get("price", 0)))

        indicator = next((item for item in visible_payload.get("indicator_points", []) if item.get("time") == point.get("time")), {})
        lines = [
            f"日期：{point.get('time', '-')}",
            f"收盘：{engine.money(float(point.get('price', 0)))}",
            f"快线：{engine.money(float(point.get('fast', 0)))}",
            f"慢线：{engine.money(float(point.get('slow', 0)))}",
        ]
        if "rsi" in indicator:
            lines.append(f"RSI：{float(indicator['rsi']):.2f}")
        elif "macd" in indicator:
            lines.append(f"MACD：{float(indicator['macd']):.4f} / Signal {float(indicator.get('signal', 0)):.4f}")
        elif "prob" in indicator:
            lines.append(f"ML概率：{float(indicator['prob']):.2f}")
        elif "spread" in indicator:
            lines.append(f"快慢差：{float(indicator['spread']):.4f}")

        canvas.delete("point_info")
        canvas.create_line(px, top, px, bottom, fill="#64748b", dash=(3, 3), tags="point_info")
        canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#1d5fd1", outline="white", width=2, tags="point_info")

        box_w = 250
        box_h = 24 + len(lines) * 20
        box_x1 = px + 12 if px < width - box_w - 24 else px - box_w - 12
        box_y1 = max(42, min(py - box_h / 2, height - box_h - 36))
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h
        canvas.create_rectangle(box_x1, box_y1, box_x2, box_y2, fill="#ffffff", outline="#94a3b8", width=1, tags="point_info")
        for i, line in enumerate(lines):
            canvas.create_text(box_x1 + 12, box_y1 + 16 + i * 20, text=line, fill="#14213d", anchor="w", font=("Microsoft YaHei", 10), tags="point_info")
        self.status_var.set(" | ".join(lines[:4]))

    def _chart_scale(self, points: list[dict[str, Any]], width: int, height: int, pl: int, pr: int, pt: int, pb: int, low: float, high: float):
        def x_at(index: int) -> float:
            if len(points) == 1:
                return pl
            return pl + index * (width - pl - pr) / (len(points) - 1)

        def y_at(value: float) -> float:
            return pt + (high - value) * (height - pt - pb) / (high - low)

        return x_at, y_at

    def _draw_axes(self, canvas: tk.Canvas, width: int, height: int, pl: int, pr: int, pt: int, pb: int, low: float, high: float) -> None:
        axis_bottom = height - pb
        canvas.create_line(pl, pt, pl, axis_bottom, fill="#d8e0ea")
        canvas.create_line(pl, axis_bottom, width - pr, axis_bottom, fill="#d8e0ea")
        canvas.create_text(pl, 16, text=f"{high:.2f}", fill="#607086", anchor="w")
        canvas.create_text(pl, axis_bottom - 2, text=f"{low:.2f}", fill="#607086", anchor="w")

    def _draw_series(self, canvas: tk.Canvas, points: list[dict[str, Any]], key: str, x_at, y_at, color: str, width: int) -> None:
        coords: list[float] = []
        step = max(1, len(points) // 900)
        sampled = list(enumerate(points))[::step]
        if sampled and sampled[-1][0] != len(points) - 1:
            sampled.append((len(points) - 1, points[-1]))
        for idx, point in sampled:
            if isinstance(point.get(key), (int, float)):
                coords.extend([x_at(idx), y_at(float(point[key]))])
        if len(coords) >= 4:
            canvas.create_line(*coords, fill=color, width=width)

    def _dashed_line(self, canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, fill: str) -> None:
        dash, gap, x = 8, 5, x1
        while x < x2:
            canvas.create_line(x, y1, min(x + dash, x2), y2, fill=fill)
            x += dash + gap

    def _triangle(self, canvas: tk.Canvas, x: float, y: float, color: str, up: bool) -> None:
        if up:
            points = [x, y - 8, x - 7, y + 7, x + 7, y + 7]
        else:
            points = [x, y + 8, x - 7, y - 7, x + 7, y - 7]
        canvas.create_polygon(points, fill=color, outline="white")

    def _nearest_time_x(self, target: str, points: list[dict[str, Any]], time_to_x: dict[str, float], x_at) -> float:
        if target in time_to_x:
            return time_to_x[target]
        try:
            target_ts = pd.Timestamp(target)
            dates = [pd.Timestamp(p["time"]) for p in points]
            idx = int(np.argmin([abs((d - target_ts).days) for d in dates]))
            return x_at(idx)
        except Exception:
            return x_at(0)


def main() -> None:
    try:
        app = StrategyDesktopApp()
        app.mainloop()
    except Exception:
        messagebox.showerror("程序错误", traceback.format_exc())


if __name__ == "__main__":
    mp.freeze_support()
    main()
