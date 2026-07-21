from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import desktop_strategy_app as desktop
from ml_decision.inference import ProductionDecisionResult
from scripts import monthly_train_desktop


def _daily_frame(offset: float, periods: int = 70) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    close = 10.0 + offset + np.linspace(0.0, 2.0, len(dates))
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=dates,
    )


def _decision_row(code: str, name: str, price: float) -> dict:
    return {
        "date": "2025-04-09",
        "code": code,
        "name": name,
        "shares": 0,
        "available_shares": 0,
        "average_cost": price,
        "current_price": price,
        "position_weight": 0.0,
        "industry": "",
        "probability_up": 0.62,
        "probability_profitable": 0.58,
        "probability_down_2pct": 0.10,
        "expected_open_to_next_open_return": 0.01,
        "expected_open_to_open_return": 0.01,
        "return_q10": -0.012,
        "return_q50": 0.005,
        "return_q90": 0.025,
        "recommended_target_weight": 0.10,
        "recommended_target_shares": 100,
        "recommended_trade_shares": 100,
        "recommended_action": "ADD_25",
        "requested_action": "ADD_25",
        "effective_action": "ADD_25",
        "display_score": 65.0,
        "utility_score": 0.01,
        "holding_risk_level": "未持仓",
        "reason": "production inference",
        "model_version": "ui-test",
        "data_completeness_score": 1.0,
        "confidence_score": 0.8,
        "confidence_level": "high",
        "event_status": {},
        "group_weights": {"technical": 1.0},
        "group_predictions": {},
        "data_availability": {"market_data_available": 1.0},
    }


def test_desktop_batch_uses_one_production_inference_call(monkeypatch, tmp_path: Path):
    frames = {"000001": _daily_frame(0.0, 250), "000002": _daily_frame(2.0, 250)}
    monkeypatch.setattr(desktop.engine, "cached_data", lambda code, _start, _adjust: frames[code])
    monkeypatch.setattr(
        desktop,
        "fetch_external_factor_frame",
        lambda _codes, force_refresh, market_df: (pd.DataFrame(), []),
    )
    monkeypatch.setattr(desktop, "_ml_display_snapshot", lambda _data: {"factor": {}, "anomaly": {}, "monte_carlo": {}})
    calls = []

    def fake_production(market, holdings, _root, **_kwargs):
        calls.append((market.copy(), holdings.copy()))
        rows = [
            _decision_row("000001", "A", float(frames["000001"]["Close"].iloc[-1])),
            _decision_row("000002", "B", float(frames["000002"]["Close"].iloc[-1])),
        ]
        return ProductionDecisionResult(
            table=pd.DataFrame(rows),
            metrics={"auc": 0.6},
            feature_columns=["ret_5"],
            source_notes=[],
            model_metadata={"latest_window": {"train_end": "2025-03-31"}},
            model_version="ui-test",
            snapshot_path=str(tmp_path / "snapshot.json"),
        )

    monkeypatch.setattr(desktop, "run_production_holding_decision", fake_production)
    payload = desktop._compute_ml_decision_payload(
        {
            "symbol": "000001",
            "batch_symbols": "000001 000002",
            "positions": {"000001": {"name": "A"}, "000002": {"name": "B"}},
            "cash": "100000",
            "target_position": "80",
        }
    )
    assert len(calls) == 1
    assert set(calls[0][0]["code"]) == {"000001", "000002"}
    assert calls[0][0].groupby("code").size().to_dict() == {"000001": 180, "000002": 180}
    assert len(payload["results"]) == 2
    assert payload["decision_meta"]["model_version"] == "ui-test"


def test_monthly_training_reads_current_nested_ml_pool(monkeypatch, tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ml_stock_pool.json").write_text(
        json.dumps(
            {
                "saved_at": "2026-07-17 21:00:00",
                "items": {
                    "002472": {"symbol": "002472", "name": "双环传动"},
                    "metadata": {"name": "ignored"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(monthly_train_desktop.engine, "CACHE_DIR", cache)

    assert monthly_train_desktop._pool_symbols() == [("002472", "双环传动")]


def test_ml_policy_report_requires_the_complete_strict_oos_bundle(tmp_path: Path):
    required = {
        "ml_policy_backtest.parquet",
        "ml_policy_stock_summary.csv",
        "ml_policy_report.json",
    }
    assert set(desktop.StrategyDesktopApp._missing_ml_policy_report_files(tmp_path)) == required

    for filename in required:
        (tmp_path / filename).touch()

    assert desktop.StrategyDesktopApp._missing_ml_policy_report_files(tmp_path) == []


def test_monthly_training_progress_tracks_real_log_counters() -> None:
    progress, stage = desktop._monthly_training_progress(["[market 5/10] 000001 A"])
    assert progress == 15.5
    assert "5/10" in stage

    progress, stage = desktop._monthly_training_progress(
        ["[market 10/10] 000001 A", "[panel] 1000 rows", "[training 2/4] window=2025-02"]
    )
    assert progress == 65.0
    assert "2/4" in stage

    progress, stage = desktop._monthly_training_progress(["[refit 1/2] all_factor"])
    assert progress == 94.0
    assert "all_factor" in stage

    progress, stage = desktop._monthly_training_progress(["[backtest] report.parquet"])
    assert progress == 100.0
    assert "报告已生成" in stage
