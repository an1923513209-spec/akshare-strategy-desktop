from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_decision.accounting import resolve_account_snapshot, save_account_snapshot
from ml_decision.actions import Holding, score_actions
from ml_decision.baselines import evaluate_stable_baselines
from ml_decision.config import AccountState, DecisionConfig
from ml_decision.cross_section import (
    RANK_BASE_COLUMNS,
    compute_full_universe_ranks,
    load_rank_cache,
    save_rank_cache,
)
from ml_decision.features import build_features, feature_columns
from ml_decision.drift import assess_model_drift, update_prediction_history
from ml_decision.factor_registry import source_requirements
from ml_decision.inference import data_availability, required_data_completeness
from ml_decision.models import NextSessionModel, PredictionPack
from ml_decision.rolling import RollingWindow
from ml_decision.trading_rules import (
    board_name,
    drop_incomplete_latest_daily_bar,
    enrich_trade_constraints,
    price_limit_rate,
)
from services.task_manager import create_task, write_task_error, write_task_result
from desktop_strategy_app import StrategyDesktopApp


def _market(codes: tuple[str, ...] = ("000001", "000002", "000003")) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=45)
    rows: list[dict] = []
    for offset, code in enumerate(codes):
        for step, date in enumerate(dates):
            close = 10.0 + offset + step * (0.02 + offset * 0.005)
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "industry": "bank" if offset < 2 else "tech",
                    "open": close * 0.998,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000 + step * 1_000,
                    "amount": close * (1_000_000 + step * 1_000),
                }
            )
    return pd.DataFrame(rows)


def _prediction() -> PredictionPack:
    return PredictionPack(
        expected_gap_return=0.0,
        expected_open_to_open_return=-0.03,
        probability_up=0.25,
        probability_profitable=0.20,
        probability_down_2pct=0.60,
        return_q10=-0.05,
        return_q50=-0.02,
        return_q90=0.01,
        confidence_level="high",
        top_positive_factors=[],
        top_negative_factors=[],
        important_factors=[],
        score_weights={},
    )


def test_cross_section_rank_is_independent_of_desktop_selection(tmp_path: Path) -> None:
    features = build_features(_market())
    ranks = compute_full_universe_ranks(features, minimum_universe_size=3)
    save_rank_cache(ranks, tmp_path)
    loaded, _status = load_rank_cache(tmp_path, required_columns=["market_rank_ret_5"])

    one = build_features(_market(("000001",)), cross_sectional_rank_frame=loaded)
    two = build_features(_market(("000001", "000002")), cross_sectional_rank_frame=loaded)
    left = one.loc[one["code"].eq("000001"), ["date", "market_rank_ret_5"]].dropna()
    right = two.loc[two["code"].eq("000001"), ["date", "market_rank_ret_5"]].dropna()
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))


def test_small_selection_never_creates_local_pseudo_ranks() -> None:
    features = build_features(_market(("000001", "000002")))
    assert not any(column.startswith(("market_rank_", "industry_rank_")) for column in features)
    with pytest.raises(ValueError, match="local ranks are forbidden"):
        compute_full_universe_ranks(features, minimum_universe_size=500)


def test_account_snapshot_migrates_old_cash_and_validates(tmp_path: Path) -> None:
    snapshot = resolve_account_snapshot(
        available_cash=20_000,
        holdings_market_value=80_000,
        total_asset=None,
        market_date="2026-07-17",
    )
    assert snapshot.total_asset == 100_000
    assert snapshot.total_asset_estimated is True
    assert save_account_snapshot(snapshot, tmp_path).exists()
    with pytest.raises(ValueError, match="cannot be below"):
        resolve_account_snapshot(
            available_cash=20_000,
            holdings_market_value=80_000,
            total_asset=90_000,
        )


def test_only_required_sources_affect_core_completeness() -> None:
    availability = {
        "market_data_available": 1.0,
        "fund_flow_data_available": 0.0,
        "news_data_available": 0.0,
        "institution_data_available": 0.0,
    }
    score, details = required_data_completeness(
        availability,
        {"market_data_available": True, "news_data_available": False},
    )
    assert score == 1.0
    assert details["news_data_available"]["degraded"] is False
    score, details = required_data_completeness(
        availability,
        {"market_data_available": True, "fund_flow_data_available": True},
    )
    assert score == 0.5
    assert details["fund_flow_data_available"]["degraded"] is True


def test_required_cross_section_cache_affects_completeness() -> None:
    required = {"market_rank_ret_5", "ret_5"}
    requirements = source_requirements(required)
    availability = data_availability(
        pd.Series({"open": 10, "high": 11, "low": 9, "close": 10, "ret_5": 0.01}),
        required,
    )
    score, details = required_data_completeness(
        availability,
        requirements,
        data_date="2026-07-17",
    )
    assert requirements["cross_section_rank_available"] is True
    assert details["cross_section_rank_available"]["degraded"] is True
    assert score < 1.0


def test_late_factor_can_activate_in_a_later_training_window() -> None:
    dataset = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=80),
            "code": "000001",
            "next_open_to_next_open_return": 0.01,
            "late_factor": [np.nan] * 40 + list(np.arange(40, dtype=float)),
        }
    )
    assert "late_factor" not in feature_columns(dataset, selection_df=dataset.iloc[:40])
    assert "late_factor" in feature_columns(dataset, selection_df=dataset.iloc[40:])
    assert "late_factor" in feature_columns(dataset, selection_df=dataset)


def test_unknown_t1_available_shares_disables_sell_sizing() -> None:
    holding = Holding(
        code="000001",
        shares=150,
        available_shares=0,
        average_cost=10.0,
        current_price=10.0,
        position_value=1500.0,
        position_weight=0.015,
        available_shares_known=False,
    )
    scores = score_actions(
        holding,
        pd.Series({"volume": 1_000_000}),
        _prediction(),
        AccountState(cash=0, total_asset=100_000),
        DecisionConfig(),
    )
    sell = next(score for score in scores if score.requested_action == "SELL_ALL")
    assert sell.trade_shares == 0
    assert sell.effective_action == "NO_TRADE_AVAILABLE_UNKNOWN"
    assert sell.feasible is False


def test_one_price_limits_block_corresponding_side() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-07-16", "code": "600000", "name": "A", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 1000},
            {"date": "2026-07-17", "code": "600000", "name": "A", "open": 11, "high": 11, "low": 11, "close": 11, "volume": 1000},
        ]
    )
    latest = enrich_trade_constraints(frame).iloc[-1]
    assert bool(latest["is_one_price_up"]) is True
    holding = Holding("600000", 0, 0, 0.0, 11.0, 0.0, 0.0)
    scores = score_actions(
        holding,
        latest,
        PredictionPack(
            expected_gap_return=0.01,
            expected_open_to_open_return=0.03,
            probability_up=0.8,
            probability_profitable=0.75,
            probability_down_2pct=0.05,
            return_q10=-0.01,
            return_q50=0.02,
            return_q90=0.06,
            confidence_level="high",
            top_positive_factors=[],
            top_negative_factors=[],
            important_factors=[],
            score_weights={},
        ),
        AccountState(cash=100_000, total_asset=100_000),
        DecisionConfig(),
    )
    assert not any(score.feasible and score.trade_shares > 0 for score in scores)


@pytest.mark.parametrize(
    ("code", "name", "board", "rate"),
    [
        ("600000", "浦发银行", "main", 0.10),
        ("300750", "宁德时代", "chinext", 0.20),
        ("688981", "中芯国际", "star", 0.20),
        ("920001", "北交测试", "beijing", 0.30),
        ("600001", "*ST测试", "main", 0.05),
    ],
)
def test_a_share_board_price_limit_rules(code: str, name: str, board: str, rate: float) -> None:
    assert board_name(code) == board
    assert price_limit_rate(code, name) == rate


def test_suspended_stock_produces_no_trade() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-07-16", "code": "600000", "name": "A", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 1000},
            {"date": "2026-07-17", "code": "600000", "name": "A", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 0},
        ]
    )
    latest = enrich_trade_constraints(frame).iloc[-1]
    scores = score_actions(
        Holding("600000", 100, 100, 10.0, 10.0, 1000.0, 0.01),
        latest,
        _prediction(),
        AccountState(cash=10_000, total_asset=100_000),
        DecisionConfig(),
    )
    assert bool(latest["is_suspended"]) is True
    assert all(score.trade_shares == 0 for score in scores)


def test_prediction_history_backfills_outcomes_and_reports_drift(tmp_path: Path) -> None:
    dates = pd.bdate_range("2026-01-02", periods=45)
    market_rows: list[dict[str, object]] = []
    for offset, code in enumerate(("000001", "000002", "000003")):
        for index, date in enumerate(dates):
            open_price = 10 + offset + index * (0.02 + offset * 0.003) + np.sin(index / 3 + offset) * 0.15
            market_rows.append({"code": code, "date": date, "open": open_price})
    market = pd.DataFrame(market_rows)
    ordered = market.sort_values(["code", "date"], kind="stable").copy()
    grouped = ordered.groupby("code", sort=False)["open"]
    ordered["actual"] = grouped.shift(-2) / grouped.shift(-1) - 1
    predictions = ordered.dropna(subset=["actual"]).copy()
    predictions["probability_up"] = np.where(predictions["actual"].gt(0), 0.55, 0.45)
    predictions["expected_open_to_open_return"] = predictions["actual"]
    predictions["data_completeness_score"] = 1.0
    predictions["feature__ret_5"] = predictions["actual"].rolling(5, min_periods=1).mean()
    predictions = predictions.rename(columns={"actual": "unused_actual"})

    history = update_prediction_history(tmp_path, predictions, market, "v1")
    report = assess_model_drift(
        history,
        "v1",
        {"minimum_realized_rows": 30},
        production_training_end=dates[20],
    )
    assert (tmp_path / "cache" / "ml_prediction_history.parquet").exists()
    assert history["realized_return"].notna().sum() >= 30
    assert np.isfinite(report["metrics"]["rank_ic_20d"])
    assert np.isfinite(report["metrics"]["feature_psi"])
    assert report["metrics"]["trading_days_since_training"] > 0


def test_today_daily_bar_is_dropped_before_close() -> None:
    frame = pd.DataFrame(
        {"Close": [10.0, 10.2]},
        index=pd.to_datetime(["2026-07-17", "2026-07-20"]),
    )
    before_close = drop_incomplete_latest_daily_bar(
        frame,
        now=datetime(2026, 7, 20, 14, 30),
    )
    after_close = drop_incomplete_latest_daily_bar(
        frame,
        now=datetime(2026, 7, 20, 15, 10),
    )
    assert list(before_close.index.strftime("%Y-%m-%d")) == ["2026-07-17"]
    assert len(after_close) == 2


def test_desktop_task_files_keep_log_result_and_error(tmp_path: Path) -> None:
    task = create_task(tmp_path, {"batch_symbols": "000001 000002"})
    write_task_result(task, {"results": [{"code": "000001"}], "errors": []})
    write_task_error(task, "network_failure", "offline")
    assert task.input_path.exists()
    assert task.log_path.exists()
    assert '"result_count": 1' in task.result_path.read_text(encoding="utf-8")
    assert '"category": "network_failure"' in task.error_path.read_text(encoding="utf-8")


def test_auto_monitor_tick_does_not_stack_workers() -> None:
    statuses: list[str] = []
    scheduled: list[int] = []
    starts: list[tuple[bool, list[str]]] = []
    fake = SimpleNamespace(
        monitor_running=True,
        selected_monitor_symbol="000001",
        monitor_worker=SimpleNamespace(is_alive=lambda: True),
        monitor_schedule_var=SimpleNamespace(set=statuses.append),
        _monitor_interval_seconds=lambda: 30,
        _start_monitor_worker=lambda loop, symbols: starts.append((loop, symbols)),
        _is_a_share_trading_time=lambda: True,
        after=lambda delay, callback: scheduled.append(delay),
    )
    fake._auto_monitor_tick = lambda: None
    StrategyDesktopApp._auto_monitor_tick(fake)
    assert not starts
    assert statuses[-1] == "上一轮仍在运行，本轮已跳过"
    assert scheduled == [30_000]


def test_auto_monitor_pauses_network_requests_outside_trading_hours() -> None:
    statuses: list[str] = []
    scheduled: list[int] = []
    starts: list[tuple[bool, list[str]]] = []
    fake = SimpleNamespace(
        monitor_running=True,
        selected_monitor_symbol="000001",
        monitor_worker=None,
        monitor_schedule_var=SimpleNamespace(set=statuses.append),
        _monitor_interval_seconds=lambda: 30,
        _is_a_share_trading_time=lambda: False,
        _start_monitor_worker=lambda loop, symbols: starts.append((loop, symbols)),
        after=lambda delay, callback: scheduled.append(delay),
    )
    fake._auto_monitor_tick = lambda: None
    StrategyDesktopApp._auto_monitor_tick(fake)
    assert not starts
    assert "非交易时段" in statuses[-1]
    assert scheduled == [300_000]


def test_stable_baselines_use_the_same_untouched_test_dates() -> None:
    dates = pd.bdate_range("2025-01-02", periods=80)
    rows = []
    for code_offset, code in enumerate(("000001", "000002", "000003")):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "factor_a": index + code_offset,
                    "factor_b": np.sin(index / 5 + code_offset),
                    "next_open_to_next_open_return": np.sin(index / 7 + code_offset) * 0.02,
                }
            )
    dataset = pd.DataFrame(rows)
    window = RollingWindow(
        "probe",
        dates[:50],
        dates[52:60],
        dates[62:70],
        dates[72:],
        2,
        2,
    )
    rows = evaluate_stable_baselines(
        dataset,
        ["factor_a", "factor_b"],
        window,
        transaction_cost=0.0016,
    )
    assert {row["model_group"] for row in rows} == {
        "baseline_logistic_ridge",
        "baseline_equal_factor",
    }
    assert all(np.isfinite(float(row["brier"])) for row in rows)


class _RefitProbe(NextSessionModel):
    def _fit_regressor(self, x, y):
        return ("regressor", len(x))

    def _fit_quantile(self, x, y, alpha):
        return ("quantile", alpha, len(x))

    def _fit_classifier(self, name, x_train, y_train, x_calibration, y_calibration):
        return ("classifier", name, len(x_train), len(x_calibration))


def test_production_refit_reaches_latest_labelled_date_without_reselection() -> None:
    dates = pd.bdate_range("2025-01-02", periods=90)
    dataset = pd.DataFrame(
        {
            "date": dates,
            "code": "000001",
            "frozen_factor": np.arange(len(dates), dtype=float),
            "next_gap_return": 0.001,
            "next_open_to_next_open_return": np.sin(np.arange(len(dates))) * 0.01,
            "label_up": (np.arange(len(dates)) % 2).astype(float),
            "label_profitable": (np.arange(len(dates)) % 3 == 0).astype(float),
            "label_down_2pct": (np.arange(len(dates)) % 5 == 0).astype(float),
        }
    )
    model = _RefitProbe().refit_production(
        dataset,
        ["frozen_factor"],
        frozen_metrics={"auc_up": 0.57},
        calibration_days=20,
        purge_days=2,
    )
    assert model.feature_columns == ["frozen_factor"]
    assert model.metrics["production_refit"] is True
    assert model.metrics["production_training_end"] == str(dates.max().date())
    assert model.metrics["auc_up"] == 0.57
    assert model.models["open_to_open"] == ("regressor", len(dataset))
