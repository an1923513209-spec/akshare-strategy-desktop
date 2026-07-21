from __future__ import annotations

import numpy as np
import pandas as pd

from ml_decision.actions import Holding, score_policy_target
from ml_decision.config import AccountState
from ml_decision.models import PredictionPack
from ml_decision.policy import (
    FactorPolicyParameters,
    calibrate_factor_policy,
    choose_target_weight,
    simulate_factor_policy,
)


def _prediction_frame(periods: int = 10) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    rows: list[dict] = []
    for code, sign in (("000001", 1.0), ("000002", -1.0)):
        for index, date in enumerate(dates):
            realized = sign * (0.006 + index * 0.0001)
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": "上涨样本" if sign > 0 else "下跌样本",
                    "next_open_to_next_open_return": realized,
                    "probability_up": 0.80 if sign > 0 else 0.20,
                    "probability_down_2pct": 0.08 if sign > 0 else 0.70,
                    "predicted_return": 0.012 if sign > 0 else -0.012,
                    "return_q10": -0.008 if sign > 0 else -0.04,
                }
            )
    return pd.DataFrame(rows)


def _prediction_pack() -> PredictionPack:
    return PredictionPack(
        expected_gap_return=0.0,
        expected_open_to_open_return=-0.02,
        probability_up=0.20,
        probability_profitable=0.20,
        probability_down_2pct=0.70,
        return_q10=-0.05,
        return_q50=-0.02,
        return_q90=0.01,
        confidence_level="high",
        top_positive_factors=[],
        top_negative_factors=[],
        important_factors=[],
        score_weights={},
    )


def test_policy_uses_factor_predictions_to_choose_absolute_weight() -> None:
    parameters = FactorPolicyParameters(
        probability_return_scale=0.02,
        downside_penalty=0.25,
        concentration_penalty=0.0,
        minimum_utility_edge=0.0,
        max_single_weight=0.25,
    )
    positive, _, _ = choose_target_weight(
        probability_up=0.90,
        predicted_return=0.03,
        probability_down=0.05,
        return_q10=-0.005,
        current_weight=0.0,
        parameters=parameters,
    )
    negative, _, _ = choose_target_weight(
        probability_up=0.15,
        predicted_return=-0.03,
        probability_down=0.75,
        return_q10=-0.06,
        current_weight=0.20,
        parameters=parameters,
    )
    assert positive == 0.25
    assert negative == 0.0


def test_each_stock_policy_backtest_has_an_independent_position_path() -> None:
    backtest = simulate_factor_policy(
        _prediction_frame(6),
        FactorPolicyParameters(minimum_utility_edge=0.0, concentration_penalty=0.0),
    )
    first_rows = backtest.sort_values(["code", "date"]).groupby("code", sort=False).head(1)
    assert first_rows["current_weight"].eq(0.0).all()
    assert backtest.groupby("code")["equity"].last().loc["000001"] > 1.0
    assert backtest.groupby("code")["target_weight"].max().le(0.25).all()


def test_policy_calibration_and_test_dates_have_two_day_embargo() -> None:
    dates = pd.DatetimeIndex(_prediction_frame(10)["date"].drop_duplicates().sort_values())
    _parameters, report, backtest, summary = calibrate_factor_policy(
        _prediction_frame(10),
        {
            "calibration_fraction": 0.5,
            "embargo_trading_days": 2,
            "parameter_grid": {
                "probability_return_scale": [0.01],
                "downside_penalty": [0.5],
                "concentration_penalty": [0.0],
                "minimum_utility_edge": [0.0],
            },
        },
    )
    calibration_end = pd.Timestamp(report["calibration_end"])
    test_start = pd.Timestamp(report["test_start"])
    assert dates.get_loc(test_start) - dates.get_loc(calibration_end) >= 3
    assert pd.to_datetime(backtest["date"]).min() == test_start
    assert set(summary["code"]) == {"000001", "000002"}


def test_policy_handles_missing_prediction_without_cross_stock_failure() -> None:
    frame = _prediction_frame(4)
    frame.loc[frame["code"].eq("000001") & frame["date"].eq(frame["date"].min()), "probability_up"] = np.nan
    backtest = simulate_factor_policy(frame, FactorPolicyParameters())
    assert set(backtest["code"]) == {"000001", "000002"}
    assert np.isfinite(backtest["equity"]).all()


def test_policy_clear_sells_all_odd_lot_available_shares() -> None:
    holding = Holding(
        code="000001",
        shares=150,
        available_shares=150,
        average_cost=10.0,
        current_price=10.0,
        position_value=1500.0,
        position_weight=0.015,
        available_shares_known=True,
    )
    latest = pd.Series({"volume": 1000, "open": 10, "high": 10.2, "low": 9.8, "close": 10})
    score = score_policy_target(
        holding,
        latest,
        _prediction_pack(),
        AccountState(total_asset=100_000, cash=20_000),
        target_weight=0.0,
        predicted_utility=0.1,
    )
    assert score.requested_action == "POLICY_SELL_CLEAR"
    assert score.trade_shares == -150
    assert score.target_shares == 0
