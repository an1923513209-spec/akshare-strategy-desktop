from __future__ import annotations

from types import MethodType

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from ml_decision.actions import Holding, apply_account_constraints, score_actions
from ml_decision.config import AccountState, DecisionConfig
from ml_decision import data_sources
from ml_decision.features import feature_columns
from ml_decision.models import NextSessionModel, PredictionPack, make_purged_date_splits


def prediction(**overrides: float) -> PredictionPack:
    values = {
        "expected_gap_return": 0.01,
        "expected_open_to_open_return": 0.02,
        "probability_up": 0.70,
        "probability_profitable": 0.65,
        "probability_down_2pct": 0.10,
        "return_q10": -0.01,
        "return_q50": 0.02,
        "return_q90": 0.05,
    }
    values.update(overrides)
    return PredictionPack(
        **values,
        confidence_level="high",
        top_positive_factors=[],
        top_negative_factors=[],
        important_factors=[],
        score_weights={"probability": 0.5, "expected": 0.35, "risk": 0.15},
    )


def holding(shares: int, available: int | None = None) -> Holding:
    available = shares if available is None else available
    return Holding(
        code="000001",
        shares=shares,
        available_shares=available,
        average_cost=10.0,
        current_price=10.0,
        position_value=shares * 10.0,
        position_weight=shares * 10.0 / 100000.0,
        industry="test",
    )


def test_existing_institution_factors_remain_model_features() -> None:
    rows = 50
    dataset = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=rows),
            "code": "000001",
            "next_open_to_next_open_return": np.linspace(-0.02, 0.02, rows),
            "institution_activity": np.arange(rows, dtype=float),
            "institution_net_buy_amount": np.arange(rows, dtype=float) * 1000,
            "institution_hold_ratio": np.linspace(0.01, 0.05, rows),
        }
    )
    columns = feature_columns(dataset)
    assert "institution_activity" in columns
    assert "institution_net_buy_amount" in columns
    assert "institution_hold_ratio" in columns


def test_feature_coverage_is_selected_from_training_rows_only() -> None:
    dataset = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=80),
            "code": "000001",
            "next_open_to_next_open_return": 0.01,
            "late_only_factor": [np.nan] * 40 + list(np.arange(40, dtype=float)),
        }
    )
    assert "late_only_factor" not in feature_columns(dataset, selection_df=dataset.iloc[:40])


def test_expired_external_cache_triggers_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_calls: list[bool] = []

    def fake_cache(_source, _code, _ttl, allow_stale=False):
        cache_calls.append(bool(allow_stale))
        return None

    fresh = pd.DataFrame({"date": [pd.Timestamp("2026-07-16")], "code": ["000001"], "main_net_ratio": [0.1]})
    monkeypatch.setattr(data_sources, "_read_cached_frame", fake_cache)
    monkeypatch.setattr(data_sources, "_fetch_fund_flow_direct_history", lambda _code: fresh)
    monkeypatch.setattr(data_sources, "_write_cached_frame", lambda *_args, **_kwargs: None)
    result, note = data_sources.fetch_fund_flow_features("000001")
    assert cache_calls[0] is False
    assert result.equals(fresh)
    assert note.status == "ok"


def test_unique_dates_never_cross_partitions_and_have_two_day_purge() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    dataset = pd.DataFrame(
        [(date, code) for date in dates for code in ("000001", "000002")],
        columns=["date", "code"],
    )
    splits = make_purged_date_splits(dataset, purge_days=2)
    parts = [splits.train_dates, splits.calibration_dates, splits.validation_dates, splits.test_dates]
    for left_index, left in enumerate(parts):
        for right in parts[left_index + 1 :]:
            assert set(left).isdisjoint(set(right))
    positions = {date: index for index, date in enumerate(dates)}
    for left, right in zip(parts, parts[1:]):
        assert positions[right.min()] - positions[left.max()] >= 3


class CalibrationModel(NextSessionModel):
    def _xgb_classifier(self):
        return LogisticRegression(max_iter=500, random_state=7)


def test_sklearn_probability_calibration_is_applied() -> None:
    rng = np.random.default_rng(4)
    x_train = pd.DataFrame({"x": rng.normal(size=120)})
    y_train = (x_train["x"] + rng.normal(scale=0.7, size=120) > 0).astype(int)
    x_cal = pd.DataFrame({"x": rng.normal(size=60)})
    y_cal = (x_cal["x"] + rng.normal(scale=0.7, size=60) > 0).astype(int)
    model = CalibrationModel()
    calibrated = model._fit_classifier("up", x_train, y_train, x_cal, y_cal)
    assert isinstance(calibrated, CalibratedClassifierCV)
    assert model._calibration_metrics["calibration_up_status"] == "applied"
    assert "calibration_up_brier_after" in model._calibration_metrics


def test_sell_all_of_150_shares_leaves_zero() -> None:
    scores = score_actions(
        holding(150), pd.Series({"volume": 1000}), prediction(expected_open_to_open_return=-0.05),
        AccountState(cash=0, total_asset=100000), DecisionConfig(),
    )
    sell = next(score for score in scores if score.requested_action == "SELL_ALL")
    assert sell.trade_shares == -150
    assert sell.target_shares == 0
    assert sell.effective_action == "SELL_ALL"


def test_unavailable_shares_are_not_marked_as_true_liquidation() -> None:
    scores = score_actions(
        holding(150, available=100), pd.Series({"volume": 1000}), prediction(expected_open_to_open_return=-0.05),
        AccountState(cash=0, total_asset=100000), DecisionConfig(),
    )
    sell = next(score for score in scores if score.requested_action == "SELL_ALL")
    assert sell.trade_shares == -100
    assert sell.target_shares == 50
    assert sell.effective_action == "SELL_AVAILABLE"


def test_invalid_actions_do_not_create_duplicate_hold_rows() -> None:
    scores = score_actions(
        holding(0), pd.Series({"volume": 0}), prediction(),
        AccountState(cash=100000, total_asset=100000), DecisionConfig(),
    )
    outcomes = [(score.effective_action, score.trade_shares, score.target_shares) for score in scores]
    assert len(outcomes) == len(set(outcomes))
    assert sum(score.effective_action == "HOLD" for score in scores) == 1


def test_multi_stock_buys_do_not_exceed_shared_cash() -> None:
    table = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "shares": [0, 0],
            "current_price": [50.0, 50.0],
            "industry": ["a", "b"],
            "requested_action": ["ADD_50", "ADD_50"],
            "effective_action": ["ADD_50", "ADD_50"],
            "recommended_action": ["ADD_50", "ADD_50"],
            "recommended_trade_shares": [200, 200],
            "utility_score": [2.0, 1.0],
            "expected_open_to_open_return": [0.02, 0.02],
        }
    )
    result = apply_account_constraints(
        table,
        AccountState(
            cash=10000,
            total_asset=100000,
            max_total_position_weight=0.8,
            max_single_position_weight=0.5,
            max_industry_weight=0.5,
        ),
    )
    buy_value = float((result["recommended_trade_shares"] * result["current_price"]).sum())
    assert buy_value <= 10000
    assert result.attrs["remaining_cash"] >= 0
    assert float(result["recommended_target_weight"].sum()) <= 0.8 + 1e-9
    assert float(result["recommended_target_weight"].max()) <= 0.5 + 1e-9


def test_portfolio_constraints_do_not_change_raw_model_predictions() -> None:
    table = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "shares": [0, 0],
            "current_price": [10.0, 20.0],
            "industry": ["a", "b"],
            "requested_action": ["ADD_25", "ADD_25"],
            "effective_action": ["ADD_25", "ADD_25"],
            "recommended_action": ["ADD_25", "ADD_25"],
            "recommended_trade_shares": [1000, 500],
            "utility_score": [2.0, 1.0],
            "probability_up": [0.73, 0.61],
            "expected_open_to_open_return": [0.021, 0.012],
        }
    )
    before = table[["code", "probability_up", "expected_open_to_open_return"]].copy()
    constrained = apply_account_constraints(
        table,
        AccountState(cash=5_000, total_asset=100_000, max_single_position_weight=0.2),
    )
    pd.testing.assert_frame_equal(
        before.reset_index(drop=True),
        constrained[["code", "probability_up", "expected_open_to_open_return"]].reset_index(drop=True),
    )


def test_shap_signs_drive_positive_and_negative_labels() -> None:
    model = NextSessionModel(use_shap=True)
    model.feature_columns = ["positive", "negative", "neutral"]

    def fake_shap(self, _x):
        return {"positive": 0.4, "negative": -0.3, "neutral": 0.0}

    model._shap_contributions = MethodType(fake_shap, model)
    positive, negative, important = model._factor_directions(pd.DataFrame([[1, 2, 3]], columns=model.feature_columns))
    assert positive == ["positive"]
    assert negative == ["negative"]
    assert important == []
