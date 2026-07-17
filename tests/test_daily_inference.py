from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_decision import AccountState, DecisionConfig
from ml_decision.inference import confidence_score, run_production_holding_decision
from ml_decision.models import PredictionPack
from ml_decision.versioning import ModelRegistry


class FakeProductionModel:
    feature_columns = ["ret_5", "rsi_14"]
    metrics = {"auc_up": 0.61}

    def fit(self, *_args, **_kwargs):
        raise AssertionError("Daily inference must never call fit")

    def predict_one(self, _row):
        return PredictionPack(
            expected_gap_return=0.001,
            expected_open_to_open_return=0.012,
            probability_up=0.68,
            probability_profitable=0.63,
            probability_down_2pct=0.08,
            return_q10=-0.012,
            return_q50=0.008,
            return_q90=0.031,
            confidence_level="high",
            top_positive_factors=[],
            top_negative_factors=[],
            important_factors=["ret_5"],
            score_weights={"probability": 0.45, "expected": 0.35, "risk": 0.20},
        )

    def _important_factors(self):
        return ["ret_5", "rsi_14"]


class OneStockFailsModel(FakeProductionModel):
    def predict_one(self, row):
        if str(row.get("code")) == "000002":
            raise RuntimeError("synthetic single-stock failure")
        return super().predict_one(row)


def _market() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=90)
    rows = []
    for code, offset in (("000001", 0.0), ("000002", 2.0)):
        close = 10 + offset + np.linspace(0, 3, len(dates))
        for date, price in zip(dates, close):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": code,
                    "open": price * 0.998,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1_000_000,
                    "amount": price * 1_000_000,
                    "market_data_available": 1.0,
                    "fund_flow_data_available": 1.0,
                    "news_data_available": 1.0,
                    "institution_data_available": 1.0,
                    "lhb_detail_available": 1.0,
                    "lhb_inst_data_available": 1.0,
                    "has_news": 0.0,
                    "lhb_flag": 0.0,
                    "lhb_count_5d": 0.0,
                    "lhb_inst_buy_count": 0.0,
                    "lhb_inst_net_buy_sum_5d": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _register(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    model = FakeProductionModel()
    registry.save_version(
        "daily-test",
        {"all_factor": model, "technical": model},
        metadata={
            "config": {
                "event_gating": {"news_enabled": True, "lhb_enabled": True},
                "dynamic_weights": {"max_group_weight": 1.0},
                "confidence": {
                    "minimum_data_completeness": 0.70,
                    "low_confidence_position_multiplier": 0.50,
                },
            },
            "latest_window": {"train_end": "2025-04-30"},
        },
        group_weights={"technical": 1.0},
        factor_columns={"all_factor": model.feature_columns, "technical": model.feature_columns},
        factor_status={"all_factor": "ACTIVE", "technical": "ACTIVE"},
        training_metrics=[{"model_group": "all_factor", "auc": 0.61, "rank_ic": 0.04}],
    )
    assert registry.promote_candidate(
        {
            "consecutive_better_windows": 3,
            "minimum_better_windows": 3,
            "net_return_not_worse": True,
            "drawdown_not_worse": True,
            "regression_tests_passed": True,
        }
    )


def test_batch_daily_inference_uses_saved_models_without_fit(tmp_path: Path):
    _register(tmp_path)
    holdings = pd.DataFrame(
        [
            {"code": "000001", "name": "A", "shares": 0, "available_shares": 0},
            {"code": "000002", "name": "B", "shares": 100, "available_shares": 100},
        ]
    )
    result = run_production_holding_decision(
        _market(),
        holdings,
        tmp_path,
        account=AccountState(cash=100_000, total_asset=100_000),
        config=DecisionConfig(train_min_rows=50),
    )
    assert set(result.table["code"]) == {"000001", "000002"}
    assert set(result.table["model_version"]) == {"daily-test"}
    assert (result.table["data_completeness_score"] == 1.0).all()
    assert Path(result.snapshot_path).exists()


def test_low_data_completeness_reduces_confidence():
    full_score, _ = confidence_score(0.68, 0.01, {"technical": 0.67}, 1.0, {"auc": 0.61})
    low_score, level = confidence_score(0.68, 0.01, {"technical": 0.67}, 0.3, {"auc": 0.61})
    assert low_score < full_score
    assert level == "unavailable"


def test_one_stock_prediction_failure_does_not_abort_batch(tmp_path: Path):
    registry = ModelRegistry(tmp_path)
    model = OneStockFailsModel()
    registry.save_version(
        "partial-test",
        {"all_factor": model, "technical": model},
        metadata={
            "config": {
                "event_gating": {"news_enabled": True, "lhb_enabled": True},
                "dynamic_weights": {"max_group_weight": 1.0},
                "confidence": {},
            }
        },
        group_weights={"technical": 1.0},
        factor_columns={"all_factor": model.feature_columns, "technical": model.feature_columns},
        factor_status={"all_factor": "ACTIVE", "technical": "ACTIVE"},
        training_metrics=[],
    )
    assert registry.promote_candidate(
        {
            "consecutive_better_windows": 3,
            "minimum_better_windows": 3,
            "net_return_not_worse": True,
            "drawdown_not_worse": True,
            "regression_tests_passed": True,
        }
    )
    holdings = pd.DataFrame(
        [
            {"code": "000001", "shares": 0, "available_shares": 0},
            {"code": "000002", "shares": 0, "available_shares": 0},
        ]
    )
    result = run_production_holding_decision(_market(), holdings, tmp_path, save_snapshot=False)
    assert list(result.table["code"]) == ["000001"]
    assert any(note.source == "production_inference" for note in result.source_notes)
