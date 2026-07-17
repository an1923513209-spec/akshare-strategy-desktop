from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_decision.audit import ABLATION_VARIANTS, ablation_feature_sets, group_shap_summary, sample_shap_directions
from ml_decision.ensemble import apply_event_gates, compute_dynamic_weights, equal_weights
from ml_decision.factor_registry import (
    build_factor_groups,
    flatten_factor_groups,
    snapshot_factor_frame,
    validate_feature_names,
)
from ml_decision.rolling import generate_rolling_windows
from ml_decision.versioning import ModelRegistry
from ml_decision import workflows


def governance_config() -> dict:
    return json.loads((Path(__file__).parents[1] / "config" / "ml_governance.json").read_text(encoding="utf-8"))


def factor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ret_5": [0.1, 0.2],
            "volume_ratio_5": [1.0, 2.0],
            "main_net_ratio": [0.01, np.nan],
            "institution_activity": [2.0, 3.0],
            "news_sentiment": [0.5, np.nan],
            "lhb_flag": [1.0, 0.0],
            "lhb_inst_buy_count": [2.0, 0.0],
            "mystery_alpha": [7.0, 8.0],
        }
    )


def test_all_existing_factor_fields_are_preserved() -> None:
    frame = factor_frame()
    groups = build_factor_groups(frame.columns)
    assert set(flatten_factor_groups(groups)) == set(frame.columns)


def test_existing_factor_values_are_unchanged_by_registry() -> None:
    frame = factor_frame()
    before = snapshot_factor_frame(frame, frame.columns)
    build_factor_groups(frame.columns)
    pd.testing.assert_frame_equal(frame, before)


def test_ordinary_institution_and_lhb_institution_are_separate() -> None:
    groups = build_factor_groups(["institution_activity", "lhb_inst_buy_count"])
    assert groups["institution"] == ["institution_activity"]
    assert groups["lhb_institution"] == ["lhb_inst_buy_count"]


def test_unmatched_factor_is_other_existing() -> None:
    assert build_factor_groups(["mystery_alpha"])["other_existing"] == ["mystery_alpha"]


@pytest.mark.parametrize("column", ["target_return", "label_up", "future_alpha", "next_return_1d", "上榜后1日"])
def test_target_and_future_fields_are_forbidden(column: str) -> None:
    with pytest.raises(ValueError):
        validate_feature_names([column])


def rolling_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-01", "2026-06-30")


def test_rolling_dates_never_overlap() -> None:
    window = generate_rolling_windows(rolling_dates())[-1]
    parts = [window.train_dates, window.calibration_dates, window.validation_dates, window.test_dates]
    for index, left in enumerate(parts):
        for right in parts[index + 1 :]:
            assert set(left).isdisjoint(set(right))


def test_rolling_boundaries_have_two_day_purge_and_embargo() -> None:
    dates = rolling_dates()
    positions = {date: index for index, date in enumerate(dates)}
    window = generate_rolling_windows(dates, purge_trading_days=2, embargo_trading_days=2)[-1]
    parts = [window.train_dates, window.calibration_dates, window.validation_dates, window.test_dates]
    for older, newer in zip(parts, parts[1:]):
        assert positions[newer.min()] - positions[older.max()] >= 5


def metric_history(months: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2025-01-31", periods=months, freq="ME")
    rows = []
    for group_index, group in enumerate(("technical", "liquidity", "fund_flow", "institution")):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "model_group": group,
                    "test_end": date,
                    "rank_ic": 0.01 * (group_index + 1) + index * 0.001,
                    "sharpe": 0.2 + group_index * 0.1,
                    "net_return": 0.001 + index * 0.0001,
                }
            )
    return pd.DataFrame(rows)


def test_dynamic_weights_use_only_past_oos_windows() -> None:
    config = governance_config()
    history = metric_history()
    effective = pd.Timestamp("2025-07-15")
    base, _ = compute_dynamic_weights(history, effective, config, groups=("technical", "liquidity", "fund_flow", "institution"))
    changed = pd.concat(
        [history, pd.DataFrame([{"model_group": "technical", "test_end": effective, "rank_ic": 99, "sharpe": 99, "net_return": 99}])],
        ignore_index=True,
    )
    after, _ = compute_dynamic_weights(changed, effective, config, groups=("technical", "liquidity", "fund_flow", "institution"))
    assert after == pytest.approx(base)


def test_news_missing_is_not_equal_to_no_news() -> None:
    weights = equal_weights(("technical", "liquidity", "news"))
    gated, status = apply_event_gates(weights, {"has_news": np.nan, "news_data_available": 0}, governance_config())
    assert status["news"] == "UNKNOWN_DATA_MISSING"
    assert gated["news"] > 0


def test_lhb_missing_is_not_equal_to_not_listed() -> None:
    weights = equal_weights(("technical", "liquidity", "lhb", "lhb_institution"))
    gated, status = apply_event_gates(weights, {"lhb_detail_available": 0, "lhb_inst_data_available": 0}, governance_config())
    assert status["lhb"] == "UNKNOWN_DATA_MISSING"
    assert gated["lhb"] > 0


def test_no_news_gates_news_and_renormalizes_other_models() -> None:
    weights = equal_weights(("technical", "liquidity", "fund_flow", "news"))
    row = {"has_news": 0, "news_data_available": 1}
    gated, status = apply_event_gates(weights, row, governance_config())
    assert status["news"] == "GATED_NO_NEWS"
    assert gated["news"] == 0
    assert sum(gated.values()) == pytest.approx(1.0)


def test_no_recent_lhb_gates_both_lhb_models() -> None:
    weights = equal_weights(("technical", "liquidity", "fund_flow", "lhb", "lhb_institution"))
    row = {
        "lhb_detail_available": 1,
        "lhb_flag": 0,
        "lhb_count_5d": 0,
        "lhb_inst_data_available": 1,
        "lhb_inst_buy_count": 0,
        "lhb_inst_net_buy_sum_5d": 0,
    }
    gated, _ = apply_event_gates(weights, row, governance_config())
    assert gated["lhb"] == 0
    assert gated["lhb_institution"] == 0
    assert sum(gated.values()) == pytest.approx(1.0)


def test_dynamic_model_weight_never_exceeds_40_percent() -> None:
    weights, _ = compute_dynamic_weights(
        metric_history(), pd.Timestamp("2026-01-01"), governance_config(),
        groups=("technical", "liquidity", "fund_flow", "institution"),
    )
    assert max(weights.values()) <= 0.4000001


def test_monthly_weight_change_never_exceeds_five_points() -> None:
    previous = {"technical": 0.25, "liquidity": 0.25, "fund_flow": 0.25, "institution": 0.25}
    weights, _ = compute_dynamic_weights(
        metric_history(), pd.Timestamp("2026-01-01"), governance_config(),
        groups=tuple(previous), previous_weights=previous,
    )
    assert max(abs(weights[group] - previous[group]) for group in previous) <= 0.0500001


def test_insufficient_windows_do_not_update_weights() -> None:
    previous = {"technical": 0.25, "liquidity": 0.25, "fund_flow": 0.25, "institution": 0.25}
    weights, report = compute_dynamic_weights(
        metric_history(3), pd.Timestamp("2026-01-01"), governance_config(),
        groups=tuple(previous), previous_weights=previous,
    )
    assert weights == previous
    assert set(report["update_reason"]) == {"insufficient_oos_windows"}


def test_ablation_contains_every_required_variant() -> None:
    groups = build_factor_groups(factor_frame().columns)
    assert tuple(ablation_feature_sets(groups)) == ABLATION_VARIANTS


def test_shap_signs_are_taken_from_current_sample() -> None:
    result = sample_shap_directions(
        pd.Series({"good": 0.4, "bad": -0.3, "flat": 0.0}),
        pd.Series({"good": 2.0, "bad": -1.0, "flat": 8.0}),
    )
    assert result["positive"][0]["factor"] == "good"
    assert result["negative"][0]["factor"] == "bad"


def test_group_shap_uses_absolute_sample_contributions() -> None:
    shap = pd.DataFrame({"ret_5": [1.0, -2.0], "news_sentiment": [3.0, 0.0]})
    groups = build_factor_groups(shap.columns)
    result = group_shap_summary(shap, groups)
    news = result[(result["group_name"] == "news") & (result["condition"] == "all_samples")].iloc[0]
    assert news["mean_abs_shap"] == pytest.approx(1.5)


def test_daily_prediction_does_not_trigger_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyPack:
        probability_up = 0.6
        expected_open_to_open_return = 0.01

    class DummyModel:
        def predict_one(self, _row):
            return DummyPack()

        def fit(self, *_args, **_kwargs):
            raise AssertionError("daily prediction must never fit")

    package = {
        "models": {"all_factor": DummyModel(), "technical": DummyModel()},
        "group_weights": {"technical": 1.0},
        "metadata": {"config": governance_config()},
    }
    monkeypatch.setattr(workflows.ModelRegistry, "load_status", lambda _self, _status: package)
    frame = pd.DataFrame({"date": ["2026-07-17"], "code": ["000001"], "ret_5": [0.1], "rsi_14": [50.0]})
    result = workflows.daily_predict(frame, tmp_path)
    assert result.iloc[0]["ensemble_probability_up"] == pytest.approx(0.6)


def test_failed_candidate_cannot_replace_production(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    registry.save_version(
        "v1", {}, metadata={}, group_weights={}, factor_columns={}, factor_status={}, training_metrics=[]
    )
    assert not registry.promote_candidate(
        {"minimum_better_windows": 3, "consecutive_better_windows": 2, "net_return_not_worse": True, "drawdown_not_worse": True, "regression_tests_passed": True}
    )
    assert registry._read_registry()["production"] is None


def test_passing_candidate_promotion_keeps_previous_pointer(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    registry.save_version("v1", {}, metadata={}, group_weights={}, factor_columns={}, factor_status={}, training_metrics=[])
    criteria = {"minimum_better_windows": 3, "consecutive_better_windows": 3, "net_return_not_worse": True, "drawdown_not_worse": True, "regression_tests_passed": True}
    assert registry.promote_candidate(criteria)
    registry.save_version("v2", {}, metadata={}, group_weights={}, factor_columns={}, factor_status={}, training_metrics=[])
    assert registry.promote_candidate(criteria)
    state = registry._read_registry()
    assert state["production"] == "v2"
    assert state["previous_production"] == "v1"
