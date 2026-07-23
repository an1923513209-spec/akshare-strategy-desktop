from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_decision.model_registry import (
    ProductionModelLoader,
    ProductionModelNotFound,
    rollback_production_model,
)
from ml_decision.versioning import ModelRegistry


class StoredModel:
    feature_columns = ["ret_5"]


def _save(registry: ModelRegistry, version: str) -> None:
    registry.save_version(
        version,
        {"all_factor": StoredModel()},
        metadata={"config": {}},
        group_weights={},
        factor_columns={"all_factor": ["ret_5"]},
        factor_status={"all_factor": "ACTIVE"},
        training_metrics=[],
    )


def _passing_gate() -> dict:
    return {
        "consecutive_better_windows": 3,
        "minimum_better_windows": 3,
        "net_return_not_worse": True,
        "drawdown_not_worse": True,
        "regression_tests_passed": True,
    }


def test_missing_production_never_trains_a_temporary_model(tmp_path: Path):
    with pytest.raises(ProductionModelNotFound):
        ProductionModelLoader(tmp_path).load()


def test_loader_reuses_package_until_version_changes(tmp_path: Path):
    registry = ModelRegistry(tmp_path)
    _save(registry, "v1")
    assert registry.promote_candidate(_passing_gate())
    loader = ProductionModelLoader(tmp_path)
    first = loader.load()
    second = loader.load()
    assert first is second
    assert loader.load_count == 1


def test_production_can_roll_back_to_previous_version(tmp_path: Path):
    registry = ModelRegistry(tmp_path)
    _save(registry, "v1")
    assert registry.promote_candidate(_passing_gate())
    _save(registry, "v2")
    assert registry.promote_candidate(_passing_gate())
    assert registry.status()["production"] == "v2"
    assert rollback_production_model(tmp_path)
    assert registry.status()["production"] == "v1"


def test_first_production_uses_absolute_oos_gate(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "ml_governance.json").write_text(
        '{"model_promotion":{"initial_production":{"minimum_oos_windows":12,"minimum_auc":0.52,'
        '"minimum_rank_ic":0.0,"minimum_net_return":0.0,"minimum_max_drawdown":-0.15,'
        '"maximum_brier":0.255}}}',
        encoding="utf-8",
    )
    registry = ModelRegistry(tmp_path)
    rows = [
        {"model_group": "dynamic_group_ensemble", "auc": 0.53, "rank_ic": 0.02,
         "net_return": 0.01, "max_drawdown": -0.08, "brier": 0.249}
        for _ in range(12)
    ]
    registry.save_version(
        "v1", {"all_factor": StoredModel()}, metadata={"config": {}}, group_weights={},
        factor_columns={"all_factor": ["ret_5"]}, factor_status={"all_factor": "ACTIVE"},
        training_metrics=rows,
    )
    comparison = {"regression_tests_passed": True, "consecutive_better_windows": 0}

    evaluation = registry.candidate_promotion_evaluation(comparison)
    assert evaluation["mode"] == "initial_absolute"
    assert evaluation["passed"]
    assert registry.promote_candidate(comparison)


def test_replacement_compares_candidate_and_production_on_shared_oos_dates(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "ml_governance.json").write_text(
        '{"model_promotion":{'
        '"replacement":{"comparison_recent_windows":12,"minimum_comparable_windows":12,'
        '"rank_ic_tolerance":0.01,"net_return_tolerance":0.003,'
        '"max_drawdown_tolerance":0.02,"brier_tolerance":0.005},'
        '"initial_production":{"minimum_oos_windows":12,"minimum_auc":0.52,'
        '"minimum_rank_ic":0.0,"minimum_net_return":0.0,"minimum_max_drawdown":-0.15,'
        '"maximum_brier":0.255}}}',
        encoding="utf-8",
    )
    dates = pd.bdate_range("2025-01-02", periods=12)

    def rows(rank_ic: float, net_return: float, drawdown: float, brier: float) -> list[dict]:
        return [
            {
                "model_group": "dynamic_group_ensemble",
                "window_id": f"w{index:02d}",
                "test_start": str(date.date()),
                "test_end": str(date.date()),
                "auc": 0.54,
                "rank_ic": rank_ic,
                "net_return": net_return,
                "max_drawdown": drawdown,
                "brier": brier,
            }
            for index, date in enumerate(dates)
        ]

    registry = ModelRegistry(tmp_path)
    registry.save_version(
        "production", {"all_factor": StoredModel()}, metadata={}, group_weights={},
        factor_columns={"all_factor": ["ret_5"]}, factor_status={"all_factor": "ACTIVE"},
        training_metrics=rows(0.020, 0.010, -0.08, 0.249),
    )
    assert registry.promote_candidate({"regression_tests_passed": True})
    registry.save_version(
        "candidate", {"all_factor": StoredModel()}, metadata={}, group_weights={},
        factor_columns={"all_factor": ["ret_5"]}, factor_status={"all_factor": "ACTIVE"},
        training_metrics=rows(0.015, 0.008, -0.09, 0.251),
    )

    evaluation = registry.candidate_promotion_evaluation({"regression_tests_passed": True})
    assert evaluation["mode"] == "replacement_shared_oos"
    assert evaluation["shared_window_count"] == 12
    assert evaluation["passed"]
    assert evaluation["candidate_version"] == "candidate"
    assert evaluation["production_version"] == "production"


def test_replacement_rejects_material_oos_net_return_deterioration(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "ml_governance.json").write_text(
        '{"model_promotion":{'
        '"replacement":{"comparison_recent_windows":3,"minimum_comparable_windows":3,'
        '"rank_ic_tolerance":0.01,"net_return_tolerance":0.001,'
        '"max_drawdown_tolerance":0.02,"brier_tolerance":0.005},'
        '"initial_production":{"minimum_oos_windows":3,"minimum_auc":0.52,'
        '"minimum_rank_ic":0.0,"minimum_net_return":0.0,"minimum_max_drawdown":-0.15,'
        '"maximum_brier":0.255}}}',
        encoding="utf-8",
    )
    base = [
        {
            "model_group": "dynamic_group_ensemble", "window_id": f"w{index}",
            "test_start": f"2025-01-0{index + 2}", "test_end": f"2025-01-0{index + 2}",
            "auc": 0.54, "rank_ic": 0.02, "net_return": 0.01,
            "max_drawdown": -0.08, "brier": 0.249,
        }
        for index in range(3)
    ]
    registry = ModelRegistry(tmp_path)
    registry.save_version(
        "production", {"all_factor": StoredModel()}, metadata={}, group_weights={},
        factor_columns={"all_factor": ["ret_5"]}, factor_status={"all_factor": "ACTIVE"},
        training_metrics=base,
    )
    assert registry.promote_candidate({"regression_tests_passed": True})
    degraded = [{**row, "net_return": 0.001} for row in base]
    registry.save_version(
        "candidate", {"all_factor": StoredModel()}, metadata={}, group_weights={},
        factor_columns={"all_factor": ["ret_5"]}, factor_status={"all_factor": "ACTIVE"},
        training_metrics=degraded,
    )
    evaluation = registry.candidate_promotion_evaluation({"regression_tests_passed": True})
    assert not evaluation["passed"]
    assert "net_return_noninferior" in evaluation["failed_checks"]


def test_user_can_manually_select_any_complete_model_version(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    _save(registry, "v1")
    _save(registry, "v2")

    assert registry.list_versions() == ["v2", "v1"]
    assert registry.set_production_version("v1")
    assert registry.status()["production"] == "v1"
    assert registry.status()["candidate"] == "v2"
    assert registry.set_production_version("v2")
    status = registry.status()
    assert status["production"] == "v2"
    assert status["previous_production"] == "v1"
    assert status["candidate"] is None
    audit = (tmp_path / "models" / "manual_promotion_log.jsonl").read_text(encoding="utf-8")
    assert '"mode": "manual_gate_bypass"' in audit
    assert '"selected_version": "v2"' in audit
