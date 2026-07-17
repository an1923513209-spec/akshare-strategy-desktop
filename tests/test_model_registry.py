from __future__ import annotations

from pathlib import Path

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
