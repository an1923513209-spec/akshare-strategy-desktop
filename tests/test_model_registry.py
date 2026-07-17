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

