"""Lazy, process-safe access to immutable production model packages."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Mapping

from .versioning import ModelRegistry


class ProductionModelError(RuntimeError):
    """Base error for production model discovery and loading."""


class ProductionModelNotFound(ProductionModelError):
    """Raised when daily inference is requested before model promotion."""


class ProductionModelCorrupt(ProductionModelError):
    """Raised when a registered model package cannot be deserialized."""


class ProductionModelLoader:
    """Load one production package per process and reuse it until version changes."""

    _instances: dict[str, "ProductionModelLoader"] = {}
    _instances_lock = threading.RLock()

    def __new__(cls, project_root: str | Path):
        key = str(Path(project_root).resolve())
        with cls._instances_lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = super().__new__(cls)
                cls._instances[key] = instance
        return instance

    def __init__(self, project_root: str | Path) -> None:
        if getattr(self, "_initialized", False):
            return
        self.project_root = Path(project_root).resolve()
        self.registry = ModelRegistry(self.project_root)
        self._lock = threading.RLock()
        self._version: str | None = None
        self._package: dict[str, Any] | None = None
        self._load_count = 0
        self._initialized = True

    @property
    def load_count(self) -> int:
        return self._load_count

    def status(self) -> dict[str, str | None]:
        return self.registry.status()

    def metadata(self, status: str = "production") -> dict[str, Any]:
        """Read JSON metadata without deserializing model files."""
        version = self.status().get(status)
        if not version:
            raise ProductionModelNotFound(
                "No production model is registered. Run monthly training and promote a candidate first."
            )
        path = self.registry.version_path(version) / "model_metadata.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProductionModelCorrupt(f"Missing model metadata: {path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductionModelCorrupt(f"Invalid model metadata: {path}: {exc}") from exc

    def load(self) -> dict[str, Any]:
        """Lazily load and cache the current production package."""
        with self._lock:
            version = self.status().get("production")
            if not version:
                # Keep the registry load method injectable for existing API
                # callers and tests. The real registry raises FileNotFoundError
                # here; no runtime training or fallback model is created.
                try:
                    package = self.registry.load_status("production")
                except FileNotFoundError as exc:
                    raise ProductionModelNotFound(
                        "No production model is registered. Run monthly training and promote a candidate first."
                    ) from exc
                if "all_factor" not in package.get("models", {}):
                    raise ProductionModelCorrupt("Production package has no all-factor model.")
                self._version = str(package.get("version") or "production")
                self._package = package
                self._load_count += 1
                return package
            if self._package is not None and version == self._version:
                return self._package
            try:
                package = self.registry.load_version(version)
            except FileNotFoundError as exc:
                raise ProductionModelCorrupt(
                    f"Production model {version!r} is incomplete: {exc}"
                ) from exc
            except Exception as exc:
                raise ProductionModelCorrupt(
                    f"Production model {version!r} could not be loaded: {type(exc).__name__}: {exc}"
                ) from exc
            if "all_factor" not in package.get("models", {}):
                raise ProductionModelCorrupt(
                    f"Production model {version!r} has no all-factor model."
                )
            self._version = version
            self._package = package
            self._load_count += 1
            return package

    def clear(self) -> None:
        with self._lock:
            self._version = None
            self._package = None


def load_production_models(project_root: str | Path) -> dict[str, Any]:
    return ProductionModelLoader(project_root).load()


def load_model_metadata(project_root: str | Path) -> dict[str, Any]:
    return ProductionModelLoader(project_root).metadata()


def load_group_weights(project_root: str | Path) -> dict[str, float]:
    return dict(load_production_models(project_root).get("group_weights", {}))


def load_factor_columns(project_root: str | Path) -> dict[str, list[str]]:
    return dict(load_production_models(project_root).get("factor_columns", {}))


def load_factor_status(project_root: str | Path) -> dict[str, Any]:
    return dict(load_production_models(project_root).get("factor_status", {}))


def promote_candidate_model(project_root: str | Path, comparison: Mapping[str, Any]) -> bool:
    promoted = ModelRegistry(project_root).promote_candidate(comparison)
    if promoted:
        ProductionModelLoader(project_root).clear()
    return promoted


def evaluate_candidate_model(project_root: str | Path, comparison: Mapping[str, Any]) -> dict[str, Any]:
    return ModelRegistry(project_root).candidate_promotion_evaluation(comparison)


def rollback_production_model(project_root: str | Path) -> bool:
    rolled_back = ModelRegistry(project_root).rollback_production()
    if rolled_back:
        ProductionModelLoader(project_root).clear()
    return rolled_back
