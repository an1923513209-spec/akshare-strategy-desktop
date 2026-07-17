"""Immutable model versions and candidate/production status pointers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import joblib

from .factor_registry import safe_version_name


MODEL_FILES = {
    "all_factor": "all_factor_model.pkl",
    "technical": "technical_model.pkl",
    "liquidity": "liquidity_model.pkl",
    "fund_flow": "fund_flow_model.pkl",
    "institution": "institution_model.pkl",
    "news": "news_model.pkl",
    "lhb": "lhb_model.pkl",
    "lhb_institution": "lhb_institution_model.pkl",
    "fundamental": "fundamental_model.pkl",
}


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def current_git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


class ModelRegistry:
    """Store immutable model directories; status changes update only registry.json."""

    def __init__(self, root: str | Path) -> None:
        self.project_root = Path(root).resolve()
        self.models_root = self.project_root / "models"
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.models_root / "registry.json"

    def _read_registry(self) -> dict[str, str | None]:
        if not self.registry_path.exists():
            return {"candidate": None, "production": None, "previous_production": None}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def status(self) -> dict[str, str | None]:
        """Return a copy of candidate/production pointers without loading models."""
        return dict(self._read_registry())

    def _write_registry(self, data: Mapping[str, str | None]) -> None:
        temporary = self.registry_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(dict(data), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.registry_path)

    def version_path(self, version: str) -> Path:
        return self.models_root / safe_version_name(version)

    def save_version(
        self,
        version: str,
        models: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
        group_weights: Mapping[str, float],
        factor_columns: Mapping[str, list[str]],
        factor_status: Mapping[str, Any],
        training_metrics: Any,
    ) -> Path:
        path = self.version_path(version)
        if path.exists():
            raise FileExistsError(f"Immutable model version already exists: {path}")
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        for group, model in models.items():
            if model is not None and group in MODEL_FILES:
                joblib.dump(model, temporary / MODEL_FILES[group])
        full_metadata = dict(metadata)
        full_metadata.setdefault("model_version", version)
        full_metadata.setdefault("training_time", datetime.now().isoformat(timespec="seconds"))
        full_metadata.setdefault("git_commit", current_git_commit(self.project_root))
        payloads = {
            "model_metadata.json": full_metadata,
            "group_weights.json": dict(group_weights),
            "factor_columns.json": dict(factor_columns),
            "factor_status.json": dict(factor_status),
            "training_metrics.json": training_metrics,
        }
        for filename, payload in payloads.items():
            (temporary / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
            )
        temporary.replace(path)
        registry = self._read_registry()
        registry["candidate"] = version
        self._write_registry(registry)
        return path

    def load_version(self, version: str) -> dict[str, Any]:
        path = self.version_path(version)
        if not path.exists():
            raise FileNotFoundError(path)
        models = {
            group: joblib.load(path / filename)
            for group, filename in MODEL_FILES.items()
            if (path / filename).exists()
        }
        result: dict[str, Any] = {"version": version, "path": path, "models": models}
        for key, filename in (
            ("metadata", "model_metadata.json"),
            ("group_weights", "group_weights.json"),
            ("factor_columns", "factor_columns.json"),
            ("factor_status", "factor_status.json"),
            ("training_metrics", "training_metrics.json"),
        ):
            result[key] = json.loads((path / filename).read_text(encoding="utf-8"))
        return result

    def load_status(self, status: str = "production") -> dict[str, Any]:
        if status not in {"candidate", "production", "previous_production"}:
            raise ValueError(f"Unknown model status: {status}")
        version = self._read_registry().get(status)
        if not version:
            raise FileNotFoundError(f"No {status} model has been registered")
        return self.load_version(version)

    @staticmethod
    def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
        values = []
        for row in rows:
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return sum(values) / len(values) if values else float("nan")

    def _initial_promotion_settings(self) -> dict[str, float]:
        path = self.project_root / "config" / "ml_governance.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            settings = payload.get("model_promotion", {}).get("initial_production", {})
            return dict(settings) if isinstance(settings, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def candidate_promotion_evaluation(self, comparison: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate bootstrap or replacement gates without changing registry pointers."""
        registry = self._read_registry()
        candidate = registry.get("candidate")
        if not candidate:
            return {"passed": False, "mode": "none", "failed_checks": ["no_candidate"]}

        if not registry.get("production"):
            package = self.load_version(candidate)
            metrics = package.get("training_metrics", [])
            rows = [
                row for row in metrics
                if isinstance(row, dict) and row.get("model_group") == "dynamic_group_ensemble"
            ]
            if not rows:
                rows = [
                    row for row in metrics
                    if isinstance(row, dict) and row.get("model_group") == "all_factor"
                ]
            settings = self._initial_promotion_settings()
            if rows and settings:
                summary = {
                    "oos_windows": len(rows),
                    "auc": self._mean_metric(rows, "auc"),
                    "rank_ic": self._mean_metric(rows, "rank_ic"),
                    "net_return": self._mean_metric(rows, "net_return"),
                    "max_drawdown": self._mean_metric(rows, "max_drawdown"),
                    "brier": self._mean_metric(rows, "brier"),
                }
                checks = {
                    "regression_tests": bool(comparison.get("regression_tests_passed", False)),
                    "oos_windows": summary["oos_windows"] >= int(settings.get("minimum_oos_windows", 12)),
                    "auc": summary["auc"] >= float(settings.get("minimum_auc", 0.52)),
                    "rank_ic": summary["rank_ic"] > float(settings.get("minimum_rank_ic", 0.0)),
                    "net_return": summary["net_return"] >= float(settings.get("minimum_net_return", 0.0)),
                    "max_drawdown": summary["max_drawdown"] >= float(settings.get("minimum_max_drawdown", -0.15)),
                    "brier": summary["brier"] <= float(settings.get("maximum_brier", 0.255)),
                }
                failed = [name for name, passed in checks.items() if not passed]
                return {
                    "passed": not failed,
                    "mode": "initial_absolute",
                    "failed_checks": failed,
                    "checks": checks,
                    "metrics": summary,
                }

        required_windows = int(comparison.get("minimum_better_windows", 3))
        checks = {
            "consecutive_better_windows": int(comparison.get("consecutive_better_windows", 0)) >= required_windows,
            "net_return_not_worse": bool(comparison.get("net_return_not_worse", False)),
            "drawdown_not_worse": bool(comparison.get("drawdown_not_worse", False)),
            "regression_tests": bool(comparison.get("regression_tests_passed", False)),
        }
        failed = [name for name, passed in checks.items() if not passed]
        return {"passed": not failed, "mode": "replacement_relative", "failed_checks": failed, "checks": checks}

    def promote_candidate(self, comparison: Mapping[str, Any]) -> bool:
        """Promote only after the applicable bootstrap or replacement gate passes."""
        evaluation = self.candidate_promotion_evaluation(comparison)
        if not evaluation["passed"]:
            return False
        registry = self._read_registry()
        candidate = registry.get("candidate")
        if not candidate:
            return False
        registry["previous_production"] = registry.get("production")
        registry["production"] = candidate
        registry["candidate"] = None
        self._write_registry(registry)
        return True

    def rollback_production(self) -> bool:
        """Atomically swap production with the immediately previous production."""
        registry = self._read_registry()
        previous = registry.get("previous_production")
        current = registry.get("production")
        if not previous:
            return False
        if not self.version_path(previous).exists():
            raise FileNotFoundError(self.version_path(previous))
        registry["production"] = previous
        registry["previous_production"] = current
        self._write_registry(registry)
        return True
