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

    def list_versions(self) -> list[str]:
        """List complete immutable model versions, newest names first."""
        versions = [
            path.name
            for path in self.models_root.iterdir()
            if path.is_dir()
            and (path / "model_metadata.json").exists()
            and (path / MODEL_FILES["all_factor"]).exists()
        ]
        return sorted(versions, reverse=True)

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

    def _promotion_settings(self) -> dict[str, Any]:
        path = self.project_root / "config" / "ml_governance.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            settings = payload.get("model_promotion", {})
            return dict(settings) if isinstance(settings, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _preferred_metric_rows(package: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows = [row for row in package.get("training_metrics", []) if isinstance(row, dict)]
        ensemble = [row for row in rows if row.get("model_group") == "dynamic_group_ensemble"]
        return ensemble or [row for row in rows if row.get("model_group") == "all_factor"]

    @staticmethod
    def _window_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
        start = str(row.get("test_start") or "")
        end = str(row.get("test_end") or "")
        if start and end:
            return start, end
        window = str(row.get("window_id") or "")
        return (window, window) if window else None

    @classmethod
    def _shared_oos_metrics(
        cls,
        candidate: Mapping[str, Any],
        production: Mapping[str, Any],
        recent_windows: int,
    ) -> tuple[dict[str, float], dict[str, float], list[tuple[str, str]]]:
        candidate_rows = {
            key: row
            for row in cls._preferred_metric_rows(candidate)
            if (key := cls._window_key(row)) is not None
        }
        production_rows = {
            key: row
            for row in cls._preferred_metric_rows(production)
            if (key := cls._window_key(row)) is not None
        }
        common = sorted(set(candidate_rows).intersection(production_rows), key=lambda value: value[1])
        selected = common[-max(int(recent_windows), 1):]

        def summarize(rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, float]:
            return {
                metric: cls._mean_metric([dict(rows[key]) for key in selected], metric)
                for metric in ("auc", "rank_ic", "net_return", "max_drawdown", "brier")
            }

        return summarize(candidate_rows), summarize(production_rows), selected

    @classmethod
    def _absolute_metric_checks(
        cls,
        package: Mapping[str, Any],
        settings: Mapping[str, Any],
    ) -> tuple[dict[str, bool], dict[str, float]]:
        rows = cls._preferred_metric_rows(package)
        summary = {
            "oos_windows": len(rows),
            "auc": cls._mean_metric(rows, "auc"),
            "rank_ic": cls._mean_metric(rows, "rank_ic"),
            "net_return": cls._mean_metric(rows, "net_return"),
            "max_drawdown": cls._mean_metric(rows, "max_drawdown"),
            "brier": cls._mean_metric(rows, "brier"),
        }
        checks = {
            "absolute_oos_windows": summary["oos_windows"] >= int(settings.get("minimum_oos_windows", 12)),
            "absolute_auc": summary["auc"] >= float(settings.get("minimum_auc", 0.52)),
            "absolute_rank_ic": summary["rank_ic"] > float(settings.get("minimum_rank_ic", 0.0)),
            "absolute_net_return": summary["net_return"] >= float(settings.get("minimum_net_return", 0.0)),
            "absolute_max_drawdown": summary["max_drawdown"] >= float(settings.get("minimum_max_drawdown", -0.15)),
            "absolute_brier": summary["brier"] <= float(settings.get("maximum_brier", 0.255)),
        }
        return checks, summary

    def candidate_promotion_evaluation(self, comparison: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate bootstrap or replacement gates without changing registry pointers."""
        registry = self._read_registry()
        candidate = registry.get("candidate")
        if not candidate:
            return {"passed": False, "mode": "none", "failed_checks": ["no_candidate"]}

        if not registry.get("production"):
            package = self.load_version(candidate)
            settings = self._initial_promotion_settings()
            if self._preferred_metric_rows(package) and settings:
                checks, summary = self._absolute_metric_checks(package, settings)
                checks = {"regression_tests": bool(comparison.get("regression_tests_passed", False)), **checks}
                failed = [name for name, passed in checks.items() if not passed]
                return {
                    "passed": not failed,
                    "mode": "initial_absolute",
                    "failed_checks": failed,
                    "checks": checks,
                    "metrics": summary,
                }

        production_version = registry.get("production")
        if production_version:
            candidate_package = self.load_version(candidate)
            production_package = self.load_version(production_version)
            settings = self._promotion_settings()
            replacement = settings.get("replacement", {})
            replacement = dict(replacement) if isinstance(replacement, dict) else {}
            recent_windows = int(replacement.get("comparison_recent_windows", 12))
            candidate_metrics, production_metrics, shared_windows = self._shared_oos_metrics(
                candidate_package,
                production_package,
                recent_windows,
            )
            minimum_windows = int(replacement.get("minimum_comparable_windows", 12))
            if shared_windows:
                rank_tolerance = float(replacement.get("rank_ic_tolerance", 0.01))
                return_tolerance = float(replacement.get("net_return_tolerance", 0.0005))
                drawdown_tolerance = float(replacement.get("max_drawdown_tolerance", 0.02))
                brier_tolerance = float(replacement.get("brier_tolerance", 0.005))
                absolute_settings = settings.get("initial_production", {})
                absolute_settings = dict(absolute_settings) if isinstance(absolute_settings, dict) else {}
                absolute_checks, absolute_metrics = self._absolute_metric_checks(
                    candidate_package,
                    absolute_settings,
                )
                checks = {
                    "regression_tests": bool(comparison.get("regression_tests_passed", False)),
                    "comparable_oos_windows": len(shared_windows) >= minimum_windows,
                    "rank_ic_noninferior": candidate_metrics["rank_ic"] >= production_metrics["rank_ic"] - rank_tolerance,
                    "net_return_noninferior": candidate_metrics["net_return"] >= production_metrics["net_return"] - return_tolerance,
                    "drawdown_noninferior": candidate_metrics["max_drawdown"] >= production_metrics["max_drawdown"] - drawdown_tolerance,
                    "brier_noninferior": candidate_metrics["brier"] <= production_metrics["brier"] + brier_tolerance,
                    **absolute_checks,
                }
                failed = [name for name, passed in checks.items() if not passed]
                deltas = {
                    key: candidate_metrics[key] - production_metrics[key]
                    for key in candidate_metrics
                }
                return {
                    "passed": not failed,
                    "mode": "replacement_shared_oos",
                    "candidate_version": candidate,
                    "production_version": production_version,
                    "failed_checks": failed,
                    "checks": checks,
                    "shared_window_count": len(shared_windows),
                    "comparison_window_start": shared_windows[0][0],
                    "comparison_window_end": shared_windows[-1][1],
                    "candidate_metrics": candidate_metrics,
                    "production_metrics": production_metrics,
                    "metric_deltas": deltas,
                    "candidate_absolute_metrics": absolute_metrics,
                    "tolerances": {
                        "rank_ic": rank_tolerance,
                        "net_return": return_tolerance,
                        "max_drawdown": drawdown_tolerance,
                        "brier": brier_tolerance,
                    },
                }

        # Backward compatibility for legacy model packages without dated OOS metrics.
        required_windows = int(comparison.get("minimum_better_windows", 3))
        checks = {
            "consecutive_better_windows": int(comparison.get("consecutive_better_windows", 0)) >= required_windows,
            "net_return_not_worse": bool(comparison.get("net_return_not_worse", False)),
            "drawdown_not_worse": bool(comparison.get("drawdown_not_worse", False)),
            "regression_tests": bool(comparison.get("regression_tests_passed", False)),
        }
        failed = [name for name, passed in checks.items() if not passed]
        return {"passed": not failed, "mode": "legacy_explicit", "failed_checks": failed, "checks": checks}

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

    def set_production_version(self, version: str, *, operator: str = "desktop_manual") -> bool:
        """Manually select a complete immutable version as production, bypassing metric gates."""
        normalized = safe_version_name(version)
        if normalized != str(version).strip():
            raise ValueError(f"Invalid model version: {version}")
        path = self.version_path(normalized)
        if not path.exists():
            raise FileNotFoundError(path)
        # Loading performs the same metadata/model integrity checks used by production inference.
        package = self.load_version(normalized)
        if "all_factor" not in package.get("models", {}):
            raise ValueError(f"Model version has no all_factor model: {normalized}")
        registry = self._read_registry()
        current = registry.get("production")
        if current == normalized:
            return False
        registry["previous_production"] = current
        registry["production"] = normalized
        if registry.get("candidate") == normalized:
            registry["candidate"] = None
        self._write_registry(registry)
        audit = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "operator": operator,
            "mode": "manual_gate_bypass",
            "selected_version": normalized,
            "previous_production": current,
        }
        log_path = self.models_root / "manual_promotion_log.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit, ensure_ascii=False, default=_json_default) + "\n")
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
