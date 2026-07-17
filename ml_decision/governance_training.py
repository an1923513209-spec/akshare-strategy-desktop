"""Fixed-window model training used by monthly and quarterly governance jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import evaluate_prediction_frame
from .features import feature_columns
from .models import NextSessionModel
from .rolling import RollingWindow


@dataclass(slots=True)
class WindowModelResult:
    window: RollingWindow
    group: str
    status: str
    feature_columns: list[str] = field(default_factory=list)
    model: Any | None = None
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def _frame_for_dates(dataset: pd.DataFrame, dates: Iterable) -> pd.DataFrame:
    normalized = pd.to_datetime(dataset["date"], errors="coerce").dt.normalize()
    return dataset.loc[normalized.isin(pd.DatetimeIndex(dates))].sort_values(["date", "code"], kind="stable")


def model_prediction_frame(model: NextSessionModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Return both probability and expected return for an evaluation partition."""
    x = frame[model.feature_columns]
    result = frame[["date", "code", "next_open_to_next_open_return"]].copy()
    result["probability_up"] = model.models["up"].predict_proba(x)[:, 1]
    result["predicted_return"] = model.models["open_to_open"].predict(x)
    return result


def model_shap_frame(model: NextSessionModel, frame: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
    """Calculate native TreeSHAP values for the regression output on test samples."""
    sample = frame.head(max(int(max_samples), 1))
    pipeline = model.models["open_to_open"]
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Open-to-open model is not an explainable pipeline")
    imputed = pipeline.named_steps["imputer"].transform(sample[model.feature_columns])
    estimator = pipeline.named_steps["model"]
    import xgboost as xgb

    values = estimator.get_booster().predict(
        xgb.DMatrix(imputed, feature_names=model.feature_columns), pred_contribs=True
    )
    if values.shape[1] != len(model.feature_columns) + 1:
        raise ValueError("Unexpected TreeSHAP contribution width")
    return pd.DataFrame(values[:, :-1], columns=model.feature_columns, index=sample.index)


def train_fixed_window(
    dataset: pd.DataFrame,
    candidate_features: Iterable[str],
    window: RollingWindow,
    *,
    group: str,
    config: Mapping,
    model_factory: Callable[..., NextSessionModel] = NextSessionModel,
) -> WindowModelResult:
    """Train on one immutable rolling window; selection sees train rows only."""
    train = _frame_for_dates(dataset, window.train_dates)
    test = _frame_for_dates(dataset, window.test_dates)
    candidates = list(dict.fromkeys(column for column in candidate_features if column in dataset.columns))
    all_eligible = feature_columns(dataset, selection_df=train)
    selected = [column for column in candidates if column in set(all_eligible)]
    if not selected:
        return WindowModelResult(window, group, "INACTIVE", reason="no_train_eligible_features")
    minimum_coverage = float(config["factor_evaluation"].get("minimum_group_coverage", 0.03))
    coverage = float(train[selected].notna().mean().mean())
    if coverage < minimum_coverage:
        return WindowModelResult(window, group, "INACTIVE", selected, reason=f"coverage_below_threshold:{coverage:.6f}")
    model_settings = config.get("model", {})
    model = model_factory(
        calibration_method=model_settings.get("calibration_method", "sigmoid"),
        random_state=int(model_settings.get("random_state", 42)),
        use_shap=bool(model_settings.get("use_shap", True)),
    )
    model.fit(dataset, selected, splits=window.as_splits())
    predictions = model_prediction_frame(model, test)
    metrics = evaluate_prediction_frame(
        predictions,
        transaction_cost=float(config["factor_evaluation"].get("transaction_cost", 0.0016)),
    )
    metrics.update(window.ranges())
    metrics.update({"model_group": group, "feature_count": len(selected), "coverage_rate": coverage})
    return WindowModelResult(window, group, "ACTIVE", selected, model, predictions, metrics)


def train_window_model_set(
    dataset: pd.DataFrame,
    groups: Mapping[str, Iterable[str]],
    window: RollingWindow,
    config: Mapping,
    *,
    model_factory: Callable[..., NextSessionModel] = NextSessionModel,
) -> dict[str, WindowModelResult]:
    """Train the all-factor model and each configured group model on identical dates."""
    controls = [column for column in config.get("control_features", []) if column in dataset.columns]
    all_features = list(dict.fromkeys(column for values in groups.values() for column in values))
    requests = {"all_factor": all_features}
    for group in (
        "technical", "liquidity", "fund_flow", "institution", "news", "lhb",
        "lhb_institution", "fundamental",
    ):
        requests[group] = list(dict.fromkeys(list(groups.get(group, [])) + controls))
    results: dict[str, WindowModelResult] = {}
    for group, features in requests.items():
        try:
            results[group] = train_fixed_window(
                dataset, features, window, group=group, config=config, model_factory=model_factory
            )
        except Exception as exc:
            results[group] = WindowModelResult(
                window=window,
                group=group,
                status="FAILED",
                reason=f"{type(exc).__name__}: {exc}",
            )
    return results
