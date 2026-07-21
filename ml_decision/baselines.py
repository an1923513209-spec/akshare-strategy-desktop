"""Stable OOS baselines used to challenge the production tree model."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import evaluate_prediction_frame
from .rolling import RollingWindow


def _frame_for_dates(dataset: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    normalized = pd.to_datetime(dataset["date"], errors="coerce").dt.normalize()
    return dataset.loc[normalized.isin(dates)].sort_values(["date", "code"], kind="stable")


def evaluate_stable_baselines(
    dataset: pd.DataFrame,
    features: Iterable[str],
    window: RollingWindow,
    *,
    transaction_cost: float,
) -> list[dict[str, float | str]]:
    """Fit simple frozen baselines on train dates and evaluate untouched test dates."""
    columns = [column for column in dict.fromkeys(features) if column in dataset.columns]
    if not columns:
        return []
    train = _frame_for_dates(dataset, window.train_dates).dropna(
        subset=["next_open_to_next_open_return"]
    )
    test = _frame_for_dates(dataset, window.test_dates).dropna(
        subset=["next_open_to_next_open_return"]
    )
    if train.empty or test.empty:
        return []
    x_train = train[columns]
    x_test = test[columns]
    y_train = train["next_open_to_next_open_return"].astype(float)
    label_train = (y_train > 0).astype(int)
    rows: list[dict[str, float | str]] = []

    ridge = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
    )
    ridge.fit(x_train, y_train)
    predicted_return = ridge.predict(x_test)

    if label_train.nunique() >= 2:
        logistic = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        logistic.fit(x_train, label_train)
        probability = logistic.predict_proba(x_test)[:, 1]
        frame = test[["date", "code", "next_open_to_next_open_return"]].copy()
        frame["probability_up"] = probability
        frame["predicted_return"] = predicted_return
        rows.append(
            {
                "model_group": "baseline_logistic_ridge",
                **window.ranges(),
                **evaluate_prediction_frame(frame, transaction_cost=transaction_cost),
            }
        )

    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(x_train)
    test_values = imputer.transform(x_test)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std <= 1e-12] = 1.0
    score = ((test_values - mean) / std).mean(axis=1)
    scale = max(float(np.nanstd(score)), 1e-6)
    probability = 1.0 / (1.0 + np.exp(-np.clip(score / scale, -20, 20)))
    equal_frame = test[["date", "code", "next_open_to_next_open_return"]].copy()
    equal_frame["probability_up"] = probability
    equal_frame["predicted_return"] = score * max(float(y_train.std()), 1e-6)
    rows.append(
        {
            "model_group": "baseline_equal_factor",
            **window.ranges(),
            **evaluate_prediction_frame(equal_frame, transaction_cost=transaction_cost),
        }
    )
    return rows
