"""Strict month-based rolling out-of-sample date windows."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import pandas as pd

from .models import PurgedDateSplits


@dataclass(frozen=True, slots=True)
class RollingWindow:
    window_id: str
    train_dates: pd.DatetimeIndex
    calibration_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    purge_trading_days: int
    embargo_trading_days: int

    def as_splits(self) -> PurgedDateSplits:
        return PurgedDateSplits(
            self.train_dates,
            self.calibration_dates,
            self.validation_dates,
            self.test_dates,
            max(self.purge_trading_days, self.embargo_trading_days),
        )

    def ranges(self) -> dict[str, str]:
        result: dict[str, str] = {"window_id": self.window_id}
        for part in ("train", "calibration", "validation", "test"):
            dates = getattr(self, f"{part}_dates")
            result[f"{part}_start"] = str(dates.min().date()) if len(dates) else ""
            result[f"{part}_end"] = str(dates.max().date()) if len(dates) else ""
        return result


def _dates_in_periods(dates: pd.DatetimeIndex, periods: Iterable[pd.Period]) -> pd.DatetimeIndex:
    wanted = set(periods)
    return dates[[period in wanted for period in dates.to_period("M")]]


def generate_rolling_windows(
    dates: Iterable,
    *,
    train_months: int = 36,
    calibration_months: int = 2,
    validation_months: int = 1,
    test_months: int = 1,
    step_months: int = 1,
    purge_trading_days: int = 2,
    embargo_trading_days: int = 2,
) -> list[RollingWindow]:
    """Create 36/2/1/1 month windows with explicit boundary exclusions."""
    unique_dates = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna().normalize().unique().sort_values()
    months = pd.PeriodIndex(unique_dates.to_period("M").unique()).sort_values()
    required = train_months + calibration_months + validation_months + test_months
    purge = max(int(purge_trading_days), 2)
    embargo = max(int(embargo_trading_days), 2)
    if len(months) < required:
        return []

    windows: list[RollingWindow] = []
    for start in range(0, len(months) - required + 1, max(int(step_months), 1)):
        cursor = start
        train_month_set = months[cursor : cursor + train_months]
        cursor += train_months
        calibration_month_set = months[cursor : cursor + calibration_months]
        cursor += calibration_months
        validation_month_set = months[cursor : cursor + validation_months]
        cursor += validation_months
        test_month_set = months[cursor : cursor + test_months]

        train = _dates_in_periods(unique_dates, train_month_set)
        calibration = _dates_in_periods(unique_dates, calibration_month_set)
        validation = _dates_in_periods(unique_dates, validation_month_set)
        test = _dates_in_periods(unique_dates, test_month_set)
        if min(map(len, (train, calibration, validation, test))) <= purge + embargo:
            continue

        # Remove both the label purge from the older set and the execution
        # embargo from the newer set. The excluded dates never enter any split.
        train = train[:-purge]
        calibration = calibration[embargo:-purge]
        validation = validation[embargo:-purge]
        test = test[embargo:]
        window = RollingWindow(
            window_id=str(test_month_set[-1]),
            train_dates=train,
            calibration_dates=calibration,
            validation_dates=validation,
            test_dates=test,
            purge_trading_days=purge,
            embargo_trading_days=embargo,
        )
        validate_rolling_window(window)
        windows.append(window)
    return windows


def validate_rolling_window(window: RollingWindow) -> None:
    parts = [window.train_dates, window.calibration_dates, window.validation_dates, window.test_dates]
    sets = [set(part) for part in parts]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            if sets[left].intersection(sets[right]):
                raise ValueError(f"Rolling split date overlap in {window.window_id}")
    for older, newer in zip(parts, parts[1:]):
        if len(older) and len(newer) and older.max() >= newer.min():
            raise ValueError(f"Rolling split is not chronological in {window.window_id}")


def rolling_windows_from_config(dataset: pd.DataFrame, config: dict) -> list[RollingWindow]:
    settings = config["rolling_training"]
    return generate_rolling_windows(dataset["date"], **settings)
