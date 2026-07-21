"""Configuration objects for the holding decision engine.

The engine deliberately keeps account state and model settings outside the
model code so transaction costs and risk limits are visible and adjustable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import json


UtilityMode = Literal["q10", "downside_probability"]
TrainingUniverse = Literal["all_market", "same_industry", "custom"]


@dataclass(slots=True)
class AccountState:
    """Trading constraints and account-level state."""

    cash: float = 0.0
    total_asset: float = 100000.0
    max_single_position_weight: float = 0.25
    max_industry_weight: float = 0.45
    max_total_position_weight: float = 0.80
    minimum_trade_amount: float = 1000.0
    lot_size: int = 100
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    slippage_rate: float = 0.001
    risk_aversion: float = 1.8
    turnover_penalty: float = 0.001

    def normalized(self) -> "AccountState":
        """Return a copy with percentages converted if the UI supplied 0-100 values."""
        data = asdict(self)
        for key in ("max_single_position_weight", "max_industry_weight", "max_total_position_weight"):
            value = float(data[key])
            data[key] = value / 100.0 if value > 1.0 else value
        return AccountState(**data)


@dataclass(slots=True)
class DecisionConfig:
    """Model, validation, feature, and action-scoring settings."""

    start_date: str = "20200101"
    training_universe: TrainingUniverse = "custom"
    external_factor_lag: int = 0
    down_threshold: float = -0.02
    utility_mode: UtilityMode = "q10"
    minimum_action_edge: float = 0.001
    add_probability_threshold: float = 0.55
    add_q10_threshold: float = -0.025
    max_add_downside_probability: float = 0.35
    profitable_cost_buffer: float = 0.0
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    test_years: int = 1
    validation_years: int = 1
    train_min_rows: int = 260
    use_shap: bool = True
    feature_exclude: tuple[str, ...] = field(default_factory=tuple)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file. Missing files return an empty dict."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def save_default_config(path: str | Path) -> None:
    """Write a readable default configuration file."""
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"account": asdict(AccountState()), "decision": asdict(DecisionConfig())}
    with cfg_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
