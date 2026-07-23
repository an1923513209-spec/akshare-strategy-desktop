"""Factor-driven target-weight policy and leakage-safe per-stock OOS backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FactorPolicyParameters:
    """Parameters selected only from the policy-calibration OOS period."""

    probability_return_scale: float = 0.01
    downside_penalty: float = 0.50
    concentration_penalty: float = 0.02
    minimum_utility_edge: float = 0.0005
    max_single_weight: float = 0.25
    weight_step: float = 0.05
    transaction_cost: float = 0.0016

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FactorPolicyParameters":
        source = value or {}
        allowed = {field: source[field] for field in cls.__dataclass_fields__ if field in source}
        return cls(**allowed)


def candidate_weights(parameters: FactorPolicyParameters, maximum: float | None = None) -> np.ndarray:
    cap = min(max(float(maximum if maximum is not None else parameters.max_single_weight), 0.0), 1.0)
    step = max(float(parameters.weight_step), 0.01)
    values = np.arange(0.0, cap + step * 0.5, step)
    values = np.clip(values, 0.0, cap)
    return np.unique(np.append(values, cap)).round(8)


def target_weight_utilities(
    *,
    probability_up: float,
    predicted_return: float,
    probability_down: float,
    return_q10: float,
    current_weight: float,
    parameters: FactorPolicyParameters,
    maximum: float | None = None,
) -> pd.DataFrame:
    """Score absolute target weights using only model predictions known at decision time."""
    weights = candidate_weights(parameters, maximum)
    probability = float(probability_up) if np.isfinite(probability_up) else 0.5
    expected_return = float(predicted_return) if np.isfinite(predicted_return) else 0.0
    downside_probability = float(probability_down) if np.isfinite(probability_down) else 0.5
    q10 = float(return_q10) if np.isfinite(return_q10) else -0.02
    probability_edge = (float(np.clip(probability, 0.0, 1.0)) - 0.5) * 2.0
    adjusted_return = expected_return + parameters.probability_return_scale * probability_edge
    tail_loss = max(-q10, 0.0)
    down_probability = float(np.clip(downside_probability, 0.0, 1.0))
    turnover = np.abs(weights - max(float(current_weight), 0.0))
    expected_component = weights * adjusted_return
    downside_component = weights * parameters.downside_penalty * tail_loss * (0.5 + down_probability)
    concentration_component = parameters.concentration_penalty * np.square(weights)
    cost_component = turnover * parameters.transaction_cost
    utility = expected_component - downside_component - concentration_component - cost_component
    utility = np.where(utility >= parameters.minimum_utility_edge, utility, np.where(weights == 0.0, utility, -np.inf))
    return pd.DataFrame(
        {
            "target_weight": weights,
            "expected_component": expected_component,
            "downside_component": downside_component,
            "concentration_component": concentration_component,
            "transaction_cost_component": cost_component,
            "utility": utility,
        }
    )


def choose_target_weight(
    *,
    probability_up: float,
    predicted_return: float,
    probability_down: float,
    return_q10: float,
    current_weight: float,
    parameters: FactorPolicyParameters,
    maximum: float | None = None,
) -> tuple[float, float, list[dict[str, float]]]:
    scores = target_weight_utilities(
        probability_up=probability_up,
        predicted_return=predicted_return,
        probability_down=probability_down,
        return_q10=return_q10,
        current_weight=current_weight,
        parameters=parameters,
        maximum=maximum,
    )
    finite = scores[np.isfinite(scores["utility"])]
    selected = finite.loc[finite["utility"].idxmax()] if not finite.empty else scores.iloc[0]
    records = [
        {key: float(value) for key, value in row.items()}
        for row in scores.replace([np.inf, -np.inf], np.nan).to_dict("records")
    ]
    return float(selected["target_weight"]), float(selected["utility"]), records


def _policy_action(current_weight: float, target_weight: float, tolerance: float = 1e-9) -> str:
    if abs(target_weight - current_weight) <= tolerance:
        return "HOLD"
    if target_weight <= tolerance:
        return "SELL/CLEAR"
    if target_weight > current_weight:
        return "BUY/ADD"
    return "REDUCE"


def simulate_factor_policy(
    predictions: pd.DataFrame,
    parameters: FactorPolicyParameters,
) -> pd.DataFrame:
    """Run an independent, path-aware OOS backtest for every stock."""
    required = {
        "date", "code", "next_open_to_next_open_return", "probability_up",
        "predicted_return", "probability_down_2pct", "return_q10",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Policy backtest is missing columns: {missing}")
    data = predictions.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data.dropna(subset=["date", "code", "next_open_to_next_open_return"])
    data = data.sort_values(["code", "date"], kind="stable").drop_duplicates(["date", "code"], keep="last")
    output: list[dict[str, Any]] = []
    for code, stock in data.groupby("code", sort=False):
        current_weight = 0.0
        equity = 1.0
        benchmark_equity = 1.0
        peak = 1.0
        for row in stock.to_dict("records"):
            target, predicted_utility, candidates = choose_target_weight(
                probability_up=float(row.get("probability_up", 0.5)),
                predicted_return=float(row.get("predicted_return", 0.0)),
                probability_down=float(row.get("probability_down_2pct", 0.5)),
                return_q10=float(row.get("return_q10", -0.02)),
                current_weight=current_weight,
                parameters=parameters,
            )
            realized = float(row["next_open_to_next_open_return"])
            trade_weight = target - current_weight
            transaction_cost = abs(trade_weight) * parameters.transaction_cost
            gross_return = target * realized
            net_return = gross_return - transaction_cost
            equity *= max(1.0 + net_return, 1e-9)
            benchmark_return = parameters.max_single_weight * realized
            benchmark_equity *= max(1.0 + benchmark_return, 1e-9)
            peak = max(peak, equity)
            output.append(
                {
                    **row,
                    "current_weight": current_weight,
                    "target_weight": target,
                    "trade_weight": trade_weight,
                    "policy_action": _policy_action(current_weight, target),
                    "predicted_policy_utility": predicted_utility,
                    "gross_strategy_return": gross_return,
                    "transaction_cost": transaction_cost,
                    "net_strategy_return": net_return,
                    "equity": equity,
                    "drawdown": equity / peak - 1.0,
                    "benchmark_return": benchmark_return,
                    "benchmark_equity": benchmark_equity,
                    "candidate_utilities": candidates,
                }
            )
            current_weight = target
    return pd.DataFrame(output)


def policy_backtest_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for code, stock in backtest.groupby("code", sort=False):
        stock = stock.sort_values("date", kind="stable")
        net = pd.to_numeric(stock["net_strategy_return"], errors="coerce").fillna(0.0)
        volatility = float(net.std(ddof=0))
        rows.append(
            {
                "code": code,
                "name": str(stock.get("name", pd.Series([""])).iloc[-1] or ""),
                "start_date": str(pd.to_datetime(stock["date"]).min().date()),
                "end_date": str(pd.to_datetime(stock["date"]).max().date()),
                "observations": int(len(stock)),
                "policy_return": float(stock["equity"].iloc[-1] - 1.0),
                "benchmark_return": float(stock["benchmark_equity"].iloc[-1] - 1.0),
                "excess_return": float(stock["equity"].iloc[-1] - stock["benchmark_equity"].iloc[-1]),
                "max_drawdown": float(pd.to_numeric(stock["drawdown"], errors="coerce").min()),
                "sharpe": float(net.mean() / volatility * np.sqrt(252.0)) if volatility > 1e-12 else 0.0,
                "turnover": float(pd.to_numeric(stock["trade_weight"], errors="coerce").abs().sum()),
                "trade_days": int(pd.to_numeric(stock["trade_weight"], errors="coerce").abs().gt(1e-9).sum()),
                "win_rate": float(net.gt(0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("policy_return", ascending=False, kind="stable").reset_index(drop=True)


def _portfolio_objective(backtest: pd.DataFrame, drawdown_penalty: float, turnover_penalty: float) -> float:
    if backtest.empty:
        return -np.inf
    daily = backtest.groupby("date", sort=True)["net_strategy_return"].mean()
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    total_return = float(equity.iloc[-1] - 1.0)
    turnover = float(backtest.groupby("date")["trade_weight"].apply(lambda values: values.abs().mean()).mean())
    return total_return - drawdown_penalty * abs(float(drawdown.min())) - turnover_penalty * turnover


def calibrate_factor_policy(
    predictions: pd.DataFrame,
    settings: Mapping[str, Any] | None = None,
) -> tuple[FactorPolicyParameters, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Select policy parameters on early OOS dates and evaluate on untouched later OOS dates."""
    config = dict(settings or {})
    data = predictions.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    dates = pd.DatetimeIndex(data["date"].dropna().drop_duplicates().sort_values())
    fraction = float(config.get("calibration_fraction", 0.70))
    embargo = max(int(config.get("embargo_trading_days", 2)), 0)
    split = min(max(int(len(dates) * fraction), 1), max(len(dates) - embargo - 1, 1))
    calibration_dates = dates[:split]
    test_dates = dates[min(split + embargo, len(dates)):]
    if len(test_dates) == 0:
        test_dates = dates[-max(1, len(dates) // 3):]
        calibration_dates = dates[: max(1, len(dates) - len(test_dates) - embargo)]

    calibration = data[data["date"].isin(calibration_dates)].copy()
    max_symbols = max(int(config.get("max_calibration_symbols", 300)), 1)
    if calibration["code"].nunique() > max_symbols:
        selected_codes = (
            calibration.groupby("code").size().sort_values(ascending=False, kind="stable").head(max_symbols).index
        )
        calibration = calibration[calibration["code"].isin(selected_codes)]

    base = {
        "max_single_weight": float(config.get("max_single_weight", 0.25)),
        "weight_step": float(config.get("weight_step", 0.05)),
        "transaction_cost": float(config.get("transaction_cost", 0.0016)),
    }
    grids = config.get("parameter_grid", {})
    probability_values = grids.get("probability_return_scale", [0.0, 0.01, 0.02])
    downside_values = grids.get("downside_penalty", [0.25, 0.50, 1.0])
    concentration_values = grids.get("concentration_penalty", [0.0, 0.02, 0.05])
    edge_values = grids.get("minimum_utility_edge", [0.0, 0.0005])
    drawdown_penalty = float(config.get("objective_drawdown_penalty", 0.50))
    turnover_penalty = float(config.get("objective_turnover_penalty", 0.10))
    best_parameters: FactorPolicyParameters | None = None
    best_objective = -np.inf
    evaluated = 0
    for probability_scale, downside, concentration, edge in product(
        probability_values, downside_values, concentration_values, edge_values
    ):
        parameters = FactorPolicyParameters(
            probability_return_scale=float(probability_scale),
            downside_penalty=float(downside),
            concentration_penalty=float(concentration),
            minimum_utility_edge=float(edge),
            **base,
        )
        trial = simulate_factor_policy(calibration, parameters)
        objective = _portfolio_objective(trial, drawdown_penalty, turnover_penalty)
        evaluated += 1
        if objective > best_objective:
            best_objective = objective
            best_parameters = parameters
    selected = best_parameters or FactorPolicyParameters(**base)
    all_backtest = simulate_factor_policy(data, selected)
    test_backtest = all_backtest[all_backtest["date"].isin(test_dates)].copy()
    summary = policy_backtest_summary(test_backtest)
    report = {
        "status": "calibrated" if len(calibration_dates) > 0 and len(test_dates) > 0 else "insufficient_oos",
        "decision_timing": "factors known after t close; rebalance at t+1 open; evaluate at t+2 open",
        "benchmark": "constant max_single_weight exposure without timing",
        "calibration_start": str(calibration_dates.min().date()) if len(calibration_dates) else None,
        "calibration_end": str(calibration_dates.max().date()) if len(calibration_dates) else None,
        "test_start": str(test_dates.min().date()) if len(test_dates) else None,
        "test_end": str(test_dates.max().date()) if len(test_dates) else None,
        "embargo_trading_days": embargo,
        "calibration_symbols": int(calibration["code"].nunique()),
        "evaluated_parameter_sets": evaluated,
        "calibration_objective": float(best_objective),
        "parameters": selected.to_dict(),
        "test_stock_count": int(test_backtest["code"].nunique()) if not test_backtest.empty else 0,
    }
    return selected, report, test_backtest, summary
