"""Action scoring under A-share trading constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import AccountState, DecisionConfig
from .models import PredictionPack


ActionName = Literal["SELL_ALL", "REDUCE_50", "REDUCE_25", "HOLD", "ADD_25", "ADD_50"]
ACTION_MULTIPLIERS: dict[ActionName, float] = {
    "SELL_ALL": -1.0,
    "REDUCE_50": -0.50,
    "REDUCE_25": -0.25,
    "HOLD": 0.0,
    "ADD_25": 0.25,
    "ADD_50": 0.50,
}


@dataclass(slots=True)
class Holding:
    """Current holding row used by the action layer."""

    code: str
    shares: int
    available_shares: int
    average_cost: float
    current_price: float
    position_value: float
    position_weight: float
    holding_days: int = 0
    industry: str = ""
    name: str = ""


@dataclass(slots=True)
class ActionScore:
    """Utility and trade details for one candidate action."""

    action: ActionName
    trade_shares: int
    target_shares: int
    target_weight: float
    expected_net_return: float
    expected_net_pnl: float
    downside_risk: float
    transaction_cost: float
    turnover: float
    utility_score: float
    feasible: bool
    reason: str


def _round_lot(shares: float, lot_size: int) -> int:
    if shares <= 0:
        return 0
    return int(shares // lot_size) * lot_size


def _trade_cost(value: float, is_sell: bool, account: AccountState) -> float:
    if value <= 0:
        return 0.0
    commission = max(value * account.commission_rate, account.minimum_commission)
    stamp = value * account.stamp_duty_rate if is_sell else 0.0
    slippage = value * account.slippage_rate
    return commission + stamp + slippage


def _is_one_price_limit(row: pd.Series, up: bool) -> bool:
    limit_col = "limit_up_price" if up else "limit_down_price"
    if limit_col not in row or pd.isna(row.get(limit_col)):
        return False
    limit_price = float(row[limit_col])
    return all(abs(float(row[col]) - limit_price) < 1e-6 for col in ("open", "high", "low", "close"))


def _ml_edge_score(prediction: PredictionPack) -> float:
    weights = prediction.score_weights or {"probability": 0.45, "expected": 0.35, "risk": 0.20}
    prob_component = (float(prediction.probability_up) - 0.5) * 2.0
    expected_component = float(np.tanh(float(prediction.expected_open_to_open_return) / 0.03))
    q10_risk = float(np.clip(-float(prediction.return_q10), 0.0, 0.08) / 0.08)
    risk_component = (float(prediction.probability_down_2pct) + q10_risk) / 2.0
    return (
        float(weights.get("probability", 0.45)) * prob_component
        + float(weights.get("expected", 0.35)) * expected_component
        - float(weights.get("risk", 0.20)) * risk_component
    )


def score_actions(
    holding: Holding,
    latest_row: pd.Series,
    prediction: PredictionPack,
    account: AccountState,
    config: DecisionConfig,
) -> list[ActionScore]:
    """Score all candidate actions and return them in declaration order."""
    account = account.normalized()
    current_value = max(float(holding.position_value), 0.0)
    current_weight = current_value / account.total_asset if account.total_asset > 0 else 0.0
    scores: list[ActionScore] = []
    suspended = bool(latest_row.get("is_suspended", False))
    no_volume = float(latest_row.get("volume", 0) or 0) <= 0
    cannot_buy = suspended or no_volume or _is_one_price_limit(latest_row, up=True)
    cannot_sell = suspended or no_volume or _is_one_price_limit(latest_row, up=False)
    edge_score = _ml_edge_score(prediction)
    score_expected_return = max(edge_score, -1.0) * 0.035

    for action, multiplier in ACTION_MULTIPLIERS.items():
        feasible = True
        reason = ""
        raw_trade_value = current_value * multiplier
        if multiplier < 0:
            sell_shares = min(holding.available_shares, int(abs(multiplier) * holding.shares))
            trade_shares = -_round_lot(sell_shares, account.lot_size)
            if cannot_sell and trade_shares < 0:
                feasible = False
                reason = "停牌/无量/一字跌停，不能假设卖出成交"
        elif multiplier > 0:
            max_value_by_single = max(account.max_single_position_weight * account.total_asset - current_value, 0.0)
            if current_value <= 0:
                desired_buy_value = account.max_single_position_weight * account.total_asset * multiplier
            else:
                desired_buy_value = raw_trade_value
            buy_value = min(desired_buy_value, account.cash, max_value_by_single)
            buy_shares = _round_lot(buy_value / max(float(holding.current_price), 0.01), account.lot_size)
            trade_shares = buy_shares
            if cannot_buy and trade_shares > 0:
                feasible = False
                reason = "停牌/无量/一字涨停，不能假设买入成交"
        else:
            trade_shares = 0

        trade_value = abs(trade_shares) * holding.current_price
        if trade_shares == 0:
            action_effective: ActionName = "HOLD"
        elif 0 < trade_value < account.minimum_trade_amount:
            trade_shares = 0
            trade_value = 0.0
            action_effective = "HOLD"
        else:
            action_effective = action

        target_shares = holding.shares + trade_shares
        target_value = max(target_shares, 0) * holding.current_price
        target_weight = target_value / account.total_asset if account.total_asset > 0 else 0.0
        if target_weight > account.max_single_position_weight + 1e-9:
            feasible = False
            reason = "超过单股最大仓位"

        buy_cost = _trade_cost(trade_value, is_sell=False, account=account) if trade_shares > 0 else 0.0
        sell_cost = _trade_cost(trade_value, is_sell=True, account=account) if trade_shares < 0 else 0.0
        transaction_cost = buy_cost + sell_cost
        exposure = target_value / account.total_asset if account.total_asset > 0 else 0.0
        expected_net_return = exposure * score_expected_return
        expected_net_pnl = target_value * score_expected_return - transaction_cost
        downside_risk = exposure * abs(min(prediction.return_q10, 0.0))
        turnover = trade_value / account.total_asset if account.total_asset > 0 else 0.0
        if config.utility_mode == "downside_probability":
            utility = expected_net_return - account.risk_aversion * prediction.probability_down_2pct * exposure
        else:
            utility = expected_net_return - account.risk_aversion * downside_risk
        utility -= account.turnover_penalty * turnover + transaction_cost / max(account.total_asset, 1.0)

        if action.startswith("ADD"):
            add_blocked = (
                edge_score <= 0.10
                or prediction.probability_up < config.add_probability_threshold
                or prediction.expected_open_to_open_return <= transaction_cost / max(trade_value, 1.0)
                or prediction.return_q10 <= config.add_q10_threshold
                or prediction.probability_down_2pct > config.max_add_downside_probability
            )
            if add_blocked:
                feasible = False
                reason = "加仓门槛未通过：超成本概率/预期收益/下行风险不足"

        if not feasible:
            utility = -np.inf
        scores.append(
            ActionScore(
                action=action_effective,
                trade_shares=trade_shares,
                target_shares=max(target_shares, 0),
                target_weight=target_weight,
                expected_net_return=expected_net_return,
                expected_net_pnl=expected_net_pnl,
                downside_risk=downside_risk,
                transaction_cost=transaction_cost,
                turnover=turnover,
                utility_score=float(utility),
                feasible=feasible,
                reason=reason,
            )
        )
    return scores


def choose_action(scores: list[ActionScore], minimum_action_edge: float) -> ActionScore:
    """Select the best action, defaulting to HOLD unless the edge is meaningful."""
    hold = next((score for score in scores if score.action == "HOLD"), None)
    feasible = [score for score in scores if score.feasible]
    if not feasible:
        return hold or scores[0]
    best = max(feasible, key=lambda score: score.utility_score)
    if hold is not None and best.action != "HOLD":
        if best.utility_score - hold.utility_score <= minimum_action_edge:
            return hold
    return best
