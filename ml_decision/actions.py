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
    available_shares_known: bool = True
    today_bought_shares: int = 0
    holding_days: int = 0
    industry: str = ""
    name: str = ""


@dataclass(slots=True)
class ActionScore:
    """Utility and trade details for one candidate action."""

    requested_action: str
    effective_action: str
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

    @property
    def action(self) -> str:
        """Backward-compatible display name for existing callers."""
        return self.effective_action


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
    explicit = "is_one_price_up" if up else "is_one_price_down"
    if explicit in row and not pd.isna(row.get(explicit)):
        return bool(row.get(explicit))
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


def _legacy_score_actions(
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
    score_expected_return = float(prediction.expected_open_to_open_return)

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
                requested_action=action,
                effective_action=action_effective,
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


def score_policy_target(
    holding: Holding,
    latest_row: pd.Series,
    prediction: PredictionPack,
    account: AccountState,
    target_weight: float,
    predicted_utility: float,
) -> ActionScore:
    """Convert an absolute ML policy weight into one executable A-share order."""
    account = account.normalized()
    price = max(float(holding.current_price), 0.01)
    current_weight = max(float(holding.position_value), 0.0) / max(account.total_asset, 1.0)
    requested_weight = float(np.clip(target_weight, 0.0, account.max_single_position_weight))
    desired_shares = _round_lot(requested_weight * account.total_asset / price, account.lot_size)
    suspended = bool(latest_row.get("is_suspended", False))
    no_volume = float(latest_row.get("volume", 0) or 0) <= 0
    cannot_buy = suspended or no_volume or _is_one_price_limit(latest_row, up=True)
    cannot_sell = suspended or no_volume or _is_one_price_limit(latest_row, up=False)
    reason = (
        f"因子仓位政策选择 {requested_weight * 100:.1f}%：比较候选仓位的预测净效用后取最高值。"
    )
    feasible = True
    if desired_shares > holding.shares:
        requested_action = "POLICY_BUY_ADD"
        effective_action = requested_action
        buy_capacity = min(
            max(account.cash - account.minimum_commission, 0.0),
            max(account.max_single_position_weight * account.total_asset - holding.position_value, 0.0),
        )
        desired_buy = desired_shares - holding.shares
        trade_shares = min(desired_buy, _round_lot(buy_capacity / price, account.lot_size))
        if cannot_buy and trade_shares > 0:
            feasible = False
            effective_action = "NO_TRADE_LIMIT_OR_SUSPENSION"
            trade_shares = 0
            reason += " 当前停牌、无量或一字涨停，买入不可执行。"
    elif desired_shares < holding.shares:
        requested_action = "POLICY_SELL_CLEAR" if desired_shares == 0 else "POLICY_REDUCE"
        effective_action = requested_action
        if not holding.available_shares_known:
            feasible = False
            effective_action = "NO_TRADE_AVAILABLE_UNKNOWN"
            trade_shares = 0
            reason += " 可卖股数未知，按 T+1 规则不假设能够卖出。"
        else:
            desired_sell = holding.shares - desired_shares
            if desired_shares == 0:
                executable_sell = min(desired_sell, max(holding.available_shares, 0))
            else:
                executable_sell = _round_lot(
                    min(desired_sell, max(holding.available_shares, 0)), account.lot_size
                )
            trade_shares = -executable_sell
            if executable_sell < desired_sell:
                effective_action = "POLICY_SELL_AVAILABLE"
                reason += " 受 T+1 可卖股数限制，仅卖出当前可用股份。"
            if cannot_sell and trade_shares < 0:
                feasible = False
                effective_action = "NO_TRADE_LIMIT_OR_SUSPENSION"
                trade_shares = 0
                reason += " 当前停牌、无量或一字跌停，卖出不可执行。"
    else:
        requested_action = "POLICY_HOLD"
        effective_action = requested_action
        trade_shares = 0

    if trade_shares == 0 and requested_action != "POLICY_HOLD" and feasible:
        effective_action = "POLICY_HOLD_LOT_LIMIT"
        reason += " 目标变化不足一个可执行交易单位。"
    target_shares = max(holding.shares + trade_shares, 0)
    executable_target_weight = target_shares * price / max(account.total_asset, 1.0)
    trade_value = abs(trade_shares) * price
    transaction_cost = _trade_cost(trade_value, trade_shares < 0, account) if trade_shares else 0.0
    expected_net_pnl = target_shares * price * float(prediction.expected_open_to_open_return) - transaction_cost
    downside_risk = executable_target_weight * abs(min(float(prediction.return_q10), 0.0))
    return ActionScore(
        requested_action=requested_action,
        effective_action=effective_action,
        trade_shares=trade_shares,
        target_shares=target_shares,
        target_weight=executable_target_weight,
        expected_net_return=expected_net_pnl / max(account.total_asset, 1.0),
        expected_net_pnl=expected_net_pnl,
        downside_risk=downside_risk,
        transaction_cost=transaction_cost,
        turnover=trade_value / max(account.total_asset, 1.0),
        utility_score=float(predicted_utility),
        feasible=feasible,
        reason=reason,
    )


def score_actions(
    holding: Holding,
    latest_row: pd.Series,
    prediction: PredictionPack,
    account: AccountState,
    config: DecisionConfig,
) -> list[ActionScore]:
    """Score requested actions while preserving their effective execution state."""
    account = account.normalized()
    current_value = max(float(holding.position_value), 0.0)
    suspended = bool(latest_row.get("is_suspended", False))
    no_volume = float(latest_row.get("volume", 0) or 0) <= 0
    cannot_buy = suspended or no_volume or _is_one_price_limit(latest_row, up=True)
    cannot_sell = suspended or no_volume or _is_one_price_limit(latest_row, up=False)
    edge_score = _ml_edge_score(prediction)
    scores: list[ActionScore] = []

    for requested_action, multiplier in ACTION_MULTIPLIERS.items():
        feasible = True
        reason = ""
        effective_action = requested_action
        if multiplier < 0:
            if not holding.available_shares_known:
                trade_shares = 0
                feasible = False
                effective_action = "NO_TRADE_AVAILABLE_UNKNOWN"
                reason = "Available shares are unknown; T+1 sell sizing is conservatively disabled."
                sell_shares = 0
            elif requested_action == "SELL_ALL":
                sell_shares = min(max(holding.available_shares, 0), max(holding.shares, 0))
            else:
                desired = min(max(holding.available_shares, 0), int(abs(multiplier) * max(holding.shares, 0)))
                sell_shares = _round_lot(desired, account.lot_size)
            if holding.available_shares_known:
                trade_shares = -sell_shares
            if requested_action == "SELL_ALL" and 0 < sell_shares < holding.shares:
                effective_action = "SELL_AVAILABLE"
                reason = "Available shares are below total shares; sell all currently available shares."
            if cannot_sell and trade_shares < 0:
                feasible = False
                reason = "Selling is unavailable due to suspension, zero volume, or a one-price down limit."
        elif multiplier > 0:
            desired_value = (
                account.max_single_position_weight * account.total_asset * multiplier
                if current_value <= 0
                else current_value * multiplier
            )
            single_capacity = max(account.max_single_position_weight * account.total_asset - current_value, 0.0)
            buy_value = min(desired_value, account.cash, single_capacity)
            trade_shares = _round_lot(buy_value / max(float(holding.current_price), 0.01), account.lot_size)
            if cannot_buy and trade_shares > 0:
                feasible = False
                reason = "Buying is unavailable due to suspension, zero volume, or a one-price up limit."
        else:
            trade_shares = 0

        trade_value = abs(trade_shares) * holding.current_price
        is_exact_liquidation = requested_action == "SELL_ALL" and trade_shares < 0
        if (
            requested_action != "HOLD"
            and trade_shares == 0
            and effective_action == requested_action
        ):
            feasible = False
            effective_action = "NO_TRADE"
            reason = reason or "Requested action produced no executable shares."
        elif trade_value < account.minimum_trade_amount and trade_shares != 0 and not is_exact_liquidation:
            feasible = False
            effective_action = "NO_TRADE"
            trade_shares = 0
            trade_value = 0.0
            reason = "Trade value is below the configured minimum."

        if requested_action.startswith("ADD"):
            add_blocked = (
                edge_score <= 0.10
                or prediction.probability_up < config.add_probability_threshold
                or prediction.expected_open_to_open_return <= _trade_cost(trade_value, False, account) / max(trade_value, 1.0)
                or prediction.return_q10 <= config.add_q10_threshold
                or prediction.probability_down_2pct > config.max_add_downside_probability
            )
            if add_blocked:
                feasible = False
                effective_action = "NO_TRADE"
                trade_shares = 0
                trade_value = 0.0
                reason = "ML add-position thresholds were not met."

        if not feasible and effective_action == requested_action:
            effective_action = "NO_TRADE"
            trade_shares = 0
            trade_value = 0.0
        target_shares = max(holding.shares + trade_shares, 0)
        target_value = target_shares * holding.current_price
        target_weight = target_value / account.total_asset if account.total_asset > 0 else 0.0
        if target_weight > account.max_single_position_weight + 1e-9:
            feasible = False
            effective_action = "NO_TRADE"
            trade_shares = 0
            target_shares = holding.shares
            target_value = current_value
            target_weight = target_value / account.total_asset if account.total_asset > 0 else 0.0
            trade_value = 0.0
            reason = "Target exceeds the single-stock position limit."

        transaction_cost = _trade_cost(trade_value, trade_shares < 0, account) if trade_shares else 0.0
        exposure = target_value / account.total_asset if account.total_asset > 0 else 0.0
        predicted_return = float(prediction.expected_open_to_open_return)
        expected_net_pnl = target_value * predicted_return - transaction_cost
        expected_net_return = expected_net_pnl / max(account.total_asset, 1.0)
        downside_risk = exposure * abs(min(float(prediction.return_q10), 0.0))
        turnover = trade_value / account.total_asset if account.total_asset > 0 else 0.0
        if config.utility_mode == "downside_probability":
            utility = expected_net_return - account.risk_aversion * prediction.probability_down_2pct * exposure
        else:
            utility = expected_net_return - account.risk_aversion * downside_risk
        utility -= account.turnover_penalty * turnover + transaction_cost / max(account.total_asset, 1.0)
        if not feasible:
            utility = -np.inf
        scores.append(
            ActionScore(
                requested_action=requested_action,
                effective_action=effective_action,
                trade_shares=trade_shares,
                target_shares=target_shares,
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

    deduplicated: list[ActionScore] = []
    seen: set[tuple[str, int, int]] = set()
    for score in scores:
        key = (score.effective_action, score.trade_shares, score.target_shares)
        if key not in seen:
            seen.add(key)
            deduplicated.append(score)
    return deduplicated


def apply_account_constraints(table: pd.DataFrame, account: AccountState) -> pd.DataFrame:
    """Apply shared cash, total, single-stock, and industry limits to recommendations."""
    if table.empty:
        return table
    account = account.normalized()
    result = table.copy()
    numeric_defaults = {
        "shares": 0,
        "current_price": 0.0,
        "recommended_trade_shares": 0,
        "utility_score": 0.0,
        "expected_open_to_open_return": 0.0,
    }
    for column, default in numeric_defaults.items():
        result[column] = pd.to_numeric(result.get(column, default), errors="coerce").fillna(default)
    if "industry" not in result.columns:
        result["industry"] = ""
    result["industry"] = result["industry"].fillna("").astype(str)

    current_values = result["shares"] * result["current_price"]
    target_values = current_values.copy()
    cash = float(account.cash)
    total_cap = account.max_total_position_weight * account.total_asset
    single_cap = account.max_single_position_weight * account.total_asset
    industry_cap = account.max_industry_weight * account.total_asset

    # Execute reductions first so their proceeds are available to later buys.
    for index in result.index[result["recommended_trade_shares"] < 0]:
        shares = int(result.at[index, "recommended_trade_shares"])
        price = float(result.at[index, "current_price"])
        value = abs(shares) * price
        cash += max(value - _trade_cost(value, True, account), 0.0)
        target_values.at[index] = max((int(result.at[index, "shares"]) + shares) * price, 0.0)

    buy_indices = list(result.index[result["recommended_trade_shares"] > 0])
    buy_indices.sort(key=lambda idx: float(result.at[idx, "utility_score"]), reverse=True)
    for index in buy_indices:
        price = max(float(result.at[index, "current_price"]), 0.01)
        requested_shares = int(result.at[index, "recommended_trade_shares"])
        current_value = float(current_values.at[index])
        current_total = float(target_values.sum())
        industry = str(result.at[index, "industry"])
        industry_value = float(target_values[result["industry"].eq(industry)].sum()) if industry else 0.0
        industry_capacity = max(industry_cap - industry_value, 0.0) if industry else float("inf")
        capacity_value = min(
            max(cash - account.minimum_commission, 0.0),
            max(total_cap - current_total, 0.0),
            max(single_cap - current_value, 0.0),
            industry_capacity,
            requested_shares * price,
        )
        allowed_shares = min(requested_shares, _round_lot(capacity_value / price, account.lot_size))
        trade_value = allowed_shares * price
        cost = _trade_cost(trade_value, False, account) if allowed_shares else 0.0
        while allowed_shares > 0 and trade_value + cost > cash:
            allowed_shares -= account.lot_size
            trade_value = max(allowed_shares, 0) * price
            cost = _trade_cost(trade_value, False, account) if allowed_shares else 0.0
        allowed_shares = max(allowed_shares, 0)
        result.at[index, "recommended_trade_shares"] = allowed_shares
        result.at[index, "effective_action"] = (
            "NO_TRADE" if allowed_shares == 0 else "ADD_CONSTRAINED" if allowed_shares < requested_shares else result.at[index, "effective_action"]
        )
        result.at[index, "recommended_action"] = result.at[index, "effective_action"]
        target_values.at[index] = current_value + allowed_shares * price
        cash -= allowed_shares * price + cost

    result["recommended_target_shares"] = (
        result["shares"] + result["recommended_trade_shares"]
    ).clip(lower=0).astype(int)
    result["recommended_target_weight"] = target_values / max(account.total_asset, 1.0)
    result["expected_net_pnl"] = (
        target_values * result["expected_open_to_open_return"]
        - result["recommended_trade_shares"].abs()
        * result["current_price"]
        * (account.commission_rate + account.slippage_rate)
    )
    result.attrs["remaining_cash"] = cash
    result.attrs["target_position_value"] = float(target_values.sum())
    return result
