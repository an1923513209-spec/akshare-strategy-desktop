"""End-to-end next-session holding decision engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd

from .actions import Holding, choose_action, score_actions
from .config import AccountState, DecisionConfig
from .data_sources import SourceNote
from .features import add_labels, build_features, feature_columns, normalize_market_df
from .models import NextSessionModel, PredictionPack


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DecisionResult:
    """Engine output: table rows, diagnostics, and source status."""

    table: pd.DataFrame
    metrics: dict[str, float]
    feature_columns: list[str]
    source_notes: list[SourceNote]


def prepare_holdings(holdings_df: pd.DataFrame, latest_market: pd.DataFrame, account: AccountState) -> pd.DataFrame:
    """Normalize holding rows and fill current price/value/weight when missing."""
    data = holdings_df.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    latest = latest_market.sort_values("date").groupby("code").tail(1)[["code", "close"]]
    data = data.merge(latest.rename(columns={"close": "latest_close"}), on="code", how="left")
    data["shares"] = pd.to_numeric(data.get("shares", 0), errors="coerce").fillna(0).astype(int)
    data["available_shares"] = pd.to_numeric(data.get("available_shares", data["shares"]), errors="coerce").fillna(data["shares"]).astype(int)
    data["average_cost"] = pd.to_numeric(data.get("average_cost", np.nan), errors="coerce")
    data["current_price"] = pd.to_numeric(data.get("current_price", data["latest_close"]), errors="coerce").fillna(data["latest_close"])
    data["position_value"] = pd.to_numeric(data.get("position_value", np.nan), errors="coerce")
    data["position_value"] = data["position_value"].fillna(data["shares"] * data["current_price"])
    data["position_weight"] = pd.to_numeric(data.get("position_weight", np.nan), errors="coerce")
    data["position_weight"] = data["position_weight"].fillna(data["position_value"] / max(account.total_asset, 1.0))
    if "holding_days" not in data.columns:
        data["holding_days"] = 0
    data["holding_days"] = pd.to_numeric(data["holding_days"], errors="coerce").fillna(0).astype(int)
    if "industry" not in data.columns:
        data["industry"] = ""
    if "name" not in data.columns:
        data["name"] = ""
    return data


def run_holding_decision(
    market_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    account: AccountState | None = None,
    config: DecisionConfig | None = None,
    source_notes: list[SourceNote] | None = None,
) -> DecisionResult:
    """Train models on a market long table and score current holdings.

    The model is trained on historical rows with future labels. Current holdings
    are used only at inference and action-scoring time.
    """
    account = (account or AccountState()).normalized()
    config = config or DecisionConfig()
    source_notes = source_notes or []
    market = normalize_market_df(market_df)
    round_trip_cost = account.commission_rate * 2 + account.stamp_duty_rate + account.slippage_rate * 2 + config.profitable_cost_buffer
    features = build_features(market, external_factor_lag=config.external_factor_lag)
    dataset = add_labels(features, round_trip_cost=round_trip_cost, down_threshold=config.down_threshold)
    columns = feature_columns(dataset, exclude=config.feature_exclude)
    valid = dataset.dropna(subset=["next_open_to_next_open_return"])
    if len(valid) < config.train_min_rows:
        raise ValueError(f"有效训练样本不足: {len(valid)} < {config.train_min_rows}")

    model = NextSessionModel(calibration_method=config.calibration_method).fit(valid, columns)
    latest_market = features.sort_values("date").groupby("code").tail(1)
    holdings = prepare_holdings(holdings_df, latest_market, account)
    rows: list[dict[str, Any]] = []
    for holding_row in holdings.to_dict("records"):
        code = str(holding_row["code"]).zfill(6)
        latest = latest_market[latest_market["code"] == code]
        if latest.empty:
            LOGGER.warning("No latest market row for %s", code)
            continue
        latest_row = latest.iloc[-1]
        try:
            prediction = model.predict_one(latest_row)
            holding = Holding(
                code=code,
                shares=int(holding_row.get("shares") or 0),
                available_shares=int(holding_row.get("available_shares") or 0),
                average_cost=float(holding_row.get("average_cost") or 0),
                current_price=float(holding_row.get("current_price") or latest_row["close"]),
                position_value=float(holding_row.get("position_value") or 0),
                position_weight=float(holding_row.get("position_weight") or 0),
                holding_days=int(holding_row.get("holding_days") or 0),
                industry=str(holding_row.get("industry") or latest_row.get("industry", "")),
                name=str(holding_row.get("name") or ""),
            )
            scores = score_actions(holding, latest_row, prediction, account, config)
            selected = choose_action(scores, config.minimum_action_edge)
            score_map = {score.action: score for score in scores}
            rows.append(_output_row(holding, latest_row, prediction, selected, score_map))
        except Exception as exc:
            LOGGER.warning("Skip ML decision for %s: %s", code, exc)
            continue

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["recommended_action", "utility_score"], ascending=[True, False]).reset_index(drop=True)
    return DecisionResult(table=table, metrics=model.metrics, feature_columns=columns, source_notes=source_notes)


def _output_row(
    holding: Holding,
    latest_row: pd.Series,
    prediction: PredictionPack,
    selected: Any,
    scores: dict[str, Any],
) -> dict[str, Any]:
    unrealized = holding.current_price / holding.average_cost - 1 if holding.average_cost > 0 else np.nan
    action_label = _action_cn(selected.action, holding.shares)
    holding_risk = _holding_risk_level(holding, prediction)
    reason = (
        f"建议{action_label}。开盘至再下一开盘预期收益 {prediction.expected_open_to_open_return * 100:.2f}%，"
        f"上涨概率 {prediction.probability_up * 100:.1f}%，超成本概率 {prediction.probability_profitable * 100:.1f}%，"
        f"下跌2%概率 {prediction.probability_down_2pct * 100:.1f}%，q10 {prediction.return_q10 * 100:.2f}%。"
        f"验证权重：概率 {prediction.score_weights.get('probability', 0):.2f}，"
        f"预期 {prediction.score_weights.get('expected', 0):.2f}，风险 {prediction.score_weights.get('risk', 0):.2f}。"
    )
    return {
        "date": pd.to_datetime(latest_row["date"]).strftime("%Y-%m-%d"),
        "code": holding.code,
        "name": holding.name,
        "shares": holding.shares,
        "available_shares": holding.available_shares,
        "average_cost": holding.average_cost,
        "current_price": holding.current_price,
        "position_weight": holding.position_weight,
        "unrealized_return": unrealized,
        "expected_gap_return": prediction.expected_gap_return,
        "expected_open_to_open_return": prediction.expected_open_to_open_return,
        "probability_up": prediction.probability_up,
        "probability_profitable": prediction.probability_profitable,
        "probability_down_2pct": prediction.probability_down_2pct,
        "return_q10": prediction.return_q10,
        "return_q50": prediction.return_q50,
        "return_q90": prediction.return_q90,
        "hold_score": _score(scores, "HOLD"),
        "sell_all_score": _score(scores, "SELL_ALL"),
        "reduce_50_score": _score(scores, "REDUCE_50"),
        "reduce_25_score": _score(scores, "REDUCE_25"),
        "add_25_score": _score(scores, "ADD_25"),
        "add_50_score": _score(scores, "ADD_50"),
        "recommended_action": selected.action,
        "recommended_trade_shares": selected.trade_shares,
        "recommended_target_weight": selected.target_weight,
        "expected_net_pnl": selected.expected_net_pnl,
        "downside_risk": selected.downside_risk,
        "utility_score": selected.utility_score,
        "display_score": _display_score(prediction),
        "confidence_level": prediction.confidence_level,
        "holding_risk_level": holding_risk,
        "score_weight_probability": prediction.score_weights.get("probability", np.nan),
        "score_weight_expected": prediction.score_weights.get("expected", np.nan),
        "score_weight_risk": prediction.score_weights.get("risk", np.nan),
        "main_net_ratio": _latest_numeric(latest_row, "main_net_ratio"),
        "main_net_ratio_3": _latest_numeric(latest_row, "main_net_ratio_3"),
        "large_net_ratio": _latest_numeric(latest_row, "large_net_ratio"),
        "news_sentiment_mean_3": _latest_numeric(latest_row, "news_sentiment_mean_3"),
        "news_count_3": _latest_numeric(latest_row, "news_count_3"),
        "institution_activity": _latest_numeric(latest_row, "institution_activity"),
        "institution_net_buy_amount": _latest_numeric(latest_row, "institution_net_buy_amount"),
        "top_positive_factors": ", ".join(prediction.top_positive_factors),
        "top_negative_factors": ", ".join(prediction.top_negative_factors),
        "reason": reason,
    }


def _latest_numeric(row: pd.Series, column: str) -> float:
    value = row.get(column, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _score(scores: dict[str, Any], action: str) -> float:
    score = scores.get(action)
    return float(score.utility_score) if score is not None else np.nan


def _action_cn(action: str, shares: int = 0) -> str:
    if action == "HOLD" and shares <= 0:
        return "暂不买入/观察"
    if action in {"SELL_ALL", "REDUCE_50", "REDUCE_25"} and shares <= 0:
        return "暂不买入/观察"
    labels = {
        "SELL_ALL": "清仓",
        "REDUCE_50": "减仓50%",
        "REDUCE_25": "减仓25%",
        "HOLD": "持有",
        "ADD_25": "加仓25%",
        "ADD_50": "加仓50%",
    }
    return labels.get(action, action)


def _display_score(prediction: PredictionPack) -> float:
    weights = prediction.score_weights or {"probability": 0.45, "expected": 0.35, "risk": 0.20}
    prob_component = (float(prediction.probability_up) - 0.5) * 2.0
    expected_component = float(np.tanh(float(prediction.expected_open_to_open_return) / 0.03))
    q10_risk = float(np.clip(-float(prediction.return_q10), 0.0, 0.08) / 0.08)
    risk_component = (float(prediction.probability_down_2pct) + q10_risk) / 2.0
    edge = (
        float(weights.get("probability", 0.45)) * prob_component
        + float(weights.get("expected", 0.35)) * expected_component
        - float(weights.get("risk", 0.20)) * risk_component
    )
    return float(np.clip((edge + 1.0) * 50.0, 0.0, 100.0))


def _holding_risk_level(holding: Holding, prediction: PredictionPack) -> str:
    if holding.shares <= 0:
        return "未持仓"
    q10_risk = max(-float(prediction.return_q10), 0.0)
    down_prob = float(prediction.probability_down_2pct)
    if down_prob >= 0.45 or q10_risk >= 0.06:
        return "高风险"
    if down_prob >= 0.30 or q10_risk >= 0.04:
        return "风险升高"
    if down_prob >= 0.18 or q10_risk >= 0.025:
        return "中等"
    return "低"


def result_to_jsonable(result: DecisionResult) -> dict[str, Any]:
    """Convert a DecisionResult to a JSON/pickle-friendly dictionary."""
    return {
        "table": result.table.to_dict("records"),
        "metrics": result.metrics,
        "feature_columns": result.feature_columns,
        "source_notes": [asdict(note) for note in result.source_notes],
    }
