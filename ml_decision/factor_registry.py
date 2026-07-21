"""Central, non-destructive registry for existing model factors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import re


GROUP_ORDER = (
    "technical",
    "liquidity",
    "fund_flow",
    "institution",
    "news",
    "lhb",
    "lhb_institution",
    "fundamental",
    "market",
    "industry",
    "other_existing",
)

FORBIDDEN_FEATURE_KEYWORDS = (
    "target",
    "label",
    "future",
    "next_return",
    "上榜后",
    "未来",
    "后续收益",
)

NEWS_FACTORS = {
    "has_news",
    "news_sentiment",
    "news_sentiment_mean_3",
    "news_sentiment_mean_5",
    "news_sentiment_change_3",
    "news_count_3",
    "news_count_5",
    "weighted_news_sentiment",
}
FUND_FLOW_FACTORS = {
    "large_net_ratio",
    "large_net_ratio_3",
    "large_net_ratio_5",
    "main_net_ratio",
    "main_net_ratio_3",
    "main_net_ratio_5",
    "main_net_ratio_10",
    "main_net_amount",
    "large_net_amount",
    "positive_main_flow_days_5",
    "main_flow_acceleration",
    "flow_price_divergence_5",
}
INSTITUTION_FACTORS = {
    "institution_activity",
    "institution_activity_ma_5",
    "institution_net_buy_amount",
    "institution_hold_count",
    "institution_hold_ratio",
    "institution_hold_ratio_change",
}

_TECHNICAL_PREFIXES = (
    "ret_", "volatility_", "downside_volatility_", "vol_ratio_", "max_drawdown_",
    "current_drawdown_", "breakout_gap_", "support_gap_", "ma", "bias_", "rsi_",
    "atr_", "body_", "upper_shadow_", "lower_shadow_", "close_location", "open_location",
    "gap_open", "intraday_return",
)
_LIQUIDITY_PREFIXES = ("volume_ratio_", "amount_ratio_", "turnover", "liquidity_")
_FUNDAMENTAL_PREFIXES = ("pe_", "pb_", "ps_", "roe", "roa", "eps", "revenue_", "profit_", "fundamental_")
_MARKET_PREFIXES = ("index_", "market_", "market_rank_")
_INDUSTRY_PREFIXES = ("industry_", "relative_industry_", "industry_rank_")


def is_forbidden_feature(name: str) -> bool:
    """Return whether a field is a target/future field and cannot be a feature."""
    lowered = str(name).lower()
    return any(keyword.lower() in lowered for keyword in FORBIDDEN_FEATURE_KEYWORDS)


def validate_feature_names(columns: Iterable[str]) -> None:
    forbidden = [str(column) for column in columns if is_forbidden_feature(str(column))]
    if forbidden:
        raise ValueError(f"Forbidden target/future fields in feature registry: {forbidden}")


def classify_factor(name: str) -> str:
    """Classify one existing factor without changing its name or value."""
    if is_forbidden_feature(name):
        raise ValueError(f"Forbidden target/future field cannot be registered: {name}")
    if name.startswith("lhb_inst_"):
        return "lhb_institution"
    if name.startswith("lhb_"):
        return "lhb"
    if name in NEWS_FACTORS or name.startswith("news_"):
        return "news"
    if name in FUND_FLOW_FACTORS or name.startswith(("main_net_", "large_net_", "flow_")):
        return "fund_flow"
    if name in INSTITUTION_FACTORS or name.startswith("institution_"):
        return "institution"
    if name.startswith(_INDUSTRY_PREFIXES):
        return "industry"
    if name.startswith(_MARKET_PREFIXES):
        return "market"
    if name.startswith(_LIQUIDITY_PREFIXES):
        return "liquidity"
    if name.startswith(_FUNDAMENTAL_PREFIXES):
        return "fundamental"
    if name.startswith(_TECHNICAL_PREFIXES):
        return "technical"
    return "other_existing"


def build_factor_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Map every allowed existing factor exactly once; unmatched fields are retained."""
    ordered = list(dict.fromkeys(str(column) for column in columns))
    validate_feature_names(ordered)
    groups = {group: [] for group in GROUP_ORDER}
    for column in ordered:
        groups[classify_factor(column)].append(column)
    assert sum(len(values) for values in groups.values()) == len(ordered)
    assert len({item for values in groups.values() for item in values}) == len(ordered)
    return groups


def flatten_factor_groups(groups: Mapping[str, Iterable[str]]) -> list[str]:
    return [str(column) for group in GROUP_ORDER for column in groups.get(group, [])]


def factor_group_counts(groups: Mapping[str, Iterable[str]]) -> dict[str, int]:
    return {group: len(list(groups.get(group, []))) for group in GROUP_ORDER}


def source_requirements(columns: Iterable[str]) -> dict[str, bool]:
    """Return the data sources actually required by a frozen feature schema."""
    ordered = [str(column) for column in columns]
    groups = {classify_factor(column) for column in ordered}
    return {
        "market_data_available": True,
        "cross_section_rank_available": any(
            column.startswith(("market_rank_", "industry_rank_")) for column in ordered
        ),
        "fund_flow_data_available": "fund_flow" in groups,
        "news_data_available": "news" in groups,
        "institution_data_available": "institution" in groups,
        "lhb_data_available": "lhb" in groups,
        "lhb_inst_data_available": "lhb_institution" in groups,
    }


def snapshot_factor_frame(frame, factor_columns: Iterable[str]):
    """Deep snapshot used by protection tests; never mutates the source frame."""
    return deepcopy(frame.loc[:, list(factor_columns)])


def safe_version_name(value: str) -> str:
    """Sanitize a user supplied version while keeping model directories unique."""
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value)).strip("-.")
    if not cleaned:
        raise ValueError("Model version must contain at least one safe character")
    return cleaned
