"""Feature and label construction for next-session A-share decisions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .lhb_data import assert_no_forbidden_features
from .cross_section import RANK_BASE_COLUMNS
from .trading_rules import TRADE_RULE_COLUMNS


REQUIRED_COLUMNS = ("date", "code", "open", "high", "low", "close", "volume", "amount")
OPTIONAL_FACTOR_COLUMNS = (
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
    "institution_activity",
    "institution_activity_ma_5",
    "institution_net_buy_amount",
    "institution_hold_count",
    "institution_hold_ratio",
    "institution_hold_ratio_change",
    "flow_price_divergence_5",
    "has_news",
    "news_sentiment",
    "news_sentiment_mean_3",
    "news_sentiment_mean_5",
    "news_sentiment_change_3",
    "news_count_3",
    "news_count_5",
    "weighted_news_sentiment",
    "index_ret_1",
    "index_ret_5",
    "index_ret_20",
    "index_volatility_20",
    "market_breadth",
    "market_amount_ratio_20",
    "industry_ret_5",
    "industry_ret_20",
    "relative_industry_ret_5",
    "relative_industry_ret_20",
    "industry_rank_ret_5",
    "industry_rank_ret_20",
    "lhb_flag",
    "lhb_reason_count",
    "lhb_net_buy_ratio",
    "lhb_amount_ratio",
    "lhb_buy_sell_balance",
    "lhb_net_buy_float_cap_ratio",
    "lhb_buy_log",
    "lhb_sell_log",
    "lhb_net_buy_signed_log",
    "lhb_reason_price_deviation",
    "lhb_reason_turnover",
    "lhb_reason_amplitude",
    "lhb_reason_abnormal",
    "lhb_reason_three_day",
    "lhb_reason_st",
    "lhb_inst_buy_count",
    "lhb_inst_sell_count",
    "lhb_inst_net_buy_ratio",
    "lhb_inst_buy_sell_balance",
    "lhb_inst_net_buy_signed_log",
    "lhb_inst_buy_flag",
    "lhb_inst_sell_flag",
    "lhb_inst_net_buy_positive",
    "lhb_count_5d",
    "lhb_count_10d",
    "lhb_count_20d",
    "lhb_count_60d",
    "lhb_net_buy_sum_5d",
    "lhb_net_buy_sum_10d",
    "lhb_net_buy_sum_20d",
    "lhb_inst_net_buy_sum_5d",
    "lhb_inst_net_buy_sum_10d",
    "lhb_inst_net_buy_sum_20d",
    "days_since_last_lhb",
    "consecutive_lhb_days",
    "lhb_positive_count_5d",
    "lhb_negative_count_5d",
    "lhb_inst_positive_count_20d",
)
TARGET_COLUMNS = (
    "next_gap_return",
    "next_open_to_close_return",
    "next_open_to_next_open_return",
    "next_close_to_close_return",
    "next_high_excursion",
    "next_low_excursion",
    "label_up",
    "label_profitable",
    "label_down_2pct",
)
MIN_FEATURE_NON_NULL = 30
MIN_OPTIONAL_FACTOR_NON_NULL = 20
MIN_OPTIONAL_FACTOR_COVERAGE = 0.03

LHB_INITIAL_MODEL_FEATURES = {
    "lhb_flag",
    "lhb_reason_count",
    "lhb_net_buy_ratio",
    "lhb_amount_ratio",
    "lhb_buy_sell_balance",
    "lhb_net_buy_float_cap_ratio",
    "lhb_inst_buy_count",
    "lhb_inst_sell_count",
    "lhb_inst_net_buy_ratio",
    "lhb_inst_buy_sell_balance",
    "lhb_count_5d",
    "lhb_count_20d",
    "lhb_net_buy_sum_5d",
    "lhb_inst_net_buy_sum_20d",
    "days_since_last_lhb",
    "consecutive_lhb_days",
}
LHB_NON_MODEL_COLUMNS = {
    "lhb_data_available",
    "lhb_detail_available",
    "lhb_inst_data_available",
    "lhb_record_count",
    "lhb_net_buy",
    "lhb_buy",
    "lhb_sell",
    "lhb_amount",
    "stock_total_amount",
    "float_market_cap",
    "lhb_inst_stock_total_amount",
    "lhb_inst_buy_amount",
    "lhb_inst_sell_amount",
    "lhb_inst_net_buy",
    "pct_change",
    "turnover_rate",
}
DATA_AVAILABILITY_COLUMNS = {
    "market_data_available",
    "fund_flow_data_available",
    "news_data_available",
    "institution_data_available",
    "lhb_data_available",
    "lhb_detail_available",
    "lhb_inst_data_available",
}


def normalize_market_df(market_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common OHLCV schemas to the long table used by the engine."""
    rename_map = {
        "Date": "date",
        "Code": "code",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Amount": "amount",
        "成交额": "amount",
    }
    data = market_df.rename(columns={key: value for key, value in rename_map.items() if key in market_df.columns}).copy()
    if "amount" not in data.columns and "volume" in data.columns and "close" in data.columns:
        data["amount"] = pd.to_numeric(data["volume"], errors="coerce") * pd.to_numeric(data["close"], errors="coerce")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"market_df 缺少字段: {missing}")
    data["date"] = pd.to_datetime(data["date"])
    data["code"] = data["code"].astype(str).str.zfill(6)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values(["code", "date"]).reset_index(drop=True)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(group: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = group["high"] - group["low"]
    high_close = (group["high"] - group["close"].shift()).abs()
    low_close = (group["low"] - group["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def _downside_volatility(ret: pd.Series, window: int) -> pd.Series:
    return ret.where(ret < 0, 0.0).rolling(window).std()


def build_features(
    market_df: pd.DataFrame,
    external_factor_lag: int = 1,
    include_optional: Iterable[str] = OPTIONAL_FACTOR_COLUMNS,
    cross_sectional_rank_frame: pd.DataFrame | None = None,
    allow_local_cross_sectional_ranks: bool = False,
) -> pd.DataFrame:
    """Build time-safe factors without inventing ranks from a desktop subset."""
    data = normalize_market_df(market_df)
    groups = []
    optional_cols = [column for column in include_optional if column in data.columns]
    for _code, group in data.groupby("code", sort=False):
        g = group.copy()
        open_ = g["open"]
        high = g["high"]
        low = g["low"]
        close = g["close"]
        volume = g["volume"]
        amount = g["amount"]
        day_range = (high - low).replace(0, np.nan)
        prev_close = close.shift(1)
        ret = close.pct_change()

        g["body_ratio"] = (close - open_) / open_
        g["body_abs_ratio"] = (close - open_).abs() / open_
        g["upper_shadow_ratio"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / day_range
        g["lower_shadow_ratio"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / day_range
        g["close_location"] = (close - low) / day_range
        g["open_location"] = (open_ - low) / day_range
        g["gap_open"] = open_ / prev_close - 1
        g["intraday_return"] = close / open_ - 1

        for window in (1, 2, 3, 5, 10, 20):
            g[f"ret_{window}"] = close.pct_change(window)
        for window in (5, 10, 20):
            g[f"volatility_{window}"] = ret.rolling(window).std()
        g["downside_volatility_20"] = _downside_volatility(ret, 20)
        g["vol_ratio_5_20"] = g["volatility_5"] / g["volatility_20"]

        rolling_max_10 = close.rolling(10).max()
        rolling_max_20 = close.rolling(20).max()
        g["max_drawdown_10"] = close / rolling_max_10 - 1
        g["max_drawdown_20"] = close / rolling_max_20 - 1
        g["current_drawdown_10"] = close / rolling_max_10 - 1
        g["current_drawdown_20"] = close / rolling_max_20 - 1

        g["breakout_gap_10"] = close / high.shift(1).rolling(10).max() - 1
        g["breakout_gap_20"] = close / high.shift(1).rolling(20).max() - 1
        g["support_gap_10"] = close / low.shift(1).rolling(10).min() - 1
        g["support_gap_20"] = close / low.shift(1).rolling(20).min() - 1

        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        g["ma5_ma10"] = ma5 / ma10 - 1
        g["ma5_ma20"] = ma5 / ma20 - 1
        g["ma10_ma20"] = ma10 / ma20 - 1
        g["ma20_ma60"] = ma20 / ma60 - 1
        g["ma5_slope"] = ma5 / ma5.shift(3) - 1
        g["ma10_slope"] = ma10 / ma10.shift(3) - 1
        g["ma20_slope"] = ma20 / ma20.shift(5) - 1
        g["bias_5"] = close / ma5 - 1
        g["bias_10"] = close / ma10 - 1
        g["bias_20"] = close / ma20 - 1
        g["bias_60"] = close / ma60 - 1

        g["rsi_6"] = _rsi(close, 6)
        g["rsi_14"] = _rsi(close, 14)
        g["rsi_6_14_diff"] = g["rsi_6"] - g["rsi_14"]
        atr = _atr(g, 14)
        g["atr_pct"] = atr / close
        g["atr_pct_14"] = g["atr_pct"]

        for window in (3, 5, 10, 20):
            g[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean()
        g["amount_ratio_20"] = amount / amount.rolling(20).mean()

        for column in optional_cols:
            g[column] = pd.to_numeric(g[column], errors="coerce").shift(external_factor_lag)

        # LHB data at date t is published after the close and may only predict
        # t+1. The caller uses external_factor_lag=0 for an after-close model.
        # Consolidate the frame after adding the optional source columns so
        # sparse LHB interactions do not emit one fragmentation warning per stock.
        g = g.copy()
        if "lhb_net_buy_ratio" in g.columns:
            g["lhb_net_buy_momentum_interaction"] = g["lhb_net_buy_ratio"] * g["ret_5"]
            g["lhb_volume_interaction"] = g["lhb_net_buy_ratio"] * g["volume_ratio_5"]
        if "lhb_inst_net_buy_ratio" in g.columns:
            g["lhb_inst_breakout_interaction"] = g["lhb_inst_net_buy_ratio"] * g["breakout_gap_20"]
            g["lhb_inst_rsi_interaction"] = g["lhb_inst_net_buy_ratio"] * g["rsi_14"]
        if "lhb_count_20d" in g.columns:
            g["lhb_count_momentum_interaction"] = g["lhb_count_20d"] * g["ret_20"]
        groups.append(g)

    data = pd.concat(groups, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True).copy()
    if cross_sectional_rank_frame is not None and not cross_sectional_rank_frame.empty:
        ranks = cross_sectional_rank_frame.copy()
        ranks["date"] = pd.to_datetime(ranks["date"], errors="coerce").dt.normalize()
        ranks["code"] = ranks["code"].astype(str).str.zfill(6)
        rank_columns = [
            column for column in ranks.columns
            if column.startswith(("market_rank_", "industry_rank_"))
        ]
        data = data.drop(columns=[column for column in rank_columns if column in data.columns], errors="ignore")
        data = data.merge(
            ranks[["date", "code", *rank_columns]],
            on=["date", "code"],
            how="left",
            validate="many_to_one",
        )
    elif allow_local_cross_sectional_ranks:
        for column in RANK_BASE_COLUMNS:
            if column not in data.columns:
                continue
            data[f"market_rank_{column}"] = data.groupby("date")[column].rank(pct=True)
            if "industry" in data.columns:
                data[f"industry_rank_{column}"] = data.groupby(["date", "industry"])[column].rank(pct=True)
    return data.replace([np.inf, -np.inf], np.nan)


def add_labels(feature_df: pd.DataFrame, round_trip_cost: float, down_threshold: float) -> pd.DataFrame:
    """Add next-session targets. Future shift is used only for labels."""
    groups = []
    for _code, group in feature_df.groupby("code", sort=False):
        g = group.sort_values("date").copy()
        next_open = g["open"].shift(-1)
        next_close = g["close"].shift(-1)
        next_high = g["high"].shift(-1)
        next_low = g["low"].shift(-1)
        next_next_open = g["open"].shift(-2)
        g["next_gap_return"] = next_open / g["close"] - 1
        g["next_open_to_close_return"] = next_close / next_open - 1
        g["next_open_to_next_open_return"] = next_next_open / next_open - 1
        g["next_close_to_close_return"] = next_close / g["close"] - 1
        g["next_high_excursion"] = next_high / next_open - 1
        g["next_low_excursion"] = next_low / next_open - 1
        g["label_up"] = (g["next_open_to_next_open_return"] > 0).astype(float)
        g["label_profitable"] = (g["next_open_to_next_open_return"] > round_trip_cost).astype(float)
        g["label_down_2pct"] = (g["next_open_to_next_open_return"] < down_threshold).astype(float)
        groups.append(g)
    return pd.concat(groups, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def feature_columns(
    dataset: pd.DataFrame,
    exclude: Iterable[str] = (),
    selection_df: pd.DataFrame | None = None,
) -> list[str]:
    """Return usable numeric model feature columns.

    Technical OHLCV factors are kept when they have enough historical samples.
    Optional external factors, such as fund flow, news sentiment and institution
    activity, are included only when the labelled training rows contain enough
    non-null values. A latest-only external snapshot is still available for
    display, but it is not used for model training because there is no history
    from which the model can learn its effect.
    """
    blocked = {
        "date",
        "code",
        "name",
        "industry",
        "board",
        *REQUIRED_COLUMNS,
        *TARGET_COLUMNS,
        *exclude,
        *DATA_AVAILABILITY_COLUMNS,
        *LHB_NON_MODEL_COLUMNS,
        *TRADE_RULE_COLUMNS,
    }
    columns = []
    selection = dataset if selection_df is None else selection_df
    if "next_open_to_next_open_return" in selection.columns:
        train_mask = selection["next_open_to_next_open_return"].notna()
    else:
        train_mask = pd.Series(True, index=selection.index)
    train_rows = max(int(train_mask.sum()), 1)
    optional_set = set(OPTIONAL_FACTOR_COLUMNS)
    for column in dataset.columns:
        if column in blocked:
            continue
        if column.startswith("lhb_") and column not in LHB_INITIAL_MODEL_FEATURES:
            # Compute and retain the wider LHB library for research/output, but
            # initially train only on the explicitly approved sparse features.
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            if column not in selection.columns:
                continue
            train_series = pd.to_numeric(selection.loc[train_mask, column], errors="coerce")
            non_null = int(train_series.notna().sum())
            if non_null < MIN_FEATURE_NON_NULL:
                continue
            if column in optional_set:
                coverage = non_null / train_rows
                if non_null < MIN_OPTIONAL_FACTOR_NON_NULL or coverage < MIN_OPTIONAL_FACTOR_COVERAGE:
                    continue
            columns.append(column)
    assert_no_forbidden_features(columns)
    return columns
