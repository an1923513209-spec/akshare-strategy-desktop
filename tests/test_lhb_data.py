from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_decision.lhb_data import (
    _call_with_retry,
    add_next_day_target,
    aggregate_lhb_detail,
    aggregate_lhb_institution,
    assert_no_forbidden_features,
    build_lhb_factor_frame,
)
from ml_decision.features import add_labels, build_features, feature_columns


def detail_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "000001",
                "trade_date": pd.Timestamp("2026-07-13"),
                "name": "平安银行",
                "close": 10.0,
                "pct_change": 2.0,
                "lhb_net_buy": 200.0,
                "lhb_buy": 600.0,
                "lhb_sell": 400.0,
                "lhb_amount": 1000.0,
                "stock_total_amount": 10000.0,
                "turnover_rate": 5.0,
                "float_market_cap": 100000.0,
                "lhb_reason": "日涨幅偏离值达到7%",
                "lhb_interpretation": "机构买入",
            },
            {
                "symbol": "000001",
                "trade_date": pd.Timestamp("2026-07-13"),
                "name": "平安银行",
                "close": 10.0,
                "pct_change": 2.0,
                "lhb_net_buy": 200.0,
                "lhb_buy": 600.0,
                "lhb_sell": 400.0,
                "lhb_amount": 1000.0,
                "stock_total_amount": 10000.0,
                "turnover_rate": 5.0,
                "float_market_cap": 100000.0,
                "lhb_reason": "连续三个交易日内涨幅偏离值累计达到20%",
                "lhb_interpretation": "机构买入",
            },
        ]
    )


def institution_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "000001",
                "trade_date": pd.Timestamp("2026-07-13"),
                "name": "平安银行",
                "lhb_inst_buy_count": 3,
                "lhb_inst_sell_count": 1,
                "lhb_inst_buy_amount": 300.0,
                "lhb_inst_sell_amount": 100.0,
                "lhb_inst_net_buy": 200.0,
                "stock_total_amount": 10000.0,
                "lhb_reason": "日涨幅偏离值达到7%",
            }
        ]
    )


def availability(dates: pd.DatetimeIndex, unavailable: set[pd.Timestamp] | None = None) -> pd.DataFrame:
    unavailable = unavailable or set()
    return pd.DataFrame(
        {
            "trade_date": dates,
            "lhb_data_available": [0 if date in unavailable else 1 for date in dates],
        }
    )


def test_duplicate_reasons_aggregate_to_one_row_without_amount_sum() -> None:
    result = aggregate_lhb_detail(detail_rows())
    assert len(result) == 1
    assert result.loc[0, "lhb_record_count"] == 2
    assert result.loc[0, "lhb_reason_count"] == 2
    assert result.loc[0, "lhb_amount"] == 1000.0
    assert result.loc[0, "lhb_buy"] == 600.0


def test_zero_denominators_never_create_infinity() -> None:
    detail = detail_rows().iloc[:1].copy()
    detail[["lhb_buy", "lhb_sell", "stock_total_amount", "float_market_cap"]] = 0.0
    institution = institution_rows().copy()
    institution[["lhb_inst_buy_amount", "lhb_inst_sell_amount", "stock_total_amount"]] = 0.0
    detail_result = aggregate_lhb_detail(detail)
    institution_result = aggregate_lhb_institution(institution)
    assert not np.isinf(detail_result.select_dtypes("number").to_numpy()).any()
    assert not np.isinf(institution_result.select_dtypes("number").to_numpy()).any()
    assert pd.isna(detail_result.loc[0, "lhb_buy_sell_balance"])
    assert pd.isna(institution_result.loc[0, "lhb_inst_buy_sell_balance"])


def test_rolling_windows_do_not_cross_symbols() -> None:
    dates = pd.bdate_range("2026-07-06", periods=6)
    calendar = pd.MultiIndex.from_product([["000001", "000002"], dates], names=["symbol", "trade_date"]).to_frame(index=False)
    factors = build_lhb_factor_frame(calendar, detail_rows(), institution_rows(), availability(dates))
    first_other = factors[(factors["code"] == "000002") & (factors["date"] == dates[0])].iloc[0]
    assert first_other["lhb_count_5d"] == 0
    assert first_other["lhb_net_buy_sum_5d"] == 0


def test_complete_day_fills_unlisted_stock_with_zero() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2026-07-13")])
    calendar = pd.DataFrame({"symbol": ["000001", "000002"], "trade_date": [dates[0], dates[0]]})
    factors = build_lhb_factor_frame(calendar, detail_rows(), institution_rows(), availability(dates))
    unlisted = factors[factors["code"] == "000002"].iloc[0]
    assert unlisted["lhb_flag"] == 0
    assert unlisted["lhb_net_buy"] == 0
    assert unlisted["lhb_inst_net_buy"] == 0


def test_failed_day_keeps_event_fields_missing() -> None:
    failed = pd.Timestamp("2026-07-14")
    calendar = pd.DataFrame({"symbol": ["000002"], "trade_date": [failed]})
    factors = build_lhb_factor_frame(calendar, detail_rows(), institution_rows(), availability(pd.DatetimeIndex([failed]), {failed}))
    row = factors.iloc[0]
    assert row["lhb_data_available"] == 0
    assert pd.isna(row["lhb_flag"])
    assert pd.isna(row["lhb_inst_net_buy"])


def test_days_since_last_lhb_uses_trading_positions() -> None:
    dates = pd.bdate_range("2026-07-13", periods=4)
    calendar = pd.DataFrame({"symbol": "000001", "trade_date": dates})
    factors = build_lhb_factor_frame(calendar, detail_rows(), institution_rows(), availability(dates))
    assert list(factors["days_since_last_lhb"]) == [0.0, 1.0, 2.0, 3.0]


def test_forbidden_future_columns_cannot_enter_features() -> None:
    with pytest.raises(ValueError, match="已禁止"):
        assert_no_forbidden_features(["ret_5", "上榜后1日"])
    with pytest.raises(ValueError, match="已禁止"):
        assert_no_forbidden_features(["future_return"])


def test_merged_factor_primary_key_is_unique() -> None:
    dates = pd.bdate_range("2026-07-13", periods=3)
    calendar = pd.DataFrame({"symbol": "000001", "trade_date": dates})
    factors = build_lhb_factor_frame(calendar, detail_rows(), institution_rows(), availability(dates))
    assert not factors.duplicated(["code", "date"]).any()


def test_factor_date_t_target_is_t_plus_one() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001"] * 3,
            "trade_date": pd.bdate_range("2026-07-13", periods=3),
            "close": [10.0, 11.0, 9.9],
        }
    )
    result = add_next_day_target(frame)
    assert result.loc[0, "target_return_1d"] == pytest.approx(0.10)
    assert result.loc[1, "target_return_1d"] == pytest.approx(-0.10)
    assert pd.isna(result.loc[2, "target_return_1d"])


def test_institution_balance_uses_total_buy_and_sell_power() -> None:
    result = aggregate_lhb_institution(institution_rows())
    assert result.loc[0, "lhb_inst_count_balance"] == 2
    assert result.loc[0, "lhb_inst_buy_sell_balance"] == pytest.approx(0.5)
    assert result.loc[0, "lhb_inst_net_buy_ratio"] == pytest.approx(0.02)


def test_lhb_institution_names_do_not_overwrite_existing_institution_factors() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2026-07-13")])
    market = pd.DataFrame(
        {"code": ["000001"], "date": dates, "institution_activity": [7.0], "institution_net_buy_amount": [123.0]}
    )
    factors = build_lhb_factor_frame(market[["code", "date"]], detail_rows(), institution_rows(), availability(dates))
    merged = market.merge(factors, on=["code", "date"], how="left")
    assert merged.loc[0, "institution_activity"] == 7.0
    assert merged.loc[0, "institution_net_buy_amount"] == 123.0
    assert merged.loc[0, "lhb_inst_net_buy"] == 200.0
    assert "institution_net_buy" not in merged.columns


def test_old_lhb_institution_cache_columns_migrate_without_data_loss() -> None:
    old = institution_rows().rename(
        columns={
            "lhb_inst_buy_count": "institution_buy_count",
            "lhb_inst_sell_count": "institution_sell_count",
            "lhb_inst_buy_amount": "institution_buy_amount",
            "lhb_inst_sell_amount": "institution_sell_amount",
            "lhb_inst_net_buy": "institution_net_buy",
        }
    )
    result = aggregate_lhb_institution(old)
    assert result.loc[0, "lhb_inst_net_buy"] == 200.0
    assert "institution_net_buy" not in result.columns


def test_lhb_sources_have_independent_availability() -> None:
    trade_date = pd.Timestamp("2026-07-13")
    calendar = pd.DataFrame({"symbol": ["000002"], "trade_date": [trade_date]})
    detail_only = pd.DataFrame(
        {
            "trade_date": [trade_date],
            "lhb_detail_available": [1],
            "lhb_inst_data_available": [0],
        }
    )
    factors = build_lhb_factor_frame(calendar, detail_rows(), institution_rows(), detail_only)
    row = factors.iloc[0]
    assert row["lhb_flag"] == 0
    assert pd.isna(row["lhb_inst_net_buy"])

    institution_only = detail_only.copy()
    institution_only[["lhb_detail_available", "lhb_inst_data_available"]] = [0, 1]
    factors = build_lhb_factor_frame(calendar, detail_rows(), institution_rows(), institution_only)
    row = factors.iloc[0]
    assert pd.isna(row["lhb_flag"])
    assert row["lhb_inst_net_buy"] == 0


def test_successful_empty_lhb_response_is_not_a_download_failure() -> None:
    frame, success, error = _call_with_retry(
        lambda **_kwargs: pd.DataFrame(), "20260713", "20260713", max_retries=1, backoff_seconds=0
    )
    assert frame.empty
    assert success is True
    assert error == ""


def test_only_approved_lhb_factors_enter_model() -> None:
    dates = pd.bdate_range("2026-01-01", periods=90)
    market = pd.DataFrame(
        {
            "date": dates,
            "code": "000001",
            "open": np.linspace(10.0, 12.0, len(dates)),
            "high": np.linspace(10.2, 12.2, len(dates)),
            "low": np.linspace(9.8, 11.8, len(dates)),
            "close": np.linspace(10.1, 12.1, len(dates)),
            "volume": np.linspace(1_000_000, 2_000_000, len(dates)),
            "amount": np.linspace(10_000_000, 24_000_000, len(dates)),
        }
    )
    events = detail_rows().copy()
    events["trade_date"] = dates[30]
    institutions = institution_rows().copy()
    institutions["trade_date"] = dates[30]
    factors = build_lhb_factor_frame(market[["code", "date"]], events, institutions, availability(dates))
    merged = market.merge(factors, on=["code", "date"], how="left", suffixes=("", "_lhb"))
    dataset = add_labels(build_features(merged, external_factor_lag=0), round_trip_cost=0.002, down_threshold=-0.02)
    columns = feature_columns(dataset)
    assert "lhb_flag" in columns
    assert "lhb_count_20d" in columns
    assert "lhb_amount" not in columns
    assert "lhb_inst_buy_amount" not in columns
    assert "lhb_net_buy_signed_log" not in columns
