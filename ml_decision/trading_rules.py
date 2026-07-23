"""Deterministic A-share execution constraints for recommendation sizing."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, time
from typing import Any

import numpy as np
import pandas as pd


TRADE_RULE_COLUMNS = {
    "previous_close",
    "price_limit_rate",
    "limit_up_price",
    "limit_down_price",
    "is_suspended",
    "is_one_price_up",
    "is_one_price_down",
    "trade_rule_supported",
}


def board_name(code: str) -> str:
    symbol = str(code).zfill(6)
    if symbol.startswith(("300", "301")):
        return "chinext"
    if symbol.startswith(("688", "689")):
        return "star"
    if symbol.startswith(("4", "8", "92")):
        return "beijing"
    return "main"


def price_limit_rate(code: str, name: str = "") -> float:
    if "ST" in str(name).upper():
        return 0.05
    board = board_name(code)
    if board in {"chinext", "star"}:
        return 0.20
    if board == "beijing":
        return 0.30
    return 0.10


def _price(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def enrich_trade_constraints(market_df: pd.DataFrame) -> pd.DataFrame:
    """Add observable suspension/limit fields without making fill assumptions."""
    frame = market_df.copy().sort_values(["code", "date"], kind="stable")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["previous_close"] = frame.groupby("code", sort=False)["close"].shift(1)
    names = frame.get("name", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["price_limit_rate"] = [price_limit_rate(code, name) for code, name in zip(frame["code"], names)]
    frame["limit_up_price"] = [
        _price(previous * (1.0 + rate)) if np.isfinite(previous) else np.nan
        for previous, rate in zip(pd.to_numeric(frame["previous_close"], errors="coerce"), frame["price_limit_rate"])
    ]
    frame["limit_down_price"] = [
        _price(previous * (1.0 - rate)) if np.isfinite(previous) else np.nan
        for previous, rate in zip(pd.to_numeric(frame["previous_close"], errors="coerce"), frame["price_limit_rate"])
    ]
    volume = pd.to_numeric(frame.get("volume", 0), errors="coerce").fillna(0.0)
    frame["is_suspended"] = volume.le(0)
    prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    frame["is_one_price_up"] = prices.sub(frame["limit_up_price"], axis=0).abs().le(0.001).all(axis=1) & volume.gt(0)
    frame["is_one_price_down"] = prices.sub(frame["limit_down_price"], axis=0).abs().le(0.001).all(axis=1) & volume.gt(0)
    # New-listing no-limit phases require a reliable listing date. Existing
    # rows remain usable, but the limitation is explicit for diagnostics.
    frame["trade_rule_supported"] = True
    if "listing_date" in frame.columns:
        listing = pd.to_datetime(frame["listing_date"], errors="coerce")
        trade_date = pd.to_datetime(frame["date"], errors="coerce")
        frame.loc[(trade_date - listing).dt.days.between(0, 6), "trade_rule_supported"] = False
    return frame


def drop_incomplete_latest_daily_bar(
    frame: pd.DataFrame,
    *,
    now: datetime | None = None,
    market_close_grace: time = time(15, 5),
) -> pd.DataFrame:
    """Exclude today's daily bar before the close data can be considered final."""
    if frame.empty:
        return frame.copy()
    current = now or datetime.now()
    if current.weekday() >= 5 or current.time() >= market_close_grace:
        return frame.copy()
    result = frame.copy()
    if "date" in result.columns:
        dates = pd.to_datetime(result["date"], errors="coerce").dt.date
        return result.loc[dates.ne(current.date())].copy()
    dates = pd.to_datetime(result.index, errors="coerce").date
    return result.loc[np.asarray(dates) != current.date()].copy()
