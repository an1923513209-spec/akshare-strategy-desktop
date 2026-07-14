"""Data helpers for A-share strategy experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def normalize_symbol(symbol: str) -> str:
    """Return the six-digit A-share code accepted by AKShare."""
    cleaned = symbol.strip().upper()
    for suffix in (".SZSE", ".SSE", ".SZ", ".SH"):
        cleaned = cleaned.replace(suffix, "")
    if len(cleaned) != 6 or not cleaned.isdigit():
        raise ValueError(f"Expected a six-digit A-share code, got: {symbol!r}")
    return cleaned


def tencent_symbol(symbol: str) -> str:
    code = normalize_symbol(symbol)
    prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{code}"


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize known AKShare A-share daily schemas to OHLCV."""
    if {"日期", "开盘", "最高", "最低", "收盘", "成交量"}.issubset(raw.columns):
        rename_map = {
            "日期": "Date",
            "开盘": "Open",
            "最高": "High",
            "最低": "Low",
            "收盘": "Close",
            "成交量": "Volume",
        }
    elif {"date", "open", "high", "low", "close", "amount"}.issubset(raw.columns):
        rename_map = {
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "amount": "Volume",
        }
    else:
        raise RuntimeError(f"Unexpected AKShare columns: {list(raw.columns)}")

    data = raw.rename(columns=rename_map)[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna().set_index("Date").sort_index()


def load_a_share_daily(
    symbol: str,
    start: str = "20180101",
    end: str | None = None,
    adjust: str = "qfq",
    cache: bool = True,
) -> pd.DataFrame:
    """Load daily OHLCV data from AKShare and return backtesting-ready columns."""
    code = normalize_symbol(symbol)
    end = end or datetime.now().strftime("%Y%m%d")
    cache_path = DATA_DIR / f"{code}_{start}_{end}_{adjust or 'raw'}.csv"

    if cache and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        requested_end = pd.to_datetime(end).date()
        today = datetime.now().date()
        latest_cached = cached.index.max().date() if not cached.empty else None
        if requested_end != today or (latest_cached is not None and latest_cached >= requested_end):
            return cached

    try:
        raw = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
    except Exception as first_error:
        print(f"AKShare Eastmoney source failed, falling back to Tencent source: {first_error}")
        raw = ak.stock_zh_a_hist_tx(
            symbol=tencent_symbol(code),
            start_date=start,
            end_date=end,
            adjust=adjust,
            timeout=20,
        )
    if raw.empty:
        raise RuntimeError(f"AKShare returned no data for {code}")

    data = normalize_ohlcv(raw)

    if cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path, encoding="utf-8-sig")

    return data
