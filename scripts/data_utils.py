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


def _read_daily_cache(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def _latest_daily_cache_path(code: str, start: str, end: str, adjust: str) -> Path | None:
    label = adjust or "raw"
    requested_end = pd.to_datetime(end).date()
    candidates: list[tuple[datetime, Path]] = []
    for path in DATA_DIR.glob(f"{code}_{start}_*_{label}.csv"):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            cache_end = datetime.strptime(parts[2], "%Y%m%d")
        except ValueError:
            continue
        if cache_end.date() <= requested_end:
            candidates.append((cache_end, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _cached_or_latest_daily(code: str, start: str, end: str, adjust: str, exact_path: Path) -> pd.DataFrame | None:
    if exact_path.exists():
        return _read_daily_cache(exact_path)
    latest_path = _latest_daily_cache_path(code, start, end, adjust)
    if latest_path is not None:
        return _read_daily_cache(latest_path)
    return None


def load_a_share_daily(
    symbol: str,
    start: str = "20180101",
    end: str | None = None,
    adjust: str = "qfq",
    cache: bool = True,
    refresh_stale_today: bool = False,
) -> pd.DataFrame:
    """Load daily OHLCV data from AKShare and return backtesting-ready columns."""
    code = normalize_symbol(symbol)
    end = end or datetime.now().strftime("%Y%m%d")
    cache_path = DATA_DIR / f"{code}_{start}_{end}_{adjust or 'raw'}.csv"

    cached: pd.DataFrame | None = None
    if cache and cache_path.exists():
        cached = _read_daily_cache(cache_path)
        requested_end = pd.to_datetime(end).date()
        today = datetime.now().date()
        latest_cached = cached.index.max().date() if not cached.empty else None
        if not refresh_stale_today or requested_end != today or (latest_cached is not None and latest_cached >= requested_end):
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
        try:
            raw = ak.stock_zh_a_hist_tx(
                symbol=tencent_symbol(code),
                start_date=start,
                end_date=end,
                adjust=adjust,
                timeout=20,
            )
        except Exception as second_error:
            if cache:
                fallback = cached if cached is not None else _cached_or_latest_daily(code, start, end, adjust, cache_path)
                if fallback is not None and not fallback.empty:
                    print(f"AKShare sources failed, using cached daily data for {code}: {fallback.index.max():%Y-%m-%d}")
                    return fallback
            raise RuntimeError(
                f"无法联网拉取 {code} 日线数据，且没有可用本地缓存。"
                f"东方财富错误：{first_error}; 腾讯源错误：{second_error}"
            ) from second_error
    if raw.empty:
        if cache:
            fallback = cached if cached is not None else _cached_or_latest_daily(code, start, end, adjust, cache_path)
            if fallback is not None and not fallback.empty:
                print(f"AKShare returned empty data, using cached daily data for {code}: {fallback.index.max():%Y-%m-%d}")
                return fallback
        raise RuntimeError(f"AKShare returned no data for {code}")

    data = normalize_ohlcv(raw)

    if cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path, encoding="utf-8-sig")

    return data
