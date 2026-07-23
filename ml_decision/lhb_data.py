"""Incremental Dragon-Tiger List data and leakage-safe factors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "lhb_factor_config.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "lhb"
DETAIL_PATH = RAW_DIR / "lhb_detail.parquet"
INSTITUTION_PATH = RAW_DIR / "lhb_institution.parquet"
AVAILABILITY_PATH = RAW_DIR / "lhb_availability.parquet"
QUALITY_DIR = PROJECT_ROOT / "data" / "quality" / "lhb"
QUALITY_PATH = QUALITY_DIR / "lhb_quality_report.parquet"
LOG_PATH = PROJECT_ROOT / "logs" / "lhb_data.log"
LOCK_PATH = RAW_DIR / ".update.lock"

DEFAULT_CONFIG: dict[str, Any] = {
    "reason_keywords": {
        "lhb_reason_price_deviation": ["涨幅偏离", "跌幅偏离", "价格偏离", "偏离值"],
        "lhb_reason_turnover": ["换手率", "换手"],
        "lhb_reason_amplitude": ["振幅"],
        "lhb_reason_abnormal": ["异常波动", "严重异常", "异常期间"],
        "lhb_reason_three_day": ["连续三个交易日", "三个交易日内", "3个交易日"],
        "lhb_reason_st": ["ST证券", "*ST", "退市整理"],
    },
    "forbidden_feature_keywords": ["future", "next", "target", "label", "上榜后", "未来", "后续"],
    "download": {
        "start_date": "20200101",
        "chunk_days": 30,
        "max_retries": 3,
        "retry_backoff_seconds": 1.5,
        "request_interval_seconds": 0.35,
    },
}

DETAIL_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("代码", "股票代码", "证券代码", "symbol", "code"),
    "trade_date": ("上榜日", "上榜日期", "交易日期", "日期", "trade_date", "date"),
    "name": ("名称", "股票名称", "证券简称", "name"),
    "close": ("收盘价", "close"),
    "pct_change": ("涨跌幅", "涨跌幅度", "pct_change"),
    "lhb_net_buy": ("龙虎榜净买额", "净买额", "lhb_net_buy"),
    "lhb_buy": ("龙虎榜买入额", "买入额", "lhb_buy"),
    "lhb_sell": ("龙虎榜卖出额", "卖出额", "lhb_sell"),
    "lhb_amount": ("龙虎榜成交额", "成交额", "lhb_amount"),
    "stock_total_amount": ("市场总成交额", "总成交额", "stock_total_amount"),
    "turnover_rate": ("换手率", "turnover_rate"),
    "float_market_cap": ("流通市值", "float_market_cap"),
    "lhb_reason": ("上榜原因", "原因", "lhb_reason"),
    "lhb_interpretation": ("解读", "龙虎榜解读", "lhb_interpretation"),
}

INSTITUTION_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("代码", "股票代码", "证券代码", "symbol", "code"),
    "trade_date": ("上榜日期", "上榜日", "交易日期", "日期", "trade_date", "date"),
    "name": ("名称", "股票名称", "证券简称", "name"),
    "lhb_inst_buy_count": ("买方机构数", "机构买入次数", "lhb_inst_buy_count", "institution_buy_count"),
    "lhb_inst_sell_count": ("卖方机构数", "机构卖出次数", "lhb_inst_sell_count", "institution_sell_count"),
    "lhb_inst_buy_amount": ("机构买入总额", "机构买入额", "lhb_inst_buy_amount", "institution_buy_amount"),
    "lhb_inst_sell_amount": ("机构卖出总额", "机构卖出额", "lhb_inst_sell_amount", "institution_sell_amount"),
    "lhb_inst_net_buy": ("机构买入净额", "机构净买额", "lhb_inst_net_buy", "institution_net_buy"),
    "stock_total_amount": ("市场总成交额", "总成交额", "stock_total_amount"),
    "lhb_reason": ("上榜原因", "原因", "lhb_reason"),
}

DETAIL_AMOUNT_COLUMNS = ("lhb_net_buy", "lhb_buy", "lhb_sell", "lhb_amount")
INSTITUTION_EVENT_COLUMNS = (
    "lhb_inst_buy_count",
    "lhb_inst_sell_count",
    "lhb_inst_buy_amount",
    "lhb_inst_sell_amount",
    "lhb_inst_net_buy",
)
EVENT_ZERO_COLUMNS = (
    "lhb_flag",
    "lhb_record_count",
    "lhb_reason_count",
    *DETAIL_AMOUNT_COLUMNS,
)
DIRECT_FACTOR_ZERO_COLUMNS = (
    "lhb_net_buy_ratio",
    "lhb_amount_ratio",
    "lhb_buy_sell_balance",
    "lhb_net_buy_float_cap_ratio",
    "lhb_buy_log",
    "lhb_sell_log",
    "lhb_net_buy_signed_log",
    "lhb_inst_count_balance",
    "lhb_inst_net_buy_ratio",
    "lhb_inst_buy_sell_balance",
    "lhb_inst_net_buy_signed_log",
)
RATIO_COLUMNS = (
    "lhb_net_buy_ratio",
    "lhb_amount_ratio",
    "lhb_buy_sell_balance",
    "lhb_net_buy_float_cap_ratio",
    "lhb_inst_net_buy_ratio",
    "lhb_inst_buy_sell_balance",
)


class LHBSchemaError(ValueError):
    """Raised when an AKShare response no longer contains required fields."""


@dataclass(slots=True)
class LHBUpdateResult:
    start_date: str
    end_date: str
    requested_dates: int
    successful_dates: int
    failed_dates: int
    detail_rows_added: int
    institution_rows_added: int
    quality_report_path: str


def _logger() -> logging.Logger:
    logger = logging.getLogger("ml_decision.lhb_data")
    if logger.handlers:
        return logger
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_lhb_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG, ensure_ascii=False))
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception:
        return json.loads(json.dumps(DEFAULT_CONFIG, ensure_ascii=False))
    config = json.loads(json.dumps(DEFAULT_CONFIG, ensure_ascii=False))
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def normalize_symbol(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    match = re.search(r"(\d{1,6})", text)
    return match.group(1).zfill(6) if match else text.zfill(6)


def _normalized_column_name(value: Any) -> str:
    return re.sub(r"[\s_\-（）()/%]+", "", str(value)).lower()


def _resolve_mapping(columns: Iterable[Any], aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    actual = {str(column): _normalized_column_name(column) for column in columns}
    result: dict[str, str] = {}
    missing: list[str] = []
    for canonical, choices in aliases.items():
        normalized_choices = {_normalized_column_name(choice) for choice in choices}
        source = next((column for column, normalized in actual.items() if normalized in normalized_choices), None)
        if source is None:
            missing.append(canonical)
        else:
            result[source] = canonical
    if missing:
        raise LHBSchemaError(f"AKShare 龙虎榜字段变化，缺少标准字段: {missing}; 实际字段: {list(actual)}")
    return result


def _coalesce_alias_frame(frame: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """Normalize aliases while preserving values across old and new cache columns."""
    actual = {str(column): _normalized_column_name(column) for column in frame.columns}
    output: dict[str, pd.Series] = {}
    missing: list[str] = []
    for canonical, choices in aliases.items():
        normalized_choices = {_normalized_column_name(choice) for choice in choices}
        matches = [column for column, normalized in actual.items() if normalized in normalized_choices]
        if not matches:
            missing.append(canonical)
            continue
        output[canonical] = frame[matches].bfill(axis=1).iloc[:, 0] if len(matches) > 1 else frame[matches[0]]
    if missing:
        raise LHBSchemaError(f"AKShare LHB schema is missing fields: {missing}; actual={list(actual)}")
    return pd.DataFrame(output, index=frame.index)


def _drop_future_columns(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    keywords = (config or load_lhb_config()).get("forbidden_feature_keywords", [])
    blocked = [column for column in frame.columns if any(str(key).lower() in str(column).lower() for key in keywords)]
    return frame.drop(columns=blocked, errors="ignore")


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def normalize_lhb_detail(raw: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Map changing AKShare columns to a stable detail schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(DETAIL_ALIASES))
    clean = _drop_future_columns(raw.copy(), config)
    clean = _coalesce_alias_frame(clean, DETAIL_ALIASES)
    clean["symbol"] = clean["symbol"].map(normalize_symbol)
    clean["trade_date"] = pd.to_datetime(clean["trade_date"], errors="coerce").dt.normalize()
    clean["name"] = clean["name"].fillna("").astype(str).str.strip()
    clean["lhb_reason"] = clean["lhb_reason"].fillna("").astype(str).str.strip()
    clean["lhb_interpretation"] = clean["lhb_interpretation"].fillna("").astype(str).str.strip()
    _coerce_numeric(
        clean,
        ("close", "pct_change", "lhb_net_buy", "lhb_buy", "lhb_sell", "lhb_amount", "stock_total_amount", "turnover_rate", "float_market_cap"),
    )
    return clean.dropna(subset=["symbol", "trade_date"]).reset_index(drop=True)


def normalize_lhb_institution(raw: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Map changing AKShare columns to a stable institution schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(INSTITUTION_ALIASES))
    clean = _drop_future_columns(raw.copy(), config)
    clean = _coalesce_alias_frame(clean, INSTITUTION_ALIASES)
    clean["symbol"] = clean["symbol"].map(normalize_symbol)
    clean["trade_date"] = pd.to_datetime(clean["trade_date"], errors="coerce").dt.normalize()
    clean["name"] = clean["name"].fillna("").astype(str).str.strip()
    clean["lhb_reason"] = clean["lhb_reason"].fillna("").astype(str).str.strip()
    _coerce_numeric(clean, (*INSTITUTION_EVENT_COLUMNS, "stock_total_amount"))
    return clean.dropna(subset=["symbol", "trade_date"]).reset_index(drop=True)


def _unique_text(values: pd.Series) -> str:
    unique = list(dict.fromkeys(str(value).strip() for value in values if pd.notna(value) and str(value).strip()))
    return " | ".join(unique)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / den


def _signed_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def aggregate_lhb_detail(raw: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Aggregate duplicate reasons without double-counting repeated amounts."""
    data = normalize_lhb_detail(raw, config)
    if data.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "lhb_flag"])
    grouped = data.groupby(["symbol", "trade_date"], sort=True, as_index=False).agg(
        name=("name", "last"),
        close=("close", "max"),
        pct_change=("pct_change", "max"),
        lhb_net_buy=("lhb_net_buy", "max"),
        lhb_buy=("lhb_buy", "max"),
        lhb_sell=("lhb_sell", "max"),
        lhb_amount=("lhb_amount", "max"),
        stock_total_amount=("stock_total_amount", "max"),
        turnover_rate=("turnover_rate", "max"),
        float_market_cap=("float_market_cap", "max"),
        lhb_reason_text=("lhb_reason", _unique_text),
        lhb_interpretation=("lhb_interpretation", _unique_text),
        lhb_record_count=("lhb_reason", "size"),
        lhb_reason_count=("lhb_reason", lambda values: len({str(value).strip() for value in values if str(value).strip()})),
    )
    grouped["lhb_flag"] = 1.0
    grouped["lhb_net_buy_ratio"] = _safe_divide(grouped["lhb_net_buy"], grouped["stock_total_amount"])
    grouped["lhb_amount_ratio"] = _safe_divide(grouped["lhb_amount"], grouped["stock_total_amount"])
    grouped["lhb_buy_sell_balance"] = _safe_divide(grouped["lhb_buy"] - grouped["lhb_sell"], grouped["lhb_buy"] + grouped["lhb_sell"])
    grouped["lhb_net_buy_float_cap_ratio"] = _safe_divide(grouped["lhb_net_buy"], grouped["float_market_cap"])
    grouped["lhb_buy_log"] = np.log1p(grouped["lhb_buy"].clip(lower=0))
    grouped["lhb_sell_log"] = np.log1p(grouped["lhb_sell"].clip(lower=0))
    grouped["lhb_net_buy_signed_log"] = _signed_log(grouped["lhb_net_buy"])
    reason_keywords = (config or load_lhb_config()).get("reason_keywords", {})
    for factor, keywords in reason_keywords.items():
        pattern = "|".join(re.escape(str(keyword)) for keyword in keywords)
        grouped[factor] = grouped["lhb_reason_text"].str.contains(pattern, case=False, na=False).astype(float)
    _assert_unique(grouped, "龙虎榜日聚合")
    return grouped.replace([np.inf, -np.inf], np.nan)


def aggregate_lhb_institution(raw: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    data = normalize_lhb_institution(raw, config)
    if data.empty:
        return pd.DataFrame(columns=["symbol", "trade_date"])
    grouped = data.groupby(["symbol", "trade_date"], sort=True, as_index=False).agg(
        lhb_inst_buy_count=("lhb_inst_buy_count", "max"),
        lhb_inst_sell_count=("lhb_inst_sell_count", "max"),
        lhb_inst_buy_amount=("lhb_inst_buy_amount", "max"),
        lhb_inst_sell_amount=("lhb_inst_sell_amount", "max"),
        lhb_inst_net_buy=("lhb_inst_net_buy", "max"),
        lhb_inst_stock_total_amount=("stock_total_amount", "max"),
    )
    grouped["lhb_inst_count_balance"] = grouped["lhb_inst_buy_count"] - grouped["lhb_inst_sell_count"]
    grouped["lhb_inst_net_buy_ratio"] = _safe_divide(grouped["lhb_inst_net_buy"], grouped["lhb_inst_stock_total_amount"])
    grouped["lhb_inst_buy_sell_balance"] = _safe_divide(
        grouped["lhb_inst_buy_amount"] - grouped["lhb_inst_sell_amount"],
        grouped["lhb_inst_buy_amount"] + grouped["lhb_inst_sell_amount"],
    )
    grouped["lhb_inst_net_buy_signed_log"] = _signed_log(grouped["lhb_inst_net_buy"])
    grouped["lhb_inst_buy_flag"] = (grouped["lhb_inst_buy_count"] > 0).astype(float)
    grouped["lhb_inst_sell_flag"] = (grouped["lhb_inst_sell_count"] > 0).astype(float)
    grouped["lhb_inst_net_buy_positive"] = (grouped["lhb_inst_net_buy"] > 0).astype(float)
    _assert_unique(grouped, "机构席位日聚合")
    return grouped.replace([np.inf, -np.inf], np.nan)


def _assert_unique(frame: pd.DataFrame, label: str) -> None:
    duplicate_count = int(frame.duplicated(["symbol", "trade_date"]).sum())
    if duplicate_count:
        raise ValueError(f"{label} symbol + trade_date 非唯一，重复 {duplicate_count} 行")


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _append_raw(existing: pd.DataFrame, incoming: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    if incoming.empty:
        return existing.copy()
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False) if not existing.empty else incoming.copy()
    return combined.drop_duplicates(subset=subset, keep="last").sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _call_with_retry(
    function: Callable[..., pd.DataFrame],
    start_date: str,
    end_date: str,
    max_retries: int,
    backoff_seconds: float,
) -> tuple[pd.DataFrame, bool, str]:
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            result = function(start_date=start_date, end_date=end_date)
            return (result if isinstance(result, pd.DataFrame) else pd.DataFrame()), True, ""
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = f"{type(exc).__name__}: {exc}"
            _logger().warning("LHB request %s..%s attempt %s failed: %s", start_date, end_date, attempt, last_error)
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    return pd.DataFrame(), False, last_error


def _date_chunks(dates: pd.DatetimeIndex, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if dates.empty:
        return []
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = dates[0]
    previous = dates[0]
    for current in dates[1:]:
        if (current - start).days >= chunk_days or (current - previous).days > 4:
            chunks.append((start, previous))
            start = current
        previous = current
    chunks.append((start, previous))
    return chunks


def _a_share_trading_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Use the exchange calendar; weekdays alone incorrectly include holidays."""
    try:
        import akshare as ak

        calendar = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(calendar["trade_date"], errors="coerce").dropna().dt.normalize()
        selected = dates[(dates >= start) & (dates <= end)]
        if not selected.empty:
            return pd.DatetimeIndex(selected.drop_duplicates().sort_values())
    except Exception as exc:  # pragma: no cover - network/source dependent
        _logger().warning("A-share trading calendar unavailable, using weekdays: %s", exc)
    return pd.bdate_range(start, end)


@contextmanager
def _update_lock(timeout_seconds: float = 220.0) -> Iterable[None]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            LOCK_PATH.mkdir()
            (LOCK_PATH / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "created_at": str(datetime.now())}), encoding="utf-8"
            )
            break
        except FileExistsError:
            try:
                age = time.time() - LOCK_PATH.stat().st_mtime
                owner_path = LOCK_PATH / "owner.json"
                owner = json.loads(owner_path.read_text(encoding="utf-8")) if owner_path.exists() else {}
                owner_pid = int(owner.get("pid") or 0)
                if (owner_pid and not _process_exists(owner_pid)) or age > 900:
                    shutil.rmtree(LOCK_PATH, ignore_errors=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("等待其他进程更新龙虎榜缓存超时")
            time.sleep(0.5)
    try:
        yield
    finally:
        shutil.rmtree(LOCK_PATH, ignore_errors=True)


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def update_lhb_data(
    start_date: str | date | datetime | pd.Timestamp | None = None,
    end_date: str | date | datetime | pd.Timestamp | None = None,
    force: bool = False,
    detail_fetcher: Callable[..., pd.DataFrame] | None = None,
    institution_fetcher: Callable[..., pd.DataFrame] | None = None,
) -> LHBUpdateResult:
    """Serialize global-cache writes across parallel ML worker processes."""
    with _update_lock():
        return _update_lhb_data_unlocked(
            start_date,
            end_date,
            force=force,
            detail_fetcher=detail_fetcher,
            institution_fetcher=institution_fetcher,
        )


def _update_lhb_data_unlocked(
    start_date: str | date | datetime | pd.Timestamp | None = None,
    end_date: str | date | datetime | pd.Timestamp | None = None,
    force: bool = False,
    detail_fetcher: Callable[..., pd.DataFrame] | None = None,
    institution_fetcher: Callable[..., pd.DataFrame] | None = None,
) -> LHBUpdateResult:
    """Download the first history or only missing business-date ranges."""
    config = load_lhb_config()
    download = config["download"]
    start = pd.Timestamp(start_date or download["start_date"]).normalize()
    end = pd.Timestamp(end_date or datetime.now().date()).normalize()
    if end < start:
        raise ValueError("龙虎榜结束日期不能早于开始日期")
    availability = _read_parquet(AVAILABILITY_PATH)
    requested = _a_share_trading_dates(start, end)
    if not availability.empty:
        availability["trade_date"] = pd.to_datetime(availability["trade_date"]).dt.normalize()
        in_range = availability["trade_date"].between(start, end)
        is_trading_date = availability["trade_date"].isin(set(requested))
        cleaned_availability = availability.loc[~in_range | is_trading_date].copy()
        if len(cleaned_availability) != len(availability):
            availability = cleaned_availability
            _atomic_parquet(availability.sort_values("trade_date"), AVAILABILITY_PATH)
            quality = _read_parquet(QUALITY_PATH)
            if not quality.empty and "trade_date" in quality.columns:
                quality["trade_date"] = pd.to_datetime(quality["trade_date"]).dt.normalize()
                quality_in_range = quality["trade_date"].between(start, end)
                quality = quality.loc[~quality_in_range | quality["trade_date"].isin(set(requested))]
                _atomic_parquet(quality, QUALITY_PATH)
    if not force and not availability.empty:
        completed = set(availability.loc[availability["lhb_data_available"].eq(1), "trade_date"])
        requested = pd.DatetimeIndex([item for item in requested if item not in completed])
    if requested.empty:
        return LHBUpdateResult(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), 0, 0, 0, 0, 0, str(QUALITY_PATH))

    if detail_fetcher is None or institution_fetcher is None:
        import akshare as ak

        detail_fetcher = detail_fetcher or ak.stock_lhb_detail_em
        institution_fetcher = institution_fetcher or ak.stock_lhb_jgmmtj_em

    detail_all = _read_parquet(DETAIL_PATH)
    institution_all = _read_parquet(INSTITUTION_PATH)
    if not detail_all.empty:
        detail_all = normalize_lhb_detail(detail_all, config)
    if not institution_all.empty:
        institution_all = normalize_lhb_institution(institution_all, config)
    availability_all = availability.copy()
    detail_rows_added = 0
    institution_rows_added = 0
    availability_rows: list[dict[str, Any]] = []
    interval = float(download["request_interval_seconds"])
    for chunk_start, chunk_end in _date_chunks(requested, int(download["chunk_days"])):
        start_text = chunk_start.strftime("%Y%m%d")
        end_text = chunk_end.strftime("%Y%m%d")
        detail_raw, detail_ok, detail_error = _call_with_retry(
            detail_fetcher, start_text, end_text, int(download["max_retries"]), float(download["retry_backoff_seconds"])
        )
        time.sleep(interval)
        institution_raw, institution_ok, institution_error = _call_with_retry(
            institution_fetcher, start_text, end_text, int(download["max_retries"]), float(download["retry_backoff_seconds"])
        )
        try:
            detail_clean = normalize_lhb_detail(detail_raw, config) if detail_ok else pd.DataFrame()
        except Exception as exc:
            detail_clean, detail_ok, detail_error = pd.DataFrame(), False, str(exc)
        try:
            institution_clean = normalize_lhb_institution(institution_raw, config) if institution_ok else pd.DataFrame()
        except Exception as exc:
            institution_clean, institution_ok, institution_error = pd.DataFrame(), False, str(exc)
        detail_rows_added += len(detail_clean)
        institution_rows_added += len(institution_clean)
        chunk_dates = requested[(requested >= chunk_start) & (requested <= chunk_end)]
        chunk_availability_rows: list[dict[str, Any]] = []
        for trade_date in chunk_dates:
            # A successful empty response means that no stock matched the event
            # source for the requested range. It is not a transport failure.
            # Availability is recorded per source and never inferred from the
            # maximum event date, because intermediate dates may have no rows.
            now = pd.Timestamp.now()
            is_published = bool(trade_date < now.normalize() or (trade_date == now.normalize() and now.hour >= 18))
            detail_date_ok = bool(detail_ok and is_published)
            institution_date_ok = bool(institution_ok and is_published)
            row = {
                    "trade_date": trade_date,
                    "lhb_detail_available": int(detail_date_ok),
                    "lhb_inst_data_available": int(institution_date_ok),
                    "detail_available": int(detail_date_ok),
                    "institution_available": int(institution_date_ok),
                    "lhb_data_available": int(detail_date_ok and institution_date_ok),
                    "downloaded_at": pd.Timestamp.now(),
                    "detail_error": detail_error,
                    "institution_error": institution_error,
                }
            availability_rows.append(row)
            chunk_availability_rows.append(row)

        # Commit every successful range independently. A timeout or manual
        # stop therefore loses at most the in-flight range, never the history
        # already downloaded during this run.
        detail_all = _append_raw(
            detail_all, detail_clean, ["symbol", "trade_date", "lhb_reason", "lhb_interpretation"]
        )
        institution_all = _append_raw(
            institution_all, institution_clean, ["symbol", "trade_date", "lhb_reason"]
        )
        chunk_availability = pd.DataFrame(chunk_availability_rows)
        availability_all = (
            pd.concat([availability_all, chunk_availability], ignore_index=True, sort=False)
            if not availability_all.empty
            else chunk_availability
        )
        availability_all = (
            availability_all.sort_values("downloaded_at")
            .drop_duplicates("trade_date", keep="last")
            .sort_values("trade_date")
        )
        if not detail_all.empty:
            _atomic_parquet(detail_all, DETAIL_PATH)
        if not institution_all.empty:
            _atomic_parquet(institution_all, INSTITUTION_PATH)
        _atomic_parquet(availability_all, AVAILABILITY_PATH)
        write_quality_report(detail_clean, institution_clean, chunk_availability)
        _logger().info(
            "LHB %s..%s detail=%s institution=%s detail_rows=%s institution_rows=%s",
            start_text,
            end_text,
            detail_ok,
            institution_ok,
            len(detail_clean),
            len(institution_clean),
        )
        time.sleep(interval)

    availability_new = pd.DataFrame(availability_rows)
    successes = int(availability_new["lhb_data_available"].sum()) if not availability_new.empty else 0
    return LHBUpdateResult(
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        len(requested),
        successes,
        len(requested) - successes,
        detail_rows_added,
        institution_rows_added,
        str(QUALITY_PATH),
    )


def _fill_available_zeros(frame: pd.DataFrame) -> pd.DataFrame:
    detail_available = frame["lhb_detail_available"].eq(1)
    institution_available = frame["lhb_inst_data_available"].eq(1)
    for column in EVENT_ZERO_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame.loc[detail_available, column] = frame.loc[detail_available, column].fillna(0.0)
    for column in INSTITUTION_EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame.loc[institution_available, column] = frame.loc[institution_available, column].fillna(0.0)
    detail_direct = [column for column in DIRECT_FACTOR_ZERO_COLUMNS if not column.startswith("lhb_inst_")]
    institution_direct = [column for column in DIRECT_FACTOR_ZERO_COLUMNS if column.startswith("lhb_inst_")]
    for column in detail_direct:
        if column not in frame.columns:
            frame[column] = np.nan
        frame.loc[detail_available, column] = frame.loc[detail_available, column].fillna(0.0)
    for column in institution_direct:
        if column not in frame.columns:
            frame[column] = np.nan
        frame.loc[institution_available, column] = frame.loc[institution_available, column].fillna(0.0)
    reason_columns = list(load_lhb_config().get("reason_keywords", {}))
    for column in reason_columns:
        if column not in frame.columns:
            frame[column] = np.nan
        frame.loc[detail_available, column] = frame.loc[detail_available, column].fillna(0.0)
    for column in ("lhb_inst_buy_flag", "lhb_inst_sell_flag", "lhb_inst_net_buy_positive"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame.loc[institution_available, column] = frame.loc[institution_available, column].fillna(0.0)
    return frame


def _rolling_sum(group: pd.DataFrame, column: str, window: int, availability_column: str) -> pd.Series:
    result = pd.to_numeric(group[column], errors="coerce").rolling(window, min_periods=1).sum()
    complete = pd.to_numeric(group[availability_column], errors="coerce").rolling(window, min_periods=1).min().eq(1)
    return result.where(complete)


def _days_since_last_lhb(flags: pd.Series) -> pd.Series:
    values = pd.to_numeric(flags, errors="coerce")
    result: list[float] = []
    last_position: int | None = None
    for position, value in enumerate(values):
        if pd.isna(value):
            last_position = None
            result.append(np.nan)
        elif value == 1:
            last_position = position
            result.append(0.0)
        elif last_position is None:
            result.append(np.nan)
        else:
            result.append(float(position - last_position))
    return pd.Series(result, index=flags.index, dtype=float)


def _consecutive_lhb(flags: pd.Series) -> pd.Series:
    values = pd.to_numeric(flags, errors="coerce")
    result: list[float] = []
    count = 0
    for value in values:
        if pd.isna(value):
            count = 0
            result.append(np.nan)
        elif value == 1:
            count += 1
            result.append(float(count))
        else:
            count = 0
            result.append(0.0)
    return pd.Series(result, index=flags.index, dtype=float)


def build_lhb_factor_frame(
    market_calendar: pd.DataFrame,
    detail_raw: pd.DataFrame | None = None,
    institution_raw: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align sparse LHB events to each stock's complete trading calendar."""
    calendar = market_calendar.copy()
    rename = {}
    if "code" in calendar.columns and "symbol" not in calendar.columns:
        rename["code"] = "symbol"
    if "date" in calendar.columns and "trade_date" not in calendar.columns:
        rename["date"] = "trade_date"
    calendar = calendar.rename(columns=rename)
    if not {"symbol", "trade_date"}.issubset(calendar.columns):
        raise ValueError("market_calendar 必须包含 symbol/trade_date 或 code/date")
    calendar = calendar[["symbol", "trade_date"]].copy()
    calendar["symbol"] = calendar["symbol"].map(normalize_symbol)
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"], errors="coerce").dt.normalize()
    calendar = calendar.dropna().drop_duplicates().sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    detail = aggregate_lhb_detail(detail_raw if detail_raw is not None else _read_parquet(DETAIL_PATH))
    institution = aggregate_lhb_institution(institution_raw if institution_raw is not None else _read_parquet(INSTITUTION_PATH))
    availability_data = availability.copy() if availability is not None else _read_parquet(AVAILABILITY_PATH)
    if availability_data.empty:
        availability_data = pd.DataFrame({"trade_date": calendar["trade_date"].drop_duplicates()})
    availability_data["trade_date"] = pd.to_datetime(availability_data["trade_date"], errors="coerce").dt.normalize()
    availability_data = availability_data.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    if "lhb_detail_available" not in availability_data.columns:
        availability_data["lhb_detail_available"] = availability_data.get("detail_available", availability_data.get("lhb_data_available", 0))
    if "lhb_inst_data_available" not in availability_data.columns:
        availability_data["lhb_inst_data_available"] = availability_data.get("institution_available", availability_data.get("lhb_data_available", 0))
    availability_data["lhb_data_available"] = (
        pd.to_numeric(availability_data["lhb_detail_available"], errors="coerce").eq(1)
        & pd.to_numeric(availability_data["lhb_inst_data_available"], errors="coerce").eq(1)
    ).astype(int)

    availability_columns = ["trade_date", "lhb_detail_available", "lhb_inst_data_available", "lhb_data_available"]
    factors = calendar.merge(availability_data[availability_columns], on="trade_date", how="left")
    for column in availability_columns[1:]:
        factors[column] = factors[column].fillna(0).astype(int)
    if not detail.empty:
        factors = factors.merge(detail, on=["symbol", "trade_date"], how="left")
    if not institution.empty:
        factors = factors.merge(institution, on=["symbol", "trade_date"], how="left")
    factors = _fill_available_zeros(factors)

    groups: list[pd.DataFrame] = []
    for _symbol, group in factors.groupby("symbol", sort=False):
        g = group.sort_values("trade_date").copy()
        for window in (5, 10, 20, 60):
            g[f"lhb_count_{window}d"] = _rolling_sum(g, "lhb_flag", window, "lhb_detail_available")
        for window in (5, 10, 20):
            g[f"lhb_net_buy_sum_{window}d"] = _rolling_sum(g, "lhb_net_buy", window, "lhb_detail_available")
            g[f"lhb_inst_net_buy_sum_{window}d"] = _rolling_sum(g, "lhb_inst_net_buy", window, "lhb_inst_data_available")
        g["days_since_last_lhb"] = _days_since_last_lhb(g["lhb_flag"])
        g["consecutive_lhb_days"] = _consecutive_lhb(g["lhb_flag"])
        g["_lhb_positive"] = (g["lhb_net_buy"] > 0).where(g["lhb_net_buy"].notna())
        g["_lhb_negative"] = (g["lhb_net_buy"] < 0).where(g["lhb_net_buy"].notna())
        g["_lhb_inst_positive"] = (g["lhb_inst_net_buy"] > 0).where(g["lhb_inst_net_buy"].notna())
        g["lhb_positive_count_5d"] = _rolling_sum(g, "_lhb_positive", 5, "lhb_detail_available")
        g["lhb_negative_count_5d"] = _rolling_sum(g, "_lhb_negative", 5, "lhb_detail_available")
        g["lhb_inst_positive_count_20d"] = _rolling_sum(g, "_lhb_inst_positive", 20, "lhb_inst_data_available")
        g = g.drop(columns=["_lhb_positive", "_lhb_negative", "_lhb_inst_positive"])
        groups.append(g)
    result = pd.concat(groups, ignore_index=True) if groups else factors
    _assert_unique(result, "龙虎榜因子表")
    validate_lhb_factor_frame(result)
    result = result.rename(columns={"symbol": "code", "trade_date": "date"})
    return result.sort_values(["code", "date"]).reset_index(drop=True)


def load_lhb_factors_for_market(market_df: pd.DataFrame, force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Update when requested, then construct factors on the supplied calendar."""
    if market_df.empty:
        return pd.DataFrame(), "empty market calendar"
    dates = pd.to_datetime(market_df["date"], errors="coerce")
    update_note = "cache"
    if force_refresh:
        result = update_lhb_data(dates.min(), dates.max())
        update_note = f"updated {result.successful_dates}/{result.requested_dates} dates"
    if not DETAIL_PATH.exists() or not AVAILABILITY_PATH.exists():
        return pd.DataFrame(), "missing cache; use refresh external data for first full download"
    factors = build_lhb_factor_frame(market_df[["code", "date"]])
    overlapping_market_columns = (set(factors.columns) & set(market_df.columns)) - {"code", "date"}
    if overlapping_market_columns:
        factors = factors.drop(columns=sorted(overlapping_market_columns))
    available = int(factors["lhb_data_available"].eq(1).sum()) if "lhb_data_available" in factors.columns else 0
    return factors, f"{update_note}; {len(factors)} stock-days; available={available}"


def validate_lhb_factor_frame(frame: pd.DataFrame) -> dict[str, int]:
    """Run deterministic quality checks used by tests and daily reports."""
    _assert_unique(frame, "龙虎榜因子表")
    if not frame.sort_values(["symbol", "trade_date"]).index.equals(frame.index):
        frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    monotonic_bad = int(
        sum(not group["trade_date"].is_monotonic_increasing for _symbol, group in frame.groupby("symbol", sort=False))
    )
    numeric = frame.select_dtypes(include=[np.number])
    infinite_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if not numeric.empty else 0
    non_binary = int((~pd.to_numeric(frame["lhb_flag"], errors="coerce").dropna().isin([0, 1])).sum())
    zero_event_bad = 0
    for column in DETAIL_AMOUNT_COLUMNS:
        if column in frame.columns:
            zero_event_bad += int((frame["lhb_flag"].eq(0) & pd.to_numeric(frame[column], errors="coerce").fillna(0).ne(0)).sum())
    unavailable_zero_bad = 0
    if "lhb_detail_available" in frame.columns:
        unavailable_zero_bad = int((frame["lhb_detail_available"].eq(0) & frame["lhb_flag"].eq(0)).sum())
    if monotonic_bad or infinite_count or non_binary or zero_event_bad or unavailable_zero_bad:
        raise ValueError(
            "龙虎榜质量检查失败: "
            f"非单调股票={monotonic_bad}, 无穷值={infinite_count}, 非法lhb_flag={non_binary}, "
            f"未上榜金额非零={zero_event_bad}, 不可用日期误填零={unavailable_zero_bad}"
        )
    return {"duplicate_primary_keys": 0, "non_monotonic_symbols": monotonic_bad, "infinite_values": infinite_count}


def assert_no_forbidden_features(columns: Iterable[str], config: dict[str, Any] | None = None) -> None:
    keywords = (config or load_lhb_config()).get("forbidden_feature_keywords", [])
    blocked = [column for column in columns if any(str(keyword).lower() in str(column).lower() for keyword in keywords)]
    if blocked:
        raise ValueError(f"模型特征包含未来/目标字段，已禁止: {blocked}")


def add_next_day_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a t-close to t+1-close target for alignment tests and research."""
    data = frame.sort_values(["symbol", "trade_date"]).copy()
    data["target_return_1d"] = data.groupby("symbol", sort=False)["close"].shift(-1) / data["close"] - 1
    data["target_up_1d"] = np.where(data["target_return_1d"].notna(), (data["target_return_1d"] > 0).astype(float), np.nan)
    return data


def write_quality_report(detail_raw: pd.DataFrame, institution_raw: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    detail = normalize_lhb_detail(detail_raw) if not detail_raw.empty else detail_raw.copy()
    institution = normalize_lhb_institution(institution_raw) if not institution_raw.empty else institution_raw.copy()
    detail_agg = aggregate_lhb_detail(detail) if not detail.empty else pd.DataFrame()
    dates = sorted(set(pd.to_datetime(availability.get("trade_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna()))
    rows: list[dict[str, Any]] = []
    for trade_date in dates:
        d = detail[detail["trade_date"].eq(trade_date)] if not detail.empty else detail
        i = institution[institution["trade_date"].eq(trade_date)] if not institution.empty else institution
        a = availability[pd.to_datetime(availability["trade_date"]).dt.normalize().eq(trade_date)]
        agg_day = detail_agg[detail_agg["trade_date"].eq(trade_date)] if not detail_agg.empty else detail_agg
        ratio_values = agg_day[[column for column in RATIO_COLUMNS if column in agg_day.columns]] if not agg_day.empty else pd.DataFrame()
        rows.append(
            {
                "download_date": pd.Timestamp.now().normalize(),
                "trade_date": trade_date,
                "raw_record_count": len(d),
                "listed_stock_count": int(d["symbol"].nunique()) if not d.empty else 0,
                "institution_stock_count": int(i["symbol"].nunique()) if not i.empty else 0,
                "duplicate_primary_key_count": int(agg_day.duplicated(["symbol", "trade_date"]).sum()) if not agg_day.empty else 0,
                "missing_field_count": int(d.isna().sum().sum() + i.isna().sum().sum()),
                "abnormal_ratio_count": int(np.isinf(ratio_values.to_numpy(dtype=float, na_value=np.nan)).sum()) if not ratio_values.empty else 0,
                "data_complete": int(a["lhb_data_available"].iloc[-1]) if not a.empty else 0,
            }
        )
    report = pd.DataFrame(rows)
    if not report.empty:
        existing = _read_parquet(QUALITY_PATH)
        combined = pd.concat([existing, report], ignore_index=True, sort=False) if not existing.empty else report
        combined = combined.sort_values("download_date").drop_duplicates(["download_date", "trade_date"], keep="last")
        _atomic_parquet(combined, QUALITY_PATH)
    return report


def update_result_dict(result: LHBUpdateResult) -> dict[str, Any]:
    return asdict(result)
