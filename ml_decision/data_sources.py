"""Free external-factor adapters for A-share decision features.

The adapters use AKShare free endpoints when available. They never raise for a
single source failure; instead they return an empty frame and a source note so
the decision engine can continue with technical factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from .lhb_data import load_lhb_factors_for_market


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_CACHE_DIR = PROJECT_ROOT / "cache" / "ml_external"
FUND_FLOW_TTL_SECONDS = 6 * 60 * 60
NEWS_TTL_SECONDS = 6 * 60 * 60
INSTITUTION_TTL_SECONDS = 24 * 60 * 60


@dataclass(slots=True)
class SourceNote:
    """A short status record for one external data source."""

    source: str
    status: str
    detail: str


def _market_for_code(code: str) -> str:
    if code.startswith(("4", "8", "9")):
        return "bj"
    if code.startswith(("5", "6", "7")):
        return "sh"
    return "sz"


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _empty(code: str) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "code": pd.Series(dtype="object")})


def _safe_cache_key(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value)


def _cache_path(source: str, key: str) -> Path:
    return EXTERNAL_CACHE_DIR / source / f"{_safe_cache_key(key)}.pkl"


def _cache_age_seconds(path: Path) -> float:
    return max(0.0, time.time() - path.stat().st_mtime)


def _cache_age_text(path: Path) -> str:
    minutes = _cache_age_seconds(path) / 60.0
    if minutes < 90:
        return f"{minutes:.0f} minutes old"
    return f"{minutes / 60.0:.1f} hours old"


def _read_cached_frame(source: str, key: str, ttl_seconds: int, allow_stale: bool = False) -> tuple[pd.DataFrame, str] | None:
    path = _cache_path(source, key)
    if not path.exists():
        return None
    if not allow_stale and _cache_age_seconds(path) > ttl_seconds:
        return None
    try:
        frame = pd.read_pickle(path)
    except Exception:
        return None
    if not isinstance(frame, pd.DataFrame):
        return None
    if source == "fund_flow":
        frame = _repair_fund_flow_units(frame)
    return frame.copy(), _cache_age_text(path)


def _write_cached_frame(source: str, key: str, frame: pd.DataFrame) -> None:
    try:
        path = _cache_path(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        cached_frame = _repair_fund_flow_units(frame) if source == "fund_flow" else frame
        cached_frame.to_pickle(path)
    except Exception:
        return


def _get_url(url: str, **kwargs: Any) -> requests.Response:
    session = requests.Session()
    session.trust_env = False
    kwargs.setdefault("timeout", 12)
    return session.get(url, **kwargs)


def _stock_market_id(code: str) -> int:
    return 1 if code.startswith(("5", "6", "7")) else 0


def _clean_html(text: Any) -> str:
    value = "" if pd.isna(text) else str(text)
    value = re.sub(r"</?em>", "", value)
    value = re.sub(r"<[^>]+>", "", value)
    return value.replace("\\u3000", "").replace("\u3000", "").replace("\r\n", " ").strip()


def _normalize_fund_flow_raw(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    data = pd.DataFrame()
    data["date"] = pd.to_datetime(raw["日期"], errors="coerce")
    data["code"] = code
    data["main_net_ratio"] = _num(raw, "主力净流入-净占比") / 100.0
    large_ratio = _num(raw, "大单净流入-净占比") / 100.0
    huge_ratio = _num(raw, "超大单净流入-净占比") / 100.0
    data["large_net_ratio"] = large_ratio + huge_ratio.fillna(0)
    data["main_net_amount"] = _num(raw, "主力净流入-净额")
    data["large_net_amount"] = _num(raw, "大单净流入-净额") + _num(raw, "超大单净流入-净额").fillna(0)
    data = data.dropna(subset=["date"]).sort_values("date")
    for column in ("main_net_ratio", "large_net_ratio"):
        data[f"{column}_3"] = data[column].rolling(3).mean()
        data[f"{column}_5"] = data[column].rolling(5).mean()
    data["positive_main_flow_days_5"] = (data["main_net_ratio"] > 0).rolling(5).sum()
    data["main_flow_acceleration"] = data["main_net_ratio_3"] - data["main_net_ratio_5"]
    data["flow_price_divergence_5"] = np.nan
    return data


def _repair_fund_flow_units(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    ratio_columns = [
        "main_net_ratio",
        "main_net_ratio_3",
        "main_net_ratio_5",
        "main_net_ratio_10",
        "large_net_ratio",
        "large_net_ratio_3",
        "large_net_ratio_5",
    ]
    for column in ratio_columns:
        if column not in data.columns:
            continue
        series = pd.to_numeric(data[column], errors="coerce")
        needs_scale = series.abs() > 1.0
        data[column] = series.where(~needs_scale, series / 100.0)
    if "main_net_ratio_3" in data.columns and "main_net_ratio_5" in data.columns:
        data["main_flow_acceleration"] = pd.to_numeric(data["main_net_ratio_3"], errors="coerce") - pd.to_numeric(
            data["main_net_ratio_5"], errors="coerce"
        )
    return data


def _fetch_fund_flow_direct_history(code: str) -> pd.DataFrame:
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "lmt": "0",
        "klt": "101",
        "secid": f"{_stock_market_id(code)}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
        "_": int(time.time() * 1000),
    }
    response = _get_url(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
    data_json = response.json()
    content_list = (data_json.get("data") or {}).get("klines") or []
    if not content_list:
        return _empty(code)
    raw = pd.DataFrame([item.split(",") for item in content_list])
    raw.columns = [
        "日期",
        "主力净流入-净额",
        "小单净流入-净额",
        "中单净流入-净额",
        "大单净流入-净额",
        "超大单净流入-净额",
        "主力净流入-净占比",
        "小单净流入-净占比",
        "中单净流入-净占比",
        "大单净流入-净占比",
        "超大单净流入-净占比",
        "收盘价",
        "涨跌幅",
        "_1",
        "_2",
    ]
    return _normalize_fund_flow_raw(raw, code)


def _fetch_fund_flow_direct_snapshot(code: str) -> pd.DataFrame:
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": f"{_stock_market_id(code)}.{code}",
        "fields": "f57,f58,f62,f66,f69,f72,f75,f78,f81,f84,f87,f164,f165,f174,f175,f184",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "_": int(time.time() * 1000),
    }
    response = _get_url(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
    item = response.json().get("data") or {}
    if not item:
        return _empty(code)
    today = pd.Timestamp(datetime.now().date())
    main_ratio = pd.to_numeric(pd.Series([item.get("f184")]), errors="coerce").iloc[0] / 100.0
    main_ratio_5 = pd.to_numeric(pd.Series([item.get("f165")]), errors="coerce").iloc[0] / 10000.0
    main_ratio_10 = pd.to_numeric(pd.Series([item.get("f175")]), errors="coerce").iloc[0] / 10000.0
    huge_amount = pd.to_numeric(pd.Series([item.get("f84")]), errors="coerce").iloc[0]
    data = pd.DataFrame({"date": [today], "code": [code]})
    data["main_net_ratio"] = main_ratio
    data["main_net_ratio_3"] = np.nanmean([main_ratio, main_ratio_5])
    data["main_net_ratio_5"] = main_ratio_5
    data["large_net_ratio"] = np.nan
    data["large_net_ratio_3"] = np.nan
    data["large_net_ratio_5"] = np.nan
    data["main_net_amount"] = pd.to_numeric(pd.Series([item.get("f62")]), errors="coerce").iloc[0]
    data["large_net_amount"] = huge_amount
    data["positive_main_flow_days_5"] = 1.0 if np.isfinite(main_ratio) and main_ratio > 0 else 0.0
    data["main_flow_acceleration"] = main_ratio - main_ratio_5 if np.isfinite(main_ratio) and np.isfinite(main_ratio_5) else np.nan
    data["flow_price_divergence_5"] = np.nan
    data["main_net_ratio_10"] = main_ratio_10
    return data


def fetch_fund_flow_features(code: str, force_refresh: bool = False) -> tuple[pd.DataFrame, SourceNote]:
    """Fetch Eastmoney individual fund-flow factors for the latest ~100 sessions."""
    cached = None if force_refresh else _read_cached_frame("fund_flow", code, FUND_FLOW_TTL_SECONDS, allow_stale=False)
    if cached is not None:
        frame, age = cached
        return frame, SourceNote("stock_individual_fund_flow", "cache", f"{len(frame)} rows; {age}")
    try:
        data = _fetch_fund_flow_direct_history(code)
        if not data.empty:
            _write_cached_frame("fund_flow", code, data)
            return data, SourceNote("stock_individual_fund_flow_direct", "ok", f"{len(data)} rows")
    except Exception as direct_exc:
        direct_error = direct_exc
    else:
        direct_error = None
    try:
        data = _fetch_fund_flow_direct_snapshot(code)
        if not data.empty:
            _write_cached_frame("fund_flow", code, data)
            return data, SourceNote("stock_fund_flow_snapshot", "ok", f"{len(data)} snapshot row; history failed: {direct_error}")
    except Exception as snapshot_exc:
        snapshot_error = snapshot_exc
    else:
        snapshot_error = None
    stale = _read_cached_frame("fund_flow", code, FUND_FLOW_TTL_SECONDS, allow_stale=True)
    if stale is not None:
        frame, age = stale
        return frame, SourceNote("stock_individual_fund_flow", "stale_cache", f"{len(frame)} rows; {age}; direct={direct_error}; snapshot={snapshot_error}")
    return _empty(code), SourceNote("stock_individual_fund_flow", "failed", f"direct={direct_error}; snapshot={snapshot_error}")


POSITIVE_WORDS = ("增长", "大增", "预增", "中标", "突破", "创新高", "回购", "增持", "签订", "盈利", "涨停")
NEGATIVE_WORDS = ("亏损", "下滑", "减持", "处罚", "立案", "问询", "风险", "跌停", "终止", "暴雷", "退市")


def _simple_sentiment(text: str) -> float:
    score = 0
    for word in POSITIVE_WORDS:
        if word in text:
            score += 1
    for word in NEGATIVE_WORDS:
        if word in text:
            score -= 1
    return float(np.tanh(score / 2.0))


def _news_daily_from_rows(code: str, rows: list[dict[str, Any]], title_key: str, content_key: str, date_key: str) -> pd.DataFrame:
    if not rows:
        return _empty(code)
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data.get(date_key), errors="coerce").dt.normalize()
    title = data.get(title_key, pd.Series("", index=data.index)).map(_clean_html)
    content = data.get(content_key, pd.Series("", index=data.index)).map(_clean_html)
    data["sentiment"] = (title + " " + content).map(_simple_sentiment)
    daily = data.dropna(subset=["date"]).groupby("date").agg(
        news_sentiment=("sentiment", "mean"),
        news_count=("sentiment", "size"),
    )
    if daily.empty:
        return _empty(code)
    daily = daily.sort_index()
    out = pd.DataFrame({"date": daily.index, "code": code})
    out["has_news"] = 1.0
    out["news_sentiment"] = daily["news_sentiment"].values
    out["news_count_3"] = daily["news_count"].rolling(3, min_periods=1).sum().values
    out["news_count_5"] = daily["news_count"].rolling(5, min_periods=1).sum().values
    out["news_sentiment_mean_3"] = daily["news_sentiment"].rolling(3, min_periods=1).mean().values
    out["news_sentiment_mean_5"] = daily["news_sentiment"].rolling(5, min_periods=1).mean().values
    out["news_sentiment_change_3"] = out["news_sentiment_mean_3"].diff()
    out["weighted_news_sentiment"] = out["news_sentiment"] * np.log1p(out["news_count_3"])
    return out


def _fetch_news_direct_em(code: str) -> pd.DataFrame:
    callback = f"jQuery351{int(time.time() * 1000)}"
    inner_param = {
        "uid": "",
        "keyword": code,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": 50,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    params = {"cb": callback, "param": json.dumps(inner_param, ensure_ascii=False), "_": int(time.time() * 1000)}
    response = _get_url(
        "https://search-api-web.eastmoney.com/search/jsonp",
        params=params,
        headers={"User-Agent": "Mozilla/5.0", "Referer": f"https://so.eastmoney.com/news/s?keyword={code}"},
    )
    text = response.text.strip()
    payload = text[text.find("(") + 1 : text.rfind(")")] if "(" in text and ")" in text else text
    data_json = json.loads(payload)
    rows = (data_json.get("result") or {}).get("cmsArticleWebOld") or []
    return _news_daily_from_rows(code, rows, "title", "content", "date")


def _fetch_notice_news(code: str) -> pd.DataFrame:
    try:
        end = datetime.now().date()
        begin = end - timedelta(days=30)
        params = {
            "sr": "-1",
            "page_size": "100",
            "page_index": "1",
            "ann_type": "A",
            "client_source": "web",
            "f_node": "0",
            "s_node": "0",
            "stock_list": code,
            "begin_time": begin.strftime("%Y-%m-%d"),
            "end_time": end.strftime("%Y-%m-%d"),
        }
        response = _get_url(
            "https://np-anotice-stock.eastmoney.com/api/security/ann",
            params=params,
            headers={"User-Agent": "Mozilla/5.0", "Referer": f"https://data.eastmoney.com/notices/stock/{code}.html"},
        )
        rows_json = ((response.json().get("data") or {}).get("list")) or []
    except Exception:
        return _empty(code)
    if not rows_json:
        return _empty(code)
    rows = [{"date": item.get("notice_date") or item.get("display_time"), "title": item.get("title", "")} for item in rows_json]
    return _news_daily_from_rows(code, rows, "title", "title", "date")


def fetch_news_features(code: str, force_refresh: bool = False) -> tuple[pd.DataFrame, SourceNote]:
    """Fetch recent Eastmoney news and convert it to daily sentiment counts."""
    cached = None if force_refresh else _read_cached_frame("news", code, NEWS_TTL_SECONDS, allow_stale=False)
    if cached is not None:
        frame, age = cached
        return frame, SourceNote("stock_news_em", "cache", f"{len(frame)} daily rows; {age}")
    try:
        out = _fetch_news_direct_em(code)
        if not out.empty:
            _write_cached_frame("news", code, out)
            return out, SourceNote("stock_news_em_direct", "ok", f"{len(out)} daily rows")
    except Exception as direct_exc:
        direct_error = direct_exc
    else:
        direct_error = None
    out = _fetch_notice_news(code)
    if not out.empty:
        _write_cached_frame("news", code, out)
        return out, SourceNote("stock_individual_notice_report", "ok", f"{len(out)} announcement daily rows; eastmoney news direct={direct_error}")
    stale = _read_cached_frame("news", code, NEWS_TTL_SECONDS, allow_stale=True)
    if stale is not None:
        frame, age = stale
        return frame, SourceNote("stock_news_em", "stale_cache", f"{len(frame)} daily rows; {age}; direct={direct_error}")
    return _empty(code), SourceNote("stock_news_em", "failed", f"direct={direct_error}; notice=empty")


def fetch_institution_features(codes: list[str], force_refresh: bool = False) -> tuple[pd.DataFrame, list[SourceNote]]:
    """Fetch weak institution activity proxies from free LHB and holding data."""
    notes: list[SourceNote] = []
    frames: list[pd.DataFrame] = []
    code_set = {str(code).zfill(6) for code in codes}
    cache_key = "_".join(sorted(code_set)) or "empty"
    cached = None if force_refresh else _read_cached_frame("institution", cache_key, INSTITUTION_TTL_SECONDS, allow_stale=False)
    if cached is not None:
        frame, age = cached
        return frame, [SourceNote("institution_features", "cache", f"{len(frame)} rows; {age}")]
    try:
        import akshare as ak

        lhb = ak.stock_lhb_jgstatistic_em(symbol="近一月")
        if lhb is not None and not lhb.empty and "代码" in lhb.columns:
            part = lhb.copy()
            part["code"] = part["代码"].astype(str).str.zfill(6)
            part = part[part["code"].isin(code_set)]
            if not part.empty:
                out = pd.DataFrame({"date": pd.Timestamp(datetime.now().date()), "code": part["code"]})
                out["institution_activity"] = _num(part, "上榜次数").values
                out["institution_activity_ma_5"] = _num(part, "机构买入次数").values - _num(part, "机构卖出次数").values
                out["institution_net_buy_amount"] = _num(part, "机构净买额").values
                frames.append(out)
            notes.append(SourceNote("stock_lhb_jgstatistic_em", "ok", f"{len(part)} matched rows"))
    except Exception as exc:  # pragma: no cover - network/source dependent
        notes.append(SourceNote("stock_lhb_jgstatistic_em", "failed", str(exc)))

    try:
        import akshare as ak

        quarter = _latest_report_quarter()
        hold = ak.stock_institute_hold(symbol=quarter)
        if hold is not None and not hold.empty and "证券代码" in hold.columns:
            part = hold.copy()
            part["code"] = part["证券代码"].astype(str).str.zfill(6)
            part = part[part["code"].isin(code_set)]
            if not part.empty:
                out = pd.DataFrame({"date": pd.Timestamp(datetime.now().date()), "code": part["code"]})
                out["institution_hold_count"] = _num(part, "机构数").values
                out["institution_hold_ratio"] = _num(part, "持股比例").values / 100.0
                out["institution_hold_ratio_change"] = _num(part, "持股比例增幅").values / 100.0
                frames.append(out)
            notes.append(SourceNote("stock_institute_hold", "ok", f"{len(part)} matched rows for {quarter}"))
    except Exception as exc:  # pragma: no cover - network/source dependent
        notes.append(SourceNote("stock_institute_hold", "failed", str(exc)))

    if not frames:
        stale = _read_cached_frame("institution", cache_key, INSTITUTION_TTL_SECONDS, allow_stale=True)
        if stale is not None:
            frame, age = stale
            notes.append(SourceNote("institution_features", "stale_cache", f"{len(frame)} rows; {age}; no fresh rows"))
            return frame, notes
        return _empty(""), notes
    merged = pd.concat(frames, ignore_index=True).sort_values(["code", "date"])
    merged = merged.groupby(["date", "code"], as_index=False).last()
    _write_cached_frame("institution", cache_key, merged)
    return merged, notes


def _latest_report_quarter(today: datetime | None = None) -> str:
    now = today or datetime.now()
    year = now.year
    month = now.month
    if month <= 4:
        return f"{year - 1}4"
    if month <= 8:
        return f"{year}1"
    if month <= 10:
        return f"{year}2"
    return f"{year}3"


def fetch_external_factor_frame(
    codes: list[str],
    force_refresh: bool = False,
    market_df: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, list[SourceNote]]:
    """Fetch all configured free external factors for a list of A-share codes."""
    notes: list[SourceNote] = []
    frames: list[pd.DataFrame] = []
    market_latest: dict[str, pd.Timestamp] = {}
    if market_df is not None and not market_df.empty:
        normalized_market = market_df.copy()
        normalized_market["date"] = pd.to_datetime(normalized_market["date"], errors="coerce").dt.normalize()
        normalized_market["code"] = normalized_market["code"].astype(str).str.zfill(6)
        market_latest = normalized_market.groupby("code")["date"].max().dropna().to_dict()
    availability_rows: list[dict[str, Any]] = []

    def fetch_one(code: str) -> tuple[list[pd.DataFrame], list[SourceNote], dict[str, Any] | None]:
        code = str(code).zfill(6)
        local_frames: list[pd.DataFrame] = []
        local_notes: list[SourceNote] = []
        fund, note = fetch_fund_flow_features(code, force_refresh=force_refresh)
        local_notes.append(note)
        if not fund.empty:
            local_frames.append(fund)
        news, note = fetch_news_features(code, force_refresh=force_refresh)
        local_notes.append(note)
        if not news.empty:
            local_frames.append(news)
        latest_date = market_latest.get(code)
        availability = None
        if latest_date is not None:
            fund_ok = local_notes[-2].status in {"ok", "cache", "stale_cache"}
            news_ok = local_notes[-1].status in {"ok", "cache", "stale_cache"}
            has_news_today = bool(
                not news.empty
                and (pd.to_datetime(news["date"], errors="coerce").dt.normalize() == latest_date).any()
            )
            availability = {
                "date": latest_date,
                "code": code,
                "fund_flow_data_available": float(fund_ok),
                "news_data_available": float(news_ok),
                "has_news": float(has_news_today) if news_ok else np.nan,
            }
        return local_frames, local_notes, availability

    normalized_codes = list(dict.fromkeys(str(code).zfill(6) for code in codes))
    workers = max_workers or int(os.environ.get("ML_EXTERNAL_FETCH_WORKERS", "6") or "6")
    workers = min(max(int(workers), 1), max(len(normalized_codes), 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ml-external") as executor:
        futures = {executor.submit(fetch_one, code): code for code in normalized_codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                local_frames, local_notes, availability = future.result()
                frames.extend(local_frames)
                notes.extend(local_notes)
                if availability is not None:
                    availability_rows.append(availability)
            except Exception as exc:
                notes.append(SourceNote(f"external:{code}", "failed", f"{type(exc).__name__}: {exc}"))
    institution, institution_notes = fetch_institution_features(codes, force_refresh=force_refresh)
    notes.extend(institution_notes)
    if not institution.empty:
        frames.append(institution)
    institution_ok = any(
        note.status in {"ok", "cache", "stale_cache"}
        for note in institution_notes
    )
    for row in availability_rows:
        row["institution_data_available"] = float(institution_ok)
    if availability_rows:
        frames.append(pd.DataFrame(availability_rows))
    if market_df is not None and not market_df.empty:
        try:
            lhb, detail = load_lhb_factors_for_market(market_df, force_refresh=force_refresh)
            notes.append(SourceNote("stock_lhb_detail_em+stock_lhb_jgmmtj_em", "ok" if not lhb.empty else "missing_cache", detail))
            if not lhb.empty:
                frames.append(lhb)
        except Exception as exc:  # pragma: no cover - network/source dependent
            notes.append(SourceNote("stock_lhb_detail_em+stock_lhb_jgmmtj_em", "failed", str(exc)))
    if not frames:
        return _empty(""), notes
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged["code"] = merged["code"].astype(str).str.zfill(6)
    return merged.groupby(["date", "code"], as_index=False).last(), notes


def merge_external_factors(market_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join external factors into the market long table."""
    if external_df.empty:
        return market_df
    left = market_df.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    left["code"] = left["code"].astype(str).str.zfill(6)
    right = external_df.copy()
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()
    right["code"] = right["code"].astype(str).str.zfill(6)
    return left.merge(right, on=["date", "code"], how="left")
