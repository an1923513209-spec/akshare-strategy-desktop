"""Out-of-sample factor and prediction evaluation utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from .factor_registry import build_factor_groups


def _safe_float(value, default=np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _rank_ic(frame: pd.DataFrame, factor: str, target: str, minimum_samples: int) -> float:
    valid = frame[[factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < minimum_samples or valid[factor].nunique() < 2 or valid[target].nunique() < 2:
        return np.nan
    return _safe_float(valid[factor].corr(valid[target], method="spearman"))


def daily_rank_ic(
    dataset: pd.DataFrame,
    factors: Iterable[str],
    target: str,
    minimum_samples: int = 20,
) -> pd.DataFrame:
    """Compute daily cross-sectional Spearman IC without pooling dates."""
    rows: list[dict] = []
    data = dataset.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for factor in factors:
        for date, frame in data.groupby("date", sort=True):
            value = _rank_ic(frame, factor, target, minimum_samples)
            if np.isfinite(value):
                rows.append({"date": date, "factor_name": factor, "ic": value})
    return pd.DataFrame(rows, columns=["date", "factor_name", "ic"])


def _period_ic(ic_frame: pd.DataFrame, end_date: pd.Timestamp, months: int) -> float:
    if ic_frame.empty:
        return np.nan
    start = end_date - pd.DateOffset(months=months)
    return _safe_float(ic_frame.loc[ic_frame["date"] > start, "ic"].mean())


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return _safe_float(drawdown.min())


def _sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    if not np.isfinite(std) or std <= 1e-12:
        return np.nan
    return _safe_float(returns.mean() / std * math.sqrt(252.0))


def _event_bucket(
    frame: pd.DataFrame,
    factor: str,
    group: str,
    thresholds: Mapping[str, float] | None = None,
) -> pd.Series:
    def numeric_column(name: str, fallback=np.nan) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(fallback, index=frame.index, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce")

    values = pd.to_numeric(frame[factor], errors="coerce")
    if group in {"lhb", "lhb_institution"}:
        flag = numeric_column("lhb_flag")
        inst = numeric_column("lhb_inst_net_buy_ratio")
        result = pd.Series("未上榜", index=frame.index, dtype="object")
        result.loc[flag.isna()] = np.nan
        result.loc[(flag == 1) & (values < 0)] = "上榜且净卖出"
        positive = (flag == 1) & (values >= 0)
        threshold = (thresholds or {}).get("lhb_net_buy_high", np.nan)
        if not np.isfinite(threshold):
            threshold = values.loc[positive].quantile(0.7) if positive.any() else np.nan
        result.loc[positive] = "上榜且弱净买入"
        if np.isfinite(threshold):
            result.loc[positive & (values >= threshold)] = "上榜且强净买入"
        result.loc[(flag == 1) & (inst > 0)] = "上榜且机构净买入"
        return result
    has_news = numeric_column("has_news")
    sentiment = numeric_column("news_sentiment", values)
    count = numeric_column("news_count_3")
    result = pd.Series("无新闻", index=frame.index, dtype="object")
    result.loc[has_news.isna()] = np.nan
    result.loc[has_news == 1] = "普通新闻"
    valid = sentiment.loc[has_news == 1]
    if not valid.empty:
        low = (thresholds or {}).get("news_sentiment_low", np.nan)
        high = (thresholds or {}).get("news_sentiment_high", np.nan)
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = valid.quantile([0.2, 0.8])
        result.loc[(has_news == 1) & (sentiment >= high)] = "强正面新闻"
        result.loc[(has_news == 1) & (sentiment <= low)] = "强负面新闻"
    if count.notna().any():
        count_high = (thresholds or {}).get("news_count_high", np.nan)
        if not np.isfinite(count_high):
            count_high = count.quantile(0.8)
        result.loc[(has_news == 1) & (count > count_high)] = "新闻数量显著增加"
    return result


def factor_group_return(
    dataset: pd.DataFrame,
    factor: str,
    target: str,
    group: str,
    quantile_groups: int,
    transaction_cost: float,
    minimum_samples: int,
    event_thresholds: Mapping[str, float] | None = None,
) -> dict[str, float | str]:
    """Evaluate dense factors by quantile and sparse event factors by event buckets."""
    daily_rows: list[dict] = []
    previous_top: set[str] = set()
    previous_bottom: set[str] = set()
    event_mode = group in {"news", "lhb", "lhb_institution"}
    for date, frame in dataset.groupby("date", sort=True):
        valid = frame[["code", factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < minimum_samples:
            continue
        if event_mode:
            source = frame.loc[valid.index]
            buckets = _event_bucket(source, factor, group, event_thresholds)
            event_returns = source.assign(_bucket=buckets).groupby("_bucket", dropna=True)[target].mean()
            if group == "news":
                top = event_returns.get("强正面新闻", event_returns.get("普通新闻", np.nan))
                bottom = event_returns.get("强负面新闻", event_returns.get("无新闻", np.nan))
            else:
                top = event_returns.get("上榜且机构净买入", event_returns.get("上榜且强净买入", np.nan))
                bottom = event_returns.get("上榜且净卖出", event_returns.get("未上榜", np.nan))
            if np.isfinite(top) and np.isfinite(bottom):
                daily_rows.append({"date": date, "gross": top - bottom, "turnover": 1.0})
            continue

        unique = int(valid[factor].nunique())
        groups = min(int(quantile_groups), unique)
        if groups < 2:
            continue
        ranks = valid[factor].rank(method="first")
        labels = pd.qcut(ranks, groups, labels=False, duplicates="drop")
        low_label, high_label = labels.min(), labels.max()
        top_rows = valid.loc[labels == high_label]
        bottom_rows = valid.loc[labels == low_label]
        top_set, bottom_set = set(top_rows["code"]), set(bottom_rows["code"])
        turnover = 0.0
        if previous_top or previous_bottom:
            top_turnover = 1.0 - len(top_set & previous_top) / max(len(top_set | previous_top), 1)
            bottom_turnover = 1.0 - len(bottom_set & previous_bottom) / max(len(bottom_set | previous_bottom), 1)
            turnover = (top_turnover + bottom_turnover) / 2.0
        previous_top, previous_bottom = top_set, bottom_set
        daily_rows.append(
            {
                "date": date,
                "gross": _safe_float(top_rows[target].mean() - bottom_rows[target].mean()),
                "turnover": turnover,
            }
        )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return {
            "top_bottom_return": np.nan,
            "net_top_bottom_return": np.nan,
            "top_minus_bottom_win_rate": np.nan,
            "annualized_long_short_return": np.nan,
            "long_short_sharpe": np.nan,
            "long_short_max_drawdown": np.nan,
            "event_grouping": event_mode,
        }
    daily["net"] = daily["gross"] - transaction_cost * daily["turnover"]
    return {
        "top_bottom_return": _safe_float(daily["gross"].mean()),
        "net_top_bottom_return": _safe_float(daily["net"].mean()),
        "top_minus_bottom_win_rate": _safe_float((daily["net"] > 0).mean()),
        "annualized_long_short_return": _safe_float(daily["net"].mean() * 252.0),
        "long_short_sharpe": _sharpe(daily["net"]),
        "long_short_max_drawdown": _max_drawdown(daily["net"]),
        "event_grouping": event_mode,
    }


def evaluate_factors(
    dataset: pd.DataFrame,
    factors: Iterable[str],
    target: str,
    *,
    minimum_samples: int = 20,
    quantile_groups: int = 5,
    transaction_cost: float = 0.0016,
    event_thresholds: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return factor-quality summary and daily/stability IC history."""
    factor_list = list(dict.fromkeys(factors))
    groups = build_factor_groups(factor_list)
    group_lookup = {factor: group for group, values in groups.items() for factor in values}
    ic_history = daily_rank_ic(dataset, factor_list, target, minimum_samples)
    rows: list[dict] = []
    total = max(len(dataset), 1)
    end_date = pd.to_datetime(dataset["date"], errors="coerce").max()
    for factor in factor_list:
        numeric = pd.to_numeric(dataset[factor], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = numeric.dropna()
        q1, q3 = valid.quantile([0.25, 0.75]) if not valid.empty else (np.nan, np.nan)
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            outlier_rate = ((valid < q1 - 3 * iqr) | (valid > q3 + 3 * iqr)).mean()
        else:
            outlier_rate = 0.0 if not valid.empty else np.nan
        factor_ic = ic_history.loc[ic_history["factor_name"] == factor]
        ic_std = factor_ic["ic"].std(ddof=1)
        returns = factor_group_return(
            dataset,
            factor,
            target,
            group_lookup[factor],
            quantile_groups,
            transaction_cost,
            minimum_samples,
            event_thresholds,
        )
        valid_oos_windows = int(factor_ic["date"].dt.to_period("M").nunique()) if not factor_ic.empty else 0
        row = {
            "factor_name": factor,
            "factor_group": group_lookup[factor],
            "dtype": str(dataset[factor].dtype),
            "non_null_count": int(valid.size),
            "coverage_rate": _safe_float(valid.size / total, 0.0),
            "unique_count": int(valid.nunique()),
            "zero_rate": _safe_float((valid == 0).mean()),
            "positive_rate": _safe_float((valid > 0).mean()),
            "negative_rate": _safe_float((valid < 0).mean()),
            "mean": _safe_float(valid.mean()),
            "std": _safe_float(valid.std(ddof=1)),
            "min": _safe_float(valid.min()),
            "max": _safe_float(valid.max()),
            "outlier_rate": _safe_float(outlier_rate),
            "ic_mean": _safe_float(factor_ic["ic"].mean()),
            "ic_median": _safe_float(factor_ic["ic"].median()),
            "ic_std": _safe_float(ic_std),
            "icir": _safe_float(factor_ic["ic"].mean() / ic_std) if np.isfinite(ic_std) and ic_std > 0 else np.nan,
            "positive_ic_ratio": _safe_float((factor_ic["ic"] > 0).mean()),
            "negative_ic_ratio": _safe_float((factor_ic["ic"] < 0).mean()),
            "valid_ic_days": int(len(factor_ic)),
            "valid_oos_windows": valid_oos_windows,
            "recent_3m_ic": _period_ic(factor_ic, end_date, 3),
            "recent_6m_ic": _period_ic(factor_ic, end_date, 6),
            "recent_12m_ic": _period_ic(factor_ic, end_date, 12),
            **returns,
        }
        rows.append(row)
    quality = pd.DataFrame(rows)
    if not quality.empty:
        quality["status"] = np.select(
            [
                (quality["valid_oos_windows"] >= 6) & (quality["positive_ic_ratio"] >= 0.55) & (quality["net_top_bottom_return"] > 0),
                (quality["valid_oos_windows"] >= 6) & ((quality["ic_mean"].abs() >= 0.005) | (quality["net_top_bottom_return"] > 0)),
            ],
            ["ACTIVE", "WATCH"],
            default="INACTIVE",
        )
    stability = factor_stability_ic(dataset, factor_list, target, minimum_samples)
    daily_output = ic_history.assign(record_type="daily", segment_type="date", segment=ic_history["date"].astype(str))
    history = pd.concat([daily_output, stability], ignore_index=True, sort=False)
    return quality, history


def factor_stability_ic(
    dataset: pd.DataFrame,
    factors: Iterable[str],
    target: str,
    minimum_samples: int = 20,
) -> pd.DataFrame:
    """Summarize factor IC by year, market regime, size bucket and volatility regime."""
    data = dataset.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    date_return = data.groupby("date")[target].mean().sort_index()
    regime_signal = date_return.rolling(20, min_periods=5).mean()
    low_regime, high_regime = regime_signal.quantile([0.33, 0.67]) if regime_signal.notna().any() else (np.nan, np.nan)
    regime = pd.Series("震荡市", index=regime_signal.index, dtype="object")
    regime.loc[regime_signal <= low_regime] = "熊市"
    regime.loc[regime_signal >= high_regime] = "牛市"
    data["_market_regime"] = data["date"].map(regime)
    volatility_source = "volatility_20" if "volatility_20" in data.columns else None
    if volatility_source:
        date_volatility = data.groupby("date")[volatility_source].mean()
    else:
        date_volatility = date_return.rolling(20, min_periods=5).std()
    volatility_median = date_volatility.median()
    data["_volatility_regime"] = data["date"].map(
        pd.Series(np.where(date_volatility >= volatility_median, "高波动期", "低波动期"), index=date_volatility.index)
    )
    size_column = next((name for name in ("float_market_cap", "market_cap", "total_market_cap", "size") if name in data.columns), None)
    if size_column:
        size_rank = data.groupby("date")[size_column].rank(pct=True)
        data["_size_bucket"] = pd.cut(size_rank, [-np.inf, 1 / 3, 2 / 3, np.inf], labels=["小盘股", "中盘股", "大盘股"])
    rows: list[dict] = []
    segment_specs = [("year", data["date"].dt.year.astype(str)), ("market_regime", data["_market_regime"]), ("volatility_regime", data["_volatility_regime"])]
    if size_column:
        segment_specs.append(("size_bucket", data["_size_bucket"].astype("object")))
    for segment_type, labels in segment_specs:
        for segment in pd.Series(labels).dropna().unique():
            subset = data.loc[pd.Series(labels, index=data.index) == segment]
            daily = daily_rank_ic(subset, factors, target, minimum_samples)
            for factor, values in daily.groupby("factor_name"):
                rows.append(
                    {
                        "date": pd.NaT,
                        "factor_name": factor,
                        "ic": np.nan,
                        "record_type": "stability",
                        "segment_type": segment_type,
                        "segment": str(segment),
                        "ic_mean": values["ic"].mean(),
                        "ic_median": values["ic"].median(),
                        "ic_std": values["ic"].std(ddof=1),
                        "positive_ic_ratio": (values["ic"] > 0).mean(),
                        "valid_ic_days": len(values),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    probability_col: str = "probability_up",
    prediction_col: str = "predicted_return",
    target_col: str = "next_open_to_next_open_return",
    transaction_cost: float = 0.0016,
) -> dict[str, float]:
    """Evaluate untouched OOS predictions, including simple cross-sectional net returns."""
    data = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[probability_col, prediction_col, target_col]).copy()
    if data.empty:
        return {key: np.nan for key in ("auc", "pr_auc", "brier", "log_loss", "mae", "rmse", "rank_ic", "net_return", "sharpe", "max_drawdown", "turnover")}
    y = (data[target_col] > 0).astype(int)
    probability = data[probability_col].clip(1e-6, 1 - 1e-6)
    metrics = {
        "auc": _safe_float(roc_auc_score(y, probability)) if y.nunique() > 1 else np.nan,
        "pr_auc": _safe_float(average_precision_score(y, probability)) if y.nunique() > 1 else np.nan,
        "brier": _safe_float(brier_score_loss(y, probability)),
        "log_loss": _safe_float(log_loss(y, probability, labels=[0, 1])),
        "mae": _safe_float(mean_absolute_error(data[target_col], data[prediction_col])),
        "rmse": _safe_float(mean_squared_error(data[target_col], data[prediction_col]) ** 0.5),
        "rank_ic": _safe_float(data[prediction_col].corr(data[target_col], method="spearman")),
    }
    daily_returns: list[float] = []
    daily_gross_returns: list[float] = []
    daily_top_bottom: list[float] = []
    daily_ics: list[float] = []
    turnovers: list[float] = []
    previous: set[str] = set()
    for _date, group in data.groupby("date", sort=True):
        count = max(1, int(math.ceil(len(group) * 0.2)))
        selected = group.nlargest(count, prediction_col)
        bottom = group.nsmallest(count, prediction_col)
        current = set(selected["code"].astype(str))
        turnover = 1.0 if not previous else 1.0 - len(current & previous) / max(len(current | previous), 1)
        previous = current
        gross_return = _safe_float(selected[target_col].mean())
        daily_gross_returns.append(gross_return)
        daily_top_bottom.append(gross_return - _safe_float(bottom[target_col].mean()))
        if len(group) >= 3:
            daily_ics.append(_safe_float(group[prediction_col].corr(group[target_col], method="spearman")))
        daily_returns.append(gross_return - transaction_cost * turnover)
        turnovers.append(turnover)
    net = pd.Series(daily_returns, dtype=float)
    metrics.update(
        {
            "net_return": _safe_float(net.mean()),
            "gross_return": _safe_float(np.mean(daily_gross_returns)),
            "top_bottom_return": _safe_float(np.mean(daily_top_bottom)),
            "icir": (
                _safe_float(np.mean(daily_ics) / np.std(daily_ics, ddof=1))
                if len(daily_ics) > 1 and np.std(daily_ics, ddof=1) > 0
                else np.nan
            ),
            "sharpe": _sharpe(net),
            "max_drawdown": _max_drawdown(net),
            "turnover": _safe_float(np.mean(turnovers)),
        }
    )
    return metrics
