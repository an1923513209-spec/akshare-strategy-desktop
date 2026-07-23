"""Persistent, outcome-backed production model drift monitoring."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _history_path(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "cache" / "ml_prediction_history.parquet"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _realized_open_returns(market_df: pd.DataFrame) -> pd.DataFrame:
    market = market_df.loc[:, ["code", "date", "open"]].copy()
    market["code"] = market["code"].astype(str).str.zfill(6)
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["open"] = pd.to_numeric(market["open"], errors="coerce")
    market = market.dropna(subset=["date"]).sort_values(["code", "date"], kind="stable")
    grouped = market.groupby("code", sort=False)["open"]
    market["realized_return"] = grouped.shift(-2) / grouped.shift(-1) - 1.0
    return market.loc[:, ["code", "date", "realized_return"]]


def update_prediction_history(
    project_root: str | Path,
    predictions: pd.DataFrame,
    market_df: pd.DataFrame,
    model_version: str,
) -> pd.DataFrame:
    """Append one batch and backfill outcomes only when future opens are now known."""
    path = _history_path(project_root)
    if path.exists():
        try:
            history = pd.read_parquet(path)
        except Exception:
            history = pd.DataFrame()
    else:
        history = pd.DataFrame()

    outcomes = _realized_open_returns(market_df)
    if not history.empty:
        history["code"] = history["code"].astype(str).str.zfill(6)
        history["data_date"] = pd.to_datetime(history["data_date"], errors="coerce").dt.normalize()
        merged = history.merge(
            outcomes.rename(columns={"date": "data_date", "realized_return": "_resolved_return"}),
            on=["code", "data_date"],
            how="left",
        )
        existing = (
            pd.to_numeric(merged["realized_return"], errors="coerce")
            if "realized_return" in merged.columns
            else pd.Series(np.nan, index=merged.index, dtype=float)
        )
        resolved = pd.to_numeric(merged.pop("_resolved_return"), errors="coerce")
        merged["realized_return"] = existing.where(existing.notna(), resolved)
        history = merged

    if not predictions.empty:
        current = pd.DataFrame(
            {
                "prediction_time": datetime.now().isoformat(timespec="seconds"),
                "model_version": str(model_version),
                "data_date": pd.to_datetime(predictions["date"], errors="coerce").dt.normalize(),
                "code": predictions["code"].astype(str).str.zfill(6),
                "probability_up": pd.to_numeric(predictions["probability_up"], errors="coerce"),
                "expected_return": pd.to_numeric(
                    predictions["expected_open_to_open_return"], errors="coerce"
                ),
                "required_missing_rate": 1.0
                - pd.to_numeric(predictions["data_completeness_score"], errors="coerce"),
            }
        )
        for column in predictions.columns:
            if str(column).startswith("feature__"):
                current[str(column)] = pd.to_numeric(predictions[column], errors="coerce")
        current = current.merge(
            outcomes.rename(columns={"date": "data_date"}),
            on=["code", "data_date"],
            how="left",
        )
        history = pd.concat([history, current], ignore_index=True, sort=False)

    if history.empty:
        return history
    history = history.dropna(subset=["data_date", "code"])
    history = history.drop_duplicates(["model_version", "data_date", "code"], keep="last")
    history = history.sort_values(["data_date", "code"], kind="stable").reset_index(drop=True)
    _atomic_parquet(history, path)
    return history


def _daily_rank_ic(realized: pd.DataFrame) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in realized.groupby("data_date", sort=True):
        valid = group.dropna(subset=["expected_return", "realized_return"])
        if len(valid) < 3:
            continue
        values[pd.Timestamp(date)] = float(
            valid["expected_return"].corr(valid["realized_return"], method="spearman")
        )
    return pd.Series(values, dtype=float).dropna().sort_index()


def _prediction_psi(reference: pd.Series, recent: pd.Series) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    recent = pd.to_numeric(recent, errors="coerce").dropna()
    if len(reference) < 30 or len(recent) < 10:
        return np.nan
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    new_counts = np.histogram(recent, bins=edges)[0].astype(float)
    ref_ratio = np.clip(ref_counts / max(ref_counts.sum(), 1.0), 1e-6, None)
    new_ratio = np.clip(new_counts / max(new_counts.sum(), 1.0), 1e-6, None)
    return float(np.sum((new_ratio - ref_ratio) * np.log(new_ratio / ref_ratio)))


def assess_model_drift(
    history: pd.DataFrame,
    model_version: str,
    thresholds: Mapping[str, Any] | None = None,
    production_training_end: Any = None,
) -> dict[str, Any]:
    """Assess only outcome-backed metrics; unavailable metrics remain explicit NaN."""
    cfg = dict(thresholds or {})
    if history.empty or "model_version" not in history.columns:
        version_rows = pd.DataFrame()
    else:
        version_rows = history.loc[
            history["model_version"].astype(str).eq(str(model_version))
        ].copy()
    if version_rows.empty:
        return {
            "status": "insufficient_history",
            "review_required": False,
            "severe": False,
            "realized_samples": 0,
            "latest_prediction_date": None,
            "metrics": {},
            "reasons": ["尚无该模型版本的预测历史"],
        }
    required_columns = {
        "realized_return",
        "probability_up",
        "expected_return",
        "required_missing_rate",
    }
    missing_columns = sorted(required_columns.difference(version_rows.columns))
    if missing_columns:
        return {
            "status": "insufficient_history",
            "review_required": False,
            "severe": False,
            "realized_samples": 0,
            "latest_prediction_date": None,
            "metrics": {},
            "reasons": [f"历史缓存缺少字段: {', '.join(missing_columns)}"],
        }
    version_rows["data_date"] = pd.to_datetime(version_rows.get("data_date"), errors="coerce")
    version_rows = version_rows.sort_values("data_date", kind="stable")
    realized = version_rows.dropna(subset=["realized_return", "probability_up", "expected_return"])
    minimum_rows = int(cfg.get("minimum_realized_rows", 30))
    report: dict[str, Any] = {
        "status": "insufficient_history",
        "review_required": False,
        "severe": False,
        "realized_samples": int(len(realized)),
        "latest_prediction_date": (
            str(version_rows["data_date"].max().date()) if not version_rows.empty else None
        ),
        "metrics": {},
        "reasons": [],
    }
    if len(realized) < minimum_rows:
        report["reasons"] = [f"已实现样本 {len(realized)} < {minimum_rows}"]
        return report

    daily_ic = _daily_rank_ic(realized)
    last_dates = list(realized["data_date"].dropna().drop_duplicates().sort_values())
    recent_dates = set(last_dates[-20:])
    prior_dates = set(last_dates[-80:-20])
    recent = realized.loc[realized["data_date"].isin(recent_dates)]
    prior = realized.loc[realized["data_date"].isin(prior_dates)]
    actual_up = (recent["realized_return"] > 0).astype(float)
    feature_psi_values = {
        column.removeprefix("feature__"): _prediction_psi(prior[column], recent[column])
        for column in version_rows.columns
        if column.startswith("feature__")
    }
    finite_feature_psi = [value for value in feature_psi_values.values() if np.isfinite(value)]
    metrics = {
        "rank_ic_20d": float(daily_ic.tail(20).mean()) if len(daily_ic) else np.nan,
        "rank_ic_60d": float(daily_ic.tail(60).mean()) if len(daily_ic) else np.nan,
        "calibration_bias_20d": float((recent["probability_up"] - actual_up).mean()),
        "prediction_psi": _prediction_psi(prior["probability_up"], recent["probability_up"]),
        "missing_rate_change": float(
            recent["required_missing_rate"].mean() - prior["required_missing_rate"].mean()
        ) if not prior.empty else np.nan,
        "return_prediction_bias": float(
            (recent["expected_return"] - recent["realized_return"]).mean()
        ),
        "feature_psi": float(np.mean(finite_feature_psi)) if finite_feature_psi else np.nan,
        "feature_psi_max": float(np.max(finite_feature_psi)) if finite_feature_psi else np.nan,
        "feature_psi_count": len(finite_feature_psi),
    }
    training_end = pd.to_datetime(production_training_end, errors="coerce")
    metrics["trading_days_since_training"] = (
        int(version_rows.loc[version_rows["data_date"].gt(training_end), "data_date"].nunique())
        if pd.notna(training_end)
        else np.nan
    )
    consecutive = 0
    for value in reversed(daily_ic.tolist()):
        if value > 0:
            break
        consecutive += 1
    metrics["consecutive_nonpositive_ic_days"] = consecutive
    report["metrics"] = metrics
    report["feature_psi"] = feature_psi_values

    review_reasons: list[str] = []
    severe_reasons: list[str] = []
    checks = (
        ("rank_ic_20d", "low", -0.02, -0.08, "近20日Rank IC"),
        ("calibration_bias_20d", "abs", 0.12, 0.20, "概率校准偏差"),
        ("prediction_psi", "high", 0.25, 0.50, "预测分布PSI"),
        ("feature_psi", "high", 0.20, 0.40, "特征PSI"),
        ("missing_rate_change", "high", 0.15, 0.30, "必需数据缺失率变化"),
        ("return_prediction_bias", "abs", 0.03, 0.06, "预测收益偏差"),
        ("consecutive_nonpositive_ic_days", "high", 5, 10, "连续失效交易日"),
    )
    for key, mode, review_default, severe_default, label in checks:
        value = metrics.get(key, np.nan)
        if not np.isfinite(value):
            continue
        review_limit = float(cfg.get(f"review_{key}", review_default))
        severe_limit = float(cfg.get(f"severe_{key}", severe_default))
        comparable = abs(value) if mode == "abs" else value
        if mode == "low":
            if comparable <= severe_limit:
                severe_reasons.append(label)
            elif comparable <= review_limit:
                review_reasons.append(label)
        elif comparable >= severe_limit:
            severe_reasons.append(label)
        elif comparable >= review_limit:
            review_reasons.append(label)
    if severe_reasons:
        report.update(status="severe", review_required=True, severe=True, reasons=severe_reasons)
    elif review_reasons:
        report.update(status="review", review_required=True, reasons=review_reasons)
    else:
        report.update(status="ok", reasons=[])
    return report
