"""Dynamic factor-group weights and event-aware ensemble gating."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_SUBMODEL_GROUPS = (
    "technical",
    "liquidity",
    "fund_flow",
    "institution",
    "news",
    "lhb",
    "lhb_institution",
    "fundamental",
)


def _normalize_positive(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    low, high = clean.min(), clean.max()
    if not np.isfinite(high - low) or high - low <= 1e-12:
        return pd.Series(np.where(clean.notna(), np.maximum(clean, 0.0), 0.0), index=values.index, dtype=float)
    return ((clean - low) / (high - low)).fillna(0.0).clip(lower=0.0)


def _project_bounded_simplex(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Project onto sum(x)=1 with per-coordinate bounds using bisection."""
    if lower.sum() > 1.0 + 1e-10 or upper.sum() < 1.0 - 1e-10:
        raise ValueError("Weight bounds cannot sum to one")
    low_lambda, high_lambda = -2.0, 2.0
    for _ in range(100):
        middle = (low_lambda + high_lambda) / 2.0
        candidate = np.clip(values + middle, lower, upper)
        if candidate.sum() < 1.0:
            low_lambda = middle
        else:
            high_lambda = middle
    result = np.clip(values + (low_lambda + high_lambda) / 2.0, lower, upper)
    return result / result.sum()


def equal_weights(groups: Sequence[str]) -> dict[str, float]:
    unique = list(dict.fromkeys(groups))
    if not unique:
        return {}
    value = 1.0 / len(unique)
    return {group: value for group in unique}


def compute_dynamic_weights(
    oos_metrics: pd.DataFrame,
    effective_date,
    config: Mapping,
    *,
    groups: Sequence[str] = DEFAULT_SUBMODEL_GROUPS,
    previous_weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Update weights using only OOS windows strictly before the effective date."""
    settings = config["dynamic_weights"] if "dynamic_weights" in config else config
    group_list = list(dict.fromkeys(groups))
    previous = equal_weights(group_list)
    if previous_weights:
        previous.update({group: float(previous_weights[group]) for group in group_list if group in previous_weights})
        total = sum(max(value, 0.0) for value in previous.values())
        previous = {group: max(value, 0.0) / total for group, value in previous.items()} if total > 0 else equal_weights(group_list)

    effective = pd.Timestamp(effective_date).normalize()
    metrics = oos_metrics.copy()
    if metrics.empty:
        report = pd.DataFrame(
            [{"effective_date": effective, "model_group": group, "raw_score": np.nan, "raw_weight": previous[group], "shrunk_weight": previous[group], "previous_weight": previous[group], "final_weight": previous[group], "weight_change": 0.0, "update_reason": "insufficient_oos_windows"} for group in group_list]
        )
        return previous, report
    metrics["test_end"] = pd.to_datetime(metrics["test_end"], errors="coerce").dt.normalize()
    history = metrics.loc[metrics["test_end"] < effective].copy()
    lookback_start = effective - pd.DateOffset(months=int(settings.get("lookback_months", 12)))
    history = history.loc[history["test_end"] >= lookback_start]
    minimum = int(settings.get("minimum_oos_windows", 6))
    records: list[dict] = []
    for group in group_list:
        sample = history.loc[history["model_group"] == group].sort_values("test_end")
        rank_ic = pd.to_numeric(sample.get("rank_ic", pd.Series(dtype=float)), errors="coerce")
        sharpe = pd.to_numeric(sample.get("sharpe", pd.Series(dtype=float)), errors="coerce")
        net_return = pd.to_numeric(sample.get("net_return", pd.Series(dtype=float)), errors="coerce")
        rank_std = rank_ic.std(ddof=1)
        icir = rank_ic.mean() / rank_std if np.isfinite(rank_std) and rank_std > 1e-12 else rank_ic.mean()
        records.append(
            {
                "model_group": group,
                "window_count": int(sample["test_end"].nunique()),
                "icir": icir,
                "net_sharpe": sharpe.mean(),
                "positive_window_ratio": (net_return > 0).mean() if len(net_return) else np.nan,
            }
        )
    scores = pd.DataFrame(records).set_index("model_group")
    eligible = scores["window_count"] >= minimum
    if not eligible.any():
        report = scores.reset_index()
        report["effective_date"] = effective
        report["raw_score"] = np.nan
        report["raw_weight"] = report["model_group"].map(previous)
        report["shrunk_weight"] = report["raw_weight"]
        report["previous_weight"] = report["raw_weight"]
        report["final_weight"] = report["raw_weight"]
        report["weight_change"] = 0.0
        report["update_reason"] = "insufficient_oos_windows"
        return previous, report

    ic_component = _normalize_positive(scores["icir"])
    sharpe_component = _normalize_positive(scores["net_sharpe"])
    positive_component = pd.to_numeric(scores["positive_window_ratio"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    raw_score = (
        float(settings.get("icir_weight", 0.60)) * ic_component
        + float(settings.get("net_sharpe_weight", 0.25)) * sharpe_component
        + float(settings.get("positive_window_ratio_weight", 0.15)) * positive_component
    ).where(eligible, 0.0).clip(lower=0.0)
    if raw_score.sum() <= 1e-12:
        data_weight = pd.Series(equal_weights(group_list))
        reason = "all_scores_unreliable_equal_weight"
    else:
        data_weight = raw_score / raw_score.sum()
        reason = "updated_from_past_oos_only"
    equal = pd.Series(equal_weights(group_list))
    shrink = float(settings.get("shrink_to_equal_weight", 0.50))
    shrunk = (1.0 - shrink) * data_weight + shrink * equal
    alpha = float(settings.get("smoothing_alpha", 0.20))
    previous_series = pd.Series(previous)
    smoothed = alpha * shrunk + (1.0 - alpha) * previous_series

    max_weight = float(settings.get("max_group_weight", 0.40))
    max_change = float(settings.get("max_monthly_change", 0.05))
    min_weight = float(settings.get("minimum_group_weight", 0.0))
    lower = np.maximum(min_weight, previous_series.to_numpy() - max_change)
    upper = np.minimum(max_weight, previous_series.to_numpy() + max_change)
    if upper.sum() < 1.0 - 1e-10:
        # With fewer than ceil(1/max_weight) active models, a normalized vector
        # cannot also satisfy the cap. Keep capped weights and leave residual
        # capital unallocated instead of silently breaking the risk limit.
        final_array = np.minimum(smoothed.to_numpy(dtype=float), upper)
    else:
        lower = np.minimum(lower, upper)
        final_array = _project_bounded_simplex(smoothed.to_numpy(dtype=float), lower, upper)
    final = pd.Series(final_array, index=group_list)

    report = scores.copy()
    report["effective_date"] = effective
    report["raw_score"] = raw_score
    report["raw_weight"] = data_weight
    report["shrunk_weight"] = shrunk
    report["previous_weight"] = previous_series
    report["final_weight"] = final
    report["weight_change"] = final - previous_series
    report["update_reason"] = np.where(eligible, reason, "group_insufficient_oos_windows")
    return final.to_dict(), report.reset_index()


def _renormalize_gated(weights: dict[str, float], maximum: float) -> dict[str, float]:
    active = [group for group, value in weights.items() if value > 0]
    if not active:
        return weights
    if len(active) * maximum < 1.0 - 1e-12:
        # A fully normalized vector cannot satisfy the cap with too few active
        # models. Preserve the cap and leave the remainder as cash/unallocated.
        total = sum(weights[group] for group in active)
        scale = min(1.0 / total, maximum / max(weights[group] for group in active))
        return {group: value * scale for group, value in weights.items()}
    values = np.array([weights[group] for group in active], dtype=float)
    values = values / values.sum()
    projected = _project_bounded_simplex(values, np.zeros(len(active)), np.full(len(active), maximum))
    result = {group: 0.0 for group in weights}
    result.update(dict(zip(active, projected)))
    return result


def apply_event_gates(
    weights: Mapping[str, float],
    latest_row: Mapping,
    config: Mapping,
) -> tuple[dict[str, float], dict[str, str]]:
    """Gate confirmed absent events while keeping missing data distinguishable."""
    settings = config["event_gating"] if "event_gating" in config else config
    gated = {group: max(float(value), 0.0) for group, value in weights.items()}
    status = {"news": "DISABLED", "lhb": "DISABLED", "lhb_institution": "DISABLED"}
    if bool(settings.get("news_enabled", True)) and "news" in gated:
        availability = pd.to_numeric(pd.Series([latest_row.get("news_data_available", np.nan)]), errors="coerce").iloc[0]
        has_news = pd.to_numeric(pd.Series([latest_row.get("has_news", np.nan)]), errors="coerce").iloc[0]
        if pd.isna(has_news) or availability == 0:
            status["news"] = "UNKNOWN_DATA_MISSING"
        elif has_news == 0:
            gated["news"] = 0.0
            status["news"] = "GATED_NO_NEWS"
        else:
            status["news"] = "ACTIVE_EVENT"

    if bool(settings.get("lhb_enabled", True)):
        detail_available = pd.to_numeric(pd.Series([latest_row.get("lhb_detail_available", latest_row.get("lhb_data_available", np.nan))]), errors="coerce").iloc[0]
        flag = pd.to_numeric(pd.Series([latest_row.get("lhb_flag", np.nan)]), errors="coerce").iloc[0]
        recent = pd.to_numeric(pd.Series([latest_row.get("lhb_count_5d", np.nan)]), errors="coerce").iloc[0]
        if detail_available == 0 or (pd.isna(flag) and pd.isna(recent)):
            status["lhb"] = "UNKNOWN_DATA_MISSING"
        elif flag == 0 and (pd.isna(recent) or recent == 0):
            if "lhb" in gated:
                gated["lhb"] = 0.0
            status["lhb"] = "GATED_NO_RECENT_EVENT"
        else:
            status["lhb"] = "ACTIVE_EVENT"

        inst_available = pd.to_numeric(pd.Series([latest_row.get("lhb_inst_data_available", np.nan)]), errors="coerce").iloc[0]
        inst_event = pd.to_numeric(pd.Series([latest_row.get("lhb_inst_buy_count", np.nan)]), errors="coerce").iloc[0]
        inst_recent = pd.to_numeric(pd.Series([latest_row.get("lhb_inst_net_buy_sum_5d", np.nan)]), errors="coerce").iloc[0]
        if inst_available == 0 or (pd.isna(inst_event) and pd.isna(inst_recent)):
            status["lhb_institution"] = "UNKNOWN_DATA_MISSING"
        elif (inst_event == 0 or pd.isna(inst_event)) and (inst_recent == 0 or pd.isna(inst_recent)):
            if "lhb_institution" in gated:
                gated["lhb_institution"] = 0.0
            status["lhb_institution"] = "GATED_NO_RECENT_EVENT"
        else:
            status["lhb_institution"] = "ACTIVE_EVENT"

    maximum = float(config.get("dynamic_weights", config).get("max_group_weight", 0.40))
    return _renormalize_gated(gated, maximum), status


def combine_group_predictions(predictions: Mapping[str, float], weights: Mapping[str, float]) -> float:
    usable = [(group, float(value)) for group, value in predictions.items() if group in weights and np.isfinite(value)]
    denominator = sum(float(weights[group]) for group, _value in usable)
    if denominator <= 1e-12:
        return np.nan
    return float(sum(float(weights[group]) * value for group, value in usable) / denominator)
