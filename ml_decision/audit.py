"""Factor-group ablation, grouped permutation and SHAP aggregation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .evaluation import evaluate_prediction_frame


ABLATION_VARIANTS = (
    "baseline",
    "baseline_plus_news",
    "baseline_plus_lhb",
    "baseline_plus_fund_flow",
    "baseline_plus_institution",
    "baseline_plus_news_lhb",
    "all_features",
    "all_without_news",
    "all_without_lhb",
    "all_without_fund_flow",
    "all_without_institution",
    "all_without_lhb_institution",
)


def _union(groups: Mapping[str, Iterable[str]], names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(column for name in names for column in groups.get(name, [])))


def ablation_feature_sets(groups: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    """Build all required variants from one immutable factor registry."""
    baseline_groups = ("technical", "liquidity", "market", "industry")
    baseline = _union(groups, baseline_groups)
    all_features = _union(groups, groups.keys())
    variants = {
        "baseline": baseline,
        "baseline_plus_news": list(dict.fromkeys(baseline + list(groups.get("news", [])))),
        "baseline_plus_lhb": list(dict.fromkeys(baseline + list(groups.get("lhb", [])) + list(groups.get("lhb_institution", [])))),
        "baseline_plus_fund_flow": list(dict.fromkeys(baseline + list(groups.get("fund_flow", [])))),
        "baseline_plus_institution": list(dict.fromkeys(baseline + list(groups.get("institution", [])))),
        "baseline_plus_news_lhb": list(dict.fromkeys(baseline + list(groups.get("news", [])) + list(groups.get("lhb", [])) + list(groups.get("lhb_institution", [])))),
        "all_features": all_features,
        "all_without_news": [column for column in all_features if column not in set(groups.get("news", []))],
        "all_without_lhb": [column for column in all_features if column not in set(groups.get("lhb", []))],
        "all_without_fund_flow": [column for column in all_features if column not in set(groups.get("fund_flow", []))],
        "all_without_institution": [column for column in all_features if column not in set(groups.get("institution", []))],
        "all_without_lhb_institution": [column for column in all_features if column not in set(groups.get("lhb_institution", []))],
    }
    return {name: variants[name] for name in ABLATION_VARIANTS}


def run_ablation(
    groups: Mapping[str, Iterable[str]],
    windows: Iterable,
    fit_predict: Callable[[list[str], object, str], pd.DataFrame],
    *,
    transaction_cost: float = 0.0016,
) -> pd.DataFrame:
    """Run identical windows through each required variant using an injected trainer."""
    rows: list[dict] = []
    for window in windows:
        window_id = getattr(window, "window_id", str(window))
        for variant, features in ablation_feature_sets(groups).items():
            if not features:
                rows.append({"window_id": window_id, "variant": variant, "status": "INACTIVE_NO_FEATURES"})
                continue
            prediction = fit_predict(features, window, variant)
            metrics = evaluate_prediction_frame(prediction, transaction_cost=transaction_cost)
            rows.append({"window_id": window_id, "variant": variant, "status": "OK", **metrics})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    baseline = result.loc[result["variant"] == "baseline"].set_index("window_id")
    all_model = result.loc[result["variant"] == "all_features"].set_index("window_id")
    for metric in ("auc", "pr_auc", "brier", "rank_ic", "net_return", "sharpe", "max_drawdown", "turnover"):
        result[f"baseline_{metric}_change"] = result[metric] - result["window_id"].map(baseline.get(metric, pd.Series(dtype=float)))
        result[f"all_{metric}_drop"] = result["window_id"].map(all_model.get(metric, pd.Series(dtype=float))) - result[metric]
    result["factor_group"] = result["variant"]
    result["rank_ic_change"] = result["baseline_rank_ic_change"]
    result["net_return_change"] = result["baseline_net_return_change"]
    result["sharpe_change"] = result["baseline_sharpe_change"]
    result["max_drawdown_change"] = result["baseline_max_drawdown_change"]
    positive_ratio = result.groupby("variant")["net_return"].transform(lambda values: (values > 0).mean())
    result["positive_window_ratio"] = positive_ratio
    return result


def grouped_permutation_importance(
    test_frame: pd.DataFrame,
    groups: Mapping[str, Iterable[str]],
    predict: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    repeats: int = 30,
    random_state: int = 42,
    transaction_cost: float = 0.0016,
) -> pd.DataFrame:
    """Permute complete groups within each date on the untouched final test set."""
    baseline = evaluate_prediction_frame(predict(test_frame.copy()), transaction_cost=transaction_cost)
    rows: list[dict] = []
    for group_name, columns in groups.items():
        present = [column for column in columns if column in test_frame.columns]
        if not present:
            continue
        drops: list[dict[str, float]] = []
        for repeat in range(max(int(repeats), 1)):
            rng = np.random.default_rng(random_state + repeat)
            shuffled = test_frame.copy()
            for _date, index in shuffled.groupby("date", sort=False).groups.items():
                positions = np.asarray(list(index))
                permutation = rng.permutation(len(positions))
                shuffled.loc[positions, present] = shuffled.loc[positions, present].to_numpy()[permutation]
            metrics = evaluate_prediction_frame(predict(shuffled), transaction_cost=transaction_cost)
            drops.append({metric: baseline.get(metric, np.nan) - metrics.get(metric, np.nan) for metric in baseline})
        drop_frame = pd.DataFrame(drops)
        auc_values = drop_frame.get("auc", pd.Series(dtype=float)).dropna()
        interval = 1.96 * auc_values.std(ddof=1) / np.sqrt(len(auc_values)) if len(auc_values) > 1 else np.nan
        rows.append(
            {
                "group_name": group_name,
                "auc_drop_mean": drop_frame.get("auc", pd.Series(dtype=float)).mean(),
                "auc_drop_std": drop_frame.get("auc", pd.Series(dtype=float)).std(ddof=1),
                "rank_ic_drop_mean": drop_frame.get("rank_ic", pd.Series(dtype=float)).mean(),
                "rank_ic_drop_std": drop_frame.get("rank_ic", pd.Series(dtype=float)).std(ddof=1),
                "net_return_drop_mean": drop_frame.get("net_return", pd.Series(dtype=float)).mean(),
                "sharpe_drop_mean": drop_frame.get("sharpe", pd.Series(dtype=float)).mean(),
                "confidence_interval_95": interval,
                "repeats": len(drop_frame),
            }
        )
    return pd.DataFrame(rows)


def group_shap_summary(
    shap_values: pd.DataFrame,
    groups: Mapping[str, Iterable[str]],
    sample_frame: pd.DataFrame | None = None,
    strong_news_threshold: float | None = None,
) -> pd.DataFrame:
    """Aggregate true per-sample signed SHAP values into group importance shares."""
    absolute_total = shap_values.abs().sum(axis=1).replace(0, np.nan)
    masks: dict[str, pd.Series] = {"all_samples": pd.Series(True, index=shap_values.index)}
    if sample_frame is not None:
        aligned = sample_frame.reindex(shap_values.index)
        def numeric_column(name: str) -> pd.Series:
            if name not in aligned.columns:
                return pd.Series(np.nan, index=aligned.index, dtype=float)
            return pd.to_numeric(aligned[name], errors="coerce")

        sentiment = numeric_column("news_sentiment")
        threshold = strong_news_threshold
        if threshold is None or not np.isfinite(threshold):
            threshold = sentiment.abs().quantile(0.8)
        masks.update(
            {
                "news_samples": numeric_column("has_news") == 1,
                "strong_news_samples": sentiment.abs() >= threshold,
                "lhb_samples": numeric_column("lhb_flag") == 1,
                "lhb_inst_positive_samples": numeric_column("lhb_inst_net_buy_ratio") > 0,
            }
        )
    rows: list[dict] = []
    for group_name, columns in groups.items():
        present = [column for column in columns if column in shap_values.columns]
        if not present:
            continue
        magnitude = shap_values[present].abs().sum(axis=1)
        signed = shap_values[present].sum(axis=1)
        for condition, mask in masks.items():
            valid_mask = mask.fillna(False)
            rows.append(
                {
                    "group_name": group_name,
                    "condition": condition,
                    "sample_count": int(valid_mask.sum()),
                    "mean_abs_shap": magnitude.loc[valid_mask].mean(),
                    "shap_share": (magnitude / absolute_total).loc[valid_mask].mean(),
                    "mean_signed_shap": signed.loc[valid_mask].mean(),
                }
            )
    return pd.DataFrame(rows)


def sample_shap_directions(shap_row: pd.Series, raw_row: pd.Series, top_n: int = 3) -> dict[str, list[dict]]:
    """Return honest positive/negative directions from signed SHAP, never importances."""
    values = pd.to_numeric(shap_row, errors="coerce").dropna()
    positive = values.loc[values > 0].sort_values(ascending=False).head(top_n)
    negative = values.loc[values < 0].sort_values().head(top_n)
    encode = lambda series: [
        {"factor": factor, "shap_value": float(value), "raw_value": raw_row.get(factor, np.nan)}
        for factor, value in series.items()
    ]
    return {"positive": encode(positive), "negative": encode(negative)}
