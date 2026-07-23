"""Daily prediction, monthly training and quarterly factor-audit workflows."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .audit import group_shap_summary, grouped_permutation_importance, run_ablation
from .baselines import evaluate_stable_baselines
from .cross_section import compute_full_universe_ranks, save_rank_cache
from .ensemble import apply_event_gates, combine_group_predictions, compute_dynamic_weights, equal_weights
from .evaluation import evaluate_factors, evaluate_prediction_frame
from .factor_registry import build_factor_groups, classify_factor, factor_group_counts, source_requirements
from .features import TARGET_COLUMNS, add_labels, build_features, feature_columns
from .models import NextSessionModel, xgboost_backend_label
from .policy import FactorPolicyParameters, calibrate_factor_policy
from .schema import FEATURE_SCHEMA_VERSION, feature_schema_hash
from .governance_training import (
    WindowModelResult,
    model_prediction_frame,
    model_shap_frame,
    train_fixed_window,
    train_window_model_set,
)
from .rolling import RollingWindow, rolling_windows_from_config
from .versioning import ModelRegistry
from .model_registry import ProductionModelLoader


REPORT_FILES = (
    "factor_quality.csv",
    "factor_ic_history.csv",
    "factor_group_ablation.csv",
    "factor_group_permutation.csv",
    "factor_group_shap.csv",
    "model_oos_metrics.csv",
    "dynamic_group_weights.csv",
    "factor_status.csv",
)


def load_governance_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "ml_governance.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_labelled_dataset(frame: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    """Reuse prepared factors when supplied; otherwise build without replacing source fields."""
    data = frame.copy()
    target = config["target"]["column"]
    if target in data.columns:
        return data
    feature_markers = {"ret_5", "rsi_14", "market_rank_ret_5"}
    if not feature_markers.intersection(data.columns):
        data = build_features(data, external_factor_lag=0)
    return add_labels(
        data,
        round_trip_cost=float(config["factor_evaluation"].get("transaction_cost", 0.0016)),
        down_threshold=-0.02,
    )


def _active_group_names(results: Mapping[str, WindowModelResult]) -> list[str]:
    return [
        group for group, result in results.items()
        if group != "all_factor" and result.status == "ACTIVE" and not result.predictions.empty
    ]


def _ensemble_prediction_frame(
    dataset: pd.DataFrame,
    results: Mapping[str, WindowModelResult],
    weights: Mapping[str, float],
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = _active_group_names(results)
    if not active:
        return pd.DataFrame(), pd.DataFrame()
    first = results[active[0]].predictions[["date", "code", "next_open_to_next_open_return"]].copy()
    for group in active:
        pred = results[group].predictions[["date", "code", "probability_up", "predicted_return"]].rename(
            columns={"probability_up": f"probability_up__{group}", "predicted_return": f"predicted_return__{group}"}
        )
        first = first.merge(pred, on=["date", "code"], how="left", validate="one_to_one")
    context_columns = [
        column for column in (
            "date", "code", "has_news", "news_data_available", "lhb_data_available",
            "lhb_detail_available", "lhb_inst_data_available", "lhb_flag", "lhb_count_5d",
            "lhb_inst_buy_count", "lhb_inst_net_buy_sum_5d",
        ) if column in dataset.columns
    ]
    context = dataset[context_columns].drop_duplicates(["date", "code"], keep="last")
    first = first.merge(context, on=["date", "code"], how="left", validate="one_to_one")
    gate_rows: list[dict] = []
    probability_values: list[float] = []
    return_values: list[float] = []
    for row in first.to_dict("records"):
        gated, status = apply_event_gates(weights, row, config)
        probability_values.append(
            combine_group_predictions({group: row.get(f"probability_up__{group}", np.nan) for group in active}, gated)
        )
        return_values.append(
            combine_group_predictions({group: row.get(f"predicted_return__{group}", np.nan) for group in active}, gated)
        )
        gate_rows.append({"date": row["date"], "code": row["code"], **status, **{f"weight_{key}": value for key, value in gated.items()}})
    first["probability_up"] = probability_values
    first["predicted_return"] = return_values
    return first, pd.DataFrame(gate_rows)


def _comparison_markdown(metrics: pd.DataFrame, weights: Mapping[str, float]) -> tuple[str, dict[str, Any]]:
    all_rows = metrics.loc[metrics["model_group"] == "all_factor"].sort_values("test_end")
    ensemble = metrics.loc[metrics["model_group"] == "dynamic_group_ensemble"].sort_values("test_end")
    merged = all_rows.merge(ensemble, on="window_id", suffixes=("_all", "_ensemble"))
    better = (merged.get("rank_ic_ensemble", pd.Series(dtype=float)) > merged.get("rank_ic_all", pd.Series(dtype=float))).astype(int)
    consecutive = 0
    for value in better.iloc[::-1]:
        if value:
            consecutive += 1
        else:
            break
    net_not_worse = bool(merged["net_return_ensemble"].mean() >= merged["net_return_all"].mean()) if not merged.empty else False
    drawdown_not_worse = bool(merged["max_drawdown_ensemble"].mean() >= merged["max_drawdown_all"].mean()) if not merged.empty else False
    comparison = {
        "consecutive_better_windows": consecutive,
        "net_return_not_worse": net_not_worse,
        "drawdown_not_worse": drawdown_not_worse,
        "regression_tests_passed": False,
    }
    weight_text = ", ".join(f"{group}={value:.3f}" for group, value in sorted(weights.items())) or "none"
    lines = [
        "# Model OOS Comparison",
        "",
        "All conclusions below use rolling test windows only.",
        "",
        "## Full-sample contribution",
        f"Comparable rolling OOS windows: {len(merged)}.",
        "",
        "## Event-conditional contribution",
        "See factor_group_shap.csv and factor_group_ablation.csv after the quarterly audit; no event claim is inferred when those reports have not run.",
        "",
        "## Prediction accuracy contribution",
        f"Ensemble consecutive Rank IC wins: {consecutive}.",
        "",
        "## Net performance contribution",
        f"Net return not worse: {net_not_worse}; drawdown not worse: {drawdown_not_worse}.",
        "",
        "## Cross-time stability",
        "Promotion requires at least three consecutive better OOS windows rather than one month.",
        "",
        "## Current model weights",
        weight_text,
        "",
        "## Continue using",
        "Candidate only. Production promotion remains pending regression tests and all OOS gates.",
    ]
    return "\n".join(lines) + "\n", comparison


def _write_csv(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    output = frame.copy()
    if output.empty and columns:
        output = pd.DataFrame(columns=columns)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def monthly_train(
    source_frame: pd.DataFrame,
    project_root: str | Path,
    *,
    config: Mapping | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Run strict rolling OOS training, update weights and save an immutable candidate."""
    root = Path(project_root).resolve()
    settings = dict(config or load_governance_config())
    dataset = ensure_labelled_dataset(source_frame, settings)
    universe_size = int(dataset["code"].astype(str).nunique())
    if universe_size >= 500:
        try:
            rank_frame = compute_full_universe_ranks(dataset)
            save_rank_cache(rank_frame, root)
            rank_columns = [
                column
                for column in rank_frame.columns
                if column.startswith(("market_rank_", "industry_rank_"))
            ]
            dataset = dataset.drop(columns=rank_columns, errors="ignore").merge(
                rank_frame[["date", "code", *rank_columns]],
                on=["date", "code"],
                how="left",
                validate="many_to_one",
            )
            print(f"[cross-section] full-universe rank cache: {len(rank_frame)} rows", flush=True)
        except ValueError as exc:
            print(f"[cross-section] disabled: {exc}", flush=True)
    target = settings["target"]["column"]
    dataset = dataset.dropna(subset=[target]).copy()
    windows = rolling_windows_from_config(dataset, settings)
    if not windows:
        raise ValueError("No complete 36/2/1/1 rolling window is available")
    # Global candidates are based only on legality and whether a factor is
    # completely unusable. Each rolling window performs its own train-only
    # coverage/variance selection inside train_fixed_window.
    candidates = feature_columns(dataset, selection_df=dataset)
    groups = build_factor_groups(candidates)
    oos_rows: list[dict] = []
    weight_reports: list[pd.DataFrame] = []
    previous_weights: dict[str, float] | None = None
    latest_results: dict[str, WindowModelResult] = {}
    policy_oos_frames: list[pd.DataFrame] = []
    history = pd.DataFrame()
    for window_index, window in enumerate(windows, start=1):
        print(
            f"[training {window_index}/{len(windows)}] window={window.window_id} "
            f"test={window.test_dates.min().date()}..{window.test_dates.max().date()}",
            flush=True,
        )
        results = train_window_model_set(dataset, groups, window, settings)
        latest_results = results
        for group, result in results.items():
            if result.status == "ACTIVE":
                oos_rows.append({"window_id": window.window_id, **result.metrics})
            else:
                oos_rows.append({"window_id": window.window_id, "model_group": group, "status": result.status, "failure_reason": result.reason, **window.ranges()})
        all_factor_result = results.get("all_factor")
        if all_factor_result is not None and all_factor_result.status == "ACTIVE" and not all_factor_result.predictions.empty:
            policy_oos_frames.append(all_factor_result.predictions.assign(window_id=window.window_id))
        if all_factor_result is not None and all_factor_result.feature_columns:
            try:
                print(
                    f"[cpu-baseline] window={window.window_id} "
                    "running Ridge/Logistic comparison only",
                    flush=True,
                )
                oos_rows.extend(
                    evaluate_stable_baselines(
                        dataset,
                        all_factor_result.feature_columns,
                        window,
                        transaction_cost=float(
                            settings["factor_evaluation"].get("transaction_cost", 0.0016)
                        ),
                    )
                )
            except Exception as exc:
                oos_rows.append(
                    {
                        "window_id": window.window_id,
                        "model_group": "stable_baselines",
                        "status": "FAILED",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        **window.ranges(),
                    }
                )
        history = pd.DataFrame(oos_rows)
        active_groups = _active_group_names(results)
        weights, report = compute_dynamic_weights(
            history,
            window.test_dates.min(),
            settings,
            groups=active_groups,
            previous_weights=previous_weights,
        )
        previous_weights = weights
        weight_reports.append(report.assign(window_id=window.window_id))
        ensemble_frame, _gates = _ensemble_prediction_frame(dataset, results, weights, settings)
        if not ensemble_frame.empty:
            ensemble_metrics = evaluate_prediction_frame(
                ensemble_frame,
                transaction_cost=float(settings["factor_evaluation"].get("transaction_cost", 0.0016)),
            )
            oos_rows.append(
                {
                    "window_id": window.window_id,
                    "model_group": "dynamic_group_ensemble",
                    **ensemble_metrics,
                    **window.ranges(),
                }
            )

    oos_metrics = pd.DataFrame(oos_rows)
    active_latest = _active_group_names(latest_results)
    if latest_results.get("all_factor") is None or latest_results["all_factor"].status != "ACTIVE":
        reason = latest_results.get("all_factor").reason if latest_results.get("all_factor") else "missing"
        raise RuntimeError(f"Latest all-factor model is not deployable: {reason}")
    next_effective = windows[-1].test_dates.max() + pd.offsets.BDay(1)
    final_weights, final_weight_report = compute_dynamic_weights(
        oos_metrics,
        next_effective,
        settings,
        groups=active_latest,
        previous_weights=previous_weights,
    )
    weight_reports.append(final_weight_report.assign(window_id="production_next"))
    weights_frame = pd.concat(weight_reports, ignore_index=True) if weight_reports else pd.DataFrame()
    policy_predictions = (
        pd.concat(policy_oos_frames, ignore_index=True, sort=False)
        .sort_values(["date", "code"], kind="stable")
        .drop_duplicates(["date", "code"], keep="last")
        if policy_oos_frames else pd.DataFrame()
    )
    policy_settings = dict(settings.get("factor_policy", {}))
    policy_settings.setdefault(
        "transaction_cost", float(settings["factor_evaluation"].get("transaction_cost", 0.0016))
    )
    print("[policy] calibrating allocation policy on rolling OOS predictions", flush=True)
    try:
        policy_parameters, policy_report, policy_backtest, policy_summary = calibrate_factor_policy(
            policy_predictions,
            policy_settings,
        )
    except Exception as exc:
        policy_parameters = FactorPolicyParameters(
            transaction_cost=float(policy_settings.get("transaction_cost", 0.0016)),
            max_single_weight=float(policy_settings.get("max_single_weight", 0.25)),
            weight_step=float(policy_settings.get("weight_step", 0.05)),
        )
        policy_report = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "parameters": policy_parameters.to_dict(),
        }
        policy_backtest = pd.DataFrame()
        policy_summary = pd.DataFrame()
    factor_status = {
        group: ("ACTIVE" if result.status == "ACTIVE" else "INACTIVE")
        for group, result in latest_results.items()
    }
    factor_columns = {group: result.feature_columns for group, result in latest_results.items()}
    model_settings = settings.get("model", {})
    models: dict[str, NextSessionModel] = {}
    refit_groups = [
        (group, result)
        for group, result in latest_results.items()
        if result.status == "ACTIVE" and result.feature_columns
    ]
    print(f"[refit] fitting {len(refit_groups)} production factor-group models", flush=True)
    for refit_index, (group, result) in enumerate(refit_groups, start=1):
        print(f"[refit {refit_index}/{len(refit_groups)}] {group}", flush=True)
        models[group] = NextSessionModel(
            calibration_method=model_settings.get("calibration_method", "sigmoid"),
            random_state=int(model_settings.get("random_state", 42)),
            use_shap=bool(model_settings.get("use_shap", True)),
            gpu_device=str(model_settings.get("device", "cuda:0")),
            host_cpu_threads=int(model_settings.get("host_cpu_threads", 2)),
        ).refit_production(
            dataset,
            result.feature_columns,
            frozen_metrics=dict(getattr(result.model, "metrics", {})),
            purge_days=int(settings.get("rolling_training", {}).get("purge_trading_days", 2)),
        )
    latest_ranges = windows[-1].ranges()
    unique_symbols = int(dataset["code"].astype(str).nunique())
    universe_definition = "all_a_share" if unique_symbols >= 500 else "custom_training_panel"
    timing_lag = 0
    production_training_end = str(pd.to_datetime(dataset["date"]).max().date())
    dataset_dates = pd.to_datetime(dataset["date"], errors="coerce").dt.normalize()
    latest_train_mask = dataset_dates.isin(windows[-1].train_dates)
    recent_dates = set(dataset_dates.dropna().drop_duplicates().sort_values().tail(60))
    recent_mask = dataset_dates.isin(recent_dates)
    active_features = set(factor_columns.get("all_factor", []))
    factor_diagnostics: dict[str, dict[str, Any]] = {}
    for column in candidates:
        values = pd.to_numeric(dataset[column], errors="coerce")
        valid_dates = dataset_dates.loc[values.notna()]
        group = classify_factor(column)
        factor_diagnostics[column] = {
            "factor_group": group,
            "status": "ACTIVE" if column in active_features else "INACTIVE",
            "first_available_date": (
                str(valid_dates.min().date()) if not valid_dates.empty else None
            ),
            "training_coverage": float(values.loc[latest_train_mask].notna().mean()),
            "recent_coverage": float(values.loc[recent_mask].notna().mean()),
            "inactive_reason": "" if column in active_features else "not_selected_in_latest_training_window",
            "data_source": group,
            "current_group_weight": float(final_weights.get(group, 0.0)),
        }
    schema_hash = feature_schema_hash(
        factor_columns,
        external_factor_lag=timing_lag,
        universe_definition=universe_definition,
    )
    metadata = {
        "target_definition": "features known after t close -> execute t+1 open -> exit t+2 open",
        "label_definition": "t close-known factors -> execute t+1 open -> exit t+2 open",
        "target_column": target,
        "factor_count": len(candidates),
        "factor_group_counts": factor_group_counts(groups),
        "factor_diagnostics": factor_diagnostics,
        "random_seed": settings.get("model", {}).get("random_state", 42),
        "transaction_cost": settings["factor_evaluation"].get("transaction_cost", 0.0016),
        "model_backend": xgboost_backend_label(),
        "calibration_status": latest_results["all_factor"].model.metrics.get("calibration_up_status", "unknown"),
        "calibration_metrics": {
            key: value
            for key, value in models["all_factor"].metrics.items()
            if key.startswith(("calibration_", "test_brier_", "test_log_loss_", "test_ece_", "test_calibration_", "test_reliability_"))
        },
        "factor_policy": {
            **policy_report,
            "parameters": policy_parameters.to_dict(),
            "decision_source": "factor_model_oos_predictions",
            "traditional_strategy_dependency": False,
        },
        "rolling_windows": [window.ranges() for window in windows],
        "latest_window": latest_ranges,
        "oos_evaluation_end": latest_ranges.get("test_end"),
        "production_training_end": production_training_end,
        "production_refit": True,
        "decision_time": "after_close",
        "execution_time": "next_open",
        "market_data_lag": 0,
        "external_factor_lag": timing_lag,
        "news_cutoff_time": "15:00:00",
        "universe_definition": universe_definition,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": schema_hash,
        "feature_columns": factor_columns,
        "source_requirements": source_requirements(
            column
            for columns in factor_columns.values()
            for column in columns
        ),
        "latest_complete_bar_only": True,
        "training_start": latest_ranges.get("train_start"),
        "training_end": latest_ranges.get("train_end"),
        "calibration_start": latest_ranges.get("calibration_start"),
        "calibration_end": latest_ranges.get("calibration_end"),
        "validation_start": latest_ranges.get("validation_start"),
        "validation_end": latest_ranges.get("validation_end"),
        "test_start": latest_ranges.get("test_start"),
        "test_end": latest_ranges.get("test_end"),
        "config": settings,
    }
    version = version or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    registry = ModelRegistry(root)
    print(f"[reports] saving candidate model and strict OOS reports as {version}", flush=True)
    model_path = registry.save_version(
        version,
        models,
        metadata=metadata,
        group_weights=final_weights,
        factor_columns=factor_columns,
        factor_status=factor_status,
        training_metrics=oos_metrics.to_dict("records"),
    )
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    persisted_policy_backtest = policy_backtest.copy()
    if "candidate_utilities" in persisted_policy_backtest.columns:
        persisted_policy_backtest["candidate_utilities"] = persisted_policy_backtest["candidate_utilities"].map(
            lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
    for directory in (reports, Path(model_path)):
        directory.mkdir(parents=True, exist_ok=True)
        backtest_path = directory / "ml_policy_backtest.parquet"
        if not persisted_policy_backtest.empty:
            persisted_policy_backtest.to_parquet(backtest_path, index=False)
        elif backtest_path.exists():
            backtest_path.unlink()
        _write_csv(policy_summary, directory / "ml_policy_stock_summary.csv")
        (directory / "ml_policy_report.json").write_text(
            json.dumps(policy_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    _write_csv(oos_metrics, reports / "model_oos_metrics.csv")
    _write_csv(weights_frame, reports / "dynamic_group_weights.csv")
    _write_csv(pd.DataFrame([{"factor_name": key, "status": value} for key, value in factor_status.items()]), reports / "factor_status.csv")
    comparison_markdown, comparison = _comparison_markdown(oos_metrics, final_weights)
    (reports / "model_comparison.md").write_text(comparison_markdown, encoding="utf-8")
    (reports / "model_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports / "latest_training_report.md").write_text(
        "\n".join(
            [
                "# Latest Monthly Training",
                "",
                f"- Candidate version: `{version}`",
                f"- Rolling OOS windows: {len(windows)}",
                f"- Training range: {latest_ranges.get('train_start')} to {latest_ranges.get('train_end')}",
                f"- Test range: {latest_ranges.get('test_start')} to {latest_ranges.get('test_end')}",
                f"- Target: {metadata['target_definition']}",
                f"- Feature count: {len(candidates)}",
                f"- Model path: `{model_path}`",
                "",
                "## Final group weights",
                *[f"- {group}: {weight:.6f}" for group, weight in sorted(final_weights.items())],
                "",
                "The candidate is not production until OOS gates and regression tests pass.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "version": version,
        "model_path": model_path,
        "factor_groups": groups,
        "oos_metrics": oos_metrics,
        "dynamic_weights": weights_frame,
        "final_weights": final_weights,
        "comparison": comparison,
        "factor_policy": policy_report,
        "policy_backtest": policy_backtest,
        "policy_stock_summary": policy_summary,
    }


def daily_predict(
    factor_frame: pd.DataFrame,
    project_root: str | Path,
    *,
    status: str = "production",
) -> pd.DataFrame:
    """Load a formal model and predict; this path contains no fit or weight update."""
    package = (
        ProductionModelLoader(project_root).load()
        if status == "production"
        else ModelRegistry(project_root).load_status(status)
    )
    models = package["models"]
    weights = package["group_weights"]
    config = package["metadata"]["config"]
    latest = factor_frame.copy()
    if not {"ret_5", "rsi_14"}.intersection(latest.columns):
        latest = build_features(latest)
    latest = latest.sort_values("date").groupby("code", sort=False).tail(1)
    rows: list[dict] = []
    for _, row in latest.iterrows():
        aligned = row.copy()
        required = {
            column
            for model in models.values()
            for column in getattr(model, "feature_columns", [])
        }
        for column in required:
            if column not in aligned.index:
                aligned[column] = np.nan
        all_pack = models["all_factor"].predict_one(aligned) if "all_factor" in models else None
        group_packs = {
            group: model.predict_one(aligned)
            for group, model in models.items()
            if group != "all_factor"
        }
        gated, gate_status = apply_event_gates(weights, aligned, config)
        ensemble_probability = combine_group_predictions(
            {group: pack.probability_up for group, pack in group_packs.items()}, gated
        )
        ensemble_return = combine_group_predictions(
            {group: pack.expected_open_to_open_return for group, pack in group_packs.items()}, gated
        )
        rows.append(
            {
                "date": pd.to_datetime(row["date"]),
                "code": str(row["code"]).zfill(6),
                "name": row.get("name", ""),
                "all_factor_probability_up": all_pack.probability_up if all_pack else np.nan,
                "all_factor_predicted_return": all_pack.expected_open_to_open_return if all_pack else np.nan,
                "ensemble_probability_up": ensemble_probability,
                "ensemble_predicted_return": ensemble_return,
                **{f"gate_{key}": value for key, value in gate_status.items()},
                **{f"weight_{key}": value for key, value in gated.items()},
            }
        )
    return pd.DataFrame(rows)


def quarterly_audit(
    source_frame: pd.DataFrame,
    project_root: str | Path,
    *,
    config: Mapping | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate OOS-only factor, ablation, permutation and SHAP audit reports."""
    root = Path(project_root).resolve()
    settings = dict(config or load_governance_config())
    dataset = ensure_labelled_dataset(source_frame, settings)
    target = settings["target"]["column"]
    dataset = dataset.dropna(subset=[target]).copy()
    windows = rolling_windows_from_config(dataset, settings)
    if not windows:
        raise ValueError("No complete rolling OOS window is available for quarterly audit")
    first_train = windows[0].as_splits().frame(dataset, "train")
    factors = feature_columns(dataset, selection_df=dataset)
    groups = build_factor_groups(factors)
    oos_dates = pd.DatetimeIndex([])
    for window in windows:
        oos_dates = oos_dates.union(window.test_dates)
    normalized = pd.to_datetime(dataset["date"]).dt.normalize()
    oos_frame = dataset.loc[normalized.isin(oos_dates)]
    news_sentiment = pd.to_numeric(first_train.get("news_sentiment", pd.Series(dtype=float)), errors="coerce")
    news_count = pd.to_numeric(first_train.get("news_count_3", pd.Series(dtype=float)), errors="coerce")
    lhb_strength = pd.to_numeric(first_train.get("lhb_net_buy_ratio", pd.Series(dtype=float)), errors="coerce")
    event_thresholds = {
        "news_sentiment_low": news_sentiment.quantile(0.2),
        "news_sentiment_high": news_sentiment.quantile(0.8),
        "news_count_high": news_count.quantile(0.8),
        "lhb_net_buy_high": lhb_strength.quantile(0.7),
    }
    quality, ic_history = evaluate_factors(
        oos_frame,
        factors,
        target,
        minimum_samples=int(settings["factor_evaluation"].get("minimum_cross_section_samples", 20)),
        quantile_groups=int(settings["factor_evaluation"].get("quantile_groups", 5)),
        transaction_cost=float(settings["factor_evaluation"].get("transaction_cost", 0.0016)),
        event_thresholds=event_thresholds,
    )

    def fit_predict(variant_features: list[str], window: RollingWindow, variant: str) -> pd.DataFrame:
        result = train_fixed_window(dataset, variant_features, window, group=variant, config=settings)
        if result.status != "ACTIVE":
            return pd.DataFrame(columns=["date", "code", target, "probability_up", "predicted_return"])
        return result.predictions

    ablation = run_ablation(
        groups,
        windows,
        fit_predict,
        transaction_cost=float(settings["factor_evaluation"].get("transaction_cost", 0.0016)),
    )
    latest = train_fixed_window(dataset, factors, windows[-1], group="all_factor", config=settings)
    if latest.status != "ACTIVE":
        raise RuntimeError(f"Latest all-factor model failed: {latest.reason}")
    test_frame = windows[-1].as_splits().frame(dataset, "test")

    def predict(frame: pd.DataFrame) -> pd.DataFrame:
        return model_prediction_frame(latest.model, frame)

    permutation = grouped_permutation_importance(
        test_frame,
        groups,
        predict,
        repeats=int(settings["factor_evaluation"].get("permutation_repeats", 30)),
        random_state=int(settings.get("model", {}).get("random_state", 42)),
        transaction_cost=float(settings["factor_evaluation"].get("transaction_cost", 0.0016)),
    )
    shap_values = model_shap_frame(latest.model, test_frame)
    strong_news_threshold = news_sentiment.abs().quantile(
        float(settings["event_gating"].get("strong_news_quantile", 0.80))
    )
    shap = group_shap_summary(
        shap_values,
        groups,
        test_frame,
        strong_news_threshold=strong_news_threshold,
    )
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    _write_csv(quality, reports / "factor_quality.csv")
    _write_csv(ic_history, reports / "factor_ic_history.csv")
    _write_csv(ablation, reports / "factor_group_ablation.csv")
    _write_csv(permutation, reports / "factor_group_permutation.csv")
    _write_csv(shap, reports / "factor_group_shap.csv")
    status_columns = [
        column for column in (
            "factor_name", "factor_group", "status", "valid_oos_windows", "coverage_rate",
            "ic_mean", "icir", "positive_ic_ratio", "net_top_bottom_return",
        ) if column in quality.columns
    ]
    _write_csv(quality[status_columns], reports / "factor_status.csv")
    return {
        "factor_quality": quality,
        "factor_ic_history": ic_history,
        "factor_group_ablation": ablation,
        "factor_group_permutation": permutation,
        "factor_group_shap": shap,
    }
