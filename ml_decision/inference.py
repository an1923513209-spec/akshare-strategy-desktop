"""Fast production inference for the desktop holding-decision workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .actions import Holding, apply_account_constraints, choose_action, score_actions, score_policy_target
from .config import AccountState, DecisionConfig
from .cross_section import load_rank_cache
from .data_sources import SourceNote
from .drift import assess_model_drift, update_prediction_history
from .engine import _output_row, prepare_holdings
from .ensemble import apply_event_gates, combine_group_predictions
from .factor_registry import classify_factor, source_requirements
from .features import build_features, normalize_market_df
from .model_registry import ProductionModelLoader
from .models import PredictionPack
from .policy import FactorPolicyParameters, choose_target_weight
from .schema import FEATURE_SCHEMA_VERSION, feature_schema_hash
from .trading_rules import enrich_trade_constraints
from .workflows import load_governance_config


@dataclass(slots=True)
class ProductionDecisionResult:
    """Daily inference output plus immutable production-model metadata."""

    table: pd.DataFrame
    metrics: dict[str, Any]
    feature_columns: list[str]
    source_notes: list[SourceNote]
    model_metadata: dict[str, Any]
    model_version: str
    snapshot_path: str | None
    drift_report: dict[str, Any] = field(default_factory=dict)


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _availability(row: pd.Series, explicit: str, feature_names: tuple[str, ...]) -> float:
    value = _finite(row.get(explicit), np.nan)
    if np.isfinite(value):
        return float(value > 0)
    return float(any(np.isfinite(_finite(row.get(name), np.nan)) for name in feature_names))


def data_availability(
    row: pd.Series,
    required_features: set[str] | None = None,
) -> dict[str, float]:
    """Return source availability without treating missing sources as zero-valued events."""
    rank_features = [
        name
        for name in (required_features or set())
        if name.startswith(("market_rank_", "industry_rank_"))
    ]
    return {
        "market_data_available": float(
            all(np.isfinite(_finite(row.get(name), np.nan)) for name in ("open", "high", "low", "close"))
        ),
        "cross_section_rank_available": float(
            not rank_features
            or all(np.isfinite(_finite(row.get(name), np.nan)) for name in rank_features)
        ),
        "fund_flow_data_available": _availability(
            row, "fund_flow_data_available", ("main_net_ratio", "large_net_ratio")
        ),
        "news_data_available": _availability(
            row, "news_data_available", ("has_news", "news_sentiment")
        ),
        "institution_data_available": _availability(
            row, "institution_data_available", ("institution_activity", "institution_hold_ratio")
        ),
        "lhb_data_available": _availability(
            row, "lhb_detail_available", ("lhb_flag", "lhb_count_5d")
        ),
        "lhb_inst_data_available": _availability(
            row, "lhb_inst_data_available", ("lhb_inst_buy_count", "lhb_inst_net_buy_sum_5d")
        ),
    }


def required_data_completeness(
    availability: Mapping[str, float],
    requirements: Mapping[str, Any] | None,
    data_date: Any = None,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Score only sources required by the frozen production feature schema."""
    required = dict(requirements or {})
    if not required:
        required = {name: name == "market_data_available" for name in availability}
    details: dict[str, dict[str, Any]] = {}
    core_values: list[float] = []
    for name, value in availability.items():
        is_required = bool(required.get(name, False))
        available = float(value) if np.isfinite(value) else 0.0
        details[name] = {
            "required": is_required,
            "available": bool(available > 0),
            "degraded": bool(is_required and available <= 0),
            "latest_date": str(data_date) if available > 0 and data_date is not None else None,
            "missing_ratio": 0.0 if available > 0 else 1.0,
            "impact": (
                "required_source_missing"
                if is_required and available <= 0
                else "optional_source_missing"
                if not is_required and available <= 0
                else "available"
            ),
        }
        if is_required:
            core_values.append(available)
    return (float(np.mean(core_values)) if core_values else 1.0), details


def _latest_oos_metrics(package: Mapping[str, Any]) -> dict[str, Any]:
    rows = package.get("training_metrics", [])
    if not isinstance(rows, list):
        return {}
    all_rows = [row for row in rows if isinstance(row, dict) and row.get("model_group") == "all_factor"]
    if not all_rows:
        return {}
    return dict(all_rows[-1])


def confidence_score(
    probability: float,
    expected_return: float,
    group_probabilities: Mapping[str, float],
    completeness: float,
    metrics: Mapping[str, Any],
) -> tuple[float, str]:
    """Score directional strength, model agreement, OOS stability and source coverage."""
    edge = float(np.clip(abs(probability - 0.5) * 2.0, 0.0, 1.0))
    direction = float((probability >= 0.5) == (expected_return >= 0.0))
    usable = [value for value in group_probabilities.values() if np.isfinite(value)]
    if usable:
        agreement = float(np.mean([(value >= 0.5) == (probability >= 0.5) for value in usable]))
    else:
        agreement = 0.5
    auc = _finite(metrics.get("auc"), _finite(metrics.get("auc_up"), 0.5))
    rank_ic = _finite(metrics.get("rank_ic"), 0.0)
    stability = float(
        0.6 * np.clip((auc - 0.5) / 0.15, 0.0, 1.0)
        + 0.4 * np.clip(rank_ic / 0.10, 0.0, 1.0)
    )
    score = float(
        np.clip(
            0.25 * edge + 0.20 * direction + 0.20 * agreement + 0.25 * completeness + 0.10 * stability,
            0.0,
            1.0,
        )
    )
    if completeness < 0.50:
        level = "unavailable"
    elif score >= 0.80:
        level = "high"
    elif score >= 0.60:
        level = "medium"
    else:
        level = "low"
    return score, level


def _shap_details(model: Any, latest_row: pd.Series) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    try:
        x = pd.DataFrame([latest_row[model.feature_columns].to_dict()])
        contributions = model._shap_contributions(x)
    except Exception:
        return [], [], list(getattr(model, "_important_factors", lambda: [])())

    def detail(name: str, value: float) -> dict[str, Any]:
        raw = _finite(latest_row.get(name), np.nan)
        return {
            "factor_name": name,
            "factor_group": classify_factor(name),
            "factor_value": raw,
            "shap_value": float(value),
            "human_readable_description": name,
        }

    positive = [detail(name, value) for name, value in sorted(contributions.items(), key=lambda x: x[1], reverse=True) if value > 0][:3]
    negative = [detail(name, value) for name, value in sorted(contributions.items(), key=lambda x: x[1]) if value < 0][:3]
    return positive, negative, []


def _prediction_pack(
    all_pack: PredictionPack,
    probability: float,
    expected_return: float,
    confidence_level: str,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    important: list[str],
) -> PredictionPack:
    return PredictionPack(
        expected_gap_return=all_pack.expected_gap_return,
        expected_open_to_open_return=expected_return,
        probability_up=probability,
        probability_profitable=all_pack.probability_profitable,
        probability_down_2pct=all_pack.probability_down_2pct,
        return_q10=all_pack.return_q10,
        return_q50=all_pack.return_q50,
        return_q90=all_pack.return_q90,
        confidence_level=confidence_level,
        top_positive_factors=[item["factor_name"] for item in positive],
        top_negative_factors=[item["factor_name"] for item in negative],
        important_factors=important,
        score_weights=all_pack.score_weights,
    )


def _limit_low_confidence_buys(table: pd.DataFrame, settings: Mapping[str, Any], account: AccountState) -> pd.DataFrame:
    if table.empty:
        return table
    result = table.copy()
    minimum = float(settings.get("minimum_data_completeness", 0.70))
    multiplier = float(settings.get("low_confidence_position_multiplier", 0.50))
    for index, row in result.iterrows():
        requested = int(row.get("recommended_trade_shares") or 0)
        if requested <= 0:
            continue
        completeness = _finite(row.get("data_completeness_score"), 0.0)
        confidence = _finite(row.get("confidence_score"), 0.0)
        if completeness >= minimum and confidence >= 0.60:
            continue
        allowed_multiplier = 0.0 if completeness < 0.50 else multiplier
        allowed = int(requested * allowed_multiplier // account.lot_size) * account.lot_size
        result.at[index, "recommended_trade_shares"] = allowed
        result.at[index, "recommended_target_shares"] = int(row.get("shares") or 0) + allowed
        result.at[index, "recommended_target_weight"] = (
            result.at[index, "recommended_target_shares"] * _finite(row.get("current_price"), 0.0)
            / max(account.total_asset, 1.0)
        )
        result.at[index, "effective_action"] = "NO_TRADE_LOW_CONFIDENCE" if allowed == 0 else "ADD_LIMITED_LOW_CONFIDENCE"
        result.at[index, "recommended_action"] = result.at[index, "effective_action"]
    return result


def _atomic_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def run_production_holding_decision(
    market_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    project_root: str | Path,
    *,
    account: AccountState | None = None,
    config: DecisionConfig | None = None,
    source_notes: list[SourceNote] | None = None,
    save_snapshot: bool = True,
    shap_top_n: int = 10,
) -> ProductionDecisionResult:
    """Predict a whole stock batch with one immutable production package and no fitting."""
    root = Path(project_root).resolve()
    account = (account or AccountState()).normalized()
    decision_config = config or DecisionConfig()
    source_notes = source_notes or []
    package = ProductionModelLoader(root).load()
    governance = package.get("metadata", {}).get("config") or load_governance_config()
    models = package["models"]
    weights = package.get("group_weights", {})
    metadata = dict(package.get("metadata", {}))
    policy_metadata = metadata.get("factor_policy", {}) if isinstance(metadata.get("factor_policy"), Mapping) else {}
    policy_parameters = FactorPolicyParameters.from_mapping(policy_metadata.get("parameters"))
    policy_enabled = bool(policy_metadata.get("parameters")) and str(
        policy_metadata.get("status", "")
    ).lower() == "calibrated"
    factor_columns = dict(package.get("factor_columns", {}))
    expected_hash = str(metadata.get("feature_schema_hash") or "")
    if expected_hash:
        inference_lag = int(metadata.get("external_factor_lag", -1))
        if inference_lag != 0:
            raise ValueError(
                f"Production model external_factor_lag={inference_lag} is incompatible with after-close inference"
            )
        current_hash = feature_schema_hash(
            factor_columns,
            external_factor_lag=0,
            universe_definition=str(metadata.get("universe_definition") or ""),
        )
        if current_hash != expected_hash:
            raise ValueError(
                f"Production feature schema mismatch: model={expected_hash[:12]} runtime={current_hash[:12]}"
            )
        if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported production feature schema version: {metadata.get('feature_schema_version')}"
            )
    else:
        source_notes.append(
            SourceNote(
                "production_schema",
                "legacy_unverified",
                "Legacy model has no timing/schema hash; inference remains compatible but must be retrained.",
            )
        )
    market = enrich_trade_constraints(normalize_market_df(market_df))
    required_model_features = {
        column for model in models.values() for column in getattr(model, "feature_columns", [])
    }
    rank_frame, rank_status = load_rank_cache(root, required_columns=required_model_features)
    if rank_frame.empty and any(
        column.startswith(("market_rank_", "industry_rank_")) for column in required_model_features
    ):
        source_notes.append(SourceNote("full_market_cross_section", "degraded", rank_status))
    elif not rank_frame.empty:
        source_notes.append(SourceNote("full_market_cross_section", "cache", rank_status))
    features = build_features(
        market,
        external_factor_lag=0,
        cross_sectional_rank_frame=rank_frame,
    )
    latest_market = features.sort_values(["date", "code"], kind="stable").groupby("code", sort=False).tail(1)
    holdings = prepare_holdings(holdings_df, latest_market, account)
    oos_metrics = _latest_oos_metrics(package)
    metadata = package.get("metadata", {})
    requirements = metadata.get("source_requirements")
    if not isinstance(requirements, Mapping) or not requirements:
        requirements = source_requirements(required_model_features)
    rows: list[dict[str, Any]] = []
    latest_rows_by_code: dict[str, pd.Series] = {}

    for holding_row in holdings.to_dict("records"):
        code = str(holding_row["code"]).zfill(6)
        latest = latest_market.loc[latest_market["code"].eq(code)]
        if latest.empty:
            continue
        latest_row = latest.iloc[-1].copy()
        latest_rows_by_code[code] = latest_row
        for column in required_model_features:
            if column not in latest_row.index:
                latest_row[column] = np.nan
        all_model = models["all_factor"]
        try:
            all_pack = all_model.predict_one(latest_row)
            group_packs = {
                group: model.predict_one(latest_row)
                for group, model in models.items()
                if group != "all_factor"
            }
        except Exception as exc:
            source_notes.append(
                SourceNote(
                    "production_inference",
                    "failed",
                    f"{code}: {type(exc).__name__}: {exc}",
                )
            )
            continue
        gated_weights, gate_status = apply_event_gates(weights, latest_row, governance)
        group_probabilities = {group: pack.probability_up for group, pack in group_packs.items()}
        group_returns = {group: pack.expected_open_to_open_return for group, pack in group_packs.items()}
        probability = combine_group_predictions(group_probabilities, gated_weights)
        expected_return = combine_group_predictions(group_returns, gated_weights)
        if not np.isfinite(probability):
            probability = all_pack.probability_up
        if not np.isfinite(expected_return):
            expected_return = all_pack.expected_open_to_open_return

        availability = data_availability(latest_row, required_model_features)
        completeness, availability_details = required_data_completeness(
            availability,
            requirements,
            data_date=latest_row.get("date"),
        )
        confidence, confidence_level = confidence_score(
            probability, expected_return, group_probabilities, completeness, oos_metrics
        )
        positive: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []
        important: list[str] = []
        prediction = _prediction_pack(
            all_pack, probability, expected_return, confidence_level, positive, negative, important
        )
        holding = Holding(
            code=code,
            shares=int(holding_row.get("shares") or 0),
            available_shares=int(holding_row.get("available_shares") or 0),
            average_cost=float(holding_row.get("average_cost") or 0),
            current_price=float(holding_row.get("current_price") or latest_row["close"]),
            position_value=float(holding_row.get("position_value") or 0),
            position_weight=float(holding_row.get("position_weight") or 0),
            available_shares_known=bool(holding_row.get("available_shares_known", False)),
            today_bought_shares=int(holding_row.get("today_bought_shares") or 0),
            holding_days=int(holding_row.get("holding_days") or 0),
            industry=str(holding_row.get("industry") or latest_row.get("industry", "")),
            name=str(holding_row.get("name") or latest_row.get("name", "")),
        )
        scores = score_actions(holding, latest_row, prediction, account, decision_config)
        policy_candidates: list[dict[str, float]] = []
        if policy_enabled:
            current_weight = holding.position_value / max(account.total_asset, 1.0)
            policy_target, policy_utility, policy_candidates = choose_target_weight(
                probability_up=prediction.probability_up,
                predicted_return=prediction.expected_open_to_open_return,
                probability_down=prediction.probability_down_2pct,
                return_q10=prediction.return_q10,
                current_weight=current_weight,
                parameters=policy_parameters,
                maximum=min(account.max_single_position_weight, policy_parameters.max_single_weight),
            )
            selected = score_policy_target(
                holding,
                latest_row,
                prediction,
                account,
                policy_target,
                policy_utility,
            )
        else:
            selected = choose_action(scores, decision_config.minimum_action_edge)
        output = _output_row(
            holding, latest_row, prediction, selected, {score.requested_action: score for score in scores}
        )
        if policy_enabled:
            output["reason"] = (
                selected.reason
                + f" 模型预测上涨概率 {prediction.probability_up * 100:.1f}%，"
                + f"预期收益 {prediction.expected_open_to_open_return * 100:.2f}%，"
                + f"Q10 {prediction.return_q10 * 100:.2f}%。"
            )
        output.update(
            {
                "model_version": package["version"],
                "model_backend": package.get("metadata", {}).get("model_backend", all_pack and "XGBoost CUDA"),
                "training_end": package.get("metadata", {}).get("latest_window", {}).get("train_end"),
                "data_completeness_score": completeness,
                "confidence_score": confidence,
                "confidence_level": confidence_level,
                "data_availability": availability,
                "data_source_details": availability_details,
                "event_status": gate_status,
                "group_weights": gated_weights,
                "group_predictions": {
                    group: {
                        "probability_up": pack.probability_up,
                        "expected_return": pack.expected_open_to_open_return,
                    }
                    for group, pack in group_packs.items()
                },
                "all_factor_prediction": {
                    "probability_up": all_pack.probability_up,
                    "expected_return": all_pack.expected_open_to_open_return,
                },
                "top_positive_factor_details": positive,
                "top_negative_factor_details": negative,
                "important_factor_details": important,
                "factor_policy_enabled": policy_enabled,
                "factor_policy_target_weight": selected.target_weight if policy_enabled else np.nan,
                "factor_policy_utility": selected.utility_score if policy_enabled else np.nan,
                "factor_policy_candidates": policy_candidates,
                "factor_policy_report": policy_metadata,
            }
        )
        rows.append(output)

    table = pd.DataFrame(rows)
    drift_report: dict[str, Any] = {}
    if not table.empty:
        try:
            history_predictions = table.copy()
            for column in sorted(required_model_features):
                history_predictions[f"feature__{column}"] = [
                    latest_rows_by_code.get(str(code).zfill(6), pd.Series(dtype=float)).get(column, np.nan)
                    for code in history_predictions["code"]
                ]
            history = update_prediction_history(
                root,
                history_predictions,
                market,
                str(package["version"]),
            )
            drift_report = assess_model_drift(
                history,
                str(package["version"]),
                governance.get("drift_monitoring", {}),
                production_training_end=metadata.get("production_training_end"),
            )
        except Exception as exc:
            drift_report = {
                "status": "monitor_error",
                "review_required": False,
                "severe": False,
                "metrics": {},
                "reasons": [f"{type(exc).__name__}: {exc}"],
            }
            source_notes.append(
                SourceNote("model_drift_monitor", "failed", drift_report["reasons"][0])
            )
        _atomic_snapshot(
            root / "cache" / "ml_drift_report.json",
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "model_version": str(package["version"]),
                **drift_report,
            },
        )
        table["drift_status"] = str(drift_report.get("status") or "-")
        multiplier = 1.0
        if drift_report.get("severe"):
            multiplier = float(
                governance.get("drift_monitoring", {}).get("severe_confidence_multiplier", 0.35)
            )
        elif drift_report.get("review_required"):
            multiplier = float(
                governance.get("drift_monitoring", {}).get("review_confidence_multiplier", 0.70)
            )
        if multiplier < 1.0:
            table["confidence_score"] = pd.to_numeric(
                table["confidence_score"], errors="coerce"
            ) * multiplier
            table["confidence_level"] = "model_review"
        if drift_report.get("severe"):
            blocked = pd.to_numeric(table["recommended_trade_shares"], errors="coerce").gt(0)
            table.loc[blocked, "effective_action"] = "NO_TRADE_MODEL_REVIEW"
            table.loc[blocked, "recommended_action"] = "NO_TRADE_MODEL_REVIEW"
            table.loc[blocked, "recommended_trade_shares"] = 0
            table.loc[blocked, "recommended_target_shares"] = table.loc[blocked, "shares"]
            table.loc[blocked, "recommended_target_weight"] = table.loc[blocked, "position_weight"]
        table = _limit_low_confidence_buys(table, governance.get("confidence", {}), account)
        table = apply_account_constraints(table, account)
        table = table.sort_values(
            ["recommended_action", "utility_score"], ascending=[True, False], kind="stable"
        ).reset_index(drop=True)
        for index in table.index[: max(int(shap_top_n), 0)]:
            code = str(table.at[index, "code"]).zfill(6)
            latest_row = latest_rows_by_code.get(code)
            if latest_row is None:
                continue
            positive, negative, important = _shap_details(models["all_factor"], latest_row)
            table.at[index, "top_positive_factor_details"] = positive
            table.at[index, "top_negative_factor_details"] = negative
            table.at[index, "important_factor_details"] = important
            table.at[index, "top_positive_factors"] = ", ".join(
                item["factor_name"] for item in positive
            )
            table.at[index, "top_negative_factors"] = ", ".join(
                item["factor_name"] for item in negative
            )
            table.at[index, "important_factors"] = ", ".join(important)

    snapshot_path: Path | None = root / "cache" / "ml_prediction_snapshot.json" if save_snapshot else None
    if snapshot_path is not None:
        snapshot_rows = []
        for row in table.to_dict("records"):
            snapshot_rows.append(
                {
                    "prediction_date": datetime.now().isoformat(timespec="seconds"),
                    "model_version": package["version"],
                    "data_date": row.get("date"),
                    "symbol": row.get("code"),
                    "probability_up": row.get("probability_up"),
                    "expected_return": row.get("expected_open_to_open_return"),
                    "risk_score": row.get("display_score"),
                    "confidence_score": row.get("confidence_score"),
                    "target_weight": row.get("recommended_target_weight"),
                    "trade_shares": row.get("recommended_trade_shares"),
                    "recommended_action": row.get("recommended_action"),
                    "top_positive_factors": row.get("top_positive_factor_details", []),
                    "top_negative_factors": row.get("top_negative_factor_details", []),
                    "data_availability": row.get("data_availability", {}),
                    "data_source_details": row.get("data_source_details", {}),
                    "drift_status": row.get("drift_status"),
                }
            )
        _atomic_snapshot(
            snapshot_path,
            {
                "snapshot_time": datetime.now().isoformat(timespec="seconds"),
                "model_version": package["version"],
                "results": snapshot_rows,
            },
        )

    return ProductionDecisionResult(
        table=table,
        metrics=oos_metrics,
        feature_columns=list(package.get("factor_columns", {}).get("all_factor", [])),
        source_notes=source_notes,
        model_metadata=dict(package.get("metadata", {})),
        model_version=str(package["version"]),
        snapshot_path=str(snapshot_path) if snapshot_path is not None else None,
        drift_report=drift_report,
    )


def production_result_to_jsonable(result: ProductionDecisionResult) -> dict[str, Any]:
    return {
        "table": result.table.to_dict("records"),
        "metrics": result.metrics,
        "feature_columns": result.feature_columns,
        "source_notes": [asdict(note) for note in result.source_notes],
        "model_metadata": result.model_metadata,
        "model_version": result.model_version,
        "snapshot_path": result.snapshot_path,
        "drift_report": result.drift_report,
    }
