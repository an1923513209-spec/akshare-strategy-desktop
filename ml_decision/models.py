"""Model training and inference for the next-session decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - compatibility with transitional sklearn releases
    from sklearn.calibration import FrozenEstimator


LOGGER = logging.getLogger(__name__)


def configured_xgboost_device() -> str:
    """Return the explicit CUDA device used by every production tree model."""
    raw = str(os.environ.get("ML_XGBOOST_DEVICE", "cuda:0") or "cuda:0").strip().lower()
    if raw == "cuda":
        return "cuda:0"
    if not raw.startswith("cuda:"):
        raise RuntimeError(f"ML_XGBOOST_DEVICE 必须是 CUDA 设备，例如 cuda:0；当前为 {raw!r}")
    return raw


def configured_xgboost_cpu_threads() -> int:
    """Keep host-side XGBoost work bounded so it cannot saturate the desktop."""
    try:
        value = int(os.environ.get("ML_XGBOOST_CPU_THREADS", "2") or "2")
    except ValueError:
        value = 2
    return min(max(value, 1), 8)


def xgboost_backend_label(runtime: dict[str, Any] | None = None) -> str:
    status = runtime or require_xgboost_cuda()
    cuda_version = status.get("cuda_version") or "unknown"
    return (
        f"XGBoost {status.get('xgboost_version', '?')} CUDA {cuda_version} "
        f"({status.get('actual_device', '?')})"
    )


@lru_cache(maxsize=4)
def require_xgboost_cuda(device: str | None = None) -> dict[str, Any]:
    """Run a real fit and verify that XGBoost created a CUDA booster.

    Checking ``build_info`` alone is insufficient: a CUDA-enabled wheel can
    still start with an unavailable or incorrectly selected device.  The tiny
    probe is cached once per process and prevents silent CPU-only training.
    """
    selected = device or configured_xgboost_device()
    try:
        import xgboost as xgb
    except Exception as exc:
        raise RuntimeError("未安装 xgboost，无法使用 GPU 训练") from exc
    try:
        info = xgb.build_info()
    except Exception as exc:
        raise RuntimeError("无法读取 xgboost 编译信息，不能确认 CUDA 可用") from exc
    if not bool(info.get("USE_CUDA")):
        raise RuntimeError("当前 xgboost 不是 CUDA 版本，已拒绝退回 CPU 训练")

    probe_x = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32
    )
    probe_y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    try:
        probe = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device=selected,
            n_jobs=1,
            verbosity=0,
        )
        probe.fit(probe_x, probe_y)
        booster_config = json.loads(probe.get_booster().save_config())
        actual_device = str(booster_config["learner"]["generic_param"]["device"])
    except Exception as exc:
        raise RuntimeError(f"CUDA 训练自检失败（请求设备 {selected}）：{exc}") from exc
    if not actual_device.startswith("cuda:"):
        raise RuntimeError(
            f"XGBoost 实际设备为 {actual_device}，已拒绝静默退回 CPU"
        )
    return {
        "backend": "XGBoost CUDA",
        "requested_device": selected,
        "actual_device": actual_device,
        "xgboost_version": str(xgb.__version__),
        "cuda_version": ".".join(str(part) for part in info.get("CUDA_VERSION", [])),
        "cpu_threads": configured_xgboost_cpu_threads(),
    }


@dataclass(slots=True)
class PurgedDateSplits:
    """Four chronological date sets separated by label-safe embargo dates."""

    train_dates: pd.DatetimeIndex
    calibration_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    purge_days: int = 2

    def frame(self, dataset: pd.DataFrame, part: str) -> pd.DataFrame:
        dates = getattr(self, f"{part}_dates")
        normalized = pd.to_datetime(dataset["date"], errors="coerce").dt.normalize()
        return dataset.loc[normalized.isin(dates)].sort_values(["date", "code"], kind="stable")


def make_purged_date_splits(dataset: pd.DataFrame, purge_days: int = 2) -> PurgedDateSplits:
    """Split unique dates, never rows, and embargo every adjacent partition."""
    dates = pd.DatetimeIndex(
        pd.to_datetime(dataset["date"], errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    )
    purge_days = max(int(purge_days), 2)
    usable_count = len(dates) - purge_days * 3
    if usable_count < 32:
        raise ValueError(f"Not enough unique trade dates for purged split: {len(dates)}")
    train_count = max(18, int(usable_count * 0.55))
    calibration_count = max(4, int(usable_count * 0.15))
    validation_count = max(4, int(usable_count * 0.15))
    test_count = usable_count - train_count - calibration_count - validation_count
    if test_count < 4:
        deficit = 4 - test_count
        train_count -= deficit
        test_count = 4

    cursor = 0
    train = dates[cursor : cursor + train_count]
    cursor += train_count + purge_days
    calibration = dates[cursor : cursor + calibration_count]
    cursor += calibration_count + purge_days
    validation = dates[cursor : cursor + validation_count]
    cursor += validation_count + purge_days
    test = dates[cursor : cursor + test_count]
    return PurgedDateSplits(train, calibration, validation, test, purge_days)


@dataclass(slots=True)
class PredictionPack:
    """Predictions required by the action scorer."""

    expected_gap_return: float
    expected_open_to_open_return: float
    probability_up: float
    probability_profitable: float
    probability_down_2pct: float
    return_q10: float
    return_q50: float
    return_q90: float
    confidence_level: str
    top_positive_factors: list[str]
    top_negative_factors: list[str]
    important_factors: list[str]
    score_weights: dict[str, float]


class _LegacyNextSessionModel:
    """Train several time-safe models and produce one prediction row."""

    MIN_USABLE_ROWS = 50

    def __init__(
        self,
        calibration_method: str = "sigmoid",
        random_state: int = 42,
        use_shap: bool = True,
        gpu_device: str | None = None,
        host_cpu_threads: int | None = None,
    ) -> None:
        self.calibration_method = calibration_method if calibration_method in {"sigmoid", "isotonic"} else "sigmoid"
        self.random_state = random_state
        self.use_shap = bool(use_shap)
        self.gpu_device = gpu_device or configured_xgboost_device()
        self.host_cpu_threads = min(
            max(int(host_cpu_threads or configured_xgboost_cpu_threads()), 1), 8
        )
        self.feature_columns: list[str] = []
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self._calibration_metrics: dict[str, Any] = {}
        self._uncalibrated_classifiers: dict[str, Any] = {}

    def fit(self, dataset: pd.DataFrame, features: list[str]) -> "NextSessionModel":
        """Fit all required models using chronological train/validation slices."""
        self.feature_columns = list(features)
        usable = dataset.dropna(subset=["next_open_to_next_open_return"]).sort_values("date")
        if len(usable) < self.MIN_USABLE_ROWS:
            raise ValueError(f"训练样本不足，至少需要约 {self.MIN_USABLE_ROWS} 行有效标签")
        split = max(35, int(len(usable) * 0.78))
        split = min(split, max(1, len(usable) - 10))
        train = usable.iloc[:split]
        valid = usable.iloc[split:]
        x_train = train[self.feature_columns]
        x_valid = valid[self.feature_columns]

        self.models["gap"] = self._fit_regressor(x_train, train["next_gap_return"])
        self.models["open_to_open"] = self._fit_regressor(x_train, train["next_open_to_next_open_return"])
        self.models["up"] = self._fit_classifier(x_train, train["label_up"], x_valid, valid["label_up"])
        self.models["profitable"] = self._fit_classifier(x_train, train["label_profitable"], x_valid, valid["label_profitable"])
        self.models["down_2pct"] = self._fit_classifier(x_train, train["label_down_2pct"], x_valid, valid["label_down_2pct"])
        self.models["q10"] = self._fit_quantile(x_train, train["next_open_to_next_open_return"], 0.10)
        self.models["q50"] = self._fit_quantile(x_train, train["next_open_to_next_open_return"], 0.50)
        self.models["q90"] = self._fit_quantile(x_train, train["next_open_to_next_open_return"], 0.90)

        self.metrics = self._score_validation(valid, x_valid)
        self.metrics["model_backend"] = xgboost_backend_label()
        self.metrics["train_rows"] = float(len(train))
        self.metrics["valid_rows"] = float(len(valid))
        return self

    def predict_one(self, latest_row: pd.Series) -> PredictionPack:
        """Predict one currently held or candidate stock."""
        x = pd.DataFrame([latest_row[self.feature_columns].to_dict()])
        expected_gap = float(self.models["gap"].predict(x)[0])
        expected_ret = float(self.models["open_to_open"].predict(x)[0])
        prob_up = self._predict_proba("up", x)
        prob_profitable = self._predict_proba("profitable", x)
        prob_down = self._predict_proba("down_2pct", x)
        q10 = float(self.models["q10"].predict(x)[0])
        q50 = float(self.models["q50"].predict(x)[0])
        q90 = float(self.models["q90"].predict(x)[0])
        positive, negative = self._factor_directions()
        confidence = "high" if len(self.metrics) and self.metrics.get("auc_up", 0.5) >= 0.57 else "medium"
        if self.metrics.get("auc_up", 0.5) < 0.52:
            confidence = "low"
        if self.metrics.get("train_rows", 999) < 80 or self.metrics.get("valid_rows", 999) < 20:
            confidence = "low"
        score_weights = {
            "probability": float(self.metrics.get("score_weight_probability", 0.45)),
            "expected": float(self.metrics.get("score_weight_expected", 0.35)),
            "risk": float(self.metrics.get("score_weight_risk", 0.20)),
        }
        return PredictionPack(
            expected_gap_return=expected_gap,
            expected_open_to_open_return=expected_ret,
            probability_up=prob_up,
            probability_profitable=prob_profitable,
            probability_down_2pct=prob_down,
            return_q10=q10,
            return_q50=q50,
            return_q90=q90,
            confidence_level=confidence,
            top_positive_factors=positive,
            top_negative_factors=negative,
            score_weights=score_weights,
        )

    def _fit_regressor(self, x: pd.DataFrame, y: pd.Series) -> Pipeline:
        model = self._xgb_regressor()
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        pipe.fit(x, y.fillna(0.0))
        return pipe

    def _fit_quantile(self, x: pd.DataFrame, y: pd.Series, alpha: float) -> Pipeline:
        model = self._xgb_quantile(alpha)
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        pipe.fit(x, y.fillna(0.0))
        return pipe

    def _fit_classifier(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> Pipeline:
        if y_train.nunique() < 2:
            raise ValueError("训练标签只有一种，无法训练 GPU 分类模型")
        base = self._xgb_classifier()
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", base)])
        pipe.fit(x_train, y_train.astype(int))
        if y_valid.nunique() >= 2 and len(y_valid) >= 40:
            try:
                calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method=self.calibration_method)
                calibrated.fit(x_valid, y_valid.astype(int))
                return calibrated
            except Exception:
                LOGGER.exception("Legacy probability calibration failed")
                return pipe
        return pipe

    def _predict_proba(self, name: str, x: pd.DataFrame) -> float:
        model = self.models[name]
        try:
            return float(model.predict_proba(x)[0, 1])
        except Exception:
            return float(np.clip(model.predict(x)[0], 0, 1))

    def _score_validation(self, valid: pd.DataFrame, x_valid: pd.DataFrame) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if valid.empty:
            return metrics
        pred = self.models["open_to_open"].predict(x_valid)
        y = valid["next_open_to_next_open_return"].fillna(0.0)
        metrics["mae"] = float(mean_absolute_error(y, pred))
        metrics["rmse"] = float(mean_squared_error(y, pred) ** 0.5)
        metrics.update(self._validate_score_weights(valid, x_valid, pd.Series(pred, index=valid.index), y))
        if valid["label_up"].nunique() >= 2:
            try:
                metrics["auc_up"] = float(roc_auc_score(valid["label_up"], self.models["up"].predict_proba(x_valid)[:, 1]))
            except Exception:
                metrics["auc_up"] = 0.5
        return metrics

    def _validate_score_weights(
        self,
        valid: pd.DataFrame,
        x_valid: pd.DataFrame,
        pred_ret: pd.Series,
        realized_ret: pd.Series,
    ) -> dict[str, float]:
        candidates = [
            {"probability": 0.50, "expected": 0.30, "risk": 0.20},
            {"probability": 0.40, "expected": 0.40, "risk": 0.20},
            {"probability": 0.35, "expected": 0.45, "risk": 0.20},
            {"probability": 0.45, "expected": 0.25, "risk": 0.30},
            {"probability": 0.30, "expected": 0.35, "risk": 0.35},
            {"probability": 0.60, "expected": 0.25, "risk": 0.15},
        ]
        if len(valid) < 12:
            best = candidates[0]
            return {
                "score_weight_probability": best["probability"],
                "score_weight_expected": best["expected"],
                "score_weight_risk": best["risk"],
                "score_weight_validation_edge": 0.0,
            }
        prob_up = pd.Series(self.models["up"].predict_proba(x_valid)[:, 1], index=valid.index)
        try:
            prob_down = pd.Series(self.models["down_2pct"].predict_proba(x_valid)[:, 1], index=valid.index)
        except Exception:
            prob_down = pd.Series(0.0, index=valid.index)
        q10 = pd.Series(self.models["q10"].predict(x_valid), index=valid.index)
        prob_component = (prob_up - 0.5) * 2.0
        expected_component = np.tanh(pd.to_numeric(pred_ret, errors="coerce") / 0.03)
        q10_risk = np.clip(-pd.to_numeric(q10, errors="coerce"), 0.0, 0.08) / 0.08
        risk_component = (pd.to_numeric(prob_down, errors="coerce").fillna(0.0) + q10_risk.fillna(0.0)) / 2.0
        realized = pd.to_numeric(realized_ret, errors="coerce").fillna(0.0)

        best = candidates[0]
        best_edge = -np.inf
        for weights in candidates:
            score = (
                weights["probability"] * prob_component
                + weights["expected"] * expected_component
                - weights["risk"] * risk_component
            )
            frame = pd.DataFrame({"score": score, "realized": realized}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(frame) < 12:
                continue
            cut = max(3, len(frame) // 3)
            ordered = frame.sort_values("score")
            low_mean = float(ordered.head(cut)["realized"].mean())
            high_mean = float(ordered.tail(cut)["realized"].mean())
            try:
                corr = float(frame["score"].corr(frame["realized"]))
            except Exception:
                corr = 0.0
            edge = high_mean - low_mean + 0.01 * (corr if np.isfinite(corr) else 0.0)
            if edge > best_edge:
                best = weights
                best_edge = edge
        return {
            "score_weight_probability": best["probability"],
            "score_weight_expected": best["expected"],
            "score_weight_risk": best["risk"],
            "score_weight_validation_edge": float(best_edge if np.isfinite(best_edge) else 0.0),
        }

    def _factor_directions(self) -> tuple[list[str], list[str]]:
        model = self.models.get("open_to_open")
        importances = None
        if hasattr(model, "named_steps"):
            estimator = model.named_steps.get("model")
            importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            return [], []
        pairs = sorted(zip(self.feature_columns, importances), key=lambda item: float(item[1]), reverse=True)[:6]
        names = [name for name, _value in pairs]
        return [], []

    def _require_xgboost_cuda(self) -> dict[str, Any]:
        return require_xgboost_cuda(self.gpu_device)

    def _xgb_regressor(self) -> Any:
        runtime = self._require_xgboost_cuda()
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=180,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device=runtime["actual_device"],
            n_jobs=self.host_cpu_threads,
            random_state=self.random_state,
            verbosity=0,
        )

    def _xgb_quantile(self, alpha: float) -> Any:
        runtime = self._require_xgboost_cuda()
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=160,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device=runtime["actual_device"],
            n_jobs=self.host_cpu_threads,
            random_state=self.random_state,
            verbosity=0,
        )

    def _xgb_classifier(self) -> Any:
        runtime = self._require_xgboost_cuda()
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=180,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device=runtime["actual_device"],
            n_jobs=self.host_cpu_threads,
            random_state=self.random_state,
            verbosity=0,
        )


class NextSessionModel(_LegacyNextSessionModel):
    """Leakage-safe model implementation used by the decision engine."""

    def fit(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        splits: PurgedDateSplits | None = None,
    ) -> "NextSessionModel":
        self.feature_columns = list(features)
        usable = dataset.dropna(subset=["next_open_to_next_open_return"]).sort_values(
            ["date", "code"], kind="stable"
        )
        if len(usable) < self.MIN_USABLE_ROWS:
            raise ValueError(f"Not enough labelled rows: {len(usable)} < {self.MIN_USABLE_ROWS}")
        splits = splits or make_purged_date_splits(usable, purge_days=2)
        train = splits.frame(usable, "train")
        calibration = splits.frame(usable, "calibration")
        validation = splits.frame(usable, "validation")
        test = splits.frame(usable, "test")
        if min(map(len, (train, calibration, validation, test))) == 0:
            raise ValueError("Purged date split produced an empty model partition")

        x_train = train[self.feature_columns]
        x_calibration = calibration[self.feature_columns]
        x_validation = validation[self.feature_columns]
        x_test = test[self.feature_columns]
        self._calibration_metrics = {}
        self.models["gap"] = self._fit_regressor(x_train, train["next_gap_return"])
        self.models["open_to_open"] = self._fit_regressor(x_train, train["next_open_to_next_open_return"])
        self.models["up"] = self._fit_classifier(
            "up", x_train, train["label_up"], x_calibration, calibration["label_up"]
        )
        self.models["profitable"] = self._fit_classifier(
            "profitable", x_train, train["label_profitable"], x_calibration, calibration["label_profitable"]
        )
        self.models["down_2pct"] = self._fit_classifier(
            "down_2pct", x_train, train["label_down_2pct"], x_calibration, calibration["label_down_2pct"]
        )
        self.models["q10"] = self._fit_quantile(x_train, train["next_open_to_next_open_return"], 0.10)
        self.models["q50"] = self._fit_quantile(x_train, train["next_open_to_next_open_return"], 0.50)
        self.models["q90"] = self._fit_quantile(x_train, train["next_open_to_next_open_return"], 0.90)

        validation_prediction = pd.Series(
            self.models["open_to_open"].predict(x_validation), index=validation.index
        )
        validation_weights = self._validate_score_weights(
            validation,
            x_validation,
            validation_prediction,
            validation["next_open_to_next_open_return"].fillna(0.0),
        )
        self.metrics = self._score_test(test, x_test)
        self.metrics.update(validation_weights)
        self.metrics.update(self._calibration_metrics)
        self.metrics.update(
            {
                "model_backend": xgboost_backend_label(),
                "train_rows": float(len(train)),
                "calibration_rows": float(len(calibration)),
                "valid_rows": float(len(validation)),
                "test_rows": float(len(test)),
                "purge_days": float(splits.purge_days),
                "split_train_end": str(splits.train_dates.max().date()),
                "split_calibration_start": str(splits.calibration_dates.min().date()),
                "split_validation_start": str(splits.validation_dates.min().date()),
                "split_test_start": str(splits.test_dates.min().date()),
            }
        )
        return self

    def refit_production(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        *,
        frozen_metrics: dict[str, Any] | None = None,
        calibration_days: int = 20,
        purge_days: int = 2,
    ) -> "NextSessionModel":
        """Refit frozen model choices through the latest labelled session.

        Regressors and quantiles use every labelled row.  Classifiers reserve
        only the trailing calibration dates, with a two-session purge, so the
        already-selected calibration method remains effective without running
        model or hyperparameter selection again.
        """
        self.feature_columns = list(features)
        usable = dataset.dropna(subset=["next_open_to_next_open_return"]).sort_values(
            ["date", "code"], kind="stable"
        )
        if len(usable) < self.MIN_USABLE_ROWS:
            raise ValueError(f"Not enough labelled rows for production refit: {len(usable)}")
        dates = pd.DatetimeIndex(pd.to_datetime(usable["date"]).dt.normalize().unique()).sort_values()
        calibration_days = min(max(int(calibration_days), 10), max(len(dates) // 4, 10))
        purge_days = max(int(purge_days), 2)
        calibration_dates = dates[-calibration_days:]
        train_dates = dates[: -(calibration_days + purge_days)]
        train = usable.loc[pd.to_datetime(usable["date"]).dt.normalize().isin(train_dates)]
        calibration = usable.loc[pd.to_datetime(usable["date"]).dt.normalize().isin(calibration_dates)]
        if train.empty or calibration.empty:
            raise ValueError("Production refit split is empty")
        x_all = usable[self.feature_columns]
        x_train = train[self.feature_columns]
        x_calibration = calibration[self.feature_columns]
        self._calibration_metrics = {}
        self.models["gap"] = self._fit_regressor(x_all, usable["next_gap_return"])
        self.models["open_to_open"] = self._fit_regressor(x_all, usable["next_open_to_next_open_return"])
        self.models["up"] = self._fit_classifier(
            "up", x_train, train["label_up"], x_calibration, calibration["label_up"]
        )
        self.models["profitable"] = self._fit_classifier(
            "profitable", x_train, train["label_profitable"], x_calibration, calibration["label_profitable"]
        )
        self.models["down_2pct"] = self._fit_classifier(
            "down_2pct", x_train, train["label_down_2pct"], x_calibration, calibration["label_down_2pct"]
        )
        self.models["q10"] = self._fit_quantile(x_all, usable["next_open_to_next_open_return"], 0.10)
        self.models["q50"] = self._fit_quantile(x_all, usable["next_open_to_next_open_return"], 0.50)
        self.models["q90"] = self._fit_quantile(x_all, usable["next_open_to_next_open_return"], 0.90)
        self.metrics = dict(frozen_metrics or {})
        self.metrics.update(self._calibration_metrics)
        self.metrics.update(
            {
                "production_refit": True,
                "production_rows": float(len(usable)),
                "production_training_end": str(dates.max().date()),
                "production_calibration_start": str(calibration_dates.min().date()),
                "production_calibration_end": str(calibration_dates.max().date()),
                "model_backend": xgboost_backend_label(),
            }
        )
        return self

    def _fit_classifier(
        self,
        name: str,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_calibration: pd.DataFrame,
        y_calibration: pd.Series,
    ) -> Any:
        if y_train.nunique() < 2:
            raise ValueError(f"Training label {name} has only one class")
        pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", self._xgb_classifier())]
        )
        pipe.fit(x_train, y_train.astype(int))
        self._uncalibrated_classifiers[name] = pipe
        status_key = f"calibration_{name}_status"
        if y_calibration.nunique() < 2 or len(y_calibration) < 20:
            self._calibration_metrics[status_key] = "skipped_insufficient_classes_or_rows"
            return pipe
        try:
            before = pipe.predict_proba(x_calibration)[:, 1]
            calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method=self.calibration_method)
            calibrated.fit(x_calibration, y_calibration.astype(int))
            after = calibrated.predict_proba(x_calibration)[:, 1]
            self._calibration_metrics[status_key] = "applied"
            self._calibration_metrics[f"calibration_{name}_brier_before"] = float(
                brier_score_loss(y_calibration.astype(int), before)
            )
            self._calibration_metrics[f"calibration_{name}_brier_after"] = float(
                brier_score_loss(y_calibration.astype(int), after)
            )
            return calibrated
        except Exception as exc:
            LOGGER.exception("Probability calibration failed for %s", name)
            self._calibration_metrics[status_key] = "failed"
            self._calibration_metrics[f"calibration_{name}_error"] = f"{type(exc).__name__}: {exc}"
            return pipe

    def _score_test(self, test: pd.DataFrame, x_test: pd.DataFrame) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if test.empty:
            return metrics
        prediction = self.models["open_to_open"].predict(x_test)
        realized = test["next_open_to_next_open_return"].fillna(0.0)
        metrics["test_mae"] = float(mean_absolute_error(realized, prediction))
        metrics["test_rmse"] = float(mean_squared_error(realized, prediction) ** 0.5)
        if test["label_up"].nunique() >= 2:
            try:
                labels = test["label_up"].astype(int)
                calibrated_probability = self.models["up"].predict_proba(x_test)[:, 1]
                metrics["auc_up"] = float(
                    roc_auc_score(labels, calibrated_probability)
                )
                metrics["test_brier_calibrated"] = float(
                    brier_score_loss(labels, calibrated_probability)
                )
                metrics["test_log_loss_calibrated"] = float(
                    log_loss(labels, np.clip(calibrated_probability, 1e-6, 1 - 1e-6))
                )
                raw_model = self._uncalibrated_classifiers.get("up")
                if raw_model is not None:
                    raw_probability = raw_model.predict_proba(x_test)[:, 1]
                    metrics["test_brier_uncalibrated"] = float(
                        brier_score_loss(labels, raw_probability)
                    )
                    metrics["test_log_loss_uncalibrated"] = float(
                        log_loss(labels, np.clip(raw_probability, 1e-6, 1 - 1e-6))
                    )
                bins = pd.cut(
                    pd.Series(calibrated_probability),
                    bins=np.linspace(0.0, 1.0, 11),
                    include_lowest=True,
                )
                calibration = pd.DataFrame(
                    {"bin": bins, "probability": calibrated_probability, "label": labels.to_numpy()}
                ).groupby("bin", observed=False).agg(
                    count=("label", "size"),
                    mean_probability=("probability", "mean"),
                    actual_rate=("label", "mean"),
                )
                valid_bins = calibration.loc[calibration["count"].gt(0)]
                metrics["test_ece_calibrated"] = float(
                    (
                        valid_bins["count"] / max(valid_bins["count"].sum(), 1)
                        * (valid_bins["mean_probability"] - valid_bins["actual_rate"]).abs()
                    ).sum()
                )
                metrics["test_calibration_bias"] = float(
                    calibrated_probability.mean() - labels.mean()
                )
                metrics["test_reliability_bins_json"] = calibration.reset_index().to_json(
                    orient="records",
                    force_ascii=False,
                )
            except Exception as exc:
                LOGGER.warning("Test AUC failed: %s", exc)
                metrics["auc_up"] = 0.5
        return metrics

    def predict_one(self, latest_row: pd.Series) -> PredictionPack:
        x = pd.DataFrame([latest_row[self.feature_columns].to_dict()])
        positive, negative, important = self._factor_directions(x)
        auc = float(self.metrics.get("auc_up", 0.5))
        confidence = "high" if auc >= 0.57 else "medium" if auc >= 0.52 else "low"
        if self.metrics.get("train_rows", 999) < 80 or self.metrics.get("valid_rows", 999) < 20:
            confidence = "low"
        return PredictionPack(
            expected_gap_return=float(self.models["gap"].predict(x)[0]),
            expected_open_to_open_return=float(self.models["open_to_open"].predict(x)[0]),
            probability_up=self._predict_proba("up", x),
            probability_profitable=self._predict_proba("profitable", x),
            probability_down_2pct=self._predict_proba("down_2pct", x),
            return_q10=float(self.models["q10"].predict(x)[0]),
            return_q50=float(self.models["q50"].predict(x)[0]),
            return_q90=float(self.models["q90"].predict(x)[0]),
            confidence_level=confidence,
            top_positive_factors=positive,
            top_negative_factors=negative,
            important_factors=important,
            score_weights={
                "probability": float(self.metrics.get("score_weight_probability", 0.45)),
                "expected": float(self.metrics.get("score_weight_expected", 0.35)),
                "risk": float(self.metrics.get("score_weight_risk", 0.20)),
            },
        )

    def _factor_directions(self, x: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        if self.use_shap:
            try:
                contributions = self._shap_contributions(x)
                positive = [name for name, value in sorted(contributions.items(), key=lambda item: item[1], reverse=True) if value > 0][:3]
                negative = [name for name, value in sorted(contributions.items(), key=lambda item: item[1]) if value < 0][:3]
                return positive, negative, []
            except Exception as exc:
                LOGGER.warning("Current-row SHAP unavailable; using unsigned importance: %s", exc)
        return [], [], self._important_factors()

    def _shap_contributions(self, x: pd.DataFrame) -> dict[str, float]:
        pipeline = self.models["open_to_open"]
        if not hasattr(pipeline, "named_steps"):
            raise TypeError("Open-to-open model is not an explainable pipeline")
        imputed = pipeline.named_steps["imputer"].transform(x)
        estimator = pipeline.named_steps["model"]
        booster = estimator.get_booster()
        import xgboost as xgb

        values = booster.predict(xgb.DMatrix(imputed, feature_names=self.feature_columns), pred_contribs=True)[0]
        if len(values) != len(self.feature_columns) + 1:
            raise ValueError("Unexpected TreeSHAP contribution width")
        return {name: float(value) for name, value in zip(self.feature_columns, values[:-1])}

    def _important_factors(self) -> list[str]:
        pipeline = self.models.get("open_to_open")
        if not hasattr(pipeline, "named_steps"):
            return []
        importances = getattr(pipeline.named_steps.get("model"), "feature_importances_", None)
        if importances is None:
            return []
        pairs = sorted(zip(self.feature_columns, importances), key=lambda item: float(item[1]), reverse=True)
        return [name for name, _value in pairs[:6]]
