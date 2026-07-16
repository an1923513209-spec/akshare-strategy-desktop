"""Model training and inference for the next-session decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline


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
    score_weights: dict[str, float]


class NextSessionModel:
    """Train several time-safe models and produce one prediction row."""

    MIN_USABLE_ROWS = 50

    def __init__(self, calibration_method: str = "sigmoid", random_state: int = 42) -> None:
        self.calibration_method = calibration_method if calibration_method in {"sigmoid", "isotonic"} else "sigmoid"
        self.random_state = random_state
        self.feature_columns: list[str] = []
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}

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
        self.metrics["model_backend"] = "XGBoost CUDA"
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
                calibrated = CalibratedClassifierCV(pipe, method=self.calibration_method, cv="prefit")
                calibrated.fit(x_valid, y_valid.astype(int))
                return calibrated
            except Exception:
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
        return names[:3], names[3:6]

    def _require_xgboost_cuda(self) -> None:
        try:
            import xgboost as xgb
        except Exception as exc:
            raise RuntimeError("未安装 xgboost，无法使用 GPU 训练") from exc
        try:
            info = xgb.build_info()
        except Exception as exc:
            raise RuntimeError("无法读取 xgboost 编译信息，不能确认 CUDA 可用") from exc
        if not bool(info.get("USE_CUDA")):
            raise RuntimeError("当前 xgboost 不是 CUDA 版本，无法使用 GPU 训练")

    def _xgb_regressor(self) -> Any:
        self._require_xgboost_cuda()
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
            device="cuda",
            random_state=self.random_state,
            verbosity=0,
        )

    def _xgb_quantile(self, alpha: float) -> Any:
        self._require_xgboost_cuda()
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
            device="cuda",
            random_state=self.random_state,
            verbosity=0,
        )

    def _xgb_classifier(self) -> Any:
        self._require_xgboost_cuda()
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
            device="cuda",
            random_state=self.random_state,
            verbosity=0,
        )
