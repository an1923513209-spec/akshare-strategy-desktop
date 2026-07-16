from __future__ import annotations

import numpy as np
import pandas as pd

from ml_decision import AccountState, DecisionConfig, run_holding_decision
from ml_decision.features import add_labels, build_features, feature_columns


def synthetic_market(rows: int = 180, codes: tuple[str, ...] = ("000001", "000002", "000003")) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=rows)
    frames = []
    for i, code in enumerate(codes):
        ret = rng.normal(0.0006 + i * 0.0001, 0.018, size=rows)
        close = 10 * np.exp(np.cumsum(ret))
        open_ = close * (1 + rng.normal(0, 0.004, size=rows))
        high = np.maximum(open_, close) * (1 + rng.random(rows) * 0.015)
        low = np.minimum(open_, close) * (1 - rng.random(rows) * 0.015)
        volume = rng.integers(800000, 5000000, size=rows)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "code": code,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                    "large_net_ratio": rng.normal(0, 0.03, size=rows),
                    "main_net_ratio": rng.normal(0, 0.04, size=rows),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_features_and_labels_build() -> None:
    market = synthetic_market()
    features = build_features(market)
    dataset = add_labels(features, round_trip_cost=0.002, down_threshold=-0.02)
    cols = feature_columns(dataset)
    assert "ret_5" in cols
    assert "large_net_ratio" in cols
    assert dataset["next_open_to_next_open_return"].notna().sum() > 100


def test_run_holding_decision_outputs_actions() -> None:
    market = synthetic_market()
    latest = market.sort_values("date").groupby("code").tail(1)
    holdings = latest[["code", "close"]].rename(columns={"close": "average_cost"})
    holdings["shares"] = [100, 0, 200]
    holdings["available_shares"] = holdings["shares"]
    result = run_holding_decision(
        market,
        holdings,
        account=AccountState(cash=100000, total_asset=120000),
        config=DecisionConfig(train_min_rows=120, use_shap=False),
    )
    assert not result.table.empty
    assert {"recommended_action", "probability_up", "return_q10", "reason"}.issubset(result.table.columns)
