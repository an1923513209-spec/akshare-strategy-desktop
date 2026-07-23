"""Reusable full-universe cross-sectional factor ranks.

Desktop selections are never a valid proxy for the A-share universe.  This
module therefore keeps rank construction explicit and cache-backed: callers
must provide a sufficiently broad universe before ranks are created.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


RANK_BASE_COLUMNS = (
    "ret_5",
    "ret_20",
    "volatility_20",
    "breakout_gap_20",
    "support_gap_20",
    "bias_20",
    "rsi_6",
    "atr_pct",
    "volume_ratio_20",
    "amount_ratio_20",
)
MARKET_RANK_COLUMNS = tuple(f"market_rank_{name}" for name in RANK_BASE_COLUMNS)
INDUSTRY_RANK_COLUMNS = tuple(f"industry_rank_{name}" for name in RANK_BASE_COLUMNS)
ALL_RANK_COLUMNS = MARKET_RANK_COLUMNS + INDUSTRY_RANK_COLUMNS
DEFAULT_MIN_UNIVERSE_SIZE = 500


def rank_cache_path(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "cache" / "ml_cross_sectional_ranks.parquet"


def compute_full_universe_ranks(
    feature_frame: pd.DataFrame,
    *,
    minimum_universe_size: int = DEFAULT_MIN_UNIVERSE_SIZE,
) -> pd.DataFrame:
    """Compute date ranks only when the supplied frame is a broad universe."""
    required = {"date", "code", *RANK_BASE_COLUMNS}
    missing = sorted(required.difference(feature_frame.columns))
    if missing:
        raise ValueError(f"Cross-sectional rank input is missing columns: {missing}")
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    counts = frame.groupby("date")["code"].nunique()
    valid_dates = counts[counts >= max(int(minimum_universe_size), 2)].index
    frame = frame.loc[frame["date"].isin(valid_dates)].copy()
    if frame.empty:
        raise ValueError(
            f"No date contains at least {minimum_universe_size} stocks; local ranks are forbidden"
        )
    result = frame[["date", "code"]].copy()
    for column in RANK_BASE_COLUMNS:
        result[f"market_rank_{column}"] = frame.groupby("date")[column].rank(pct=True)
        if "industry" in frame.columns:
            result[f"industry_rank_{column}"] = frame.groupby(["date", "industry"])[column].rank(pct=True)
    return result.drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"])


def save_rank_cache(frame: pd.DataFrame, project_root: str | Path) -> Path:
    path = rank_cache_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path


def load_rank_cache(
    project_root: str | Path,
    *,
    required_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, str]:
    """Load an immutable rank cache, returning an explicit degradation reason."""
    path = rank_cache_path(project_root)
    if not path.exists():
        return pd.DataFrame(), f"full-market rank cache is missing: {path}"
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return pd.DataFrame(), f"full-market rank cache cannot be read: {type(exc).__name__}: {exc}"
    if not {"date", "code"}.issubset(frame.columns):
        return pd.DataFrame(), "full-market rank cache has no date/code key"
    required = [name for name in required_columns if name.startswith(("market_rank_", "industry_rank_"))]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        return pd.DataFrame(), f"full-market rank cache is missing model fields: {missing}"
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if frame.duplicated(["date", "code"]).any():
        return pd.DataFrame(), "full-market rank cache has duplicate date/code keys"
    return frame, f"full-market rank cache loaded: {len(frame)} rows"
