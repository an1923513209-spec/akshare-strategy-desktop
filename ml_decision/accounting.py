"""Account migration, validation and snapshot helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    available_cash: float
    total_asset: float
    holdings_market_value: float
    total_asset_estimated: bool
    calculated_at: str
    market_date: str


def resolve_account_snapshot(
    *,
    available_cash: float,
    holdings_market_value: float,
    total_asset: float | None,
    market_date: str = "",
) -> AccountSnapshot:
    cash = float(available_cash)
    holdings = max(float(holdings_market_value), 0.0)
    if cash < 0:
        raise ValueError("Available cash cannot be negative")
    estimated = total_asset is None
    assets = cash + holdings if estimated else float(total_asset)
    if assets <= 0:
        raise ValueError("Total assets must be positive")
    if assets + 1e-6 < cash:
        raise ValueError("Total assets cannot be below available cash")
    if assets + 1e-6 < cash + holdings:
        raise ValueError("Total assets cannot be below available cash plus current holdings market value")
    return AccountSnapshot(
        available_cash=cash,
        total_asset=assets,
        holdings_market_value=holdings,
        total_asset_estimated=estimated,
        calculated_at=datetime.now().isoformat(timespec="seconds"),
        market_date=str(market_date or ""),
    )


def save_account_snapshot(snapshot: AccountSnapshot, project_root: str | Path) -> Path:
    path = Path(project_root).resolve() / "cache" / "ml_account_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path
