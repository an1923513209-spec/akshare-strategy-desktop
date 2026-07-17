"""Build the desktop ML pool panel and run the explicit monthly training workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as engine
from ml_decision.data_sources import fetch_external_factor_frame, merge_external_factors
from ml_decision.workflows import monthly_train


def _pool_symbols() -> list[tuple[str, str]]:
    path = engine.CACHE_DIR / "ml_stock_pool.json"
    if not path.exists():
        raise FileNotFoundError(f"ML stock pool does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ML stock pool must be a JSON object")
    items = payload.get("items", payload)
    if not isinstance(items, dict):
        raise ValueError("ML stock pool items must be a JSON object")
    rows: list[tuple[str, str]] = []
    for raw_code, item in items.items():
        candidate = item.get("symbol", raw_code) if isinstance(item, dict) else raw_code
        try:
            code = engine.normalize_symbol(candidate)
        except (TypeError, ValueError):
            continue
        name = str(item.get("name") or "") if isinstance(item, dict) else ""
        rows.append((code, name))
    if not rows:
        raise ValueError("ML stock pool does not contain any valid six-digit A-share code")
    return rows


def build_training_panel(start: str = "20200101", adjust: str = "qfq") -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    symbols = _pool_symbols()
    for index, (code, name) in enumerate(symbols, start=1):
        print(f"[market {index}/{len(symbols)}] {code} {name}", flush=True)
        try:
            data = engine.cached_data(code, start, adjust).copy()
            if data.empty or len(data) < 80:
                raise ValueError(f"only {len(data)} rows")
            frame = data.reset_index()
            frame = frame.rename(
                columns={
                    frame.columns[0]: "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            frame["code"] = code
            frame["name"] = name
            frame["amount"] = frame["volume"] * frame["close"]
            frame["market_data_available"] = 1.0
            frames.append(
                frame[[
                    "date", "code", "name", "open", "high", "low", "close", "volume", "amount",
                    "market_data_available",
                ]]
            )
        except Exception as exc:
            errors.append(f"{code}: {type(exc).__name__}: {exc}")
            print(f"[skip] {errors[-1]}", flush=True)
    if not frames:
        raise ValueError("No stock in the ML pool has enough market history")
    market = pd.concat(frames, ignore_index=True, sort=False)
    codes = sorted(market["code"].unique())
    print(f"[external] fetching cached/free factors for {len(codes)} stocks", flush=True)
    external, notes = fetch_external_factor_frame(codes, force_refresh=False, market_df=market)
    for note in notes:
        print(f"[source] {note.source}: {note.status}: {note.detail}", flush=True)
    return merge_external_factors(market, external), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop monthly production-model training")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "ml_training_panel.parquet"))
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--version")
    args = parser.parse_args()
    panel, errors = build_training_panel(args.start, args.adjust)
    data_path = Path(args.data)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(data_path, index=False)
    print(f"[panel] {len(panel)} rows -> {data_path}", flush=True)
    if errors:
        print(f"[panel] skipped {len(errors)} stocks", flush=True)
    result = monthly_train(panel, PROJECT_ROOT, version=args.version)
    print(f"[candidate] {result['model_path']}", flush=True)


if __name__ == "__main__":
    main()
