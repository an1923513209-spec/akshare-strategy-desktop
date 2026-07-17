"""Command line entrypoints for production prediction and model governance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_decision.workflows import daily_predict, monthly_train, quarterly_audit
from ml_decision.model_registry import ProductionModelLoader, rollback_production_model


def _read_table(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source, dtype={"code": str})


def main() -> None:
    parser = argparse.ArgumentParser(description="ML daily/monthly/quarterly governance workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)
    daily = subparsers.add_parser("daily-predict", help="load production model; never retrain")
    daily.add_argument("--data", required=True)
    daily.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "daily_predictions.csv"))
    daily.add_argument("--status", choices=("production", "candidate", "previous_production"), default="production")
    monthly = subparsers.add_parser("monthly-train", help="rolling OOS train and save candidate")
    monthly.add_argument("--data", required=True)
    monthly.add_argument("--version")
    quarterly = subparsers.add_parser("quarterly-audit", help="factor quality and group audit")
    quarterly.add_argument("--data", required=True)
    subparsers.add_parser("status", help="show candidate/production model pointers")
    subparsers.add_parser("rollback", help="swap production with previous_production")
    args = parser.parse_args()
    if args.command == "daily-predict":
        frame = _read_table(args.data)
        result = daily_predict(frame, PROJECT_ROOT, status=args.status)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, encoding="utf-8-sig")
        print(output)
    elif args.command == "monthly-train":
        frame = _read_table(args.data)
        result = monthly_train(frame, PROJECT_ROOT, version=args.version)
        print(result["model_path"])
    elif args.command == "quarterly-audit":
        frame = _read_table(args.data)
        result = quarterly_audit(frame, PROJECT_ROOT)
        print({key: len(value) for key, value in result.items()})
    elif args.command == "status":
        print(ProductionModelLoader(PROJECT_ROOT).status())
    else:
        if not rollback_production_model(PROJECT_ROOT):
            raise SystemExit("No previous production model is available for rollback.")
        print(ProductionModelLoader(PROJECT_ROOT).status())


if __name__ == "__main__":
    main()
