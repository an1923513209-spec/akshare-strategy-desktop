"""Example CLI for the A-share next-session holding decision engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml_decision import AccountState, DecisionConfig, run_holding_decision
from ml_decision.data_sources import fetch_external_factor_frame, merge_external_factors
from scripts.data_utils import load_a_share_daily, normalize_symbol


def parse_symbols(text: str) -> list[str]:
    """Parse comma/space separated A-share symbols."""
    for sep in ("\n", "\r", "\t", ",", "，", ";", "；", "、", "|"):
        text = text.replace(sep, " ")
    return [normalize_symbol(token) for token in text.split() if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="A股持仓次日操作决策示例")
    parser.add_argument("--symbols", required=True, help="股票代码，逗号或空格分隔")
    parser.add_argument("--holdings", default="", help="CSV: code,shares,available_shares,average_cost")
    parser.add_argument("--cash", type=float, default=100000)
    parser.add_argument("--total-asset", type=float, default=100000)
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--no-external", action="store_true", help="不拉资金流/新闻/机构免费因子")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    frames = []
    for symbol in symbols:
        data = load_a_share_daily(symbol, start=args.start, adjust=args.adjust)
        frame = data.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        frame["code"] = symbol
        frame["amount"] = frame["volume"] * frame["close"]
        frames.append(frame[["date", "code", "open", "high", "low", "close", "volume", "amount"]])
    market_df = pd.concat(frames, ignore_index=True)
    notes = []
    if not args.no_external:
        external_df, notes = fetch_external_factor_frame(symbols, market_df=market_df)
        market_df = merge_external_factors(market_df, external_df)

    if args.holdings:
        holdings_df = pd.read_csv(Path(args.holdings), dtype={"code": str})
    else:
        latest = market_df.sort_values("date").groupby("code").tail(1)
        holdings_df = latest[["code", "close"]].rename(columns={"close": "average_cost"})
        holdings_df["shares"] = 0
        holdings_df["available_shares"] = 0

    result = run_holding_decision(
        market_df,
        holdings_df,
        account=AccountState(cash=args.cash, total_asset=args.total_asset),
        config=DecisionConfig(train_min_rows=120),
        source_notes=notes,
    )
    print(result.table.to_string(index=False))


if __name__ == "__main__":
    main()
