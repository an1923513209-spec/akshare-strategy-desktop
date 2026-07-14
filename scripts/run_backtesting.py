"""Run a simple moving-average crossover strategy with backtesting.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from data_utils import PROJECT_ROOT, load_a_share_daily, normalize_symbol


def sma(values, window: int):
    return pd.Series(values).rolling(window).mean()


class SmaCross(Strategy):
    fast = 10
    slow = 30

    def init(self) -> None:
        self.fast_ma = self.I(sma, self.data.Close, self.fast)
        self.slow_ma = self.I(sma, self.data.Close, self.slow)

    def next(self) -> None:
        if crossover(self.fast_ma, self.slow_ma):
            self.position.close()
            self.buy()
        elif crossover(self.slow_ma, self.fast_ma):
            self.position.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="002472", help="A-share code, for example 002472")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default=None)
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--fast", type=int, default=10)
    parser.add_argument("--slow", type=int, default=30)
    args = parser.parse_args()

    if args.fast >= args.slow:
        raise ValueError("--fast must be smaller than --slow")

    data = load_a_share_daily(args.symbol, args.start, args.end, args.adjust)
    bt = Backtest(
        data,
        SmaCross,
        cash=args.cash,
        commission=args.commission,
        trade_on_close=True,
        finalize_trades=True,
    )
    stats = bt.run(fast=args.fast, slow=args.slow)

    code = normalize_symbol(args.symbol)
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"{code}_backtesting_sma_{args.fast}_{args.slow}.csv"
    stats.to_csv(out_path, encoding="utf-8-sig")

    keys = [
        "Start",
        "End",
        "Duration",
        "Return [%]",
        "Buy & Hold Return [%]",
        "Max. Drawdown [%]",
        "# Trades",
        "Win Rate [%]",
        "Best Trade [%]",
        "Worst Trade [%]",
        "Sharpe Ratio",
    ]
    print(f"Symbol: {code}")
    print(f"Rows: {len(data)}")
    for key in keys:
        if key in stats:
            print(f"{key}: {stats[key]}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
