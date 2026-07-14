"""Scan many SMA parameter pairs with vectorbt."""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd
import vectorbt as vbt

from data_utils import PROJECT_ROOT, load_a_share_daily, normalize_symbol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="002472")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default=None)
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--fee", type=float, default=0.0003)
    parser.add_argument("--fast", default="5,10,15,20,25")
    parser.add_argument("--slow", default="30,40,50,60,90,120")
    args = parser.parse_args()

    data = load_a_share_daily(args.symbol, args.start, args.end, args.adjust)
    close = data["Close"]
    fast_windows = [int(x) for x in args.fast.split(",") if x.strip()]
    slow_windows = [int(x) for x in args.slow.split(",") if x.strip()]

    rows = []
    for fast, slow in product(fast_windows, slow_windows):
        if fast >= slow:
            continue
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            init_cash=args.cash,
            fees=args.fee,
            freq="1D",
        )
        rows.append(
            {
                "fast": fast,
                "slow": slow,
                "total_return_pct": float(portfolio.total_return()) * 100,
                "max_drawdown_pct": float(portfolio.max_drawdown()) * 100,
                "sharpe": float(portfolio.sharpe_ratio()) if np.isfinite(portfolio.sharpe_ratio()) else np.nan,
                "trades": int(portfolio.trades.count()),
                "final_value": float(portfolio.final_value()),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["sharpe", "total_return_pct", "max_drawdown_pct"],
        ascending=[False, False, True],
    )

    code = normalize_symbol(args.symbol)
    out_path = PROJECT_ROOT / "reports" / f"{code}_vectorbt_sma_scan.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(result.head(10).to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
