"""Build an interactive Plotly HTML dashboard for one A-share strategy scan."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbt as vbt
from plotly.subplots import make_subplots

from data_utils import PROJECT_ROOT, load_a_share_daily, normalize_symbol


def make_signals(close: pd.Series, fast: int, slow: int) -> tuple[pd.Series, pd.Series]:
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    return entries, exits


def scan_or_load(symbol: str, close: pd.Series, cash: float, fee: float) -> pd.DataFrame:
    code = normalize_symbol(symbol)
    scan_path = PROJECT_ROOT / "reports" / f"{code}_vectorbt_sma_scan.csv"
    if scan_path.exists():
        return pd.read_csv(scan_path)

    rows = []
    for fast, slow in product([5, 10, 15, 20, 25], [30, 40, 50, 60, 90, 120]):
        if fast >= slow:
            continue
        entries, exits = make_signals(close, fast, slow)
        portfolio = vbt.Portfolio.from_signals(close, entries, exits, init_cash=cash, fees=fee, freq="1D")
        sharpe = portfolio.sharpe_ratio()
        rows.append(
            {
                "fast": fast,
                "slow": slow,
                "total_return_pct": float(portfolio.total_return()) * 100,
                "max_drawdown_pct": float(portfolio.max_drawdown()) * 100,
                "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
                "trades": int(portfolio.trades.count()),
                "final_value": float(portfolio.final_value()),
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe", "total_return_pct"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="002472")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default=None)
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--fee", type=float, default=0.0003)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    code = normalize_symbol(args.symbol)
    data = load_a_share_daily(code, args.start, args.end, args.adjust)
    close = data["Close"]
    scan = scan_or_load(code, close, args.cash, args.fee).dropna(subset=["fast", "slow"]).copy()
    scan = scan.sort_values(["sharpe", "total_return_pct"], ascending=[False, False]).head(args.top)
    if scan.empty:
        raise RuntimeError("No strategy scan rows are available to visualize.")

    best = scan.iloc[0]
    best_fast = int(best["fast"])
    best_slow = int(best["slow"])
    best_entries, best_exits = make_signals(close, best_fast, best_slow)
    best_portfolio = vbt.Portfolio.from_signals(
        close,
        best_entries,
        best_exits,
        init_cash=args.cash,
        fees=args.fee,
        freq="1D",
    )
    equity = best_portfolio.value()
    drawdown = (equity / equity.cummax() - 1) * 100

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.44, 0.14, 0.27, 0.15],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=(
            f"{code} Daily Price with SMA {best_fast}/{best_slow}",
            "Volume",
            "Equity and Drawdown",
            "Top SMA Parameter Sets",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=data.index, y=close.rolling(best_fast).mean(), name=f"SMA {best_fast}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=close.rolling(best_slow).mean(), name=f"SMA {best_slow}"), row=1, col=1)

    buy_points = close[best_entries.fillna(False)]
    sell_points = close[best_exits.fillna(False)]
    fig.add_trace(
        go.Scatter(
            x=buy_points.index,
            y=buy_points,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#16a34a"),
            name="Entry",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points.index,
            y=sell_points,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#dc2626"),
            name="Exit",
        ),
        row=1,
        col=1,
    )

    volume_colors = np.where(data["Close"] >= data["Open"], "#ef4444", "#22c55e")
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], marker_color=volume_colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity", line=dict(color="#2563eb", width=2)), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown %", line=dict(color="#f97316", width=1.6)),
        row=3,
        col=1,
    )

    table = scan.copy()
    for column in ["total_return_pct", "max_drawdown_pct", "sharpe", "final_value"]:
        table[column] = table[column].map(lambda value: f"{value:.2f}")
    fig.add_trace(
        go.Table(
            header=dict(
                values=["fast", "slow", "return %", "max DD %", "sharpe", "trades", "final value"],
                fill_color="#334155",
                font=dict(color="white"),
                align="center",
            ),
            cells=dict(
                values=[
                    table["fast"],
                    table["slow"],
                    table["total_return_pct"],
                    table["max_drawdown_pct"],
                    table["sharpe"],
                    table["trades"],
                    table["final_value"],
                ],
                fill_color="#f8fafc",
                align="center",
            ),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title=f"{code} Strategy Dashboard | Best SMA {best_fast}/{best_slow}",
        template="plotly_white",
        height=1100,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=90, b=40),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Value / DD%", row=3, col=1)

    out_path = PROJECT_ROOT / "reports" / f"{code}_strategy_dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {out_path}")
    print(f"Best SMA: fast={best_fast}, slow={best_slow}")


if __name__ == "__main__":
    main()
