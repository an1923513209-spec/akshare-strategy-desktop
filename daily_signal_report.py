"""Print a daily trading signal report from the web cockpit logic."""

from __future__ import annotations

import argparse
from html import unescape
from pathlib import Path
import re

from app import analyze, default_form


TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return unescape(TAG_RE.sub("", text)).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="002472")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--cash", default="5000")
    parser.add_argument("--fee", default="0.0003")
    parser.add_argument("--risk", default="normal", choices=["tight", "normal", "loose"])
    parser.add_argument("--horizon", default="short", choices=["short", "swing", "trend"])
    parser.add_argument("--strategy-type", default="auto", choices=["auto", "breakout", "rsi", "macd", "ml", "sma"])
    parser.add_argument("--shares", default="0")
    parser.add_argument("--buy-price", default="")
    parser.add_argument("--buy-date", default="")
    args = parser.parse_args()

    form = default_form()
    form.update(
        {
            "symbol": args.symbol,
            "start": args.start,
            "cash": args.cash,
            "fee": args.fee,
            "risk": args.risk,
            "horizon": args.horizon,
            "strategy_type": args.strategy_type,
            "shares": args.shares,
            "buy_price": args.buy_price,
            "buy_date": args.buy_date,
        }
    )
    result = analyze(form)

    print(f"Daily Signal Report: {args.symbol}")
    print("\n[Action]")
    for line in result["action_lines"]:
        print("- " + strip_html(str(line)))

    print("\n[Signal]")
    for line in result["signal_lines"]:
        print("- " + strip_html(str(line)))

    print("\n[Intraday Reminders]")
    for line in result["reminder_lines"]:
        print("- " + strip_html(str(line)))


if __name__ == "__main__":
    main()
