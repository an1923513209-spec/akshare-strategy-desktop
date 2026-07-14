"""Use Optuna to search SMA strategy parameters with backtesting.py."""

from __future__ import annotations

import argparse

import optuna
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
    parser.add_argument("--symbol", default="002472")
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default=None)
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    data = load_a_share_daily(args.symbol, args.start, args.end, args.adjust)
    bt = Backtest(
        data,
        SmaCross,
        cash=args.cash,
        commission=args.commission,
        trade_on_close=True,
        finalize_trades=True,
    )

    def objective(trial: optuna.Trial) -> float:
        fast = trial.suggest_int("fast", 5, 40)
        slow = trial.suggest_int("slow", fast + 5, 160)
        stats = bt.run(fast=fast, slow=slow)
        ret = float(stats["Return [%]"])
        drawdown = abs(float(stats["Max. Drawdown [%]"]))
        trades = float(stats["# Trades"])
        if trades < 3:
            return -1e6
        return ret - drawdown * 0.5

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    code = normalize_symbol(args.symbol)
    out_path = PROJECT_ROOT / "reports" / f"{code}_optuna_sma_trials.csv"
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "value": trial.value,
                "fast": trial.params.get("fast"),
                "slow": trial.params.get("slow"),
                "state": trial.state.name,
            }
        )
    pd.DataFrame(rows).sort_values("value", ascending=False).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
