# backtest/engine.py

import numpy as np
import pandas as pd


class BacktestEngine:

    def __init__(
        self,
        initial_capital=100000,
        transaction_cost=0.0005
    ):

        self.initial_capital = initial_capital

        self.transaction_cost = (
            transaction_cost
        )

    def run(
        self,
        actual_returns,
        signals
    ):

        actual_returns = np.asarray(
            actual_returns
        )

        signals = np.asarray(
            signals
        )

        # Position
        position = signals

        # Strategy return
        strategy_returns = (
            position *
            actual_returns
        )

        # Transaction cost
        position_change = np.abs(
            np.diff(
                np.insert(
                    position,
                    0,
                    0
                )
            )
        )

        costs = (
            position_change *
            self.transaction_cost
        )

        net_returns = (
            strategy_returns -
            costs
        )

        equity = (
            self.initial_capital *
            np.cumprod(
                1 + net_returns
            )
        )

        result = pd.DataFrame({

            "Actual_Return":
                actual_returns,

            "Signal":
                signals,

            "Strategy_Return":
                net_returns,

            "Equity":
                equity

        })

        return result

    def metrics(
        self,
        result
    ):

        returns = result[
            "Strategy_Return"
        ]

        equity = result[
            "Equity"
        ]

        # Total return
        total_return = (
            equity.iloc[-1] /
            self.initial_capital
            - 1
        )

        # Sharpe
        if returns.std() != 0:

            sharpe = (
                returns.mean() /
                returns.std()
            ) * np.sqrt(252)

        else:

            sharpe = 0

        # Drawdown
        running_max = equity.cummax()

        drawdown = (
            equity /
            running_max
            - 1
        )

        max_drawdown = (
            drawdown.min()
        )

        # Number of trades
        trades = (
            np.abs(
                np.diff(
                    result["Signal"]
                )
            ) > 0
        ).sum()

        return {

            "Total Return":
                total_return,

            "Sharpe Ratio":
                sharpe,

            "Max Drawdown":
                max_drawdown,

            "Trades":
                trades

        }