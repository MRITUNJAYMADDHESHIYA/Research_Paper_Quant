import pandas as pd
import numpy as np

class PerformanceAnalyzer:
    def __init__(self, broker, periods_per_year=252):
        self.broker = broker
        self.periods_per_year = periods_per_year

    def claculate(self):
        df = pd.DataFrame(self.broker.equity_curve)
        df = df.set_index("timestamp")

        equity = df["equity"]
        returns = equity.pct_change().dropna()

        total_return = (equity.iloc[-1] / self.broker.initial_cash - 1)
        running_max  = equity.cummax()
        drawdown     = (equity / running_max -1)
        max_drawdown = drawdown.min()

        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(self.periods_per_year)
        else:
            sharpe = 0.0

        return {
            "initial_cash": self.broker.initial_cash,
            "final_equity": equity.iloc[-1],
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "filled_orders": len(self.broker.filled_orders)
        }

    
