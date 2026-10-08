import pandas as pd
import numpy as np
from analytics.trade_analyzer import TradeAnalyzer

class PerformanceAnalyzer:
    def __init__(self, broker, periods_per_year=365*24, risk_free_rate=0.0):
        self.broker           = broker
        self.periods_per_year = periods_per_year
        self.risk_free_rate   = risk_free_rate

    def analyze(self):
        equity_df = pd.DataFrame(self.broker.equity_curve)
        trades_df = TradeAnalyzer(self.broker).get_trades()

        if equity_df.empty:
            raise ValueError("No equity data available")
        
        equity  = equity_df["equity"].astype(float)
        returns = equity.pct_change().dropna()
        initial = (self.broker.initial_cash)
        final   = equity.iloc[-1]
        total_return = (final / initial - 1)

        running_max  = np.maximum(equity.cummax(), initial)
        drawdown     = (equity / running_max -1)
        max_drawdown = drawdown.min()

        ###### sharpe ratio and sortino ratio
        periodic_rf     = ((1 + self.risk_free_rate) ** (1 / self.periods_per_year) - 1)
        excess_returns  = returns - periodic_rf
        volatility      = (returns.std(ddof=1) if len(returns) > 1 else 0.0)
        sharpe          = ( excess_returns.mean()  / volatility * np.sqrt(self.periods_per_year)if volatility > 0 else np.nan)
        downside        = np.minimum(excess_returns, 0.0)
        downside_deviation = (np.sqrt(np.mean(downside ** 2)) if len(downside) else 0.0)
        sortino         = (excess_returns.mean() / downside_deviation * np.sqrt(self.periods_per_year) if downside_deviation > 0 else np.nan)

        ########## Trades
        total_trades = len(trades_df)
        if total_trades > 0:
            winners      = trades_df[trades_df["pnl"] > 0]
            losers       = trades_df[trades_df["pnl"] < 0]
            win_rate     = (len(winners) / total_trades)
            gross_profit = (winners["pnl"].sum())
            gross_loss   = abs(losers["pnl"].sum())
            profit_factor = (gross_profit / gross_loss if gross_loss > 0 else np.inf)
            average_trade = trades_df["pnl"].mean()
            average_win   = (winners["pnl"].mean() if len(winners) else 0)
            average_loss  = (losers["pnl"].mean() if len(losers) else 0)
            best_trade    = (trades_df["pnl"].max())
            worst_trade   = (trades_df["pnl"].min())

        else:
            win_rate = 0
            profit_factor = 0
            average_trade = 0
            average_win   = 0
            average_loss  = 0
            best_trade    = 0
            worst_trade   = 0

        return {
            "Initial Capital": initial,
            "Final Equity": final,
            "Net Profit": final - initial,
            "Total Return %": total_return * 100,
            "Max Drawdown %": max_drawdown * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Total Trades": total_trades,
            "Win Rate %": win_rate * 100,
            "Profit Factor": profit_factor,
            "Average Trade": average_trade,
            "Average Win": average_win,
            "Average Loss": average_loss,
            "Best Trade": best_trade,
            "Worst Trade": worst_trade
        }

    def print_report(self):
        results = (self.analyze())

        print("\n")
        print("=" * 50)
        print("          BACKTEST REPORT")
        print("=" * 50)

        for key, value in results.items():
            if isinstance(value, (float, np.floating)):
                print(f"{key:<25}: "f"{value:,.2f}")
            else:
                print(f"{key:<25}: "f"{value}")
        print("=" * 50)
    
