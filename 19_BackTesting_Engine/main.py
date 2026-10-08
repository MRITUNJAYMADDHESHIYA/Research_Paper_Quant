from data.loader import CSVLoader
from broker.broker import Broker
from strategies.sma_strategy import SMAStrategy
from strategies.donchian_strategy import DonchianStrategy

from risk.risk_manager import RiskManager
from engine.backtest import BacktestEngine
from analytics.performance import PerformanceAnalyzer
from analytics.trade_analyzer import TradeAnalyzer

def main():
    INITIAL_CAPITAL = 100000
    loader = CSVLoader(filepath="C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/4_Time_Series/1m_SOL.csv")
    bars   = loader.load()

    broker       = Broker(initial_cash=INITIAL_CAPITAL, commission_rate=0.0002, slippage_rate=0.0001, max_volume_participation=0.10, allow_short=True)
    risk_manager = RiskManager(initial_capital=INITIAL_CAPITAL, risk_per_trade=0.01, stop_loss_pct=0.02, daily_drawdown_limit=0.10,max_position_pct=0.95)
    #strategy     = SMAStrategy(broker=broker, risk_manager=risk_manager, fast=10, slow=30)
    strategy     = DonchianStrategy(broker=broker, risk_manager=risk_manager, entry_period=20, exit_period=10, atr_period=14, atr_multiplier=2.0)
    engine       = BacktestEngine(bars=bars, strategy=strategy, broker=broker, risk_manager=risk_manager)
    results      = engine.run()

    
    analyzer = PerformanceAnalyzer(broker=results, periods_per_year=365*24*60)
    analyzer.print_report()
    trades   = TradeAnalyzer(results).get_trades()
    print(trades)
    trades.to_csv("trades.csv", index=False)


if __name__ == "__main__":
    main()


### Broker:-       executes orders
### risk_manager:- position, sl, daily DD, 
### strategy:-     generates decision
### engine:-       controls time
### analyzer:-     results