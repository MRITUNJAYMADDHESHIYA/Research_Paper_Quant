from data.loader import CSVLoader
from broker.broker import Broker
from strategies.sma_strategy import SMAStrategy
from risk.risk_manager import RiskManager
from engine.backtest import BacktestEngine
from analytics.performance import PerformanceAnalyzer

def main():
    INITIAL_CAPITAL = 10000
    loader = CSVLoader(filepath="C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/4_Time_Series/1m_SOL.csv")
    bars   = loader.load()

    broker       = Broker(initial_cash=INITIAL_CAPITAL, commission=0.0002, slippage=0.0001)
    risk_manager = RiskManager(initial_capital=INITIAL_CAPITAL, risk_per_trade=0.01, stop_loss_pct=0.02, daily_drawdown_limit=0.10,max_position_pct=0.95)
    strategy     = SMAStrategy(broker=broker, risk_manager=risk_manager, fast=10, slow=30)
    engine       = BacktestEngine(bars=bars, strategy=strategy, broker=broker, risk_manager=risk_manager)
    results      = engine.run()

    analyzer = PerformanceAnalyzer(results, periods_per_year=365*24)
    analyzer.print_report()


if __name__ == "__main__":
    main()


### Broker:-       executes orders
### risk_manager:- position, sl, daily DD, 
### strategy:-     generates decision
### engine:-       controls time
### analyzer:-     results