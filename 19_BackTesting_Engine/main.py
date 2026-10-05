from data.loader import CSVLoader
from broker.broker import Broker
from strategies.sma_strategy import SMAStrategy
from engine.backtest import BacktestEngine
from analytics.performance import PerformanceAnalyzer

def main():
    loader = CSVLoader(filepath="C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/Research_Paper/4_Time_Series/1m_SOL.csv", column_map={"Date": "datetime"})
    bars   = loader.load()

    broker   = Broker(initial_cash=100000, commission=0.0002, slippage=0.0001)
    strategy = SMAStrategy(broker=broker, fast_period=10, slow_period=30)
    engine   = BacktestEngine(bars=bars, strategy=strategy, broker=broker)
    results  = engine.run()

    analyzer = PerformanceAnalyzer(results, periods_per_year=365*24)
    analyzer.print_report()


if __name__ == "__main__":
    main()
