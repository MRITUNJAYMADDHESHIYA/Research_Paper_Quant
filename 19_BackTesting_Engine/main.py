from data.loader import CSVLoader
from broker.broker import Broker
from strategies.sma_strategy import SMAStrategy
from engine.backtest import BacktestEngine
from analytics.performance import PerformanceAnalyzer

def main():
    loader = CSVLoader(filepath="datasets/sample.csv", column_map={"Date": "datetime"})
    bars   = loader.load()

    broker   = Broker(initial_cash=10000, commission=0.002, slippage=0.001)
    strategy = SMAStrategy(broker=broker, fast_period=10, slow_period=30)
    engine   = BacktestEngine(bars=bars, strategy=strategy, broker=broker)

    results  = engine.run()
    analyzer = PerformanceAnalyzer(results, periods_per_year=252)
    metrics  = analyzer.claculate()

    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
    