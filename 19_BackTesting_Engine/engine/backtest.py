

class BacktestEngine:
    def __init__(self, bars, strategy, broker):
        self.bars      = bars
        self.strategy  = strategy
        self.broker    = broker

    def run(self):
        print("\n Starting Backtest....")
        print("-"*50)

        for bar in self.bars:

            ## 1. execute previous candle orders using current candle open
            self.broker.execute_orders(bar)

            ## 2. generate signals using current bar
            self.strategy.on_bar(bar)

            ## 3. mark portfolio at close
            self.broker.update_equity(bar)


        print("Backtest Finished")
        print("-"*50)

        return self.broker
    