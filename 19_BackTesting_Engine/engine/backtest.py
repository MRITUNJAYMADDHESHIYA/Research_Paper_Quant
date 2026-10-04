

class BacktestEngine:
    def __init__(self, bars, strategy, broker):
        self.bars      = bars
        self.strategy  = strategy
        self.broker    = broker

    def run(self):
        if not self.bars:
            raise ValueError("No market data provided")

        for bar in self.bars:

            ## 1. fill orders from previous bars
            self.broker.execute_orders(bar)

            ## 2. generate signals using current bar
            self.strategy.on_bar(bar)

            ## 3. Record current portfolio equity
            self.broker.update_equity(bar)


        ## Order submitted on the final candle
        ## cannot executes without another candle
        self.broker.pending_orders.clear()

        return self.broker