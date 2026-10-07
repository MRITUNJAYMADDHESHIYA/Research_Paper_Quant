
class BacktestEngine:
    def __init__(self, bars, strategy, broker, risk_manager):
        self.bars      = bars
        self.strategy  = strategy
        self.broker    = broker
        self.risk_manager = (risk_manager)

    def run(self):
        print("\n Starting Backtest....")
        print("-"*50)

        for bar in self.bars:

            ## 1. execute previous candle orders using current candle open
            self.broker.execute_orders(bar)

            ### 1.1 check intrabar stop loss
            self.broker.check_stop_loss(bar)

            equity = (self.broker.get_equity(bar.close))
            risk   = (self.risk_manager.update(timestamp=bar.datetime, equity = equity))

            ### portfolio kill switch
            if(self.risk_manager.kill_switch and self.broker.position > 0):
                self.broker.sell(quantity = self.broker.position, signal_time = bar.datetime)
            elif self.risk_manager.can_trade():
                self.strategy.on_bar(bar)
                self.broker.update_equity(bar)


        print("Backtest Finished")
        print("-"*50)

        return self.broker
    