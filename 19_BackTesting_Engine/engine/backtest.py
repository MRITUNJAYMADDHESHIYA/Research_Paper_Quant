
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
            ######## execute previous candle orders using current candle open
            self.broker.process_bar(bar)
            equity = (self.broker.get_equity(bar.close))
            
            if self.risk_manager:    
                self.risk_manager.update(timestamp=bar.datetime, equity = equity)

            can_trade = (self.risk_manager is None or self.risk_manager.can_trade())
            if can_trade:
                self.broker.update_equity(bar)

        ###### end of dataset #########
        final_bar = self.bars[-1]
        if not self.broker.position.is_flat:
            self.broker.liquidate(final_bar, reason="END_OF_DATA")
            self.broker.process_bar(final_bar)
            self.broker.update_equity(final_bar)  

        print("Backtest Finished")
        print("-"*50)

        return self.broker
    