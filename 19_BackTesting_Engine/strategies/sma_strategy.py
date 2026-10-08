## 1.SMA strategy
## buy when fast sma crosses above the slow sma
## sell when fast sma crosses below the slow sma
## don't enter, when a position open
## trade one unit at a time

from collections import deque
from strategies.base import Strategy

class SMAStrategy(Strategy):
    def __init__(self, broker, risk_manager, fast=10, slow=30):
        super().__init__(broker)

        self.fast           = fast
        self.slow           = slow
        self.prices         = deque(maxlen=slow)
        self.previous_fast  = None
        self.previous_slow  = None
        
        self.risk_manager   = (risk_manager)


    def on_bar(self, bar, allow_entry=True):
        self.prices.append(bar.close)
        if len(self.prices) < self.slow:
            return

        prices   = list(self.prices)
        fast_sma = (sum(prices[-self.fast:]) / self.fast)
        slow_sma = (sum(prices) / self.slow)

        if self.previous_fast is None:
            self.previous_fast = fast_sma
            self.previous_slow = slow_sma
            return
        
        bullish_cross   = (self.previous_fast <= self.previous_slow and fast_sma > slow_sma)
        bearish_cross   = (self.previous_fast >= self.previous_slow and fast_sma < slow_sma)

        has_pending_orders = any(order.is_active for order in self.broker.pending_orders)

        ########### Entry ###########
        if(bullish_cross and allow_entry and self.broker.position.is_flat and not has_pending_orders):
            equity   = self.broker.get_equity(bar.close)
            quantity = (self.risk_manager.calculate_position_size(equity=equity, cash=self.broker.cash, entry_price=bar.close))

            ##### reserve some cash for fees and price movement
            max_affordable = (self.broker.cash * 0.99 / (bar.close *(1 + self.broker.commission_model.rate) * (1 + self.broker.slippage_model.rate)))
            quantity       = min(quantity, max_affordable)

            if quantity > 0:
                self.broker.buy(quantity=quantity, signal_time=bar.datetime, tag = "SMA_ENTRY")
        elif(bearish_cross and self.broker.position.is_long and not has_pending_orders):
            self.broker.sell(quantity=self.broker.position.quantity, signal_time=bar.datetime, reduce_only=True, tag = "SMA_EXIT")
            

        self.previous_fast = fast_sma
        self.previous_slow = slow_sma