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


    def on_bar(self, bar):
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

        ########### Entry ###########
        if bullish_cross:
            if self.broker.position.is_flat:
                self.broker.buy(quantity=100, signal_time=bar.datetime, tag="SMA_ENTRY")
        if bearish_cross:
            if self.broker.position.is_long:
                self.broker.sell(quantity=self.broker.position.quantity, signal_time = bar.datetime, reduce_only=True, tag="SMA_EXIT")
        self.previous_fast = fast_sma
        self.previous_slow = slow_sma