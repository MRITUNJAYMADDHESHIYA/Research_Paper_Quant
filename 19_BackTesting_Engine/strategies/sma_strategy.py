## 1.SMA strategy
## buy when fast sma crosses above the slow sma
## sell when fast sma crosses below the slow sma
## don't enter, when a position open
## trade one unit at a time

from collections import deque
from strategies.base import Strategy

class SMAStrategy(Strategy):
    def __init__(self, broker, fast=10, slow=30):
        super().__init__(broker)

        self.fast           = fast
        self.slow           = slow
        self.prices         = deque(maxlen=slow)
        self.previous_fast  = None
        self.previous_slow  = None
        self.pending_singal = False


    def on_bar(self, bar):
        self.prices.append(bar.close)
        if len(self.prices) < self.slow:
            return

        prices   = list(self.prices)
        fast_sma = (sum(prices[-self.fast:]) / self.fast)
        slow_sma = (sum(prices) / self.slow)

        if self.previous_fast is not None:
            self.previous_fast = fast_sma
            self.previous_slow = slow_sma
            return
        
        bullish_cross   = (self.previous_fast <= self.previous_slow and fast_sma > slow_sma)
        bearish_cross   = (self.previous_fast >= self.previous_slow and fast_sma < slow_sma)

        if bullish_cross and self.broker.position == 0:
            capital = (self.broker.cash * 0.95) ## invest 95% capital
            quantity = (capital / bar.close)
            self.broker.buy(quantity = quantity, signal_time = bar.datetime)

        elif bearish_cross and self.broker.position > 0:
            self.broker.sell(quantity = self.broker.position, signal_time=bar.datetime)

        self.previous_fast = fast_sma
        self.previous_slow = slow_sma