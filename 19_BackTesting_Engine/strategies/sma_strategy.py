## 1.SMA strategy
## buy when fast sma crosses above the slow sma
## sell when fast sma crosses below the slow sma
## don't enter, when a position open
## trade one unit at a time

from collections import deque
from strategies.base import BaseStrategy

class SMAStrategy(BaseStrategy):
    def __init__(self, broker, fast_period=10, slow_period=30):
        super().__init__(broker)

        if not 0< fast_period < slow_period:
            raise ValueError("0 < fast < slow")

        self.fast_period    = fast_period
        self.slow_period    = slow_period
        self.prices         = deque(maxlen=slow_period)
        self.previous_fast  = None
        self.previous_slow  = None
        self.pending_singal = False


    def on_bar(self, bar):
        self.prices.append(bar.close)
        if len(self.prices) < self.slow_period:
            return

        prices = list(self.prices)
        fast_sma = (sum(prices[-self.fast_period:]) / self.fast_period)
        slow_sma = (sum(prices) / self.slow_period)

        if self.previous_fast is not None:
            cross_up   = (self.previous_fast <= self.previous_slow and fast_sma > slow_sma)
            cross_down = (self.previous_fast >= self.previous_slow and fast_sma < slow_sma)

            if not self.broker.pending_orders:
                if cross_up and self.broker.position == 0:
                    self.broker.buy(quantity = 1)
                    
                elif cross_down and self.broker.position > 0:
                    self.broker.sell(quantity = self.broker.position)

        self.previous_fast = fast_sma
        self.previous_slow = slow_sma