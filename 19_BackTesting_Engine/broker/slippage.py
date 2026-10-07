from abc import ABC, abstractmethod
from broker.enums import OrderSide

class SlippageModel(ABC):
    @abstractmethod
    def apply(self, price, side, quantity=None, bar=None):
        pass

class PercentageSlippage(SlippageModel):
    def __init__(self, rate=0.0001):
        if rate < 0:
            raise ValueError("Slippage cannot be negative")
        self.rate = rate

    def apply(self, price, side, quantity=None, bar=None):
        if side == OrderSide.BUY:
            return (price * (1+ self.rate))

        return (price * (1- self.rate))

    