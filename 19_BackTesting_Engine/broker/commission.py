from abc import ABC, abstractmethod

class CommissionModel(ABC):

    @abstractmethod
    def calculate(self, quantity, price):
        pass

class PercentageCommission(CommissionModel):
    def __init__(self, rate=0.0002):
        if rate < 0:
            raise ValueError("Commission cannot be negative")
        self.rate = rate

    def calculate(self, quantity, price):
        notional = (abs(quantity) * price)
        return (notional * self.rate)


######## later I can add commission