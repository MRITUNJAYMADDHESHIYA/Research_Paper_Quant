from abc import ABC, abstractmethod

class BaseStrategy(ABC):

    @abstractmethod
    def generate_signal(self, prediction, current_price):
        pass

    