from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, broker):
        self.broker = broker

    @abstractmethod
    def on_bar(self, bar):
        pass