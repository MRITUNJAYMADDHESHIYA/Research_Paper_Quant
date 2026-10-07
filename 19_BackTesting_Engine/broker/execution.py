from dataclasses import dataclass
from broker.enums import (OrderSide, OrderType)

@dataclass
class ExecutionDecision:
    should_fill: bool
    price:       float = None
    triggered:   bool = False

class ExecutionEngine:
    def __init__(self, max_volume_participation=0.10):  #10% of total volume of the market
        if not (0 < max_volume_participation <= 1):
            raise ValueError("volume between 0 to 1")
        
        self.max_volume_participation = (max_volume_participation)

    ########## Max fill quantity #############
    def available_quantity(self, bar):
        return (bar.volume * self.max_volume_participation)

    ######### Execution Decision #############
    def evaluate(self, order, bar):
        if (order.order_type == OrderType.MARKET):
            return ExecutionDecision(True, bar.open)

        if(order.order_type == OrderType.LIMIT):
            return self._limit(order, bar)

        if(order.order_type == OrderType.STOP):
            return self._stop(order, bar)

        if(order.order_type == OrderType.STOP_LIMT):
            return self._stop_limit(order, bar)

        return ExecutionDecision(False)

    ######### Limit ###################
    def _limit(self, order, bar):
        limit = 

    ############ Stop #################
    def _stop(self, order, bar):
        pass

    ########### stop limit ############
    def _stop_limit(self, order, bar):
        pass