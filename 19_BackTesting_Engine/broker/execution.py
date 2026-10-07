#### This tell you about (can it execute + at what price)

from dataclasses import dataclass
from broker.enums import (OrderSide, OrderType)

@dataclass
class ExecutionDecision:
    should_fill: bool
    price:       float = None
    triggered:   bool  = False  ### spefic for (stop orders):- True(stop condition occurred)

class ExecutionEngine:
    def __init__(self, max_volume_participation=0.10):  #10% of candle volume of the market
        if not (0 < max_volume_participation <= 1):
            raise ValueError("volume between 0 to 1")
        
        self.max_volume_participation = (max_volume_participation)

    ########## Max fill quantity #############
    def available_quantity(self, bar):
        return (bar.volume * self.max_volume_participation)

    ######### Execution Decision #############
    def evaluate(self, order, bar):
        if (order.order_type == OrderType.MARKET):
            return ExecutionDecision(True, bar.open) ### not need any other function, execute at market price

        if(order.order_type == OrderType.LIMIT):
            return self._limit(order, bar)

        if(order.order_type == OrderType.STOP):
            return self._stop(order, bar)

        if(order.order_type == OrderType.STOP_LIMT):
            return self._stop_limit(order, bar)

        return ExecutionDecision(False)

    ######### Limit ###################
    def _limit(self, order, bar):
        limit = order.limit_price
        if limit is None:
            return ExecutionDecision(False)

        #### Buy limit
        if order.side == OrderSide.BUY:
            ### Gap/open better than limit
            if bar.open <= limit:
                return ExecutionDecision(True, bar.open)
            ### Intrabar touch
            if bar.low <= limit:
                return ExecutionDecision(True, limit)
        else:
            if bar.open >= limit:
                return ExecutionDecision(True, bar.open)
            if bar.high >= limit:
                return ExecutionDecision(True, limit)

        return ExecutionDecision(False)
    

    ############ Stop #################
    def _stop(self, order, bar):
        stop = order.stop_price

        if stop is None:
            return ExecutionDecision(False)
        ### Buy stop
        if order.side == OrderSide.Buy:
            ### gap above stop
            if bar.open >= stop:
                return ExecutionDecision(True, bar.open, True)
            if bar.high >= stop:
                return ExecutionDecision(True, stop, True)
        else:
            ### gap below stop
            if bar.open <= stop:
                return ExecutionDecision(True, bar.open, True)
            if bar.low <= stop:
                return ExecutionDecision(True, stop, True)

        return ExecutionDecision(False)


    ########### stop limit ############
    def _stop_limit(self, order, bar):
        stop  = order.stop_price
        limit = order.limit_price

        if(stop is None or limit is None):
            return ExecutionDecision(False)
        
        ##### Trigger first
        if not order.triggered:
            if order.side == OrderSide.BUY:
                if(bar.open >= stop or bar.high >= stop):
                    order.triggered = True

            else:
                if(bar.open <= stop or bar.low <= stop):
                    order.triggered = True

        if not order.triggered:
            return ExecutionDecision(False)

        ### Once triggered it bahaves like a limit order
        return self._limit(order, bar)

    