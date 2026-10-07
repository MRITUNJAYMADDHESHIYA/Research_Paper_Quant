#### One order can generate multiple fills
from datetime import dataclass
from itertools import count
from broker.enums import OrderSide

_fill_counter = count(1)

@dataclass
class Fill:
    order_id:   int
    side:       OrderSide
    quantity:   float
    price:      float
    commission: float
    timestamp:  object
    fill_id:    int =None


    def __post_init__(self):    ### automatically calculate 
        if self.fill_id is None:
            self.fill_id = next(_fill_counter)

    @property
    def notional(self):
        return (self.quantity * self.price)