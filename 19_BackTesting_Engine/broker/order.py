from dataclasses import dataclass
from enum import Enum

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING  = "PENDING"
    FILLED   = "FILLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    side:        OrderSide
    quantity:    float

    status:      OrderStatus = OrderStatus.PENDING
    signal_time: object = None
    fill_time:   object = None

    fill_price:  float = 0.0
    commission:  float = 0.0

    stop_price:  float = None
    exit_reason: str   = None

