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
    side:     OrderSide
    quantity: float
    status:   OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    commission: float = 0.0

