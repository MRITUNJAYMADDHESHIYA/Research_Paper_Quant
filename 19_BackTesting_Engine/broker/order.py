from dataclasses import dataclass, field
from typing import Optional
from itertools import count
from broker.enums import OrderSide, OrderStatus, OrderType, TimeInForce, ExitReason

_order_counter = count(1)

@dataclass
class Order:
    side:        OrderSide
    quantity:    float

    order_type:  OrderType = OrderType.MARKET 
    limit_price: Optional[float] = None
    stop_price:  Optional[float] = None
    signal_time: object = None

    time_in_frame: TimeInForce = TimeInForce.GTC  
    reduce_only: bool = False
    tag:         Optional[str] = None
    id:          int = field(default_factory=lambda: next(_order_counter))
    status:      OrderStatus = OrderStatus.NEW

    filled_quantity:    float = 0.0
    average_fill_price: float = 0.0
    commission:         float = 0.0
    rejection_reason:   Optional[str] = None
    triggered:          bool = False

    @property
    def remaining_quantity(self):
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def is_active(self):
        return self.status in {OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}