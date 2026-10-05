### Manages capital, positions, pending orders, transaction consts and the equity

from broker.order import Order, OrderStatus, OrderSide

class Broker:
    def __init__(self, initial_cash = 100000, commission=0.0002, slippage=0.0001):
        self.initial_cash = initial_cash
        self.cash         = float(initial_cash)
        self.position     = 0.0
        self.average_price = 0.0

        self.commission = commission
        self.slippage   = slippage

        self.pending_orders = []
        self.filled_orderes = []
        self.equity_curve   = []
        self.realized_pnl   = 0.0

    def buy(self, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        self.pending_orders.append(Order(OrderSide.BUY, quantity))

    def sell(self, quantity):
        if quantity <=0:
            raise ValueError("Quantity must be positive")

        self.pending_orders.append(Order(OrderSide.SELL, quantity))

    def execute_orders(self, bar):
        for order in self.pending_orders:
            if order.side == OrderSide.BUY:

                price = bar.open * (1 + self.slippage)
                cost  = price * order.quantity
                fee   = cost * self.commission

                if cost + fee > self.cash:
                    order.status = OrderStatus.REJECTED
                    continue

                old_cost           = self.position * self.average_price
                self.cash         -= cost + fee
                self.position     += order.quantity
                self.average_price = (old_cost + cost) / self.position

            else:
                if order.quantity > self.position:
                    order.status = OrderStatus.REJECTED
                    continue
                price    = bar.open * (1 - self.slippage)
                proceeds = price * order.quantity
                fee      = proceeds * self.commission

                self.cash         += proceeds - fee
                self.realized_pnl += ((price - self.average_price) * order.quantity - fee)
                self.position     -= order.quantity

                if self.position == 0:
                    self.average_price = 0.0

            order.fill_price = price
            order.commission = fee
            order.status = OrderStatus.FILLED

            self.filled_orderes.append({
                "timestamp": bar.timestamp,
                "side":      order.side.value,
                "quantity":  order.quantity,
                "price":     price,
                "commission":fee
            })

        self.pending_orders.clear()


    def update_equity(self, bar):
        equity = (self.cash + self.position*bar.close)
        self.equity_curve.append({
            "timestamp": bar.timestamp,
            "equity": equity
        })

    def get_equity(self, current_price):
        return (self.cash + self.position * current_price)

        