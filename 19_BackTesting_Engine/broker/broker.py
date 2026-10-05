### Manages capital, positions, pending orders, transaction consts and the equity

from broker.order import Order, OrderStatus, OrderSide

class Broker:
    def __init__(self, initial_cash = 100000, commission=0.0002, slippage=0.0001):
        self.initial_cash = float(initial_cash)
        self.cash         = float(initial_cash)

        self.position     = 0.0
        self.average_price = 0.0

        self.commission_rate = commission
        self.slippage   = slippage

        self.pending_orders = []

        self.order_history  = []
        self.trade_history  = []

        self.equity_curve   = []

        self.entry_price    = None
        self.entry_time     = None
        self.stop_price     = None
        self.exit_reason    = None
        self.stop_loss_pct  = None

        self.realized_pnl   = 0.0

    def buy(self, quantity, signal_time=None, stop_loss_pct=None,):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        order = Order(side=OrderSide.BUY, quantity=quantity, signal_time=signal_time)
        order.stop_price = stop_loss_pct

        self.pending_orders.append(order)

    def sell(self, quantity, signal_time=None):
        if quantity <=0:
            raise ValueError("Quantity must be positive")

        order = Order(side=OrderSide.SELL, quantity=quantity, signal_time=signal_time)
        self.pending_orders.append(order)

    def execute_orders(self, bar):
        orders = self.pending_orders.copy()
        self.pending_orders.clear()

        for order in orders:
            if order.side == OrderSide.BUY:
                self._execute_buy(order, bar)
            elif order.side == OrderSide.SELL:
                self._execute_sell(order, bar)

    def _execute_buy(self, order, bar):
        price = (bar.open * (1 + self.slippage))
        value = (price * order.quantity)
        commission = (value * self.commission_rate)
        total_cost = (value + commission)

        if total_cost > self.cash:
            order.status = (OrderStatus.REJECTED)
            self.order_history.append(order)
            return

        old_value          = (self.position * self.average_price)
        self.cash         -= total_cost
        self.position     += order.quantity
        self.average_price = (old_value + value) / self.position
        order.status       = OrderStatus.FILLED

        order.fill_price = price
        order.fill_time  = bar.datetime
        order.commission = commission

        self.order_history.append(order)

        ##### New position
        if self.entry_price is None:
            self.entry_price = price
            self.entry_time  = bar.datetime

            ### calculate stop from actual fill price
            if order.stop_price is not None:
                self.stop_price = (price *(1 - order.stop_price))

    def _execute_sell(self, order, bar):
        if order.quantity > self.position:
            order.status = (OrderStatus.REJECTED)
            self.order_history.append(order)
            return

        price = (bar.open * (1 - self.slippage))
        value = (price * order.quantity)
        commission = (value * self.commission_rate)
        self.cash += (value - commission)
        pnl = (price - self.average_price) * order.quantity
        pnl -= commission
        self.position -= order.quantity

        order.status     = OrderStatus.FILLED
        order.fill_price = price
        order.fill_time  = bar.datetime
        order.commission = commission

        self.order_history.append(order)

        #### For version 1 we expect full exits.
        if self.position == 0:
            trade_return = (price / self.entry_price) - 1
            self.trade_history.append({
                "entry_time": self.entry_time,
                "exit_time": bar.datetime,
                "entry_price": self.entry_price,
                "exit_price": price,
                "quantity": order.quantity,
                "pnl": pnl,
                "return": trade_return
            })

            self.average_price = 0
            self.entry_price   = None
            self.entry_time    = None

    
    def update_equity(self, bar):
        market_value = (self.position * bar.close)
        equity = (self.cash + market_value)
        self.equity_curve.append({
            "datetime": bar.datetime,
            "cash": self.cash,
            "position": self.position,
            "close": bar.close,
            "market_value": market_value,
            "equity": equity
        })

    ####### current equity ###########
    def get_equity(self, price):
        return (self.cash + self.position * price)

    def _close_position(self, price, timestamp, reason):
            if self.position <= 0:
                return

            quantity   = self.position
            value      = (quantity * price)
            commission = (value *self.commission_rate)
            proceeds   = (value -commission)
            self.cash += proceeds

            gross_pnl    = (price - self.entry_price) * quantity
            net_pnl      = (gross_pnl - commission)
            trade_return = (price / self.entry_price - 1)

            self.trade_history.append({
                "entry_time":self.entry_time,
                "exit_time":timestamp,
                "entry_price":self.entry_price,
                "exit_price":price,
                "quantity":quantity,
                "pnl":net_pnl,
                "return":trade_return,
                "exit_reason":reason
            })

            self.position = 0
            self.average_price = 0
            self.entry_price = None
            self.entry_time = None
            self.stop_price = None

    def check_stop_loss(self, bar):
        if self.position <= 0:
            return

        if self.stop_price is None:
            return

        #### stop not touched
        if bar.low > self.stop_price:
            return

        #### gap below stop
        if bar.open <= self.stop_price:
            execution_price = (bar.open * (1 - self.slippage))
        else:
            execution_price = (self.stop_price * (1 - self.slippage))

        self._close_position(price = execution_price, timestamp=bar.datetime, reason="STOP_LOSS")