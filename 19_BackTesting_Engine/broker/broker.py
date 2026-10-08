### Manages capital, positions, pending orders, transaction consts and the equity

from broker.enums import (OrderSide, OrderStatus, OrderType,TimeInForce,ExitReason)
from broker.order import Order
from broker.fill import Fill
from broker.position import Position
from broker.commission import PercentageCommission
from broker.slippage import PercentageSlippage
from broker.execution import ExecutionEngine

class Broker:
    def __init__(self, initial_cash = 10000, commission_rate=0.0002, slippage_rate=0.0001, max_volume_participation=0.10, allow_short=True):
        self.initial_cash = float(initial_cash)
        self.cash         = float(initial_cash)

        self.commission_model = PercentageCommission(commission_rate)
        self.slippage_model   = PercentageSlippage(slippage_rate)
        self.execution_engine = ExecutionEngine(max_volume_participation)
        self.allow_short      = allow_short

        self.position         = Position()

        self.pending_orders = []
        self.order_history  = []
        self.fill_history   = []

        self.total_commission = 0.0
        self.realized_pnl     = 0.0
        self.equity_curve     = []

    def submit_order(self, side, quantity, order_type=OrderType.MARKET, signal_time=None, limit_price=None, stop_price=None, time_in_force=TimeInForce.GTC, reduce_only=False, tag=None):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        if(order_type == OrderType.LIMIT and limit_price is None):
            raise ValueError("LIMIT order requires limit_price")

        if (order_type == OrderType.STOP and stop_price is None):
            raise ValueError("STOP order requires stop_price")

        if (order_type == OrderType.STOP_LIMIT and (stop_price is None or limit_price is None)):
            raise ValueError("STOP_LIMIT requires stop_price " "and limit_price")

        order = Order(
            side=side,
            quantity=float(quantity),
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            signal_time=signal_time,
            time_in_frame=time_in_force,
            reduce_only=reduce_only,
            tag=tag
        )

        order.status = (OrderStatus.PENDING)
        self.pending_orders.append(order)
        return order

    def buy(self, quantity, signal_time=None, tag=None):
        return self.submit_order(side=OrderSide.BUY, quantity=quantity, order_type=OrderType.MARKET, signal_time=signal_time, tag=tag)

    def sell(self, quantity, signal_time=None, reduce_only=False, tag=None):
        return self.submit_order(side=OrderSide.SELL, quantity=quantity, order_type=OrderType.MARKET, signal_time=signal_time, reduce_only=reduce_only, tag=tag)

    def limit_buy(self, quantity, limit_price, signal_time=None):
        return self.submit_order(OrderSide.BUY, quantity, OrderType.LIMIT, signal_time, limit_price=limit_price)

    def limit_sell(self, quantity, limit_price, signal_time=None):
        return self.submit_order(OrderSide.SELL,  quantity, OrderType.LIMIT, signal_time, limit_price=limit_price)

    def stop_buy(self, quantity, stop_price, signal_time=None):
        return self.submit_order(OrderSide.BUY, quantity, OrderType.STOP, signal_time, stop_price=stop_price)

    def stop_sell(self, quantity, stop_price, signal_time=None, reduce_only=True, tag="STOP_LOSS"):
        return self.submit_order(side=OrderSide.SELL, quantity=quantity, order_type=OrderType.STOP, signal_time=signal_time, stop_price=stop_price, reduce_only=reduce_only, tag=tag)

    def cancel_order(self, order_id):
        for order in self.pending_orders:
            if(order.id == order_id and order.is_active):
                order.status = (OrderStatus.CANCELLED)
                self.order_history.append(order)
                self.pending_orders = [x for x in self.pending_orders if x.id != order_id]
                return True
        return False

    ######### Execution ##############
    def process_bar(self, bar):
        if not self.pending_orders:
            return

        #### Liquidity available for this bar
        available_volume = (self.execution_engine.available_quantity(bar))
        surviving_orders = []

        for order in list(self.pending_orders):
            if not order.is_active:
                continue

            decision = (self.execution_engine.evaluate(order, bar))
            if not decision.should_fill:
                if (order.time_in_force == TimeInForce.IOC):
                    order.status = (OrderStatus.CANCELLED)
                    self.order_history.append(order)
                else:
                    surviving_orders.append(order)
                continue

            ########## FILL QUANTITY
            fill_quantity = min(order.remaining_quantity, available_volume)
            if fill_quantity <= 0:
                surviving_orders.append(order)
                continue

            ########## REDUCE ONLY VALIDATION
            fill_quantity = (self._apply_reduce_only_limit(order, fill_quantity))
            if fill_quantity <= 0:
                order.status = (OrderStatus.REJECTED)
                order.rejection_reason = ("Reduce-only order would increase or reverse position")
                self.order_history.append(order)
                continue

            ########### SLIPPAGE
            execution_price = (self.slippage_model.apply(price=decision.price, side=order.side, quantity=fill_quantity, bar=bar))

            ####### PRE-TRADE CHECKS
            allowed, reason = (self._validate_fill(order, fill_quantity, execution_price))

            if not allowed:
                order.status = (OrderStatus.REJECTED)
                order.rejection_reason = (reason)
                self.order_history.append(order)
                continue

            ######### COMMISSION
            commission = (self.commission_model.calculate(fill_quantity, execution_price))
           
            ########## CREATE FILL
            fill = Fill(order_id=order.id, side=order.side, quantity=fill_quantity, price=execution_price, commission=commission, timestamp=bar.datetime)

            ########## APPLY FILL
            self._apply_fill(order, fill)
            available_volume -= (fill_quantity)

            ########## ORDER STATE
            if (order.remaining_quantity <= 1e-12):
                order.status = (OrderStatus.FILLED)
                self.order_history.append(order)
            else:
                order.status = (OrderStatus.PARTIALLY_FILLED)
                if (order.time_in_force == TimeInForce.IOC):
                    order.status = (OrderStatus.CANCELLED)
                    self.order_history.append(order)
                else:
                    surviving_orders.append(order)
        self.pending_orders = (surviving_orders)

    def _apply_reduce_only_limit(self, order, requested_quantity):
        if not order.reduce_only:
            return requested_quantity

        position_qty = (self.position.quantity)
        if position_qty > 0:
            if order.side != OrderSide.SELL:
                return 0.0
            return min(requested_quantity, position_qty)
        if position_qty < 0:
            if order.side != OrderSide.BUY:
                return 0.0
            return min(requested_quantity, abs(position_qty))
        return 0.0

    def _validate_fill(self, order, quantity, price):
        commission = (self.commission_model.calculate(quantity, price))

        ##### BUY
        if order.side == OrderSide.BUY:
            #### if buy is opening position, cash must be available
            if(self.position.quantity >= 0):
                required_cash = (quantity * price + commission)
                if(required_cash > self.cash + 1e-12):
                    return (False, "Insufficient cash")

        else:
            #### selling beyond existing long creates a short
            resulting_position = (self.position.quantity - quantity)
            if(resulting_position < 0 and not self.allow_short):
                return (False, "short selling disabled")

        return True, None

    def _apply_fill(self, order, fill):
        notional = (fill.quantity * fill.price)
        if fill.side == OrderSide.BUY:
            self.cash -= (notional + fill.commission)
            signed_quantity = (fill.quantity)
        else:
            self.cash += (notional - fill.commission)
            signed_quantity = (-fill.quantity)

        ### Position
        realized               = (self.position.apply_fill(signed_quantity, fill.price))
        net_realized_change    = (realized - fill.commission)
        self.realized_pnl     += (net_realized_change)
        self.total_commission += (fill.commission)

        #### Order #########
        previous_filled = (order.filled_quantity)
        new_filled      = (previous_filled + fill.quantity)
        if new_filled > 0:
            order.average_fill_price = ((order.average_fill_price * previous_filled) + (fill.price * fill.quantity)) / new_filled
            order.filled_quantity    = new_filled
            order.commission        += (fill.commission)
            self.fill_history.append(fill)

    ####### Account value ##########
    def get_market_value(self, price):
        return (self.position.market_value(price))

    def get_unrealized_pnl(self, price):
        return (self.position.unrealized_pnl(price))

    def get_equity(self, price):
        return (self.cash + self.get_market_value(price))

    ########## Record ###############
    def update_equity(self, bar):
        equity = (self.get_equity(bar.close))
        self.equity_curve.append({
            "datetime":     bar.datetime,
            "cash":         self.cash,
            "position":     self.position.quantity,
            "average_price":self.position.average_price,
            "close":        bar.close,
            "market_value": self.get_market_value(bar.close),
            "unrealized_pnl":self.get_unrealized_pnl(bar.close),
            "realized_pnl": self.realized_pnl,
            "commission":   self.total_commission,
            "equity":       equity
        })

    ####### force close ############
    def liquidate(self, bar, reason=ExitReason.MANUAL.value):
        qty = (self.position.quantity)
        if abs(qty) < 1e-12:
            return None
        
        if qty > 0:
            order = self.sell(quantity=qty, signal_time=bar.datetime, reduce_only=True, tag=reason)
        else:
            order = self.submit_order(side=OrderSide.BUY, quantity=abs(qty), order_type=OrderType.MARKET, signal_time=bar.datetime, reduce_only=True, tag=reason)

        return order



