from collections import deque
from strategies.base import Strategy
from broker.enums import OrderSide, OrderType

class DonchianStrategy(Strategy):
    def __init__(self, broker, risk_manager, entry_period=20, exit_period=10, atr_period=14, atr_multiplier=2.0):
        super().__init__(broker)

        self.risk_manager = risk_manager

        self.entry_period   = entry_period
        self.exit_period    = exit_period
        self.atr_period     = atr_period
        self.atr_multiplier = atr_multiplier

        history_length   = max(entry_period, exit_period)+1
        self.highs       = deque(maxlen=history_length)
        self.lows        = deque(maxlen=history_length)
        self.true_ranges = deque(maxlen=atr_period)

        self.previous_close = None
        self.trailing_stop  = None

    def on_bar(self, bar, allow_entry=True):
        ######## True range
        if self.previous_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(bar.high - bar.low, abs(bar.high - self.previous_close), abs(bar.low - self.previous_close))

        self.true_ranges.append(tr)
        self.highs.append(bar.high)
        self.lows.append(bar.low)

        self.previous_close = bar.close

        ########## warm-up
        if (len(self.highs) < self.entry_period + 1 or len(self.true_ranges) < self.atr_period):
            return

        atr = (sum(self.true_ranges) / len(self.true_ranges))
        ######## Exclude current candle
        entry_high = max(list(self.highs)[:-1])
        exit_low   = min(list(self.lows)[-(self.exit_period + 1):-1])

        entry_low  = min(list(self.lows)[-self.entry_period-1 : -1])
        exit_high  = max(list(self.highs)[-self.exit_period-1 : -1])

        ###### Avoid duplicate orders
        has_pending_orders = any(order.is_active for order in self.broker.pending_orders)

        if has_pending_orders:
            return

        ######### Entry
        # ==========================================
        # LONG AND SHORT ENTRY
        # ==========================================

        if self.broker.position.is_flat:

            if not allow_entry:
                return

            equity = self.broker.get_equity(bar.close)

            stop_distance = atr * self.atr_multiplier

            if stop_distance <= 0:
                return

            risk_quantity = (
                equity * self.risk_manager.risk_per_trade
                / stop_distance
            )

            max_position_quantity = (
                equity * self.risk_manager.max_position_pct
                / bar.close
            )

            quantity = min(
                risk_quantity,
                max_position_quantity
            )

            # LONG ENTRY
            if bar.close > entry_high:

                # Cash-funded long
                affordable_quantity = (
                    self.broker.cash * 0.98
                    / (
                        bar.close
                        * (1 + self.broker.commission_model.rate)
                        * (1 + self.broker.slippage_model.rate)
                    )
                )

                quantity = min(
                    quantity,
                    affordable_quantity
                )

                if quantity > 0:

                    self.broker.buy(
                        quantity=quantity,
                        signal_time=bar.datetime,
                        tag="LONG_ENTRY"
                    )

            # SHORT ENTRY
            elif bar.close < entry_low:

                if quantity > 0:

                    self.broker.sell(
                        quantity=quantity,
                        signal_time=bar.datetime,
                        reduce_only=False,
                        tag="SHORT_ENTRY"
                    )


        # ==========================================
        # LONG EXIT
        # ==========================================

        elif self.broker.position.is_long:

            new_stop = (
                bar.close
                - self.atr_multiplier * atr
            )

            self.trailing_stop = (
                new_stop
                if self.trailing_stop is None
                else max(self.trailing_stop, new_stop)
            )

            if (
                bar.close < exit_low
                or bar.close < self.trailing_stop
            ):

                self.broker.sell(
                    quantity=self.broker.position.quantity,
                    signal_time=bar.datetime,
                    reduce_only=True,
                    tag="LONG_EXIT"
                )


        # ==========================================
        # SHORT EXIT
        # ==========================================

        elif self.broker.position.is_short:

            new_stop = (
                bar.close
                + self.atr_multiplier * atr
            )

            self.trailing_stop = (
                new_stop
                if self.trailing_stop is None
                else min(self.trailing_stop, new_stop)
            )

            if (
                bar.close > exit_high
                or bar.close > self.trailing_stop
            ):

                self.broker.submit_order(
                    side=OrderSide.BUY,
                    quantity=abs(self.broker.position.quantity),
                    order_type=OrderType.MARKET,
                    signal_time=bar.datetime,
                    reduce_only=True,
                    tag="SHORT_EXIT"
                )