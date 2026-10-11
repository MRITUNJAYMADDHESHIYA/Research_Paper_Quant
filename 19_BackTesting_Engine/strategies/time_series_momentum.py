## Strategy:- Rt = Pt/P(t-k) - 1    :k-->lookback period, t-->current price time
## Volatility-adjusted sizing(positions scaled):- High-volatility markets receive smaller positions, while lower-volatility markets receive larger positions.


from collections import deque
import numpy as np
from strategies.base import Strategy


class TimeSeriesMomentumStrategy(Strategy):
    def __init__(self, broker, risk_manager, momentum_period=20, volatility_period=120, target_volatility=0.20, periods_per_year=365 * 24 * 60,):
        super().__init__(broker)

        self.risk_manager      = risk_manager
        self.momentum_period   = momentum_period
        self.volatility_period = volatility_period
        self.target_volatility = target_volatility
        self.periods_per_year  = periods_per_year

        history     = max(momentum_period, volatility_period) + 1
        self.closes = deque(maxlen=history)

    def on_bar(self, bar, allow_entry=True):
        self.closes.append(float(bar.close))
        required = max(self.momentum_period, self.volatility_period) + 1

        if len(self.closes) < required:
            return

        prices = np.asarray(self.closes, dtype=float)

        ########### Historical return ####################
        momentum          = (prices[-1] / prices[-1 - self.momentum_period]) - 1.0
        ########### Realized volatility ##################
        log_returns       = np.diff(np.log(prices[-self.volatility_period - 1:]))
        annual_volatility = (np.std(log_returns, ddof=1) * np.sqrt(self.periods_per_year))

        if annual_volatility <= 1e-12:
            return

        ############### Target exposure
        exposure        = min(self.target_volatility / annual_volatility, self.risk_manager.max_position_pct)
        equity          = self.broker.get_equity(bar.close)
        target_quantity = (equity * exposure / bar.close)
        has_pending     = any(order.is_active for order in self.broker.pending_orders)

        if has_pending:
            return

        ######### Positive momentum -> Long
        if momentum > 0 and allow_entry:
            if self.broker.position.is_flat:
                affordable = (self.broker.cash * 0.98 / (bar.close * (1 + self.broker.commission_model.rate) * (1 + self.broker.slippage_model.rate)))
                quantity   = min(target_quantity, affordable)

                if quantity > 0:
                    self.broker.buy(quantity=quantity, signal_time=bar.datetime, tag="TSMOM_LONG")

        ######### Negative momentum -> Flat
        elif momentum < 0:
            if self.broker.position.is_long:
                self.broker.sell(quantity=self.broker.position.quantity, signal_time=bar.datetime, reduce_only=True, tag="TSMOM_EXIT")



