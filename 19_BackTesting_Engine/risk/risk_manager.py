from datetime import date

class RiskManager:
    def __init__(self, initial_capital, risk_per_trade=0.01, stop_loss_pct=0.02, daily_drawdown_limit=0.03, max_drawdown_limit=0.10, max_position_pct=0.95):
        self.initial_capital      = float(initial_capital)

        self.risk_per_trade       = risk_per_trade         ### position size
        self.stop_loss_pct        = stop_loss_pct          ### stop loss size
        self.daily_drawdown_limit = (daily_drawdown_limit) ### daily-drawdown 
        self.max_drawdown_limit   = (max_drawdown_limit)   ### max total
        self.max_position_pct     = (max_position_pct)

        self.peak_equity      = float(initial_capital)
        self.current_drawdown = 0.0

        self.current_date          = None
        self.daily_start_equity    = float(initial_capital)
        self.daily_drawdown        = 0.0
        self.daily_trading_allowed = True

        self.kill_switch           = False

    ######### update portfolio risk ####################
    def update(self, timestamp, equity):
        current_date = timestamp.date()

        ####### new day
        if(self.current_date is None or current_date != self.current_date):
            self.current_date           = current_date
            self.daily_start_equity     = equity
            self.daily_drawdown         = 0.0
            self.daily_trading_allowed  = True

            #### Peak equity
            if equity > self.peak_equity:
                self.peak_equity = equity
            #### total drawdown
            if self.peak_equity > 0:
                self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
            #### daiky drawdown
            if self.daily_start_equity > 0:
                self.daily_drawdown = (self.daily_start_equity - equity) / self.daily_start_equity
            #### daily loss limit
            if (self.daily_drawdown >= self.daily_drawdown_limit):
                self.daily_trading_allowed = False
            #### kill switch
            if (self.current_drawdown >= self.max_drawdown_limit):
                self.kill_switch = True

        return {
            "drawdown": self.current_drawdown,
            "daily_drawdown": self.daily_drawdown,
            "kill_switch": self.kill_switch,
            "daily_allowed": self.daily_trading_allowed
        }


    ############# open new trade #############
    def can_trade(self):
        if self.kill_switch:
            return False

        if not self.daily_trading_allowed:
            return False

        return True

    ######### Position size ########
    def calculate_position_size(self, equity, cash, entry_price):
        if entry_price <= 0:
            return 0
        risk_amount = (equity * self.risk_per_trade)

        stop_distance = (entry_price * self.stop_loss_pct)
        if stop_distance <= 0:
            return 0

        risk_quantity = (risk_amount / stop_distance)

        max_capital = (equity * self.max_position_pct)
        capital_quantity = (max_capital/ entry_price)
        cash_quantity = (cash / entry_price)
        quantity      = min(risk_quantity, capital_quantity, cash_quantity)

        return max(quantity, 0)

    def calculate_stop_price(self, entry_price):
        return (entry_price * (1 - self.stop_loss_pct))

    