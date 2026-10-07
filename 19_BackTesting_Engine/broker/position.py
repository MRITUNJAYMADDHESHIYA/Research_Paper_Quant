# BUY 100
# BUY another 50
# SELL 75
# SELL remaining 75

# and

# SELL short 100
# BUY 50
# BUY 100 → closes short + reverses long

class Position:
    def __init__(self):
        self.quantity      = 0.0
        self.average_price = 0.0
        self.realized_pnl  = 0.0

    @property
    def is_flat(self):
        return abs(self.quantity) < 1e-12
    @property
    def is_long(self):
        return self.quantity > 0
    @property
    def is_short(self):
        return self.quantity < 0

    def market_value(self, price):
        return (self.quantity * price)

    def unrealized_pnl(self, price):
        if self.is_flat:
            return 0.0
        return (price - self.average_price)* self.quantity

    def apply_fill(self, signed_quantity, price):
        ## signed_quantity: buy:- positive, sell:- negative

        if signed_quantity == 0:
            return 0.0

        old_qty = self.quantity
        new_qty = (old_qty + signed_quantity)

        realized = 0.0

        ######### Open NEW POSITION ##########
        if self.is_flat:
            self.quantity = (signed_quantity)
            self.average_price = (price)

            return 0.0
        
        ########## Add to same direction ######
        same_direction = (old_qty * signed_quantity > 0)
        if same_direction:
            old_notional   = (abs(old_qty) * self.average_price)
            added_notional = (abs(signed_quantity) * price)
            total_quantity = (abs(old_qty) + abs(signed_quantity))

            self.average_price = (old_notional + added_notional) / total_quantity
            self.quantity      = new_qty
            return 0.0

        ############## Reduce, close, reverse ############
        closing_quantity = min(abs(old_qty), abs(signed_quantity)) 
        if old_qty > 0:
            #### closing long
            realized = (price - self.average_price)* closing_quantity
        else:
            #### closing short
            realized = (self.average_price - price)* closing_quantity


        self.realized_pnl += realized
        ##### completely flat
        if abs(new_qty) < 1e-12:
            self.quantity = 0.0
            self.average_price = 0.0

        ####### position reversed
        elif(old_qty * new_qty < 0):
            self.quantity     = new_qty
            self.average_price = price

        ###### partial close
        else:
            self.quantity = new_qty

        return realized