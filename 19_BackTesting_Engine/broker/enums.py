from enum import Enum

class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET    = "MARKET"
    LIMIT     = "LIMIT"
    STOP      = "STOP"
    STOP_LIMT = "STOP_LIMIT"

class OrderStatus(Enum):
    NEW              = "NEW"
    PENDING          = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED           = "FILLED"
    CANCELLED        = "CANCELLED"
    REJECTED         = "REJECTED"
    EXPIRED          = "EXPIRED"

class TimeInForce(Enum):
    GTC = "GTC"    ## good till canceled:- remains active until filled or manually canceled
    DAY = "DAY"    ## day order:- active only for the current trading session
    IOC = "IOC"    ## Immediate or cancel:- execute available shares immediately at the target price
    FOK = "FOK"    ## fill or kill:- executes the entire order immediately
    GTD = "GTD"    ## good till date:- active until the market closes on a specific calendar date

class ExitReason(Enum):
    SIGNAL      = "SIGNAL"
    STOP_LOSS   = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    RISK_LIMIT  = "RISK_LIMIT"
    END_OF_DATA = "END_OF_DATA"
    MANUAL      = "MANUAL"

